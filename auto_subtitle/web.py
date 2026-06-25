import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from .config import get_openai_model, load_env
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import SubtitleConfig, generate_vietsub, reburn_subtitles
from .subtitle_renderer import (
    FONT_FAMILY_CHOICES,
    default_layout_dict,
)
from .translation_topics import DEFAULT_TOPIC, list_topics, normalize_topic
from .utils import hex_color_to_ass

load_env()

STATIC_DIR = Path(__file__).parent / "static"
JOBS_ROOT = Path(tempfile.gettempdir()) / "drakonsub_jobs"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    message: str = "Waiting..."
    progress: int = 0
    output_name: str = ""
    output_path: Optional[str] = None
    error: Optional[str] = None
    input_path: Optional[str] = None
    srt_path: Optional[str] = None
    layout_path: Optional[str] = None
    layout_saved: bool = False
    translation_topic: str = DEFAULT_TOPIC
    subtitle_font_size: Optional[int] = None
    subtitle_font_color: Optional[str] = None
    source_language: str = "en"


jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()

app = FastAPI(title="DrakonSub")


def prepare_output_name(name: str, fallback: str) -> str:
    name = (name or "").strip()
    if name.lower().endswith(".mp4"):
        name = name[:-4].rstrip()
    name = os.path.basename(name)
    return name or fallback


def parse_font_size(value: Optional[str], default: int) -> int:
    if value is None or not str(value).strip():
        return default
    size = int(value)
    if size < 12 or size > 200:
        raise ValueError("Font size must be between 12 and 200")
    return size


def parse_font_color(value: Optional[str], default: str) -> str:
    if value is None or not str(value).strip():
        return default
    raw = value.strip()
    if not raw.startswith("#"):
        raw = f"#{raw}"
    hex_color_to_ass(raw)
    return raw.upper()


def build_subtitle_config(job: Job) -> SubtitleConfig:
    config = SubtitleConfig.from_env()
    config.translation_topic = job.translation_topic
    config.source_language = job.source_language
    if job.subtitle_font_size is not None:
        config.subtitle_font_size = job.subtitle_font_size
    if job.subtitle_font_color is not None:
        config.subtitle_font_color = job.subtitle_font_color
    return config


def _job_paths(job_id: str) -> tuple[Path, Path, Path]:
    job_dir = JOBS_ROOT / job_id
    return job_dir, job_dir / "vi.srt", job_dir / "layout.json"


def _read_layout_file(path: str) -> Dict[str, Any]:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)
    return default_layout_dict()


def _write_layout_file(path: str, layout: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(layout, fp, ensure_ascii=False, indent=2)


def validate_layout(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize layout JSON from the client."""
    if not isinstance(data, dict):
        raise ValueError("Layout must be a JSON object")

    base = default_layout_dict()
    layout = {**base, **data}

    layout["mode"] = str(layout.get("mode", "rounded")).strip().lower()
    if layout["mode"] not in ("rounded", "classic"):
        raise ValueError("mode must be 'rounded' or 'classic'")

    layout["x_ratio"] = float(layout["x_ratio"])
    layout["y_ratio"] = float(layout["y_ratio"])
    if not 0.0 <= layout["x_ratio"] <= 1.0:
        raise ValueError("x_ratio must be between 0 and 1")
    if not 0.0 <= layout["y_ratio"] <= 1.0:
        raise ValueError("y_ratio must be between 0 and 1")

    layout["width_ratio"] = float(layout.get("width_ratio", layout.get("max_width_ratio", 0.86)))
    if not 0.4 <= layout["width_ratio"] <= 1.0:
        raise ValueError("width_ratio must be between 0.4 and 1.0")

    layout["font_size"] = int(layout["font_size"])
    if not 12 <= layout["font_size"] <= 200:
        raise ValueError("font_size must be between 12 and 200")

    for color_key in ("text_color", "background_color"):
        raw = str(layout[color_key]).strip()
        if not raw.startswith("#"):
            raw = f"#{raw}"
        hex_color_to_ass(raw)
        layout[color_key] = raw.upper()

    layout["background_opacity"] = float(layout["background_opacity"])
    if not 0.0 <= layout["background_opacity"] <= 1.0:
        raise ValueError("background_opacity must be between 0 and 1")

    layout["border_radius"] = int(layout["border_radius"])
    layout["padding_x"] = int(layout["padding_x"])
    layout["padding_y"] = int(layout["padding_y"])

    font_family = str(layout.get("font_family", "arial_bold")).strip().lower()
    if font_family not in FONT_FAMILY_CHOICES:
        raise ValueError(f"font_family must be one of: {', '.join(FONT_FAMILY_CHOICES)}")
    layout["font_family"] = font_family

    return layout


def _job_to_dict(job: Job) -> Dict[str, Any]:
    can_render_again = (
        job.status in (JobStatus.DONE, JobStatus.ERROR)
        and job.layout_saved
        and job.srt_path
        and os.path.isfile(job.srt_path)
        and job.input_path
        and os.path.isfile(job.input_path)
    )
    return {
        "job_id": job.id,
        "status": job.status.value,
        "message": job.message,
        "progress": job.progress,
        "output_name": job.output_name,
        "error": job.error,
        "layout_saved": job.layout_saved,
        "can_render_again": can_render_again,
        "preview_url": f"/api/jobs/{job.id}/preview"
        if job.status == JobStatus.DONE
        else None,
        "download_url": f"/api/jobs/{job.id}/download"
        if job.status == JobStatus.DONE
        else None,
    }


def _run_job(job_id: str) -> None:
    def on_progress(message: str, percent: int) -> None:
        with jobs_lock:
            jobs[job_id].message = message
            jobs[job_id].progress = percent

    try:
        with jobs_lock:
            job = jobs[job_id]
            job.status = JobStatus.PROCESSING
            job.message = "Starting..."
            job.progress = 0
            job_dir, srt_path, layout_path = _job_paths(job_id)
            output_path = str(job_dir / f"{job.output_name}.mp4")
            video_path = job.input_path
            config = build_subtitle_config(job)
            job.srt_path = str(srt_path)
            job.layout_path = str(layout_path)

        generate_vietsub(
            video_path,
            output_path,
            config=config,
            on_progress=on_progress,
            persist_srt_path=str(srt_path),
            persist_layout_path=str(layout_path),
        )

        with jobs_lock:
            jobs[job_id].status = JobStatus.DONE
            jobs[job_id].output_path = output_path
            jobs[job_id].layout_saved = True
            jobs[job_id].message = "Complete"
            jobs[job_id].progress = 100
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].status = JobStatus.ERROR
            jobs[job_id].error = str(exc)
            jobs[job_id].message = "Failed"


def reburn_job_subtitles(
    job_id: str,
    layout: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Any] = None,
) -> str:
    """
    Re-burn subtitles for a completed job using persisted input, vi.srt, and layout.

    Does not transcribe or translate.
    """
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise RuntimeError("Job not found")
        if not job.input_path or not os.path.isfile(job.input_path):
            raise RuntimeError("Original video not found")
        if not job.srt_path or not os.path.isfile(job.srt_path):
            raise RuntimeError("vi.srt not found — run Generate first")
        layout_path = job.layout_path or str(_job_paths(job_id)[2])
        if layout is None:
            if not os.path.isfile(layout_path):
                raise RuntimeError("layout.json not found — save layout first")
            layout = _read_layout_file(layout_path)
        else:
            layout = validate_layout(layout)
        output_path = job.output_path or str(
            JOBS_ROOT / job_id / f"{job.output_name}.mp4"
        )
        config = build_subtitle_config(job)

    reburn_subtitles(
        job.input_path,
        job.srt_path,
        output_path,
        layout,
        config=config,
        on_progress=on_progress,
    )
    return output_path


def _run_reburn(job_id: str) -> None:
    def on_progress(message: str, percent: int) -> None:
        with jobs_lock:
            jobs[job_id].message = message
            jobs[job_id].progress = percent

    try:
        with jobs_lock:
            jobs[job_id].status = JobStatus.PROCESSING
            jobs[job_id].message = "Re-rendering..."
            jobs[job_id].progress = 0

        output_path = reburn_job_subtitles(job_id, on_progress=on_progress)

        with jobs_lock:
            jobs[job_id].status = JobStatus.DONE
            jobs[job_id].output_path = output_path
            jobs[job_id].message = "Re-render complete"
            jobs[job_id].progress = 100
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].status = JobStatus.ERROR
            jobs[job_id].error = str(exc)
            jobs[job_id].message = "Re-render failed"


@app.get("/api/topics")
def get_topics():
    return {"topics": list_topics(), "default": DEFAULT_TOPIC}


@app.get("/api/defaults")
def get_defaults():
    config = SubtitleConfig.from_env()
    return {
        "subtitle_font_size": config.subtitle_font_size,
        "subtitle_font_color": config.subtitle_font_color,
        "openai_model": get_openai_model(),
        "default_layout": default_layout_dict(),
        "font_presets": list(FONT_FAMILY_CHOICES),
    }


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    output_name: str = Form(...),
    topic: str = Form(DEFAULT_TOPIC),
    font_size: Optional[str] = Form(None),
    font_color: Optional[str] = Form(None),
    source_language: str = Form("en"),
):
    if not video.filename:
        raise HTTPException(400, "No file uploaded")

    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, "Unsupported format. Use MP4, MOV, MKV, or WEBM.")

    try:
        safe_topic = normalize_topic(topic)
        if source_language not in ("en", "vi"):
            raise ValueError("source_language must be 'en' or 'vi'")
        defaults = SubtitleConfig.from_env()
        parsed_font_size = (
            parse_font_size(font_size, defaults.subtitle_font_size)
            if font_size and font_size.strip()
            else None
        )
        parsed_font_color = (
            parse_font_color(font_color, defaults.subtitle_font_color)
            if font_color and font_color.strip()
            else None
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    fallback_name = f"{Path(video.filename).stem}-vietsub"
    final_name = prepare_output_name(output_name, fallback_name)

    job_id = str(uuid.uuid4())
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{ext}"
    input_path.write_bytes(await video.read())

    _, srt_path, layout_path = _job_paths(job_id)

    job = Job(
        id=job_id,
        output_name=final_name,
        input_path=str(input_path),
        srt_path=str(srt_path),
        layout_path=str(layout_path),
        translation_topic=safe_topic,
        subtitle_font_size=parsed_font_size,
        subtitle_font_color=parsed_font_color,
        source_language=source_language,
    )

    with jobs_lock:
        jobs[job_id] = job

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()

    return {"job_id": job_id, "output_name": final_name, "topic": safe_topic}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return _job_to_dict(job)


@app.get("/api/jobs/{job_id}/layout")
def get_job_layout(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        layout_path = job.layout_path

    if layout_path and os.path.isfile(layout_path):
        return _read_layout_file(layout_path)
    return default_layout_dict()


@app.put("/api/jobs/{job_id}/layout")
def put_job_layout(job_id: str, body: Dict[str, Any] = Body(...)):
    try:
        layout = validate_layout(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status not in (JobStatus.DONE, JobStatus.ERROR):
            raise HTTPException(400, "Layout can only be saved after Generate completes")
        layout_path = job.layout_path or str(_job_paths(job_id)[2])

    _write_layout_file(layout_path, layout)

    with jobs_lock:
        jobs[job_id].layout_path = layout_path
        jobs[job_id].layout_saved = True

    return {"ok": True, "layout": layout}


@app.post("/api/jobs/{job_id}/render")
def render_job_again(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status == JobStatus.PROCESSING:
            raise HTTPException(409, "Job is already processing")
        if not job.layout_saved:
            raise HTTPException(400, "Save layout before re-rendering")
        if not job.srt_path or not os.path.isfile(job.srt_path):
            raise HTTPException(400, "vi.srt not found — run Generate first")
        if not job.input_path or not os.path.isfile(job.input_path):
            raise HTTPException(400, "Original video not found")

    threading.Thread(target=_run_reburn, args=(job_id,), daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.get("/api/jobs/{job_id}/preview")
def preview_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status != JobStatus.DONE or not job.output_path:
            raise HTTPException(400, "Video is not ready yet")
        path = job.output_path

    if not os.path.isfile(path):
        raise HTTPException(404, "Output file not found")

    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Content-Disposition": "inline"},
    )


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status != JobStatus.DONE or not job.output_path:
            raise HTTPException(400, "Video is not ready yet")
        path = job.output_path
        name = job.output_name

    if not os.path.isfile(path):
        raise HTTPException(404, "Output file not found")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{name}.mp4",
    )


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main():
    import uvicorn

    host = os.getenv("DRAKONSUB_HOST", "127.0.0.1")
    port = int(os.getenv("DRAKONSUB_PORT", "8000"))
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    model = get_openai_model()
    print(f"DrakonSub web UI: http://{host}:{port}")
    print(f"OpenAI model: {model}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
