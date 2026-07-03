import json
import os
import tempfile
import threading
import traceback
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .config import (
    SUPPORTED_TRANSLATION_ENGINES,
    get_openai_model,
    get_translation_engine,
    load_env,
)
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import SubtitleConfig, generate_vietsub, reburn_subtitles
from .subtitle_edit_service import (
    SubtitleEditError,
    apply_text_edits,
    cue_indices_from_request,
    get_effective_vi_srt,
    load_srt,
    merge_subtitle_views,
    reset_cues_to_original,
    write_srt,
    write_user_edits,
)
from .subtitle_renderer import (
    FONT_FAMILY_CHOICES,
    default_layout_dict,
    resolve_background_visible,
)
from .translation_topics import DEFAULT_TOPIC, list_topics, normalize_topic
from .utils import hex_color_to_ass

load_env()

STATIC_DIR = Path(__file__).parent / "static"
FONTS_DIR = Path(__file__).parent / "fonts"
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
    source_srt_path: Optional[str] = None
    srt_path: Optional[str] = None
    layout_path: Optional[str] = None
    layout_saved: bool = False
    translation_topic: str = DEFAULT_TOPIC
    translation_engine: str = "openai"
    subtitle_font_size: Optional[int] = None
    subtitle_font_color: Optional[str] = None
    source_language: str = "en"
    last_render_subtitle_source: Optional[str] = None


jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()

app = FastAPI(title="DrakonSub")


def _iter_file_chunks(path: str, start: int, end: int, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    with open(path, "rb") as fp:
        fp.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            size = min(chunk_size, remaining)
            chunk = fp.read(size)
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _video_preview_response(path: str, request: Request) -> Response:
    file_size = os.path.getsize(path)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": "inline",
    }
    range_header = request.headers.get("range")
    if not range_header:
        headers["Content-Length"] = str(file_size)
        return FileResponse(path, media_type="video/mp4", headers=headers)

    try:
        units, raw_range = range_header.strip().split("=", 1)
        if units != "bytes":
            raise ValueError("Invalid range unit")
        start_text, end_text = raw_range.split("-", 1)
        if start_text == "":
            length = int(end_text)
            start = max(file_size - length, 0)
            end = file_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        if start > end or start < 0 or end >= file_size:
            raise ValueError("Invalid byte range")
    except (ValueError, IndexError):
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    headers.update({
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Content-Length": str(end - start + 1),
    })
    return StreamingResponse(
        _iter_file_chunks(path, start, end),
        status_code=206,
        media_type="video/mp4",
        headers=headers,
    )


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
    config.translation_engine = job.translation_engine
    config.source_language = job.source_language
    if job.subtitle_font_size is not None:
        config.subtitle_font_size = job.subtitle_font_size
    if job.subtitle_font_color is not None:
        config.subtitle_font_color = job.subtitle_font_color
    return config


def _validate_job_id(job_id: str) -> str:
    raw = (job_id or "").strip()
    if (
        not raw
        or "/" in raw
        or "\\" in raw
        or ".." in raw
        or Path(raw).is_absolute()
    ):
        raise HTTPException(404, "Job not found")
    return raw


def _job_paths(job_id: str) -> tuple[Path, Path, Path, Path]:
    job_id = _validate_job_id(job_id)
    job_dir = JOBS_ROOT / job_id
    return job_dir, job_dir / "source.srt", job_dir / "vi_final.srt", job_dir / "layout.json"


def _edited_vi_path(job_id: str) -> Path:
    return _job_paths(job_id)[0] / "edited_vi.srt"


def _user_edits_path(job_id: str) -> Path:
    return _job_paths(job_id)[0] / "user_edits.json"


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

    def _as_float(value: Any, field: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be a number") from exc

    def _as_int(value: Any, field: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be an integer") from exc

    base = default_layout_dict()
    layout = {**base, **data}

    layout["mode"] = str(layout.get("mode", "rounded")).strip().lower()
    if layout["mode"] not in ("rounded", "classic"):
        raise ValueError("mode must be 'rounded' or 'classic'")

    layout["x_ratio"] = _as_float(layout["x_ratio"], "x_ratio")
    layout["y_ratio"] = _as_float(layout["y_ratio"], "y_ratio")
    if not 0.0 <= layout["x_ratio"] <= 1.0:
        raise ValueError("x_ratio must be between 0 and 1")
    if not 0.0 <= layout["y_ratio"] <= 1.0:
        raise ValueError("y_ratio must be between 0 and 1")

    layout["width_ratio"] = _as_float(
        layout.get("width_ratio", layout.get("max_width_ratio", 0.86)),
        "width_ratio",
    )
    if not 0.4 <= layout["width_ratio"] <= 1.0:
        raise ValueError("width_ratio must be between 0.4 and 1.0")

    layout["font_size"] = _as_int(layout["font_size"], "font_size")
    if not 12 <= layout["font_size"] <= 200:
        raise ValueError("font_size must be between 12 and 200")

    for color_key in ("text_color", "background_color"):
        raw = str(layout[color_key]).strip()
        if not raw.startswith("#"):
            raw = f"#{raw}"
        hex_color_to_ass(raw)
        layout[color_key] = raw.upper()

    layout["background_opacity"] = _as_float(
        layout["background_opacity"], "background_opacity"
    )
    if not 0.0 <= layout["background_opacity"] <= 1.0:
        raise ValueError("background_opacity must be between 0 and 1")

    layout["background_visible"] = resolve_background_visible(layout)

    layout["border_radius"] = _as_int(layout["border_radius"], "border_radius")
    layout["padding_x"] = _as_int(layout["padding_x"], "padding_x")
    layout["padding_y"] = _as_int(layout["padding_y"], "padding_y")
    if not 0 <= layout["border_radius"] <= 200:
        raise ValueError("border_radius must be between 0 and 200")
    if not 0 <= layout["padding_x"] <= 200:
        raise ValueError("padding_x must be between 0 and 200")
    if not 0 <= layout["padding_y"] <= 200:
        raise ValueError("padding_y must be between 0 and 200")

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
        "source_srt_download_url": f"/api/jobs/{job.id}/download/source-srt"
        if job.status == JobStatus.DONE and job.source_srt_path and os.path.isfile(job.source_srt_path)
        else None,
        "vi_srt_download_url": f"/api/jobs/{job.id}/download/vi-srt"
        if job.status == JobStatus.DONE and job.srt_path and os.path.isfile(job.srt_path)
        else None,
        "subtitle_source": job.last_render_subtitle_source,
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
            job_dir, source_srt_path, srt_path, layout_path = _job_paths(job_id)
            output_path = str(job_dir / f"{job.output_name}.mp4")
            video_path = job.input_path
            config = build_subtitle_config(job)
            job.source_srt_path = str(source_srt_path)
            job.srt_path = str(srt_path)
            job.layout_path = str(layout_path)

        generate_vietsub(
            video_path,
            output_path,
            config=config,
            on_progress=on_progress,
            persist_srt_path=str(srt_path),
            persist_source_srt_path=str(source_srt_path),
            persist_layout_path=str(layout_path),
        )

        with jobs_lock:
            jobs[job_id].status = JobStatus.DONE
            jobs[job_id].output_path = output_path
            jobs[job_id].layout_saved = True
            jobs[job_id].message = "Complete"
            jobs[job_id].progress = 100
            jobs[job_id].last_render_subtitle_source = Path(srt_path).name
    except Exception as exc:
        traceback.print_exc()
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
            raise RuntimeError("Vietnamese subtitle not found — run Generate first")
        layout_path = job.layout_path or str(_job_paths(job_id)[3])
        if layout is None:
            if not os.path.isfile(layout_path):
                raise RuntimeError("layout.json not found — save layout first")
            layout = _read_layout_file(layout_path)
        else:
            layout = validate_layout(layout)
        output_path = job.output_path or str(
            JOBS_ROOT / job_id / f"{job.output_name}.mp4"
        )
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        tmp_output_path = os.path.join(
            output_dir, f".{job.output_name}.tmp-{uuid.uuid4().hex}.mp4"
        )
        config = build_subtitle_config(job)

    try:
        effective_srt = str(get_effective_vi_srt(_job_paths(job_id)[0]))
        subtitle_source = Path(effective_srt).name
        print(f"[render] effective subtitle: {subtitle_source}")
        reburn_subtitles(
            job.input_path,
            effective_srt,
            tmp_output_path,
            layout,
            config=config,
            on_progress=on_progress,
        )
        os.replace(tmp_output_path, output_path)
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].last_render_subtitle_source = subtitle_source
    finally:
        if os.path.isfile(tmp_output_path):
            try:
                os.remove(tmp_output_path)
            except OSError:
                pass

    return output_path


def _run_reburn(job_id: str) -> None:
    def on_progress(message: str, percent: int) -> None:
        with jobs_lock:
            jobs[job_id].message = message
            jobs[job_id].progress = percent

    try:
        with jobs_lock:
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
        "translation_engine": config.translation_engine,
        "translation_engines": list(SUPPORTED_TRANSLATION_ENGINES),
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
    translation_engine: str = Form(get_translation_engine()),
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
        translation_engine = (translation_engine or "").strip().lower()
        if translation_engine not in SUPPORTED_TRANSLATION_ENGINES:
            raise ValueError(
                "translation_engine must be one of: "
                + ", ".join(SUPPORTED_TRANSLATION_ENGINES)
            )
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

    _, source_srt_path, srt_path, layout_path = _job_paths(job_id)

    job = Job(
        id=job_id,
        output_name=final_name,
        input_path=str(input_path),
        source_srt_path=str(source_srt_path),
        srt_path=str(srt_path),
        layout_path=str(layout_path),
        translation_topic=safe_topic,
        translation_engine=translation_engine,
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
        layout_path = job.layout_path or str(_job_paths(job_id)[3])

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
            raise HTTPException(400, "Vietnamese SRT is not ready yet")
        if not job.input_path or not os.path.isfile(job.input_path):
            raise HTTPException(400, "Original video not found")
        job.status = JobStatus.PROCESSING
        job.message = "Re-rendering..."
        job.progress = 0
        job.error = None

    try:
        threading.Thread(target=_run_reburn, args=(job_id,), daemon=True).start()
    except Exception:
        with jobs_lock:
            job = jobs.get(job_id)
            if job:
                job.status = JobStatus.ERROR
                job.message = "Re-render failed"
                job.error = "Failed to start re-render worker"
        raise

    subtitle_source = get_effective_vi_srt(_job_paths(job_id)[0]).name
    return {"ok": True, "job_id": job_id, "subtitle_source": subtitle_source}


@app.get("/api/jobs/{job_id}/preview")
def preview_job(job_id: str, request: Request):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status != JobStatus.DONE or not job.output_path:
            raise HTTPException(400, "Video is not ready yet")
        path = job.output_path

    if not os.path.isfile(path):
        raise HTTPException(404, "Output file not found")

    return _video_preview_response(path, request)


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


@app.get("/api/jobs/{job_id}/download/source-srt")
def download_job_source_srt(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status != JobStatus.DONE or not job.source_srt_path:
            raise HTTPException(400, "Source SRT is not ready yet")
        path = job.source_srt_path
        name = job.output_name

    if not os.path.isfile(path):
        raise HTTPException(404, "Source SRT file not found")

    return FileResponse(
        path,
        media_type="application/x-subrip",
        filename=f"{name}.source.srt",
    )


@app.get("/api/jobs/{job_id}/download/vi-srt")
def download_job_vi_srt(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status != JobStatus.DONE or not job.srt_path:
            raise HTTPException(400, "Vietnamese SRT is not ready yet")
        path = job.srt_path
        name = job.output_name

    if not os.path.isfile(path):
        raise HTTPException(404, "Vietnamese SRT file not found")

    return FileResponse(
        path,
        media_type="application/x-subrip",
        filename=f"{name}.vi.srt",
    )


@app.get("/api/jobs/{job_id}/subtitles")
def get_job_subtitles(job_id: str):
    _validate_job_id(job_id)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        job_dir = _job_paths(job_id)[0]
        source_path = Path(job.source_srt_path) if job.source_srt_path else job_dir / "source.srt"
        vi_final_path = Path(job.srt_path) if job.srt_path else job_dir / "vi_final.srt"

    if not vi_final_path.exists():
        raise HTTPException(404, "Vietnamese subtitle not found")
    try:
        source_cues = load_srt(source_path) if source_path.exists() else None
        original_vi_cues = load_srt(vi_final_path)
        current_path = get_effective_vi_srt(job_dir)
        current_vi_cues = load_srt(current_path)
        subtitles = merge_subtitle_views(source_cues, original_vi_cues, current_vi_cues)
    except SubtitleEditError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "job_id": job_id,
        "has_edits": any(item["edited"] for item in subtitles),
        "subtitle_source": current_path.name,
        "subtitles": subtitles,
    }


@app.post("/api/jobs/{job_id}/subtitles/save")
def save_job_subtitles(job_id: str, body: Dict[str, Any] = Body(...)):
    _validate_job_id(job_id)
    edits = body.get("edits")
    if not isinstance(edits, list) or not edits:
        raise HTTPException(400, "Invalid cue index")
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        job_dir = _job_paths(job_id)[0]
        vi_final_path = Path(job.srt_path) if job.srt_path else job_dir / "vi_final.srt"
    if not vi_final_path.exists():
        raise HTTPException(404, "Vietnamese subtitle not found")

    edited_path = _edited_vi_path(job_id)
    try:
        original_vi_cues = load_srt(vi_final_path)
        current_vi_cues = load_srt(edited_path) if edited_path.exists() else load_srt(vi_final_path)
        updated_cues = apply_text_edits(original_vi_cues, current_vi_cues, edits)
        write_srt(updated_cues, edited_path)
        payload = write_user_edits(job_id, _user_edits_path(job_id), original_vi_cues, updated_cues)
    except SubtitleEditError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "job_id": job_id,
        "saved": True,
        "edited_count": len(payload["edits"]),
        "edited_srt": edited_path.name,
        "message": "Subtitle edits saved. Re-render video to apply changes.",
    }


@app.post("/api/jobs/{job_id}/subtitles/reset")
def reset_job_subtitles(job_id: str, body: Dict[str, Any] = Body(...)):
    _validate_job_id(job_id)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        vi_final_path = Path(job.srt_path) if job.srt_path else _job_paths(job_id)[0] / "vi_final.srt"
    if not vi_final_path.exists():
        raise HTTPException(404, "Vietnamese subtitle not found")

    edited_path = _edited_vi_path(job_id)
    try:
        cue_indices = cue_indices_from_request(body)
        original_vi_cues = load_srt(vi_final_path)
        current_vi_cues = load_srt(edited_path) if edited_path.exists() else load_srt(vi_final_path)
        updated_cues = reset_cues_to_original(original_vi_cues, current_vi_cues, cue_indices)
        payload = write_user_edits(job_id, _user_edits_path(job_id), original_vi_cues, updated_cues)
        if payload["edits"]:
            write_srt(updated_cues, edited_path)
        elif edited_path.exists():
            edited_path.unlink()
    except SubtitleEditError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "job_id": job_id,
        "reset_count": len(cue_indices),
        "has_edits": bool(payload["edits"]),
    }


@app.post("/api/jobs/{job_id}/subtitles/reset-all")
def reset_all_job_subtitles(job_id: str):
    _validate_job_id(job_id)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        vi_final_path = Path(job.srt_path) if job.srt_path else _job_paths(job_id)[0] / "vi_final.srt"
    if not vi_final_path.exists():
        raise HTTPException(404, "Vietnamese subtitle not found")

    edited_path = _edited_vi_path(job_id)
    user_edits_path = _user_edits_path(job_id)
    original_vi_cues = load_srt(vi_final_path)
    payload = write_user_edits(job_id, user_edits_path, original_vi_cues, original_vi_cues)
    if edited_path.exists():
        edited_path.unlink()

    return {"job_id": job_id, "reset_all": True, "has_edits": bool(payload["edits"])}


app.mount("/fonts", StaticFiles(directory=str(FONTS_DIR)), name="fonts")
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
