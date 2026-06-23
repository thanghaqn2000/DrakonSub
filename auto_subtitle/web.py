import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from .config import get_openai_model, load_env
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import SubtitleConfig, generate_vietsub
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
            output_path = str(JOBS_ROOT / job_id / f"{job.output_name}.mp4")
            video_path = job.input_path
            config = build_subtitle_config(job)

        generate_vietsub(
            video_path,
            output_path,
            config=config,
            on_progress=on_progress,
        )

        with jobs_lock:
            jobs[job_id].status = JobStatus.DONE
            jobs[job_id].output_path = output_path
            jobs[job_id].message = "Complete"
            jobs[job_id].progress = 100
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].status = JobStatus.ERROR
            jobs[job_id].error = str(exc)
            jobs[job_id].message = "Failed"


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

    job = Job(
        id=job_id,
        output_name=final_name,
        input_path=str(input_path),
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
        return {
            "job_id": job.id,
            "status": job.status.value,
            "message": job.message,
            "progress": job.progress,
            "output_name": job.output_name,
            "error": job.error,
            "preview_url": f"/api/jobs/{job_id}/preview"
            if job.status == JobStatus.DONE
            else None,
            "download_url": f"/api/jobs/{job_id}/download"
            if job.status == JobStatus.DONE
            else None,
        }


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
