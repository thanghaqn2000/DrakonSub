import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import SubtitleConfig, generate_vietsub

load_dotenv()

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


jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()

app = FastAPI(title="DrakonSub")


def sanitize_output_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Output name is required")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError(
            "Output name may only contain letters, numbers, dots, dashes, and underscores"
        )
    return name


def _run_job(job_id: str) -> None:
    def on_progress(message: str, percent: int) -> None:
        with jobs_lock:
            jobs[job_id].message = message
            jobs[job_id].progress = percent

    try:
        with jobs_lock:
            jobs[job_id].status = JobStatus.PROCESSING
            jobs[job_id].message = "Starting..."
            jobs[job_id].progress = 0
            output_path = str(
                JOBS_ROOT / job_id / f"{jobs[job_id].output_name}.mp4"
            )
            video_path = jobs[job_id].input_path

        generate_vietsub(
            video_path,
            output_path,
            config=SubtitleConfig.from_env(),
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


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    output_name: str = Form(...),
):
    if not video.filename:
        raise HTTPException(400, "No file uploaded")

    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, "Unsupported format. Use MP4, MOV, MKV, or WEBM.")

    try:
        safe_name = sanitize_output_name(output_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = str(uuid.uuid4())
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{ext}"
    input_path.write_bytes(await video.read())

    job = Job(
        id=job_id,
        output_name=safe_name,
        input_path=str(input_path),
    )

    with jobs_lock:
        jobs[job_id] = job

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()

    return {"job_id": job_id, "output_name": safe_name}


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
    print(f"DrakonSub web UI: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
