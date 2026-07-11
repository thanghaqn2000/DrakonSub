import json
import os
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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
    resolve_font_bold,
)
from .translation_topics import DEFAULT_TOPIC, list_topics, normalize_topic
from .url_import_service import (
    UrlImportError,
    detect_provider,
    validate_url_with_selected_provider,
    validate_video_url,
)
from .utils import hex_color_to_ass
from .voiceover.from_video import (
    VOICEOVER_TTS_STAGE_FROM_VIDEO,
    prepare_voiceover_srt_from_video,
)
from .voiceover.job_service import (
    VoiceoverJobError,
    VoiceoverJobOptions,
    run_voiceover_job,
)
from .voiceover.saydi_tts import (
    SaydiConfigError,
    SAYDI_SPEED_MAX,
    SAYDI_SPEED_MIN,
    load_saydi_config,
    validate_saydi_sample,
    validate_saydi_speed,
)
from .gemini_keys import gemini_configured, load_gemini_api_keys
from .voiceover.script_job import (
    DEFAULT_ORIGINAL_VOLUME,
    ScriptRenderOptions,
    cues_to_response,
    edited_voiceover_srt_path,
    effective_voiceover_srt_path,
    load_source_cues,
    load_voiceover_cues,
    render_script_job,
    run_script_generation_job,
    save_edited_voiceover_cues,
    source_srt_path,
    validate_edited_cues,
    voiceover_srt_path,
)

load_env()

STATIC_DIR = Path(__file__).parent / "static"
FONTS_DIR = Path(__file__).parent / "fonts"


def _resolve_jobs_root() -> Path:
    raw = os.getenv("DRAKONSUB_JOBS_ROOT", "").strip()
    if raw:
        return Path(raw)
    return Path(tempfile.gettempdir()) / "drakonsub_jobs"


JOBS_ROOT = _resolve_jobs_root()
def _resolve_voiceover_jobs_root() -> Path:
    raw = os.getenv("DRAKONSUB_VOICEOVER_JOBS_ROOT", "").strip()
    if raw:
        return Path(raw)
    return Path("voiceover_jobs")


VOICEOVER_JOBS_ROOT = _resolve_voiceover_jobs_root()
JOB_META_FILENAME = "job.json"
JOB_RELOAD_MESSAGE = "Không tìm thấy video đã tải. Vui lòng tải lại video từ link."
TRANSLATION_ENGINE_LABELS = {
    "openai": "OpenAI",
    "gemini": "Gemini",
    "google": "Google Translate",
}


class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
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
    url_provider: Optional[str] = None


jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()

app = FastAPI(title="DrakonSub")


def _voiceover_validate_job_id(job_id: str) -> str:
    raw = (job_id or "").strip()
    if (
        not raw
        or "/" in raw
        or "\\" in raw
        or ".." in raw
        or Path(raw).is_absolute()
    ):
        raise HTTPException(404, "Voiceover job not found")
    return raw


def _voiceover_job_dir(job_id: str) -> Path:
    return VOICEOVER_JOBS_ROOT / _voiceover_validate_job_id(job_id)


def _voiceover_job_json_path(job_id: str) -> Path:
    return _voiceover_job_dir(job_id) / "job.json"


def _sanitize_voiceover_error(message: str) -> str:
    text = (message or "").strip()
    for token_key in (
        "SAYDI_TTS_API_TOKEN",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_1",
        "GEMINI_API_KEY_2",
        "GEMINI_API_KEY_3",
        "GEMINI_API_KEY_4",
    ):
        text = text.replace(os.getenv(token_key, ""), "") if os.getenv(token_key, "") else text
    for api_key in load_gemini_api_keys():
        if api_key:
            text = text.replace(api_key, "")
    return text or "Voiceover job failed"


def _parse_optional_saydi_sample(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise HTTPException(400, "Giọng đọc Saydi không hợp lệ. Vui lòng kiểm tra mã giọng/sample.")
    try:
        return validate_saydi_sample(raw)
    except SaydiConfigError as exc:
        raise HTTPException(400, str(exc)) from exc


def _parse_optional_saydi_speed(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        return validate_saydi_speed(raw)
    except SaydiConfigError as exc:
        raise HTTPException(400, str(exc)) from exc


def _voiceover_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


VOICEOVER_STAGE_PROGRESS = {
    "queued": 0,
    "starting": 5,
    "extracting_audio": 10,
    "transcribing": 25,
    "translating_voiceover": 45,
    "script_ready": 50,
    "preparing_text": 15,
    "generating_voice": 35,
    "mixing_audio": 80,
    "completed": 100,
    "failed": 100,
}


def _voiceover_update_job_json(job_id: str, updates: Dict[str, Any]) -> None:
    payload = None
    for _ in range(8):
        payload = _read_voiceover_job_json(job_id)
        if payload is not None:
            break
        time.sleep(0.025)
    if payload is None:
        payload = {"job_id": job_id}
        if "job_type" in updates:
            payload["job_type"] = updates["job_type"]
    payload.update(updates)
    payload["updated_at"] = _voiceover_utc_now()
    _write_voiceover_job_json(job_id, payload)


def _voiceover_normalize_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    status = payload.get("status", "processing")
    output_path = Path(payload.get("output_video") or "")
    manifest_path = Path(payload.get("manifest") or "")
    if status == "processing" and output_path.is_file() and manifest_path.is_file():
        payload = {**payload, "status": "completed", "stage": "completed", "progress_percent": 100}
    return payload


def _voiceover_build_status_response(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = _voiceover_normalize_status(payload)
    status = payload.get("status", "processing")
    output_path = Path(payload.get("output_video") or "")
    manifest_path = Path(payload.get("manifest") or "")
    output_ready = status == "completed" and output_path.is_file()
    manifest_ready = status == "completed" and manifest_path.is_file()
    return {
        "job_id": job_id,
        "status": status,
        "stage": payload.get("stage"),
        "progress_percent": payload.get("progress_percent", VOICEOVER_STAGE_PROGRESS.get(payload.get("stage") or "", 0)),
        "summary": payload.get("summary"),
        "error": payload.get("error"),
        "output_ready": output_ready,
        "manifest_ready": manifest_ready,
        "output_video_url": f"/api/voiceover/jobs/{job_id}/output-video" if output_ready else None,
        "manifest_url": f"/api/voiceover/jobs/{job_id}/manifest" if manifest_ready else None,
        "status_url": f"/api/voiceover/jobs/{job_id}",
    }


def _voiceover_build_script_status_response(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = _voiceover_normalize_status(payload)
    status = payload.get("status", "processing")
    output_path = Path(payload.get("output_video") or "")
    manifest_path = Path(payload.get("manifest") or "")
    output_ready = status == "completed" and output_path.is_file()
    manifest_ready = status == "completed" and manifest_path.is_file()
    voiceover_srt = voiceover_srt_path(_voiceover_job_dir(job_id))
    edited_srt = edited_voiceover_srt_path(_voiceover_job_dir(job_id))
    source_srt = source_srt_path(_voiceover_job_dir(job_id))
    voiceover_ready = voiceover_srt.is_file()
    source_ready = source_srt.is_file()
    return {
        "job_id": job_id,
        "status": status,
        "stage": payload.get("stage"),
        "progress_percent": payload.get(
            "progress_percent", VOICEOVER_STAGE_PROGRESS.get(payload.get("stage") or "", 0)
        ),
        "source_srt_ready": source_ready,
        "voiceover_srt_ready": voiceover_ready,
        "edited_srt_ready": bool(payload.get("edited_srt_ready")) or edited_srt.is_file(),
        "cue_count": payload.get("cue_count"),
        "summary": payload.get("summary"),
        "error": payload.get("error"),
        "output_ready": output_ready,
        "manifest_ready": manifest_ready,
        "output_video_url": f"/api/voiceover/jobs/{job_id}/output-video" if output_ready else None,
        "manifest_url": f"/api/voiceover/jobs/{job_id}/manifest" if manifest_ready else None,
        "voiceover_srt_download_url": (
            f"/api/voiceover/script-jobs/{job_id}/download/voiceover-srt" if voiceover_ready else None
        ),
        "source_srt_download_url": (
            f"/api/voiceover/script-jobs/{job_id}/download/source-srt" if source_ready else None
        ),
        "cues_url": f"/api/voiceover/script-jobs/{job_id}/cues",
        "status_url": f"/api/voiceover/script-jobs/{job_id}",
        "url_provider": payload.get("url_provider"),
        "source_title": payload.get("source_title"),
    }


def _run_script_generation_background(
    job_id: str,
    input_video: Path,
    job_dir: Path,
    *,
    voiceover_topic: str,
) -> None:
    def stage_callback(stage: str, percent: int) -> None:
        _voiceover_update_job_json(
            job_id,
            {"status": "processing", "stage": stage, "progress_percent": percent},
        )

    try:
        stage_callback("starting", 5)
        source_srt, voiceover_srt = run_script_generation_job(
            input_video,
            job_dir,
            voiceover_topic,
            on_progress=stage_callback,
        )
        cue_count = len(load_srt(voiceover_srt))
        _voiceover_update_job_json(
            job_id,
            {
                "status": "script_ready",
                "stage": "script_ready",
                "progress_percent": 50,
                "error": None,
                "source_srt": str(source_srt),
                "voiceover_srt": str(voiceover_srt),
                "source_srt_ready": True,
                "voiceover_srt_ready": True,
                "edited_srt_ready": False,
                "cue_count": cue_count,
            },
        )
    except VoiceoverJobError as exc:
        _voiceover_update_job_json(
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress_percent": 100,
                "error": _sanitize_voiceover_error(str(exc)),
            },
        )
    except Exception as exc:
        _voiceover_update_job_json(
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress_percent": 100,
                "error": _sanitize_voiceover_error(str(exc)),
            },
        )


def _run_script_generation_from_url_background(
    job_id: str,
    url: str,
    job_dir: Path,
    *,
    voiceover_topic: str,
) -> None:
    from .url_import_service import cleanup_partial_downloads, download_video_from_url

    input_path = job_dir / "input.mp4"
    try:
        _voiceover_update_job_json(
            job_id,
            {
                "status": "processing",
                "stage": "downloading",
                "progress_percent": 2,
                "error": None,
            },
        )
        download_result = download_video_from_url(
            url,
            job_dir,
            output_filename="input.mp4",
        )
        resolved = Path(download_result.get("path") or input_path)
        if not resolved.is_file():
            raise UrlImportError(
                "Tải video thất bại. Vui lòng thử lại hoặc tải file video trực tiếp."
            )
        if resolved.resolve() != input_path.resolve():
            resolved.replace(input_path)
        _voiceover_update_job_json(
            job_id,
            {
                "input_video": str(input_path),
                "url_provider": download_result.get("provider"),
                "source_title": download_result.get("title"),
                "stage": "starting",
                "progress_percent": 5,
            },
        )
    except UrlImportError as exc:
        cleanup_partial_downloads(job_dir)
        _voiceover_update_job_json(
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress_percent": 100,
                "error": _sanitize_voiceover_error(str(exc)),
            },
        )
        return
    except Exception as exc:
        cleanup_partial_downloads(job_dir)
        _voiceover_update_job_json(
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress_percent": 100,
                "error": _sanitize_voiceover_error(str(exc)),
            },
        )
        return

    _run_script_generation_background(
        job_id,
        input_path,
        job_dir,
        voiceover_topic=voiceover_topic,
    )


def _run_script_render_background(
    job_id: str,
    job_dir: Path,
    options: ScriptRenderOptions,
) -> None:
    output_path = job_dir / "output_voiceover.mp4"
    manifest_path = job_dir / "manifest.json"

    def progress_callback(stage: str, percent: int) -> None:
        mapped_stage, mapped_percent = VOICEOVER_TTS_STAGE_FROM_VIDEO.get(
            stage, (stage, percent)
        )
        _voiceover_update_job_json(
            job_id,
            {
                "status": "rendering",
                "stage": mapped_stage,
                "progress_percent": mapped_percent,
            },
        )

    try:
        _voiceover_update_job_json(
            job_id,
            {"status": "rendering", "stage": "starting", "progress_percent": 55},
        )
        result = render_script_job(job_dir, options, on_progress=progress_callback)
        _voiceover_update_job_json(
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress_percent": 100,
                "error": None,
                "output_video": str(result.output_video),
                "manifest": str(result.manifest_path),
                "summary": result.summary,
            },
        )
    except VoiceoverJobError as exc:
        _voiceover_update_job_json(
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress_percent": 100,
                "error": _sanitize_voiceover_error(str(exc)),
            },
        )
    except Exception as exc:
        _voiceover_update_job_json(
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress_percent": 100,
                "error": _sanitize_voiceover_error(str(exc)),
            },
        )


def _run_voiceover_from_video_background(
    job_id: str,
    input_video: Path,
    job_dir: Path,
    *,
    prepare_text: bool,
    voiceover_topic: str,
    original_volume: float,
    voice_volume: float,
    max_chars_per_second: float,
    min_gap_ms: int,
    max_borrow_after_ms: int,
    severe_overflow_ms: int,
    saydi_sample: Optional[str] = None,
    saydi_speed: Optional[float] = None,
) -> None:
    output_path = job_dir / "output_voiceover.mp4"
    manifest_path = job_dir / "manifest.json"
    prepared_srt_path = job_dir / "prepared_voiceover.srt"

    def stage_callback(stage: str, percent: int) -> None:
        _voiceover_update_job_json(
            job_id,
            {"status": "processing", "stage": stage, "progress_percent": percent},
        )

    def tts_progress_callback(stage: str, percent: int) -> None:
        mapped_stage, mapped_percent = VOICEOVER_TTS_STAGE_FROM_VIDEO.get(
            stage, (stage, percent)
        )
        stage_callback(mapped_stage, mapped_percent)

    try:
        stage_callback("starting", 5)
        _, source_srt, voiceover_srt = prepare_voiceover_srt_from_video(
            input_video,
            job_dir,
            voiceover_topic,
            on_progress=stage_callback,
        )
        _voiceover_update_job_json(
            job_id,
            {
                "source_srt": str(source_srt),
                "voiceover_srt": str(voiceover_srt),
            },
        )

        options = VoiceoverJobOptions(
            input_video=input_video,
            voiceover_srt=voiceover_srt,
            output_video=output_path,
            workdir=job_dir,
            original_volume=original_volume,
            voice_volume=voice_volume,
            prepare_text=prepare_text,
            voiceover_topic=voiceover_topic,
            max_chars_per_second=max_chars_per_second,
            prepared_srt_output=prepared_srt_path if prepare_text else None,
            min_gap_ms=min_gap_ms,
            max_borrow_after_ms=max_borrow_after_ms,
            severe_overflow_ms=severe_overflow_ms,
            saydi_sample=saydi_sample,
            saydi_speed=saydi_speed,
            force=True,
        )
        result = run_voiceover_job(options, progress_callback=tts_progress_callback)
        _voiceover_update_job_json(
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress_percent": 100,
                "error": None,
                "output_video": str(result.output_video),
                "manifest": str(result.manifest_path),
                "summary": result.summary,
            },
        )
    except VoiceoverJobError as exc:
        _voiceover_update_job_json(
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress_percent": 100,
                "error": _sanitize_voiceover_error(str(exc)),
                "summary": None,
            },
        )
    except Exception as exc:
        _voiceover_update_job_json(
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress_percent": 100,
                "error": _sanitize_voiceover_error(str(exc)),
                "summary": None,
            },
        )


def _run_voiceover_job_background(job_id: str, options: VoiceoverJobOptions) -> None:
    def progress_callback(stage: str, percent: int) -> None:
        _voiceover_update_job_json(
            job_id,
            {
                "status": "processing",
                "stage": stage,
                "progress_percent": percent,
            },
        )

    try:
        result = run_voiceover_job(options, progress_callback=progress_callback)
        _voiceover_update_job_json(
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress_percent": 100,
                "error": None,
                "output_video": str(result.output_video),
                "manifest": str(result.manifest_path),
                "summary": result.summary,
            },
        )
    except VoiceoverJobError as exc:
        _voiceover_update_job_json(
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress_percent": 100,
                "error": _sanitize_voiceover_error(str(exc)),
                "summary": None,
            },
        )
    except Exception as exc:
        _voiceover_update_job_json(
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress_percent": 100,
                "error": _sanitize_voiceover_error(str(exc)),
                "summary": None,
            },
        )


def _write_voiceover_job_json(job_id: str, payload: Dict[str, Any]) -> None:
    job_dir = _voiceover_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    path = _voiceover_job_json_path(job_id)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _read_voiceover_job_json(job_id: str) -> Optional[Dict[str, Any]]:
    path = _voiceover_job_json_path(job_id)
    for attempt in range(6):
        if not path.is_file():
            if attempt < 5:
                time.sleep(0.03)
                continue
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if attempt < 5:
                time.sleep(0.03)
                continue
            return None
    return None


def _voiceover_fallback_payload(job_id: str) -> Optional[Dict[str, Any]]:
    job_dir = _voiceover_job_dir(job_id)
    manifest_path = job_dir / "manifest.json"
    output_path = job_dir / "output_voiceover.mp4"
    if not manifest_path.is_file() and not output_path.is_file():
        return None
    summary = None
    if manifest_path.is_file():
        try:
            summary = json.loads(manifest_path.read_text(encoding="utf-8")).get("summary")
        except (OSError, json.JSONDecodeError):
            summary = None
    return {
        "job_id": job_id,
        "status": "completed" if output_path.is_file() else "processing",
        "stage": "completed" if output_path.is_file() else "processing",
        "progress_percent": 100 if output_path.is_file() else 0,
        "created_at": None,
        "updated_at": None,
        "error": None,
        "input_video": None,
        "voiceover_srt": None,
        "output_video": str(output_path),
        "manifest": str(manifest_path),
        "summary": summary,
    }


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


def _job_meta_path(job_id: str) -> Path:
    return _job_paths(job_id)[0] / JOB_META_FILENAME


def _job_to_meta_dict(job: Job) -> Dict[str, Any]:
    return {
        "job_id": job.id,
        "source": "url",
        "provider": job.url_provider,
        "status": job.status.value,
        "input_filename": "input.mp4",
        "output_name": job.output_name,
        "source_language": job.source_language,
        "topic": job.translation_topic,
        "translation_engine": job.translation_engine,
        "subtitle_font_size": job.subtitle_font_size,
        "subtitle_font_color": job.subtitle_font_color,
    }


def _write_job_meta(job: Job) -> None:
    job_dir = _job_paths(job.id)[0]
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(_job_meta_path(job.id), "w", encoding="utf-8") as fp:
        json.dump(_job_to_meta_dict(job), fp, ensure_ascii=False, indent=2)


def _read_job_meta(job_id: str) -> Optional[Dict[str, Any]]:
    meta_path = _job_meta_path(job_id)
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, encoding="utf-8") as fp:
            return json.load(fp)
    except (json.JSONDecodeError, OSError):
        return None


def _job_from_meta(job_id: str, meta: Dict[str, Any]) -> Job:
    job_dir, source_srt_path, srt_path, layout_path = _job_paths(job_id)
    input_name = str(meta.get("input_filename") or "input.mp4")
    input_path = job_dir / input_name
    status_raw = str(meta.get("status") or JobStatus.DOWNLOADED.value)
    try:
        status = JobStatus(status_raw)
    except ValueError:
        status = JobStatus.DOWNLOADED
    if status not in (JobStatus.DOWNLOADED, JobStatus.ERROR):
        status = JobStatus.DOWNLOADED
    return Job(
        id=job_id,
        status=status,
        message="Đã tải video thành công."
        if status == JobStatus.DOWNLOADED
        else "Waiting...",
        progress=100 if status == JobStatus.DOWNLOADED else 0,
        output_name=str(meta.get("output_name") or "url-video-vietsub"),
        input_path=str(input_path) if input_path.is_file() else None,
        source_srt_path=str(source_srt_path),
        srt_path=str(srt_path),
        layout_path=str(layout_path),
        translation_topic=str(meta.get("topic") or DEFAULT_TOPIC),
        translation_engine=str(meta.get("translation_engine") or get_translation_engine()),
        subtitle_font_size=meta.get("subtitle_font_size"),
        subtitle_font_color=meta.get("subtitle_font_color"),
        source_language=str(meta.get("source_language") or "en"),
        url_provider=meta.get("provider"),
    )


def _rehydrate_url_job(job_id: str) -> Optional[Job]:
    try:
        job_id = _validate_job_id(job_id)
    except HTTPException:
        return None

    job_dir = JOBS_ROOT / job_id
    input_path = job_dir / "input.mp4"
    if not input_path.is_file():
        return None

    meta = _read_job_meta(job_id)
    if meta:
        job = _job_from_meta(job_id, meta)
    else:
        _, source_srt_path, srt_path, layout_path = _job_paths(job_id)
        job = Job(
            id=job_id,
            status=JobStatus.DOWNLOADED,
            message="Đã tải video thành công.",
            progress=100,
            output_name="url-video-vietsub",
            input_path=str(input_path),
            source_srt_path=str(source_srt_path),
            srt_path=str(srt_path),
            layout_path=str(layout_path),
            translation_topic=DEFAULT_TOPIC,
            translation_engine=get_translation_engine(),
            source_language="en",
        )

    if not job.input_path or not os.path.isfile(job.input_path):
        return None

    with jobs_lock:
        jobs[job_id] = job
    return job


def _get_job_or_rehydrate(job_id: str) -> Optional[Job]:
    with jobs_lock:
        job = jobs.get(job_id)
    if job:
        return job
    return _rehydrate_url_job(job_id)


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
    layout["font_bold"] = resolve_font_bold(layout, layout["font_family"])

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
    input_ready = bool(job.input_path and os.path.isfile(job.input_path))
    can_render_again = (
        job.status in (JobStatus.DONE, JobStatus.ERROR)
        and job.layout_saved
        and job.srt_path
        and os.path.isfile(job.srt_path)
        and job.input_path
        and os.path.isfile(job.input_path)
    )
    can_process_subtitles = job.status == JobStatus.DOWNLOADED and input_ready
    return {
        "job_id": job.id,
        "status": job.status.value,
        "message": job.message,
        "progress": job.progress,
        "output_name": job.output_name,
        "error": job.error,
        "layout_saved": job.layout_saved,
        "input_ready": input_ready,
        "provider": job.url_provider,
        "can_process_subtitles": can_process_subtitles,
        "input_video_url": f"/api/jobs/{job.id}/input-video" if input_ready else None,
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


def _parse_job_creation_fields(
    *,
    output_name: str,
    topic: str,
    source_language: str,
    translation_engine: str,
    font_size: Optional[str] = None,
    font_color: Optional[str] = None,
    fallback_name: str = "video-vietsub",
) -> Dict[str, Any]:
    safe_topic = normalize_topic(topic)
    if source_language not in ("en", "vi"):
        raise ValueError("source_language must be 'en' or 'vi'")
    engine = (translation_engine or "").strip().lower()
    if engine not in SUPPORTED_TRANSLATION_ENGINES:
        raise ValueError(
            "translation_engine must be one of: " + ", ".join(SUPPORTED_TRANSLATION_ENGINES)
        )
    defaults = SubtitleConfig.from_env()
    parsed_font_size = (
        parse_font_size(font_size, defaults.subtitle_font_size)
        if font_size and str(font_size).strip()
        else None
    )
    parsed_font_color = (
        parse_font_color(font_color, defaults.subtitle_font_color)
        if font_color and str(font_color).strip()
        else None
    )
    final_name = prepare_output_name(output_name, fallback_name)
    return {
        "output_name": final_name,
        "translation_topic": safe_topic,
        "translation_engine": engine,
        "subtitle_font_size": parsed_font_size,
        "subtitle_font_color": parsed_font_color,
        "source_language": source_language,
    }


def _run_url_import_job(job_id: str, url: str) -> None:
    from .url_import_service import download_video_from_url

    try:
        with jobs_lock:
            jobs[job_id].status = JobStatus.DOWNLOADING
            jobs[job_id].message = "Đang tải video..."
            jobs[job_id].progress = 5
            job_dir = _job_paths(job_id)[0]

        result = download_video_from_url(url, job_dir)

        with jobs_lock:
            jobs[job_id].input_path = result["path"]
            jobs[job_id].url_provider = result.get("provider")
            jobs[job_id].status = JobStatus.DOWNLOADED
            jobs[job_id].message = "Đã tải video thành công."
            jobs[job_id].progress = 100
            _write_job_meta(jobs[job_id])
    except UrlImportError as exc:
        from .url_import_service import cleanup_partial_downloads

        cleanup_partial_downloads(_job_paths(job_id)[0])
        with jobs_lock:
            jobs[job_id].status = JobStatus.ERROR
            jobs[job_id].error = str(exc)
            jobs[job_id].message = str(exc)
            jobs[job_id].input_path = None
            jobs[job_id].progress = 0
    except Exception:  # noqa: BLE001 - worker boundary must report unexpected failures to the job.
        from .url_import_service import cleanup_partial_downloads

        traceback.print_exc()
        cleanup_partial_downloads(_job_paths(job_id)[0])
        with jobs_lock:
            jobs[job_id].status = JobStatus.ERROR
            jobs[job_id].error = (
                "Tải video thất bại. Vui lòng thử lại hoặc tải file video trực tiếp."
            )
            jobs[job_id].message = jobs[job_id].error
            jobs[job_id].input_path = None
            jobs[job_id].progress = 0


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


@app.get("/api/health")
def health_check():
    """Lightweight readiness probe for production smoke tests (no secrets)."""
    config = SubtitleConfig.from_env()
    jobs_writable = False
    try:
        JOBS_ROOT.mkdir(parents=True, exist_ok=True)
        probe = JOBS_ROOT / ".health_probe"
        probe.write_text("ok", encoding="utf-8")
        jobs_writable = probe.is_file()
        probe.unlink(missing_ok=True)
    except OSError:
        pass
    return {
        "status": "ok",
        "jobs_root": str(JOBS_ROOT),
        "jobs_root_writable": jobs_writable,
        "translation_engine": config.translation_engine,
        "whisper_model": config.model,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "gemini_configured": gemini_configured(),
    }


@app.get("/api/defaults")
def get_defaults():
    config = SubtitleConfig.from_env()
    return {
        "subtitle_font_size": config.subtitle_font_size,
        "subtitle_font_color": config.subtitle_font_color,
        "openai_model": get_openai_model(),
        "translation_engine": config.translation_engine,
        "translation_engines": list(SUPPORTED_TRANSLATION_ENGINES),
        "translation_engine_labels": {
            engine: TRANSLATION_ENGINE_LABELS.get(engine, engine.title())
            for engine in SUPPORTED_TRANSLATION_ENGINES
        },
        "default_layout": default_layout_dict(),
        "font_presets": list(FONT_FAMILY_CHOICES),
    }


@app.get("/api/voiceover/config")
def get_voiceover_config():
    cfg = load_saydi_config()
    return {
        "default_original_volume": DEFAULT_ORIGINAL_VOLUME,
        "default_voice_volume": 1.0,
        "default_saydi_sample": cfg.sample,
        "default_saydi_speed": cfg.speed,
        "saydi_speed_min": SAYDI_SPEED_MIN,
        "saydi_speed_max": SAYDI_SPEED_MAX,
    }


@app.post("/api/voiceover/jobs")
async def create_voiceover_job(
    input_video: UploadFile = File(...),
    voiceover_srt: UploadFile = File(...),
    prepare_text: bool = Form(True),
    voiceover_topic: str = Form("catholic"),
    original_volume: float = Form(DEFAULT_ORIGINAL_VOLUME),
    voice_volume: float = Form(1.00),
    max_chars_per_second: float = Form(13.0),
    min_gap_ms: int = Form(120),
    max_borrow_after_ms: int = Form(1200),
    severe_overflow_ms: int = Form(2000),
    saydi_sample: str = Form(""),
    saydi_speed: str = Form(""),
):
    if not input_video.filename or Path(input_video.filename).suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, "Unsupported input video format")
    if not voiceover_srt.filename or Path(voiceover_srt.filename).suffix.lower() != ".srt":
        raise HTTPException(400, "Unsupported voiceover SRT format")
    parsed_saydi_sample = _parse_optional_saydi_sample(saydi_sample)
    parsed_saydi_speed = _parse_optional_saydi_speed(saydi_speed)

    job_id = str(uuid.uuid4())
    job_dir = VOICEOVER_JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.mp4"
    srt_path = job_dir / "voiceover.srt"
    prepared_srt_path = job_dir / "prepared_voiceover.srt"
    output_path = job_dir / "output_voiceover.mp4"
    manifest_path = job_dir / "manifest.json"

    input_path.write_bytes(await input_video.read())
    srt_path.write_bytes(await voiceover_srt.read())

    now = _voiceover_utc_now()
    initial_payload = {
        "job_id": job_id,
        "status": "processing",
        "stage": "queued",
        "progress_percent": 0,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "input_video": str(input_path),
        "voiceover_srt": str(srt_path),
        "output_video": str(output_path),
        "manifest": str(manifest_path),
        "summary": None,
    }
    _write_voiceover_job_json(job_id, initial_payload)

    options = VoiceoverJobOptions(
        input_video=input_path,
        voiceover_srt=srt_path,
        output_video=output_path,
        workdir=job_dir,
        original_volume=original_volume,
        voice_volume=voice_volume,
        prepare_text=prepare_text,
        voiceover_topic=voiceover_topic,
        max_chars_per_second=max_chars_per_second,
        prepared_srt_output=prepared_srt_path if prepare_text else None,
        min_gap_ms=min_gap_ms,
        max_borrow_after_ms=max_borrow_after_ms,
        severe_overflow_ms=severe_overflow_ms,
        saydi_sample=parsed_saydi_sample,
        saydi_speed=parsed_saydi_speed,
        force=True,
    )
    threading.Thread(
        target=_run_voiceover_job_background,
        args=(job_id, options),
        daemon=True,
    ).start()

    return {
        "job_id": job_id,
        "status": "processing",
        "status_url": f"/api/voiceover/jobs/{job_id}",
        "message": "Đã bắt đầu tạo video thuyết minh.",
    }


@app.post("/api/voiceover/jobs/from-video")
async def create_voiceover_job_from_video(
    input_video: UploadFile = File(...),
    prepare_text: bool = Form(True),
    voiceover_topic: str = Form("catholic"),
    original_volume: float = Form(DEFAULT_ORIGINAL_VOLUME),
    voice_volume: float = Form(1.00),
    max_chars_per_second: float = Form(13.0),
    min_gap_ms: int = Form(120),
    max_borrow_after_ms: int = Form(1200),
    severe_overflow_ms: int = Form(2000),
    saydi_sample: str = Form(""),
    saydi_speed: str = Form(""),
):
    if not input_video.filename or Path(input_video.filename).suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, "Unsupported input video format")
    parsed_saydi_sample = _parse_optional_saydi_sample(saydi_sample)
    parsed_saydi_speed = _parse_optional_saydi_speed(saydi_speed)

    job_id = str(uuid.uuid4())
    job_dir = VOICEOVER_JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.mp4"
    output_path = job_dir / "output_voiceover.mp4"
    manifest_path = job_dir / "manifest.json"

    input_path.write_bytes(await input_video.read())

    now = _voiceover_utc_now()
    initial_payload = {
        "job_id": job_id,
        "status": "processing",
        "stage": "queued",
        "progress_percent": 0,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "input_video": str(input_path),
        "voiceover_srt": None,
        "output_video": str(output_path),
        "manifest": str(manifest_path),
        "summary": None,
        "source": "from-video",
    }
    _write_voiceover_job_json(job_id, initial_payload)

    threading.Thread(
        target=_run_voiceover_from_video_background,
        kwargs={
            "job_id": job_id,
            "input_video": input_path,
            "job_dir": job_dir,
            "prepare_text": prepare_text,
            "voiceover_topic": voiceover_topic,
            "original_volume": original_volume,
            "voice_volume": voice_volume,
            "max_chars_per_second": max_chars_per_second,
            "min_gap_ms": min_gap_ms,
            "max_borrow_after_ms": max_borrow_after_ms,
            "severe_overflow_ms": severe_overflow_ms,
            "saydi_sample": parsed_saydi_sample,
            "saydi_speed": parsed_saydi_speed,
        },
        daemon=True,
    ).start()

    return {
        "job_id": job_id,
        "status": "processing",
        "status_url": f"/api/voiceover/jobs/{job_id}",
        "message": "Đã bắt đầu tạo video thuyết minh.",
    }


@app.post("/api/voiceover/script-jobs/from-video")
async def create_voiceover_script_job_from_video(
    input_video: UploadFile = File(...),
    voiceover_topic: str = Form("catholic"),
    max_chars_per_second: float = Form(13.0),
):
    if not input_video.filename or Path(input_video.filename).suffix.lower() not in {
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
    }:
        raise HTTPException(400, "Unsupported input video format")

    job_id = str(uuid.uuid4())
    job_dir = VOICEOVER_JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.mp4"
    output_path = job_dir / "output_voiceover.mp4"
    manifest_path = job_dir / "manifest.json"
    input_path.write_bytes(await input_video.read())

    now = _voiceover_utc_now()
    initial_payload = {
        "job_id": job_id,
        "job_type": "script",
        "status": "processing",
        "stage": "queued",
        "progress_percent": 0,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "input_video": str(input_path),
        "voiceover_srt": None,
        "output_video": str(output_path),
        "manifest": str(manifest_path),
        "summary": None,
        "source_srt_ready": False,
        "voiceover_srt_ready": False,
        "edited_srt_ready": False,
        "cue_count": None,
        "max_chars_per_second": max_chars_per_second,
        "voiceover_topic": voiceover_topic,
    }
    _write_voiceover_job_json(job_id, initial_payload)

    threading.Thread(
        target=_run_script_generation_background,
        kwargs={
            "job_id": job_id,
            "input_video": input_path,
            "job_dir": job_dir,
            "voiceover_topic": voiceover_topic,
        },
        daemon=True,
    ).start()

    return {
        "job_id": job_id,
        "status": "processing",
        "status_url": f"/api/voiceover/script-jobs/{job_id}",
        "cues_url": f"/api/voiceover/script-jobs/{job_id}/cues",
        "message": "Đã bắt đầu tạo lời thuyết minh.",
    }


@app.post("/api/voiceover/script-jobs/from-url")
def create_voiceover_script_job_from_url(body: dict = Body(default_factory=dict)):
    url = str(body.get("url", "")).strip()
    voiceover_topic = str(body.get("voiceover_topic", "catholic")).strip() or "catholic"
    try:
        max_chars_per_second = float(body.get("max_chars_per_second", 13.0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "max_chars_per_second không hợp lệ.") from exc

    try:
        safe_url = validate_video_url(url)
        provider = detect_provider(safe_url)
    except UrlImportError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = str(uuid.uuid4())
    job_dir = VOICEOVER_JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.mp4"
    output_path = job_dir / "output_voiceover.mp4"
    manifest_path = job_dir / "manifest.json"

    now = _voiceover_utc_now()
    initial_payload = {
        "job_id": job_id,
        "job_type": "script",
        "status": "processing",
        "stage": "downloading",
        "progress_percent": 0,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "input_video": str(input_path),
        "voiceover_srt": None,
        "output_video": str(output_path),
        "manifest": str(manifest_path),
        "summary": None,
        "source_srt_ready": False,
        "voiceover_srt_ready": False,
        "edited_srt_ready": False,
        "cue_count": None,
        "max_chars_per_second": max_chars_per_second,
        "voiceover_topic": voiceover_topic,
        "source_url": safe_url,
        "url_provider": provider,
    }
    _write_voiceover_job_json(job_id, initial_payload)

    threading.Thread(
        target=_run_script_generation_from_url_background,
        kwargs={
            "job_id": job_id,
            "url": safe_url,
            "job_dir": job_dir,
            "voiceover_topic": voiceover_topic,
        },
        daemon=True,
    ).start()

    return {
        "job_id": job_id,
        "status": "processing",
        "provider": provider,
        "status_url": f"/api/voiceover/script-jobs/{job_id}",
        "cues_url": f"/api/voiceover/script-jobs/{job_id}/cues",
        "message": "Đã bắt đầu tải video và tạo lời thuyết minh.",
    }


@app.get("/api/voiceover/script-jobs/{job_id}")
def get_voiceover_script_job(job_id: str):
    _voiceover_validate_job_id(job_id)
    payload = _read_voiceover_job_json(job_id)
    if not payload or payload.get("job_type") != "script":
        raise HTTPException(404, "Voiceover script job not found")
    return _voiceover_build_script_status_response(job_id, payload)


@app.get("/api/voiceover/script-jobs/{job_id}/cues")
def get_voiceover_script_job_cues(job_id: str):
    _voiceover_validate_job_id(job_id)
    payload = _read_voiceover_job_json(job_id)
    if not payload or payload.get("job_type") != "script":
        raise HTTPException(404, "Voiceover script job not found")
    if payload.get("status") not in {"script_ready", "rendering", "completed"}:
        raise HTTPException(409, "Lời thuyết minh chưa sẵn sàng để chỉnh sửa.")
    job_dir = _voiceover_job_dir(job_id)
    try:
        cues, source = load_voiceover_cues(job_dir)
        source_cues = load_source_cues(job_dir)
    except VoiceoverJobError as exc:
        raise HTTPException(409, _sanitize_voiceover_error(str(exc))) from exc
    except SubtitleEditError as exc:
        raise HTTPException(400, str(exc)) from exc
    return cues_to_response(job_id, cues, source, source_cues)


@app.get("/api/voiceover/script-jobs/{job_id}/download/voiceover-srt")
def download_voiceover_script_srt(job_id: str):
    _voiceover_validate_job_id(job_id)
    payload = _read_voiceover_job_json(job_id)
    if not payload or payload.get("job_type") != "script":
        raise HTTPException(404, "Voiceover script job not found")
    if payload.get("status") not in {"script_ready", "rendering", "completed"}:
        raise HTTPException(409, "Lời thuyết minh chưa sẵn sàng để tải.")
    job_dir = _voiceover_job_dir(job_id)
    path = effective_voiceover_srt_path(job_dir)
    if not path.is_file():
        raise HTTPException(404, "Voiceover SRT not found")
    return FileResponse(
        str(path),
        media_type="application/x-subrip",
        filename=f"voiceover-{job_id[:8]}.srt",
    )


@app.get("/api/voiceover/script-jobs/{job_id}/download/source-srt")
def download_voiceover_source_srt(job_id: str):
    _voiceover_validate_job_id(job_id)
    payload = _read_voiceover_job_json(job_id)
    if not payload or payload.get("job_type") != "script":
        raise HTTPException(404, "Voiceover script job not found")
    if payload.get("status") not in {"script_ready", "rendering", "completed"}:
        raise HTTPException(409, "Lời thuyết minh chưa sẵn sàng để tải.")
    job_dir = _voiceover_job_dir(job_id)
    path = source_srt_path(job_dir)
    if not path.is_file():
        raise HTTPException(404, "Source SRT not found")
    return FileResponse(
        str(path),
        media_type="application/x-subrip",
        filename=f"source-{job_id[:8]}.srt",
    )


@app.put("/api/voiceover/script-jobs/{job_id}/cues")
def save_voiceover_script_job_cues(job_id: str, body: dict = Body(...)):
    _voiceover_validate_job_id(job_id)
    payload = _read_voiceover_job_json(job_id)
    if not payload or payload.get("job_type") != "script":
        raise HTTPException(404, "Voiceover script job not found")
    if payload.get("status") not in {"script_ready", "rendering", "completed"}:
        raise HTTPException(409, "Lời thuyết minh chưa sẵn sàng để chỉnh sửa.")
    if payload.get("status") == "rendering":
        raise HTTPException(409, "Job đang render, không thể chỉnh sửa.")

    job_dir = _voiceover_job_dir(job_id)
    base_path = voiceover_srt_path(job_dir)
    if not base_path.is_file():
        raise HTTPException(409, "Chưa có file lời thuyết minh.")

    try:
        original_cues = load_srt(base_path)
        submitted = body.get("cues")
        if not isinstance(submitted, list):
            raise SubtitleEditError("Dữ liệu cue không hợp lệ.")
        updated = validate_edited_cues(original_cues, submitted)
        save_edited_voiceover_cues(job_dir, updated)
        _voiceover_update_job_json(
            job_id,
            {
                "edited_srt_ready": True,
                "cue_count": len(updated),
            },
        )
    except SubtitleEditError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "job_id": job_id,
        "edited_srt_ready": True,
        "cue_count": len(updated),
        "message": "Đã lưu lời thuyết minh đã chỉnh sửa.",
    }


@app.post("/api/voiceover/script-jobs/{job_id}/render")
def render_voiceover_script_job(job_id: str, body: dict = Body(default_factory=dict)):
    _voiceover_validate_job_id(job_id)
    payload = _read_voiceover_job_json(job_id)
    if not payload or payload.get("job_type") != "script":
        raise HTTPException(404, "Voiceover script job not found")
    if payload.get("status") == "rendering":
        raise HTTPException(409, "Job đang render, vui lòng đợi hoàn tất.")
    if payload.get("status") == "processing":
        raise HTTPException(409, "Lời thuyết minh chưa sẵn sàng để render.")
    if payload.get("status") not in {"script_ready", "failed", "completed"}:
        raise HTTPException(409, "Job chưa sẵn sàng để render.")

    job_dir = _voiceover_job_dir(job_id)
    if not voiceover_srt_path(job_dir).is_file():
        raise HTTPException(409, "Chưa có file lời thuyết minh.")

    options = ScriptRenderOptions(
        original_volume=float(body.get("original_volume", DEFAULT_ORIGINAL_VOLUME)),
        voice_volume=float(body.get("voice_volume", 1.00)),
        prepare_text=bool(body.get("prepare_text", True)),
        voiceover_topic=str(
            body.get("voiceover_topic", payload.get("voiceover_topic", "catholic"))
        ),
        max_chars_per_second=float(
            body.get("max_chars_per_second", payload.get("max_chars_per_second", 13.0))
        ),
        min_gap_ms=int(body.get("min_gap_ms", 120)),
        max_borrow_after_ms=int(body.get("max_borrow_after_ms", 1200)),
        severe_overflow_ms=int(body.get("severe_overflow_ms", 2000)),
        saydi_sample=_parse_optional_saydi_sample(body.get("saydi_sample")),
        saydi_speed=_parse_optional_saydi_speed(body.get("saydi_speed")),
    )

    resolved_saydi = load_saydi_config(
        sample_override=options.saydi_sample,
        speed_override=options.saydi_speed,
    )
    _voiceover_update_job_json(
        job_id,
        {
            "status": "rendering",
            "stage": "starting",
            "progress_percent": 55,
            "error": None,
            "render_options": {
                "original_volume": options.original_volume,
                "voice_volume": options.voice_volume,
                "prepare_text": options.prepare_text,
                "saydi_sample": resolved_saydi.sample,
                "saydi_speed": resolved_saydi.speed,
            },
        },
    )

    threading.Thread(
        target=_run_script_render_background,
        kwargs={"job_id": job_id, "job_dir": job_dir, "options": options},
        daemon=True,
    ).start()

    return {
        "job_id": job_id,
        "status": "rendering",
        "status_url": f"/api/voiceover/script-jobs/{job_id}",
        "output_video_url": f"/api/voiceover/jobs/{job_id}/output-video",
        "manifest_url": f"/api/voiceover/jobs/{job_id}/manifest",
    }


@app.get("/api/voiceover/jobs/{job_id}")
def get_voiceover_job(job_id: str):
    _voiceover_validate_job_id(job_id)
    payload = _read_voiceover_job_json(job_id) or _voiceover_fallback_payload(job_id)
    if not payload:
        raise HTTPException(404, "Voiceover job not found")
    return _voiceover_build_status_response(job_id, payload)


@app.get("/api/voiceover/jobs/{job_id}/manifest")
def get_voiceover_manifest(job_id: str):
    _voiceover_validate_job_id(job_id)
    payload = _read_voiceover_job_json(job_id) or _voiceover_fallback_payload(job_id)
    if not payload:
        raise HTTPException(404, "Voiceover job not found")
    status = _voiceover_build_status_response(job_id, payload)
    if not status["manifest_ready"]:
        raise HTTPException(409, "Manifest chưa sẵn sàng. Vui lòng đợi job hoàn tất.")
    manifest_path = Path(payload.get("manifest") or "")
    if not manifest_path.is_file():
        raise HTTPException(404, "Manifest not found")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@app.get("/api/voiceover/jobs/{job_id}/output-video")
def get_voiceover_output_video(job_id: str):
    _voiceover_validate_job_id(job_id)
    payload = _read_voiceover_job_json(job_id) or _voiceover_fallback_payload(job_id)
    if not payload:
        raise HTTPException(404, "Voiceover job not found")
    status = _voiceover_build_status_response(job_id, payload)
    if not status["output_ready"]:
        raise HTTPException(409, "Video thuyết minh chưa sẵn sàng. Vui lòng đợi job hoàn tất.")
    output_path = Path(payload.get("output_video") or "")
    if not output_path.is_file():
        raise HTTPException(404, "Output video not found")
    return FileResponse(str(output_path), media_type="video/mp4", filename=output_path.name)


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
        fields = _parse_job_creation_fields(
            output_name=output_name,
            topic=topic,
            source_language=source_language,
            translation_engine=translation_engine,
            font_size=font_size,
            font_color=font_color,
            fallback_name=f"{Path(video.filename).stem}-vietsub",
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    final_name = fields["output_name"]
    safe_topic = fields["translation_topic"]

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
        translation_engine=fields["translation_engine"],
        subtitle_font_size=fields["subtitle_font_size"],
        subtitle_font_color=fields["subtitle_font_color"],
        source_language=fields["source_language"],
    )

    with jobs_lock:
        jobs[job_id] = job

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()

    return {"job_id": job_id, "output_name": final_name, "topic": safe_topic}


@app.post("/api/jobs/from-url")
async def create_job_from_url(body: Dict[str, Any] = Body(...)):
    url = str(body.get("url", "")).strip()
    selected_provider = str(body.get("selected_provider", "youtube")).strip().lower()
    try:
        safe_url, provider = validate_url_with_selected_provider(url, selected_provider)
        fields = _parse_job_creation_fields(
            output_name=str(body.get("output_name", "")),
            topic=str(body.get("topic", DEFAULT_TOPIC)),
            source_language=str(body.get("source_language", "en")),
            translation_engine=str(body.get("translation_engine", get_translation_engine())),
            font_size=body.get("font_size"),
            font_color=body.get("font_color"),
            fallback_name="url-video-vietsub",
        )
    except UrlImportError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = str(uuid.uuid4())
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _, source_srt_path, srt_path, layout_path = _job_paths(job_id)

    job = Job(
        id=job_id,
        status=JobStatus.DOWNLOADING,
        message="Đang kiểm tra link...",
        progress=0,
        output_name=fields["output_name"],
        source_srt_path=str(source_srt_path),
        srt_path=str(srt_path),
        layout_path=str(layout_path),
        translation_topic=fields["translation_topic"],
        translation_engine=fields["translation_engine"],
        subtitle_font_size=fields["subtitle_font_size"],
        subtitle_font_color=fields["subtitle_font_color"],
        source_language=fields["source_language"],
        url_provider=provider,
    )

    with jobs_lock:
        jobs[job_id] = job

    _write_job_meta(job)

    threading.Thread(target=_run_url_import_job, args=(job_id, safe_url), daemon=True).start()

    return {
        "job_id": job_id,
        "output_name": fields["output_name"],
        "topic": fields["translation_topic"],
        "source": "url",
        "provider": provider,
        "status": JobStatus.DOWNLOADING.value,
        "input_ready": False,
    }


@app.post("/api/jobs/{job_id}/process")
def process_downloaded_job(job_id: str):
    job_id = _validate_job_id(job_id)
    job = _get_job_or_rehydrate(job_id)
    if not job:
        raise HTTPException(404, JOB_RELOAD_MESSAGE)

    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, JOB_RELOAD_MESSAGE)
        if job.status in (JobStatus.PROCESSING, JobStatus.DOWNLOADING):
            raise HTTPException(409, "Job is already processing")
        if job.status != JobStatus.DOWNLOADED:
            raise HTTPException(400, "Job is not ready for subtitle processing")
        if not job.input_path or not os.path.isfile(job.input_path):
            raise HTTPException(404, JOB_RELOAD_MESSAGE)
        job.status = JobStatus.PROCESSING
        job.message = "Đang tạo phụ đề..."
        job.progress = 0
        job.error = None

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return {"ok": True, "job_id": job_id, "status": JobStatus.PROCESSING.value}


@app.get("/api/jobs/{job_id}/input-video")
def download_input_video(job_id: str):
    job_id = _validate_job_id(job_id)
    job = _get_job_or_rehydrate(job_id)
    if not job:
        raise HTTPException(404, JOB_RELOAD_MESSAGE)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, JOB_RELOAD_MESSAGE)
        path = job.input_path
        name = job.output_name or "video"

    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Source video not found")

    safe_name = f"{name}-source.mp4"
    return FileResponse(path, media_type="video/mp4", filename=safe_name)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job_id = _validate_job_id(job_id)
    job = _get_job_or_rehydrate(job_id)
    if not job:
        raise HTTPException(404, JOB_RELOAD_MESSAGE)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, JOB_RELOAD_MESSAGE)
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
