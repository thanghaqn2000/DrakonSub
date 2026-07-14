from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auto_subtitle.subtitle_edit_service import SubtitleEditError, load_srt
from auto_subtitle.voiceover.audio_builder import probe_audio_duration_ms
from auto_subtitle.voiceover.saydi_tts import load_saydi_config, synthesize_to_file

from .audio_track import build_srt_audio_track, convert_wav_to_mp3
from .cue_service import annotate_cues, has_blocking_issues, load_effective_cues
from .timing import parse_srt_timestamp_to_ms


class SrtAudioJobError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_job_json(job_dir: Path, payload: dict[str, Any]) -> None:
    path = job_dir / "job.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_job_json(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "job.json"
    return json.loads(path.read_text(encoding="utf-8"))


def create_job_from_srt_bytes(jobs_root: Path, srt_bytes: bytes) -> tuple[str, Path]:
    jobs_root.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    input_srt = job_dir / "input.srt"
    text = srt_bytes.decode("utf-8-sig")
    input_srt.write_text(text, encoding="utf-8")

    try:
        cues = load_srt(input_srt)
    except SubtitleEditError as exc:
        raise SrtAudioJobError(str(exc)) from exc
    if not cues:
        raise SrtAudioJobError("File SRT không có cue.")

    payload = {
        "job_id": job_id,
        "job_type": "srt_audio",
        "status": "ready",
        "stage": "ready",
        "progress_percent": 10,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "error": None,
        "cue_count": len(cues),
        "output_wav": str(job_dir / "output.wav"),
        "output_mp3": str(job_dir / "output.mp3"),
        "manifest": str(job_dir / "manifest.json"),
        "output_format": None,
    }
    _write_job_json(job_dir, payload)
    return job_id, job_dir


def run_synthesize_job(
    job_dir: Path,
    *,
    saydi_sample: str | None,
    saydi_speed: float | None,
    output_format: str = "wav",
    chars_per_second: float = 13.0,
) -> dict[str, Any]:
    fmt = (output_format or "wav").strip().lower()
    if fmt not in {"wav", "mp3"}:
        raise SrtAudioJobError("output_format phải là wav hoặc mp3.")

    speed = float(saydi_speed) if saydi_speed is not None else 1.0
    cues = load_effective_cues(job_dir)
    rows = annotate_cues(cues, chars_per_second=chars_per_second, saydi_speed=speed)
    if has_blocking_issues(rows):
        raise SrtAudioJobError(
            "SRT còn lỗi timing (trống, chồng cue, timestamp sai). Hãy sửa trước khi thuyết minh."
        )

    saydi_config = load_saydi_config(
        sample_override=saydi_sample,
        speed_override=speed,
    )
    if not getattr(saydi_config, "token", ""):
        raise SrtAudioJobError("SAYDI_TTS_API_TOKEN is not configured")

    segments_dir = job_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    segment_paths: list[Path] = []
    segment_starts_ms: list[int] = []
    overflow_warnings: list[str] = []
    last_audio_end_ms = 0
    last_cue_end_ms = 0

    for cue in cues:
        start_ms = parse_srt_timestamp_to_ms(cue.start)
        end_ms = parse_srt_timestamp_to_ms(cue.end)
        last_cue_end_ms = max(last_cue_end_ms, end_ms)
        segment_path = segments_dir / f"{cue.index:04d}.wav"
        synthesize_to_file(cue.text, segment_path, config=saydi_config)
        tts_ms = probe_audio_duration_ms(segment_path)
        if tts_ms > max(0, end_ms - start_ms):
            overflow_warnings.append(f"cue_{cue.index}:tts_longer_than_slot")
        segment_paths.append(segment_path)
        segment_starts_ms.append(start_ms)
        last_audio_end_ms = max(last_audio_end_ms, start_ms + tts_ms)

    track_duration_ms = max(last_cue_end_ms, last_audio_end_ms, 1)
    output_wav = job_dir / "output.wav"
    build_srt_audio_track(
        segment_starts_ms=segment_starts_ms,
        segment_paths=segment_paths,
        track_duration_ms=track_duration_ms,
        output_path=output_wav,
    )

    output_mp3 = job_dir / "output.mp3"
    if fmt == "mp3":
        convert_wav_to_mp3(output_wav, output_mp3)

    manifest = {
        "job_type": "srt_audio",
        "created_at": _utc_now(),
        "cue_count": len(cues),
        "saydi_sample": saydi_config.sample,
        "saydi_speed": saydi_config.speed,
        "chars_per_second": chars_per_second,
        "output_format": fmt,
        "track_duration_ms": track_duration_ms,
        "overflow_warnings": overflow_warnings,
        "cues": rows,
    }
    manifest_path = job_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    payload = _read_job_json(job_dir)
    payload.update(
        {
            "status": "completed",
            "stage": "completed",
            "progress_percent": 100,
            "error": None,
            "updated_at": _utc_now(),
            "output_format": fmt,
            "saydi_sample": saydi_config.sample,
            "saydi_speed": saydi_config.speed,
            "summary": {
                "cue_count": len(cues),
                "track_duration_ms": track_duration_ms,
                "overflow_warning_count": len(overflow_warnings),
            },
        }
    )
    _write_job_json(job_dir, payload)

    return {
        "output_wav": str(output_wav),
        "output_mp3": str(output_mp3) if fmt == "mp3" else None,
        "manifest": str(manifest_path),
        "summary": payload["summary"],
    }
