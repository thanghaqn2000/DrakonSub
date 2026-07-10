"""Voiceover script generation, cue editing, and render-after-review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..subtitle_edit_service import SubtitleCue, SubtitleEditError, load_srt, write_srt
from .from_video import prepare_voiceover_srt_from_video
from .job_service import VoiceoverJobError, VoiceoverJobOptions, VoiceoverJobResult, run_voiceover_job

DEFAULT_ORIGINAL_VOLUME = 0.18
ProgressCallback = Callable[[str, int], None]


@dataclass(frozen=True)
class ScriptRenderOptions:
    original_volume: float = DEFAULT_ORIGINAL_VOLUME
    voice_volume: float = 1.00
    prepare_text: bool = True
    voiceover_topic: str = "catholic"
    max_chars_per_second: float = 13.0
    min_gap_ms: int = 120
    max_borrow_after_ms: int = 1200
    severe_overflow_ms: int = 2000
    saydi_sample: str | None = None
    saydi_speed: float | None = None


def voiceover_srt_path(job_dir: Path) -> Path:
    return job_dir / "voiceover.srt"


def source_srt_path(job_dir: Path) -> Path:
    return job_dir / "source.srt"


def edited_voiceover_srt_path(job_dir: Path) -> Path:
    return job_dir / "edited_voiceover.srt"


def effective_voiceover_srt_path(job_dir: Path) -> Path:
    edited = edited_voiceover_srt_path(job_dir)
    if edited.is_file():
        return edited
    return voiceover_srt_path(job_dir)


def load_voiceover_cues(job_dir: Path) -> tuple[list[SubtitleCue], str]:
    base = voiceover_srt_path(job_dir)
    if not base.is_file():
        raise VoiceoverJobError("Chưa có file lời thuyết minh.")
    edited = edited_voiceover_srt_path(job_dir)
    if edited.is_file():
        return load_srt(edited), "edited_voiceover.srt"
    return load_srt(base), "voiceover.srt"


def load_source_cues(job_dir: Path) -> list[SubtitleCue]:
    path = source_srt_path(job_dir)
    if not path.is_file():
        return []
    return load_srt(path)


def validate_edited_cues(
    original_cues: list[SubtitleCue], submitted: list[dict]
) -> list[SubtitleCue]:
    if len(submitted) != len(original_cues):
        raise SubtitleEditError("Số lượng cue không khớp.")

    seen_indexes: set[int] = set()
    by_index = {cue.index: cue for cue in original_cues}
    updated: list[SubtitleCue] = []

    for item in submitted:
        if not isinstance(item, dict):
            raise SubtitleEditError("Dữ liệu cue không hợp lệ.")
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError) as exc:
            raise SubtitleEditError("Index cue không hợp lệ.") from exc
        if index in seen_indexes:
            raise SubtitleEditError("Index cue bị trùng.")
        seen_indexes.add(index)
        original = by_index.get(index)
        if original is None:
            raise SubtitleEditError("Index cue không khớp.")

        start = str(item.get("start", "")).strip()
        end = str(item.get("end", "")).strip()
        text = str(item.get("text", "")).strip()
        if start != original.start or end != original.end:
            raise SubtitleEditError("Không được thay đổi timing.")
        if not text:
            raise SubtitleEditError("Nội dung lời thuyết minh không được để trống.")
        updated.append(
            SubtitleCue(index=original.index, start=original.start, end=original.end, text=text)
        )

    if seen_indexes != set(by_index):
        raise SubtitleEditError("Số lượng cue không khớp.")

    updated.sort(key=lambda cue: cue.index)
    return updated


def save_edited_voiceover_cues(job_dir: Path, cues: list[SubtitleCue]) -> None:
    write_srt(cues, edited_voiceover_srt_path(job_dir))


def cues_to_response(
    job_id: str,
    cues: list[SubtitleCue],
    source: str,
    source_cues: list[SubtitleCue] | None = None,
) -> dict:
    source_by_index = {cue.index: cue.text for cue in (source_cues or [])}
    return {
        "job_id": job_id,
        "source": source,
        "cues": [
            {
                "index": cue.index,
                "start": cue.start,
                "end": cue.end,
                "text": cue.text,
                "source_text": source_by_index.get(cue.index, ""),
            }
            for cue in cues
        ],
    }


def run_script_generation_job(
    input_video: Path,
    job_dir: Path,
    voiceover_topic: str,
    on_progress: Optional[ProgressCallback] = None,
) -> tuple[Path, Path]:
    """Generate source.srt and voiceover.srt only; no TTS."""
    _, source_srt, voiceover_srt = prepare_voiceover_srt_from_video(
        input_video,
        job_dir,
        voiceover_topic,
        on_progress=on_progress,
    )
    return source_srt, voiceover_srt


def render_script_job(
    job_dir: Path,
    options: ScriptRenderOptions,
    on_progress: Optional[ProgressCallback] = None,
) -> VoiceoverJobResult:
    input_video = job_dir / "input.mp4"
    if not input_video.is_file():
        raise VoiceoverJobError("Input video not found.")
    voiceover_srt = effective_voiceover_srt_path(job_dir)
    if not voiceover_srt.is_file():
        raise VoiceoverJobError("Chưa có file lời thuyết minh.")

    output_video = job_dir / "output_voiceover.mp4"
    prepared_srt = job_dir / "prepared_voiceover.srt"
    job_options = VoiceoverJobOptions(
        input_video=input_video,
        voiceover_srt=voiceover_srt,
        output_video=output_video,
        workdir=job_dir,
        original_volume=options.original_volume,
        voice_volume=options.voice_volume,
        prepare_text=options.prepare_text,
        voiceover_topic=options.voiceover_topic,
        max_chars_per_second=options.max_chars_per_second,
        prepared_srt_output=prepared_srt if options.prepare_text else None,
        min_gap_ms=options.min_gap_ms,
        max_borrow_after_ms=options.max_borrow_after_ms,
        severe_overflow_ms=options.severe_overflow_ms,
        saydi_sample=options.saydi_sample,
        saydi_speed=options.saydi_speed,
        force=True,
    )
    return run_voiceover_job(job_options, progress_callback=on_progress)
