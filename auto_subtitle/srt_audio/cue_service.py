from __future__ import annotations

from pathlib import Path

from auto_subtitle.subtitle_edit_service import SubtitleCue, load_srt, write_srt

from .timing import estimate_speech_ms, validate_cue_timings


class SrtAudioCueError(ValueError):
    pass


def input_srt_path(job_dir: Path) -> Path:
    return job_dir / "input.srt"


def edited_srt_path(job_dir: Path) -> Path:
    return job_dir / "edited.srt"


def load_effective_cues(job_dir: Path) -> list[SubtitleCue]:
    edited = edited_srt_path(job_dir)
    if edited.is_file():
        return load_srt(edited)
    return load_srt(input_srt_path(job_dir))


def annotate_cues(
    cues: list[SubtitleCue],
    *,
    chars_per_second: float = 13.0,
    saydi_speed: float = 1.0,
) -> list[dict]:
    payload = [
        {"index": c.index, "start": c.start, "end": c.end, "text": c.text}
        for c in cues
    ]
    issues = validate_cue_timings(
        payload, chars_per_second=chars_per_second, saydi_speed=saydi_speed
    )
    rows: list[dict] = []
    for cue, cue_issues in zip(cues, issues):
        estimated = estimate_speech_ms(
            cue.text, chars_per_second=chars_per_second, saydi_speed=saydi_speed
        )
        rows.append(
            {
                "index": cue.index,
                "start": cue.start,
                "end": cue.end,
                "text": cue.text,
                "estimated_ms": estimated,
                "issues": cue_issues,
            }
        )
    return rows


def save_edited_cues(job_dir: Path, submitted: list[dict]) -> list[SubtitleCue]:
    original = load_srt(input_srt_path(job_dir))
    if len(submitted) != len(original):
        raise SrtAudioCueError("Số lượng cue không khớp.")
    by_index = {c.index: c for c in original}
    updated: list[SubtitleCue] = []
    seen: set[int] = set()
    for item in submitted:
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError) as exc:
            raise SrtAudioCueError("Index cue không hợp lệ.") from exc
        if index in seen or index not in by_index:
            raise SrtAudioCueError("Index cue không khớp.")
        seen.add(index)
        start = str(item.get("start", "")).strip()
        end = str(item.get("end", "")).strip()
        text = str(item.get("text", "")).strip()
        if not text:
            raise SrtAudioCueError("Nội dung cue không được để trống.")
        updated.append(SubtitleCue(index=index, start=start, end=end, text=text))
    updated.sort(key=lambda c: c.index)
    if [c.index for c in updated] != [c.index for c in original]:
        raise SrtAudioCueError("Index cue không khớp.")
    write_srt(updated, edited_srt_path(job_dir))
    return updated


def has_blocking_issues(rows: list[dict]) -> bool:
    return any(row.get("issues") for row in rows)
