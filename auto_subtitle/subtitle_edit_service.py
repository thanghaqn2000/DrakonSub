from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_TIMING_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}$"
)


class SubtitleEditError(ValueError):
    pass


@dataclass
class SubtitleCue:
    index: int
    start: str
    end: str
    text: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_srt(path: Path) -> list[SubtitleCue]:
    if not path.exists():
        raise SubtitleEditError("Subtitle file not found")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    cues: list[SubtitleCue] = []
    blocks = re.split(r"\r?\n\r?\n", content)
    for block in blocks:
        lines = [line.rstrip("\r") for line in block.splitlines()]
        if len(lines) < 3:
            raise SubtitleEditError("Malformed SRT")
        try:
            index = int(lines[0].strip())
        except ValueError as exc:
            raise SubtitleEditError("Malformed SRT") from exc
        timing = lines[1].strip()
        if not _TIMING_RE.match(timing):
            raise SubtitleEditError("Malformed SRT")
        start, end = [part.strip() for part in timing.split("-->")]
        text = "\n".join(lines[2:]).strip()
        cues.append(SubtitleCue(index=index, start=start, end=end, text=text))

    for pos, cue in enumerate(cues, start=1):
        if cue.index != pos:
            raise SubtitleEditError("Malformed SRT")
    return cues


def write_srt(cues: list[SubtitleCue], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for expected_index, cue in enumerate(cues, start=1):
        parts.append(
            f"{expected_index}\n{cue.start} --> {cue.end}\n{cue.text.strip() or ' '}"
        )
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def merge_subtitle_views(
    source_cues: list[SubtitleCue] | None,
    original_vi_cues: list[SubtitleCue],
    current_vi_cues: list[SubtitleCue],
) -> list[dict]:
    if len(original_vi_cues) != len(current_vi_cues):
        raise SubtitleEditError("Cue count mismatch")
    if source_cues is not None and len(source_cues) != len(original_vi_cues):
        raise SubtitleEditError("Cue count mismatch")

    merged = []
    for idx, original in enumerate(original_vi_cues):
        current = current_vi_cues[idx]
        source = source_cues[idx] if source_cues is not None else None
        merged.append(
            {
                "cue_index": original.index,
                "start": original.start,
                "end": original.end,
                "source_text": source.text if source else "",
                "original_vi_text": original.text,
                "current_vi_text": current.text,
                "edited": current.text.strip() != original.text.strip(),
            }
        )
    return merged


def apply_text_edits(
    original_vi_cues: list[SubtitleCue],
    current_vi_cues: list[SubtitleCue],
    edits: list[dict],
) -> list[SubtitleCue]:
    if len(original_vi_cues) != len(current_vi_cues):
        raise SubtitleEditError("Cue count mismatch")

    updated = [
        SubtitleCue(index=cue.index, start=cue.start, end=cue.end, text=cue.text)
        for cue in current_vi_cues
    ]
    by_index = {cue.index: cue for cue in updated}

    for edit in edits:
        cue_index = edit.get("cue_index")
        text = str(edit.get("text", "")).strip()
        if not isinstance(cue_index, int) or cue_index < 1 or cue_index > len(updated):
            raise SubtitleEditError("Invalid cue index")
        if not text:
            raise SubtitleEditError("Subtitle text cannot be empty")
        by_index[cue_index].text = text

    return updated


def build_user_edits_payload(
    job_id: str,
    original_vi_cues: list[SubtitleCue],
    current_vi_cues: list[SubtitleCue],
) -> dict:
    edits = []
    now = _utc_now()
    for original, current in zip(original_vi_cues, current_vi_cues):
        if original.text.strip() == current.text.strip():
            continue
        edits.append(
            {
                "cue_index": current.index,
                "start": current.start,
                "end": current.end,
                "old_text": original.text,
                "new_text": current.text,
                "edited_at": now,
            }
        )
    return {"job_id": job_id, "updated_at": now, "edits": edits}


def write_user_edits(
    job_id: str,
    path: Path,
    original_vi_cues: list[SubtitleCue],
    current_vi_cues: list[SubtitleCue],
) -> dict:
    payload = build_user_edits_payload(job_id, original_vi_cues, current_vi_cues)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def get_effective_vi_srt(job_dir: Path) -> Path:
    vi_final = job_dir / "vi_final.srt"
    if not vi_final.exists():
        raise SubtitleEditError("Vietnamese subtitle not found")
    edited = job_dir / "edited_vi.srt"
    if not edited.exists():
        return vi_final

    try:
        original = load_srt(vi_final)
        current = load_srt(edited)
    except SubtitleEditError:
        return vi_final
    if len(original) != len(current):
        return vi_final
    return edited


def cue_indices_from_request(payload: dict) -> list[int]:
    raw = payload.get("cue_indices")
    if not isinstance(raw, list) or not raw:
        raise SubtitleEditError("Invalid cue index")
    indices: list[int] = []
    for item in raw:
        if not isinstance(item, int):
            raise SubtitleEditError("Invalid cue index")
        indices.append(item)
    return indices


def reset_cues_to_original(
    original_vi_cues: list[SubtitleCue],
    current_vi_cues: list[SubtitleCue],
    cue_indices: list[int],
) -> list[SubtitleCue]:
    if len(original_vi_cues) != len(current_vi_cues):
        raise SubtitleEditError("Cue count mismatch")
    reset_set = set(cue_indices)
    updated: list[SubtitleCue] = []
    for original, current in zip(original_vi_cues, current_vi_cues):
        if current.index in reset_set:
            updated.append(
                SubtitleCue(
                    index=original.index,
                    start=original.start,
                    end=original.end,
                    text=original.text,
                )
            )
        else:
            updated.append(
                SubtitleCue(
                    index=current.index,
                    start=current.start,
                    end=current.end,
                    text=current.text,
                )
            )
    return updated


def cues_to_dicts(cues: list[SubtitleCue]) -> list[dict]:
    return [asdict(cue) for cue in cues]
