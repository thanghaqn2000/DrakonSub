"""Voiceover SRT quality: phrase grouping, compaction, CPS, continuous reindex."""

from __future__ import annotations

import re
from typing import Any

from ..utils import parse_srt
from .srt_parser import parse_timestamp_to_ms

_STRONG_END_RE = re.compile(r"[.!?…]\s*[\"'”’)]*\s*$")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Phrase grouping (source EN, before translate)
_DEFAULT_MAX_GAP_MS = 500
_DEFAULT_MAX_MERGED_CHARS = 180
_DEFAULT_MAX_MERGED_DURATION_MS = 8_000
_SHORT_CUE_WORDS = 4

# Compaction (VI, after translate)
_DEFAULT_TARGET_CPS = 18.0
_MIN_CUE_DURATION_MS = 1_200
_SHORT_TEXT_CHARS = 12
_COMPACT_MAX_GAP_MS = 500
_COMPACT_MAX_MERGED_CHARS = 160
_COMPACT_MAX_MERGED_DURATION_MS = 10_000
_MIN_GAP_KEEP_MS = 80


def _ms_to_srt_timestamp(value: int) -> str:
    value = max(0, int(value))
    hours = value // 3_600_000
    minutes = (value % 3_600_000) // 60_000
    seconds = (value % 60_000) // 1_000
    millis = value % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _entry_start_ms(entry: dict) -> int:
    return parse_timestamp_to_ms(str(entry["start_str"]))


def _entry_end_ms(entry: dict) -> int:
    return parse_timestamp_to_ms(str(entry["end_str"]))


def _entry_duration_ms(entry: dict) -> int:
    return max(0, _entry_end_ms(entry) - _entry_start_ms(entry))


def _normalize_text(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", (text or "").strip())


def _ends_strong(text: str) -> bool:
    return bool(_STRONG_END_RE.search(_normalize_text(text)))


def _word_count(text: str) -> int:
    return len(_normalize_text(text).split())


def _char_count(text: str) -> int:
    return len(_normalize_text(text).replace(" ", ""))


def _cps(entry: dict) -> float:
    duration_s = _entry_duration_ms(entry) / 1000.0
    if duration_s <= 0:
        return float("inf")
    return _char_count(entry.get("text", "")) / duration_s


def _gap_ms(prev: dict, nxt: dict) -> int:
    return max(0, _entry_start_ms(nxt) - _entry_end_ms(prev))


def _merge_pair(a: dict, b: dict) -> dict:
    text_a = _normalize_text(a.get("text", ""))
    text_b = _normalize_text(b.get("text", ""))
    if text_a and text_b:
        joined = f"{text_a} {text_b}"
    else:
        joined = text_a or text_b
    return {
        "start_str": a["start_str"],
        "end_str": b["end_str"],
        "text": _normalize_text(joined),
    }


def reindex_drop_empty(entries: list[dict]) -> list[dict]:
    """Drop blank/whitespace cues and return a clean list (indexes applied on write)."""
    out: list[dict] = []
    for entry in entries:
        text = _normalize_text(entry.get("text", ""))
        if not text:
            continue
        out.append(
            {
                "start_str": entry["start_str"],
                "end_str": entry["end_str"],
                "text": text,
            }
        )
    return out


def group_source_cues_for_voiceover(
    entries: list[dict],
    *,
    max_gap_ms: int = _DEFAULT_MAX_GAP_MS,
    max_merged_chars: int = _DEFAULT_MAX_MERGED_CHARS,
    max_merged_duration_ms: int = _DEFAULT_MAX_MERGED_DURATION_MS,
) -> list[dict]:
    """Merge fragmented Whisper cues into phrase-level blocks before translation."""
    cleaned = reindex_drop_empty(entries)
    if not cleaned:
        return []

    grouped: list[dict] = [dict(cleaned[0])]
    for current in cleaned[1:]:
        prev = grouped[-1]
        gap = _gap_ms(prev, current)
        merged_text = _normalize_text(f"{prev['text']} {current['text']}")
        merged_duration = _entry_end_ms(current) - _entry_start_ms(prev)
        short_current = _word_count(current["text"]) <= _SHORT_CUE_WORDS
        mid_phrase = not _ends_strong(prev["text"])
        should_merge = (
            mid_phrase
            and gap <= max_gap_ms
            and len(merged_text) <= max_merged_chars
            and merged_duration <= max_merged_duration_ms
            and (short_current or gap <= 350 or _word_count(prev["text"]) <= 8)
        )
        if should_merge:
            grouped[-1] = _merge_pair(prev, current)
        else:
            grouped.append(dict(current))
    return grouped


def _should_compact_pair(prev: dict, nxt: dict, *, target_cps: float) -> bool:
    gap = _gap_ms(prev, nxt)
    if gap > _COMPACT_MAX_GAP_MS:
        return False
    merged = _merge_pair(prev, nxt)
    if len(merged["text"]) > _COMPACT_MAX_MERGED_CHARS:
        return False
    if _entry_duration_ms(merged) > _COMPACT_MAX_MERGED_DURATION_MS:
        return False
    if _cps(merged) > max(target_cps + 4.0, 24.0):
        return False

    mid_phrase = not _ends_strong(prev["text"])
    short_next = (
        _entry_duration_ms(nxt) < 1_000
        or len(_normalize_text(nxt["text"])) <= _SHORT_TEXT_CHARS
        or _word_count(nxt["text"]) <= 3
    )
    short_prev = _entry_duration_ms(prev) < _MIN_CUE_DURATION_MS
    return mid_phrase and (short_next or short_prev)


def _borrow_timing_for_cps(entries: list[dict], *, target_cps: float) -> list[dict]:
    """Stretch cue ends into following gaps when CPS is too high."""
    if not entries:
        return []
    out = [dict(e) for e in entries]
    for i, entry in enumerate(out):
        text = _normalize_text(entry.get("text", ""))
        if not text:
            continue
        duration_ms = _entry_duration_ms(entry)
        needed_ms = int((_char_count(text) / target_cps) * 1000)
        if duration_ms >= needed_ms:
            continue
        start_ms = _entry_start_ms(entry)
        end_ms = _entry_end_ms(entry)
        if i + 1 < len(out):
            next_start = _entry_start_ms(out[i + 1])
            max_end = next_start - _MIN_GAP_KEEP_MS
        else:
            max_end = end_ms + 5_000
        new_end = min(max(end_ms, start_ms + needed_ms), max_end)
        if new_end > end_ms:
            out[i] = {
                "start_str": entry["start_str"],
                "end_str": _ms_to_srt_timestamp(new_end),
                "text": text,
            }
    return out


def compact_voiceover_cues(
    entries: list[dict],
    *,
    target_cps: float = _DEFAULT_TARGET_CPS,
) -> list[dict]:
    """Merge short/mid-phrase VI cues and borrow timing for TTS CPS."""
    working = reindex_drop_empty(entries)
    if not working:
        return []

    changed = True
    while changed and len(working) > 1:
        changed = False
        compacted: list[dict] = [dict(working[0])]
        i = 1
        while i < len(working):
            prev = compacted[-1]
            current = working[i]
            if _should_compact_pair(prev, current, target_cps=target_cps):
                compacted[-1] = _merge_pair(prev, current)
                changed = True
            else:
                compacted.append(dict(current))
            i += 1
        working = compacted

    return _borrow_timing_for_cps(working, target_cps=target_cps)


def optimize_voiceover_srt_entries(
    entries: list[dict],
    *,
    target_cps: float = _DEFAULT_TARGET_CPS,
) -> list[dict]:
    """Post-translate voiceover SRT: compact, CPS stretch, drop empty."""
    return compact_voiceover_cues(entries, target_cps=target_cps)


def compute_voiceover_srt_metrics(entries: list[dict]) -> dict[str, Any]:
    cleaned = reindex_drop_empty(entries)
    cue_count = len(cleaned)
    durations = [_entry_duration_ms(e) for e in cleaned]
    cps_values = [_cps(e) for e in cleaned]
    total_duration_ms = (
        (_entry_end_ms(cleaned[-1]) - _entry_start_ms(cleaned[0])) if cleaned else 0
    )
    return {
        "cue_count": cue_count,
        "max_index": cue_count,
        "missing_indexes": [],
        "total_duration_ms": total_duration_ms,
        "total_duration_s": round(total_duration_ms / 1000.0, 1),
        "cps_gt_20": sum(1 for c in cps_values if c > 20),
        "cps_gt_24": sum(1 for c in cps_values if c > 24),
        "cps_gt_26": sum(1 for c in cps_values if c > 26),
        "duration_lt_1s": sum(1 for d in durations if d < 1_000),
        "worst_cps": sorted(
            (
                {
                    "index": i,
                    "cps": round(cps_values[i - 1], 1),
                    "duration_ms": durations[i - 1],
                    "text": cleaned[i - 1]["text"][:80],
                }
                for i in range(1, cue_count + 1)
            ),
            key=lambda item: item["cps"],
            reverse=True,
        )[:5],
    }


def voiceover_narration_translation_context() -> dict[str, Any]:
    """Prompt context: TTS narration style, not on-screen subtitle fragments."""
    return {
        "video_context": {
            "detected_topic": "voiceover_narration",
            "audience_level": "general_adult",
            "translation_style": (
                "natural Vietnamese narration for TTS voiceover "
                "(spoken aloud, not on-screen subtitle lines)"
            ),
            "tone": "spoken narration",
            "short_summary": (
                "Rewrite as complete, spoken Vietnamese sentences suitable for "
                "text-to-speech voiceover. Prefer natural narration over literal "
                "cue-by-cue subtitle translation."
            ),
            "translation_warnings": [
                "This output will be read aloud by TTS — avoid fragmented tails.",
                "Prefer complete spoken sentences; allow light restructuring.",
                "Avoid machine-like or overly literal phrasing.",
                "Keep religious terms respectful and consistent when present "
                "(Chúa, Chúa Kitô, Kinh Thánh, Vương quốc của Chúa, Hội Thánh).",
                "Do not leave any non-empty source cue blank.",
            ],
        }
    }


def load_srt_entries_from_path(path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    return parse_srt(content)


def write_voiceover_srt_entries(entries: list[dict], path) -> None:
    """Write SRT with continuous indexes; never emit blank/space-only cues."""
    cleaned = reindex_drop_empty(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    for index, entry in enumerate(cleaned, start=1):
        text = _normalize_text(entry["text"]).replace("-->", "->")
        parts.append(
            f"{index}\n{entry['start_str']} --> {entry['end_str']}\n{text}"
        )
    path.write_text("\n\n".join(parts) + ("\n" if parts else ""), encoding="utf-8")
