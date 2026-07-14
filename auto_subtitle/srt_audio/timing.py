from __future__ import annotations

import math
from typing import Any


def parse_srt_timestamp_to_ms(timestamp: str) -> int:
    hours, minutes, rest = timestamp.strip().split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis)
    )


def ms_to_srt_timestamp(value: int) -> str:
    value = max(0, int(value))
    hours = value // 3_600_000
    value %= 3_600_000
    minutes = value // 60_000
    value %= 60_000
    seconds = value // 1_000
    millis = value % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def plan_cascade_starts(
    intent_starts_ms: list[int],
    durations_ms: list[int],
    *,
    gap_ms: int = 280,
    max_gap_ms: int = 2000,
) -> list[int]:
    """Place cues with min rest (gap_ms) and max rest (max_gap_ms) after previous speech.

    - If intent is earlier than prev_end + gap_ms → push later (overflow cascade).
    - If intent is later than prev_end + max_gap_ms → pull earlier (cap long silence).
    - Otherwise keep intent.
    """
    if len(intent_starts_ms) != len(durations_ms):
        raise ValueError("intent_starts_ms and durations_ms length mismatch")
    if gap_ms < 0:
        raise ValueError("gap_ms must be >= 0")
    if max_gap_ms < gap_ms:
        raise ValueError("max_gap_ms must be >= gap_ms")
    planned: list[int] = []
    prev_end: int | None = None
    for intent, duration in zip(intent_starts_ms, durations_ms):
        intent_i = max(0, int(intent))
        dur_i = max(0, int(duration))
        if prev_end is None:
            start = intent_i
        else:
            earliest = prev_end + int(gap_ms)
            latest = prev_end + int(max_gap_ms)
            start = max(earliest, min(intent_i, latest))
        planned.append(start)
        prev_end = start + dur_i
    return planned


def estimate_speech_ms(
    text: str,
    *,
    chars_per_second: float = 13.0,
    saydi_speed: float = 1.0,
) -> int:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return 0
    cps = max(float(chars_per_second), 0.1)
    speed = max(float(saydi_speed), 0.01)
    return int(math.ceil((len(cleaned) / cps) * 1000.0 / speed))


def validate_cue_timings(
    cues: list[dict[str, Any]],
    *,
    chars_per_second: float = 13.0,
    saydi_speed: float = 1.0,
) -> list[list[str]]:
    """Return per-cue issue codes: empty_text, start_after_end, overlap_next, too_long."""
    result: list[list[str]] = [[] for _ in cues]
    starts: list[int] = []
    ends: list[int] = []
    for idx, cue in enumerate(cues):
        text = str(cue.get("text") or "").strip()
        if not text:
            result[idx].append("empty_text")
        try:
            start_ms = parse_srt_timestamp_to_ms(str(cue.get("start") or ""))
            end_ms = parse_srt_timestamp_to_ms(str(cue.get("end") or ""))
        except (ValueError, AttributeError):
            result[idx].append("invalid_timestamp")
            starts.append(0)
            ends.append(0)
            continue
        starts.append(start_ms)
        ends.append(end_ms)
        if start_ms >= end_ms:
            result[idx].append("start_after_end")
        duration = max(0, end_ms - start_ms)
        estimated = estimate_speech_ms(
            text, chars_per_second=chars_per_second, saydi_speed=saydi_speed
        )
        if duration > 0 and estimated > duration:
            result[idx].append("too_long")

    for idx in range(len(cues) - 1):
        if "invalid_timestamp" in result[idx] or "invalid_timestamp" in result[idx + 1]:
            continue
        if "start_after_end" in result[idx]:
            continue
        if ends[idx] > starts[idx + 1]:
            result[idx].append("overlap_next")
    return result
