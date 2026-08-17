from __future__ import annotations

from dataclasses import dataclass

from .srt_parser import VoiceoverCue
from .saydi_tts import SAYDI_SPEED_MAX, SAYDI_SPEED_MIN


@dataclass(frozen=True)
class TimingPlan:
    index: int
    text: str
    original_start_ms: int
    original_end_ms: int
    cue_duration_ms: int
    tts_duration_ms: int
    planned_start_ms: int
    planned_end_ms: int
    overflow_ms: int
    borrowed_gap_after_ms: int
    overlap_next_ms: int
    status: str
    shifted_ms: int = 0


def cue_audio_budget_ms(
    cue: VoiceoverCue,
    *,
    next_start_ms: int,
    video_duration_ms: int,
    is_last_cue: bool,
    min_gap_ms: int = 120,
    max_borrow_after_ms: int = 1_200,
) -> int:
    """Max audio length that fits the cue slot plus borrowable trailing gap."""
    if is_last_cue:
        available_gap_after = max(0, video_duration_ms - cue.end_ms)
        safe_gap_after = available_gap_after
    else:
        available_gap_after = max(0, next_start_ms - cue.end_ms)
        safe_gap_after = max(0, available_gap_after - min_gap_ms)
    borrowable_ms = min(safe_gap_after, max_borrow_after_ms)
    return max(1, cue.duration_ms + borrowable_ms)


def suggest_saydi_speed_for_budget(
    *,
    base_speed: float,
    tts_duration_ms: int,
    budget_ms: int,
    max_speed: float = SAYDI_SPEED_MAX,
    min_speed: float = SAYDI_SPEED_MIN,
    headroom: float = 1.05,
) -> float:
    """Pick a higher Saydi speed so estimated duration fits the budget."""
    base = max(min_speed, min(max_speed, float(base_speed)))
    if tts_duration_ms <= 0 or budget_ms <= 0 or tts_duration_ms <= budget_ms:
        return base
    needed = base * (tts_duration_ms / budget_ms) * headroom
    return round(min(max_speed, max(base, needed)), 2)


def plan_timing(
    cues: list[VoiceoverCue],
    tts_durations_ms: list[int],
    *,
    video_duration_ms: int,
    min_gap_ms: int = 120,
    max_borrow_after_ms: int = 1_200,
    severe_overflow_ms: int = 2_000,
    resolve_overlaps: bool = True,
) -> list[TimingPlan]:
    if len(cues) != len(tts_durations_ms):
        raise ValueError("Cue and TTS duration lists must align")

    plans: list[TimingPlan] = []
    earliest_free_ms = 0

    for idx, (cue, tts_duration_ms) in enumerate(zip(cues, tts_durations_ms)):
        is_last_cue = idx + 1 >= len(cues)
        next_start_ms = cues[idx + 1].start_ms if idx + 1 < len(cues) else video_duration_ms
        available_gap_after = max(0, next_start_ms - cue.end_ms)
        safe_gap_after = available_gap_after if is_last_cue else max(0, available_gap_after - min_gap_ms)

        overflow_ms = max(0, tts_duration_ms - cue.duration_ms)
        borrowable_ms = 0
        if overflow_ms > 0:
            borrowable_ms = min(safe_gap_after, max_borrow_after_ms, overflow_ms)

        planned_start_ms = cue.start_ms
        shifted_ms = 0
        if resolve_overlaps:
            planned_start_ms = max(cue.start_ms, earliest_free_ms)
            shifted_ms = max(0, planned_start_ms - cue.start_ms)

        # True audio end used by the mixer (adelay + full WAV).
        planned_end_ms = planned_start_ms + tts_duration_ms

        remaining_overflow_ms = max(0, overflow_ms - borrowable_ms)
        overlap_next_ms = 0
        if not resolve_overlaps and not is_last_cue:
            overlap_next_ms = max(0, (cue.start_ms + tts_duration_ms) - next_start_ms)

        if shifted_ms > 0:
            status = "shifted_to_avoid_overlap"
        elif overflow_ms == 0:
            status = "ok"
        elif remaining_overflow_ms == 0 and borrowable_ms > 0:
            status = "extended_into_gap"
        elif remaining_overflow_ms > severe_overflow_ms:
            status = "severe_overflow"
        else:
            status = "overflow_warning"

        plans.append(
            TimingPlan(
                index=cue.index,
                text=cue.text,
                original_start_ms=cue.start_ms,
                original_end_ms=cue.end_ms,
                cue_duration_ms=cue.duration_ms,
                tts_duration_ms=tts_duration_ms,
                planned_start_ms=planned_start_ms,
                planned_end_ms=planned_end_ms,
                overflow_ms=overflow_ms,
                borrowed_gap_after_ms=borrowable_ms,
                overlap_next_ms=overlap_next_ms,
                status=status,
                shifted_ms=shifted_ms,
            )
        )

        if resolve_overlaps:
            earliest_free_ms = planned_start_ms + tts_duration_ms + min_gap_ms

    return plans
