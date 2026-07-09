from __future__ import annotations

from dataclasses import dataclass

from .srt_parser import VoiceoverCue


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


def plan_timing(
    cues: list[VoiceoverCue],
    tts_durations_ms: list[int],
    *,
    video_duration_ms: int,
    min_gap_ms: int = 120,
    max_borrow_after_ms: int = 1_200,
    severe_overflow_ms: int = 2_000,
) -> list[TimingPlan]:
    if len(cues) != len(tts_durations_ms):
        raise ValueError("Cue and TTS duration lists must align")

    plans: list[TimingPlan] = []
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
        planned_end_ms = cue.start_ms + tts_duration_ms

        if not is_last_cue:
            max_safe_end_ms = max(cue.end_ms, next_start_ms - min_gap_ms)
            planned_end_ms = min(planned_end_ms, max_safe_end_ms)
        else:
            planned_end_ms = min(planned_end_ms, video_duration_ms)

        remaining_overflow_ms = max(0, overflow_ms - borrowable_ms)
        overlap_next_ms = 0
        if not is_last_cue:
            overlap_next_ms = max(0, (cue.start_ms + tts_duration_ms) - next_start_ms)

        if overflow_ms == 0:
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
            )
        )
    return plans
