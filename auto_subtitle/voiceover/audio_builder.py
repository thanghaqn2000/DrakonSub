from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .srt_parser import VoiceoverCue
from .text_preparer import PreparedVoiceoverCue
from .timing_planner import TimingPlan

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class SegmentManifest:
    index: int
    text: str
    original_text: str
    prepared_text: str
    original_char_count: int
    prepared_char_count: int
    target_char_count: int
    reduction_ratio: float
    text_status: str
    text_warnings: list[str]
    original_start_ms: int
    original_end_ms: int
    cue_duration_ms: int
    segment_path: str
    tts_duration_ms: int
    planned_start_ms: int
    planned_end_ms: int
    overflow_ms: int
    borrowed_gap_after_ms: int
    overlap_next_ms: int
    status: str
    shifted_ms: int = 0
    saydi_speed: float = 1.0

    @property
    def has_overflow(self) -> bool:
        return self.overflow_ms > 0


def probe_audio_duration_ms(path: Path) -> int:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"ffprobe failed for {path}: {stderr}")
    try:
        seconds = float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned invalid duration for {path}") from exc
    return max(0, int(round(seconds * 1_000)))


def probe_video_duration_ms(path: Path) -> int:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"ffprobe failed for video {path}: {stderr}")
    try:
        seconds = float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned invalid video duration for {path}") from exc
    return max(0, int(round(seconds * 1_000)))


def build_segment_manifests(
    cues: list[VoiceoverCue],
    segment_paths: list[Path],
    plans: list[TimingPlan],
    prepared_cues: list[PreparedVoiceoverCue] | None = None,
    saydi_speeds: list[float] | None = None,
) -> list[SegmentManifest]:
    if not (len(cues) == len(segment_paths) == len(plans)):
        raise ValueError("Cue, segment path, and timing plan lists must align")
    if prepared_cues is not None and len(prepared_cues) != len(cues):
        raise ValueError("Prepared cue list must align with cues")
    if saydi_speeds is not None and len(saydi_speeds) != len(cues):
        raise ValueError("Saydi speed list must align with cues")

    manifests: list[SegmentManifest] = []
    for idx, (cue, segment_path, plan) in enumerate(zip(cues, segment_paths, plans)):
        prepared = prepared_cues[idx] if prepared_cues is not None else None
        speed = saydi_speeds[idx] if saydi_speeds is not None else 1.0
        if plan.overflow_ms > 0 or plan.shifted_ms > 0:
            logger.warning(
                "Cue %s planned as %s; overflow=%sms borrowed=%sms overlap_next=%sms shifted=%sms speed=%.2f",
                cue.index,
                plan.status,
                plan.overflow_ms,
                plan.borrowed_gap_after_ms,
                plan.overlap_next_ms,
                plan.shifted_ms,
                speed,
            )
        manifests.append(
            SegmentManifest(
                index=cue.index,
                text=cue.text,
                original_text=prepared.original_text if prepared else cue.text,
                prepared_text=prepared.prepared_text if prepared else cue.text,
                original_char_count=prepared.original_char_count if prepared else len(cue.text),
                prepared_char_count=prepared.prepared_char_count if prepared else len(cue.text),
                target_char_count=prepared.target_char_count if prepared else max(20, len(cue.text)),
                reduction_ratio=prepared.reduction_ratio if prepared else 0.0,
                text_status=prepared.status if prepared else "ok",
                text_warnings=list(prepared.warnings) if prepared else [],
                original_start_ms=cue.start_ms,
                original_end_ms=cue.end_ms,
                cue_duration_ms=cue.duration_ms,
                segment_path=str(segment_path),
                tts_duration_ms=plan.tts_duration_ms,
                planned_start_ms=plan.planned_start_ms,
                planned_end_ms=plan.planned_end_ms,
                overflow_ms=plan.overflow_ms,
                borrowed_gap_after_ms=plan.borrowed_gap_after_ms,
                overlap_next_ms=plan.overlap_next_ms,
                status=plan.status,
                shifted_ms=plan.shifted_ms,
                saydi_speed=speed,
            )
        )
    return manifests


def manifests_to_dicts(manifests: list[SegmentManifest]) -> list[dict]:
    return [asdict(item) for item in manifests]


def build_manifest_summary(plans: list[TimingPlan]) -> dict:
    return {
        "cue_count": len(plans),
        "ok_count": sum(1 for item in plans if item.status == "ok"),
        "extended_count": sum(1 for item in plans if item.status == "extended_into_gap"),
        "shifted_count": sum(1 for item in plans if item.status == "shifted_to_avoid_overlap"),
        "overflow_warning_count": sum(1 for item in plans if item.status == "overflow_warning"),
        "severe_overflow_count": sum(1 for item in plans if item.status == "severe_overflow"),
        "max_overflow_ms": max((item.overflow_ms for item in plans), default=0),
        "max_shifted_ms": max((item.shifted_ms for item in plans), default=0),
        "total_borrowed_gap_ms": sum(item.borrowed_gap_after_ms for item in plans),
        "total_shifted_ms": sum(item.shifted_ms for item in plans),
        "overlap_next_count": sum(1 for item in plans if item.overlap_next_ms > 0),
    }


def build_voiceover_track(
    *,
    manifests: list[SegmentManifest],
    video_duration_ms: int,
    output_path: Path,
) -> None:
    if not manifests:
        raise ValueError("Cannot build voiceover track without segments")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_seconds = max(video_duration_ms, 1) / 1_000.0

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        f"anullsrc=channel_layout=mono:sample_rate={SAMPLE_RATE}",
    ]
    for manifest in manifests:
        cmd.extend(["-i", manifest.segment_path])

    delayed_labels: list[str] = []
    filter_parts: list[str] = []
    for idx, manifest in enumerate(manifests):
        label = f"a{idx}"
        delay = manifest.planned_start_ms
        filter_parts.append(
            f"[{idx + 1}:a]aresample={SAMPLE_RATE},aformat=channel_layouts=mono,"
            f"adelay={delay}|{delay}[{label}]"
        )
        delayed_labels.append(f"[{label}]")

    mix_inputs = "[0:a]" + "".join(delayed_labels)
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(manifests) + 1}:duration=first:"
        "dropout_transition=0:normalize=0[out]"
    )

    cmd.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[out]",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(output_path),
        ]
    )

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg voiceover track build failed: {stderr[-2000:]}")
