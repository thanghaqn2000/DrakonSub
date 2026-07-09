from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .srt_parser import VoiceoverCue

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class SegmentManifest:
    index: int
    text: str
    cue_start_ms: int
    cue_end_ms: int
    cue_duration_ms: int
    segment_path: str
    tts_duration_ms: int
    overflow_ms: int
    placement: str

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
    tts_durations_ms: list[int],
) -> list[SegmentManifest]:
    if not (len(cues) == len(segment_paths) == len(tts_durations_ms)):
        raise ValueError("Cue, segment path, and duration lists must align")

    manifests: list[SegmentManifest] = []
    for cue, segment_path, tts_duration_ms in zip(cues, segment_paths, tts_durations_ms):
        overflow_ms = max(0, tts_duration_ms - cue.duration_ms)
        if overflow_ms > 0:
            logger.warning(
                "Cue %s TTS (%sms) exceeds cue window (%sms) by %sms; allowing mild overflow",
                cue.index,
                tts_duration_ms,
                cue.duration_ms,
                overflow_ms,
            )
        manifests.append(
            SegmentManifest(
                index=cue.index,
                text=cue.text,
                cue_start_ms=cue.start_ms,
                cue_end_ms=cue.end_ms,
                cue_duration_ms=cue.duration_ms,
                segment_path=str(segment_path),
                tts_duration_ms=tts_duration_ms,
                overflow_ms=overflow_ms,
                placement="start_aligned",
            )
        )
    return manifests


def manifests_to_dicts(manifests: list[SegmentManifest]) -> list[dict]:
    return [asdict(item) for item in manifests]


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
        delay = manifest.cue_start_ms
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
