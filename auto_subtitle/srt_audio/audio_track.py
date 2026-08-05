from __future__ import annotations

import subprocess
from pathlib import Path

SAMPLE_RATE = 24_000


def build_srt_audio_track(
    *,
    segment_starts_ms: list[int],
    segment_paths: list[Path],
    track_duration_ms: int,
    output_path: Path,
) -> None:
    if not segment_paths:
        raise ValueError("Cannot build audio track without segments")
    if len(segment_starts_ms) != len(segment_paths):
        raise ValueError("Segment starts and paths must align")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_seconds = max(track_duration_ms, 1) / 1_000.0

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
    for path in segment_paths:
        cmd.extend(["-i", str(path)])

    delayed_labels: list[str] = []
    filter_parts: list[str] = []
    for idx, start_ms in enumerate(segment_starts_ms):
        label = f"a{idx}"
        filter_parts.append(
            f"[{idx + 1}:a]aresample={SAMPLE_RATE},aformat=channel_layouts=mono,"
            f"adelay={start_ms}|{start_ms}[{label}]"
        )
        delayed_labels.append(f"[{label}]")

    mix_inputs = "[0:a]" + "".join(delayed_labels)
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(segment_paths) + 1}:duration=first:"
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
        raise RuntimeError(f"ffmpeg srt-audio track build failed: {stderr[-2000:]}")


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(mp3_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg mp3 convert failed: {stderr[-2000:]}")
