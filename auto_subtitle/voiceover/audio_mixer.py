from __future__ import annotations

import json
import subprocess
from pathlib import Path


def video_has_audio_stream(path: Path) -> bool:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return False
    streams = payload.get("streams") or []
    return bool(streams)


def mix_audio_tracks(
    *,
    original_audio_source: Path,
    voiceover_track: Path,
    output_path: Path,
    original_volume: float,
    voice_volume: float,
    has_original_audio: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if has_original_audio:
        filter_complex = (
            f"[0:a]volume={original_volume:.4f}[orig];"
            f"[1:a]volume={voice_volume:.4f}[voice];"
            "[orig][voice]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(original_audio_source),
            "-i",
            str(voiceover_track),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(voiceover_track),
            "-filter:a",
            f"volume={voice_volume:.4f}",
            str(output_path),
        ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg audio mix failed: {stderr[-2000:]}")


def mux_video_with_audio(
    *,
    input_video: Path,
    mixed_audio: Path,
    output_video: Path,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(mixed_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg mux failed: {stderr[-2000:]}")
