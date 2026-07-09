#!/usr/bin/env python3
"""Isolated voiceover CLI prototype (Phase 1): SRT cues -> Saydi TTS -> mixed MP4."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.voiceover.audio_builder import (  # noqa: E402
    build_manifest_summary,
    build_segment_manifests,
    build_voiceover_track,
    manifests_to_dicts,
    probe_audio_duration_ms,
    probe_video_duration_ms,
)
from auto_subtitle.voiceover.audio_mixer import (  # noqa: E402
    mix_audio_tracks,
    mux_video_with_audio,
    video_has_audio_stream,
)
from auto_subtitle.voiceover.saydi_tts import load_saydi_config, synthesize_to_file  # noqa: E402
from auto_subtitle.voiceover.srt_parser import parse_voiceover_srt  # noqa: E402
from auto_subtitle.voiceover.text_preparer import (  # noqa: E402
    prepare_voiceover_cues,
    prepared_to_voiceover_cues,
    summarize_prepared_cues,
    write_prepared_srt,
)
from auto_subtitle.voiceover.timing_planner import plan_timing  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _segment_filename(index: int) -> str:
    return f"{index:04d}.wav"


def run_prototype(
    *,
    input_video: Path,
    voiceover_srt: Path,
    output_video: Path,
    job_dir: Path,
    original_volume: float,
    voice_volume: float,
    min_gap_ms: int,
    max_borrow_after_ms: int,
    severe_overflow_ms: int,
    prepare_text: bool,
    voiceover_topic: str,
    max_chars_per_second: float,
    prepared_srt_output: Path | None,
    force: bool,
) -> dict:
    if output_video.exists() and not force:
        raise FileExistsError(f"Output exists: {output_video} (pass --force to overwrite)")

    effective_job_dir = job_dir or output_video.parent
    cues = parse_voiceover_srt(voiceover_srt)
    if not cues:
        raise ValueError(f"No cues found in {voiceover_srt}")

    prepared_cues = None
    tts_cues = cues
    prepared_srt_path = None
    text_summary = {
        "text_ok_count": len(cues),
        "text_shortened_count": 0,
        "text_too_long_count": 0,
        "total_original_chars": sum(len(cue.text) for cue in cues),
        "total_prepared_chars": sum(len(cue.text) for cue in cues),
        "average_reduction_ratio": 0.0,
    }
    if prepare_text:
        prepared_cues = prepare_voiceover_cues(
            cues,
            topic=voiceover_topic,
            max_chars_per_second=max_chars_per_second,
        )
        tts_cues = prepared_to_voiceover_cues(prepared_cues)
        text_summary = summarize_prepared_cues(prepared_cues)
        prepared_srt_path = prepared_srt_output or (effective_job_dir / "prepared_voiceover.srt")
        write_prepared_srt(prepared_cues, prepared_srt_path)

    video_duration_ms = probe_video_duration_ms(input_video)
    has_original_audio = video_has_audio_stream(input_video)

    segments_dir = effective_job_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    segment_paths: list[Path] = []
    tts_durations_ms: list[int] = []
    saydi_config = load_saydi_config()

    for cue in tts_cues:
        segment_path = segments_dir / _segment_filename(cue.index)
        synthesize_to_file(cue.text, segment_path, config=saydi_config)
        segment_paths.append(segment_path)
        tts_durations_ms.append(probe_audio_duration_ms(segment_path))

    timing_plans = plan_timing(
        tts_cues,
        tts_durations_ms,
        video_duration_ms=video_duration_ms,
        min_gap_ms=min_gap_ms,
        max_borrow_after_ms=max_borrow_after_ms,
        severe_overflow_ms=severe_overflow_ms,
    )
    manifests = build_segment_manifests(tts_cues, segment_paths, timing_plans, prepared_cues=prepared_cues)
    summary = build_manifest_summary(timing_plans)
    summary.update(text_summary)

    voiceover_track = effective_job_dir / "voiceover_track.wav"
    build_voiceover_track(
        manifests=manifests,
        video_duration_ms=video_duration_ms,
        output_path=voiceover_track,
    )

    mixed_audio = effective_job_dir / "mixed_audio.wav"
    mix_audio_tracks(
        original_audio_source=input_video,
        voiceover_track=voiceover_track,
        output_path=mixed_audio,
        original_volume=original_volume,
        voice_volume=voice_volume,
        has_original_audio=has_original_audio,
    )

    mux_video_with_audio(
        input_video=input_video,
        mixed_audio=mixed_audio,
        output_video=output_video,
    )

    manifest = {
        "created_at": _utc_now(),
        "input_video": str(input_video.resolve()),
        "voiceover_srt": str(voiceover_srt.resolve()),
        "output_video": str(output_video.resolve()),
        "video_duration_ms": video_duration_ms,
        "has_original_audio": has_original_audio,
        "original_volume": original_volume,
        "voice_volume": voice_volume,
        "text_preparation": {
            "enabled": prepare_text,
            "topic": voiceover_topic,
            "max_chars_per_second": max_chars_per_second,
            **({"prepared_srt_output": str(prepared_srt_path.resolve())} if prepared_srt_path else {}),
        },
        "timing_config": {
            "min_gap_ms": min_gap_ms,
            "max_borrow_after_ms": max_borrow_after_ms,
            "severe_overflow_ms": severe_overflow_ms,
        },
        "segments": manifests_to_dicts(manifests),
        "summary": summary,
        "artifacts": {
            "segments_dir": str(segments_dir.resolve()),
            "voiceover_track": str(voiceover_track.resolve()),
            "mixed_audio": str(mixed_audio.resolve()),
        },
        "overflow_warnings": summary["overflow_warning_count"] + summary["severe_overflow_count"],
    }

    manifest_path = effective_job_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--voiceover-srt", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--job-dir", default=None, type=Path)
    parser.add_argument("--original-volume", type=float, default=0.30)
    parser.add_argument("--voice-volume", type=float, default=1.00)
    parser.add_argument("--min-gap-ms", type=int, default=120)
    parser.add_argument("--max-borrow-after-ms", type=int, default=1200)
    parser.add_argument("--severe-overflow-ms", type=int, default=2000)
    parser.add_argument("--prepare-text", action="store_true")
    parser.add_argument("--voiceover-topic", default="catholic")
    parser.add_argument("--max-chars-per-second", type=float, default=13.0)
    parser.add_argument("--prepared-srt-output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        manifest = run_prototype(
            input_video=args.input_video,
            voiceover_srt=args.voiceover_srt,
            output_video=args.output_video,
            job_dir=args.job_dir,
            original_volume=args.original_volume,
            voice_volume=args.voice_volume,
            min_gap_ms=args.min_gap_ms,
            max_borrow_after_ms=args.max_borrow_after_ms,
            severe_overflow_ms=args.severe_overflow_ms,
            prepare_text=args.prepare_text,
            voiceover_topic=args.voiceover_topic,
            max_chars_per_second=args.max_chars_per_second,
            prepared_srt_output=args.prepared_srt_output,
            force=args.force,
        )
    except Exception as exc:
        logging.error("%s", exc)
        return 1

    print(f"output_video={manifest['output_video']}")
    print(f"manifest={manifest['manifest_path']}")
    print(f"segments={len(manifest['segments'])}")
    print(f"overflow_warnings={manifest['overflow_warnings']}")
    print(f"summary={json.dumps(manifest['summary'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
