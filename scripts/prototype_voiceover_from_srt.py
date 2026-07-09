#!/usr/bin/env python3
"""Thin CLI wrapper for the internal voiceover job service."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.voiceover.job_service import (  # noqa: E402
    VoiceoverJobOptions,
    run_voiceover_job,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--voiceover-srt", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--job-dir", default=None, type=Path)
    parser.add_argument("--original-volume", type=float, default=0.18)
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
        result = run_voiceover_job(
            VoiceoverJobOptions(
            input_video=args.input_video,
            voiceover_srt=args.voiceover_srt,
            output_video=args.output_video,
            workdir=args.job_dir or args.output_video.parent,
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
        )
    except Exception as exc:
        logging.error("%s", exc)
        return 1

    print(f"output_video={result.output_video}")
    print(f"manifest={result.manifest_path}")
    print(f"segments={result.segment_count}")
    overflow_warnings = result.summary.get("overflow_warning_count", 0) + result.summary.get("severe_overflow_count", 0)
    print(f"overflow_warnings={overflow_warnings}")
    print(f"summary={json.dumps(result.summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
