#!/usr/bin/env python3
"""Acceptance test: final SRT has zero cue overlaps after timing normalize."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_subtitle.config import TIMING_NORMALIZE_MIN_GAP  # noqa: E402
from auto_subtitle.subtitle_timing_optimizer import (  # noqa: E402
    _count_overlaps,
    _parse_ts,
    normalize_final_srt_timing,
    optimize_srt_timing_file,
)
from auto_subtitle.utils import parse_srt  # noqa: E402

INPUT = ROOT / "artifacts" / "vi_editor_acceptance_test" / "vi_after_readability.srt"
OUT = ROOT / "artifacts" / "timing_normalize_test"


def main() -> None:
    if not INPUT.exists():
        print(f"SKIP: missing input {INPUT}")
        sys.exit(0)

    OUT.mkdir(parents=True, exist_ok=True)
    working = OUT / "vi_working.srt"
    shutil.copy2(INPUT, working)

    optimize_srt_timing_file(str(working))
    shutil.copy2(working, OUT / "vi_before_timing_normalize.srt")
    normalize_final_srt_timing(str(working))
    shutil.copy2(working, OUT / "vi_final.srt")

    entries = parse_srt(working.read_text(encoding="utf-8"))
    starts = [_parse_ts(e["start_str"]) for e in entries]
    ends = [_parse_ts(e["end_str"]) for e in entries]

    overlaps = _count_overlaps(starts, ends, TIMING_NORMALIZE_MIN_GAP)
    monotonic = all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1))
    valid_duration = all(ends[i] >= starts[i] for i in range(len(starts)))

    print(f"cue_count={len(entries)} overlaps={overlaps} monotonic_starts={monotonic}")
    if overlaps != 0 or not monotonic or not valid_duration:
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
