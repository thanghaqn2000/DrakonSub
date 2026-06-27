#!/usr/bin/env python3
"""Acceptance test for English domain ASR correction."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_subtitle.en_domain_corrector import correct_en_domain_srt_file  # noqa: E402
from auto_subtitle.utils import parse_srt  # noqa: E402

JOB_SOURCE = Path(
    "/var/folders/kc/wq9gs6yd0pl2b0q6ddqfqvfc0000gn/T/"
    "drakonsub_jobs/6bfeabd2-7d63-4d0c-8561-bbbc61df5891/source.srt"
)
OUT = ROOT / "artifacts" / "en_domain_correction_test"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    raw = OUT / "source_raw.srt"
    corrected = OUT / "source_corrected.srt"
    raw.write_bytes(JOB_SOURCE.read_bytes())

    correct_en_domain_srt_file(str(raw), str(corrected), debug_dir=str(OUT))

    entries = parse_srt(corrected.read_text(encoding="utf-8"))
    buffer_cue = next(
        (e for e in entries if "buffer" in e["text"].lower() or "buffett coin" in e["text"].lower()),
        None,
    )
    if not buffer_cue:
        print("FAIL: no cue with buffer/Buffett Coin found")
        sys.exit(1)

    text = buffer_cue["text"]
    print(f"Corrected cue: {text}")
    ok = "Buffett Coin" in text and "buffer coin" not in text.lower()
    if not ok:
        print("FAIL: expected Buffett Coin without buffer coin")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
