#!/usr/bin/env python3
"""Acceptance test: improved VI editor pass on Buffett/Bitcoin sample."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_subtitle.config import TIMING_NORMALIZE_MIN_GAP  # noqa: E402
from auto_subtitle.en_domain_corrector import correct_en_domain_srt_file  # noqa: E402
from auto_subtitle.pipeline import SubtitleConfig, translate_srt_file  # noqa: E402
from auto_subtitle.subtitle_readability_optimizer import optimize_readability_file  # noqa: E402
from auto_subtitle.subtitle_timing_optimizer import (  # noqa: E402
    _count_overlaps,
    _parse_ts,
    normalize_final_srt_timing,
    optimize_srt_timing_file,
)
from auto_subtitle.utils import parse_srt  # noqa: E402
from auto_subtitle.vi_editor import edit_vi_srt_file  # noqa: E402
from auto_subtitle.vi_compression import compress_vi_srt_file  # noqa: E402
from auto_subtitle.vi_flow import flow_vi_srt_file  # noqa: E402
import shutil  # noqa: E402

JOB_SOURCE = Path(
    "/var/folders/kc/wq9gs6yd0pl2b0q6ddqfqvfc0000gn/T/"
    "drakonsub_jobs/6bfeabd2-7d63-4d0c-8561-bbbc61df5891/source.srt"
)
OUT = ROOT / "artifacts" / "vi_editor_acceptance_test"

TARGET_CUES = {7, 8, 14, 15, 16, 20, 21, 22, 23, 24, 25, 27, 28, 29}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    os.environ["TRANSLATION_ENGINE"] = "openai"
    os.environ["ENABLE_VI_READABILITY_OPTIMIZER"] = "true"
    os.environ["ENABLE_VI_TIMING_OPTIMIZER"] = "true"

    source_raw = OUT / "source_raw.srt"
    source_corrected = OUT / "source_corrected.srt"
    vi_raw = OUT / "vi_raw.srt"
    vi_editor = OUT / "vi_editor.srt"
    vi_compression = OUT / "vi_after_compression.srt"
    vi_flow = OUT / "vi_after_flow.srt"
    vi_final = OUT / "vi_final.srt"
    working = OUT / "vi_working.srt"

    shutil.copy2(JOB_SOURCE, source_raw)
    correct_en_domain_srt_file(str(source_raw), str(source_corrected), debug_dir=str(OUT))

    config = SubtitleConfig.from_env()
    config.translation_engine = "openai"
    config.source_language = "en"

    translate_srt_file(str(source_corrected), str(vi_raw), config)
    shutil.copy2(vi_raw, working)

    edit_vi_srt_file(
        str(source_corrected),
        str(working),
        str(vi_editor),
        translation_engine=config.translation_engine,
        topic=config.translation_topic,
        debug_dir=str(OUT),
    )
    shutil.copy2(vi_editor, working)

    compress_vi_srt_file(str(working))
    shutil.copy2(working, vi_compression)

    flow_vi_srt_file(str(source_corrected), str(working))
    shutil.copy2(working, vi_flow)

    optimize_readability_file(str(working))
    optimize_srt_timing_file(str(working))
    normalize_final_srt_timing(str(working))
    shutil.copy2(working, vi_final)

    entries = parse_srt(vi_final.read_text(encoding="utf-8"))
    source_entries = parse_srt(source_corrected.read_text(encoding="utf-8"))
    starts = [_parse_ts(e["start_str"]) for e in entries]
    ends = [_parse_ts(e["end_str"]) for e in entries]
    overlaps = _count_overlaps(starts, ends, TIMING_NORMALIZE_MIN_GAP)

    counts = {
        "source_corrected": len(source_entries),
        "vi_raw": len(parse_srt(vi_raw.read_text(encoding="utf-8"))),
        "vi_editor": len(parse_srt(vi_editor.read_text(encoding="utf-8"))),
        "vi_compression": len(parse_srt(vi_compression.read_text(encoding="utf-8"))),
        "vi_flow": len(parse_srt(vi_flow.read_text(encoding="utf-8"))),
        "vi_final": len(entries),
    }
    print("Cue counts:", json.dumps(counts, indent=2))
    print(f"overlaps={overlaps}")

    print("\n--- Target cues (vi_final.srt) ---")
    for i in sorted(TARGET_CUES):
        if i <= len(entries):
            print(f"[{i}] {entries[i - 1]['text'].strip()}")

    final_text = vi_final.read_text(encoding="utf-8")
    checks = {
        "cue_counts_match": len(set(counts.values())) == 1,
        "zero_overlap": overlaps == 0,
        "buffett_coin_fixed": "Buffett Coin" in source_corrected.read_text(),
        "no_ve_no_fragment": "về nó." not in final_text.lower()
        and "về điều đó." not in final_text.lower(),
        "cue7_clear": "mua lại" in entries[6]["text"].lower() if len(entries) >= 7 else False,
        "cue14_15_flow": (
            "giữ hết" in entries[13]["text"].lower()
            and "bí ẩn" in entries[14]["text"].lower()
            if len(entries) >= 15
            else False
        ),
        "cue24_25_flow": (
            "bỏ tiền mua" in entries[23]["text"].lower()
            if len(entries) >= 25
            else False
        ),
    }
    print("\nChecks:", json.dumps(checks, ensure_ascii=False, indent=2))

    if not all(checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
