#!/usr/bin/env python3
"""Isolation test: readability optimizer ON vs OFF (same translation base)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_subtitle.pipeline import (  # noqa: E402
    SubtitleConfig,
    extract_audio,
    transcribe_to_srt,
    translate_srt_file,
)
from auto_subtitle.subtitle_readability_optimizer import optimize_readability_file
from auto_subtitle.subtitle_timing_optimizer import optimize_srt_timing_file
from auto_subtitle.utils import parse_srt

VIDEO = Path(
    "/var/folders/kc/wq9gs6yd0pl2b0q6ddqfqvfc0000gn/T/"
    "drakonsub_jobs/6bfeabd2-7d63-4d0c-8561-bbbc61df5891/input.mp4"
)
JOB_SOURCE = VIDEO.parent / "source.srt"
OUT = ROOT / "artifacts" / "readability_isolation_test"


def _compare_srt(before_path: Path, after_path: Path) -> dict:
    before = parse_srt(before_path.read_text(encoding="utf-8"))
    after = parse_srt(after_path.read_text(encoding="utf-8"))
    changes = []
    for i, (b, a) in enumerate(zip(before, after), start=1):
        bt, at = b["text"].strip(), a["text"].strip()
        if bt != at:
            changes.append(
                {
                    "cue": i,
                    "before": bt,
                    "after": at,
                    "before_len": len(bt),
                    "after_len": len(at),
                    "delta_len": len(at) - len(bt),
                }
            )
    return {
        "total_cues": len(before),
        "changed_cues": len(changes),
        "changes": changes,
    }


def _run_branch(
    *,
    name: str,
    source_srt: Path,
    vi_base: Path,
    readability_enabled: bool,
) -> None:
    branch = OUT / name
    branch.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_srt, branch / "source.srt")
    shutil.copy2(vi_base, branch / "vi_before_readability.srt")

    working = branch / "vi_working.srt"
    shutil.copy2(vi_base, working)

    os.environ["ENABLE_VI_READABILITY_OPTIMIZER"] = (
        "true" if readability_enabled else "false"
    )
    os.environ["DRAKONSUB_VI_BEFORE_READABILITY_SRT"] = str(
        branch / "vi_before_readability.srt"
    )
    os.environ["DRAKONSUB_VI_AFTER_READABILITY_SRT"] = str(
        branch / "vi_after_readability.srt"
    )

    optimize_readability_file(str(working))
    optimize_srt_timing_file(str(working))
    shutil.copy2(working, branch / "vi_final.srt")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    os.environ["TRANSLATION_ENGINE"] = "openai"
    os.environ["TRANSLATION_TOPIC"] = "economics"
    os.environ["ENABLE_VI_TIMING_OPTIMIZER"] = "true"
    os.environ["DRAKONSUB_DEBUG"] = "false"

    config = SubtitleConfig.from_env()
    config.translation_engine = "openai"
    config.source_language = "en"

    shared = OUT / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    source_srt = shared / "source.srt"
    vi_base = shared / "vi_before_readability.srt"

    if JOB_SOURCE.exists():
        shutil.copy2(JOB_SOURCE, source_srt)
        print(f"Reused existing source.srt from job: {JOB_SOURCE}")
    else:
        audio = shared / "audio.wav"
        extract_audio(str(VIDEO), str(audio))
        transcribe_to_srt(str(audio), str(source_srt), config)
        print(f"Transcribed fresh source.srt from: {VIDEO}")

    translate_srt_file(str(source_srt), str(vi_base), config)
    print(f"Translated once with OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})")

    _run_branch(
        name="run_a_readability_on",
        source_srt=source_srt,
        vi_base=vi_base,
        readability_enabled=True,
    )
    _run_branch(
        name="run_b_readability_off",
        source_srt=source_srt,
        vi_base=vi_base,
        readability_enabled=False,
    )

    readability_diff = _compare_srt(
        OUT / "run_a_readability_on" / "vi_before_readability.srt",
        OUT / "run_a_readability_on" / "vi_after_readability.srt",
    )
    final_diff = _compare_srt(
        OUT / "run_b_readability_off" / "vi_final.srt",
        OUT / "run_a_readability_on" / "vi_final.srt",
    )

    report = {
        "video": str(VIDEO),
        "translation_engine": "openai",
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "run_a": {
            "enable_readability": True,
            "readability_changes": readability_diff,
        },
        "run_b": {
            "enable_readability": False,
            "readability_changes": _compare_srt(
                OUT / "run_b_readability_off" / "vi_before_readability.srt",
                OUT / "run_b_readability_off" / "vi_after_readability.srt",
            ),
        },
        "final_text_diff_a_vs_b": final_diff,
    }

    report_path = OUT / "comparison_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nArtifacts saved under: {OUT}")


if __name__ == "__main__":
    main()
