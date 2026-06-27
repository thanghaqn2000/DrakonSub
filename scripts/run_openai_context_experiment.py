#!/usr/bin/env python3
"""Controlled experiment: OpenAI context-aware raw translation + strengthened editor."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_subtitle.config import TIMING_NORMALIZE_MIN_GAP, load_env  # noqa: E402
from auto_subtitle.en_domain_corrector import correct_en_domain_srt_file  # noqa: E402
from auto_subtitle.pipeline import SubtitleConfig, translate_srt_file  # noqa: E402
from auto_subtitle.subtitle_readability_optimizer import optimize_readability_file  # noqa: E402
from auto_subtitle.subtitle_timing_optimizer import (  # noqa: E402
    _count_overlaps,
    _parse_ts,
    normalize_final_srt_timing,
    optimize_srt_timing_file,
)
from auto_subtitle.utils import parse_srt, write_srt_entries  # noqa: E402
from auto_subtitle.vi_compression import _cps  # noqa: E402
from auto_subtitle.vi_editor import edit_vi_srt_file  # noqa: E402
from auto_subtitle.vi_flow import _STANDALONE_FRAGMENT_RE, _is_tiny_fragment  # noqa: E402
from auto_subtitle.vi_flow import flow_vi_srt_file  # noqa: E402
from auto_subtitle.vi_compression import compress_vi_srt_file  # noqa: E402

JOB_SOURCE = Path(
    "/var/folders/kc/wq9gs6yd0pl2b0q6ddqfqvfc0000gn/T/"
    "drakonsub_jobs/6bfeabd2-7d63-4d0c-8561-bbbc61df5891/source.srt"
)
OUT = ROOT / "debug" / "openai_context_experiment"
BEFORE_RAW = (
    ROOT / "debug" / "provider_comparison" / "openai" / "openai_raw_translation.srt"
)

TRACK_CASES = {
    "B_cue7_8": [7, 8],
    "C_cue14_15": [14, 15],
    "D_cue24_25": [24, 25],
    "E_cue27_29": [27, 28, 29],
}

MAX_CPS = 18.0


def _cue_map(entries: list[dict]) -> dict[int, str]:
    return {i + 1: e.get("text", "").strip() for i, e in enumerate(entries)}


def _save_srt(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        write_srt_entries(entries, file=f)


def _empty_cue_nums(entries: list[dict]) -> list[int]:
    return [i + 1 for i, e in enumerate(entries) if not e.get("text", "").strip()]


def _metrics(entries: list[dict]) -> dict:
    cps_values: list[float] = []
    over_cps = 0
    fragments = 0
    for e in entries:
        text = e.get("text", "").strip()
        if not text:
            continue
        dur = _parse_ts(e["end_str"]) - _parse_ts(e["start_str"])
        if dur <= 0:
            continue
        c = _cps(text, dur)
        cps_values.append(c)
        if c > MAX_CPS:
            over_cps += 1
        if _is_tiny_fragment(text) or _STANDALONE_FRAGMENT_RE.match(text):
            fragments += 1
    starts = [_parse_ts(e["start_str"]) for e in entries]
    ends = [_parse_ts(e["end_str"]) for e in entries]
    return {
        "cue_count": len(entries),
        "empty_cue_count": len(_empty_cue_nums(entries)),
        "max_cps": round(max(cps_values), 2) if cps_values else 0.0,
        "over_cps_count": over_cps,
        "fragment_count": fragments,
        "overlap_count": _count_overlaps(starts, ends, TIMING_NORMALIZE_MIN_GAP),
    }


def _case_snapshot(cmap: dict[int, str], cue_nums: list[int]) -> dict[str, str]:
    return {str(n): cmap.get(n, "") for n in cue_nums}


def _editor_changed(before: dict[str, str], after: dict[str, str]) -> bool:
    return before != after


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    os.environ["TRANSLATION_ENGINE"] = "openai"
    os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
    os.environ["ENABLE_VI_READABILITY_OPTIMIZER"] = "true"
    os.environ["ENABLE_VI_TIMING_OPTIMIZER"] = "true"
    os.environ["OPENAI_CONTEXT_PROMPT_DUMP"] = str(
        OUT / "openai_context_prompt_dump.txt"
    )

    load_env()

    if not JOB_SOURCE.exists():
        print(f"Missing sample source: {JOB_SOURCE}", file=sys.stderr)
        sys.exit(1)

    if BEFORE_RAW.exists():
        shutil.copy2(BEFORE_RAW, OUT / "openai_raw_before.srt")
    else:
        print(f"Warning: baseline before raw not found at {BEFORE_RAW}")

    source = OUT / "source_corrected.srt"
    shutil.copy2(JOB_SOURCE, OUT / "source_raw.srt")
    correct_en_domain_srt_file(
        str(OUT / "source_raw.srt"), str(source), debug_dir=str(OUT)
    )

    working = OUT / "working.srt"
    config = SubtitleConfig.from_env()
    config.translation_engine = "openai"
    config.source_language = "en"

    translate_srt_file(str(source), str(working), config)
    raw_after = OUT / "openai_raw_after.srt"
    shutil.copy2(working, raw_after)

    edit_vi_srt_file(
        str(source),
        str(working),
        str(OUT / "openai_after_editor.srt"),
        translation_engine="openai",
        topic=config.translation_topic,
        debug_dir=str(OUT / "editor_debug"),
    )
    shutil.copy2(OUT / "openai_after_editor.srt", working)

    compress_vi_srt_file(str(working))
    flow_vi_srt_file(str(source), str(working))
    optimize_readability_file(str(working))
    optimize_srt_timing_file(str(working))
    normalize_final_srt_timing(str(working))
    final_path = OUT / "openai_final.srt"
    shutil.copy2(working, final_path)

    stages: dict[str, Path] = {
        "raw_before": OUT / "openai_raw_before.srt",
        "raw_after": raw_after,
        "after_editor": OUT / "openai_after_editor.srt",
        "final": final_path,
    }

    snapshots: dict[str, dict[int, str]] = {}
    stage_metrics: dict[str, dict] = {}
    for name, path in stages.items():
        if not path.exists():
            continue
        entries = parse_srt(path.read_text(encoding="utf-8"))
        snapshots[name] = _cue_map(entries)
        stage_metrics[name] = _metrics(entries)

    source_count = len(parse_srt(source.read_text(encoding="utf-8")))
    final_entries = parse_srt(final_path.read_text(encoding="utf-8"))
    final_metrics = _metrics(final_entries)

    known_cases: dict[str, dict] = {}
    for case_id, cue_nums in TRACK_CASES.items():
        before = {}
        after_editor = {}
        final = {}
        if "raw_before" in snapshots:
            before = _case_snapshot(snapshots["raw_before"], cue_nums)
        if "raw_after" in snapshots:
            after_raw = _case_snapshot(snapshots["raw_after"], cue_nums)
        else:
            after_raw = {}
        if "after_editor" in snapshots:
            after_editor = _case_snapshot(snapshots["after_editor"], cue_nums)
        if "final" in snapshots:
            final = _case_snapshot(snapshots["final"], cue_nums)

        editor_changed = _editor_changed(after_raw, after_editor) if after_raw else None
        known_cases[case_id] = {
            "cue_nums": cue_nums,
            "before_raw": before,
            "after_raw": after_raw,
            "after_editor": after_editor,
            "final": final,
            "editor_changed_from_raw": editor_changed,
        }

    report = {
        "experiment": "openai_context_experiment",
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "source_cue_count": source_count,
        "empty_cue_count": {
            stage: stage_metrics.get(stage, {}).get("empty_cue_count", None)
            for stage in stages
            if stage_metrics.get(stage)
        },
        "cue_count_match": {
            "source_vs_final": source_count == final_metrics["cue_count"],
            "raw_after_vs_source": stage_metrics.get("raw_after", {}).get("cue_count")
            == source_count,
            "final_vs_source": final_metrics["cue_count"] == source_count,
        },
        "known_cases": known_cases,
        "metrics_by_stage": stage_metrics,
        "acceptance": {
            "no_empty_final_cues": final_metrics["empty_cue_count"] == 0,
            "cue5_not_blank": bool(snapshots.get("final", {}).get(5, "").strip()),
            "case_B_good": (
                "mua lại" in snapshots.get("final", {}).get(7, "").lower()
                and "tạo ra" in snapshots.get("final", {}).get(8, "").lower()
            ),
            "case_C_no_standalone_ve": not any(
                re.search(r"(?i)^về (điều đó|nó)\.?$", snapshots.get("final", {}).get(n, ""))
                for n in (14, 15)
            ),
            "case_D_no_literal_cho": "tôi sẽ không cho bạn" not in " ".join(
                snapshots.get("final", {}).get(n, "") for n in (24, 25)
            ).lower(),
            "zero_overlap": final_metrics["overlap_count"] == 0,
        },
        "artifact_paths": {k: str(v) for k, v in stages.items()},
        "prompt_dump": str(OUT / "openai_context_prompt_dump.txt"),
    }

    report_path = OUT / "openai_context_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["acceptance"], ensure_ascii=False, indent=2))
    print(f"\nReport: {report_path}")

    if not all(report["acceptance"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
