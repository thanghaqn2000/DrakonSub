#!/usr/bin/env python3
"""Build long-form success_232 diagnosis reports from benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_subtitle.subtitle_timing_optimizer import _parse_ts
from auto_subtitle.utils import parse_srt
from auto_subtitle.vi_compression import _cps

OUT = ROOT / "artifacts" / "translation_quality_review"
SAMPLE_ID = "success_232"
WINDOW_SIZE = 20
WINDOW_STRIDE = 20


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_srt_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    return {i + 1: (e.get("text") or "").strip() for i, e in enumerate(parse_srt(path.read_text()))}


def _duration_seconds(entries: List[dict]) -> float:
    if not entries:
        return 0.0
    return _parse_ts(entries[-1]["end_str"]) - _parse_ts(entries[0]["start_str"])


def _dominant_layer(errors: List[str]) -> str:
    if not errors:
        return "none"
    rawish = {
        "semantic_alignment_error",
        "semantic_drift_error",
        "literal_translation_error",
    }
    flowish = {"repeated_meaning_error", "cue_flow_error"}
    readish = {"readability_cps_error"}
    shiftish = {"cue_alignment_warning", "split_term_across_cues_error"}
    families = set(errors)
    if families & rawish:
        return "raw_translation"
    if families & flowish:
        return "editor_flow"
    if families & readish:
        return "timing_readability"
    if families & shiftish:
        return "cue_shift"
    return "qa_other"


def _window_analysis(cue_assessments: List[dict], entries: List[dict]) -> List[dict]:
    windows: List[dict] = []
    n = len(cue_assessments)
    for start in range(0, n, WINDOW_STRIDE):
        end = min(start + WINDOW_SIZE, n)
        if start >= n:
            break
        chunk = cue_assessments[start:end]
        risky = sum(1 for c in chunk if c.get("is_risky"))
        sem = sum(
            1
            for c in chunk
            if "semantic_alignment_error" in (c.get("detected_translation_errors") or [])
        )
        rep = sum(
            1
            for c in chunk
            if "repeated_meaning_error" in (c.get("detected_translation_errors") or [])
        )
        cps_err = sum(
            1
            for c in chunk
            if "readability_cps_error" in (c.get("detected_translation_errors") or [])
        )
        shift = sum(
            1
            for c in chunk
            if (c.get("alignment_warnings") or []) or "cue_alignment_warning" in (
                c.get("detected_translation_errors") or []
            )
        )
        all_errs: List[str] = []
        for c in chunk:
            all_errs.extend(c.get("detected_translation_errors") or [])
        layer = _dominant_layer(all_errs)
        scores = [0 if c.get("is_risky") else 1 for c in chunk]
        win_score = round(100 * sum(scores) / max(len(scores), 1))
        windows.append(
            {
                "start_cue": start + 1,
                "end_cue": end,
                "quality_score": win_score,
                "risky_cues": risky,
                "semantic_errors": sem,
                "repeated_meaning_errors": rep,
                "cps_errors": cps_err,
                "cue_shift_suspicions": shift,
                "dominant_failure_layer": layer,
                "notes": "",
            }
        )
    return windows


def _cue_rows(artifact_dir: Path, quality: dict) -> List[dict]:
    source = _load_srt_map(artifact_dir / "source_corrected.srt") or _load_srt_map(
        artifact_dir / "source.srt"
    )
    vi_raw = _load_srt_map(artifact_dir / "vi_raw.srt")
    vi_editor = _load_srt_map(artifact_dir / "vi_after_editor.srt")
    vi_flow = _load_srt_map(artifact_dir / "vi_after_flow.srt")
    vi_final = _load_srt_map(artifact_dir / "final_vi.srt")
    entries = []
    if (artifact_dir / "final_vi.srt").exists():
        entries = parse_srt((artifact_dir / "final_vi.srt").read_text())
    entry_by_idx = {i + 1: e for i, e in enumerate(entries)}

    rows: List[dict] = []
    for c in quality.get("cue_assessments") or []:
        idx = int(c["cue_index"])
        entry = entry_by_idx.get(idx, {})
        dur = 0.01
        if entry:
            dur = max(0.01, _parse_ts(entry["end_str"]) - _parse_ts(entry["start_str"]))
        vi = vi_final.get(idx, "")
        errs = c.get("detected_translation_errors") or []
        rows.append(
            {
                "cue_index": idx,
                "time": f"{entry.get('start_str','')} --> {entry.get('end_str','')}",
                "source_en": c.get("en") or source.get(idx, ""),
                "vi_raw": vi_raw.get(idx, ""),
                "vi_after_editor": vi_editor.get(idx, ""),
                "vi_after_flow": vi_flow.get(idx, ""),
                "vi_final": vi,
                "auto_error_families": ";".join(errs),
                "cps": round(_cps(vi, dur), 1) if vi else 0,
                "chars_per_line": len(vi),
                "likely_failure_layer": _dominant_layer(errs),
                "is_risky": c.get("is_risky", False),
                "risk_score": len(errs) * 3 + (5 if c.get("is_risky") else 0),
                "SA_semantic_score": "",
                "SA_comment": "",
            }
        )
    return rows


def _worst_slice(rows: List[dict]) -> List[dict]:
    semantic = sorted(
        [r for r in rows if "semantic" in r["auto_error_families"]],
        key=lambda r: r["risk_score"],
        reverse=True,
    )[:15]
    cps = sorted(
        [r for r in rows if "readability_cps" in r["auto_error_families"]],
        key=lambda r: r["cps"],
        reverse=True,
    )[:10]
    repeat = sorted(
        [
            r
            for r in rows
            if "repeated_meaning" in r["auto_error_families"]
            or "cue_flow" in r["auto_error_families"]
        ],
        key=lambda r: r["risk_score"],
        reverse=True,
    )[:10]
    good = [r for r in rows if not r["is_risky"]][:5]
    picked: List[dict] = []
    seen = set()

    def add(items: List[dict], reason: str) -> None:
        for r in items:
            if r["cue_index"] in seen:
                continue
            seen.add(r["cue_index"])
            row = dict(r)
            row["selection_reason"] = reason
            picked.append(row)

    add(semantic, "worst_semantic")
    add(cps, "worst_cps")
    add(repeat, "repeat_or_flow")
    add(good, "control_good")
    return picked[:40]


def build_report(artifact_dir: Path, benchmark_report: Optional[dict], mode: str) -> dict:
    quality = _load_json(artifact_dir / "translation_quality_report.json") or {}
    contract = _load_json(artifact_dir / "pipeline_contract_report.json") or {}
    overlap = _load_json(OUT / "post_raw_overlap_guard_v1_report.json") or {}
    fragment = _load_json(OUT / "fragment_overlap_repair_v1_report.json") or {}
    llm_cache = _load_json(OUT / "raw_llm_response_cache_report.json") or {}
    shift = _load_json(artifact_dir / "cue_shift_repair_report.json") or {}
    intel = _load_json(artifact_dir / "accepted_repairs.json") or []

    sample_row = None
    if benchmark_report:
        for s in benchmark_report.get("samples", []):
            if s.get("id") == SAMPLE_ID:
                sample_row = s
                break

    entries = []
    if (artifact_dir / "final_vi.srt").exists():
        entries = parse_srt((artifact_dir / "final_vi.srt").read_text())

    overlap_sample = {}
    for s in overlap.get("samples", []):
        if s.get("sample_id") == SAMPLE_ID:
            overlap_sample = s
            break

    err_counts = Counter()
    for c in quality.get("cue_assessments") or []:
        for e in c.get("detected_translation_errors") or []:
            err_counts[e] += 1

    repeated_count = err_counts.get("repeated_meaning_error", 0)
    cps_count = err_counts.get("readability_cps_error", 0)
    sem_count = quality.get("semantic_alignment", {}).get("summary", {}).get(
        "semantic_alignment_errors", 0
    )

    cue_rows = _cue_rows(artifact_dir, quality)
    windows = _window_analysis(quality.get("cue_assessments") or [], entries)

    context_drift_cues = [
        c["cue_index"]
        for c in (quality.get("cue_assessments") or [])
        if "semantic_drift_error" in (c.get("detected_translation_errors") or [])
    ][:10]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": SAMPLE_ID,
        "benchmark_mode": mode,
        "cue_count": contract.get("source_cue_count") or len(entries),
        "total_duration_seconds": round(_duration_seconds(entries), 2),
        "runtime_seconds": sample_row.get("elapsed_seconds") if sample_row else None,
        "raw_llm_calls": llm_cache.get("total_calls"),
        "raw_llm_cache_hits": llm_cache.get("cache_hits"),
        "raw_llm_cache_misses": llm_cache.get("cache_misses"),
        "estimated_cost_if_available": llm_cache.get("estimated_cost_usd"),
        "contract_status": contract.get("pipeline_contract_status"),
        "text_lock_status": contract.get("post_final_repair_text_lock_status"),
        "quality_score": quality.get("quality_score"),
        "risky_cue_count": quality.get("summary", {}).get("risky_cue_count"),
        "semantic_error_count": sem_count,
        "repeated_meaning_count": repeated_count,
        "readability_cps_count": cps_count,
        "cue_shift_count": shift.get("windows_detected", 0) if shift else 0,
        "empty_or_missing_cue_count": len(contract.get("missing_or_empty_cue_errors") or []),
        "post_raw_overlap_repairs": overlap_sample.get("summary", {}),
        "span_guard_repairs": {},
        "editor_flow_repair_count": len(intel) if isinstance(intel, list) else 0,
        "segment_windows": windows,
        "long_form_checks": {
            "context_drift_detected": bool(context_drift_cues),
            "context_drift_example_cues": context_drift_cues,
            "repeated_meaning_explosion": repeated_count >= 15,
            "repeated_meaning_count": repeated_count,
            "repeated_meaning_example_cues": [
                c["cue_index"]
                for c in (quality.get("cue_assessments") or [])
                if "repeated_meaning_error" in (c.get("detected_translation_errors") or [])
            ][:10],
            "readability_collapse": cps_count >= 20,
            "readability_cps_error_count": cps_count,
            "readability_top_windows": sorted(
                windows, key=lambda w: w["cps_errors"], reverse=True
            )[:3],
        },
        "top_failure_families": err_counts.most_common(5),
        "worst_cues": sorted(cue_rows, key=lambda r: r["risk_score"], reverse=True)[:10],
    }
    return report, cue_rows


def _write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _md(report: dict) -> str:
    lines = [
        "# Long-form success_232 Diagnosis v1",
        "",
        f"Generated: {report['generated_at']}",
        f"Mode: {report['benchmark_mode']}",
        "",
        f"- cue_count: {report['cue_count']}",
        f"- duration: {report['total_duration_seconds']}s",
        f"- runtime: {report['runtime_seconds']}s",
        f"- quality_score: {report['quality_score']}",
        f"- contract: {report['contract_status']}",
        f"- risky_cues: {report['risky_cue_count']}",
        "",
        "## Top failure families",
        "",
    ]
    for fam, cnt in report.get("top_failure_families", []):
        lines.append(f"- {fam}: {cnt}")
    lines.extend(["", "## Segment windows (first/last)", ""])
    wins = report.get("segment_windows") or []
    for w in wins[:3] + wins[-3:]:
        lines.append(
            f"- cues {w['start_cue']}-{w['end_cue']}: score~{w['quality_score']} "
            f"risky={w['risky_cues']} layer={w['dominant_failure_layer']}"
        )
    lines.extend(["", "## Long-form checks", "", json.dumps(report.get("long_form_checks"), indent=2)])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="pipeline_regression")
    args = parser.parse_args()
    mode = args.mode
    bench_root = ROOT / "artifacts" / "multi_sample_benchmark" / mode
    artifact_dir = bench_root / SAMPLE_ID
    bench_report = _load_json(bench_root / "benchmark_report.json")
    if not artifact_dir.exists():
        print(f"Missing artifact dir: {artifact_dir}", file=sys.stderr)
        return 1
    report, cue_rows = build_report(artifact_dir, bench_report, mode)
    worst = _worst_slice(cue_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "long_form_success_232_v1.json"
    md_path = OUT / "long_form_success_232_v1.md"
    csv_path = OUT / "long_form_success_232_v1.csv"
    slice_md = OUT / "success_232_worst_slice_v1.md"
    slice_csv = OUT / "success_232_worst_slice_v1.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_md(report), encoding="utf-8")
    _write_csv(
        csv_path,
        report.get("segment_windows", []),
        [
            "start_cue",
            "end_cue",
            "quality_score",
            "risky_cues",
            "semantic_errors",
            "repeated_meaning_errors",
            "cps_errors",
            "cue_shift_suspicions",
            "dominant_failure_layer",
            "notes",
        ],
    )
    slice_csv_fields = [
        "cue_index",
        "time",
        "selection_reason",
        "source_en",
        "vi_raw",
        "vi_after_editor",
        "vi_after_flow",
        "vi_final",
        "auto_error_families",
        "cps",
        "chars_per_line",
        "likely_failure_layer",
        "SA_semantic_score",
        "SA_comment",
    ]
    _write_csv(slice_csv, worst, slice_csv_fields)
    slice_md.write_text(
        "# success_232 worst slice v1\n\n"
        + "\n".join(
            f"## Cue {r['cue_index']} ({r['selection_reason']})\n"
            f"- EN: {r['source_en']}\n"
            f"- final: {r['vi_final']}\n"
            f"- errors: {r['auto_error_families']}\n"
            for r in worst[:15]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"quality_score": report["quality_score"], "contract": report["contract_status"]}, indent=2))
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
