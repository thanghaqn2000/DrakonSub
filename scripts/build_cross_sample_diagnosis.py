#!/usr/bin/env python3
"""Build cross-sample error diagnosis from benchmark artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "artifacts" / "multi_sample_benchmark"

FAMILIES = [
    "semantic_drift_error",
    "repeated_meaning_error",
    "cue_flow_error",
    "semantic_alignment_error",
    "readability_cps_error",
]


def _load_benchmark() -> dict:
    path = OUT_ROOT / "benchmark_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_errors() -> Dict[str, dict]:
    agg: Dict[str, dict] = {f: {"sample_count": 0, "cue_count": 0, "samples": set(), "cases": []} for f in FAMILIES}
    for sample_dir in sorted(OUT_ROOT.iterdir()):
        if not sample_dir.is_dir():
            continue
        qr = sample_dir / "translation_quality_report.json"
        if not qr.exists():
            continue
        rep = json.loads(qr.read_text(encoding="utf-8"))
        sid = sample_dir.name
        for c in rep.get("cue_assessments", []):
            for e in c.get("detected_translation_errors", []):
                if e not in agg:
                    continue
                agg[e]["cue_count"] += 1
                agg[e]["samples"].add(sid)
                if len(agg[e]["cases"]) < 4:
                    agg[e]["cases"].append(
                        {
                            "sample": sid,
                            "cue_index": c["cue_index"],
                            "en": c.get("en", "")[:100],
                            "vi": c.get("vi", "")[:100],
                            "reasons": (c.get("alignment_reasons") or [])[:3],
                        }
                    )
    for f in agg:
        agg[f]["sample_count"] = len(agg[f]["samples"])
        agg[f]["samples"] = sorted(agg[f]["samples"])
    return agg


def build_diagnosis(benchmark: dict, errors: Dict[str, dict]) -> dict:
    agg = benchmark.get("aggregate", {})
    families_out = []
    recommendations = {
        "semantic_drift_error": "Calibrate discourse-marker vs named-entity detection; downgrade unit-level fragment drift",
        "repeated_meaning_error": "Skip legitimate cross-cue continuation within same meaning unit",
        "cue_flow_error": "Relax weak-fragment heuristics for complete short noun/verb phrases",
        "semantic_alignment_error": "Tighten glossary misplacement + cue-shift false positives",
        "readability_cps_error": "Score on post-timing delivery (out of scope this phase)",
    }
    for family in FAMILIES:
        data = errors.get(family, {})
        families_out.append(
            {
                "family": family.replace("_error", ""),
                "sample_count": data.get("sample_count", 0),
                "cue_count": data.get("cue_count", 0),
                "representative_cases": data.get("cases", []),
                "likely_root_causes": [],
                "recommended_fix": recommendations.get(family, ""),
            }
        )
    drift = errors.get("semantic_drift_error", {})
    repeat = errors.get("repeated_meaning_error", {})
    flow = errors.get("cue_flow_error", {})
    selected = "semantic_drift_and_qa_calibration"
    if repeat.get("cue_count", 0) >= drift.get("cue_count", 0):
        selected = "repeated_meaning_and_fragment_qa"
    return {
        "benchmark_summary": {
            "sample_count": agg.get("samples_run", 0),
            "contract_pass": f"{agg.get('contract_pass', 0)}/{agg.get('samples_run', 0)}",
            "quality_min": agg.get("quality_score_min"),
            "quality_avg": agg.get("quality_score_avg"),
            "quality_max": agg.get("quality_score_max"),
            "total_risky_cues": agg.get("total_risky_cues"),
        },
        "error_families": families_out,
        "selected_fix_for_this_task": {
            "family": selected,
            "reason": (
                "semantic_drift false positives from capitalized discourse words + "
                "repeated_meaning on split cues + cue_flow on valid micro-fragments"
            ),
            "expected_impact": "quality_avg +15–25 on informal/motivational samples",
            "risk_level": "medium — QA calibration only, no pipeline contract change",
        },
        "notes": {
            "false_positive_patterns": [
                "named entity missing for Suppose/Well/Embrace",
                "repeated meaning on same-unit cue splits",
                "cue_flow on complete 1–2 word translations",
            ],
            "true_errors_remaining": [
                "no_rush semantic alignment on cue 17–18",
                "some genuine semantic drift on long units",
            ],
        },
    }


def main() -> None:
    benchmark = _load_benchmark()
    errors = _aggregate_errors()
    report = build_diagnosis(benchmark, errors)
    out = OUT_ROOT / "cross_sample_error_diagnosis.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
