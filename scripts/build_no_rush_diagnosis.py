#!/usr/bin/env python3
"""Build no_rush_19 error diagnosis artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "multi_sample_benchmark"
SAMPLE_DIR = OUT / "no_rush_19"


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def build() -> dict:
    quality = _load(SAMPLE_DIR / "translation_quality_report.json")
    alignment = _load(SAMPLE_DIR / "semantic_alignment_report_after_repair.json")
    shift = _load(SAMPLE_DIR / "cue_shift_diagnosis_sample.json")

    top_errors: dict = {}
    for c in quality.get("cue_assessments", []):
        for e in c.get("detected_translation_errors", []):
            top_errors[e] = top_errors.get(e, 0) + 1

    semantic_cases = []
    glossary_cases = []
    repeated_cases = []
    cps_cases = []

    for c in quality.get("cue_assessments", []):
        errs = c.get("detected_translation_errors") or []
        reasons = c.get("alignment_reasons") or []
        item = {
            "cue_index": c["cue_index"],
            "en": c.get("en", "")[:120],
            "vi": c.get("vi", "")[:120],
            "errors": errs,
            "reasons": reasons[:3],
        }
        if "semantic_alignment_error" in errs:
            semantic_cases.append(item)
        if any("glossary" in r for r in reasons):
            glossary_cases.append(item)
        if "repeated_meaning_error" in errs:
            repeated_cases.append(item)
        if "readability_cps_error" in errs:
            cps_cases.append(item)

    shift_windows = shift.get("shift_windows") or []
    is_cue_shift = bool(shift_windows) or shift.get("has_local_cue_shift")

    root_causes = []
    if semantic_cases and any("duplicate VI" in str(c.get("reasons")) for c in semantic_cases):
        root_causes.append(
            "Informal rhetorical EN repetition translated to identical VI on non-equivalent cues"
        )
    if glossary_cases:
        root_causes.append("Glossary phrase literal-match false positives on paraphrased VI")
    if repeated_cases:
        root_causes.append("Adjacent VI overlap across meaning units")
    if cps_cases:
        root_causes.append("CPS over limit post-timing on short cues")

    selected = {
        "family": "duplicate_vi_rhetorical_calibration",
        "reason": "no_rush semantic_alignment dominated by duplicate VI on rhetorically similar EN",
        "expected_impact": "Clear 4 semantic_alignment errors; quality +15-25",
        "risk_level": "low",
    }
    if not semantic_cases and cps_cases:
        selected = {
            "family": "readability_cps",
            "reason": "semantic cleared; CPS remains",
            "expected_impact": "Moderate quality lift",
            "risk_level": "medium",
        }

    return {
        "sample": "no_rush_19",
        "quality_score": quality.get("quality_score"),
        "top_errors": [{"error": k, "count": v} for k, v in sorted(top_errors.items())],
        "semantic_alignment_cases": semantic_cases[:8],
        "glossary_drift_cases": glossary_cases[:8],
        "repeated_adjacent_cases": repeated_cases[:8],
        "cps_cases": cps_cases[:8],
        "is_cue_shift_problem": is_cue_shift,
        "likely_root_causes": root_causes,
        "selected_fix": selected,
        "alignment_error_count": alignment.get("alignment_error_count"),
    }


def main() -> None:
    report = build()
    out = OUT / "no_rush_diagnosis.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
