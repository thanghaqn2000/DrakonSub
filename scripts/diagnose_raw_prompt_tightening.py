#!/usr/bin/env python3
"""Diagnose borderline quality misses before raw prompt tightening."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "translation_quality_review"
PR = ROOT / "artifacts" / "multi_sample_benchmark" / "pipeline_regression"

FOCUS = {
    "no_rush_19": (1, 4, 6, 12, 17, 18),
    "buffett_bitcoin_29": (1, 2, 7, 8, 9, 10),
}


def _load_qa(sample: str) -> dict:
    p = PR / sample / "translation_quality_report.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _cue_row(report: dict, idx: int) -> dict:
    for a in report.get("cue_assessments") or []:
        if a.get("cue_index") == idx:
            return a
    return {}


def main() -> int:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version_before": "raw_v1",
        "prompt_version_after": "raw_v2",
        "samples": {},
    }
    lines = ["# Raw Prompt Tightening Diagnosis v1", ""]

    for sid, cues in FOCUS.items():
        qa = _load_qa(sid)
        risky = [a for a in qa.get("cue_assessments") or [] if a.get("is_risky")]
        focus = []
        for c in cues:
            row = _cue_row(qa, c)
            focus.append(
                {
                    "cue_index": c,
                    "en": row.get("en", ""),
                    "vi": row.get("vi", ""),
                    "errors": row.get("detected_translation_errors", []),
                    "is_risky": row.get("is_risky", False),
                    "calibration_notes": row.get("calibration_notes", []),
                }
            )
        if sid == "no_rush_19":
            root_cause = (
                "Cue 17 semantic_alignment (fragment too generic); cue 6 CPS; "
                "cue 1 uses kiếm tiền for take no money"
            )
        else:
            root_cause = (
                "Cues 4-5 repeated_meaning from batch overlap; "
                "raw absorbed multi-cue meaning into single lines"
            )
        report["samples"][sid] = {
            "quality_score": qa.get("quality_score"),
            "risky_cue_count": len(risky),
            "risky_cues": [
                {
                    "cue_index": r["cue_index"],
                    "errors": r.get("detected_translation_errors"),
                    "vi": r.get("vi"),
                }
                for r in risky
            ],
            "focus_cues": focus,
            "root_cause_summary": root_cause,
        }
        lines.append(f"## {sid} (score {qa.get('quality_score')})")
        lines.append(f"- {root_cause}")
        lines.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_prompt_tightening_diagnosis_v1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "raw_prompt_tightening_diagnosis_v1.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote diagnosis to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
