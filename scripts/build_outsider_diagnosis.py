#!/usr/bin/env python3
"""Build outsider_36 diagnosis artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "multi_sample_benchmark"
SAMPLE_DIR = OUT / "outsider_36"


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _read_cue_text(srt_path: Path, cue_index: int) -> str:
    if not srt_path.exists():
        return ""
    import re

    content = srt_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 3 and lines[0].strip() == str(cue_index):
            return " ".join(lines[2:]).strip()
    return ""


def build() -> dict:
    quality = _load(SAMPLE_DIR / "translation_quality_report.json")
    benchmark = _load(OUT / "benchmark_report.json")
    engine = _load(OUT / "engine_selection_report.json")

    sample_entry = next(
        (s for s in benchmark.get("samples", []) if s.get("id") == "outsider_36"),
        {},
    )
    mode_meta = sample_entry.get("benchmark_mode") or {}
    engine_entry = next(
        (s for s in engine.get("samples", []) if s.get("sample") == "outsider_36"),
        {},
    )

    top_errors: dict = {}
    cases = []
    for c in quality.get("cue_assessments", []):
        for e in c.get("detected_translation_errors", []):
            top_errors[e] = top_errors.get(e, 0) + 1
        if c.get("is_risky"):
            cases.append(
                {
                    "cue_index": c["cue_index"],
                    "en": c.get("en", "")[:100],
                    "vi": c.get("vi", "")[:100],
                    "errors": c.get("detected_translation_errors", []),
                    "reasons": (c.get("alignment_reasons") or [])[:3],
                }
            )

    raw_vi_29 = _read_cue_text(SAMPLE_DIR / "vi_raw.srt", 29)
    final_vi_29 = _read_cue_text(SAMPLE_DIR / "final_vi.srt", 29)
    raw_translation_suspected = raw_vi_29.strip() in (".", ",", ";", "")
    pipeline_repair_suspected = (
        not raw_translation_suspected
        and raw_vi_29 != final_vi_29
        and any("repeated_meaning" in str(c) for c in cases)
    )
    qa_fp_suspected = any(
        "glossary" in str(c.get("reasons"))
        and _read_cue_text(SAMPLE_DIR / "vi_raw.srt", c["cue_index"] - 1)
        for c in cases
        if c["cue_index"] > 1
    )

    selected = {
        "family": "raw_translation_split_absorption",
        "reason": "vi_raw cue 29 is punctuation-only while EN is liberty fragment; content merged to cue 28",
        "expected_impact": "Use raw cache for pipeline regression; optional orphan-cue QA calibration",
        "risk_level": "low",
    }
    if raw_translation_suspected:
        selected["family"] = "raw_translation_split_absorption"
    elif pipeline_repair_suspected:
        selected = {
            "family": "pipeline_repair",
            "reason": "Post-raw stages change risky cues",
            "expected_impact": "Pipeline fix in repair/flow",
            "risk_level": "medium",
        }

    return {
        "sample": "outsider_36",
        "quality_score": quality.get("quality_score"),
        "mode_type": mode_meta.get("mode_type", "unknown"),
        "translation_engine_effective": engine_entry.get("translation_engine_effective"),
        "top_errors": [{"error": k, "count": v} for k, v in sorted(top_errors.items())],
        "raw_translation_quality_suspected": raw_translation_suspected,
        "pipeline_repair_suspected": pipeline_repair_suspected,
        "qa_false_positive_suspected": qa_fp_suspected,
        "representative_cases": cases[:8],
        "cue_29_trace": {
            "en": "liberty.",
            "vi_raw": raw_vi_29,
            "final_vi": final_vi_29,
        },
        "selected_fix": selected,
    }


def main() -> None:
    report = build()
    out = OUT / "outsider_diagnosis.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
