#!/usr/bin/env python3
"""Diagnose current raw translation path and alignment risks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_subtitle.config import get_raw_translation_mode  # noqa: E402

OUT = ROOT / "artifacts" / "translation_quality_review" / "raw_translation_diagnosis_v1.json"


def main() -> int:
    mode = get_raw_translation_mode()
    diagnosis = {
        "current_strategy": (
            "cue_keyed batches with strict cue_index JSON"
            if mode == "cue_keyed"
            else "meaning-unit/phrase-grouped batch translation with indexed or order-based JSON"
        ),
        "uses_grouped_translation": mode != "cue_keyed",
        "output_keyed_by_cue_index": mode == "cue_keyed",
        "order_based_mapping_risks": [
            "grouped mode: Gemini uses flat translations[] array mapped by batch order",
            "grouped mode: batch retry splits groups then maps zip(local_idx, results) by position",
            "meaning_unit_builder can group 2-5 cues — model may bleed context across unit",
            "_parse_json_strings validates count only, not semantic cue alignment",
        ],
        "json_parse_fallbacks": [
            "OpenAI grouped: repair retry on validation failure",
            "OpenAI grouped: cascade to per-group then per-cue on batch failure",
            "per-cue failure: fallback to source EN text",
        ],
        "context_bleed_risks": [
            "±5 cue English context window in grouped OpenAI prompts",
            "meaning units merge fragmented cues before translation",
            "prompt encourages reading full group as complete thought (legacy grouped prompt)",
        ],
        "likely_root_causes": [
            "batch/group translation lets model assign neighbor meaning to wrong cue_index",
            "order-based JSON mapping cannot detect semantic misassignment when count matches",
            "large context window + localization rewrite style increases hallucinated additions",
        ],
        "recommended_fix": (
            "RAW_TRANSLATION_MODE=cue_keyed with raw_translation_alignment_guard "
            "and single-cue repair for flagged cues"
            if mode != "cue_keyed"
            else "cue_keyed active; monitor raw_alignment_guard_report.json after runs"
        ),
        "active_raw_translation_mode": mode,
        "implementation_files": [
            "auto_subtitle/openai_translate.py",
            "auto_subtitle/gemini_translate.py",
            "auto_subtitle/meaning_unit_builder.py",
            "auto_subtitle/raw_cue_keyed_translate.py",
            "auto_subtitle/raw_translation_alignment_guard.py",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
