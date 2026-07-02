"""Hybrid guarded raw translation — grouped baseline + selective cue_keyed repair."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .raw_cue_keyed_translate import _call_cue_keyed_batch, translate_single_cue_keyed
from .raw_translation_alignment_guard import (
    _FLAG_GENERIC,
    _FLAG_LENGTH,
    _FLAG_NEIGHBOR,
    _neighbor_bleed_score,
    _word_count,
    analyze_raw_alignment,
)
from .config import get_openai_model
from .translation_topics import normalize_topic

_CUE_KEYED_BATCH = 8
_HIGH_FLAGS = {_FLAG_LENGTH, _FLAG_NEIGHBOR}


def _classify_cue_severity(
    en: str,
    vi: str,
    prev_en: str,
    next_en: str,
    prev_vi: str,
    next_vi: str,
    cue_flags: List[str],
) -> str:
    if not cue_flags:
        return "NONE"
    en_wc = _word_count(en)
    vi_wc = _word_count(vi)
    neighbor_max, cur_overlap = _neighbor_bleed_score(vi, en, prev_en, next_en)

    if _FLAG_NEIGHBOR in cue_flags:
        if neighbor_max >= 0.5 and neighbor_max > cur_overlap + 0.25:
            return "HIGH"
        if prev_vi and len(vi) > 12 and vi[:20] == prev_vi[:20]:
            return "HIGH"
        if next_vi and len(vi) > 12 and vi[:20] == next_vi[:20]:
            return "HIGH"
        return "MEDIUM"

    if _FLAG_LENGTH in cue_flags:
        if en_wc <= 3 and vi_wc >= 10:
            return "HIGH"
        return "MEDIUM"

    if _FLAG_GENERIC in cue_flags:
        return "LOW"

    return "MEDIUM"


def analyze_with_severity(
    source_entries: List[dict],
    vi_entries: List[dict],
) -> Dict[str, Any]:
    base = analyze_raw_alignment(source_entries, vi_entries)
    n = min(len(source_entries), len(vi_entries))
    high: List[dict] = []
    medium: List[dict] = []
    low: List[dict] = []

    for item in base.get("flags", []):
        i = item["cue_index"] - 1
        en = item["en"]
        vi = item["vi"]
        prev_en = source_entries[i - 1].get("text", "").strip() if i > 0 else ""
        next_en = source_entries[i + 1].get("text", "").strip() if i + 1 < n else ""
        prev_vi = vi_entries[i - 1].get("text", "").strip() if i > 0 else ""
        next_vi = vi_entries[i + 1].get("text", "").strip() if i + 1 < n else ""
        severity = _classify_cue_severity(
            en, vi, prev_en, next_en, prev_vi, next_vi, item.get("flags", [])
        )
        enriched = {**item, "severity": severity}
        if severity == "HIGH":
            high.append(enriched)
        elif severity == "MEDIUM":
            medium.append(enriched)
        else:
            low.append(enriched)

    return {
        **base,
        "high_flags": high,
        "medium_flags": medium,
        "low_flags": low,
        "high_flag_count": len(high),
    }


def _batch_windows(entry_indexes: List[int], batch_size: int) -> List[List[int]]:
    return [
        entry_indexes[i : i + batch_size]
        for i in range(0, len(entry_indexes), batch_size)
    ]


def hybrid_guard_and_repair(
    source_entries: List[dict],
    vi_entries: List[dict],
    *,
    topic: Optional[str] = None,
    debug_dir: Optional[str] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    from openai import OpenAI

    analysis = analyze_with_severity(source_entries, vi_entries)
    working = [dict(e) for e in vi_entries]
    repairs: List[dict] = []
    batch_repairs = 0
    single_repairs = 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not analysis["high_flags"]:
        report = {
            "mode": "hybrid_guarded",
            "before": analysis,
            "after": analysis,
            "repairs": [],
            "repair_summary": {
                "total_high_flags": analysis["high_flag_count"],
                "single_cue_repairs": 0,
                "batch_repairs": 0,
                "unchanged_cues": analysis["cue_count"] - analysis["high_flag_count"],
            },
        }
        if debug_dir:
            out = Path(debug_dir) / "hybrid_guarded_report.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts = Path("artifacts/translation_quality_review")
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "hybrid_guarded_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return working, report

    client = OpenAI(api_key=api_key)
    model = get_openai_model()
    topic = normalize_topic(topic or os.environ.get("TRANSLATION_TOPIC"))

    high_indexes = sorted({f["cue_index"] - 1 for f in analysis["high_flags"]})
    non_empty = [i for i, e in enumerate(source_entries) if e.get("text", "").strip()]
    windows = _batch_windows(non_empty, _CUE_KEYED_BATCH)

    high_set = set(high_indexes)
    batch_rerun: set[int] = set()
    for window in windows:
        highs_in_window = [i for i in window if i in high_set]
        if len(highs_in_window) >= 2:
            batch_rerun.update(window)

    for window in windows:
        highs_in_window = [i for i in window if i in high_set]
        if len(highs_in_window) < 2:
            continue
        try:
            parsed = _call_cue_keyed_batch(
                client, model, source_entries, window, "vi", topic
            )
            batch_repairs += 1
            for entry_idx in window:
                before = working[entry_idx].get("text", "")
                after = parsed.get(entry_idx + 1, before)
                if after != before:
                    working[entry_idx] = {**working[entry_idx], "text": after}
                repairs.append(
                    {
                        "cue_index": entry_idx + 1,
                        "repair_type": "batch_cue_keyed",
                        "before": before,
                        "after": after,
                    }
                )
        except Exception as exc:
            for entry_idx in highs_in_window:
                repairs.append(
                    {
                        "cue_index": entry_idx + 1,
                        "repair_type": "batch_cue_keyed_failed",
                        "error": str(exc),
                    }
                )

    for item in analysis["high_flags"]:
        entry_idx = item["cue_index"] - 1
        if entry_idx in batch_rerun:
            continue
        before = working[entry_idx].get("text", "").strip()
        try:
            after = translate_single_cue_keyed(
                client, model, source_entries, entry_idx, "vi", topic
            )
            single_repairs += 1
            if after:
                working[entry_idx] = {**working[entry_idx], "text": after}
            repairs.append(
                {
                    "cue_index": item["cue_index"],
                    "repair_type": "single_cue_keyed",
                    "severity": "HIGH",
                    "flags": item.get("flags"),
                    "before": before,
                    "after": after,
                }
            )
        except Exception as exc:
            repairs.append(
                {
                    "cue_index": item["cue_index"],
                    "repair_type": "single_cue_keyed_failed",
                    "error": str(exc),
                    "before": before,
                }
            )

    after_analysis = analyze_with_severity(source_entries, working)
    report = {
        "mode": "hybrid_guarded",
        "before": analysis,
        "after": after_analysis,
        "repairs": repairs,
        "repair_summary": {
            "total_high_flags": analysis["high_flag_count"],
            "single_cue_repairs": single_repairs,
            "batch_repairs": batch_repairs,
            "unchanged_cues": analysis["cue_count"] - len(high_indexes),
        },
    }

    if debug_dir:
        out = Path(debug_dir) / "hybrid_guarded_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts = Path("artifacts/translation_quality_review")
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "hybrid_guarded_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return working, report


def translate_srt_entries_hybrid_openai(
    entries: List[dict],
    target_lang: str = "vi",
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    topic: Optional[str] = None,
    polish: Optional[bool] = None,
    translation_context: Optional[dict] = None,
    *,
    strict_cue_count: bool = False,
) -> List[dict]:
    """Grouped baseline translation, then selective HIGH-severity cue_keyed repair."""
    from .openai_translate import translate_srt_entries_openai

    prev_mode = os.environ.get("RAW_TRANSLATION_MODE")
    os.environ["RAW_TRANSLATION_MODE"] = "grouped"
    try:
        baseline = translate_srt_entries_openai(
            entries,
            target_lang=target_lang,
            model=model,
            batch_size=batch_size,
            topic=topic,
            polish=polish,
            translation_context=translation_context,
            strict_cue_count=strict_cue_count,
        )
    finally:
        if prev_mode is None:
            os.environ.pop("RAW_TRANSLATION_MODE", None)
        else:
            os.environ["RAW_TRANSLATION_MODE"] = prev_mode

    repaired, _report = hybrid_guard_and_repair(
        entries,
        baseline,
        topic=topic,
    )
    return repaired
