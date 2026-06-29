"""CPS diagnosis for subtitle delivery quality — timing vs text vs duration constraint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .semantic_alignment_guard import analyze_semantic_alignment
from .subtitle_timing_optimizer import _parse_ts, load_timing_config
from .translation_quality_analyzer import (
    _DEFAULT_MAX_CPS,
    _DEFAULT_RISKY_CPS,
    _detect_cue_assessment,
    _find_unit_for_cue,
)
from .vi_compression import _cps

_MICRO_CUE_MAX_DURATION = 0.55
_SHORT_TEXT_MAX_CHARS = 12


def _cue_duration(entry: dict) -> float:
    return max(0.0, _parse_ts(entry["end_str"]) - _parse_ts(entry["start_str"]))


def _estimate_timing_headroom(
    cue_idx: int,
    entries: List[dict],
    cfg,
) -> float:
    """Seconds of end-time extension still available before next cue hard ceiling."""
    if cue_idx < 1 or cue_idx > len(entries):
        return 0.0
    entry = entries[cue_idx - 1]
    end = _parse_ts(entry["end_str"])
    if cue_idx >= len(entries):
        return cfg.max_extend
    next_start = _parse_ts(entries[cue_idx]["start_str"])
    hard_ceil = next_start + cfg.max_overlap_allowed
    soft_ceil = end + cfg.max_extend
    return max(0.0, min(hard_ceil, soft_ceil) - end)


def _can_shorten_without_meaning_loss(text: str, source: str) -> bool:
    """Conservative heuristic: only suggest text shortening when clearly verbose."""
    t = text.strip()
    s = source.strip()
    if len(t) <= _SHORT_TEXT_MAX_CHARS:
        return False
    if len(t.split()) <= 4:
        return False
    filler_markers = ("thì ", "vì ", "rằng ", "là ", "đã ", "sẽ ")
    return any(t.lower().startswith(m) for m in filler_markers) or len(t) > len(s) * 2


def build_cps_diagnosis_report(
    source_entries: List[dict],
    pre_timing_entries: List[dict],
    post_timing_entries: List[dict],
    meaning_units: Optional[List[dict]] = None,
    video_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Diagnose CPS issues comparing pre- and post-timing delivery entries.

    Pre-timing reflects semantic pipeline output; post-timing reflects viewer-facing
    subtitles after the timing-only stage.
    """
    cfg = load_timing_config()
    alignment = analyze_semantic_alignment(
        source_entries,
        post_timing_entries,
        meaning_units or [],
        video_context,
    )
    semantic_error_cues = {
        issue["cue_index"]
        for issue in alignment.get("cue_issues", [])
        if issue.get("errors")
    }

    cps_cues: List[dict] = []
    summary = {
        "reducible_by_text": 0,
        "reducible_by_timing": 0,
        "duration_constraint_only": 0,
        "should_not_penalize_heavily": 0,
    }

    n = min(len(source_entries), len(pre_timing_entries), len(post_timing_entries))
    for i in range(1, n + 1):
        src = source_entries[i - 1].get("text", "")
        pre_e = pre_timing_entries[i - 1]
        post_e = post_timing_entries[i - 1]
        text = (pre_e.get("text") or "").strip()
        if not text:
            continue

        pre_dur = _cue_duration(pre_e)
        post_dur = _cue_duration(post_e)
        pre_cps = _cps(text, pre_dur)
        post_cps = _cps(text, post_dur)

        if pre_cps <= _DEFAULT_RISKY_CPS and post_cps <= _DEFAULT_RISKY_CPS:
            continue

        _, pre_errors = _detect_cue_assessment(
            i, src, text, pre_e, meaning_units or [], video_context
        )
        is_semantically_clean = i not in semantic_error_cues and not any(
            e for e in pre_errors if e != "readability_cps_error"
        )

        timing_delta = post_dur - pre_dur
        timing_reduced_cps = post_cps < pre_cps - 0.5
        timing_fixed = post_cps <= _DEFAULT_RISKY_CPS
        headroom = _estimate_timing_headroom(i, pre_timing_entries, cfg)
        can_extend = headroom >= 0.08 and not timing_fixed
        can_reduce = _can_shorten_without_meaning_loss(text, src)

        micro_cue = pre_dur <= _MICRO_CUE_MAX_DURATION or len(text) <= _SHORT_TEXT_MAX_CHARS
        duration_constraint = (
            micro_cue
            and is_semantically_clean
            and not can_reduce
            and (timing_fixed or (not can_extend and post_cps <= _DEFAULT_RISKY_CPS + 2))
        )

        if timing_fixed and timing_reduced_cps:
            recommended_action = "timing_extended"
            reason = (
                f"CPS {pre_cps:.1f}→{post_cps:.1f} via timing-only stage "
                f"(duration {pre_dur:.2f}s→{post_dur:.2f}s)"
            )
            summary["reducible_by_timing"] += 1
            summary["should_not_penalize_heavily"] += 1
        elif can_reduce and pre_cps > _DEFAULT_RISKY_CPS:
            recommended_action = "text_shorten"
            reason = "Text may be shortened without losing core meaning (pre-timing)"
            summary["reducible_by_text"] += 1
        elif can_extend:
            recommended_action = "extend_timing"
            reason = f"Additional timing headroom ~{headroom:.2f}s before next cue"
            summary["reducible_by_timing"] += 1
        elif duration_constraint:
            recommended_action = "duration_constraint_only"
            reason = (
                "Micro-cue with natural short text; CPS high pre-timing but "
                "acceptable or unfixable without meaning/timing contract risk"
            )
            summary["duration_constraint_only"] += 1
            summary["should_not_penalize_heavily"] += 1
        else:
            recommended_action = "monitor"
            reason = "CPS elevated; review manually"

        cps_cues.append(
            {
                "cue_index": i,
                "duration_pre_timing": round(pre_dur, 3),
                "duration_post_timing": round(post_dur, 3),
                "text": text,
                "source": src.strip(),
                "cps_pre_timing": round(pre_cps, 1),
                "cps_post_timing": round(post_cps, 1),
                "max_cps": _DEFAULT_MAX_CPS,
                "risky_cps_threshold": _DEFAULT_RISKY_CPS,
                "is_semantically_clean": is_semantically_clean,
                "can_reduce_text": can_reduce,
                "can_extend_timing": can_extend,
                "duration_constraint_only": duration_constraint,
                "meaning_unit_id": (_find_unit_for_cue(meaning_units or [], i) or {}).get(
                    "unit_id"
                ),
                "reason": reason,
                "recommended_action": recommended_action,
            }
        )

    return {
        "cps_error_count_pre_timing": len(cps_cues),
        "cps_error_count_post_timing": sum(
            1 for c in cps_cues if c["cps_post_timing"] > _DEFAULT_RISKY_CPS
        ),
        "cps_cues": cps_cues,
        "summary": summary,
    }


def save_cps_diagnosis_report(path: str | Path, report: Dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def merge_delivery_quality_report(
    pre_timing_report: Dict[str, Any],
    post_timing_report: Dict[str, Any],
    cps_diagnosis: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge pre-timing repair QA with post-timing delivery QA for artifacts."""
    merged = dict(post_timing_report)
    merged["pre_timing_after_repair"] = {
        "quality_score": pre_timing_report.get("quality_score"),
        "score_band": pre_timing_report.get("score_band"),
        "risky_cue_count": pre_timing_report.get("summary", {}).get(
            "risky_cue_count", len(pre_timing_report.get("risky_cues", []))
        ),
        "confirmed_error_counts": pre_timing_report.get("summary", {}).get(
            "confirmed_error_counts", {}
        ),
        "note": (
            "QA on vi_after_final_repair before timing-only stage; "
            "CPS warnings here may be resolved by timing optimizer."
        ),
    }
    if "before_repair" in pre_timing_report:
        merged["before_repair"] = pre_timing_report["before_repair"]
    if "repair" in pre_timing_report:
        merged["repair"] = pre_timing_report["repair"]
    merged["cps_diagnosis"] = {
        "summary": cps_diagnosis.get("summary", {}),
        "cps_error_count_pre_timing": cps_diagnosis.get("cps_error_count_pre_timing"),
        "cps_error_count_post_timing": cps_diagnosis.get("cps_error_count_post_timing"),
    }
    merged["delivery_stage"] = "vi_after_timing"
    return merged
