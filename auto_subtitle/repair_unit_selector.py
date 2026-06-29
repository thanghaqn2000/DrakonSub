"""Severity-based selection of meaning units for repair."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

MAX_REPAIR_UNITS = 6

_PRIORITY_1 = frozenset(
    {
        "semantic_drift_error",
        "semantic_alignment_error",
        "repair_contract_violation",
        "missing_or_empty_cue_error",
    }
)
_PRIORITY_2 = frozenset(
    {
        "cue_flow_error",
        "repeated_meaning_error",
        "possible_asr_term_unresolved",
        "literal_translation_error",
        "unnatural_vietnamese_error",
        "domain_term_error",
    }
)
_PRIORITY_3 = frozenset(
    {
        "readability_cps_error",
        "cue_alignment_warning",
        "over_compression_error",
        "split_term_across_cues_error",
    }
)

_ERROR_SEVERITY: Dict[str, int] = {
    "semantic_drift_error": 100,
    "semantic_alignment_error": 90,
    "repair_contract_violation": 85,
    "missing_or_empty_cue_error": 80,
    "cue_flow_error": 50,
    "repeated_meaning_error": 45,
    "possible_asr_term_unresolved": 40,
    "literal_translation_error": 35,
    "unnatural_vietnamese_error": 30,
    "domain_term_error": 28,
    "readability_cps_error": 10,
    "cue_alignment_warning": 5,
    "over_compression_error": 8,
    "split_term_across_cues_error": 6,
}


def _unit_error_set(unit: dict, cue_assessments: Optional[List[dict]] = None) -> Set[str]:
    errors: Set[str] = set(unit.get("detected_translation_errors") or [])
    cues = set(unit.get("cue_indexes") or [])
    if cue_assessments:
        for a in cue_assessments:
            if a.get("cue_index") in cues:
                errors.update(a.get("detected_translation_errors") or [])
                errors.update(a.get("alignment_warnings") or [])
    return errors


def _priority_tier(errors: Set[str]) -> int:
    if errors & _PRIORITY_1:
        return 1
    if errors & _PRIORITY_2:
        return 2
    return 3


def score_repair_unit(unit: dict, cue_assessments: Optional[List[dict]] = None) -> Dict[str, Any]:
    errors = _unit_error_set(unit, cue_assessments)
    tier = _priority_tier(errors)
    severity = sum(_ERROR_SEVERITY.get(e, 5) for e in errors)
    has_severe_semantic = bool(errors & {"semantic_alignment_error", "semantic_drift_error"})
    non_readability = errors - _PRIORITY_3 - {"asr_possible_error"}
    if has_severe_semantic and not (non_readability & _PRIORITY_2):
        severity += 15
    if errors == {"semantic_alignment_error"} or errors == {"semantic_drift_error"}:
        severity += 10
    if len(unit.get("cue_indexes") or []) == 1 and has_severe_semantic:
        severity += 25
    return {
        "unit_id": unit.get("unit_id"),
        "cue_indexes": unit.get("cue_indexes", []),
        "detected_errors": sorted(errors),
        "priority_tier": tier,
        "severity_score": severity,
        "has_severe_semantic": has_severe_semantic,
    }


def select_units_for_repair(
    risky_units: List[dict],
    *,
    cue_assessments: Optional[List[dict]] = None,
    max_units: int = MAX_REPAIR_UNITS,
) -> Dict[str, Any]:
    """
    Select units by severity (P1 semantic > P2 flow/repeat > P3 readability).
    Returns selected units and a full selection report.
    """
    candidates: List[dict] = []
    for unit in risky_units:
        scored = score_repair_unit(unit, cue_assessments)
        candidates.append({**unit, **scored})

    candidates.sort(
        key=lambda u: (u["priority_tier"], -u["severity_score"], u.get("unit_id", 0))
    )

    selected: List[dict] = []
    selected_ids: Set[int] = set()
    slots = max_units

    reserved_single_cue: Optional[dict] = None
    for u in sorted(candidates, key=lambda x: (-x["severity_score"], x.get("unit_id", 0))):
        errs = set(u.get("detected_errors") or [])
        if (
            u.get("has_severe_semantic")
            and len(u.get("cue_indexes") or []) == 1
            and "semantic_alignment_error" in errs
        ):
            reserved_single_cue = u
            break

    for tier in (1, 2, 3):
        tier_units = sorted(
            [u for u in candidates if u["priority_tier"] == tier],
            key=lambda x: -x["severity_score"],
        )
        for u in tier_units:
            if slots <= 0:
                break
            uid = u["unit_id"]
            if uid in selected_ids:
                continue
            selected.append(u)
            selected_ids.add(uid)
            slots -= 1

    if reserved_single_cue and reserved_single_cue["unit_id"] not in selected_ids:
        selected.append(reserved_single_cue)
        selected_ids.add(reserved_single_cue["unit_id"])

    effective_max = max_units + (1 if reserved_single_cue else 0)

    report_entries: List[dict] = []
    for u in candidates:
        uid = u["unit_id"]
        is_selected = uid in selected_ids
        entry: Dict[str, Any] = {
            "unit_id": uid,
            "cue_indexes": u.get("cue_indexes", []),
            "priority_tier": u["priority_tier"],
            "severity_score": u["severity_score"],
            "detected_errors": u.get("detected_errors", []),
            "has_severe_semantic": u.get("has_severe_semantic", False),
            "selected": is_selected,
        }
        if is_selected:
            rank = next(i for i, s in enumerate(selected, 1) if s["unit_id"] == uid)
            if (
                reserved_single_cue
                and uid == reserved_single_cue["unit_id"]
                and rank > max_units
            ):
                entry["selection_reason"] = "reserved_single_cue_semantic_slot"
            else:
                entry["selection_reason"] = f"severity_rank_{rank}_tier_{u['priority_tier']}"
        elif u.get("has_severe_semantic"):
            entry["selection_reason"] = (
                "skipped_severe_semantic"
                if len(selected) >= effective_max and u["priority_tier"] == 1
                else "not_in_top_budget"
            )
            entry["skip_reason"] = (
                f"repair budget ({effective_max}) exhausted by higher-severity units"
            )
        elif u["priority_tier"] == 1:
            entry["skip_reason"] = f"repair budget ({effective_max}) exhausted"
        else:
            entry["skip_reason"] = (
                f"lower priority (tier {u['priority_tier']}) — budget reserved for severe units"
            )
        report_entries.append(entry)

    return {
        "max_units": max_units,
        "effective_max_units": effective_max,
        "reserved_single_cue_semantic": reserved_single_cue is not None,
        "candidates_count": len(candidates),
        "selected_count": len(selected),
        "selected_unit_ids": [u["unit_id"] for u in selected],
        "candidates": report_entries,
        "selected_units": [
            {
                "unit_id": u["unit_id"],
                "cue_indexes": u.get("cue_indexes", []),
                "detected_translation_errors": u.get("detected_translation_errors", []),
                "source_risk_flags": u.get("source_risk_flags", []),
                "source_text": u.get("source_text", ""),
                "priority_tier": u["priority_tier"],
                "severity_score": u["severity_score"],
            }
            for u in selected
        ],
    }
