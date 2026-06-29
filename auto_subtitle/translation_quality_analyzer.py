"""Rule-first translation quality analysis with taxonomy-backed risk detection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .meaning_unit_builder import build_meaning_units
from .semantic_alignment_guard import (
    analyze_semantic_alignment,
    detect_repeated_meaning,
    _content_absorbed_in_previous_cue,
)
from .subtitle_timing_optimizer import _parse_ts
from .translation_error_taxonomy import ERROR_TYPES, get_error_type
from .vi_compression import _cps

_DEFAULT_MAX_CPS = 24.0
_DEFAULT_RISKY_CPS = 30.0
_MIN_MEANINGFUL_CHARS = 3

# Severity weights for confirmed translation errors (not source risk signals).
_ERROR_WEIGHTS: Dict[str, float] = {
    "missing_or_empty_cue_error": 20.0,
    "repair_contract_violation": 18.0,
    "semantic_drift_error": 14.0,
    "semantic_alignment_error": 15.0,
    "repeated_meaning_error": 5.0,
    "possible_asr_term_unresolved": 4.0,
    "cue_flow_error": 4.0,
    "cue_alignment_warning": 3.0,
    "domain_term_error": 6.0,
    "literal_translation_error": 5.0,
    "unnatural_vietnamese_error": 5.0,
    "over_compression_error": 5.0,
    "split_term_across_cues_error": 3.0,
    "pronoun_reference_error": 3.0,
    "idiom_metaphor_error": 4.0,
    "readability_cps_error": 2.0,
    "asr_possible_error": 1.0,
}

# Source-side flags from meaning units — never treated as confirmed VI errors.
_SOURCE_ONLY_FLAGS = frozenset(
    {
        "cue_fragmentation_error",
        "split_term_across_cues_error",
        "pronoun_reference_error",
    }
)

_WEAK_FRAGMENT_PATTERNS = [
    re.compile(r"(?i)^về\s+(điều đó|nó|điều này|cái đó|chuyện đó)\.?$"),
    re.compile(r"(?i)^về\s+.{1,18}\.?$"),
    re.compile(r"(?i)^của\s+(điều này|điều đó|nó)\.?$"),
]

_WEAK_FRAGMENT_START_RE = re.compile(
    r"(?i)^(?:về|với|của|rằng|mà|thì|và|hoặc|nhưng|nếu|khi|cho|để|là)\b"
)

_LITERAL_PATTERNS = [
    re.compile(r"(?i)\bphụ thuộc vào việc\b"),
    re.compile(r"(?i)\btrong việc\b"),
    re.compile(r"(?i)\bmột cách\b"),
    re.compile(r"(?i)\bđối với việc\b"),
]

_EN_REMNANT_RE = re.compile(
    r"(?i)\b(?:the|and|that|with|for|you|your|is|are|was|were)\b"
)


def _detect_asr_term_unresolved(
    source: str,
    vi: str,
    video_context: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    if not video_context:
        return False, ""
    src_l = source.lower()
    vi_l = vi.lower()
    for risk in video_context.get("possible_asr_risks") or []:
        phrase = str(risk).strip().lower()
        if len(phrase) < 3 or phrase not in src_l:
            continue
        if phrase in vi_l:
            return True, f"possible ASR term '{risk}' kept literally in VI"
        risk_tokens = [t for t in re.findall(r"[a-z']+", phrase) if len(t) >= 3]
        if len(risk_tokens) >= 2 and all(t in vi_l for t in risk_tokens):
            return True, f"possible ASR term '{risk}' appears verbatim in VI"
    return False, ""


def _cue_duration(entry: dict) -> float:
    return max(0.01, _parse_ts(entry["end_str"]) - _parse_ts(entry["start_str"]))


def _find_unit_for_cue(meaning_units: List[dict], cue_1based: int) -> Optional[dict]:
    for unit in meaning_units:
        if cue_1based in unit.get("cue_indexes", []):
            return unit
    return None


def _unit_source_risk_flags(unit: Optional[dict]) -> List[str]:
    if not unit:
        return []
    flags = unit.get("source_risk_flags") or unit.get("risk_flags") or []
    return [f for f in flags if f in ERROR_TYPES]


def _is_standalone_weak_fragment(text: str) -> bool:
    """True when VI cue cannot stand alone on screen."""
    text = text.strip()
    if not text:
        return False
    for pat in _WEAK_FRAGMENT_PATTERNS:
        if pat.match(text):
            return True
    words = text.split()
    if len(words) > 6:
        return False
    if len(words) == 1 and len(text) >= 4 and not _WEAK_FRAGMENT_START_RE.match(text):
        return False
    if len(words) >= 2 and not _WEAK_FRAGMENT_START_RE.match(text) and len(text) < 12:
        return False
    if len(text) < 12 and not text.endswith("?"):
        return True
    if _WEAK_FRAGMENT_START_RE.match(text) and len(words) <= 3:
        return True
    if re.fullmatch(r"(?i)(về|với|của)\s+.{1,20}\.?", text) and len(words) <= 4:
        return True
    return False


def _detect_split_term_issue(
    cue_idx: int,
    source: str,
    vi: str,
    meaning_units: List[dict],
) -> bool:
    """Warn only when VI looks like a broken partial term at a unit boundary."""
    unit = _find_unit_for_cue(meaning_units, cue_idx)
    if not unit or "split_term_across_cues_error" not in _unit_source_risk_flags(unit):
        return False
    vi = vi.strip()
    if not vi or len(vi.split()) > 4:
        return False
    if vi[0].islower() and not vi.endswith((".", "?", "!")):
        return True
    return False


def _detect_cue_assessment(
    cue_idx: int,
    source: str,
    vi: str,
    entry: dict,
    meaning_units: List[dict],
    video_context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Return (source_risk_flags, detected_translation_errors).
    Source risks inform context; only detected errors mark a cue risky.
    """
    source_risks: List[str] = []
    errors: List[str] = []
    src = source.strip()
    text = vi.strip()

    unit = _find_unit_for_cue(meaning_units, cue_idx)
    source_risks = _unit_source_risk_flags(unit)

    if src and not text:
        errors.append("missing_or_empty_cue_error")
        return source_risks, errors

    if not text:
        return source_risks, errors

    if len(text) < _MIN_MEANINGFUL_CHARS and src:
        errors.append("cue_flow_error")

    if _is_standalone_weak_fragment(text):
        errors.append("cue_flow_error")

    dur = _cue_duration(entry)
    cps_val = _cps(text, dur)
    if cps_val > _DEFAULT_RISKY_CPS:
        errors.append("readability_cps_error")

    for pat in _LITERAL_PATTERNS:
        if pat.search(text):
            errors.append("literal_translation_error")
            break

    if _EN_REMNANT_RE.search(text) and not re.search(
        r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
        text.lower(),
    ):
        errors.append("unnatural_vietnamese_error")

    if _detect_split_term_issue(cue_idx, src, text, meaning_units):
        errors.append("split_term_across_cues_error")

    asr_hit, _ = _detect_asr_term_unresolved(src, text, video_context)
    if asr_hit:
        errors.append("possible_asr_term_unresolved")

    return source_risks, errors


def _compute_quality_score(
    total_cues: int,
    empty_count: int,
    cue_assessments: List[dict],
    over_cps: int,
) -> int:
    """Score 0–100 with severity weights; source risks do not deduct."""
    if total_cues == 0:
        return 0

    deduction = empty_count * _ERROR_WEIGHTS["missing_or_empty_cue_error"]
    confirmed_counts: Dict[str, int] = {}

    for item in cue_assessments:
        for eid in (item.get("detected_translation_errors") or []) + (
            item.get("alignment_warnings") or []
        ):
            confirmed_counts[eid] = confirmed_counts.get(eid, 0) + 1

    for eid, count in confirmed_counts.items():
        weight = _ERROR_WEIGHTS.get(eid, 2.0)
        deduction += count * weight

    # Cap deduction so minor issues across many cues cannot force score to 0.
    max_deduction = 70.0
    deduction = min(deduction, max_deduction)

    # Small bonus when most cues are clean.
    clean = sum(
        1 for item in cue_assessments if not item.get("detected_translation_errors")
    )
    clean_ratio = clean / max(total_cues, 1)
    if clean_ratio >= 0.85 and empty_count == 0:
        deduction = max(0.0, deduction - 5.0)

    score = int(round(max(0.0, min(100.0, 100.0 - deduction))))
    return score


def _score_band(score: int) -> str:
    if score >= 85:
        return "good"
    if score >= 70:
        return "usable_minor_issues"
    if score >= 50:
        return "needs_review"
    if score >= 30:
        return "bad"
    return "broken"


def analyze_translation_quality(
    source_entries: List[dict],
    vi_entries: List[dict],
    video_context: Optional[Dict[str, Any]] = None,
    meaning_units: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Produce quality report with score, risky cues/units, and repair recommendations."""
    meaning_units = meaning_units or build_meaning_units(source_entries)
    video_context = video_context or {}

    if len(source_entries) != len(vi_entries):
        return {
            "quality_score": 0,
            "score_band": "broken",
            "cue_count_match": False,
            "cue_assessments": [],
            "risky_cues": [],
            "risky_units": [],
            "source_risk_summary": {},
            "repair_recommendations": [
                {
                    "error_type": "missing_or_empty_cue_error",
                    "action": "retranslate_all",
                    "reason": "Cue count mismatch between source and translation",
                }
            ],
            "human_review_needed": True,
            "summary": {
                "empty_cue_count": 0,
                "max_cps": 0.0,
                "over_cps_count": 0,
                "fragment_count": 0,
                "confirmed_error_counts": {},
            },
        }

    cue_assessments: List[dict] = []
    risky_cues: List[dict] = []
    risky_unit_ids: Set[int] = set()
    confirmed_error_counts: Dict[str, int] = {}
    source_risk_counts: Dict[str, int] = {}
    max_cps = 0.0
    over_cps = 0
    fragment_count = 0
    empty_count = 0

    alignment_report = analyze_semantic_alignment(
        source_entries, vi_entries, meaning_units, video_context
    )
    alignment_by_cue: Dict[int, List[str]] = {}
    alignment_warnings_by_cue: Dict[int, List[str]] = {}
    alignment_reasons: Dict[int, List[str]] = {}
    for issue in alignment_report.get("cue_issues", []):
        idx = issue["cue_index"]
        alignment_by_cue[idx] = issue.get("errors", [])
        alignment_reasons[idx] = issue.get("reasons", [])
    for issue in alignment_report.get("cue_warnings", []):
        idx = issue["cue_index"]
        alignment_warnings_by_cue[idx] = issue.get("errors", [])
        alignment_reasons.setdefault(idx, []).extend(issue.get("reasons", []))

    source_texts = [e.get("text", "") for e in source_entries]
    vi_texts = [e.get("text", "") for e in vi_entries]
    repeated_meaning = detect_repeated_meaning(source_texts, vi_texts, meaning_units)

    for i, (src_e, vi_e) in enumerate(zip(source_entries, vi_entries), start=1):
        src = src_e.get("text", "")
        vi = vi_e.get("text", "")
        source_risks, detected_errors = _detect_cue_assessment(
            i, src, vi, vi_e, meaning_units, video_context
        )
        if _content_absorbed_in_previous_cue(i, source_texts, vi_texts, video_context):
            detected_errors = [
                e
                for e in detected_errors
                if e
                not in (
                    "cue_flow_error",
                    "semantic_drift_error",
                    "semantic_alignment_error",
                    "missing_or_empty_cue_error",
                )
            ]
        alignment_warnings: List[str] = []
        for eid in alignment_by_cue.get(i, []):
            if eid not in detected_errors:
                detected_errors.append(eid)
        for eid in alignment_warnings_by_cue.get(i, []):
            if eid not in alignment_warnings:
                alignment_warnings.append(eid)
        if i in repeated_meaning:
            if "repeated_meaning_error" not in detected_errors:
                detected_errors.append("repeated_meaning_error")
            alignment_reasons.setdefault(i, []).append(repeated_meaning[i])

        if src.strip() and not vi.strip():
            empty_count += 1

        if vi.strip():
            c = _cps(vi.strip(), _cue_duration(vi_e))
            max_cps = max(max_cps, c)
            if c > _DEFAULT_MAX_CPS:
                over_cps += 1
            if _is_standalone_weak_fragment(vi.strip()):
                fragment_count += 1

        for flag in source_risks:
            source_risk_counts[flag] = source_risk_counts.get(flag, 0) + 1

        assessment = {
            "cue_index": i,
            "en": src.strip(),
            "vi": vi.strip(),
            "source_risk_flags": source_risks,
            "detected_translation_errors": detected_errors,
            "alignment_warnings": alignment_warnings,
            "alignment_reasons": alignment_reasons.get(i, []),
            "is_risky": bool(detected_errors),
        }
        cue_assessments.append(assessment)

        if detected_errors or alignment_warnings:
            for eid in detected_errors + alignment_warnings:
                confirmed_error_counts[eid] = confirmed_error_counts.get(eid, 0) + 1
        if detected_errors:
            risky_cues.append(
                {
                    "cue_index": i,
                    "source_risk_flags": source_risks,
                    "detected_translation_errors": detected_errors,
                    "alignment_reasons": alignment_reasons.get(i, []),
                    "vi": vi.strip(),
                    "en": src.strip(),
                }
            )
            unit = _find_unit_for_cue(meaning_units, i)
            if unit:
                risky_unit_ids.add(unit["unit_id"])

    risky_units: List[dict] = []
    repair_recommendations: List[dict] = []
    human_review_needed = False

    for unit in meaning_units:
        unit_confirmed: Set[str] = set()
        unit_source_risks: Set[str] = set(set(_unit_source_risk_flags(unit)))
        for a in cue_assessments:
            if a["cue_index"] not in unit.get("cue_indexes", []):
                continue
            unit_source_risks.update(a.get("source_risk_flags") or [])
            unit_confirmed.update(a.get("detected_translation_errors") or [])

        if not unit_confirmed:
            continue

        risky_units.append(
            {
                "unit_id": unit["unit_id"],
                "cue_indexes": unit["cue_indexes"],
                "source_risk_flags": sorted(unit_source_risks),
                "detected_translation_errors": sorted(unit_confirmed),
                "source_text": unit.get("source_text", ""),
            }
        )
        auto_errors = [e for e in unit_confirmed if get_error_type(e).auto_repair]
        review_errors = [e for e in unit_confirmed if get_error_type(e).human_review]
        if auto_errors:
            repair_recommendations.append(
                {
                    "unit_id": unit["unit_id"],
                    "cue_indexes": unit["cue_indexes"],
                    "error_types": auto_errors,
                    "action": "repair_unit",
                }
            )
        if review_errors:
            human_review_needed = True

    if empty_count:
        human_review_needed = True
    if alignment_report.get("human_review_needed"):
        human_review_needed = True
    if confirmed_error_counts.get("possible_asr_term_unresolved"):
        human_review_needed = True

    asr_risks_in_source = [
        r
        for r in (video_context.get("possible_asr_risks") or [])
        if any(
            str(r).lower() in (e.get("text", "") or "").lower()
            for e in source_entries
        )
    ]

    quality_score = _compute_quality_score(
        len(source_entries), empty_count, cue_assessments, over_cps
    )

    return {
        "quality_score": quality_score,
        "score_band": _score_band(quality_score),
        "cue_count_match": True,
        "cue_assessments": cue_assessments,
        "risky_cues": risky_cues,
        "risky_units": risky_units,
        "source_risk_summary": source_risk_counts,
        "repair_recommendations": repair_recommendations,
        "human_review_needed": human_review_needed,
        "semantic_alignment": {
            "summary": alignment_report.get("summary", {}),
            "alignment_error_count": alignment_report.get("alignment_error_count", 0),
            "alignment_warning_count": alignment_report.get("alignment_warning_count", 0),
        },
        "asr_risks": {
            "flagged_in_video_context": video_context.get("possible_asr_risks") or [],
            "present_in_source": asr_risks_in_source,
            "unresolved_cue_count": confirmed_error_counts.get(
                "possible_asr_term_unresolved", 0
            ),
        },
        "summary": {
            "empty_cue_count": empty_count,
            "max_cps": round(max_cps, 2),
            "over_cps_count": over_cps,
            "fragment_count": fragment_count,
            "confirmed_error_counts": confirmed_error_counts,
            "risky_cue_count": len(risky_cues),
            "clean_cue_count": len(source_entries) - len(risky_cues),
        },
    }


def save_quality_report(path: str | Path, report: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
