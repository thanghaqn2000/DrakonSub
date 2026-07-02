"""Span-level alignment guard — detect fragmented-span misalignment, conservative repair."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .raw_cue_keyed_translate import _call_cue_keyed_batch, translate_single_cue_keyed
from .raw_translation_alignment_guard import (
    _neighbor_bleed_score,
    _word_count,
    analyze_raw_alignment,
)
from .semantic_alignment_guard import _extract_concepts, _overlap_ratio

WINDOW_SIZES = (3, 4, 5)
MIN_SIGNALS_FOR_HIGH = 2

RISK_FRAGMENT_SPAN = "fragment_span_misalignment"
RISK_NEIGHBOR_BLEED = "neighbor_concept_bleed"
RISK_SHORT_LONG = "short_source_long_vi"
RISK_LOW_HIGH_NEIGHBOR = "low_source_high_neighbor_overlap"
RISK_TOPIC_JUMP = "span_topic_jump"
RISK_ORPHAN = "orphan_fragment_translation"
RISK_REPEATED = "repeated_neighbor_meaning"

SEMANTIC_BLEED_SIGNALS = {
    RISK_NEIGHBOR_BLEED,
    RISK_TOPIC_JUMP,
    RISK_LOW_HIGH_NEIGHBOR,
}
FRAGMENT_RISK_SIGNALS = {
    RISK_FRAGMENT_SPAN,
    RISK_SHORT_LONG,
    RISK_REPEATED,
}


def _cue_signals(
    source_entries: List[dict],
    vi_entries: List[dict],
    i: int,
) -> Tuple[List[str], dict]:
    """Return risk signatures and metrics for cue at index i."""
    n = min(len(source_entries), len(vi_entries))
    en = source_entries[i].get("text", "").strip()
    vi = vi_entries[i].get("text", "").strip()
    if not en:
        return [], {}

    prev_en = source_entries[i - 1].get("text", "").strip() if i > 0 else ""
    next_en = source_entries[i + 1].get("text", "").strip() if i + 1 < n else ""
    prev_vi = vi_entries[i - 1].get("text", "").strip() if i > 0 else ""
    next_vi = vi_entries[i + 1].get("text", "").strip() if i + 1 < n else ""

    en_wc = _word_count(en)
    vi_wc = _word_count(vi)
    neighbor_max, cur_overlap = _neighbor_bleed_score(vi, en, prev_en, next_en)

    signals: List[str] = []
    if en_wc <= 6 and vi_wc >= max(en_wc * 2, en_wc + 4):
        signals.append(RISK_SHORT_LONG)

    if neighbor_max >= 0.35 and neighbor_max > cur_overlap + 0.18:
        signals.append(RISK_NEIGHBOR_BLEED)
        signals.append(RISK_LOW_HIGH_NEIGHBOR)

    if cur_overlap < 0.12 and en_wc >= 3:
        signals.append(RISK_ORPHAN)

    if not en.rstrip().endswith((".", "?", "!")) and en_wc <= 8:
        signals.append(RISK_FRAGMENT_SPAN)

    span_en = " ".join(
        source_entries[j].get("text", "").strip()
        for j in range(max(0, i - 1), min(n, i + 2))
    )
    span_vi = " ".join(
        vi_entries[j].get("text", "").strip()
        for j in range(max(0, i - 1), min(n, i + 2))
    )
    span_ratio = _overlap_ratio(_extract_concepts(span_vi, "vi"), _extract_concepts(en, "en"))
    if span_ratio > 0.5 and cur_overlap < 0.15:
        signals.append(RISK_TOPIC_JUMP)

    if prev_vi and vi and _overlap_ratio(
        _extract_concepts(vi, "vi"), _extract_concepts(prev_vi, "vi")
    ) >= 0.7:
        signals.append(RISK_REPEATED)

    return sorted(set(signals)), {
        "cur_overlap": round(cur_overlap, 3),
        "neighbor_max": round(neighbor_max, 3),
        "en_wc": en_wc,
        "vi_wc": vi_wc,
        "span_ratio": round(span_ratio, 3),
    }


def _severity_from_signals(signals: List[str], *, conservative: bool = False) -> str:
    if conservative:
        return _severity_conservative(signals)
    if len(signals) >= MIN_SIGNALS_FOR_HIGH:
        return "HIGH"
    if signals:
        return "MEDIUM"
    return "NONE"


def _severity_conservative(signals: List[str]) -> str:
    sig_set = set(signals)
    has_bleed = bool(sig_set & SEMANTIC_BLEED_SIGNALS)
    has_frag = bool(sig_set & FRAGMENT_RISK_SIGNALS)
    strong_repeat = RISK_REPEATED in sig_set and RISK_NEIGHBOR_BLEED in sig_set

    if (has_bleed or strong_repeat) and (has_frag or RISK_ORPHAN in sig_set):
        return "HIGH"
    if RISK_ORPHAN in sig_set and not has_bleed:
        return "LOW" if len(sig_set) == 1 else "MEDIUM"
    if sig_set:
        return "MEDIUM"
    return "NONE"


def _old_severity_from_signals(signals: List[str]) -> str:
    if len(signals) >= MIN_SIGNALS_FOR_HIGH:
        return "HIGH"
    if signals:
        return "MEDIUM"
    return "NONE"


def _window_looks_good(
    source_entries: List[dict],
    vi_entries: List[dict],
    entry_indexes: List[int],
) -> bool:
    """Skip repair when grouped window has no semantic bleed signals."""
    for entry_idx in entry_indexes:
        signals, metrics = _cue_signals(source_entries, vi_entries, entry_idx)
        if sig_set := set(signals):
            if sig_set & SEMANTIC_BLEED_SIGNALS:
                return False
            if metrics.get("neighbor_max", 0) >= 0.42:
                return False
            if metrics.get("cur_overlap", 0) < 0.08 and metrics.get("en_wc", 0) >= 5:
                return False
            if metrics.get("span_ratio", 0) > 0.55 and metrics.get("cur_overlap", 0) < 0.12:
                return False
        en_wc = metrics.get("en_wc", 0) if metrics else _word_count(
            source_entries[entry_idx].get("text", "")
        )
        vi_wc = metrics.get("vi_wc", 0) if metrics else _word_count(
            vi_entries[entry_idx].get("text", "")
        )
        if en_wc > 0 and (vi_wc / en_wc > 4.5 or vi_wc / en_wc < 0.2):
            return False
    return True


def _risk_profile(
    source_entries: List[dict],
    vi_entries: List[dict],
    entry_idx: int,
    text: str,
) -> dict:
    n = min(len(source_entries), len(vi_entries))
    en = source_entries[entry_idx].get("text", "").strip()
    prev_en = source_entries[entry_idx - 1].get("text", "").strip() if entry_idx > 0 else ""
    next_en = source_entries[entry_idx + 1].get("text", "").strip() if entry_idx + 1 < n else ""
    neighbor_max, cur_overlap = _neighbor_bleed_score(text, en, prev_en, next_en)
    en_wc = _word_count(en)
    vi_wc = _word_count(text)
    return {
        "neighbor_bleed": round(neighbor_max, 3),
        "cur_overlap": round(cur_overlap, 3),
        "length_ratio": round(vi_wc / en_wc, 2) if en_wc else 0,
    }


def analyze_span_alignment(
    source_entries: List[dict],
    vi_entries: List[dict],
    *,
    conservative: bool = False,
) -> Dict[str, Any]:
    """Analyze sliding windows for span-level alignment risk."""
    n = min(len(source_entries), len(vi_entries))
    cue_analysis: Dict[int, dict] = {}
    windows: List[dict] = []

    for i in range(n):
        if not source_entries[i].get("text", "").strip():
            continue
        signals, metrics = _cue_signals(source_entries, vi_entries, i)
        old_severity = _old_severity_from_signals(signals)
        severity = _severity_from_signals(signals, conservative=conservative)
        cue_analysis[i + 1] = {
            "cue_index": i + 1,
            "en": source_entries[i].get("text", "").strip(),
            "vi": vi_entries[i].get("text", "").strip(),
            "risk_signatures": signals,
            "signals": {s: True for s in signals},
            "old_severity": old_severity,
            "severity": severity,
            "new_severity": severity,
            "metrics": metrics,
        }

    seen_windows: Set[Tuple[int, ...]] = set()
    for window_size in WINDOW_SIZES:
        for start in range(0, n - window_size + 1):
            indices = tuple(range(start + 1, start + window_size + 1))
            if indices in seen_windows:
                continue
            texts = [source_entries[j - 1].get("text", "").strip() for j in indices]
            if not any(texts):
                continue
            seen_windows.add(indices)

            span_signals: Set[str] = set()
            severities: List[str] = []
            for idx in indices:
                ca = cue_analysis.get(idx, {})
                span_signals.update(ca.get("risk_signatures") or [])
                severities.append(ca.get("severity", "NONE"))

            high_in_window = sum(1 for s in severities if s == "HIGH")
            medium_in_window = sum(1 for s in severities if s == "MEDIUM")
            if high_in_window >= 2:
                severity = "HIGH"
                decision = "span_repair_candidate"
            elif high_in_window == 1:
                severity = "HIGH"
                decision = "single_cue_repair_candidate"
            elif medium_in_window >= 2 and len(span_signals) >= 2:
                severity = "MEDIUM"
                decision = "log_only"
            elif medium_in_window:
                severity = "MEDIUM"
                decision = "log_only"
            else:
                continue

            before = {
                str(idx): cue_analysis.get(idx, {}).get("vi", "")
                for idx in indices
            }
            why_not = None
            if severity == "MEDIUM":
                why_not = (
                    f"only {medium_in_window} MEDIUM cue(s), "
                    f"signals={sorted(span_signals)} — need {MIN_SIGNALS_FOR_HIGH}+ agreeing signals per cue for HIGH"
                )

            windows.append(
                {
                    "cue_indices": list(indices),
                    "window_size": window_size,
                    "risk_signatures": sorted(span_signals),
                    "signals": {s: s in span_signals for s in sorted(span_signals)},
                    "severity": severity,
                    "decision": decision,
                    "before": before,
                    "why_not_repaired": why_not,
                    "skip_reason": why_not,
                }
            )

    high_cues = [c for c in cue_analysis.values() if c["severity"] == "HIGH"]
    medium_cues = [c for c in cue_analysis.values() if c["severity"] == "MEDIUM"]

    mode_name = "span_guarded_conservative" if conservative else "span_guarded"
    return {
        "mode": mode_name,
        "conservative": conservative,
        "cue_count": n,
        "cues": list(cue_analysis.values()),
        "windows": windows,
        "high_cues": high_cues,
        "medium_cues": medium_cues,
        "high_count": len(high_cues),
        "medium_count": len(medium_cues),
        "cue_level": analyze_raw_alignment(source_entries, vi_entries),
    }


def _repair_rejected(
    before: str,
    after: str,
    en: str,
    source_entries: List[dict],
    vi_entries: List[dict],
    entry_idx: int,
    reason: str,
    *,
    conservative: bool = False,
) -> Tuple[bool, Optional[str]]:
    if not after or not after.strip():
        return True, "empty_vi"
    en_wc = _word_count(en)
    vi_wc = _word_count(after)
    if en_wc > 0 and vi_wc > en_wc * 4 and en_wc <= 6:
        return True, "length_blowup"
    n = min(len(source_entries), len(vi_entries))
    prev_en = source_entries[entry_idx - 1].get("text", "").strip() if entry_idx > 0 else ""
    next_en = source_entries[entry_idx + 1].get("text", "").strip() if entry_idx + 1 < n else ""
    nb_after, cur_after = _neighbor_bleed_score(after, en, prev_en, next_en)
    nb_before, cur_before = _neighbor_bleed_score(before, en, prev_en, next_en)
    if nb_after > nb_before + 0.15:
        return True, "higher_neighbor_bleed"

    if conservative:
        before_risk = _risk_profile(source_entries, vi_entries, entry_idx, before)
        after_risk = _risk_profile(source_entries, vi_entries, entry_idx, after)
        if after_risk["neighbor_bleed"] > before_risk["neighbor_bleed"] + 0.05:
            return True, "repair_not_better_than_before"
        if after_risk["cur_overlap"] < before_risk["cur_overlap"] - 0.1:
            return True, "repair_not_better_than_before"
        if (
            en_wc <= 6
            and after_risk["length_ratio"] > before_risk["length_ratio"] + 1.5
        ):
            return True, "repair_worse_readability"
        if nb_after > 0.45 and cur_after < 0.1:
            return True, "new_topic_jump"
        if (
            before_risk["neighbor_bleed"] <= before_risk["cur_overlap"] + 0.1
            and after_risk["neighbor_bleed"] > after_risk["cur_overlap"] + 0.1
        ):
            return True, "repair_rejected_uncertain"
    return False, None


def _group_adjacent_high(indexes: List[int]) -> List[List[int]]:
    if not indexes:
        return []
    sorted_idx = sorted(set(indexes))
    groups: List[List[int]] = []
    current = [sorted_idx[0]]
    for idx in sorted_idx[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    return groups


def span_guard_and_repair(
    source_entries: List[dict],
    vi_entries: List[dict],
    *,
    topic: Optional[str] = None,
    debug_dir: Optional[str] = None,
    sample_id: Optional[str] = None,
    conservative: bool = False,
) -> Tuple[List[dict], Dict[str, Any]]:
    import os

    from openai import OpenAI

    from .config import get_openai_model
    from .translation_topics import normalize_topic

    analysis = analyze_span_alignment(source_entries, vi_entries, conservative=conservative)
    working = [dict(e) for e in vi_entries]
    repairs: List[dict] = []
    single_repairs = 0
    span_repairs = 0
    repair_rejected = 0
    skipped_windows = 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not analysis["high_cues"]:
        report = _build_report(
            analysis,
            repairs,
            sample_id,
            single_repairs,
            span_repairs,
            repair_rejected,
            skipped_windows,
            conservative,
        )
        _write_reports(report, debug_dir, conservative=conservative)
        return working, report

    client = OpenAI(api_key=api_key)
    model = get_openai_model()
    topic = normalize_topic(topic or os.environ.get("TRANSLATION_TOPIC"))

    high_entry_indexes = sorted(
        {c["cue_index"] - 1 for c in analysis["high_cues"]}
    )
    span_groups = _group_adjacent_high([i + 1 for i in high_entry_indexes])
    span_groups_entry = [
        [idx - 1 for idx in group] for group in span_groups
    ]

    repaired_indexes: Set[int] = set()

    for group in span_groups_entry:
        if conservative and _window_looks_good(source_entries, working, group):
            skipped_windows += 1
            for entry_idx in group:
                repairs.append(
                    {
                        "cue_index": entry_idx + 1,
                        "repair_type": "skip_repair",
                        "skip_reason": "window_quality_precheck_passed",
                    }
                )
            continue
        if len(group) >= 2:
            try:
                parsed = _call_cue_keyed_batch(
                    client, model, source_entries, group, "vi", topic
                )
                span_repairs += 1
                for entry_idx in group:
                    cue_index = entry_idx + 1
                    before = working[entry_idx].get("text", "").strip()
                    after = parsed.get(cue_index, before)
                    en = source_entries[entry_idx].get("text", "").strip()
                    rejected, reject_reason = _repair_rejected(
                        before,
                        after,
                        en,
                        source_entries,
                        working,
                        entry_idx,
                        "span",
                        conservative=conservative,
                    )
                    if rejected:
                        repair_rejected += 1
                        repairs.append(
                            {
                                "cue_index": cue_index,
                                "repair_type": "span_cue_keyed_rejected",
                                "before": before,
                                "after": after,
                                "skip_reason": reject_reason,
                                "before_risk": _risk_profile(
                                    source_entries, working, entry_idx, before
                                ),
                                "after_risk": _risk_profile(
                                    source_entries, working, entry_idx, after
                                ),
                            }
                        )
                        continue
                    if after != before:
                        working[entry_idx] = {**working[entry_idx], "text": after}
                    repaired_indexes.add(entry_idx)
                    repairs.append(
                        {
                            "cue_index": cue_index,
                            "repair_type": "span_cue_keyed",
                            "before": before,
                            "after": after,
                            "repair_status": "accepted",
                        }
                    )
            except Exception as exc:
                for entry_idx in group:
                    repairs.append(
                        {
                            "cue_index": entry_idx + 1,
                            "repair_type": "span_cue_keyed_failed",
                            "error": str(exc),
                        }
                    )
        else:
            entry_idx = group[0]
            if entry_idx in repaired_indexes:
                continue
            cue_index = entry_idx + 1
            before = working[entry_idx].get("text", "").strip()
            en = source_entries[entry_idx].get("text", "").strip()
            try:
                after = translate_single_cue_keyed(
                    client, model, source_entries, entry_idx, "vi", topic
                )
                single_repairs += 1
                rejected, reject_reason = _repair_rejected(
                    before,
                    after,
                    en,
                    source_entries,
                    working,
                    entry_idx,
                    "single",
                    conservative=conservative,
                )
                if rejected:
                    repair_rejected += 1
                    repairs.append(
                        {
                            "cue_index": cue_index,
                            "repair_type": "single_cue_keyed_rejected",
                            "before": before,
                            "after": after,
                            "skip_reason": reject_reason,
                            "before_risk": _risk_profile(
                                source_entries, working, entry_idx, before
                            ),
                            "after_risk": _risk_profile(
                                source_entries, working, entry_idx, after
                            ),
                        }
                    )
                    continue
                if after:
                    working[entry_idx] = {**working[entry_idx], "text": after}
                repairs.append(
                    {
                        "cue_index": cue_index,
                        "repair_type": "single_cue_keyed",
                        "before": before,
                        "after": after,
                        "repair_status": "accepted",
                    }
                )
            except Exception as exc:
                repairs.append(
                    {
                        "cue_index": cue_index,
                        "repair_type": "single_cue_keyed_failed",
                        "error": str(exc),
                    }
                )

    for w in analysis["windows"]:
        if w.get("severity") == "HIGH" and w.get("decision", "").endswith("_candidate"):
            idxs = w["cue_indices"]
            w["after"] = {str(i): working[i - 1].get("text", "") for i in idxs}
            w["repair_status"] = "accepted" if any(
                r.get("repair_status") == "accepted"
                and r.get("cue_index") in idxs
                for r in repairs
            ) else "skipped"

    report = _build_report(
        analysis,
        repairs,
        sample_id,
        single_repairs,
        span_repairs,
        repair_rejected,
        skipped_windows,
        conservative,
    )
    _write_reports(report, debug_dir, conservative=conservative)
    return working, report


def _build_report(
    analysis: dict,
    repairs: List[dict],
    sample_id: Optional[str],
    single_repairs: int,
    span_repairs: int,
    repair_rejected: int,
    skipped_windows: int,
    conservative: bool,
) -> dict:
    mode = "span_guarded_conservative" if conservative else "span_guarded"
    return {
        "mode": mode,
        "conservative": conservative,
        "sample_id": sample_id,
        "analysis": analysis,
        "windows": analysis.get("windows", []),
        "repairs": repairs,
        "summary": {
            "high_count": analysis.get("high_count", 0),
            "medium_count": analysis.get("medium_count", 0),
            "single_cue_repairs": single_repairs,
            "span_repairs": span_repairs,
            "repair_rejected": repair_rejected,
            "skipped_windows": skipped_windows,
        },
    }


def _write_reports(report: dict, debug_dir: Optional[str], *, conservative: bool = False) -> None:
    report_name = (
        "span_guarded_conservative_report.json"
        if conservative
        else "span_alignment_guard_v2_report.json"
    )
    if debug_dir:
        out = Path(debug_dir) / report_name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts = Path("artifacts/translation_quality_review")
    artifacts.mkdir(parents=True, exist_ok=True)
    existing: dict = {"samples": []}
    main_path = artifacts / report_name
    if main_path.exists():
        try:
            existing = json.loads(main_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {"samples": []}
    samples = [s for s in existing.get("samples", []) if s.get("sample_id") != report.get("sample_id")]
    samples.append(
        {
            "sample_id": report.get("sample_id"),
            "windows": report.get("windows", []),
            "summary": report.get("summary", {}),
            "repairs": report.get("repairs", []),
        }
    )
    existing["samples"] = samples
    existing["mode"] = report.get("mode", "span_guarded")
    main_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
