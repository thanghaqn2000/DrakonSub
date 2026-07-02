"""Evidence-based tiered span guard — conservative base + LLM classifier for candidate windows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import get_openai_model, llm_temperature
from .openai_chat import create_chat_completion
from .raw_llm_response_cache import raw_llm_complete
from .raw_cue_keyed_translate import _call_cue_keyed_batch, translate_single_cue_keyed
from .raw_span_alignment_guard import (
    RISK_FRAGMENT_SPAN,
    RISK_LOW_HIGH_NEIGHBOR,
    RISK_NEIGHBOR_BLEED,
    RISK_ORPHAN,
    RISK_REPEATED,
    RISK_SHORT_LONG,
    RISK_TOPIC_JUMP,
    SEMANTIC_BLEED_SIGNALS,
    WINDOW_SIZES,
    _cue_signals,
    _group_adjacent_high,
    _repair_rejected,
    _risk_profile,
    _severity_conservative,
    _window_looks_good,
    _word_count,
    analyze_span_alignment,
)
from .translation_topics import normalize_topic

SPAN_HIGH_CONFIDENCE = 0.75
REPORT_NAME = "span_guarded_tiered_report.json"


def _strip_json_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return content


def _candidate_signals_for_window(
    source_entries: List[dict],
    vi_entries: List[dict],
    cue_indices: List[int],
    cue_analysis: Dict[int, dict],
) -> Tuple[List[str], int]:
    """Return candidate signal names and count of criteria met."""
    signals: Set[str] = set()
    medium_count = 0
    for idx in cue_indices:
        ca = cue_analysis.get(idx, {})
        medium_count += 1 if ca.get("severity") == "MEDIUM" else 0
        signals.update(ca.get("risk_signatures") or [])

    criteria = 0
    if medium_count >= 2:
        criteria += 1
    if RISK_FRAGMENT_SPAN in signals:
        criteria += 1
    if RISK_REPEATED in signals:
        criteria += 1
    if RISK_LOW_HIGH_NEIGHBOR in signals:
        criteria += 1
    for idx in cue_indices:
        i = idx - 1
        en = source_entries[i].get("text", "").strip()
        vi = vi_entries[i].get("text", "").strip()
        en_wc = _word_count(en)
        vi_wc = _word_count(vi)
        if en_wc <= 8 and not en.rstrip().endswith((".", "?", "!")) and vi_wc >= en_wc + 6:
            criteria += 1
            signals.add("fragment_to_full_statement")
            break
    if RISK_TOPIC_JUMP in signals and RISK_NEIGHBOR_BLEED not in signals:
        criteria += 1
    return sorted(signals), criteria


def _good_window_skip(
    source_entries: List[dict],
    vi_entries: List[dict],
    entry_indexes: List[int],
    candidate_signals: List[str],
) -> Optional[str]:
    sig = set(candidate_signals)
    only_orphan_short = sig <= {
        RISK_ORPHAN,
        RISK_FRAGMENT_SPAN,
        RISK_SHORT_LONG,
        "fragment_to_full_statement",
    }
    if only_orphan_short and not (sig & SEMANTIC_BLEED_SIGNALS):
        return "orphan_fragment_without_bleed"
    if _window_looks_good(source_entries, vi_entries, entry_indexes):
        return "window_quality_precheck_passed"
    return None


def _classify_window_llm(
    client,
    model: str,
    source_entries: List[dict],
    vi_entries: List[dict],
    cue_indices: List[int],
) -> Dict[str, Any]:
    items = []
    for idx in cue_indices:
        i = idx - 1
        items.append(
            {
                "cue_index": idx,
                "source_en": source_entries[i].get("text", "").strip(),
                "vi": vi_entries[i].get("text", "").strip(),
            }
        )
    payload = {
        "task": "Detect whether Vietnamese translations preserve cue-to-cue meaning alignment.",
        "rules": [
            "Do not judge style.",
            "Only detect whether meaning from one source cue is assigned to the wrong Vietnamese cue.",
            "Return severity SPAN_HIGH only when there is clear cross-cue meaning drift.",
            "Return MEDIUM or LOW when uncertain.",
        ],
        "items": items,
    }
    system = (
        "You are a subtitle alignment classifier. Output strict JSON only. "
        "severity must be one of: LOW, MEDIUM, HIGH, SPAN_HIGH. "
        "confidence is 0.0-1.0. misaligned_cues is a list of cue_index integers."
    )
    user = json.dumps(payload, ensure_ascii=False, indent=2)
    content = raw_llm_complete(
        client,
        model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        llm_task_type="span_drift_classifier",
        batch_indices=indices,
        source_texts=[source_entries[i - 1].get("text", "") for i in indices],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = content or "{}"
    data = json.loads(_strip_json_fence(raw))
    return {
        "window_indices": data.get("window_indices") or cue_indices,
        "severity": str(data.get("severity", "LOW")).upper(),
        "misaligned_cues": [int(x) for x in (data.get("misaligned_cues") or [])],
        "reason": data.get("reason", ""),
        "confidence": float(data.get("confidence", 0.0)),
    }


def analyze_tiered(
    source_entries: List[dict],
    vi_entries: List[dict],
) -> Dict[str, Any]:
    base = analyze_span_alignment(source_entries, vi_entries, conservative=True)
    cue_analysis = {c["cue_index"]: c for c in base["cues"]}
    n = min(len(source_entries), len(vi_entries))
    windows: List[dict] = []
    seen: Set[Tuple[int, ...]] = set()

    for window_size in WINDOW_SIZES:
        for start in range(0, n - window_size + 1):
            indices = list(range(start + 1, start + window_size + 1))
            key = tuple(indices)
            if key in seen:
                continue
            if not any(source_entries[i - 1].get("text", "").strip() for i in indices):
                continue
            seen.add(key)

            candidate_signals, criteria = _candidate_signals_for_window(
                source_entries, vi_entries, indices, cue_analysis
            )
            entry_indexes = [i - 1 for i in indices]
            skip_reason = _good_window_skip(
                source_entries, vi_entries, entry_indexes, candidate_signals
            )

            window = {
                "cue_indices": indices,
                "window_size": window_size,
                "candidate_signals": candidate_signals,
                "candidate_criteria_met": criteria,
                "classifier_used": False,
                "classifier_result": None,
                "severity": "LOW",
                "decision": "skip_repair",
                "skip_reason": skip_reason or "not_candidate",
                "before": {
                    str(i): vi_entries[i - 1].get("text", "").strip() for i in indices
                },
            }

            if skip_reason:
                window["decision"] = "skip_repair"
                windows.append(window)
                continue

            if criteria < 2:
                window["skip_reason"] = "not_candidate"
                windows.append(window)
                continue

            window["decision"] = "classifier_pending"
            window["severity"] = "MEDIUM"
            windows.append(window)

    high_cues = [c for c in base["cues"] if c.get("severity") == "HIGH"]
    return {
        "mode": "span_guarded_tiered",
        "cue_count": n,
        "cues": base["cues"],
        "windows": windows,
        "high_cues": high_cues,
        "high_count": len(high_cues),
        "medium_count": sum(1 for c in base["cues"] if c.get("severity") == "MEDIUM"),
    }


def span_tiered_guard_and_repair(
    source_entries: List[dict],
    vi_entries: List[dict],
    *,
    topic: Optional[str] = None,
    debug_dir: Optional[str] = None,
    sample_id: Optional[str] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    import os

    from openai import OpenAI

    analysis = analyze_tiered(source_entries, vi_entries)
    working = [dict(e) for e in vi_entries]
    repairs: List[dict] = []
    classifier_calls = 0
    span_high = 0
    single_repairs = 0
    span_repairs = 0
    repair_rejected = 0
    skipped_good = 0
    candidate_windows = 0

    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else None
    model = get_openai_model()
    topic = normalize_topic(topic or os.environ.get("TRANSLATION_TOPIC"))

    # Cue-level HIGH (conservative semantic bleed)
    if client and analysis["high_cues"]:
        for item in analysis["high_cues"]:
            entry_idx = item["cue_index"] - 1
            cue_index = item["cue_index"]
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
                    conservative=True,
                )
                if rejected:
                    repair_rejected += 1
                    repairs.append(
                        {
                            "cue_index": cue_index,
                            "repair_type": "single_cue_high_rejected",
                            "skip_reason": reject_reason,
                            "before": before,
                            "after": after,
                        }
                    )
                    continue
                if after:
                    working[entry_idx] = {**working[entry_idx], "text": after}
                repairs.append(
                    {
                        "cue_index": cue_index,
                        "repair_type": "single_cue_high",
                        "repair_status": "accepted",
                        "before": before,
                        "after": after,
                    }
                )
            except Exception as exc:
                repairs.append(
                    {
                        "cue_index": cue_index,
                        "repair_type": "single_cue_high_failed",
                        "error": str(exc),
                    }
                )

    repaired_in_span: Set[int] = set()

    for window in analysis["windows"]:
        if window.get("decision") != "classifier_pending":
            if window.get("skip_reason") in (
                "orphan_fragment_without_bleed",
                "window_quality_precheck_passed",
            ):
                skipped_good += 1
            continue

        candidate_windows += 1
        indices = window["cue_indices"]
        entry_indexes = [i - 1 for i in indices]

        if not client:
            window["skip_reason"] = "no_api_key"
            continue

        try:
            classifier_calls += 1
            window["classifier_used"] = True
            result = _classify_window_llm(
                client, model, source_entries, working, indices
            )
            window["classifier_result"] = result
        except Exception as exc:
            window["skip_reason"] = "classifier_failed"
            window["classifier_result"] = {"error": str(exc)}
            repairs.append(
                {
                    "cue_indices": indices,
                    "repair_type": "classifier_failed",
                    "error": str(exc),
                }
            )
            continue

        severity = result.get("severity", "LOW")
        confidence = result.get("confidence", 0.0)
        if severity != "SPAN_HIGH" or confidence < SPAN_HIGH_CONFIDENCE:
            window["severity"] = severity
            window["decision"] = "skip_repair"
            window["skip_reason"] = "classifier_not_span_high"
            continue

        span_high += 1
        window["severity"] = "SPAN_HIGH"
        window["decision"] = "span_repair"
        misaligned = result.get("misaligned_cues") or indices
        repair_indexes = [c - 1 for c in misaligned if c in indices]
        if not repair_indexes:
            repair_indexes = entry_indexes

        if len(repair_indexes) == 1:
            entry_idx = repair_indexes[0]
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
                    "span_high_single",
                    conservative=True,
                )
                if rejected:
                    repair_rejected += 1
                    window["repair_status"] = "rejected"
                    window["skip_reason"] = reject_reason
                    continue
                if after:
                    working[entry_idx] = {**working[entry_idx], "text": after}
                repaired_in_span.add(entry_idx)
                window["repair_status"] = "accepted"
                repairs.append(
                    {
                        "cue_index": cue_index,
                        "repair_type": "span_high_single",
                        "repair_status": "accepted",
                        "classifier_confidence": confidence,
                        "before": before,
                        "after": after,
                    }
                )
            except Exception as exc:
                window["repair_status"] = "failed"
                repairs.append({"cue_index": cue_index, "error": str(exc)})
        else:
            try:
                parsed = _call_cue_keyed_batch(
                    client, model, source_entries, repair_indexes, "vi", topic
                )
                span_repairs += 1
                accepted_any = False
                for entry_idx in repair_indexes:
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
                        "span_high_batch",
                        conservative=True,
                    )
                    if rejected:
                        repair_rejected += 1
                        repairs.append(
                            {
                                "cue_index": cue_index,
                                "repair_type": "span_high_batch_rejected",
                                "skip_reason": reject_reason,
                            }
                        )
                        continue
                    if after != before:
                        working[entry_idx] = {**working[entry_idx], "text": after}
                    repaired_in_span.add(entry_idx)
                    accepted_any = True
                    repairs.append(
                        {
                            "cue_index": cue_index,
                            "repair_type": "span_high_batch",
                            "repair_status": "accepted",
                            "classifier_confidence": confidence,
                            "before": before,
                            "after": after,
                        }
                    )
                window["repair_status"] = "accepted" if accepted_any else "rejected"
                window["after"] = {
                    str(i): working[i - 1].get("text", "") for i in indices
                }
            except Exception as exc:
                window["repair_status"] = "failed"
                repairs.append(
                    {"cue_indices": indices, "repair_type": "span_high_batch_failed", "error": str(exc)}
                )

    report = {
        "mode": "span_guarded_tiered",
        "sample_id": sample_id,
        "windows": analysis["windows"],
        "repairs": repairs,
        "summary": {
            "candidate_windows": candidate_windows,
            "classifier_calls": classifier_calls,
            "span_high": span_high,
            "single_cue_repairs": single_repairs,
            "span_repairs": span_repairs,
            "repair_rejected": repair_rejected,
            "skipped_good_windows": skipped_good,
            "high_cue_count": analysis.get("high_count", 0),
        },
    }
    _write_tiered_report(report, debug_dir)
    return working, report


def _write_tiered_report(report: dict, debug_dir: Optional[str]) -> None:
    if debug_dir:
        out = Path(debug_dir) / REPORT_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts = Path("artifacts/translation_quality_review")
    artifacts.mkdir(parents=True, exist_ok=True)
    main_path = artifacts / REPORT_NAME
    existing: dict = {"samples": [], "mode": "span_guarded_tiered"}
    if main_path.exists():
        try:
            existing = json.loads(main_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    samples = [
        s
        for s in existing.get("samples", [])
        if s.get("sample_id") != report.get("sample_id")
    ]
    samples.append(
        {
            "sample_id": report.get("sample_id"),
            "windows": report.get("windows", []),
            "summary": report.get("summary", {}),
            "repairs": report.get("repairs", []),
        }
    )
    existing["samples"] = samples
    existing["mode"] = "span_guarded_tiered"
    main_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
