"""Post-raw overlap guard — detect adjacent VI duplication and repair conservatively."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .qa_calibration import _en_rhetorical_repeat
from .semantic_alignment_guard import _extract_concepts, _overlap_ratio

_RISK_OVERLAP = "post_raw_adjacent_overlap"
_RISK_RHETORIC = "rhetorical_repeat_protected"
_VI_OVERLAP_HIGH = 0.5
_EN_OVERLAP_LOW = 0.35
_SHORT_FRAGMENT_WORDS = 6

_ARTIFACT_PATH = Path("artifacts/translation_quality_review/post_raw_overlap_guard_v1_report.json")
_FRAGMENT_REPORT_PATH = Path(
    "artifacts/translation_quality_review/fragment_overlap_repair_v1_report.json"
)

_GENERIC_VI_RE = re.compile(
    r"(?i)^(điều này|chuyện đó|làm gì đó|như vậy|tóm lại|nói chung)"
)


def _en_similarity(en1: str, en2: str) -> float:
    c1 = _extract_concepts(en1, "en")
    c2 = _extract_concepts(en2, "en")
    if not c1 or not c2:
        return 0.0
    return len(c1 & c2) / min(len(c1), len(c2))


def _vi_similarity(vi1: str, vi2: str) -> float:
    c1 = _extract_concepts(vi1, "vi")
    c2 = _extract_concepts(vi2, "vi")
    if not c1 or not c2:
        return 0.0
    return len(c1 & c2) / min(len(c1), len(c2))


def _is_short_fragment(text: str) -> bool:
    words = re.findall(r"[\w']+", text.strip(), flags=re.UNICODE)
    return len(words) <= _SHORT_FRAGMENT_WORDS


def _alignment_to_source(vi: str, en: str) -> float:
    return _overlap_ratio(_extract_concepts(vi, "vi"), _extract_concepts(en, "en"))


def _tail_phrase_overlap(vi_a: str, vi_b: str, min_words: int = 3) -> bool:
    return _shared_tail_word_count(vi_a, vi_b, min_words) >= min_words


def _shared_tail_word_count(vi_a: str, vi_b: str, min_words: int = 2) -> int:
    a = re.sub(r"[^\w\s]", "", vi_a.lower()).split()
    b = re.sub(r"[^\w\s]", "", vi_b.lower()).split()
    if len(a) < min_words or len(b) < min_words:
        return 0
    for n in range(min(len(a), len(b)), min_words - 1, -1):
        if a[-n:] == b[-n:]:
            return n
    return 0


def _shared_tail_phrase(vi_a: str, vi_b: str) -> str:
    n = _shared_tail_word_count(vi_a, vi_b)
    if n < 2:
        return ""
    a = re.sub(r"[^\w\s]", "", vi_a.lower()).split()
    return " ".join(a[-n:])


def _is_en_fragment(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    words = re.findall(r"[\w']+", s, flags=re.UNICODE)
    if len(words) <= _SHORT_FRAGMENT_WORDS:
        return True
    return not s.rstrip().endswith((".", "?", "!"))


def _is_syntactically_connected(prev_en: str, target_en: str) -> bool:
    prev = prev_en.strip()
    target = target_en.strip()
    if not prev or not target:
        return False
    if not prev.rstrip().endswith((".", "?", "!")):
        return True
    if target[0].islower():
        return True
    if _is_en_fragment(target) and len(re.findall(r"[\w']+", target)) <= 5:
        return True
    return False


def _normalize_vi(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _needs_fragment_repair(prev_en: str, target_en: str, prev_vi: str, target_vi: str) -> bool:
    if not _is_en_fragment(target_en):
        return False
    if not _is_syntactically_connected(prev_en, target_en):
        return False
    if _normalize_vi(prev_vi) == _normalize_vi(target_vi):
        return False
    nv, tv = _normalize_vi(prev_vi), _normalize_vi(target_vi)
    if len(tv) > 12 and (tv in nv or nv in tv):
        return False
    return _shared_tail_word_count(prev_vi, target_vi) >= 2 or _tail_phrase_overlap(prev_vi, target_vi)


def _build_fragment_repair_prompt(
    *,
    cue_a: int,
    cue_b: int,
    prev_en: str,
    prev_vi: str,
    target_en: str,
    target_vi: str,
    next_en: str,
    next_vi: str,
    forbidden_tail: str,
) -> str:
    payload = {
        "task": (
            "Repair the Vietnamese translation of a subtitle fragment that overlaps "
            "too much with the previous cue."
        ),
        "rules": [
            "Translate only the target cue.",
            "The target cue may be a fragment continuing the previous cue.",
            "Do not repeat the previous cue's Vietnamese tail.",
            "Preserve the target cue's actual meaning.",
            "Keep the Vietnamese short and fragment-like if the English is fragment-like.",
            "Do not add new information.",
            f"Do not reuse this forbidden tail phrase: {forbidden_tail!r}" if forbidden_tail else (
                "Avoid repeating words from the previous cue's ending."
            ),
            "Return strict JSON with cue_index and vi.",
        ],
        "previous": {"cue_index": cue_a, "source_en": prev_en, "vi": prev_vi},
        "target": {
            "cue_index": cue_b,
            "source_en": target_en,
            "current_vi": target_vi,
        },
        "next": {"cue_index": cue_b + 1, "source_en": next_en, "vi": next_vi},
    }
    return (
        "Fragment-aware overlap repair.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f'Return JSON: {{"cue_index": {cue_b}, "vi": "..."}}'
    )


def _parse_fragment_repair_response(content: str, cue_index: int) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if isinstance(data, dict) and "vi" in data:
        idx = data.get("cue_index", cue_index)
        if idx != cue_index:
            raise ValueError(f"Unexpected cue_index {idx}")
        return str(data["vi"]).strip()
    raise ValueError("Fragment repair response missing vi")


def _repair_fragment_overlap(
    client,
    model: str,
    *,
    cue_a: int,
    cue_b: int,
    source_entries: List[dict],
    working: List[dict],
    repair_idx: int,
    neighbor_idx: int,
    topic: str,
) -> str:
    from .openai_translate import _build_openai_raw_system_prompt
    from .raw_llm_response_cache import raw_llm_complete

    prev_en = source_entries[neighbor_idx].get("text", "").strip()
    target_en = source_entries[repair_idx].get("text", "").strip()
    prev_vi = working[neighbor_idx].get("text", "").strip()
    target_vi = working[repair_idx].get("text", "").strip()
    next_en = (
        source_entries[repair_idx + 1].get("text", "").strip()
        if repair_idx + 1 < len(source_entries)
        else ""
    )
    next_vi = (
        working[repair_idx + 1].get("text", "").strip()
        if repair_idx + 1 < len(working)
        else ""
    )
    forbidden = _shared_tail_phrase(prev_vi, target_vi)
    user = _build_fragment_repair_prompt(
        cue_a=cue_a,
        cue_b=cue_b,
        prev_en=prev_en,
        prev_vi=prev_vi,
        target_en=target_en,
        target_vi=target_vi,
        next_en=next_en,
        next_vi=next_vi,
        forbidden_tail=forbidden,
    )
    system = _build_openai_raw_system_prompt(topic) + (
        "\n\nFRAGMENT OVERLAP REPAIR MODE:\n"
        "- Output one short Vietnamese fragment for the target cue only.\n"
        "- Never repeat the previous cue's ending phrase.\n"
        "- JSON only."
    )
    content = raw_llm_complete(
        client,
        model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        llm_task_type="post_raw_fragment_overlap_repair",
        batch_indices=[cue_b],
        source_texts=[target_en],
        repair=True,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return _parse_fragment_repair_response(content, cue_b)


def _rule_connective_fragment_repair(prev_vi: str, target_vi: str, target_en: str) -> Optional[str]:
    """Deterministic fallback: minimal fragment without shared tail words."""
    tail = _shared_tail_phrase(prev_vi, target_vi)
    if len(tail.split()) < 3:
        return None
    tail_words = set(tail.split())
    target_words = re.sub(r"[^\w\s]", "", target_vi.lower()).split()
    kept = [w for w in target_words if w not in tail_words]
    punct = "?" if target_en.strip().endswith("?") else ""
    if kept and len(kept) >= 2:
        return " ".join(kept) + punct
    en_l = target_en.lower().strip()
    if re.search(r"\b(do with it|with it)\b", en_l):
        subj = "tôi" if re.match(r"^\s*i\b", en_l) else "mình"
        return f"{subj} làm gì với nó{punct or '?'}"
    words = tail.split()
    drop_aux = {"sẽ", "đã", "đang", "vẫn", "cũng"}
    if words[0] in drop_aux and len(words) > 3:
        core = " ".join(words[1:])
    else:
        core = tail
    return f"thì {core}{punct}"


def detect_adjacent_overlap_flags(
    source_entries: List[dict],
    vi_entries: List[dict],
) -> List[dict]:
    """Flag adjacent cue pairs with high VI overlap but low EN overlap."""
    flags: List[dict] = []
    source_texts = [e.get("text", "") for e in source_entries]
    vi_texts = [e.get("text", "") for e in vi_entries]
    n = min(len(source_entries), len(vi_entries))

    for i in range(1, n):
        cue_a, cue_b = i, i + 1
        en_a = source_texts[i - 1].strip()
        en_b = source_texts[i].strip()
        vi_a = vi_texts[i - 1].strip()
        vi_b = vi_texts[i].strip()
        if not vi_a or not vi_b:
            continue

        en_sim = _en_similarity(en_a, en_b)
        vi_sim = _vi_similarity(vi_a, vi_b)
        tail_overlap = _tail_phrase_overlap(vi_a, vi_b)

        if _en_rhetorical_repeat(source_texts, i):
            flags.append(
                {
                    "cue_indices": [cue_a, cue_b],
                    "risk_signature": _RISK_RHETORIC,
                    "severity": "PROTECTED",
                    "en_similarity": round(en_sim, 3),
                    "vi_similarity": round(vi_sim, 3),
                    "rhetorical_repeat_protected": True,
                    "decision": "skip",
                }
            )
            continue

        if (_is_short_fragment(en_a) or _is_short_fragment(en_b)) and vi_sim < 0.65 and not tail_overlap:
            continue

        if vi_sim >= _VI_OVERLAP_HIGH and en_sim < _EN_OVERLAP_LOW:
            severity = "HIGH" if vi_sim >= 0.55 or tail_overlap else "MEDIUM"
        elif tail_overlap and en_sim < _EN_OVERLAP_LOW and vi_sim >= 0.35:
            severity = "HIGH"
        else:
            continue

        flags.append(
            {
                "cue_indices": [cue_a, cue_b],
                "risk_signature": _RISK_OVERLAP,
                "severity": severity,
                "en_similarity": round(en_sim, 3),
                "vi_similarity": round(vi_sim, 3),
                "tail_phrase_overlap": tail_overlap,
                "rhetorical_repeat_protected": False,
                "decision": "repair_later_cue",
            }
        )
    return flags


def _accept_fragment_repair(
    *,
    before_vi: str,
    after_vi: str,
    neighbor_vi: str,
    next_vi: str,
    target_en: str,
) -> Tuple[bool, str]:
    if not after_vi.strip():
        return False, "repair_rejected_empty"
    if _GENERIC_VI_RE.match(after_vi.strip()):
        return False, "repair_rejected_generic"
    tail_before = _shared_tail_word_count(neighbor_vi, before_vi)
    tail_after = _shared_tail_word_count(neighbor_vi, after_vi)
    if tail_after >= tail_before:
        return False, "repair_rejected_tail_overlap_not_decreased"
    if next_vi and _vi_similarity(after_vi, next_vi) >= 0.55:
        return False, "repair_rejected_duplicates_next"
    old_align = _alignment_to_source(before_vi, target_en)
    new_align = _alignment_to_source(after_vi, target_en)
    if new_align < old_align - 0.15:
        return False, "repair_rejected_semantic_worse"
    if len(after_vi) > max(len(before_vi), 1) * 1.4:
        return False, "repair_rejected_not_concise"
    return True, "accepted"


def _accept_repair(
    *,
    before_vi: str,
    after_vi: str,
    neighbor_vi: str,
    target_en: str,
) -> Tuple[bool, str]:
    if not after_vi.strip():
        return False, "repair_rejected_empty"
    old_overlap = _vi_similarity(before_vi, neighbor_vi)
    new_overlap = _vi_similarity(after_vi, neighbor_vi)
    if new_overlap >= old_overlap - 0.02:
        return False, "repair_rejected_overlap_not_decreased"
    old_align = _alignment_to_source(before_vi, target_en)
    new_align = _alignment_to_source(after_vi, target_en)
    if new_align < old_align - 0.12:
        return False, "repair_rejected_semantic_worse"
    if len(after_vi) > max(len(before_vi), 1) * 1.6:
        return False, "repair_rejected_readability"
    return True, "accepted"


def repair_overlap_flags(
    source_entries: List[dict],
    vi_entries: List[dict],
    flags: List[dict],
    *,
    topic: Optional[str] = None,
) -> Tuple[List[dict], List[dict]]:
    """Re-translate later cue for HIGH overlap flags."""
    from openai import OpenAI

    from .config import get_openai_model
    from .raw_cue_keyed_translate import translate_single_cue_keyed
    from .translation_topics import normalize_topic

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return vi_entries, []

    client = OpenAI(api_key=api_key)
    model = get_openai_model()
    topic = normalize_topic(topic or os.environ.get("TRANSLATION_TOPIC"))
    working = [dict(e) for e in vi_entries]
    repairs: List[dict] = []
    repaired_indices: set[int] = set()

    for flag in flags:
        if flag.get("risk_signature") != _RISK_OVERLAP:
            continue
        if flag.get("severity") != "HIGH":
            continue
        cue_a, cue_b = flag["cue_indices"]
        repair_idx = cue_b - 1
        if repair_idx in repaired_indices:
            continue
        neighbor_idx = cue_a - 1
        if repair_idx < 0 or repair_idx >= len(working):
            continue

        before = working[repair_idx].get("text", "").strip()
        neighbor_vi = working[neighbor_idx].get("text", "").strip()
        next_vi = (
            working[repair_idx + 1].get("text", "").strip()
            if repair_idx + 1 < len(working)
            else ""
        )
        target_en = source_entries[repair_idx].get("text", "").strip()
        prev_en = source_entries[neighbor_idx].get("text", "").strip()
        flag["before"] = {"cue_index": cue_b, "vi": before}
        fragment_aware = _needs_fragment_repair(prev_en, target_en, neighbor_vi, before)
        flag["fragment_aware_repair"] = fragment_aware
        if fragment_aware:
            flag["previous_vi_tail"] = _shared_tail_phrase(neighbor_vi, before)
            flag["tail_overlap_before"] = _shared_tail_word_count(neighbor_vi, before)
        else:
            flag["repair_status"] = "repair_skipped_non_fragment"
            flag["reject_reason"] = "non_fragment_high_overlap_preserved"
            repairs.append(dict(flag))
            continue

        try:
            after = ""
            if fragment_aware:
                rule_after = _rule_connective_fragment_repair(
                    neighbor_vi, before, target_en
                )
                if rule_after:
                    ok_rule, status_rule = _accept_fragment_repair(
                        before_vi=before,
                        after_vi=rule_after,
                        neighbor_vi=neighbor_vi,
                        next_vi=next_vi,
                        target_en=target_en,
                    )
                    if ok_rule:
                        after = rule_after
                        flag["repair_method"] = "rule_connective"
                if not after:
                    flag["repair_status"] = "repair_skipped_no_safe_rule"
                    flag["reject_reason"] = "fragment_llm_disabled_conservative"
                    repairs.append(dict(flag))
                    continue
            else:
                raise RuntimeError("unreachable")
        except Exception as exc:
            after = ""
            if fragment_aware:
                after = _rule_connective_fragment_repair(
                    neighbor_vi, before, target_en
                ) or ""
            if not after:
                flag["repair_status"] = "repair_rejected_error"
                flag["reject_reason"] = str(exc)
                flag["error"] = str(exc)
                repairs.append(dict(flag))
                continue

        if fragment_aware and not flag.get("repair_method"):
            ok, status = _accept_fragment_repair(
                before_vi=before,
                after_vi=after,
                neighbor_vi=neighbor_vi,
                next_vi=next_vi,
                target_en=target_en,
            )
            if not ok and status != "accepted":
                rule_after = _rule_connective_fragment_repair(
                    neighbor_vi, before, target_en
                )
                if rule_after:
                    ok2, status2 = _accept_fragment_repair(
                        before_vi=before,
                        after_vi=rule_after,
                        neighbor_vi=neighbor_vi,
                        next_vi=next_vi,
                        target_en=target_en,
                    )
                    if ok2:
                        after = rule_after
                        ok, status = ok2, status2
                        flag["repair_method"] = "rule_connective_fallback"
            flag["tail_overlap_after"] = _shared_tail_word_count(neighbor_vi, after)
            flag["before_target_vi"] = before
            flag["after_target_vi"] = after
        elif fragment_aware:
            ok, status = True, "accepted"
            flag["tail_overlap_after"] = _shared_tail_word_count(neighbor_vi, after)
            flag["before_target_vi"] = before
            flag["after_target_vi"] = after
        else:
            ok, status = _accept_repair(
                before_vi=before,
                after_vi=after,
                neighbor_vi=neighbor_vi,
                target_en=target_en,
            )
        if not ok and status != "accepted":
            flag["reject_reason"] = status
        flag["after"] = {"cue_index": cue_b, "vi": after}
        flag["repair_status"] = status
        if ok:
            working[repair_idx] = {**working[repair_idx], "text": after}
            repaired_indices.add(repair_idx)
        repairs.append(dict(flag))

    return working, repairs


def _summarize_flags(flags: List[dict]) -> dict:
    high = [f for f in flags if f.get("risk_signature") == _RISK_OVERLAP and f.get("severity") == "HIGH"]
    repairs_attempted = sum(1 for f in flags if f.get("decision") == "repair_later_cue" and f.get("severity") == "HIGH")
    repairs_accepted = sum(1 for f in flags if f.get("repair_status") == "accepted")
    repairs_rejected = sum(
        1
        for f in flags
        if str(f.get("repair_status", "")).startswith("repair_rejected")
    )
    protected = sum(1 for f in flags if f.get("risk_signature") == _RISK_RHETORIC)
    return {
        "high_flags": len(high),
        "repairs_attempted": repairs_attempted,
        "repairs_accepted": repairs_accepted,
        "repairs_rejected": repairs_rejected,
        "protected_rhetorical_repeats": protected,
    }


def _save_aggregate_report(sample_report: dict) -> None:
    _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {"samples": []}
    if _ARTIFACT_PATH.exists():
        try:
            existing = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {"samples": []}
    by_id = {s.get("sample_id"): s for s in existing.get("samples", []) if s.get("sample_id")}
    sid = sample_report.get("sample_id")
    if sid:
        by_id[sid] = sample_report
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": list(by_id.values()),
    }
    _ARTIFACT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_fragment_report(sample_report: dict) -> None:
    _FRAGMENT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {"samples": []}
    if _FRAGMENT_REPORT_PATH.exists():
        try:
            existing = json.loads(_FRAGMENT_REPORT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {"samples": []}
    by_id = {s.get("sample_id"): s for s in existing.get("samples", []) if s.get("sample_id")}
    sid = sample_report.get("sample_id")
    if sid:
        by_id[sid] = sample_report
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": list(by_id.values()),
    }
    _FRAGMENT_REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def guard_post_raw_overlap(
    source_entries: List[dict],
    vi_entries: List[dict],
    *,
    topic: Optional[str] = None,
    debug_dir: Optional[str] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Detect adjacent overlap, repair HIGH severity, return updated entries + report."""
    flags = detect_adjacent_overlap_flags(source_entries, vi_entries)
    repaired_entries, _repair_log = repair_overlap_flags(
        source_entries,
        vi_entries,
        flags,
        topic=topic,
    )

    n = min(len(source_entries), len(vi_entries))
    sample_id = os.environ.get("BENCHMARK_SAMPLE_ID", "pipeline")
    report: Dict[str, Any] = {
        "sample_id": sample_id,
        "checked_pairs": max(0, n - 1),
        "flags": flags,
        "summary": _summarize_flags(flags),
    }

    if debug_dir:
        out = Path(debug_dir) / "post_raw_overlap_guard_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if sample_id and sample_id != "pipeline":
        _save_aggregate_report(report)
        fragment_flags = [
            {
                "cue_indices": f.get("cue_indices"),
                "risk_signature": f.get("risk_signature"),
                "fragment_aware_repair": f.get("fragment_aware_repair", False),
                "previous_vi_tail": f.get("previous_vi_tail", ""),
                "before_target_vi": f.get("before_target_vi", f.get("before", {}).get("vi", "")),
                "after_target_vi": f.get("after_target_vi", f.get("after", {}).get("vi", "")),
                "tail_overlap_before": f.get("tail_overlap_before"),
                "tail_overlap_after": f.get("tail_overlap_after"),
                "repair_status": f.get("repair_status"),
                "reject_reason": f.get("reject_reason"),
            }
            for f in flags
            if f.get("fragment_aware_repair") or f.get("tail_overlap_before") is not None
        ]
        if fragment_flags:
            _save_fragment_report(
                {
                    "sample_id": sample_id,
                    "flags": fragment_flags,
                    "summary": _summarize_flags(flags),
                }
            )

    return repaired_entries, report
