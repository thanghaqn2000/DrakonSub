"""Unit-level rewrite + deterministic cue redistribution for risky-unit repair."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .semantic_alignment_guard import (
    _unit_alignment_ok,
    analyze_semantic_alignment,
    detect_repeated_meaning,
    validate_repair_contract,
)
from .cue_shift_detector import count_shifts_in_cues
from .subtitle_timing_optimizer import _parse_ts
from .translation_quality_analyzer import _is_standalone_weak_fragment
from .vi_compression import _cps
from .vi_cue_fluency_guard import analyze_unit_fluency

_DEFAULT_TARGET_CPS = 20.0
_DEFAULT_MAX_CPS = 24.0
_RELAXED_TARGET_CPS = 32.0
_RELAXED_MAX_CPS = 42.0
_MIN_CUE_CHARS = 4
_MIN_MEANING_COVERAGE = 0.85

_WEAK_ONLY_RE = re.compile(
    r"(?i)^(?:về|với|của)\s+(?:điều đó|nó|điều này|cái đó|chuyện đó|đó)\.?$"
    r"|^(?:mà|rằng|và|nhưng|thì|nên|vì|nếu|hoặc)\.?$"
)

_DANGLING_END_RE = re.compile(
    r"(?i)\s+(?:là|và|nhưng|mà|vì|nếu|thì|hoặc|rằng|cho|với|về|của)$"
)

_CONNECTIVE_ONLY_RE = re.compile(
    r"(?i)^(?:và|nhưng|mà|rằng|vì|nên|thì|hoặc|để)\.?$"
)
_CONTINUATION_LEAD_RE = re.compile(r"(?i)^(vào|về|cho|của|mà|rằng)\s+")

_SEMANTIC_REPAIR_ERROR_TYPES = frozenset(
    {
        "semantic_alignment_error",
        "semantic_drift_error",
        "missing_or_empty_cue",
        "repeated_meaning_error",
        "cue_flow_error",
    }
)

_PHRASE_SPLIT_RE = re.compile(
    r"(?<=[.,;:!?])\s+"
    r"|(?<=\s)(?:và|nhưng|mà|nên|vì|nếu|thì|hoặc|hoặc là|vì vậy|tuy nhiên|để|cho)\s+",
    re.IGNORECASE,
)
_PHRASE_BOUNDARY_RE = re.compile(r"(?i)(?<=[,;])\s+|(?:\s)(?:nhưng|mà|và)\s+")


@dataclass
class RedistributionConfig:
    target_cps: float = _DEFAULT_TARGET_CPS
    max_cps: float = _DEFAULT_MAX_CPS
    min_cue_chars: int = _MIN_CUE_CHARS
    allow_cps_truncate: bool = False
    relaxed: bool = False
    prefer_fluency: bool = False


def relaxed_redistribution_config() -> RedistributionConfig:
    return RedistributionConfig(
        target_cps=_RELAXED_TARGET_CPS,
        max_cps=_RELAXED_MAX_CPS,
        allow_cps_truncate=False,
        relaxed=True,
    )


def fluency_coherent_config() -> RedistributionConfig:
    return RedistributionConfig(
        target_cps=_RELAXED_TARGET_CPS,
        max_cps=_RELAXED_MAX_CPS,
        prefer_fluency=True,
        relaxed=True,
    )


def _cue_duration(entry: dict) -> float:
    return max(0.01, _parse_ts(entry["end_str"]) - _parse_ts(entry["start_str"]))


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = re.sub(r"([.?!,;:])(?=\S)", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_phrases(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []
    parts = _PHRASE_SPLIT_RE.split(text)
    phrases = [p.strip() for p in parts if p.strip()]
    return phrases if phrases else [text]


def _token_words(text: str) -> Set[str]:
    return {w for w in re.findall(r"[\w']+", text.lower()) if len(w) >= 2}


def _meaning_coverage(unit_translation: str, redistributed: Dict[int, str], cue_indexes: List[int]) -> float:
    unit_w = _token_words(unit_translation)
    if not unit_w:
        return 1.0
    combined = " ".join(redistributed.get(i, "") for i in cue_indexes)
    redist_w = _token_words(combined)
    return len(unit_w & redist_w) / len(unit_w)


def _is_weak_only_fragment(text: str) -> bool:
    text = text.strip()
    if not text:
        return True
    if _WEAK_ONLY_RE.match(text):
        return True
    return _is_standalone_weak_fragment(text)


def _redistribute_proportional_words(
    text: str,
    cue_indexes: List[int],
    durations: List[float],
) -> Dict[int, str]:
    words = text.split()
    if not words:
        return {i: "" for i in cue_indexes}
    total = len(words)
    total_dur = sum(durations) or 1.0
    result: Dict[int, str] = {}
    pos = 0
    for i, idx in enumerate(cue_indexes):
        if i == len(cue_indexes) - 1:
            result[idx] = " ".join(words[pos:])
            break
        remaining_cues = len(cue_indexes) - i - 1
        share = max(1, round(total * durations[i] / total_dur))
        share = min(share, total - pos - remaining_cues)
        result[idx] = " ".join(words[pos : pos + share])
        pos += share
    return result


def _char_budgets(durations: List[float], total_chars: int, cfg: RedistributionConfig) -> List[int]:
    caps = [max(cfg.min_cue_chars, int(d * cfg.target_cps)) for d in durations]
    total_cap = sum(caps) or 1
    budgets = [max(cfg.min_cue_chars, round(total_chars * c / total_cap)) for c in caps]
    diff = total_chars - sum(budgets)
    i = 0
    while diff != 0 and budgets:
        idx = i % len(budgets)
        if diff > 0:
            budgets[idx] += 1
            diff -= 1
        elif budgets[idx] > cfg.min_cue_chars:
            budgets[idx] -= 1
            diff += 1
        i += 1
        if i > total_chars * 3:
            break
    return budgets


def _merge_weak_fragments(
    cue_indexes: List[int],
    texts: Dict[int, str],
    source_entries: List[dict],
    *,
    allow_donor_steal: bool = True,
) -> Dict[int, str]:
    result = dict(texts)
    for pos, idx in enumerate(cue_indexes):
        t = result.get(idx, "").strip()
        if not _is_weak_only_fragment(t):
            continue
        if pos > 0:
            prev = cue_indexes[pos - 1]
            result[prev] = _normalize_text(f"{result.get(prev, '')} {t}")
            result[idx] = ""
        elif pos + 1 < len(cue_indexes):
            nxt = cue_indexes[pos + 1]
            result[nxt] = _normalize_text(f"{t} {result.get(nxt, '')}")
            result[idx] = ""
    for idx in cue_indexes:
        src = source_entries[idx - 1].get("text", "").strip() if idx <= len(source_entries) else ""
        if src and not result.get(idx, "").strip():
            if not allow_donor_steal:
                continue
            neighbors = [c for c in cue_indexes if c != idx and result.get(c, "").strip()]
            if neighbors:
                donor = max(neighbors, key=lambda c: len(result.get(c, "")))
                donor_text = result[donor]
                words = donor_text.split()
                if len(words) > 2:
                    take = max(1, len(words) // 3)
                    result[idx] = " ".join(words[:take])
                    result[donor] = " ".join(words[take:])
    return result


def _dedupe_adjacent_overlap(cue_indexes: List[int], texts: Dict[int, str]) -> Dict[int, str]:
    result = dict(texts)
    for i in range(len(cue_indexes) - 1):
        a, b = cue_indexes[i], cue_indexes[i + 1]
        ta, tb = result.get(a, "").strip(), result.get(b, "").strip()
        if not ta or not tb:
            continue
        ta_l, tb_l = ta.lower().rstrip("?"), tb.lower().rstrip("?")
        if ta_l == tb_l or (len(ta_l) > 8 and ta_l in tb_l):
            result[b] = tb[len(ta) :].strip() if tb.lower().startswith(ta_l) else tb
            if not result[b]:
                result[b] = tb
        elif len(tb_l) > 8 and tb_l in ta_l:
            result[a] = ta[: -len(tb)].strip() if ta.lower().endswith(tb_l) else ta
    return result


def _fix_dangling_from_unit_text(
    unit_translation: str,
    cue_indexes: List[int],
    result: Dict[int, str],
) -> Dict[int, str]:
    combined = _normalize_text(unit_translation)
    out = dict(result)
    for idx in cue_indexes:
        t = out.get(idx, "").strip()
        if not t or not _has_dangling_end(t):
            continue
        pos = combined.lower().find(t.lower())
        if pos < 0:
            continue
        rest = combined[pos + len(t) :].strip()
        m = re.match(r"^[^\s,.;!?]+(?:\s+[^\s,.;!?]+)*[,.;!?]?", rest)
        if m and m.group().strip():
            out[idx] = _normalize_text(t + m.group())
    return out


def _polish_connective_redistribution(
    unit_translation: str,
    cue_indexes: List[int],
    result: Dict[int, str],
    source_entries: List[dict],
    original_vi_entries: Optional[List[dict]] = None,
) -> Dict[int, str]:
    """Merge orphan connectives; restore emptied cues from original VI or proportional split."""
    out = dict(result)

    for i, idx in enumerate(cue_indexes[:-1]):
        t = out.get(idx, "").strip()
        if _CONNECTIVE_ONLY_RE.match(t):
            nxt = cue_indexes[i + 1]
            out[nxt] = _normalize_text(f"{t} {out.get(nxt, '')}")
            out[idx] = ""

    for i, idx in enumerate(cue_indexes[:-1]):
        t = out.get(idx, "").strip()
        if t and _CONTINUATION_LEAD_RE.match(t):
            nxt = cue_indexes[i + 1]
            out[nxt] = _normalize_text(f"{out.get(nxt, '')} {t}")
            out[idx] = ""

    empty = [idx for idx in cue_indexes if not out.get(idx, "").strip()]
    for idx in empty:
        if original_vi_entries and 0 < idx <= len(original_vi_entries):
            fallback = (original_vi_entries[idx - 1].get("text") or "").strip()
            if fallback:
                out[idx] = fallback
                continue
        durations = [
            _cue_duration(source_entries[i - 1]) if 0 < i <= len(source_entries) else 1.0
            for i in cue_indexes
        ]
        proportional = _redistribute_proportional_words(
            unit_translation, cue_indexes, durations
        )
        out[idx] = proportional.get(idx, "")

    return out


def _split_clauses(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.?!])\s+", text)
    clauses = [p.strip() for p in parts if p.strip()]
    if len(clauses) <= 1:
        return _split_phrases(text)
    return clauses


def _expand_clauses_to_count(clauses: List[str], count: int) -> List[str]:
    parts = list(clauses)
    while len(parts) < count:
        splittable = [
            (len(chunk), i)
            for i, chunk in enumerate(parts)
            if _PHRASE_BOUNDARY_RE.search(chunk)
        ]
        if not splittable:
            break
        _, longest_i = max(splittable)
        chunk = parts[longest_i]
        split = _PHRASE_BOUNDARY_RE.split(chunk, maxsplit=1)
        if len(split) < 2 or not split[1].strip():
            break
        parts[longest_i : longest_i + 1] = [split[0].strip(), split[1].strip()]
    return parts


def _redistribute_fluency_coherent(
    text: str,
    cue_indexes: List[int],
    durations: List[float],
    source_entries: List[dict],
) -> Dict[int, str]:
    """Assign whole clauses/phrases to cues; allow uneven lengths; never break mid-clause."""
    clauses = _expand_clauses_to_count(_split_clauses(text), len(cue_indexes))
    if not clauses:
        return {cue_indexes[0]: text} if cue_indexes else {}

    if len(clauses) >= len(cue_indexes):
        result = {
            cue_indexes[i]: _normalize_text(clauses[i])
            for i in range(len(cue_indexes))
        }
    else:
        result = {}
        base, rem = divmod(len(clauses), len(cue_indexes))
        pos = 0
        for i, idx in enumerate(cue_indexes):
            take = base + (1 if i < rem else 0)
            chunk = clauses[pos : pos + take]
            result[idx] = _normalize_text(" ".join(chunk))
            pos += take

    if sum(1 for idx in cue_indexes if result.get(idx, "").strip()) < len(cue_indexes):
        result = _redistribute_proportional_words(text, cue_indexes, durations)

    result = _fix_dangling_from_unit_text(text, cue_indexes, result)
    result = _merge_weak_fragments(cue_indexes, result, source_entries, allow_donor_steal=False)
    return result


def redistribute_unit_translation(
    unit_translation: str,
    cue_indexes: List[int],
    source_entries: List[dict],
    *,
    cfg: Optional[RedistributionConfig] = None,
) -> Dict[int, str]:
    """
    Split unit_translation across cue_indexes using duration-weighted budgets
    and Vietnamese phrase boundaries.
    """
    cfg = cfg or RedistributionConfig()
    text = _normalize_text(unit_translation)
    if not text or not cue_indexes:
        return {}

    if len(cue_indexes) == 1:
        return {cue_indexes[0]: text}

    durations = [
        _cue_duration(source_entries[i - 1]) if 0 < i <= len(source_entries) else 1.0
        for i in cue_indexes
    ]

    if cfg.prefer_fluency:
        result = _redistribute_fluency_coherent(text, cue_indexes, durations, source_entries)
        coverage = _meaning_coverage(text, result, cue_indexes)
        if coverage >= _MIN_MEANING_COVERAGE:
            return result

    phrases = _split_phrases(text)
    budgets = _char_budgets(durations, len(text), cfg)

    cue_texts: List[List[str]] = [[] for _ in cue_indexes]
    cue_idx = 0
    current_len = 0

    for phrase in phrases:
        if cue_idx == len(cue_indexes) - 1:
            cue_texts[cue_idx].append(phrase)
            continue
        plen = len(phrase)
        if current_len > 0 and current_len + plen + 1 > budgets[cue_idx]:
            cue_idx += 1
            current_len = 0
        cue_texts[cue_idx].append(phrase)
        current_len += plen + (1 if current_len else 0)

    result: Dict[int, str] = {}
    for i, idx in enumerate(cue_indexes):
        joined = _normalize_text(" ".join(cue_texts[i]))
        result[idx] = joined

    if phrases and not any(result.values()):
        per = max(1, len(text) // len(cue_indexes))
        for i, idx in enumerate(cue_indexes):
            start = i * per
            end = len(text) if i == len(cue_indexes) - 1 else (i + 1) * per
            result[idx] = text[start:end].strip()

    result = _merge_weak_fragments(cue_indexes, result, source_entries)
    result = _dedupe_adjacent_overlap(cue_indexes, result)
    result = _fix_dangling_from_unit_text(text, cue_indexes, result)

    coverage = _meaning_coverage(text, result, cue_indexes)
    if coverage < _MIN_MEANING_COVERAGE:
        result = _redistribute_proportional_words(text, cue_indexes, durations)
        result = _merge_weak_fragments(cue_indexes, result, source_entries)

    return result


def redistribution_to_repair_unit(
    unit_id: int,
    cue_indexes: List[int],
    redistributed: Dict[int, str],
) -> dict:
    cues = [
        {"cue_index": idx, "text": redistributed.get(idx, "")}
        for idx in cue_indexes
    ]
    return {"unit_id": unit_id, "cues": cues}


def parse_unit_rewrite_response(content: str) -> List[dict]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    data = json.loads(content)
    units = data.get("units") or []
    parsed = []
    for u in units:
        uid = u.get("unit_id")
        translation = _normalize_text(str(u.get("unit_translation", "")))
        if uid is None or not translation:
            continue
        parsed.append(
            {
                "unit_id": uid,
                "unit_translation": translation,
                "notes": str(u.get("notes", "")).strip(),
            }
        )
    return parsed


def _unit_quality_score(
    source_entries: List[dict],
    vi_entries: List[dict],
    cue_indexes: List[int],
    meaning_units: Optional[List[dict]],
    video_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    source_texts = [e.get("text", "") for e in source_entries]
    vi_texts = [e.get("text", "") for e in vi_entries]

    alignment = analyze_semantic_alignment(
        source_entries, vi_entries, meaning_units, video_context
    )
    severe = 0
    warnings = 0
    flow = 0
    for issue in alignment.get("cue_issues", []):
        if issue["cue_index"] in cue_indexes:
            severe += len(issue.get("errors", []))
    for issue in alignment.get("cue_warnings", []):
        if issue["cue_index"] in cue_indexes:
            warnings += len(issue.get("errors", []))

    repeated = detect_repeated_meaning(source_texts, vi_texts, meaning_units)
    repeated_count = sum(1 for c in cue_indexes if c in repeated)

    for idx in cue_indexes:
        if 0 < idx <= len(vi_texts) and _is_weak_only_fragment(vi_texts[idx - 1]):
            flow += 1

    unit_ok, unit_score = _unit_alignment_ok(
        cue_indexes, source_texts, vi_texts, video_context
    )

    cps_over = 0
    for idx in cue_indexes:
        if 0 < idx <= len(vi_entries):
            t = vi_texts[idx - 1].strip()
            if t and _cps(t, _cue_duration(vi_entries[idx - 1])) > _DEFAULT_MAX_CPS:
                cps_over += 1

    return {
        "severe_alignment": severe,
        "warnings": warnings,
        "repeated_meaning": repeated_count,
        "weak_fragments": flow,
        "cps_over": cps_over,
        "unit_alignment_ok": unit_ok,
        "unit_alignment_score": unit_score,
        "semantic_total": severe * 3 + repeated_count * 2 + flow * 2,
        "total": severe * 3 + repeated_count * 2 + flow * 2,
    }


def _has_dangling_end(text: str) -> bool:
    return bool(_DANGLING_END_RE.search(text.strip()))


def _semantic_improved(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    if after["severe_alignment"] < before["severe_alignment"]:
        return True
    if after["unit_alignment_ok"] and not before["unit_alignment_ok"]:
        return True
    if after["unit_alignment_score"] > before["unit_alignment_score"] + 0.5:
        return True
    if before["severe_alignment"] > 0 and after["severe_alignment"] == 0:
        return True
    return False


def evaluate_redistributed_unit(
    source_entries: List[dict],
    vi_entries: List[dict],
    unit: dict,
    redistributed: Dict[int, str],
    meaning_units: Optional[List[dict]],
    video_context: Optional[Dict[str, Any]],
    *,
    unit_translation: str = "",
) -> Dict[str, Any]:
    """Validate redistributed cues; compare quality vs original unit."""
    uid = unit["unit_id"]
    cue_indexes = unit.get("cue_indexes") or []

    repair_unit = redistribution_to_repair_unit(uid, cue_indexes, redistributed)
    contract = validate_repair_contract([repair_unit], [unit], source_entries)

    trial = list(vi_entries)
    for idx in cue_indexes:
        if idx in redistributed and 0 < idx <= len(trial):
            trial[idx - 1] = {**trial[idx - 1], "text": redistributed[idx]}

    before = _unit_quality_score(
        source_entries, vi_entries, cue_indexes, meaning_units, video_context
    )
    after = _unit_quality_score(
        source_entries, trial, cue_indexes, meaning_units, video_context
    )

    source_texts = [e.get("text", "") for e in source_entries]
    vi_texts_before = [e.get("text", "") for e in vi_entries]
    vi_texts_after = [e.get("text", "") for e in trial]
    shift_before = count_shifts_in_cues(
        cue_indexes, source_texts, vi_texts_before, video_context
    )
    shift_after = count_shifts_in_cues(
        cue_indexes, source_texts, vi_texts_after, video_context
    )

    accept = True
    reasons: List[str] = []
    needs_human_review = False

    if not contract.get("valid", False):
        accept = False
        reasons.append("repair_contract_failed")

    if unit_translation and _meaning_coverage(unit_translation, redistributed, cue_indexes) < _MIN_MEANING_COVERAGE:
        accept = False
        reasons.append("meaning_coverage_low")

    for idx in cue_indexes:
        src = source_entries[idx - 1].get("text", "").strip() if idx <= len(source_entries) else ""
        t = redistributed.get(idx, "").strip()
        if src and not t:
            accept = False
            reasons.append(f"empty_cue_{idx}")
            break
        if t and _has_dangling_end(t):
            accept = False
            reasons.append(f"dangling_fragment_cue_{idx}")
            break

    error_types = set(unit.get("detected_translation_errors") or [])
    if "repeated_meaning_error" in error_types and after["repeated_meaning"] > before["repeated_meaning"]:
        accept = False
        reasons.append("repeated_meaning_worsened")
    if "cue_flow_error" in error_types and after["weak_fragments"] > before["weak_fragments"]:
        accept = False
        reasons.append("weak_fragments_increased")

    if after["severe_alignment"] > before["severe_alignment"]:
        accept = False
        reasons.append("severe_alignment_worse")

    if shift_after > shift_before and not _semantic_improved(before, after):
        accept = False
        reasons.append("cue_shift_worsened")

    if accept:
        had_semantic_issues = (
            before["severe_alignment"] > 0
            or bool(error_types & _SEMANTIC_REPAIR_ERROR_TYPES)
        )
        if had_semantic_issues and not _semantic_improved(before, after):
            if after["semantic_total"] >= before["semantic_total"]:
                accept = False
                reasons.append("semantic_not_improved")
        elif after["semantic_total"] > before["semantic_total"] + 1:
            accept = False
            reasons.append("semantic_quality_regressed")
        elif (
            not had_semantic_issues
            and error_types <= {"readability_cps_error"}
            and after["cps_over"] <= before["cps_over"]
            and after["severe_alignment"] <= before["severe_alignment"]
        ):
            pass  # CPS-only touch-up: allow when CPS same or better

    if accept and after["cps_over"] > before["cps_over"]:
        needs_human_review = True

    source_unit_text = unit.get("source_text", "")
    vi_fluency = analyze_unit_fluency(
        redistributed,
        cue_indexes,
        unit_translation=unit_translation,
        source_unit_text=source_unit_text,
        video_context=video_context,
    )
    if vi_fluency.get("has_severe_fluency"):
        semantic_win = (
            _semantic_improved(before, after)
            and before["severe_alignment"] > 0
            and after["severe_alignment"] == 0
        )
        if semantic_win and vi_fluency.get("severe_count", 0) <= 1:
            needs_human_review = True
        else:
            accept = False
            reasons.append("severe_vi_fluency_errors")

    return {
        "unit_id": uid,
        "accept": accept,
        "reasons": reasons,
        "needs_human_review": needs_human_review,
        "contract": contract,
        "before": before,
        "after": after,
        "vi_fluency": vi_fluency,
        "vi_fluency_total": vi_fluency.get("vi_fluency_total", 0),
        "redistributed": {str(k): v for k, v in redistributed.items()},
    }


def _eval_redistribution_rank(eval_report: Dict[str, Any]) -> tuple:
    """Lower is better: accepted > fewer severe fluency > lower semantic issues."""
    fluency = eval_report.get("vi_fluency") or {}
    after = eval_report.get("after") or {}
    return (
        0 if eval_report.get("accept") else 1,
        fluency.get("severe_count", 99),
        after.get("semantic_total", 999),
        eval_report.get("vi_fluency_total", fluency.get("vi_fluency_total", 99)),
        after.get("repeated_meaning", 99),
    )


def _try_redistribute_unit(
    translation: str,
    cue_indexes: List[int],
    source_entries: List[dict],
    unit: dict,
    vi_entries: List[dict],
    meaning_units: Optional[List[dict]],
    video_context: Optional[Dict[str, Any]],
) -> Tuple[Dict[int, str], Dict[str, Any], str]:
    """Try standard, relaxed, then fluency-coherent redistribution; return best eval report."""
    configs = [
        ("standard", RedistributionConfig()),
        ("relaxed_cps", relaxed_redistribution_config()),
        ("fluency_coherent", fluency_coherent_config()),
    ]
    best_redistributed: Dict[int, str] = {}
    best_eval: Optional[Dict[str, Any]] = None
    best_mode = "none"

    for mode, cfg in configs:
        redistributed = redistribute_unit_translation(
            translation, cue_indexes, source_entries, cfg=cfg
        )
        redistributed = _polish_connective_redistribution(
            translation,
            cue_indexes,
            redistributed,
            source_entries,
            original_vi_entries=vi_entries,
        )
        eval_report = evaluate_redistributed_unit(
            source_entries,
            vi_entries,
            unit,
            redistributed,
            meaning_units,
            video_context,
            unit_translation=translation,
        )
        eval_report["redistribution_mode"] = mode
        if eval_report.get("accept"):
            return redistributed, eval_report, mode
        if best_eval is None or _eval_redistribution_rank(eval_report) < _eval_redistribution_rank(
            best_eval
        ):
            best_redistributed = redistributed
            best_eval = eval_report
            best_mode = mode

    assert best_eval is not None
    return best_redistributed, best_eval, best_mode


def apply_unit_rewrite_repairs(
    source_entries: List[dict],
    vi_entries: List[dict],
    unit_rewrites: List[dict],
    expected_units: List[dict],
    meaning_units: Optional[List[dict]],
    video_context: Optional[Dict[str, Any]],
    *,
    cfg: Optional[RedistributionConfig] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Redistribute unit translations, validate, apply accepted units only."""
    cfg = cfg or RedistributionConfig()
    expected_by_id = {u["unit_id"]: u for u in expected_units}
    result = list(vi_entries)
    vi_texts = [e.get("text", "") for e in vi_entries]

    redistribution_reports: List[dict] = []
    vi_fluency_reports: List[dict] = []
    accepted: Dict[int, dict] = {}
    rejected: Dict[int, dict] = {}
    rejected_units: Dict[int, dict] = {}
    human_review_units: List[int] = []
    repair_units: List[dict] = []

    for rewrite in unit_rewrites:
        uid = rewrite.get("unit_id")
        unit = expected_by_id.get(uid)
        if not unit:
            continue

        cue_indexes = unit.get("cue_indexes") or []
        translation = rewrite.get("unit_translation", "")
        redistributed, eval_report, mode = _try_redistribute_unit(
            translation,
            cue_indexes,
            source_entries,
            unit,
            vi_entries,
            meaning_units,
            video_context,
        )

        redistribution_reports.append(
            {
                **eval_report,
                "unit_translation": translation,
                "notes": rewrite.get("notes", ""),
                "redistribution_mode_used": mode,
            }
        )
        if eval_report.get("vi_fluency"):
            vi_fluency_reports.append(
                {
                    "unit_id": uid,
                    "cue_indexes": cue_indexes,
                    "redistribution_mode": mode,
                    "accepted": bool(eval_report.get("accept")),
                    **eval_report["vi_fluency"],
                }
            )

        repair_unit = redistribution_to_repair_unit(uid, cue_indexes, redistributed)
        repair_units.append(repair_unit)

        if not eval_report.get("accept"):
            rejected_units[uid] = {
                "reason": "; ".join(eval_report.get("reasons", [])) or "rejected",
                "unit_id": uid,
                "cue_indexes": cue_indexes,
                "unit_translation": translation,
                "redistribution_mode": mode,
                "human_review_needed": True,
            }
            human_review_units.append(uid)
            for idx in cue_indexes:
                rejected[idx] = {
                    "reason": rejected_units[uid]["reason"],
                    "proposed": redistributed.get(idx, ""),
                    "kept": vi_texts[idx - 1].strip(),
                    "unit_id": uid,
                }
            continue

        if eval_report.get("needs_human_review"):
            human_review_units.append(uid)

        for idx in cue_indexes:
            new_text = redistributed.get(idx, "").strip()
            old_text = vi_texts[idx - 1].strip()
            if new_text and new_text != old_text:
                result[idx - 1] = {**result[idx - 1], "text": new_text}
                vi_texts[idx - 1] = new_text
                accepted[idx] = {"before": old_text, "after": new_text}

    combined_contract = validate_repair_contract(
        repair_units, expected_units, source_entries
    )

    return result, {
        "contract": combined_contract,
        "redistribution_reports": redistribution_reports,
        "vi_fluency_reports": vi_fluency_reports,
        "accepted": accepted,
        "rejected": rejected,
        "rejected_units": rejected_units,
        "human_review_units": human_review_units,
        "applied": bool(accepted),
    }


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
