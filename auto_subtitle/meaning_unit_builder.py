"""Rule-based meaning unit builder — group neighboring cues before translation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

_MAX_UNIT_CUES = 5

_INCOMPLETE_END_RE = re.compile(
    r"(?i)(?:\b(?:but|because|that|which|who|whom|whose|if|when|while|although|though|"
    r"and|or|so|as|than|for|to|with|of|in|on|at|by|from|into|about|over|under|"
    r"the|a|an|my|your|his|her|its|our|their|this|these|that|those)\s*)$|"
    r"[,;:—-]\s*$"
)

_CONTINUATION_START_RE = re.compile(
    r"(?i)^(?:and|but|or|so|because|that|which|who|whom|whose|if|when|while|"
    r"although|though|with|for|to|of|in|on|at|by|from|into|about|over|under|"
    r"the|a|an|my|your|his|her|its|our|their|this|these|that|those|it|they|he|she|we|you)\b"
)

_PRONOUN_END_RE = re.compile(
    r"(?i)\b(?:it|this|that|these|those|they|them|he|she|we|you|who|which)\s*[,;:]?\s*$"
)

_SENTENCE_END_RE = re.compile(r"[.!?…][\"')\]}]*\s*$")

_LOW_STANDALONE_RE = re.compile(
    r"(?i)^(?:\w{1,3}\.?|ok\.?|yeah\.?|right\.?|so\.?|well\.?)$"
)


def _cue_texts_from_entries(entries: List[dict]) -> Dict[int, str]:
    return {i + 1: e.get("text", "").strip() for i, e in enumerate(entries)}


def _has_clear_standalone_meaning(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if _LOW_STANDALONE_RE.match(text):
        return False
    if _INCOMPLETE_END_RE.search(text):
        return False
    if _PRONOUN_END_RE.search(text) and not _SENTENCE_END_RE.search(text):
        return False
    if len(text.split()) >= 4 and _SENTENCE_END_RE.search(text):
        return True
    return len(text.split()) >= 6


def _should_continue_unit(prev_text: str, next_text: str) -> tuple[bool, str, List[str]]:
    risks: List[str] = []
    prev = prev_text.strip()
    nxt = next_text.strip()
    if not prev or not nxt:
        return False, "", risks

    if _INCOMPLETE_END_RE.search(prev):
        risks.append("cue_fragmentation_error")
        return True, "incomplete clause continues across cues", risks

    if _CONTINUATION_START_RE.match(nxt) and not _SENTENCE_END_RE.search(prev):
        risks.append("cue_fragmentation_error")
        return True, "next cue continues previous clause", risks

    if _PRONOUN_END_RE.search(prev):
        risks.append("pronoun_reference_error")
        return True, "pronoun reference depends on following cue", risks

    if prev[-1:].islower() and nxt and nxt[0].islower() and not _SENTENCE_END_RE.search(prev):
        risks.append("split_term_across_cues_error")
        return True, "lowercase continuation suggests same phrase", risks

    if not _has_clear_standalone_meaning(prev):
        risks.append("cue_fragmentation_error")
        return True, "previous cue lacks clear standalone meaning", risks

    return False, "", risks


def build_meaning_units(entries: List[dict]) -> List[dict]:
    """
    Group 1-based cue indexes into meaning units (max 5 cues each).
    Preserves all cue indexes; does not merge timestamps.
    """
    texts = _cue_texts_from_entries(entries)
    cue_nums = sorted(n for n, t in texts.items() if t)
    if not cue_nums:
        return []

    units: List[dict] = []
    unit_id = 0
    i = 0
    while i < len(cue_nums):
        unit_id += 1
        start_cue = cue_nums[i]
        group = [start_cue]
        reasons: List[str] = []
        risk_flags: Set[str] = set()

        while i + 1 < len(cue_nums) and len(group) < _MAX_UNIT_CUES:
            cur = texts[cue_nums[i]]
            nxt_cue = cue_nums[i + 1]
            if nxt_cue != cue_nums[i] + 1:
                break
            cont, reason, risks = _should_continue_unit(cur, texts[nxt_cue])
            if not cont:
                break
            group.append(nxt_cue)
            if reason:
                reasons.append(reason)
            risk_flags.update(risks)
            i += 1

        source_text = " ".join(texts[c] for c in group)
        units.append(
            {
                "unit_id": unit_id,
                "cue_indexes": group,
                "source_text": source_text,
                "reason": "; ".join(dict.fromkeys(reasons)) or "single complete cue",
                "source_risk_flags": sorted(risk_flags),
            }
        )
        i += 1

    return units


def meaning_units_to_local_groups(
    meaning_units: List[dict],
    entry_index_to_local: Dict[int, int],
) -> List[List[int]]:
    """
    Convert meaning units (1-based entry indexes) to local index groups
    for the non-empty translation list.
    """
    groups: List[List[int]] = []
    for unit in meaning_units:
        local = [entry_index_to_local[c] for c in unit["cue_indexes"] if c in entry_index_to_local]
        if local:
            groups.append(local)
    return groups


def resolve_translation_groups(
    all_texts: List[str],
    non_empty_indices: List[int],
    max_cues_per_group: int,
    translation_context: Optional[dict],
    fallback_group_fn,
) -> List[List[int]]:
    """Prefer meaning-unit groups; fall back to phrase/sentence grouping."""
    ctx = translation_context or {}
    meaning_units = ctx.get("meaning_units")
    if meaning_units:
        entry_to_local = {
            entry_idx + 1: local_idx
            for local_idx, entry_idx in enumerate(non_empty_indices)
        }
        groups = meaning_units_to_local_groups(meaning_units, entry_to_local)
        if groups:
            return groups
    return fallback_group_fn(all_texts, max_cues_per_group=max_cues_per_group)


def save_meaning_units(path: str | Path, units: List[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")
