"""Detect local cue-shift misalignment (VI matches wrong EN cue index)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .semantic_alignment_guard import (
    _association_score,
    _detect_cue_shift,
    _glossary_bridges,
    _unit_alignment_ok,
)

_SHIFT_WINDOW = 3
_MIN_CONFIDENCE = 0.55
_WINDOW_ID_BASE = 10_000


def _cue_to_unit(meaning_units: Optional[List[dict]]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if not meaning_units:
        return mapping
    for unit in meaning_units:
        uid = unit.get("unit_id")
        for c in unit.get("cue_indexes") or []:
            mapping[c] = uid
    return mapping


def profile_cue_alignment(
    cue_idx: int,
    source_texts: List[str],
    vi_texts: List[str],
    video_context: Optional[Dict[str, Any]] = None,
    *,
    search_window: int = _SHIFT_WINDOW,
) -> Dict[str, Any]:
    """Score VI cue against own EN and neighbors; return best-matching source index."""
    n = len(source_texts)
    vi = vi_texts[cue_idx - 1].strip()
    en = source_texts[cue_idx - 1].strip()
    if not vi or not en:
        return {
            "cue_index": cue_idx,
            "en_current": en,
            "vi_current": vi,
            "current_score": 0.0,
            "best_matching_source_cue": cue_idx,
            "shifted_score": 0.0,
            "confidence": 0.0,
            "is_shifted": False,
            "pattern": "none",
        }

    bridges = _glossary_bridges(video_context)
    own_score = _association_score(vi, en, bridges)
    best_j = cue_idx
    best_score = own_score
    for j in range(max(1, cue_idx - search_window), min(n, cue_idx + search_window) + 1):
        if j == cue_idx:
            continue
        score = _association_score(vi, source_texts[j - 1], bridges)
        if score > best_score:
            best_score = score
            best_j = j

    shifted, shift_reason, conf = _detect_cue_shift(
        cue_idx, source_texts, vi_texts, video_context, window=search_window
    )
    pattern = "none"
    if shifted and best_j > cue_idx:
        pattern = "vi_matches_next_source"
    elif shifted and best_j < cue_idx:
        pattern = "vi_matches_prev_source"
    elif shifted:
        pattern = "vi_matches_neighbor_source"

    return {
        "cue_index": cue_idx,
        "en_current": en,
        "vi_current": vi,
        "current_score": round(own_score, 2),
        "best_matching_source_cue": best_j,
        "shifted_score": round(best_score, 2),
        "confidence": round(conf, 3),
        "is_shifted": shifted,
        "pattern": pattern,
        "shift_reason": shift_reason if shifted else "",
    }


def _is_continuation_fragment(
    cue_idx: int,
    profile: Dict[str, Any],
    meaning_units: Optional[List[dict]],
    source_texts: List[str],
    vi_texts: List[str],
    video_context: Optional[Dict[str, Any]],
) -> bool:
    """Skip shift flag when cue is a valid fragment inside an aligned meaning unit."""
    cue_to_unit = _cue_to_unit(meaning_units)
    uid = cue_to_unit.get(cue_idx)
    if uid is None:
        return False
    unit_cues = [
        c
        for unit in (meaning_units or [])
        if unit.get("unit_id") == uid
        for c in (unit.get("cue_indexes") or [])
    ]
    if not unit_cues or cue_idx not in unit_cues:
        return False
    ok, score = _unit_alignment_ok(unit_cues, source_texts, vi_texts, video_context)
    if not ok or score < 6.0:
        return False
    if profile.get("is_shifted") and profile.get("confidence", 0) < 0.8:
        return True
    en = profile.get("en_current", "")
    if len(en.split()) <= 4 and profile.get("current_score", 0) >= 1.5:
        return True
    return False


def _merge_shift_windows(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group consecutive forward/backward shift cues into windows."""
    if not profiles:
        return []

    windows: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        indexes = [p["cue_index"] for p in current]
        patterns = {p["pattern"] for p in current}
        if "vi_matches_next_source" in patterns and all(
            current[i]["best_matching_source_cue"] >= current[i]["cue_index"]
            for i in range(len(current))
        ):
            pattern = "local_forward_shift"
        elif "vi_matches_prev_source" in patterns:
            pattern = "local_backward_shift"
        else:
            pattern = current[0].get("pattern", "vi_matches_neighbor_source")
        conf = sum(p.get("confidence", 0) for p in current) / len(current)
        windows.append(
            {
                "cue_indexes": indexes,
                "pattern": pattern,
                "confidence": round(conf, 3),
                "evidence": [
                    {
                        "cue_index": p["cue_index"],
                        "en_current": p.get("en_current", ""),
                        "vi_current": p.get("vi_current", ""),
                        "best_matching_source_cue": p.get("best_matching_source_cue"),
                        "current_score": p.get("current_score"),
                        "shifted_score": p.get("shifted_score"),
                    }
                    for p in current
                ],
                "likely_root_cause": (
                    "VI text aligned to adjacent EN cue(s); local window off by one or more beats"
                ),
            }
        )
        current.clear()

    for p in sorted(profiles, key=lambda x: x["cue_index"]):
        if not current:
            current.append(p)
            continue
        prev = current[-1]
        forward_chain = (
            p["cue_index"] == prev["cue_index"] + 1
            and p.get("best_matching_source_cue", p["cue_index"]) >= p["cue_index"]
            and prev.get("best_matching_source_cue", prev["cue_index"]) >= prev["cue_index"]
        )
        backward_chain = (
            p["cue_index"] == prev["cue_index"] + 1
            and p.get("best_matching_source_cue", p["cue_index"]) <= p["cue_index"]
            and prev.get("best_matching_source_cue", prev["cue_index"]) <= prev["cue_index"]
        )
        if forward_chain or backward_chain:
            current.append(p)
        else:
            flush()
            current.append(p)
    flush()
    return windows


def detect_local_shift_windows(
    source_entries: List[dict],
    vi_entries: List[dict],
    meaning_units: Optional[List[dict]] = None,
    video_context: Optional[Dict[str, Any]] = None,
    *,
    min_confidence: float = _MIN_CONFIDENCE,
    min_window_cues: int = 2,
) -> List[Dict[str, Any]]:
    """Return shift windows with window_id for repair."""
    source_texts = [e.get("text", "") for e in source_entries]
    vi_texts = [e.get("text", "") for e in vi_entries]
    n = len(source_texts)
    shifted_profiles: List[Dict[str, Any]] = []

    for i in range(1, n + 1):
        profile = profile_cue_alignment(i, source_texts, vi_texts, video_context)
        if not profile.get("is_shifted"):
            continue
        if profile.get("confidence", 0) < min_confidence:
            continue
        if _is_continuation_fragment(
            i, profile, meaning_units, source_texts, vi_texts, video_context
        ):
            continue
        shifted_profiles.append(profile)

    raw_windows = _merge_shift_windows(shifted_profiles)
    windows: List[Dict[str, Any]] = []
    profile_by_idx = {p["cue_index"]: p for p in shifted_profiles}
    for idx, w in enumerate(raw_windows):
        if len(w["cue_indexes"]) < min_window_cues:
            if w.get("confidence", 0) < 0.85:
                continue
            # Expand single high-confidence shift to adjacent shifted cues
            expanded = list(w["cue_indexes"])
            for delta in (1, -1):
                neighbor = expanded[-1] + delta if delta > 0 else expanded[0] + delta
                if neighbor in profile_by_idx and neighbor not in expanded:
                    expanded.append(neighbor)
            expanded.sort()
            w = {**w, "cue_indexes": expanded}
        windows.append({**w, "window_id": _WINDOW_ID_BASE + idx})
    return windows


def diagnose_sample(
    sample_id: str,
    source_entries: List[dict],
    vi_entries: List[dict],
    meaning_units: Optional[List[dict]] = None,
    video_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    windows = detect_local_shift_windows(
        source_entries, vi_entries, meaning_units, video_context
    )
    return {
        "sample": sample_id,
        "has_local_cue_shift": bool(windows),
        "shift_windows": windows,
    }


def build_aggregate_diagnosis(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    with_shift = [s for s in samples if s.get("has_local_cue_shift")]
    patterns: Dict[str, int] = {}
    total_windows = 0
    for s in samples:
        for w in s.get("shift_windows") or []:
            total_windows += 1
            p = w.get("pattern", "unknown")
            patterns[p] = patterns.get(p, 0) + 1
    most_common = max(patterns, key=patterns.get) if patterns else ""
    return {
        "samples": samples,
        "summary": {
            "sample_count_with_shift": len(with_shift),
            "total_shift_windows": total_windows,
            "most_common_pattern": most_common,
            "pattern_counts": patterns,
        },
    }


def count_shifts_in_cues(
    cue_indexes: List[int],
    source_texts: List[str],
    vi_texts: List[str],
    video_context: Optional[Dict[str, Any]] = None,
) -> int:
    """Count cues in index list that are shifted."""
    count = 0
    for idx in cue_indexes:
        shifted, _, conf = _detect_cue_shift(idx, source_texts, vi_texts, video_context)
        if shifted and conf >= _MIN_CONFIDENCE:
            count += 1
    return count
