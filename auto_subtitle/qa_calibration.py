"""QA calibration — downgrade documented false positives without hiding true errors."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .semantic_alignment_guard import _extract_concepts, _overlap_ratio
from .subtitle_timing_optimizer import _parse_ts
from .vi_compression import _cps

_FRAGMENT_CPS_MAX_DURATION = 0.8
_FRAGMENT_CPS_CHAR_CAP = 42

_TAKE_NO_MONEY_RE = re.compile(r"(?i)take\s+no\s+money|no\s+rush\s+to\s+take")
_KIEM_TIEN_RE = re.compile(r"(?i)kiếm\s+tiền")
_NHAN_TIEN_RE = re.compile(r"(?i)(nhận|lấy)\s+tiền")

_BUILD_REPEAT_RE = re.compile(r"(?i)\bbuild\b")


def _cue_duration(entry: dict) -> float:
    return max(0.01, _parse_ts(entry["end_str"]) - _parse_ts(entry["start_str"]))


def _is_source_fragment(source: str) -> bool:
    s = source.strip()
    if not s:
        return False
    return not s.rstrip().endswith((".", "?", "!")) or len(s.split()) <= 6


def _fragment_continues_sentence(
    source_texts: List[str], cue_index: int
) -> bool:
    i = cue_index - 1
    n = len(source_texts)
    cur = source_texts[i].strip() if 0 <= i < n else ""
    nxt = source_texts[i + 1].strip() if i + 1 < n else ""
    if not cur.rstrip().endswith((".", "?", "!")):
        return True
    if nxt and nxt[0].islower():
        return True
    return False


def _en_rhetorical_repeat(source_texts: List[str], cue_index: int) -> bool:
    i = cue_index - 1
    if i < 0 or i + 1 >= len(source_texts):
        return False
    en1 = source_texts[i].lower()
    en2 = source_texts[i + 1].lower()
    if _BUILD_REPEAT_RE.search(en1) and _BUILD_REPEAT_RE.search(en2):
        return True
    c1 = _extract_concepts(en1, "en")
    c2 = _extract_concepts(en2, "en")
    if c1 and c2:
        overlap = len(c1 & c2) / min(len(c1), len(c2))
        if overlap >= 0.35:
            return True
    return False


def _minor_nuance_take_no_money(source: str, vi: str) -> bool:
    if not _TAKE_NO_MONEY_RE.search(source):
        return False
    if _NHAN_TIEN_RE.search(vi):
        return False
    return bool(_KIEM_TIEN_RE.search(vi))


def apply_qa_calibration(
    cue_index: int,
    source: str,
    vi: str,
    entry: dict,
    source_texts: List[str],
    vi_texts: List[str],
    errors: List[str],
    alignment_warnings: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Return (filtered_errors, filtered_warnings, calibration_notes)."""
    notes: List[str] = []
    out = list(errors)
    warns = list(alignment_warnings)
    src = source.strip()
    text = vi.strip()
    dur = _cue_duration(entry)
    cps_val = _cps(text, dur) if text else 0.0

    if "readability_cps_error" in out:
        if (
            dur < _FRAGMENT_CPS_MAX_DURATION
            and _is_source_fragment(src)
            and _fragment_continues_sentence(source_texts, cue_index)
            and len(text) <= _FRAGMENT_CPS_CHAR_CAP
        ):
            out.remove("readability_cps_error")
            notes.append("cps_downgrade_fragment_timing")

    if "repeated_meaning_error" in out and _en_rhetorical_repeat(source_texts, cue_index):
        out.remove("repeated_meaning_error")
        notes.append("repeated_meaning_downgrade_en_rhetoric")

    if "repeated_meaning_error" in out and _en_rhetorical_repeat(
        source_texts, max(1, cue_index - 1)
    ):
        out.remove("repeated_meaning_error")
        notes.append("repeated_meaning_downgrade_en_rhetoric")

    if _minor_nuance_take_no_money(src, text):
        for eid in ("semantic_alignment_error", "semantic_drift_error"):
            if eid in out:
                out.remove(eid)
                notes.append("minor_nuance_loss_take_no_money")
        if "semantic_alignment_error" in warns:
            warns.remove("semantic_alignment_error")
            notes.append("minor_nuance_loss_take_no_money")

    return out, warns, notes


def classify_cue_attribution(
    errors: List[str],
    calibration_notes: List[str],
    cps_val: float,
    duration: float,
) -> dict:
  """Heuristic labels for attribution report."""
  is_fp = bool(calibration_notes)
  semantic_true = any(
      e in errors for e in ("semantic_alignment_error", "semantic_drift_error")
  )
  readability_true = "readability_cps_error" in errors
  repeated_true = "repeated_meaning_error" in errors

  layer = "unknown"
  if semantic_true:
      layer = "raw_translation"
  elif repeated_true:
      layer = "qa_calibration"
  elif readability_true:
      layer = "timing" if duration < 1.0 else "compression"
  elif not errors and calibration_notes:
      layer = "qa_calibration"

  recommended = "none"
  if semantic_true:
      recommended = "raw_prompt_polish"
  elif readability_true:
      recommended = "timing_optimizer"
  elif repeated_true:
      recommended = "qa_calibration"
  elif is_fp:
      recommended = "qa_calibration"

  return {
      "is_semantic_true_error": semantic_true,
      "is_readability_true_error": readability_true and not is_fp,
      "is_repeated_meaning_true_error": repeated_true,
      "is_qa_false_positive_candidate": is_fp,
      "likely_failure_layer": layer,
      "recommended_fix": recommended,
      "calibration_notes": calibration_notes,
  }
