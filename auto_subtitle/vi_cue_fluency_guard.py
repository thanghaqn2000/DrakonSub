"""Vietnamese cue fluency validation for redistributed subtitle text."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

# Natural short cues that are valid on screen.
_VALID_SHORT_PATTERNS = [
    re.compile(r"(?i)^gì cơ\??$"),
    re.compile(r"(?i)^không có đâu\.?$"),
    re.compile(r"(?i)^đúng vậy\.?$"),
    re.compile(r"(?i)^làm gì đi\.?$"),
    re.compile(r"(?i)^cái gì\??$"),
    re.compile(r"(?i)^gì\??$"),
    re.compile(r"(?i)^ừ\.?$"),
    re.compile(r"(?i)^vâng\.?$"),
]

_DANGLING_START_RE = re.compile(
    r"(?i)^(?:bạn|anh|chị|họ|người đó|người ta|nó|điều đó|cái đó|chuyện đó|"
    r"bất cứ|mọi người|về|với|của|cho|mà|rằng|để|nên|vì|nhưng|và|hoặc|thì)\b"
)

_UNFINISHED_END_PHRASES_RE = re.compile(
    r"(?i)(?:tôi sẽ không|tại sao tôi|nếu tôi|bởi vì|vì tôi|mà tôi|"
    r"anh sẽ không|bạn sẽ không|người đó|về điều|về nó|về cái)$"
)
_UNFINISHED_END_CONNECTIVE_RE = re.compile(
    r"(?i)(?:và|là|mà|rằng|vì|nếu|khi|cho|với|về|của|nên|thì|hoặc)$"
)

_CONNECTIVE_ONLY_RE = re.compile(
    r"(?i)^(?:và|nhưng|mà|rằng|vì|nếu|khi|cho|với|về|của|nên|thì|hoặc|để)\.?$"
)

_OBJECT_TAIL_RE = re.compile(
    r"(?i)^(?:bạn|anh|chị|họ|tôi|nó)\s+.+(?:cả|đó|này|đây)\.?$"
)

_PREDICATE_HINT_RE = re.compile(
    r"(?i)\b(?:là|được|bị|có|không|sẽ|đã|đang|phải|nên|cần|muốn|"
    r"gọi|làm|mua|bán|nói|hỏi|biết|đưa|nhận|giải thích|đúng|sai)\b"
)

_CONNECTIVE_OBJECT_START_RE = re.compile(
    r"(?i)^(?:cho|về|với|của|mà|rằng)\s+"
)

_DANGLING_OBJECT_PHRASE_RE = re.compile(
    r"(?i)^(?:bạn|anh|chị|họ|tôi|nó)\s+"
    r"(?:bất cứ|mọi|một|vài|những|điều|cái|thứ|chuyện).+(?:cả|đều|này|đó|đây)\.?$"
)

_QUESTION_RE = re.compile(r"\?$|^(?:tại sao|sao|gì|ai|ở đâu|khi nào|thế nào)\b", re.I)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _continues_in_next_cue(text: str, next_text: str) -> bool:
    next_text = _normalize(next_text)
    if not next_text:
        return False
    t = text.strip()
    if t.rstrip().endswith(",") and _PREDICATE_HINT_RE.search(next_text):
        return True
    if _CONNECTIVE_ONLY_RE.match(t) and _PREDICATE_HINT_RE.search(next_text):
        return True
    if re.match(r"(?i)^(và|nhưng|mà|rằng|vì|nên|thì|hoặc|để)$", t) and next_text:
        return True
    return False


def _is_valid_short_cue(text: str) -> bool:
    t = text.strip()
    for pat in _VALID_SHORT_PATTERNS:
        if pat.match(t):
            return True
    if _QUESTION_RE.search(t) and len(t.split()) <= 4:
        if _UNFINISHED_END_PHRASES_RE.search(t):
            return False
        if t.endswith("?"):
            return True
    return False


def analyze_cue_fluency(
    cue_index: int,
    text: str,
    *,
    prev_text: str = "",
    next_text: str = "",
    position_in_unit: int = 0,
    unit_size: int = 1,
) -> Dict[str, Any]:
    """Return fluency assessment for one cue."""
    text = _normalize(text)
    errors: List[str] = []
    severity = "none"

    if not text:
        return {
            "cue_index": cue_index,
            "text": text,
            "is_fluent": True,
            "errors": [],
            "severity": "none",
        }

    if _is_valid_short_cue(text):
        return {
            "cue_index": cue_index,
            "text": text,
            "is_fluent": True,
            "errors": [],
            "severity": "none",
        }

    if _CONNECTIVE_ONLY_RE.match(text):
        errors.append("connective_fragment")
    elif _DANGLING_START_RE.match(text) and not _PREDICATE_HINT_RE.search(text):
        if not _continues_in_next_cue(text, next_text):
            if position_in_unit == 0 or not prev_text.rstrip().endswith((",", "—", "…")):
                errors.append("dangling_object")

    if _UNFINISHED_END_PHRASES_RE.search(text) or _UNFINISHED_END_CONNECTIVE_RE.search(text):
        errors.append("unfinished_phrase")

    if _CONNECTIVE_OBJECT_START_RE.match(text) and not _PREDICATE_HINT_RE.search(text):
        errors.append("dangling_object")

    if _DANGLING_OBJECT_PHRASE_RE.match(text):
        errors.append("dangling_object")
    elif _OBJECT_TAIL_RE.match(text) and not _PREDICATE_HINT_RE.search(text):
        errors.append("dangling_object")

    words = text.split()
    if len(words) <= 6 and not _PREDICATE_HINT_RE.search(text) and not text.endswith("?"):
        if _DANGLING_START_RE.match(text) or text[0].islower():
            errors.append("missing_predicate")

    if position_in_unit > 0:
        prev = prev_text.rstrip()
        if text[0].islower() and not prev.endswith((",", ":", "—", "…", ".", "?", "!")):
            if (_DANGLING_START_RE.match(text) and not _PREDICATE_HINT_RE.search(text)) or (
                _UNFINISHED_END_PHRASES_RE.search(text)
            ):
                errors.append("bad_sentence_order")
            elif len(words) <= 5 and not _PREDICATE_HINT_RE.search(text):
                errors.append("leftover_tail")

    if next_text and text.lower() in next_text.lower() and len(text) >= 8:
        errors.append("leftover_tail")

    if len(words) <= 3 and not _PREDICATE_HINT_RE.search(text) and not text.endswith("?"):
        if any(text.lower().startswith(p) for p in ("về ", "của ", "cho ", "mà ", "rằng ")):
            if not _continues_in_next_cue(text, next_text):
                errors.append("connective_fragment")

    if _continues_in_next_cue(text, next_text):
        errors = [
            e
            for e in errors
            if e
            not in {
                "connective_fragment",
                "missing_predicate",
                "dangling_object",
                "bad_sentence_order",
                "unfinished_phrase",
            }
        ]

    if errors:
        severe_types = {
            "dangling_object",
            "unfinished_phrase",
            "missing_predicate",
            "leftover_tail",
            "bad_sentence_order",
        }
        severity = "severe" if any(e in severe_types for e in errors) else "mild"
        errors = list(dict.fromkeys(errors))

    return {
        "cue_index": cue_index,
        "text": text,
        "is_fluent": not errors or severity == "mild",
        "errors": errors,
        "severity": severity if errors else "none",
    }


def analyze_unit_fluency(
    redistributed: Dict[int, str],
    cue_indexes: List[int],
    *,
    unit_translation: str = "",
    source_unit_text: str = "",
    video_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze fluency for all cues in a redistributed unit."""
    cue_issues: List[dict] = []
    severe_count = 0
    mild_count = 0

    for pos, idx in enumerate(cue_indexes):
        prev_t = redistributed.get(cue_indexes[pos - 1], "") if pos > 0 else ""
        next_t = redistributed.get(cue_indexes[pos + 1], "") if pos + 1 < len(cue_indexes) else ""
        issue = analyze_cue_fluency(
            idx,
            redistributed.get(idx, ""),
            prev_text=prev_t,
            next_text=next_t,
            position_in_unit=pos,
            unit_size=len(cue_indexes),
        )
        cue_issues.append(issue)
        if issue["severity"] == "severe":
            severe_count += 1
        elif issue["severity"] == "mild":
            mild_count += 1

    return {
        "cue_issues": cue_issues,
        "vi_fluency_total": severe_count * 3 + mild_count,
        "severe_count": severe_count,
        "mild_count": mild_count,
        "has_severe_fluency": severe_count > 0,
    }
