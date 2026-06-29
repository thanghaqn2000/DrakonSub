"""Semantic alignment guard — detect cue-level meaning drift and repair contract violations."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_EN_STOP = frozenset(
    """
    the a an and or but if so to of in on at by for with from as is are was were be been
    being have has had do does did will would could should may might must shall can need
    you your yours we they he she it its this that these those what which who whom whose
    when where why how all any some no not than then there here just very also only even
    still already about into over after before because while though although i me my mine
    our us him her his hers their them than up out off down
    """.split()
)

_VI_STOP = frozenset(
    """
    và hoặc nhưng mà thì là có được không một các những này kia đó đây cho với từ trong
    trên dưới khi nếu vì nên cũng rất quá đã sẽ bị bạn tôi chúng ta họ anh ấy cô ấy nó
    gì ai đâu sao thế này kia đi à ừ ồ
    """.split()
)

_GENERIC_VI_PATTERNS = [
    re.compile(r"(?i)^làm gì đó"),
    re.compile(r"(?i)đó là cách"),
    re.compile(r"(?i)thị trường hoạt động"),
    re.compile(r"(?i)^điều này"),
    re.compile(r"(?i)^chuyện đó"),
]

_QUESTION_WORDS = frozenset(
    {"what", "when", "where", "why", "how", "who", "which", "that", "this", "these", "those"}
)
# Sentence-initial / discourse words often capitalized in EN cues — not named entities.
_DISCOURSE_CAPITALIZED = frozenset(
    {
        "suppose", "well", "because", "embrace", "now", "then", "but", "however", "although",
        "though", "yes", "okay", "right", "sure", "look", "listen", "remember", "imagine",
        "consider", "actually", "basically", "honestly", "literally", "seriously", "anyway",
        "so", "and", "or", "if", "when", "while", "after", "before", "since", "until",
        "here", "there", "just", "maybe", "perhaps", "really", "simply", "clearly",
    }
)
_QUESTION_MARKERS_EN = re.compile(r"\?|^(why|how|what|when|where|who)\b", re.I)
_QUESTION_MARKERS_VI = re.compile(r"\?|^(tại sao|sao|gì|ai|ở đâu|khi nào|thế nào)\b", re.I)


def _tokenize(text: str, lang: str) -> List[str]:
    text = text.lower().strip()
    if lang == "en":
        return re.findall(r"[a-z0-9']+", text)
    return re.findall(r"[\w']+", text, flags=re.UNICODE)


def _extract_concepts(text: str, lang: str = "en") -> Set[str]:
    if not text.strip():
        return set()
    stop = _EN_STOP if lang == "en" else _VI_STOP
    concepts: Set[str] = set()
    for m in re.finditer(r"\$?\d+(?:\.\d+)?", text):
        concepts.add(m.group().lower())
    for tok in _tokenize(text, lang):
        if len(tok) >= 3 and tok not in stop:
            concepts.add(tok)
    # Loanwords / Latin in Vietnamese subtitles
    for tok in re.findall(r"[a-z]{3,}", text.lower()):
        if tok not in stop:
            concepts.add(tok)
    return concepts


def _glossary_bridges(video_context: Optional[Dict[str, Any]]) -> List[dict]:
    bridges = []
    for item in (video_context or {}).get("key_terms") or []:
        source = str(item.get("source", "")).strip().lower()
        suggested = str(item.get("suggested_vi", "")).strip().lower()
        if source:
            bridges.append(
                {
                    "source": source,
                    "suggested_vi": suggested,
                    "en_tokens": _extract_concepts(source, "en"),
                    "vi_tokens": _extract_concepts(suggested, "vi") if suggested else set(),
                }
            )
    return bridges


def _association_score(
    vi: str,
    en: str,
    bridges: List[dict],
) -> float:
    vi_l = vi.lower()
    en_l = en.lower()
    vi_c = _extract_concepts(vi, "vi")
    en_c = _extract_concepts(en, "en")
    score = len(vi_c & en_c) * 2.0
    score += len(set(re.findall(r"[a-z]{4,}", vi_l)) & set(re.findall(r"[a-z]{4,}", en_l))) * 2.5
    for b in bridges:
        src = b["source"]
        sug = b["suggested_vi"]
        if src and src in en_l:
            if sug and sug in vi_l:
                score += 4.0
            if b["en_tokens"] & vi_c:
                score += 2.0
        if sug and sug in vi_l and src in en_l:
            score += 3.0
    return score


def _overlap_ratio(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def _unit_texts(
    cue_indexes: List[int],
    texts: List[str],
) -> str:
    return " ".join(texts[i - 1].strip() for i in cue_indexes if 0 < i <= len(texts))


def _unit_alignment_ok(
    cue_indexes: List[int],
    source_texts: List[str],
    vi_texts: List[str],
    video_context: Optional[Dict[str, Any]],
) -> Tuple[bool, float]:
    en = _unit_texts(cue_indexes, source_texts)
    vi = _unit_texts(cue_indexes, vi_texts)
    if not en.strip() or not vi.strip():
        return False, 0.0
    bridges = _glossary_bridges(video_context)
    score = _association_score(vi, en, bridges)
    en_c = _extract_concepts(en, "en")
    vi_c = _extract_concepts(vi, "vi")
    overlap = _overlap_ratio(en_c, vi_c) if en_c else 0.0
    ok = score >= 4.0 or overlap >= 0.25
    return ok, score


def _concept_in_unit_text(concept: str, cue_indexes: List[int], vi_texts: List[str]) -> bool:
    unit_vi = _unit_texts(cue_indexes, vi_texts).lower()
    return concept.lower() in unit_vi


def _downgrade_to_unit_warnings(
    cue_issues: List[dict],
    meaning_units: Optional[List[dict]],
    source_texts: List[str],
    vi_texts: List[str],
    video_context: Optional[Dict[str, Any]],
) -> Tuple[List[dict], List[dict]]:
    """Split severe cue issues vs mild warnings when unit-level alignment is OK."""
    if not meaning_units:
        return cue_issues, []

    cue_to_unit: Dict[int, dict] = {}
    for unit in meaning_units:
        for c in unit.get("cue_indexes") or []:
            cue_to_unit[c] = unit

    severe: List[dict] = []
    warnings: List[dict] = []
    _DOWNGRADE_IF_ONLY = frozenset(
        {
            "semantic_alignment_error",
            "semantic_drift_error",
        }
    )
    _KEEP_SEVERE_REASONS = (
        "VI aligns with",
        "glossary concept",
        "duplicate VI text",
    )

    for issue in cue_issues:
        idx = issue["cue_index"]
        unit = cue_to_unit.get(idx)
        errors = list(issue.get("errors") or [])
        reasons = list(issue.get("reasons") or [])

        if not unit or not errors:
            severe.append(issue)
            continue

        unit_cues = unit.get("cue_indexes") or []
        unit_ok, _ = _unit_alignment_ok(unit_cues, source_texts, vi_texts, video_context)
        if not unit_ok:
            severe.append(issue)
            continue

        if any(any(k in r for k in _KEEP_SEVERE_REASONS) for r in reasons):
            severe.append(issue)
            continue

        downgradable = [e for e in errors if e in _DOWNGRADE_IF_ONLY]
        if not downgradable or len(errors) != len(downgradable):
            severe.append(issue)
            continue

        drift_only_entity = (
            "semantic_drift_error" in errors
            and all("named entity missing" in r for r in reasons)
        )
        if drift_only_entity:
            ents = re.findall(r"\['([^']+)'\]", " ".join(reasons))
            if ents and all(_concept_in_unit_text(e, unit_cues, vi_texts) for e in ents):
                warnings.append(
                    {
                        **issue,
                        "errors": ["cue_alignment_warning"],
                        "reasons": [f"entity present in unit VI; cue {idx} is a fragment"],
                    }
                )
                continue

        if any("too generic" in r or "question/statement mismatch" in r for r in reasons):
            warnings.append(
                {
                    **issue,
                    "errors": ["cue_alignment_warning"],
                    "reasons": [f"cue {idx} fragment OK within accurate unit {unit.get('unit_id')}"],
                }
            )
            continue

        severe.append(issue)

    return severe, warnings


def detect_repeated_meaning(
    source_texts: List[str],
    vi_texts: List[str],
    meaning_units: Optional[List[dict]] = None,
) -> Dict[int, str]:
    """Flag adjacent cues that repeat the same VI idea without EN justification."""
    issues: Dict[int, str] = {}
    cue_to_unit: Dict[int, int] = {}
    if meaning_units:
        for unit in meaning_units:
            uid = unit.get("unit_id")
            for c in unit.get("cue_indexes") or []:
                cue_to_unit[c] = uid

    n = len(source_texts)
    for i in range(1, n):
        vi1 = vi_texts[i - 1].strip()
        vi2 = vi_texts[i].strip()
        if not vi1 or not vi2:
            continue
        c1 = _extract_concepts(vi1, "vi")
        c2 = _extract_concepts(vi2, "vi")
        if not c1 or not c2:
            continue
        vi_overlap = len(c1 & c2) / min(len(c1), len(c2))
        en1 = _extract_concepts(source_texts[i - 1], "en")
        en2 = _extract_concepts(source_texts[i], "en")
        en_overlap = len(en1 & en2) / min(len(en1), len(en2)) if en1 and en2 else 0.0
        same_unit = cue_to_unit.get(i) is not None and cue_to_unit.get(i) == cue_to_unit.get(i + 1)

        v1l, v2l = vi1.lower(), vi2.lower()
        if same_unit and (v1l in v2l or v2l in v1l) and min(len(vi1), len(vi2)) >= 8:
            continue

        v1q = vi1.rstrip("?").strip().lower()
        v2q = vi2.rstrip("?").strip().lower()
        question_repeat = (
            vi1.endswith("?")
            and vi2.endswith("?")
            and (v1q in v2q or v2q in v1q or vi_overlap >= 0.55)
        )

        en_overlap_ok = en_overlap > 0.35 or (same_unit and en_overlap > 0.15)
        if en_overlap_ok:
            continue
        vi_threshold = 0.65 if same_unit else 0.5
        if question_repeat:
            reason = "adjacent cues repeat the same question"
        elif vi_overlap >= vi_threshold:
            reason = f"adjacent cues repeat meaning (VI overlap {vi_overlap:.0%})"
        else:
            continue
        issues[i] = reason
        issues[i + 1] = reason
    return issues


def _glossary_term_sets(
    video_context: Optional[Dict[str, Any]],
) -> List[Tuple[str, Set[str], str]]:
    terms: List[Tuple[str, Set[str], str]] = []
    for item in (video_context or {}).get("key_terms") or []:
        source = str(item.get("source", "")).strip()
        suggested = str(item.get("suggested_vi", "")).strip()
        if not source:
            continue
        concepts = _extract_concepts(source, "en") | _extract_concepts(suggested, "vi")
        if concepts:
            terms.append((source.lower(), concepts, suggested.lower()))
    return terms


def _detect_cue_shift(
    cue_idx: int,
    source_texts: List[str],
    vi_texts: List[str],
    video_context: Optional[Dict[str, Any]] = None,
    window: int = 3,
) -> Tuple[bool, str, float]:
    """Return (misaligned, reason, confidence)."""
    n = len(source_texts)
    vi = vi_texts[cue_idx - 1].strip()
    en = source_texts[cue_idx - 1].strip()
    if not vi or not en:
        return False, "", 0.0

    bridges = _glossary_bridges(video_context)
    own_score = _association_score(vi, en, bridges)
    if own_score >= 4.0:
        return False, "", own_score

    best_j = cue_idx
    best_score = own_score
    for j in range(max(1, cue_idx - window), min(n, cue_idx + window) + 1):
        if j == cue_idx:
            continue
        score = _association_score(vi, source_texts[j - 1], bridges)
        if score > best_score:
            best_score = score
            best_j = j

    if best_j != cue_idx and best_score >= 3.0 and own_score < best_score * 0.45:
        direction = "later" if best_j > cue_idx else "earlier"
        return (
            True,
            f"VI aligns with {direction} EN cue {best_j} (score {best_score:.1f} vs {own_score:.1f})",
            min(0.95, best_score / 6.0),
        )
    return False, "", own_score


def _detect_missing_source_concepts(
    cue_idx: int,
    source_texts: List[str],
    vi_texts: List[str],
    video_context: Optional[Dict[str, Any]] = None,
    meaning_units: Optional[List[dict]] = None,
) -> Tuple[bool, str]:
    """Flag when named entities or glossary terms are absent from VI."""
    en = source_texts[cue_idx - 1].strip()
    vi = vi_texts[cue_idx - 1].strip()
    if not en or not vi:
        return False, ""
    bridges = _glossary_bridges(video_context)
    if _association_score(vi, en, bridges) >= 3.5:
        return False, ""

    for b in bridges:
        src = b["source"]
        sug = b["suggested_vi"]
        kind = _glossary_entry_kind(src)
        if not (src and src in en.lower()):
            continue
        if kind in ("phrase", "idiom", "discourse_marker"):
            unit_cues = _find_unit_texts(cue_idx, source_texts, vi_texts, meaning_units)
            unit_en = " ".join(source_texts[c - 1] for c in unit_cues)
            unit_vi = " ".join(vi_texts[c - 1] for c in unit_cues)
            if _association_score(unit_vi, unit_en, bridges) >= 3.0:
                continue
            if _association_score(vi, en, bridges) >= 2.0:
                continue
            en_c = _extract_concepts(en, "en")
            vi_c = _extract_concepts(vi, "vi")
            if en_c and vi_c and _overlap_ratio(en_c, vi_c) >= 0.2:
                continue
        if sug and sug not in vi.lower() and len(sug) >= 4:
            return True, f"glossary '{src}' not reflected in VI"

    proper = re.findall(r"\b[A-Z][a-z]{3,}\b", en)
    en_words = re.findall(r"\b[\w']+\b", en)
    first_word = en_words[0].rstrip(".,?!") if en_words else ""
    named_in_context = [
        str(ne).lower().strip()
        for ne in (video_context or {}).get("named_entities") or []
        if str(ne).strip()
    ]
    missing = []
    for p in proper:
        pl = p.lower()
        if pl in _QUESTION_WORDS or pl in _DISCOURSE_CAPITALIZED:
            continue
        if p == first_word:
            continue
        if pl in vi.lower():
            continue
        idx = en.find(p)
        mid_sentence = idx > 0 and not en[:idx].strip().endswith((".", "!", "?"))
        in_context = any(pl in ne or ne in pl for ne in named_in_context)
        if in_context or (mid_sentence and not named_in_context):
            missing.append(p)
    if missing:
        return True, f"named entity missing in VI: {missing[:3]}"

    return False, ""


def _find_unit_texts(
    cue_idx: int,
    source_texts: List[str],
    vi_texts: List[str],
    meaning_units: Optional[List[dict]] = None,
) -> List[int]:
    if meaning_units:
        for unit in meaning_units:
            if cue_idx in unit.get("cue_indexes", []):
                return unit["cue_indexes"]
    return [cue_idx]


def _detect_generic_vi_when_specific_en(en: str, vi: str) -> bool:
    en = en.strip()
    vi = vi.strip()
    if len(_tokenize(en, "en")) < 5:
        return False
    for pat in _GENERIC_VI_PATTERNS:
        if pat.search(vi):
            return True
    if len(vi.split()) < 4 and len(_tokenize(en, "en")) >= 6:
        if _overlap_ratio(_extract_concepts(vi, "vi"), _extract_concepts(en, "en")) < 0.05:
            return True
    return False


def _detect_question_mismatch(en: str, vi: str) -> bool:
    en_stripped = en.strip()
    if re.search(r"\b(why|what|how|when|where|who)\s*$", en_stripped, re.I) and not en_stripped.endswith("?"):
        return False
    en_q = bool(_QUESTION_MARKERS_EN.search(en))
    vi_q = bool(_QUESTION_MARKERS_VI.search(vi))
    if en_q and not vi_q and _overlap_ratio(_extract_concepts(vi, "vi"), _extract_concepts(en, "en")) < 0.1:
        return True
    if vi_q and not en_q and _overlap_ratio(_extract_concepts(vi, "vi"), _extract_concepts(en, "en")) < 0.1:
        return True
    return False


def _detect_glossary_misplacement(
    cue_idx: int,
    source_texts: List[str],
    vi_texts: List[str],
    video_context: Optional[Dict[str, Any]],
    window: int = 4,
) -> Tuple[bool, str]:
    vi = vi_texts[cue_idx - 1].strip()
    if not vi:
        return False, ""
    vi_c = _extract_concepts(vi, "vi")
    for label, concepts, suggested_vi in _glossary_term_sets(video_context):
        if not (vi_c & concepts):
            continue
        if suggested_vi:
            if suggested_vi in vi.lower():
                pass  # strong glossary presence
            else:
                suggested_words = [w for w in suggested_vi.split() if len(w) >= 2]
                matches = sum(1 for w in suggested_words if w in vi.lower())
                min_matches = max(2, int(len(suggested_words) * 0.5))
                if matches < min_matches:
                    continue
        own_en = _extract_concepts(source_texts[cue_idx - 1], "en")
        if concepts & own_en:
            continue
        if any(
            o == c or o.rstrip("s") == c.rstrip("s")
            for o in own_en
            for c in concepts
            if len(o) >= 4 and len(c) >= 4
        ):
            continue
        for j in range(max(1, cue_idx - window), min(len(source_texts), cue_idx + window) + 1):
            if j == cue_idx:
                continue
            if concepts & _extract_concepts(source_texts[j - 1], "en"):
                return True, f"glossary concept '{label}' appears in VI cue {cue_idx} but EN cue {j}"
    return False, ""


def _glossary_entry_kind(source: str) -> str:
    words = [w for w in source.lower().split() if w]
    if len(words) >= 3:
        return "idiom"
    if len(words) == 2:
        return "phrase"
    if source.lower() in _DISCOURSE_CAPITALIZED:
        return "discourse_marker"
    return "term"


def _en_rhetorical_repeat(en_a: str, en_b: str) -> bool:
    """True when two EN cues repeat the same rhetorical idea (informal speech)."""
    ca = _extract_concepts(en_a, "en")
    cb = _extract_concepts(en_b, "en")
    if ca and cb:
        overlap = len(ca & cb) / min(len(ca), len(cb))
        if overlap >= 0.45:
            return True
    na = re.sub(r"\s+", " ", en_a.lower().strip())
    nb = re.sub(r"\s+", " ", en_b.lower().strip())
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= 10 and shorter in longer:
        return True
    return False


def _detect_duplicate_vi_misalignment(
    source_texts: List[str],
    vi_texts: List[str],
) -> Dict[int, str]:
    """Map cue_index -> reason for duplicate VI on different EN."""
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, vi in enumerate(vi_texts, start=1):
        key = re.sub(r"\s+", " ", vi.strip().lower())
        if len(key) >= 12:
            buckets[key].append(i)

    issues: Dict[int, str] = {}
    for _key, cues in buckets.items():
        if len(cues) < 2:
            continue
        rhetorical = True
        for a in range(len(cues)):
            for b in range(a + 1, len(cues)):
                if not _en_rhetorical_repeat(
                    source_texts[cues[a] - 1], source_texts[cues[b] - 1]
                ):
                    rhetorical = False
                    break
            if not rhetorical:
                break
        if rhetorical:
            continue
        en_sigs = [_extract_concepts(source_texts[c - 1], "en") for c in cues]
        if len({frozenset(s) for s in en_sigs}) > 1:
            for c in cues:
                issues[c] = f"duplicate VI text shared across cues {cues} with different EN"
    return issues


def analyze_semantic_alignment(
    source_entries: List[dict],
    vi_entries: List[dict],
    meaning_units: Optional[List[dict]] = None,
    video_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rule-first semantic alignment report per cue and meaning unit."""
    source_texts = [e.get("text", "") for e in source_entries]
    vi_texts = [e.get("text", "") for e in vi_entries]
    n = len(source_texts)

    duplicate_issues = _detect_duplicate_vi_misalignment(source_texts, vi_texts)
    cue_issues: List[dict] = []
    unit_issues: List[dict] = []
    alignment_error_cues: Set[int] = set()

    for i in range(1, n + 1):
        en = source_texts[i - 1].strip()
        vi = vi_texts[i - 1].strip()
        errors: List[str] = []
        reasons: List[str] = []
        confidence = 0.0

        if en and not vi.strip():
            errors.append("missing_or_empty_cue_error")
            reasons.append("non-empty source cue has empty VI")

        if en and vi:
            shifted, shift_reason, conf = _detect_cue_shift(
                i, source_texts, vi_texts, video_context
            )
            if shifted:
                errors.append("semantic_alignment_error")
                reasons.append(shift_reason)
                confidence = max(confidence, conf)

            missing, miss_reason = _detect_missing_source_concepts(
                i, source_texts, vi_texts, video_context, meaning_units
            )
            if missing:
                errors.append("semantic_drift_error")
                reasons.append(miss_reason)
                confidence = max(confidence, 0.7)

            if _detect_generic_vi_when_specific_en(en, vi):
                errors.append("semantic_alignment_error")
                reasons.append("VI too generic for specific EN cue")
                confidence = max(confidence, 0.65)

            if _detect_question_mismatch(en, vi):
                errors.append("semantic_alignment_error")
                reasons.append("question/statement mismatch between EN and VI")
                confidence = max(confidence, 0.6)

            gloss, gloss_reason = _detect_glossary_misplacement(
                i, source_texts, vi_texts, video_context
            )
            if gloss:
                errors.append("semantic_alignment_error")
                reasons.append(gloss_reason)
                confidence = max(confidence, 0.75)

            if i in duplicate_issues:
                errors.append("semantic_alignment_error")
                reasons.append(duplicate_issues[i])
                confidence = max(confidence, 0.6)

        if errors:
            alignment_error_cues.add(i)
            cue_issues.append(
                {
                    "cue_index": i,
                    "en": en,
                    "vi": vi,
                    "errors": sorted(set(errors)),
                    "reasons": reasons,
                    "confidence": round(confidence, 2),
                }
            )

    cue_issues, cue_warnings = _downgrade_to_unit_warnings(
        cue_issues, meaning_units, source_texts, vi_texts, video_context
    )

    if meaning_units:
        for unit in meaning_units:
            cues = unit.get("cue_indexes") or []
            unit_errors: Set[str] = set()
            unit_reasons: List[str] = []
            for c in cues:
                for issue in cue_issues + cue_warnings:
                    if issue["cue_index"] == c:
                        unit_errors.update(issue["errors"])
                        unit_reasons.extend(issue["reasons"])
            if unit_errors:
                unit_issues.append(
                    {
                        "unit_id": unit.get("unit_id"),
                        "cue_indexes": cues,
                        "errors": sorted(unit_errors),
                        "reasons": unit_reasons,
                        "unit_alignment_ok": _unit_alignment_ok(
                            cues, source_texts, vi_texts, video_context
                        )[0],
                    }
                )

    human_review = any(
        e in ("semantic_alignment_error", "semantic_drift_error")
        for i in cue_issues
        for e in i.get("errors", [])
    )

    return {
        "alignment_error_count": len(cue_issues),
        "alignment_warning_count": len(cue_warnings),
        "human_review_needed": human_review,
        "cue_issues": cue_issues,
        "cue_warnings": cue_warnings,
        "unit_issues": unit_issues,
        "summary": {
            "total_cues": n,
            "misaligned_cue_count": len(alignment_error_cues),
            "semantic_alignment_errors": sum(
                1 for i in cue_issues if "semantic_alignment_error" in i["errors"]
            ),
            "semantic_drift_errors": sum(
                1 for i in cue_issues if "semantic_drift_error" in i["errors"]
            ),
            "cue_alignment_warnings": len(cue_warnings),
        },
    }


def parse_repair_response_structured(content: str) -> List[dict]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    data = json.loads(content)
    return data.get("units") or []


def validate_repair_contract(
    repair_units: List[dict],
    expected_units: List[dict],
    source_entries: List[dict],
) -> Dict[str, Any]:
    """Validate repair JSON against unit/cue contract."""
    expected_by_id = {u["unit_id"]: u for u in expected_units}
    violations: List[dict] = []
    per_cue: Dict[int, dict] = {}
    per_unit: Dict[int, dict] = {}

    for expected in expected_units:
        uid = expected["unit_id"]
        per_unit[uid] = {"unit_id": uid, "valid": True, "violations": []}

    for unit in repair_units:
        uid = unit.get("unit_id")
        cues = unit.get("cues") or []
        expected = expected_by_id.get(uid)
        if expected is None:
            violations.append({"unit_id": uid, "error": "unexpected_unit_id"})
            if isinstance(uid, int):
                per_unit.setdefault(uid, {"unit_id": uid, "valid": False, "violations": []})
                per_unit[uid]["valid"] = False
                per_unit[uid]["violations"].append("unexpected_unit_id")
            continue

        expected_indexes = expected.get("cue_indexes") or []
        returned_indexes = [c.get("cue_index") for c in cues if isinstance(c.get("cue_index"), int)]
        unit_invalid = False

        if len(returned_indexes) != len(expected_indexes):
            unit_invalid = True
            violations.append(
                {
                    "unit_id": uid,
                    "error": "cue_count_mismatch",
                    "expected": expected_indexes,
                    "got": returned_indexes,
                }
            )

        if sorted(returned_indexes) != sorted(expected_indexes):
            unit_invalid = True
            violations.append(
                {
                    "unit_id": uid,
                    "error": "cue_index_mismatch",
                    "expected": expected_indexes,
                    "got": returned_indexes,
                }
            )

        returned_set = {c.get("cue_index") for c in cues if isinstance(c.get("cue_index"), int)}
        for idx in expected_indexes:
            if idx not in returned_set:
                unit_invalid = True
                violations.append(
                    {"unit_id": uid, "cue_index": idx, "error": "missing_cue_in_repair"}
                )
                per_cue[idx] = {
                    "unit_id": uid,
                    "cue_index": idx,
                    "text": "",
                    "valid": False,
                    "violations": ["repair_contract_violation:missing_cue"],
                }

        for cue in cues:
            idx = cue.get("cue_index")
            text = str(cue.get("text", "")).strip()
            if not isinstance(idx, int) or idx < 1 or idx > len(source_entries):
                unit_invalid = True
                violations.append({"unit_id": uid, "cue_index": idx, "error": "invalid_cue_index"})
                continue
            src = source_entries[idx - 1].get("text", "").strip()
            entry = {"unit_id": uid, "cue_index": idx, "text": text, "valid": True, "violations": []}
            if src and not text:
                unit_invalid = True
                entry["valid"] = False
                entry["violations"].append("repair_contract_violation:empty_text")
                violations.append(
                    {"unit_id": uid, "cue_index": idx, "error": "empty_text_for_non_empty_source"}
                )
            per_cue[idx] = entry

        if unit_invalid and uid in per_unit:
            per_unit[uid]["valid"] = False
            per_unit[uid]["violations"] = [
                v.get("error", "") for v in violations if v.get("unit_id") == uid
            ]

    invalid_unit_ids = [uid for uid, u in per_unit.items() if not u.get("valid", True)]

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "per_cue": per_cue,
        "per_unit": per_unit,
        "invalid_unit_ids": invalid_unit_ids,
    }


def _alignment_issues_for_cue(
    cue_idx: int,
    source_entries: List[dict],
    vi_entries: List[dict],
    meaning_units: Optional[List[dict]],
    video_context: Optional[Dict[str, Any]],
) -> int:
    report = analyze_semantic_alignment(
        source_entries, vi_entries, meaning_units, video_context
    )
    for issue in report.get("cue_issues", []):
        if issue["cue_index"] == cue_idx:
            return len(issue.get("errors", []))
    return 0


def apply_validated_repairs(
    source_entries: List[dict],
    vi_entries: List[dict],
    repair_units: List[dict],
    expected_units: List[dict],
    meaning_units: Optional[List[dict]],
    video_context: Optional[Dict[str, Any]],
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Validate contract + semantic alignment; apply only improvements.
  """
    source_texts = [e.get("text", "") for e in source_entries]
    vi_texts = [e.get("text", "") for e in vi_entries]
    contract = validate_repair_contract(repair_units, expected_units, source_entries)
    invalid_units = set(contract.get("invalid_unit_ids") or [])

    accepted: Dict[int, dict] = {}
    rejected: Dict[int, dict] = {}
    rejected_units: Dict[int, dict] = {}
    result = list(vi_entries)
    vi_texts = [e.get("text", "") for e in vi_entries]

    expected_by_id = {u["unit_id"]: u for u in expected_units}

    for uid in invalid_units:
        expected = expected_by_id.get(uid)
        if not expected:
            continue
        rejected_units[uid] = {
            "reason": "repair_contract_violation",
            "unit_id": uid,
            "cue_indexes": expected.get("cue_indexes", []),
            "violations": contract.get("per_unit", {}).get(uid, {}).get("violations", []),
        }
        for idx in expected.get("cue_indexes") or []:
            rejected[idx] = {
                "reason": "repair_contract_violation_unit_rejected",
                "proposed": contract.get("per_cue", {}).get(idx, {}).get("text", ""),
                "kept": vi_texts[idx - 1].strip(),
                "unit_id": uid,
            }

    for unit in repair_units:
        uid = unit.get("unit_id")
        if uid not in expected_by_id or uid in invalid_units:
            continue
        for cue in unit.get("cues") or []:
            idx = cue.get("cue_index")
            new_text = str(cue.get("text", "")).strip()
            if not isinstance(idx, int):
                continue
            src = source_entries[idx - 1].get("text", "").strip()
            old_text = vi_texts[idx - 1].strip()

            cue_contract = contract.get("per_cue", {}).get(idx, {})
            if not cue_contract.get("valid", True) or (src and not new_text):
                rejected[idx] = {
                    "reason": "repair_contract_violation",
                    "proposed": new_text,
                    "kept": old_text,
                }
                continue

            if not new_text or new_text == old_text:
                continue

            old_err = _alignment_issues_for_cue(
                idx, source_entries, vi_entries, meaning_units, video_context
            )
            trial_entries = list(result)
            trial_entries[idx - 1] = {**trial_entries[idx - 1], "text": new_text}
            new_err = _alignment_issues_for_cue(
                idx, source_entries, trial_entries, meaning_units, video_context
            )
            bridges = _glossary_bridges(video_context)
            en_text = source_texts[idx - 1]
            old_score = _association_score(old_text, en_text, bridges)
            new_score = _association_score(new_text, en_text, bridges)
            new_overlap = _overlap_ratio(
                _extract_concepts(new_text, "vi"),
                _extract_concepts(en_text, "en"),
            )

            if new_err > old_err:
                rejected[idx] = {
                    "reason": "semantic_alignment_worse",
                    "proposed": new_text,
                    "kept": old_text,
                }
                continue

            if old_err > 0 and new_err < old_err:
                result[idx - 1] = {**result[idx - 1], "text": new_text}
                vi_texts[idx - 1] = new_text
                accepted[idx] = {"before": old_text, "after": new_text}
                continue

            if new_score >= old_score + 2.0 and new_err <= old_err:
                result[idx - 1] = {**result[idx - 1], "text": new_text}
                vi_texts[idx - 1] = new_text
                accepted[idx] = {"before": old_text, "after": new_text}
                continue

            if old_err > 0 and new_err >= old_err and new_overlap < 0.15 and new_score <= old_score:
                rejected[idx] = {
                    "reason": "repair_did_not_improve_alignment",
                    "proposed": new_text,
                    "kept": old_text,
                }
                continue

            result[idx - 1] = {**result[idx - 1], "text": new_text}
            vi_texts[idx - 1] = new_text
            accepted[idx] = {"before": old_text, "after": new_text}

    for idx_key, cue_contract in contract.get("per_cue", {}).items():
        idx = int(idx_key) if isinstance(idx_key, str) else idx_key
        if cue_contract.get("valid", True) or idx in rejected:
            continue
        old_text = vi_texts[idx - 1].strip()
        rejected[idx] = {
            "reason": "repair_contract_violation",
            "proposed": cue_contract.get("text", ""),
            "kept": old_text,
            "violations": cue_contract.get("violations", []),
        }

    meta = {
        "contract": contract,
        "accepted": accepted,
        "rejected": rejected,
        "rejected_units": rejected_units,
        "applied": bool(accepted),
    }
    return result, meta


def save_alignment_report(path: str | Path, report: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
