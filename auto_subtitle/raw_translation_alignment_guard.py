"""Post-raw translation alignment guard — detect neighbor bleed and length anomalies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .semantic_alignment_guard import _extract_concepts, _overlap_ratio

_FLAG_LENGTH = "raw_alignment_length_anomaly"
_FLAG_NEIGHBOR = "raw_neighbor_bleed_suspected"
_FLAG_GENERIC = "raw_generic_translation"
_FLAG_HALLUCINATION = "raw_hallucinated_context"

_GENERIC_VI = re.compile(
    r"(?i)^(điều này|chuyện đó|làm gì đó|như vậy|tóm lại|nói chung)"
)


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w']+", text, flags=re.UNICODE))


def _neighbor_bleed_score(
    vi: str,
    en: str,
    prev_en: str,
    next_en: str,
) -> Tuple[float, float]:
    vi_c = _extract_concepts(vi, "vi")
    if not vi_c:
        return 0.0, 0.0
    cur = _overlap_ratio(vi_c, _extract_concepts(en, "en"))
    prev = _overlap_ratio(vi_c, _extract_concepts(prev_en, "en")) if prev_en else 0.0
    nxt = _overlap_ratio(vi_c, _extract_concepts(next_en, "en")) if next_en else 0.0
    return max(prev, nxt), cur


def analyze_raw_alignment(
    source_entries: List[dict],
    vi_entries: List[dict],
) -> Dict[str, Any]:
    """Run heuristic checks on raw VI vs source EN per cue."""
    flags: List[dict] = []
    n = min(len(source_entries), len(vi_entries))

    for i in range(n):
        en = source_entries[i].get("text", "").strip()
        vi = vi_entries[i].get("text", "").strip()
        cue_index = i + 1
        if not en:
            continue

        cue_flags: List[str] = []
        prev_en = source_entries[i - 1].get("text", "").strip() if i > 0 else ""
        next_en = source_entries[i + 1].get("text", "").strip() if i + 1 < n else ""

        en_wc = _word_count(en)
        vi_wc = _word_count(vi)
        if en_wc <= 3 and vi_wc >= 8:
            cue_flags.append(_FLAG_LENGTH)
        elif en_wc > 0 and vi_wc / en_wc >= 4.0 and en_wc <= 5:
            cue_flags.append(_FLAG_LENGTH)

        neighbor_max, cur_overlap = _neighbor_bleed_score(vi, en, prev_en, next_en)
        if neighbor_max >= 0.45 and neighbor_max > cur_overlap + 0.2:
            cue_flags.append(_FLAG_NEIGHBOR)

        if _GENERIC_VI.match(vi) and en_wc >= 4:
            cue_flags.append(_FLAG_GENERIC)

        # Skip cross-language concept overlap — EN/VI token sets rarely intersect.

        if cue_flags:
            flags.append(
                {
                    "cue_index": cue_index,
                    "en": en,
                    "vi": vi,
                    "flags": sorted(set(cue_flags)),
                    "recommended_fix_layer": "raw_translation",
                }
            )

    return {
        "cue_count": n,
        "flagged_cue_count": len(flags),
        "flags": flags,
    }


def repair_flagged_raw_cues(
    source_entries: List[dict],
    vi_entries: List[dict],
    report: Dict[str, Any],
    *,
    target_lang: str = "vi",
    topic: Optional[str] = None,
) -> Tuple[List[dict], List[dict]]:
    """Re-translate flagged cues one at a time via cue_keyed repair."""
    import os

    from openai import OpenAI

    from .raw_cue_keyed_translate import translate_single_cue_keyed
    from .translation_topics import normalize_topic
    from .config import get_openai_model

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return vi_entries, []

    client = OpenAI(api_key=api_key)
    model = get_openai_model()
    topic = normalize_topic(topic or os.environ.get("TRANSLATION_TOPIC"))

    working = [dict(e) for e in vi_entries]
    repairs: List[dict] = []

    repair_flags = {_FLAG_LENGTH, _FLAG_NEIGHBOR}
    for item in report.get("flags", []):
        cue_index = item["cue_index"]
        if not set(item.get("flags", [])) & repair_flags:
            continue
        entry_idx = cue_index - 1
        if entry_idx < 0 or entry_idx >= len(working):
            continue
        before = working[entry_idx].get("text", "").strip()
        try:
            after = translate_single_cue_keyed(
                client,
                model,
                source_entries,
                entry_idx,
                target_lang,
                topic,
            )
        except Exception as exc:
            repairs.append(
                {
                    "cue_index": cue_index,
                    "reason": item.get("flags", []),
                    "before": before,
                    "after": before,
                    "error": str(exc),
                }
            )
            continue
        if after and after != before:
            working[entry_idx] = {**working[entry_idx], "text": after}
        repairs.append(
            {
                "cue_index": cue_index,
                "reason": item.get("flags", []),
                "before": before,
                "after": after,
            }
        )

    return working, repairs


def guard_and_repair_raw_translations(
    source_entries: List[dict],
    vi_entries: List[dict],
    *,
    topic: Optional[str] = None,
    debug_dir: Optional[str] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Analyze raw VI, repair flagged cues, return updated entries + report."""
    report = analyze_raw_alignment(source_entries, vi_entries)
    repaired_entries, repair_log = repair_flagged_raw_cues(
        source_entries,
        vi_entries,
        report,
        topic=topic,
    )
    after = analyze_raw_alignment(source_entries, repaired_entries)

    full_report = {
        "before": report,
        "after": after,
        "repairs": repair_log,
        "repair_applied_count": sum(
            1 for r in repair_log if r.get("before") != r.get("after")
        ),
    }

    if debug_dir:
        out = Path(debug_dir) / "raw_alignment_guard_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts_dir = Path("artifacts/translation_quality_review")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "raw_alignment_guard_report.json").write_text(
        json.dumps(full_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return repaired_entries, full_report


def save_raw_alignment_guard_report(path: str | Path, report: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
