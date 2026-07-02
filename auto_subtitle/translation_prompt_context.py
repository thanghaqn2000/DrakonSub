"""Inject video context, glossary, and meaning-unit guidance into translation prompts."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def format_video_context_block(video_context: Optional[Dict[str, Any]]) -> str:
    if not video_context:
        return ""

    lines = [
        "=== VIDEO CONTEXT ===",
        f"Detected topic: {video_context.get('detected_topic', 'general')}",
        f"Audience: {video_context.get('audience_level', 'general_beginner')}",
        f"Style: {video_context.get('translation_style', 'simple natural Vietnamese subtitles')}",
        f"Tone: {video_context.get('tone', 'neutral')}",
    ]
    summary = video_context.get("short_summary", "").strip()
    if summary:
        lines.append(f"Summary: {summary}")

    warnings = video_context.get("translation_warnings") or []
    if warnings:
        lines.append("Translation warnings:")
        lines.extend(f"- {w}" for w in warnings[:8])

    asr_risks = video_context.get("possible_asr_risks") or []
    if asr_risks:
        lines.append("Possible ASR risks:")
        lines.extend(f"- {r}" for r in asr_risks[:6])

    return "\n".join(lines)


def format_glossary_block(video_context: Optional[Dict[str, Any]]) -> str:
    if not video_context:
        return ""

    key_terms = video_context.get("key_terms") or []
    if not key_terms:
        return ""

    lines = ["=== DYNAMIC GLOSSARY (from this video) ==="]
    for term in key_terms[:20]:
        source = term.get("source", "").strip()
        vi = term.get("suggested_vi", "").strip()
        explain = term.get("plain_explanation", "").strip()
        if not source:
            continue
        piece = f"- {source} → {vi}" if vi else f"- {source}"
        if explain:
            piece += f" ({explain})"
        lines.append(piece)
    return "\n".join(lines) if len(lines) > 1 else ""


def format_meaning_unit_guidance() -> str:
    return (
        "=== MEANING-UNIT TRANSLATION RULES ===\n"
        "- Translate the meaning of the whole unit, not isolated cue fragments.\n"
        "- Then distribute the Vietnamese meaning back to the original cue indexes.\n"
        "- Preserve cue count — one Vietnamese line per input cue.\n"
        "- Do not leave any non-empty source cue blank.\n"
        "- Prefer natural, easy Vietnamese for general adult viewers.\n"
        "- Avoid word-by-word translation when it makes Vietnamese unnatural.\n"
        "- Preserve correct meaning over literal English structure."
    )


def format_batch_meaning_units(
    batch_cue_indexes_1based: List[int],
    meaning_units: Optional[List[Dict[str, Any]]],
    source_texts_1based: Dict[int, str],
) -> str:
    """Describe meaning units overlapping this batch."""
    if not meaning_units:
        return ""

    batch_set = set(batch_cue_indexes_1based)
    lines = ["=== MEANING UNITS IN THIS BATCH ==="]
    for unit in meaning_units:
        cues = unit.get("cue_indexes") or []
        if not batch_set.intersection(cues):
            continue
        source = unit.get("source_text", "").strip()
        if not source:
            source = " / ".join(
                source_texts_1based.get(i, "") for i in cues if source_texts_1based.get(i)
            )
        lines.append(
            f"Unit {unit.get('unit_id')}: cues {cues}\n"
            f"  Source: {source}\n"
            f"  Reason: {unit.get('reason', '')}\n"
            f"  Risks: {', '.join(unit.get('source_risk_flags') or unit.get('risk_flags') or [])}"
        )
    return "\n".join(lines) if len(lines) > 1 else ""


def enrich_user_prompt(
    base_prompt: str,
    *,
    video_context: Optional[Dict[str, Any]] = None,
    meaning_units: Optional[List[Dict[str, Any]]] = None,
    batch_cue_indexes_1based: Optional[List[int]] = None,
    source_texts_1based: Optional[Dict[int, str]] = None,
) -> str:
    """Prepend general intelligence blocks to a translation/editor user prompt."""
    blocks = [
        format_video_context_block(video_context),
        format_glossary_block(video_context),
        format_meaning_unit_guidance(),
    ]
    if meaning_units and batch_cue_indexes_1based and source_texts_1based:
        blocks.append(
            format_batch_meaning_units(
                batch_cue_indexes_1based,
                meaning_units,
                source_texts_1based,
            )
        )
    prefix = "\n\n".join(b for b in blocks if b.strip())
    if not prefix:
        return base_prompt
    return f"{prefix}\n\n{base_prompt}"


def video_context_to_json(video_context: Dict[str, Any]) -> str:
    return json.dumps(video_context, ensure_ascii=False, indent=2)
