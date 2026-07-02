"""Span-guarded raw translation — grouped baseline + span-level alignment guard."""

from __future__ import annotations

import os
from typing import List, Optional

from .raw_span_alignment_guard import span_guard_and_repair


def translate_srt_entries_span_guarded_openai(
    entries: List[dict],
    target_lang: str = "vi",
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    topic: Optional[str] = None,
    polish: Optional[bool] = None,
    translation_context: Optional[dict] = None,
    *,
    strict_cue_count: bool = False,
    sample_id: Optional[str] = None,
    conservative: bool = False,
) -> List[dict]:
    from .openai_translate import translate_srt_entries_openai

    prev_mode = os.environ.get("RAW_TRANSLATION_MODE")
    os.environ["RAW_TRANSLATION_MODE"] = "grouped"
    try:
        baseline = translate_srt_entries_openai(
            entries,
            target_lang=target_lang,
            model=model,
            batch_size=batch_size,
            topic=topic,
            polish=polish,
            translation_context=translation_context,
            strict_cue_count=strict_cue_count,
        )
    finally:
        if prev_mode is None:
            os.environ.pop("RAW_TRANSLATION_MODE", None)
        else:
            os.environ["RAW_TRANSLATION_MODE"] = prev_mode

    repaired, _report = span_guard_and_repair(
        entries,
        baseline,
        topic=topic,
        sample_id=sample_id,
        conservative=conservative,
    )
    return repaired


def translate_srt_entries_span_guarded_conservative_openai(
    entries: List[dict],
    target_lang: str = "vi",
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    topic: Optional[str] = None,
    polish: Optional[bool] = None,
    translation_context: Optional[dict] = None,
    *,
    strict_cue_count: bool = False,
    sample_id: Optional[str] = None,
) -> List[dict]:
    return translate_srt_entries_span_guarded_openai(
        entries,
        target_lang=target_lang,
        model=model,
        batch_size=batch_size,
        topic=topic,
        polish=polish,
        translation_context=translation_context,
        strict_cue_count=strict_cue_count,
        sample_id=sample_id,
        conservative=True,
    )


def translate_srt_entries_span_guarded_tiered_openai(
    entries: List[dict],
    target_lang: str = "vi",
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    topic: Optional[str] = None,
    polish: Optional[bool] = None,
    translation_context: Optional[dict] = None,
    *,
    strict_cue_count: bool = False,
    sample_id: Optional[str] = None,
) -> List[dict]:
    from .openai_translate import translate_srt_entries_openai
    from .raw_span_tiered_guard import span_tiered_guard_and_repair

    prev_mode = os.environ.get("RAW_TRANSLATION_MODE")
    os.environ["RAW_TRANSLATION_MODE"] = "grouped"
    try:
        baseline = translate_srt_entries_openai(
            entries,
            target_lang=target_lang,
            model=model,
            batch_size=batch_size,
            topic=topic,
            polish=polish,
            translation_context=translation_context,
            strict_cue_count=strict_cue_count,
        )
    finally:
        if prev_mode is None:
            os.environ.pop("RAW_TRANSLATION_MODE", None)
        else:
            os.environ["RAW_TRANSLATION_MODE"] = prev_mode

    repaired, _report = span_tiered_guard_and_repair(
        entries,
        baseline,
        topic=topic,
        sample_id=sample_id,
    )
    return repaired
