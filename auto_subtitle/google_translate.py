"""Google Translate backend via deep-translator (no API key; works from geo-blocked regions)."""

import time
from typing import List, Optional

from deep_translator import GoogleTranslator

_DEFAULT_BATCH_SIZE = 40
_BATCH_DELAY_SEC = 0.35


def _target_lang_code(target_lang: str) -> str:
    code = (target_lang or "vi").strip().lower()
    if code in ("vi", "vietnamese"):
        return "vi"
    return code


def _translate_texts(
    texts: List[str],
    *,
    source_lang: str = "en",
    target_lang: str = "vi",
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> List[str]:
    translator = GoogleTranslator(
        source=source_lang or "auto",
        target=_target_lang_code(target_lang),
    )
    results: List[str] = []
    pending_indices: List[int] = []
    pending_texts: List[str] = []

    def flush_batch() -> None:
        nonlocal pending_indices, pending_texts
        if not pending_texts:
            return
        try:
            translated = translator.translate_batch(pending_texts)
        except Exception:
            translated = [
                translator.translate(text) if text.strip() else text
                for text in pending_texts
            ]
        for idx, vi_text in zip(pending_indices, translated):
            results[idx] = (vi_text or "").strip()
        pending_indices = []
        pending_texts = []
        time.sleep(_BATCH_DELAY_SEC)

    results = [""] * len(texts)
    for i, text in enumerate(texts):
        stripped = text.strip()
        if not stripped:
            results[i] = text
            continue
        pending_indices.append(i)
        pending_texts.append(stripped)
        if len(pending_texts) >= batch_size:
            flush_batch()
    flush_batch()
    return results


def translate_srt_entries_google(
    entries: List[dict],
    target_lang: str = "vi",
    source_lang: str = "en",
    topic: Optional[str] = None,
    translation_context: Optional[dict] = None,
    *,
    strict_cue_count: bool = False,
) -> List[dict]:
    del topic, translation_context, strict_cue_count  # unused for free Google backend

    texts = [e.get("text", "") for e in entries]
    non_empty_positions = [i for i, t in enumerate(texts) if t.strip()]
    if not non_empty_positions:
        return [{**e} for e in entries]

    print(
        f"  [Google translate] {len(non_empty_positions)} cue(s) "
        f"({source_lang} → {target_lang})…"
    )
    source_texts = [texts[i] for i in non_empty_positions]
    translated_non_empty = _translate_texts(
        source_texts,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    translated_texts = list(texts)
    for pos, vi_text in zip(non_empty_positions, translated_non_empty):
        translated_texts[pos] = vi_text or texts[pos]

    return [
        {**entry, "text": translated_texts[i]}
        for i, entry in enumerate(entries)
    ]
