import json
import os
import re
from typing import List, Optional

from .translation_topics import build_system_prompt, normalize_topic


def _build_user_prompt(segments: List[str], target_lang: str) -> str:
    lines = [f"[{i + 1}] {text}" for i, text in enumerate(segments)]
    n = len(segments)
    return (
        f"Translate these {n} English subtitle segments to {target_lang}.\n\n"
        + "\n".join(lines)
        + f'\n\nRespond with JSON: {{"translations": ["...", ...]}} '
        f"containing exactly {n} strings in the same order."
    )


def _parse_translations(content: str, expected_count: int) -> List[str]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)
    translations = data.get("translations", data if isinstance(data, list) else None)
    if not isinstance(translations, list):
        raise ValueError("OpenAI response missing 'translations' array")

    if len(translations) != expected_count:
        raise ValueError(
            f"Expected {expected_count} translations, got {len(translations)}"
        )

    return [str(t).strip() for t in translations]


def _call_openai_translate(
    client,
    model: str,
    segments: List[str],
    target_lang: str,
    topic: str,
) -> List[str]:
    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": build_system_prompt(topic)},
            {"role": "user", "content": _build_user_prompt(segments, target_lang)},
        ],
    )
    return _parse_translations(
        response.choices[0].message.content or "",
        len(segments),
    )


def _translate_batch(
    client,
    model: str,
    segments: List[str],
    target_lang: str,
    topic: str,
) -> List[str]:
    try:
        return _call_openai_translate(client, model, segments, target_lang, topic)
    except ValueError:
        if len(segments) == 1:
            raise

        mid = len(segments) // 2
        return (
            _translate_batch(client, model, segments[:mid], target_lang, topic)
            + _translate_batch(client, model, segments[mid:], target_lang, topic)
        )


def translate_srt_entries_openai(
    entries: List[dict],
    target_lang: str = "vi",
    model: Optional[str] = None,
    batch_size: int = 15,
    topic: Optional[str] = None,
) -> List[dict]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to a .env file in the project directory."
        )

    topic = normalize_topic(topic or os.environ.get("TRANSLATION_TOPIC"))
    client = OpenAI(api_key=api_key)
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    pending_indices: List[int] = []
    pending_texts: List[str] = []

    def flush_batch() -> None:
        nonlocal pending_indices, pending_texts
        if not pending_texts:
            return

        print(f"  Translating segments {pending_indices[0] + 1}-{pending_indices[-1] + 1}...")
        batch_translations = _translate_batch(
            client, model, pending_texts, target_lang, topic
        )

        for idx, vi_text in zip(pending_indices, batch_translations):
            translated[idx] = {**entries[idx], "text": vi_text}

        pending_indices = []
        pending_texts = []

    translated = [None] * len(entries)

    for i, entry in enumerate(entries):
        text = entry["text"].strip()
        if not text:
            translated[i] = {**entry, "text": text}
            continue

        pending_indices.append(i)
        pending_texts.append(text)

        if len(pending_texts) >= batch_size:
            flush_batch()

    if pending_texts:
        flush_batch()

    return translated
