import json
import os
import re
from typing import List, Optional

from .config import get_openai_model, get_translation_batch_size, translation_polish_enabled
from .translation_topics import build_polish_system_prompt, build_system_prompt, normalize_topic
from .openai_chat import create_chat_completion


def _build_translate_user_prompt(segments: List[str], target_lang: str) -> str:
    lines = [f"[{i + 1}] {text}" for i, text in enumerate(segments)]
    n = len(segments)
    return (
        f"You receive {n} English subtitle lines from a video.\n\n"
        "Your task is NOT word-by-word translation. Create natural, easy-to-understand "
        f"{target_lang} subtitles that Vietnamese viewers would actually read on screen.\n\n"
        "Read all lines together for context, then rewrite each line naturally. "
        "You may change sentence structure so Vietnamese flows smoothly. "
        "Keep one output string per input line — same count, same order.\n\n"
        + "\n".join(lines)
        + f'\n\nRespond with JSON: {{"translations": ["...", ...]}} '
        f"containing exactly {n} strings in the same order."
    )


def _build_polish_user_prompt(segments: List[str]) -> str:
    lines = [f"[{i + 1}] {text}" for i, text in enumerate(segments)]
    n = len(segments)
    return (
        f"Polish these {n} Vietnamese subtitle lines.\n\n"
        "Remove stiff or machine-translated phrasing. Make each line shorter and more natural "
        "where possible, while keeping the exact meaning. "
        "One polished string per input line — same count, same order.\n\n"
        + "\n".join(lines)
        + f'\n\nRespond with JSON: {{"subtitles": ["...", ...]}} '
        f"containing exactly {n} strings in the same order."
    )


def _parse_json_strings(content: str, key: str, expected_count: int) -> List[str]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)
    values = data.get(key, data if isinstance(data, list) else None)
    if not isinstance(values, list):
        raise ValueError(f"OpenAI response missing '{key}' array")

    if len(values) != expected_count:
        raise ValueError(f"Expected {expected_count} strings, got {len(values)}")

    return [str(v).strip() for v in values]


def _call_openai_translate(
    client,
    model: str,
    segments: List[str],
    target_lang: str,
    topic: str,
) -> List[str]:
    response = create_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": build_system_prompt(topic)},
            {"role": "user", "content": _build_translate_user_prompt(segments, target_lang)},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    return _parse_json_strings(
        response.choices[0].message.content or "",
        "translations",
        len(segments),
    )


def _call_openai_polish(
    client,
    model: str,
    segments: List[str],
    topic: str,
) -> List[str]:
    response = create_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": build_polish_system_prompt(topic)},
            {"role": "user", "content": _build_polish_user_prompt(segments)},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return _parse_json_strings(
        response.choices[0].message.content or "",
        "subtitles",
        len(segments),
    )


def _split_retry_batch(
    client,
    model: str,
    segments: List[str],
    call_fn,
    topic: str,
    target_lang: Optional[str] = None,
) -> List[str]:
    try:
        if target_lang is not None:
            return call_fn(client, model, segments, target_lang, topic)
        return call_fn(client, model, segments, topic)
    except ValueError:
        if len(segments) == 1:
            raise
        mid = len(segments) // 2
        first = _split_retry_batch(
            client, model, segments[:mid], call_fn, topic, target_lang
        )
        second = _split_retry_batch(
            client, model, segments[mid:], call_fn, topic, target_lang
        )
        return first + second


def _polish_translations(
    client,
    model: str,
    texts: List[str],
    topic: str,
    batch_size: int,
) -> List[str]:
    polished: List[Optional[str]] = [None] * len(texts)
    pending_indices: List[int] = []
    pending_texts: List[str] = []

    def flush_batch() -> None:
        nonlocal pending_indices, pending_texts
        if not pending_texts:
            return

        print(
            f"  Polishing segments {pending_indices[0] + 1}-{pending_indices[-1] + 1}..."
        )
        batch_polished = _split_retry_batch(
            client, model, pending_texts, _call_openai_polish, topic
        )
        for idx, text in zip(pending_indices, batch_polished):
            polished[idx] = text
        pending_indices = []
        pending_texts = []

    for i, text in enumerate(texts):
        if not text.strip():
            polished[i] = text
            continue
        pending_indices.append(i)
        pending_texts.append(text)
        if len(pending_texts) >= batch_size:
            flush_batch()

    if pending_texts:
        flush_batch()

    return [t if t is not None else "" for t in polished]


def translate_srt_entries_openai(
    entries: List[dict],
    target_lang: str = "vi",
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    topic: Optional[str] = None,
    polish: Optional[bool] = None,
) -> List[dict]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to a .env file in the project directory."
        )

    topic = normalize_topic(topic or os.environ.get("TRANSLATION_TOPIC"))
    client = OpenAI(api_key=api_key)
    model = model or get_openai_model()
    batch_size = batch_size or get_translation_batch_size()
    polish = translation_polish_enabled() if polish is None else polish

    pending_indices: List[int] = []
    pending_texts: List[str] = []
    translated_texts: List[Optional[str]] = [None] * len(entries)

    def flush_translate_batch() -> None:
        nonlocal pending_indices, pending_texts
        if not pending_texts:
            return

        print(
            f"  Translating segments {pending_indices[0] + 1}-{pending_indices[-1] + 1}..."
        )
        batch_translations = _split_retry_batch(
            client,
            model,
            pending_texts,
            _call_openai_translate,
            topic,
            target_lang,
        )
        for idx, vi_text in zip(pending_indices, batch_translations):
            translated_texts[idx] = vi_text
        pending_indices = []
        pending_texts = []

    for i, entry in enumerate(entries):
        text = entry["text"].strip()
        if not text:
            translated_texts[i] = text
            continue

        pending_indices.append(i)
        pending_texts.append(text)

        if len(pending_texts) >= batch_size:
            flush_translate_batch()

    if pending_texts:
        flush_translate_batch()

    final_texts = [t if t is not None else "" for t in translated_texts]

    if polish and any(t.strip() for t in final_texts):
        print("  Running subtitle polish pass...")
        final_texts = _polish_translations(
            client, model, final_texts, topic, batch_size
        )

    return [
        {**entry, "text": final_texts[i]}
        for i, entry in enumerate(entries)
    ]
