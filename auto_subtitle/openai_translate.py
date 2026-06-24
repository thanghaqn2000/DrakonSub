import json
import os
import re
from typing import Dict, List, Optional

from .config import (
    get_openai_model,
    get_phrase_group_max_cues,
    get_translation_batch_size,
    translation_polish_enabled,
)
from .translation_topics import build_polish_system_prompt, build_system_prompt, normalize_topic
from .openai_chat import create_chat_completion

# Punctuation that signals the end of a sentence / phrase group.
_SENTENCE_END_CHARS = frozenset(".!?…")


# ---------------------------------------------------------------------------
# Phrase grouping helpers
# ---------------------------------------------------------------------------

def _group_cues_by_sentence(
    texts: List[str],
    max_cues_per_group: int = 6,
) -> List[List[int]]:
    """
    Partition 0-based cue indices into phrase groups.

    A new group is started when the current cue ends with sentence-ending
    punctuation, or when the group would exceed *max_cues_per_group*.

    Returns a list of groups; each group is an ordered list of indices
    into *texts*.
    """
    groups: List[List[int]] = []
    current: List[int] = []

    for i, text in enumerate(texts):
        current.append(i)
        last_char = text.strip()[-1:] if text.strip() else ""
        if last_char in _SENTENCE_END_CHARS or len(current) >= max_cues_per_group:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


def _pack_groups_into_batches(
    groups: List[List[int]],
    batch_size: int,
) -> List[List[List[int]]]:
    """
    Pack phrase groups into API-call batches so total cues per batch
    does not exceed *batch_size*.

    A single group that is larger than *batch_size* is kept as-is in its
    own batch (over-size edge case).
    """
    batches: List[List[List[int]]] = []
    current_batch: List[List[int]] = []
    current_count = 0

    for group in groups:
        if current_count + len(group) > batch_size and current_batch:
            batches.append(current_batch)
            current_batch = [group]
            current_count = len(group)
        else:
            current_batch.append(group)
            current_count += len(group)

    if current_batch:
        batches.append(current_batch)

    return batches


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_grouped_user_prompt(
    batch_groups: List[List[int]],
    all_texts: List[str],
    target_lang: str,
) -> str:
    """
    Build the user prompt for a batch of phrase groups.

    Groups are shown as labelled sections so the model can read each group
    as a coherent thought before translating its individual cues.
    """
    total_cues = sum(len(g) for g in batch_groups)
    n_groups = len(batch_groups)

    header = (
        f"You receive {total_cues} English subtitle cue{'s' if total_cues != 1 else ''} "
        f"from a video, organised into {n_groups} phrase group{'s' if n_groups != 1 else ''}.\n\n"
        "Read each group as a COMPLETE THOUGHT before translating. "
        "Use the full group context to write natural, idiomatic Vietnamese subtitles "
        "— NOT word-by-word translation.\n\n"
        "Rules:\n"
        "- Output exactly 1 Vietnamese line per input cue, same order\n"
        "- Do NOT merge, skip, or reorder cues\n"
        "- Vietnamese must flow naturally as a subtitle read on screen\n"
        "- Each line must be self-contained and readable on its own\n"
    )

    group_lines: List[str] = []
    flat_idx = 1  # sequential [1][2][3] across entire batch

    for g_num, group in enumerate(batch_groups, 1):
        group_lines.append(f"\n--- Group {g_num} ({len(group)} cue{'s' if len(group) != 1 else ''}) ---")
        for local_idx in group:
            group_lines.append(f"[{flat_idx}] {all_texts[local_idx]}")
            flat_idx += 1

    footer = (
        f'\n\nRespond with JSON: {{"translations": ["...", ...]}} '
        f"containing exactly {total_cues} strings in the same order as input."
    )

    return header + "\n".join(group_lines) + footer


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


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# OpenAI callers
# ---------------------------------------------------------------------------

def _call_openai_translate_grouped(
    client,
    model: str,
    batch_groups: List[List[int]],
    all_texts: List[str],
    target_lang: str,
    topic: str,
) -> List[str]:
    """Translate a batch of phrase groups in a single API call."""
    total_cues = sum(len(g) for g in batch_groups)
    response = create_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": build_system_prompt(topic)},
            {
                "role": "user",
                "content": _build_grouped_user_prompt(batch_groups, all_texts, target_lang),
            },
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    return _parse_json_strings(
        response.choices[0].message.content or "",
        "translations",
        total_cues,
    )


def _call_openai_translate(
    client,
    model: str,
    segments: List[str],
    target_lang: str,
    topic: str,
) -> List[str]:
    """Legacy single-cue-list translator used as a last-resort fallback."""
    lines = [f"[{i + 1}] {text}" for i, text in enumerate(segments)]
    n = len(segments)
    user_prompt = (
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
    response = create_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": build_system_prompt(topic)},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    return _parse_json_strings(
        response.choices[0].message.content or "",
        "translations",
        n,
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


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

def _translate_batch_with_retry(
    client,
    model: str,
    batch_groups: List[List[int]],
    all_texts: List[str],
    target_lang: str,
    topic: str,
) -> List[str]:
    """
    Translate a batch of groups with cascading retry:
      1. Try the full batch in one call.
      2. On failure, retry each group individually.
      3. On a single-group failure, fall back to per-cue legacy translation.
    """
    try:
        return _call_openai_translate_grouped(
            client, model, batch_groups, all_texts, target_lang, topic
        )
    except Exception as exc:
        print(f"  Batch failed ({exc}), retrying group-by-group…")

    results: List[str] = []
    for group in batch_groups:
        try:
            group_results = _call_openai_translate_grouped(
                client, model, [group], all_texts, target_lang, topic
            )
            results.extend(group_results)
        except Exception as exc2:
            print(f"  Group failed ({exc2}), falling back to per-cue…")
            for local_idx in group:
                try:
                    per_cue = _call_openai_translate(
                        client, model, [all_texts[local_idx]], target_lang, topic
                    )
                    results.extend(per_cue)
                except Exception:
                    results.append(all_texts[local_idx])  # keep English on total failure

    return results


def _split_retry_batch(
    client,
    model: str,
    segments: List[str],
    call_fn,
    topic: str,
    target_lang: Optional[str] = None,
) -> List[str]:
    """Binary-split retry for the polish pass (operates on flat segment lists)."""
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


# ---------------------------------------------------------------------------
# Polish pass
# ---------------------------------------------------------------------------

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
            f"  Polishing segments {pending_indices[0] + 1}-{pending_indices[-1] + 1}…"
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


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

def _log_grouping_stats(
    groups: List[List[int]],
    all_texts: List[str],
) -> None:
    total = sum(len(g) for g in groups)
    avg = total / len(groups) if groups else 0.0
    print(f"\n[Phrase Grouping] Original cue count : {total}")
    print(f"[Phrase Grouping] Phrase group count  : {len(groups)}")
    print(f"[Phrase Grouping] Avg cues per group  : {avg:.1f}")

    sample_count = min(2, len(groups))
    for ex_i, group in enumerate(groups[:sample_count]):
        print(f"[Phrase Grouping] Example group {ex_i + 1} (pre-translation):")
        for local_idx in group:
            print(f"    [{local_idx + 1}] {all_texts[local_idx]}")


def _log_translation_examples(
    groups: List[List[int]],
    all_texts: List[str],
    non_empty_indices: List[int],
    translated_texts: List[str],
) -> None:
    sample_count = min(2, len(groups))
    print("\n[Phrase Grouping] Translation examples:")
    for ex_i, group in enumerate(groups[:sample_count]):
        print(f"  Group {ex_i + 1}:")
        for local_idx in group:
            orig_entry_idx = non_empty_indices[local_idx]
            print(f"    EN: {all_texts[local_idx]}")
            print(f"    VI: {translated_texts[orig_entry_idx]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    max_cues_per_group = get_phrase_group_max_cues()
    polish = translation_polish_enabled() if polish is None else polish

    # -----------------------------------------------------------------------
    # Separate non-empty entries so grouping ignores blank cues.
    # -----------------------------------------------------------------------
    all_entry_texts = [e["text"].strip() for e in entries]
    non_empty_indices: List[int] = [i for i, t in enumerate(all_entry_texts) if t]
    non_empty_texts: List[str] = [all_entry_texts[i] for i in non_empty_indices]

    if not non_empty_texts:
        return list(entries)

    # -----------------------------------------------------------------------
    # Phrase grouping + batching
    # -----------------------------------------------------------------------
    groups = _group_cues_by_sentence(non_empty_texts, max_cues_per_group=max_cues_per_group)
    _log_grouping_stats(groups, non_empty_texts)

    batches = _pack_groups_into_batches(groups, batch_size)

    # -----------------------------------------------------------------------
    # Translation loop
    # -----------------------------------------------------------------------
    translated_non_empty: Dict[int, str] = {}

    local_offset = 0
    for batch_groups in batches:
        batch_local_indices = [idx for g in batch_groups for idx in g]
        first_cue = batch_local_indices[0] + 1
        last_cue = batch_local_indices[-1] + 1
        print(
            f"  Translating cues {first_cue}-{last_cue} "
            f"({len(batch_groups)} group{'s' if len(batch_groups) != 1 else ''}, "
            f"{len(batch_local_indices)} cues)…"
        )

        results = _translate_batch_with_retry(
            client, model, batch_groups, non_empty_texts, target_lang, topic
        )

        for local_idx, vi_text in zip(batch_local_indices, results):
            translated_non_empty[local_idx] = vi_text

        local_offset += len(batch_local_indices)

    # -----------------------------------------------------------------------
    # Reconstruct full translated_texts list (entry-indexed)
    # -----------------------------------------------------------------------
    translated_texts = list(all_entry_texts)
    for local_idx, orig_entry_idx in enumerate(non_empty_indices):
        translated_texts[orig_entry_idx] = translated_non_empty.get(local_idx, "")

    _log_translation_examples(groups, non_empty_texts, non_empty_indices, translated_texts)

    # -----------------------------------------------------------------------
    # Optional polish pass
    # -----------------------------------------------------------------------
    if polish and any(t.strip() for t in translated_texts):
        print("  Running subtitle polish pass…")
        translated_texts = _polish_translations(
            client, model, translated_texts, topic, batch_size
        )

    return [
        {**entry, "text": translated_texts[i]}
        for i, entry in enumerate(entries)
    ]
