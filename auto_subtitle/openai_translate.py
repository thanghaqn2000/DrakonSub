import json
import os
import re
from typing import Dict, List, Optional

from .config import (
    get_openai_model,
    get_phrase_group_max_cues,
    get_raw_translation_mode,
    get_translation_batch_size,
    translation_polish_enabled,
)
from .translation_topics import TOPICS, build_polish_system_prompt, build_system_prompt, normalize_topic
from .openai_chat import create_chat_completion
from .raw_llm_response_cache import raw_llm_complete
from .translation_prompt_context import enrich_user_prompt

# Punctuation that signals the end of a sentence / phrase group.
_SENTENCE_END_CHARS = frozenset(".!?…")

_CONTEXT_WINDOW = 5

_OPENAI_RAW_LOCALIZATION_SYSTEM = """You are a professional Vietnamese subtitle localization editor.

Your job is not to translate words.
Your job is to rewrite the speaker's meaning into natural Vietnamese subtitles.

Audience: Vietnamese Facebook/Reels/TikTok viewers.

Style:
- Very easy to understand.
- Natural spoken Vietnamese.
- Concise enough for subtitles.
- Avoid robotic or literal translation.
- Preserve the speaker's intent and tone.
- Do not add new ideas or remove important ideas.
- Keep names, numbers, and financial terms accurate.

Important:
- Read neighboring cues for context before translating each line.
- Use context to resolve pronouns, fragments, and unfinished clauses.
- Return exactly one Vietnamese subtitle per input cue.
- Same count, same order.
- Do not merge, split, skip, or leave any non-empty source cue blank.
- Return JSON only."""

_REPAIR_USER_SUFFIX = (
    "\n\nREPAIR MODE: Your previous response failed validation "
    "(wrong count, missing index, or empty text for a non-empty source cue). "
    "Return exactly one non-empty Vietnamese subtitle for every non-empty source cue."
)

_STRICT_CUE_COUNT_SUFFIX = (
    "\n\nSTRICT CUE COUNT MODE: The output must contain exactly the same number "
    "of cues as the input. Return one Vietnamese subtitle per input cue index. "
    "Do not merge, split, skip, or drop any cue."
)


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
        # Strip trailing closing punctuation (quotes, brackets) so that cues
        # like  'like stocks."'  or  'like stocks.)'  still close the group.
        inner = text.strip().rstrip("\"')}]")
        last_char = inner[-1:] if inner else ""
        if last_char in _SENTENCE_END_CHARS or len(current) >= max_cues_per_group:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


from .meaning_unit_builder import resolve_translation_groups


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

def _parse_ts(ts: str) -> float:
    ts = ts.strip()
    time_part, millis_str = ts.split(",")
    h, m, s = time_part.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(millis_str) / 1000.0


def _build_openai_raw_system_prompt(topic: str) -> str:
    topic_id = normalize_topic(topic)
    topic_def = TOPICS[topic_id]
    parts = [
        _OPENAI_RAW_LOCALIZATION_SYSTEM,
        f"Context topic/tone: {topic_def.label}",
        topic_def.guidance,
    ]
    if topic_def.glossary:
        parts.append(topic_def.glossary)
    return "\n\n".join(parts)


def _build_context_user_prompt(
    batch_groups: List[List[int]],
    all_texts: List[str],
    target_lang: str,
    durations: Optional[List[float]] = None,
    *,
    repair: bool = False,
    strict: bool = False,
    translation_context: Optional[dict] = None,
    non_empty_indices: Optional[List[int]] = None,
) -> str:
    """Build user prompt with ±5 cue English context (Gemini-parity experiment)."""
    batch_local_indices = [idx for group in batch_groups for idx in group]
    total_cues = len(batch_local_indices)
    first_idx = batch_local_indices[0]
    last_idx = batch_local_indices[-1]

    prev_start = max(0, first_idx - _CONTEXT_WINDOW)
    previous_context = all_texts[prev_start:first_idx]
    next_context = all_texts[last_idx + 1 : last_idx + 1 + _CONTEXT_WINDOW]

    previous_lines = (
        "\n".join(
            f"- [{prev_start + offset + 1}] {line}"
            for offset, line in enumerate(previous_context)
        )
        if previous_context
        else "- (none)"
    )
    next_lines = (
        "\n".join(
            f"- [{last_idx + offset + 2}] {line}"
            for offset, line in enumerate(next_context)
        )
        if next_context
        else "- (none)"
    )

    current_lines: List[str] = []
    for local_i, idx in enumerate(batch_local_indices, start=1):
        parts = [f"[{local_i}] global_cue={idx + 1}"]
        if durations is not None and idx < len(durations):
            parts.append(f"duration={durations[idx]:.2f}s")
        parts.append(f"EN: {all_texts[idx]}")
        current_lines.append(" | ".join(parts))

    prompt = (
        f"Translate these {total_cues} English subtitle cues into natural {target_lang}.\n\n"
        "Instructions:\n"
        f"- Translate for Vietnamese short-form subtitles.\n"
        "- Preserve the main meaning, not every English word.\n"
        "- Use natural spoken Vietnamese.\n"
        "- Keep text short enough for subtitles.\n"
        "- Use neighboring cues to resolve pronouns, fragments, and unfinished clauses.\n"
        "- Do not merge cues.\n"
        "- Do not leave any non-empty source cue blank.\n"
        "- Return exactly one Vietnamese text for each input cue.\n\n"
        f"previous_context (up to {_CONTEXT_WINDOW} English cues before current batch):\n"
        f"{previous_lines}\n\n"
        "current_batch (translate ONLY these cues):\n"
        + "\n".join(current_lines)
        + f"\n\nnext_context (up to {_CONTEXT_WINDOW} English cues after current batch):\n"
        f"{next_lines}\n\n"
        "Output rules:\n"
        f"- Return exactly {total_cues} objects in current_batch order\n"
        '- JSON format: {"items": [{"index": 1, "text": "..."}, ...]}\n'
        "- index must be 1..N matching current_batch order\n"
        "- text must not be empty when the source EN cue is not empty"
    )
    if repair:
        prompt += _REPAIR_USER_SUFFIX
    if strict:
        prompt += _STRICT_CUE_COUNT_SUFFIX

    ctx = translation_context or {}
    if ctx.get("video_context"):
        batch_local = [idx for g in batch_groups for idx in g]
        batch_1based = (
            [non_empty_indices[i] + 1 for i in batch_local]
            if non_empty_indices
            else [i + 1 for i in batch_local]
        )
        source_1based = ctx.get("source_texts_1based") or {
            (non_empty_indices[i] + 1 if non_empty_indices else i + 1): all_texts[i]
            for i in range(len(all_texts))
        }
        prompt = enrich_user_prompt(
            prompt,
            video_context=ctx.get("video_context"),
            meaning_units=ctx.get("meaning_units"),
            batch_cue_indexes_1based=batch_1based,
            source_texts_1based=source_1based,
        )
    return prompt


def _build_grouped_user_prompt(
    batch_groups: List[List[int]],
    all_texts: List[str],
    target_lang: str,
) -> str:
    """
    Build the user prompt for a batch of phrase groups.

    Groups are shown as labelled sections so the model can read each group
    as a coherent thought before translating its individual cues.
    Each cue is numbered sequentially across the whole batch so the model
    emits a flat ``translations`` JSON array of the same length.
    """
    total_cues = sum(len(g) for g in batch_groups)
    n_groups = len(batch_groups)

    header = (
        f"You receive {total_cues} English subtitle cue{'s' if total_cues != 1 else ''} "
        f"from a video, organised into {n_groups} phrase group{'s' if n_groups != 1 else ''}.\n\n"
        "Read each group as a COMPLETE THOUGHT before translating. "
        f"Use the full group context to write natural, idiomatic {target_lang} subtitles "
        "— NOT word-by-word translation.\n\n"
        "Rules:\n"
        f"- Output exactly 1 {target_lang} line per input cue, same order\n"
        "- Do NOT merge, skip, or reorder cues\n"
        f"- {target_lang} text must flow naturally as a subtitle read on screen\n"
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
    """Build the user prompt asking the model to polish Vietnamese subtitle lines."""
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

def _parse_indexed_translations(content: str, expected_count: int) -> List[str]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)
    items = data.get("items")
    if items is None and isinstance(data, list):
        items = data
    if not isinstance(items, list):
        raise ValueError("OpenAI response missing 'items' array")

    if items and isinstance(items[0], str):
        if len(items) != expected_count:
            raise ValueError(f"Expected {expected_count} strings, got {len(items)}")
        return [str(v).strip() for v in items]

    if len(items) != expected_count:
        raise ValueError(f"Expected {expected_count} items, got {len(items)}")

    by_index: Dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each translation item must be an object")
        index = item.get("index")
        text = str(item.get("text", item.get("text_vi", ""))).strip()
        if not isinstance(index, int) or index < 1 or index > expected_count:
            raise ValueError(f"Invalid item index {index}")
        if index in by_index:
            raise ValueError(f"Duplicate item index {index}")
        by_index[index] = text

    missing = [i for i in range(1, expected_count + 1) if i not in by_index]
    if missing:
        raise ValueError(f"Missing item indices: {missing}")
    return [by_index[i] for i in range(1, expected_count + 1)]


def _validate_translation_output(
    translations: List[str],
    source_texts: List[str],
) -> None:
    if len(translations) != len(source_texts):
        raise ValueError(
            f"Translation count mismatch: {len(translations)} vs {len(source_texts)}"
        )
    for i, (src, vi) in enumerate(zip(source_texts, translations), start=1):
        if src.strip() and not vi.strip():
            raise ValueError(f"Empty translation for non-empty source cue at position {i}")


def _parse_json_strings(content: str, key: str, expected_count: int) -> List[str]:
    """
    Parse a JSON string returned by OpenAI and extract a string array.

    Strips optional markdown code fences, loads the JSON, locates the array
    under *key*, and validates that exactly *expected_count* strings are
    present.  Raises ``ValueError`` on any mismatch.
    """
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
    durations: Optional[List[float]] = None,
    *,
    repair: bool = False,
    strict: bool = False,
    translation_context: Optional[dict] = None,
    non_empty_indices: Optional[List[int]] = None,
) -> List[str]:
    """Translate a batch with neighboring English context and indexed JSON output."""
    batch_local_indices = [idx for g in batch_groups for idx in g]
    source_segments = [all_texts[idx] for idx in batch_local_indices]
    system_prompt = _build_openai_raw_system_prompt(topic)
    user_prompt = _build_context_user_prompt(
        batch_groups,
        all_texts,
        target_lang,
        durations,
        repair=repair,
        strict=strict,
        translation_context=translation_context,
        non_empty_indices=non_empty_indices,
    )

    dump_path = os.environ.get("OPENAI_CONTEXT_PROMPT_DUMP", "").strip()
    if dump_path and not repair:
        from pathlib import Path

        Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dump_path).write_text(
            f"=== SYSTEM ===\n{system_prompt}\n\n=== USER ===\n{user_prompt}",
            encoding="utf-8",
        )

    response_content = raw_llm_complete(
        client,
        model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        llm_task_type="raw_grouped_translate",
        batch_indices=[i + 1 for i in batch_local_indices],
        source_texts=source_segments,
        repair=repair,
        temperature=0.35 if repair else 0.4,
        response_format={"type": "json_object"},
    )
    translations = _parse_indexed_translations(
        response_content,
        len(source_segments),
    )
    _validate_translation_output(translations, source_segments)
    return translations


def _translate_grouped_with_validation_retry(
    client,
    model: str,
    batch_groups: List[List[int]],
    all_texts: List[str],
    target_lang: str,
    topic: str,
    durations: Optional[List[float]] = None,
    translation_context: Optional[dict] = None,
    non_empty_indices: Optional[List[int]] = None,
    *,
    strict_cue_count: bool = False,
) -> List[str]:
    batch_local_indices = [idx for g in batch_groups for idx in g]
    source_segments = [all_texts[idx] for idx in batch_local_indices]

    try:
        return _call_openai_translate_grouped(
            client,
            model,
            batch_groups,
            all_texts,
            target_lang,
            topic,
            durations,
            repair=False,
            strict=strict_cue_count,
            translation_context=translation_context,
            non_empty_indices=non_empty_indices,
        )
    except Exception as exc:
        print(f"  [OpenAI translate] validation failed ({exc}), repair retry…")
        try:
            return _call_openai_translate_grouped(
                client,
                model,
                batch_groups,
                all_texts,
                target_lang,
                topic,
                durations,
                repair=True,
                translation_context=translation_context,
                non_empty_indices=non_empty_indices,
            )
        except Exception as exc2:
            if len(source_segments) == 1:
                print(
                    f"  [OpenAI translate] repair failed ({exc2}), "
                    "safe fallback (source EN for non-empty cue)"
                )
                return [src if src.strip() else "" for src in source_segments]
            raise


def _call_openai_translate(
    client,
    model: str,
    segments: List[str],
    target_lang: str,
    topic: str,
    all_texts: Optional[List[str]] = None,
    durations: Optional[List[float]] = None,
    local_indices: Optional[List[int]] = None,
) -> List[str]:
    """Per-cue or small-list fallback using context-aware indexed JSON output."""
    if all_texts is not None and local_indices is not None:
        batch_groups = [[idx] for idx in local_indices]
        return _translate_grouped_with_validation_retry(
            client,
            model,
            batch_groups,
            all_texts,
            target_lang,
            topic,
            durations,
            translation_context,
            non_empty_indices,
            strict_cue_count=strict_cue_count,
        )

    lines = [f"[{i + 1}] {text}" for i, text in enumerate(segments)]
    n = len(segments)
    user_prompt = (
        f"You receive {n} English subtitle lines from a video.\n\n"
        "Your task is NOT word-by-word translation. Create natural, easy-to-understand "
        f"{target_lang} subtitles that Vietnamese viewers would actually read on screen.\n\n"
        "Read all lines together for context, then rewrite each line naturally. "
        "You may change sentence structure so Vietnamese flows smoothly. "
        "Keep one output string per input line — same count, same order.\n"
        "Do not leave any non-empty source cue blank.\n\n"
        + "\n".join(lines)
        + f'\n\nRespond with JSON: {{"items": [{{"index": 1, "text": "..."}}, ...]}} '
        f"containing exactly {n} objects."
    )
    response = create_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": _build_openai_raw_system_prompt(topic)},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    translations = _parse_indexed_translations(
        response.choices[0].message.content or "",
        n,
    )
    _validate_translation_output(translations, segments)
    return translations


def _call_openai_polish(
    client,
    model: str,
    segments: List[str],
    topic: str,
) -> List[str]:
    """Call OpenAI to polish a flat list of Vietnamese subtitle segments."""
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
    durations: Optional[List[float]] = None,
    translation_context: Optional[dict] = None,
    non_empty_indices: Optional[List[int]] = None,
    *,
    strict_cue_count: bool = False,
) -> List[str]:
    """
    Translate a batch of groups with cascading retry:
      1. Try the full batch with context + validation repair.
      2. On failure, retry each group individually.
      3. On a single-group failure, fall back to per-cue translation.
    """
    try:
        return _translate_grouped_with_validation_retry(
            client,
            model,
            batch_groups,
            all_texts,
            target_lang,
            topic,
            durations,
            translation_context,
            non_empty_indices,
            strict_cue_count=strict_cue_count,
        )
    except Exception as exc:
        print(f"  Batch failed ({exc}), retrying group-by-group…")

    results: List[str] = []
    for group in batch_groups:
        try:
            group_results = _translate_grouped_with_validation_retry(
                client,
                model,
                [group],
                all_texts,
                target_lang,
                topic,
                durations,
            )
            results.extend(group_results)
        except Exception as exc2:
            print(f"  Group failed ({exc2}), falling back to per-cue…")
            for local_idx in group:
                src = all_texts[local_idx]
                try:
                    per_cue = _call_openai_translate(
                        client,
                        model,
                        [src],
                        target_lang,
                        topic,
                        all_texts=all_texts,
                        durations=durations,
                        local_indices=[local_idx],
                    )
                    results.extend(per_cue)
                except Exception:
                    results.append(src)

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
    """
    Run a polish pass over all translated texts in *batch_size* chunks.

    Empty strings are passed through unchanged.  Returns a list of the
    same length as *texts* with each line polished in place.
    """
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

def _is_debug() -> bool:
    """Return True when DRAKONSUB_DEBUG is set to a truthy value."""
    return os.getenv("DRAKONSUB_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _log_grouping_stats(
    groups: List[List[int]],
    all_texts: List[str],
) -> None:
    """
    Print aggregate grouping statistics and, when debug mode is active,
    show a sample of the first two groups with their cue text.
    """
    total = sum(len(g) for g in groups)
    avg = total / len(groups) if groups else 0.0
    print(f"\n[Phrase Grouping] Original cue count : {total}")
    print(f"[Phrase Grouping] Phrase group count  : {len(groups)}")
    print(f"[Phrase Grouping] Avg cues per group  : {avg:.1f}")

    if not _is_debug():
        return

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
    """
    Print EN→VI sample pairs for the first two groups.
    Only emitted when DRAKONSUB_DEBUG is enabled.
    """
    if not _is_debug():
        return

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
    translation_context: Optional[dict] = None,
    *,
    strict_cue_count: bool = False,
) -> List[dict]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to a .env file in the project directory."
        )

    topic = normalize_topic(topic or os.environ.get("TRANSLATION_TOPIC"))

    if get_raw_translation_mode() == "cue_keyed":
        from .raw_cue_keyed_translate import translate_srt_entries_cue_keyed_openai

        result, _stats = translate_srt_entries_cue_keyed_openai(
            entries,
            target_lang=target_lang,
            model=model,
            topic=topic,
            batch_size=batch_size,
        )
        return result

    if get_raw_translation_mode() == "hybrid_guarded":
        from .raw_hybrid_guarded_translate import translate_srt_entries_hybrid_openai

        return translate_srt_entries_hybrid_openai(
            entries,
            target_lang=target_lang,
            model=model,
            batch_size=batch_size,
            topic=topic,
            polish=polish,
            translation_context=translation_context,
            strict_cue_count=strict_cue_count,
        )

    if get_raw_translation_mode() == "span_guarded":
        from .raw_span_guarded_translate import translate_srt_entries_span_guarded_openai

        return translate_srt_entries_span_guarded_openai(
            entries,
            target_lang=target_lang,
            model=model,
            batch_size=batch_size,
            topic=topic,
            polish=polish,
            translation_context=translation_context,
            strict_cue_count=strict_cue_count,
        )

    if get_raw_translation_mode() == "span_guarded_conservative":
        from .raw_span_guarded_translate import (
            translate_srt_entries_span_guarded_conservative_openai,
        )

        return translate_srt_entries_span_guarded_conservative_openai(
            entries,
            target_lang=target_lang,
            model=model,
            batch_size=batch_size,
            topic=topic,
            polish=polish,
            translation_context=translation_context,
            strict_cue_count=strict_cue_count,
        )

    if get_raw_translation_mode() == "span_guarded_tiered":
        from .raw_span_guarded_translate import (
            translate_srt_entries_span_guarded_tiered_openai,
        )

        return translate_srt_entries_span_guarded_tiered_openai(
            entries,
            target_lang=target_lang,
            model=model,
            batch_size=batch_size,
            topic=topic,
            polish=polish,
            translation_context=translation_context,
            strict_cue_count=strict_cue_count,
        )

    if get_raw_translation_mode() == "longform_chunked":
        from .longform_chunked_translate import translate_srt_entries_longform_chunked_openai

        return translate_srt_entries_longform_chunked_openai(
            entries,
            target_lang=target_lang,
            model=model,
            batch_size=batch_size,
            topic=topic,
            polish=polish,
            translation_context=translation_context,
            strict_cue_count=strict_cue_count,
        )

    client = OpenAI(api_key=api_key)
    model = model or get_openai_model()
    batch_size = batch_size or get_translation_batch_size()
    # Clamp so a single group can never exceed one full batch.
    max_cues_per_group = min(get_phrase_group_max_cues(), batch_size)
    polish = translation_polish_enabled() if polish is None else polish

    # -----------------------------------------------------------------------
    # Separate non-empty entries so grouping ignores blank cues.
    # -----------------------------------------------------------------------
    all_entry_texts = [e["text"].strip() for e in entries]
    non_empty_indices: List[int] = [i for i, t in enumerate(all_entry_texts) if t]
    non_empty_texts: List[str] = [all_entry_texts[i] for i in non_empty_indices]

    if not non_empty_texts:
        return list(entries)

    if translation_context is not None:
        translation_context = {
            **translation_context,
            "source_texts_1based": {
                i + 1: e.get("text", "").strip() for i, e in enumerate(entries)
            },
        }

    # -----------------------------------------------------------------------
    # Phrase / meaning-unit grouping + batching
    # -----------------------------------------------------------------------
    groups = resolve_translation_groups(
        non_empty_texts,
        non_empty_indices,
        max_cues_per_group,
        translation_context,
        _group_cues_by_sentence,
    )
    _log_grouping_stats(groups, non_empty_texts)

    batches = _pack_groups_into_batches(groups, batch_size)

    entry_durations = [
        _parse_ts(e["end_str"]) - _parse_ts(e["start_str"]) for e in entries
    ]
    non_empty_durations = [entry_durations[i] for i in non_empty_indices]

    # -----------------------------------------------------------------------
    # Translation loop
    # -----------------------------------------------------------------------
    translated_non_empty: Dict[int, str] = {}

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
            client,
            model,
            batch_groups,
            non_empty_texts,
            target_lang,
            topic,
            non_empty_durations,
            translation_context,
            non_empty_indices,
            strict_cue_count=strict_cue_count,
        )

        for local_idx, vi_text in zip(batch_local_indices, results):
            translated_non_empty[local_idx] = vi_text

    # -----------------------------------------------------------------------
    # Reconstruct full translated_texts list (entry-indexed)
    # -----------------------------------------------------------------------
    translated_texts = list(all_entry_texts)
    for local_idx, orig_entry_idx in enumerate(non_empty_indices):
        src = non_empty_texts[local_idx]
        vi_text = translated_non_empty.get(local_idx, "").strip()
        if src.strip() and not vi_text:
            vi_text = src
            print(
                f"  [OpenAI translate] cue {orig_entry_idx + 1} empty after batch; "
                f"fallback to source EN"
            )
        translated_texts[orig_entry_idx] = vi_text

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
