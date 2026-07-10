import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from .gemini_keys import (
    GeminiQuotaError,
    call_gemini_with_key_rotation,
    load_gemini_api_keys,
    resolve_gemini_model_for_keys,
)

from .config import (
    get_gemini_model,
    get_phrase_group_max_cues,
    get_translation_batch_size,
)
from .meaning_unit_builder import resolve_translation_groups
from .translation_prompt_context import enrich_user_prompt
from .translation_topics import TOPICS, normalize_topic

# Punctuation that signals the end of a sentence / phrase group.
_SENTENCE_END_CHARS = frozenset(".!?…")

_GEMINI_SYSTEM_PROMPT = """You are a professional Vietnamese subtitle localization editor.

Your job is not to translate words.
Your job is to rewrite the speaker’s meaning into natural Vietnamese subtitles.

Audience:
Vietnamese Facebook/Reels viewers.

Style:
- Very easy to understand.
- Natural spoken Vietnamese.
- Concise enough for subtitles.
- Avoid robotic or literal translation.
- Avoid academic wording.
- Preserve the speaker’s intent and tone.
- Do not add new ideas.
- Do not remove important ideas.
- Keep names, numbers, financial terms, Bitcoin, Fed, ETF, inflation, interest rate accurate.

Important:
Read all lines together for context first.
Then rewrite each line into Vietnamese.
Return exactly one Vietnamese subtitle for each input line.
Same count, same order.
Return JSON only."""

_DOMAIN_GLOSSARY = """Domain glossary (use clear Vietnamese that ordinary viewers understand):
- investing: đầu tư
- economics: kinh tế
- finance: tài chính
- business: kinh doanh
- stocks: cổ phiếu
- assets: tài sản
- inflation: lạm phát
- interest rates: lãi suất
- cash flow: dòng tiền
- value investing: đầu tư giá trị
- speculation: đầu cơ
- shareholders: cổ đông
- productive assets: tài sản tạo ra dòng tiền"""


class GeminiNonRetryableError(RuntimeError):
    """Hard Gemini failures that should fail fast without retry loops."""


_MODEL_ALIASES = {
    "gemini-3.1-pro": [
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview-customtools",
    ],
}


def _list_gemini_models(api_key: str) -> List[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini listModels HTTP {exc.code}: {detail}") from exc

    data = json.loads(body)
    models = data.get("models") or []
    names: List[str] = []
    for model in models:
        raw_name = str(model.get("name") or "").strip()
        if raw_name.startswith("models/"):
            raw_name = raw_name[len("models/"):]
        if raw_name:
            names.append(raw_name)
    return names


def _resolve_gemini_model(api_key: str, requested_model: str) -> str:
    requested = requested_model.strip()
    available = _list_gemini_models(api_key)
    if requested in available:
        return requested

    for alias in _MODEL_ALIASES.get(requested, []):
        if alias in available:
            print(f"[Translation] gemini model alias: {requested} -> {alias}")
            return alias

    close_matches = [name for name in available if requested.split("-")[0] in name]
    hint = ", ".join(sorted(close_matches)[:8]) or ", ".join(sorted(available)[:8])
    raise GeminiNonRetryableError(
        f"Gemini model '{requested}' is not available for this API key. "
        f"Available candidates: {hint}"
    )


def _group_cues_by_sentence(
    texts: List[str],
    max_cues_per_group: int = 6,
) -> List[List[int]]:
    groups: List[List[int]] = []
    current: List[int] = []

    for i, text in enumerate(texts):
        current.append(i)
        inner = text.strip().rstrip("\"')}]")
        last_char = inner[-1:] if inner else ""
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


def _build_grouped_user_prompt(
    batch_groups: List[List[int]],
    all_texts: List[str],
    target_lang: str,
    translation_context: Optional[dict] = None,
    non_empty_indices: Optional[List[int]] = None,
) -> str:
    batch_local_indices = [idx for group in batch_groups for idx in group]
    total_cues = len(batch_local_indices)
    first_idx = batch_local_indices[0]
    last_idx = batch_local_indices[-1]

    prev_start = max(0, first_idx - 5)
    previous_context = all_texts[prev_start:first_idx]
    next_context = all_texts[last_idx + 1:last_idx + 6]

    previous_lines = (
        "\n".join(f"- {line}" for line in previous_context)
        if previous_context
        else "- (none)"
    )
    current_lines = "\n".join(
        f"[{i}] {all_texts[idx]}" for i, idx in enumerate(batch_local_indices, 1)
    )
    next_lines = (
        "\n".join(f"- {line}" for line in next_context)
        if next_context
        else "- (none)"
    )

    base = (
        "Use context sections below to understand meaning.\n\n"
        "previous_context (up to 5 lines before current batch):\n"
        f"{previous_lines}\n\n"
        "current_batch (translate ONLY these lines):\n"
        f"{current_lines}\n\n"
        "next_context (up to 5 lines after current batch):\n"
        f"{next_lines}\n\n"
        "Output rules:\n"
        f"- Translate only current_batch into natural {target_lang} subtitles\n"
        "- Output exactly one subtitle per current_batch line\n"
        "- Same count, same order\n"
        "- Do not add/remove/merge/split lines\n"
        '- Return JSON only in this format: {"translations": ["...", "..."]}\n'
        f"- translations must contain exactly {total_cues} strings"
    )
    if translation_context and translation_context.get("video_context"):
        batch_local = batch_local_indices
        batch_1based = (
            [non_empty_indices[i] + 1 for i in batch_local]
            if non_empty_indices
            else [i + 1 for i in batch_local]
        )
        source_1based = translation_context.get("source_texts_1based") or {
            (non_empty_indices[i] + 1 if non_empty_indices else i + 1): all_texts[i]
            for i in range(len(all_texts))
        }
        base = enrich_user_prompt(
            base,
            video_context=translation_context.get("video_context"),
            meaning_units=translation_context.get("meaning_units"),
            batch_cue_indexes_1based=batch_1based,
            source_texts_1based=source_1based,
        )
    return base


def _build_gemini_system_prompt(topic: str) -> str:
    topic_info = TOPICS.get(topic)
    pieces = [_GEMINI_SYSTEM_PROMPT, _DOMAIN_GLOSSARY]
    if topic_info:
        pieces.append(f"Context topic/tone:\n{topic_info.guidance}")
        if topic_info.glossary:
            pieces.append(f"Topic-specific glossary:\n{topic_info.glossary}")
    return "\n\n".join(pieces)


def _parse_json_strings(content: str, key: str, expected_count: int) -> List[str]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)
    values = data.get(key, data if isinstance(data, list) else None)
    if not isinstance(values, list):
        raise ValueError(f"Gemini response missing '{key}' array")
    if len(values) != expected_count:
        raise ValueError(f"Expected {expected_count} strings, got {len(values)}")
    return [str(v).strip() for v in values]


def _call_gemini_json(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.4,
) -> Tuple[str, Dict]:
    model_id = urllib.parse.quote(model, safe="")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if exc.code == 404 and ("is not found" in detail or "NOT_FOUND" in detail):
            raise GeminiNonRetryableError(
                f"Gemini model '{model}' is unavailable for generateContent (v1beta). "
                "Use a supported model like 'gemini-2.5-flash'. "
                f"Raw error: {detail}"
            ) from exc
        if exc.code == 429 and "quota" in detail.lower():
            raise GeminiQuotaError(
                f"Gemini quota exceeded for model '{model}'. "
                "Enable billing or request higher quota to run Pro models. "
                f"Raw error: {detail}"
            ) from exc
        if exc.code == 429:
            raise GeminiQuotaError(
                f"Gemini rate limit exceeded for model '{model}'. Raw error: {detail}"
            ) from exc
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc

    data = json.loads(body)
    if "error" in data:
        raise RuntimeError(f"Gemini API error: {data['error']}")

    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates")

    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    text = "".join(part.get("text", "") for part in parts if part.get("text"))
    if not text.strip():
        raise ValueError("Gemini returned empty content")

    usage = data.get("usageMetadata") or {}
    return text, usage


def call_gemini_json_with_key_rotation(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.4,
    *,
    api_keys: Optional[List[str]] = None,
    action: str = "Gemini request",
) -> Tuple[str, Dict]:
    return call_gemini_with_key_rotation(
        lambda api_key: _call_gemini_json(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        ),
        api_keys=api_keys,
        action=action,
    )


def _call_gemini_translate_grouped(
    model: str,
    batch_groups: List[List[int]],
    all_texts: List[str],
    target_lang: str,
    topic: str,
    translation_context: Optional[dict] = None,
    non_empty_indices: Optional[List[int]] = None,
    *,
    api_keys: Optional[List[str]] = None,
) -> Tuple[List[str], Dict]:
    total_cues = sum(len(g) for g in batch_groups)
    content, usage = call_gemini_json_with_key_rotation(
        model=model,
        system_prompt=_build_gemini_system_prompt(topic),
        user_prompt=_build_grouped_user_prompt(
            batch_groups,
            all_texts,
            target_lang,
            translation_context,
            non_empty_indices,
        ),
        temperature=0.4,
        api_keys=api_keys,
        action="Gemini translation batch",
    )
    return _parse_json_strings(content, "translations", total_cues), usage


def _translate_batch_with_retry(
    model: str,
    batch_groups: List[List[int]],
    all_texts: List[str],
    target_lang: str,
    topic: str,
    max_retries: int = 2,
    translation_context: Optional[dict] = None,
    non_empty_indices: Optional[List[int]] = None,
    *,
    api_keys: Optional[List[str]] = None,
) -> Tuple[List[str], Dict]:
    attempts = 0
    last_usage: Dict = {}
    while attempts <= max_retries:
        try:
            result, usage = _call_gemini_translate_grouped(
                model,
                batch_groups,
                all_texts,
                target_lang,
                topic,
                translation_context,
                non_empty_indices,
                api_keys=api_keys,
            )
            return result, {
                "retry_count": attempts,
                "usage": usage,
                "fallback_mode": "batch",
            }
        except (GeminiNonRetryableError, GeminiQuotaError):
            raise
        except Exception as exc:
            attempts += 1
            if attempts > max_retries:
                print(f"  Gemini batch failed after retries: {exc}")
                break
            print(f"  Gemini batch retry {attempts}/{max_retries} because: {exc}")

    results: List[str] = []
    total_retries = attempts - 1
    for group in batch_groups:
        group_attempt = 0
        while group_attempt <= max_retries:
            try:
                group_result, usage = _call_gemini_translate_grouped(
                    model,
                    [group],
                    all_texts,
                    target_lang,
                    topic,
                    translation_context,
                    non_empty_indices,
                    api_keys=api_keys,
                )
                results.extend(group_result)
                last_usage = usage
                total_retries += group_attempt
                break
            except (GeminiNonRetryableError, GeminiQuotaError):
                raise
            except Exception as exc:
                group_attempt += 1
                if group_attempt > max_retries:
                    raise RuntimeError(
                        f"Gemini translation failed for one subtitle group after retries: {exc}"
                    ) from exc
                print(
                    f"  Gemini group retry {group_attempt}/{max_retries} because: {exc}"
                )

    return results, {
        "retry_count": total_retries,
        "usage": last_usage,
        "fallback_mode": "group_or_cue",
    }


def _log_batch_metrics(
    model: str,
    batch_size: int,
    input_count: int,
    output_count: int,
    retry_count: int,
    usage: Dict,
) -> None:
    usage_text = ""
    if usage:
        usage_text = (
            f", usage(prompt={usage.get('promptTokenCount')}, "
            f"candidates={usage.get('candidatesTokenCount')}, "
            f"total={usage.get('totalTokenCount')})"
        )
    print(
        "[Translation] engine=gemini "
        f"model={model} batch_size={batch_size} "
        f"input_line_count={input_count} output_line_count={output_count} "
        f"retry_count={retry_count}{usage_text}"
    )


def translate_srt_entries_gemini(
    entries: List[dict],
    target_lang: str = "vi",
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    topic: Optional[str] = None,
    on_progress=None,
    translation_context: Optional[dict] = None,
    *,
    strict_cue_count: bool = False,
) -> List[dict]:
    api_keys = load_gemini_api_keys()
    if not api_keys:
        raise ValueError(
            "No Gemini API keys configured. Set GEMINI_API_KEY_1..4 in .env "
            "when TRANSLATION_ENGINE=gemini."
        )

    topic = normalize_topic(topic or os.getenv("TRANSLATION_TOPIC"))
    model = model or get_gemini_model()
    model, _resolved_key = resolve_gemini_model_for_keys(
        api_keys, model, _resolve_gemini_model
    )
    batch_size = batch_size or get_translation_batch_size()
    max_cues_per_group = min(get_phrase_group_max_cues(), batch_size)

    all_entry_texts = [entry["text"].strip() for entry in entries]
    non_empty_indices: List[int] = [i for i, text in enumerate(all_entry_texts) if text]
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

    groups = resolve_translation_groups(
        non_empty_texts,
        non_empty_indices,
        max_cues_per_group,
        translation_context,
        _group_cues_by_sentence,
    )
    batches = _pack_groups_into_batches(groups, batch_size)

    translated_non_empty: Dict[int, str] = {}
    done = 0
    total = len(non_empty_texts)
    for batch_groups in batches:
        batch_local_indices = [idx for group in batch_groups for idx in group]
        first_cue = batch_local_indices[0] + 1
        last_cue = batch_local_indices[-1] + 1
        print(
            f"  Gemini translating cues {first_cue}-{last_cue} "
            f"({len(batch_local_indices)} lines)..."
        )
        results, stats = _translate_batch_with_retry(
            model=model,
            batch_groups=batch_groups,
            all_texts=non_empty_texts,
            target_lang=target_lang,
            topic=topic,
            translation_context=translation_context,
            non_empty_indices=non_empty_indices,
            api_keys=api_keys,
        )
        _log_batch_metrics(
            model=model,
            batch_size=batch_size,
            input_count=len(batch_local_indices),
            output_count=len(results),
            retry_count=stats.get("retry_count", 0),
            usage=stats.get("usage") or {},
        )

        for local_idx, vi_text in zip(batch_local_indices, results):
            translated_non_empty[local_idx] = vi_text

        done += len(batch_local_indices)
        if on_progress:
            percent = min(99, int(100 * done / max(total, 1)))
            on_progress(f"Gemini translating subtitles... {done}/{total}", percent)

    translated_texts = list(all_entry_texts)
    for local_idx, orig_entry_idx in enumerate(non_empty_indices):
        translated_texts[orig_entry_idx] = translated_non_empty.get(
            local_idx,
            all_entry_texts[orig_entry_idx],
        )

    return [{**entry, "text": translated_texts[i]} for i, entry in enumerate(entries)]
