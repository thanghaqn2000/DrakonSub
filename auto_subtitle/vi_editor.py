"""
Shared Vietnamese subtitle editor pass.

Runs after raw translation for both OpenAI and Gemini engines.
Uses English source cues + raw Vietnamese + surrounding context to polish
subtitle wording without changing cue count, order, or timestamps.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    get_vi_editor_batch_size,
    get_vi_editor_context_window,
    get_vi_editor_temperature,
    llm_chat_kwargs,
    load_env,
    resolve_vi_editor_model,
    resolve_vi_editor_provider,
    vi_editor_openai_few_shot_enabled,
    vi_editor_save_debug,
)
from .translation_prompt_context import enrich_user_prompt
from .gemini_translate import (
    GeminiNonRetryableError,
    _call_gemini_json,
    _resolve_gemini_model,
)
from .openai_chat import create_chat_completion
from .translation_topics import normalize_topic

VI_EDITOR_SYSTEM_PROMPT = """You are a senior Vietnamese subtitle localization editor for short-form videos about economics, investing, business, finance, Bitcoin, markets, famous investors, entrepreneurs, and public speeches.

You will receive English source subtitle cues and raw Vietnamese subtitle translations.

Your job is to rewrite the Vietnamese subtitles so they sound like natural spoken Vietnamese for Facebook/Reels/TikTok viewers — sharp, easy to hear, and easy to read on screen.

Style priorities:
* Natural spoken Vietnamese, not textbook or news-anchor tone.
* Short, punchy lines. Cut filler and stiff connectors.
* Avoid machine-translated phrases like "bằng cách này hay cách khác", "điều đó giải thích", "một cách nào đó", "trong việc", "đối với".
* Avoid long, bookish sentences. Split the idea across cues only when each cue still reads naturally on its own.
* When the English is a joke, sarcasm, or jab, make the Vietnamese feel natural and conversational — not literal.
* When one idea is split across several cues, read the full context and make each fragment sound natural while keeping the thread clear.
* For very short reactive cues (What?, Huh?, Do something.), read nearby cues — they are often jokes or sarcasm, not literal commands.
* You may rewrite boldly if the raw line is stiff, but never change the speaker's meaning.

Do not translate word by word.
Do not make tiny cosmetic edits when a real rewrite is needed.
Use the English source to verify the real meaning, then rewrite the Vietnamese line.

Rules:
* Preserve the speaker's exact meaning.
* Do not add new facts, opinions, or claims.
* Do not exaggerate.
* Keep names, numbers, companies, financial terms, and logic accurate.
* Do not merge, split, skip, or reorder cues.
* Keep the same cue count and same order.
* Return JSON only.

Few-shot examples:

EN: And you'd be right, incidentally.
BAD VI: Và bạn hoàn toàn đúng, nhân tiện.
GOOD VI: Và thực ra, bạn đúng.

EN: But I'm not going to give you anything for it.
BAD VI: Nhưng tôi sẽ không cho bạn bất cứ điều gì về điều đó.
GOOD VI: Nhưng bảo tôi bỏ tiền mua nó thì không có chuyện đó.

EN: it isn't going to do anything.
BAD VI: nó sẽ không mang lại kết quả gì cả.
GOOD VI: bản thân nó chẳng tạo ra thứ gì cả.

EN: I have to sell it back to you one way or another.
BAD VI: Tôi sẽ phải bán lại cho bạn bằng cách này hay cách khác.
GOOD VI: Kiểu gì tôi cũng phải bán lại nó cho anh.

EN: But everybody knows what I'm like.
BAD VI: Nhưng mọi người đều biết tôi là người như thế nào.
GOOD VI: Nhưng ai cũng biết tôi thế nào rồi.

EN: But everybody knows what I'm like.
BAD VI: Nhưng ai cũng biết tôi như thế nào.
GOOD VI: Nhưng ai cũng biết tôi thế nào rồi.

EN: Why don't you call it Buffett Coin?
BAD VI: Tại sao bạn không gọi nó là Buffett Coin?
GOOD VI: Sao không tự đặt tên Buffett Coin luôn đi?

EN: You know, make your own or something.
BAD VI: Bạn biết đấy, hãy tự làm hoặc gì đó.
GOOD VI: Tự nghĩ ra coin riêng của bạn đi.

EN: What?
BAD VI: Gì cơ?
GOOD VI: Hả?

EN: Do something.
BAD VI: Hãy hành động đi.
GOOD VI: Nó phải làm được gì đó chứ.

EN: Do something.
BAD VI: Làm gì đó đi.
GOOD VI: Nó phải tạo ra được gì đó chứ.

Multi-cue joke sequence example (read together, edit each cue separately):

[20] EN: Why don't you call it Buffett Coin?
GOOD VI: Sao không tự đặt tên Buffett Coin luôn đi?

[21] EN: You know, make your own or something.
GOOD VI: Tự nghĩ ra coin riêng đi.

[22] EN: What?
GOOD VI: Hả?

[23] EN: Do something. (Buffett means the coin must actually produce value — not "go take action")
GOOD VI: Nó phải làm được gì đó chứ.

EN: But I'm not going to give you anything for
BAD VI: Nhưng tôi sẽ không cho bạn bất cứ điều gì.
GOOD VI: Nhưng bảo tôi bỏ tiền mua thì không có đâu.

EN: it.
BAD VI: Về chuyện đó.
GOOD VI: Không có chuyện đó.

EN: But that explains the difference between productive assets
BAD VI: Nhưng điều đó giải thích sự khác biệt giữa tài sản tạo ra dòng tiền
GOOD VI: Nhưng đó chính là khác biệt giữa tài sản tạo ra dòng tiền

EN: depends on the next guy paying you more
BAD VI: phụ thuộc vào việc người tiếp theo trả cho bạn nhiều hơn
GOOD VI: chỉ sống nhờ việc người sau mua lại giá cao hơn

EN: guy paying you more than the last guy got now.
BAD VI: trả cho bạn nhiều hơn người trước.
GOOD VI: có người sau trả giá cao hơn người trước

EN: and something that depends on the next
BAD VI: và những thứ phụ thuộc vào việc người sau
GOOD VI: và thứ chỉ sống nhờ người sau

EN: The apartments are going to produce rental.
BAD VI: Các căn hộ tạo ra thu nhập từ cho thuê.
GOOD VI: Căn hộ thì còn tạo ra tiền thuê nhà."""

VI_EDITOR_OPENAI_REWRITE_BLOCK = """
OpenAI editor mode — rewrite aggressively when needed:
* Compare each English source cue with its Vietnamese draft.
* If the draft is literal, unnatural, misleading, or wrong in financial/business context, rewrite it fully.
* Do not make cosmetic edits when a real rewrite is needed.
* Preserve cue count, order, and timestamps (text only).
* Use previous/next cues for cross-cue meaning.
* Prefer concise spoken Vietnamese for short-form video.

Multi-cue rewrite examples (edit each cue separately):

Example B:
EN [7] I mean, maybe I'll have the same people, but
EN [8] it isn't going to do anything.
BAD VI [7] cũng chỉ có những người đó thôi, nhưng
BAD VI [8] nó sẽ chẳng làm được gì cả.
GOOD VI [7] Có thể vẫn có người mua lại, nhưng
GOOD VI [8] bản thân nó chẳng tạo ra gì.

Example C:
EN [14] If I've got it all, he could create a mystery
EN [15] about it.
BAD VI [14] Nếu tôi giữ hết, sẽ thành bí ẩn.
BAD VI [15] về điều đó.
GOOD VI [14] Nếu tôi giữ hết, nó vẫn chỉ là
GOOD VI [15] một điều bí ẩn.

Example D:
EN [24] But I'm not going to give you anything for
EN [25] it.
BAD VI [24] Nhưng tôi sẽ không cho bạn bất cứ điều gì.
BAD VI [25] Không có chuyện đó.
GOOD VI [24] Nhưng bảo tôi bỏ tiền mua
GOOD VI [25] thì không có đâu.

Example E:
EN [27] productive assets and something that depends on the next guy paying you more
BAD VI tài sản sản xuất và thứ phụ thuộc vào người tiếp theo trả nhiều hơn
GOOD VI tài sản tạo ra giá trị / và thứ chỉ chờ người sau / mua lại với giá cao hơn."""


def _resolve_editor_system_prompt(provider: str) -> str:
    if provider == "openai" and vi_editor_openai_few_shot_enabled():
        return VI_EDITOR_SYSTEM_PROMPT + "\n\n" + VI_EDITOR_OPENAI_REWRITE_BLOCK
    return VI_EDITOR_SYSTEM_PROMPT


def _align_vi_entries_to_source(
    source_entries: List[dict],
    vi_entries: List[dict],
) -> List[dict]:
    """Align VI cues to source timing when parse/write dropped empty cues."""
    if len(source_entries) == len(vi_entries):
        return list(vi_entries)

    print(
        f"[VI Editor] Aligning VI cues to source timings "
        f"({len(vi_entries)} -> {len(source_entries)})"
    )
    vi_by_time = {
        (e["start_str"], e["end_str"]): e for e in vi_entries
    }
    aligned: List[dict] = []
    for source in source_entries:
        key = (source["start_str"], source["end_str"])
        if key in vi_by_time:
            aligned.append({**source, "text": vi_by_time[key]["text"]})
        else:
            aligned.append({**source, "text": ""})
    return aligned


def _format_context_block(
    label: str,
    entry_indices: List[int],
    source_texts: List[str],
    vi_texts: List[str],
) -> str:
    if not entry_indices:
        return f"{label}:\n- (none)"
    lines = []
    for idx in entry_indices:
        lines.append(f"[{idx + 1}] EN: {source_texts[idx]}")
        lines.append(f"    VI: {vi_texts[idx]}")
    return f"{label}:\n" + "\n".join(lines)


def _build_editor_user_prompt(
    batch_indices: List[int],
    source_texts: List[str],
    vi_texts: List[str],
    context_window: int,
    translation_context: Optional[dict] = None,
) -> str:
    first = batch_indices[0]
    last = batch_indices[-1]
    prev_start = max(0, first - context_window)
    prev_indices = list(range(prev_start, first))
    next_indices = list(range(last + 1, min(len(source_texts), last + 1 + context_window)))
    n = len(batch_indices)

    current_lines = []
    for local_i, idx in enumerate(batch_indices, start=1):
        current_lines.append(f"[{local_i}] EN: {source_texts[idx]}")
        current_lines.append(f"    VI: {vi_texts[idx]}")

    base = (
        "Use previous_context and next_context to understand meaning, jokes, and ideas "
        "that span multiple cues.\n"
        "Edit ONLY current_batch. Rewrite stiff raw lines into natural spoken Vietnamese.\n\n"
        + _format_context_block(
            f"previous_context (up to {context_window} cues before current batch)",
            prev_indices,
            source_texts,
            vi_texts,
        )
        + "\n\n"
        + "current_batch (edit ONLY these cues):\n"
        + "\n".join(current_lines)
        + "\n\n"
        + _format_context_block(
            f"next_context (up to {context_window} cues after current batch)",
            next_indices,
            source_texts,
            vi_texts,
        )
        + "\n\n"
        "Output rules:\n"
        f"- Return exactly {n} edited Vietnamese lines for current_batch\n"
        "- Same count, same order\n"
        "- Do not merge, split, skip, or reorder cues\n"
        "- For quick jokes, questions, or sarcasm (What?, Do something., make your own), "
        "make each cue sound like real spoken Vietnamese, not literal translation\n"
        "- Do not add explanations\n"
        '- Return JSON only in this format:\n'
        '  {"items": [{"index": 1, "text_vi": "..."}, ...]}\n'
        f"- items must contain exactly {n} objects\n"
        "- index must be 1..N matching current_batch order\n"
        "- text_vi must not be empty unless the raw VI cue was empty"
    )
    if translation_context and translation_context.get("video_context"):
        batch_1based = [i + 1 for i in batch_indices]
        source_1based = translation_context.get("source_texts_1based") or {
            i + 1: source_texts[i] for i in range(len(source_texts))
        }
        base = enrich_user_prompt(
            base,
            video_context=translation_context.get("video_context"),
            meaning_units=translation_context.get("meaning_units"),
            batch_cue_indexes_1based=batch_1based,
            source_texts_1based=source_1based,
        )
    return base


def _parse_editor_response(content: str, expected_count: int) -> List[str]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("Editor response missing 'items' array")
    if len(items) != expected_count:
        raise ValueError(f"Expected {expected_count} items, got {len(items)}")

    by_index: Dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each item must be an object")
        index = item.get("index")
        if not isinstance(index, int):
            raise ValueError("Each item must have integer index")
        text_vi = str(item.get("text_vi", "")).strip()
        if index < 1 or index > expected_count:
            raise ValueError(f"Item index {index} out of range 1..{expected_count}")
        if index in by_index:
            raise ValueError(f"Duplicate item index {index}")
        by_index[index] = text_vi

    missing = [i for i in range(1, expected_count + 1) if i not in by_index]
    if missing:
        raise ValueError(f"Missing item indices: {missing}")

    return [by_index[i] for i in range(1, expected_count + 1)]


def _save_debug_file(debug_dir: Optional[str], name: str, content: str) -> None:
    if not debug_dir:
        return
    path = Path(debug_dir) / "editor_debug"
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(content, encoding="utf-8")


def _call_openai_editor(
    client,
    model: str,
    user_prompt: str,
    temperature: float,
    system_prompt: str,
) -> Tuple[str, Dict]:
    response = create_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
        **llm_chat_kwargs(),
    )
    content = response.choices[0].message.content or ""
    usage = {}
    if getattr(response, "usage", None):
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return content, usage


def _call_editor_batch(
    *,
    provider: str,
    model: str,
    batch_indices: List[int],
    source_texts: List[str],
    vi_texts: List[str],
    context_window: int,
    temperature: float,
    debug_dir: Optional[str],
    batch_tag: str,
    translation_context: Optional[dict] = None,
) -> Tuple[List[str], Dict]:
    user_prompt = _build_editor_user_prompt(
        batch_indices, source_texts, vi_texts, context_window, translation_context
    )
    expected = len(batch_indices)
    system_prompt = _resolve_editor_system_prompt(provider)

    if vi_editor_save_debug() and debug_dir:
        _save_debug_file(
            debug_dir,
            f"{batch_tag}_system.txt",
            system_prompt,
        )
        _save_debug_file(debug_dir, f"{batch_tag}_user.txt", user_prompt)

    if provider == "openai":
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found for VI editor pass.")
        client = OpenAI(api_key=api_key)
        content, usage = _call_openai_editor(
            client, model, user_prompt, temperature, system_prompt
        )
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found for VI editor pass.")
        model = _resolve_gemini_model(api_key, model)
        content, usage = _call_gemini_json(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported VI editor provider: {provider}")

    if vi_editor_save_debug() and debug_dir:
        _save_debug_file(debug_dir, f"{batch_tag}_response.json", content)

    edited = _parse_editor_response(content, expected)

    # Preserve empty raw cues as empty; fall back to raw when model returns empty.
    result: List[str] = []
    for local_i, entry_idx in enumerate(batch_indices):
        raw = vi_texts[entry_idx].strip()
        candidate = edited[local_i].strip()
        if not raw:
            result.append("")
        elif not candidate:
            result.append(vi_texts[entry_idx])
        else:
            result.append(candidate)

    return result, usage


def _log_batch_metrics(
    provider: str,
    model: str,
    batch_size: int,
    input_count: int,
    output_count: int,
    retry_count: int,
    usage: Dict,
    fallback_mode: str,
) -> None:
    usage_text = ""
    if usage:
        if provider == "openai":
            usage_text = (
                f", usage(prompt={usage.get('prompt_tokens')}, "
                f"completion={usage.get('completion_tokens')}, "
                f"total={usage.get('total_tokens')})"
            )
        else:
            usage_text = (
                f", usage(prompt={usage.get('promptTokenCount')}, "
                f"candidates={usage.get('candidatesTokenCount')}, "
                f"total={usage.get('totalTokenCount')})"
            )
    print(
        "[VI Editor] "
        f"provider={provider} model={model} batch_size={batch_size} "
        f"input_line_count={input_count} output_line_count={output_count} "
        f"retry_count={retry_count} fallback_mode={fallback_mode}{usage_text}"
    )


def _edit_batch_with_retry(
    *,
    provider: str,
    model: str,
    batch_indices: List[int],
    source_texts: List[str],
    vi_texts: List[str],
    context_window: int,
    temperature: float,
    debug_dir: Optional[str],
    batch_tag: str,
    max_retries: int = 1,
    translation_context: Optional[dict] = None,
) -> Tuple[List[str], Dict]:
    attempts = 0
    last_usage: Dict = {}
    while attempts <= max_retries:
        try:
            result, usage = _call_editor_batch(
                provider=provider,
                model=model,
                batch_indices=batch_indices,
                source_texts=source_texts,
                vi_texts=vi_texts,
                context_window=context_window,
                temperature=temperature,
                debug_dir=debug_dir,
                batch_tag=batch_tag if attempts == 0 else f"{batch_tag}_retry{attempts}",
                translation_context=translation_context,
            )
            return result, {
                "retry_count": attempts,
                "usage": usage,
                "fallback_mode": "batch",
            }
        except GeminiNonRetryableError:
            raise
        except Exception as exc:
            attempts += 1
            if attempts > max_retries:
                print(f"  [VI Editor] batch failed after retries: {exc}")
                break
            print(f"  [VI Editor] batch retry {attempts}/{max_retries}: {exc}")

    if len(batch_indices) == 1:
        idx = batch_indices[0]
        return [vi_texts[idx]], {
            "retry_count": attempts,
            "usage": last_usage,
            "fallback_mode": "raw_single",
        }

    mid = len(batch_indices) // 2
    left_indices = batch_indices[:mid]
    right_indices = batch_indices[mid:]
    left_texts, left_stats = _edit_batch_with_retry(
        provider=provider,
        model=model,
        batch_indices=left_indices,
        source_texts=source_texts,
        vi_texts=vi_texts,
        context_window=context_window,
        temperature=temperature,
        debug_dir=debug_dir,
        batch_tag=f"{batch_tag}_L",
        max_retries=max_retries,
        translation_context=translation_context,
    )
    right_texts, right_stats = _edit_batch_with_retry(
        provider=provider,
        model=model,
        batch_indices=right_indices,
        source_texts=source_texts,
        vi_texts=vi_texts,
        context_window=context_window,
        temperature=temperature,
        debug_dir=debug_dir,
        batch_tag=f"{batch_tag}_R",
        max_retries=max_retries,
        translation_context=translation_context,
    )
    return left_texts + right_texts, {
        "retry_count": left_stats.get("retry_count", 0) + right_stats.get("retry_count", 0),
        "usage": right_stats.get("usage") or left_stats.get("usage") or {},
        "fallback_mode": "split",
    }


def edit_vi_srt_entries(
    source_entries: List[dict],
    vi_entries: List[dict],
    *,
    translation_engine: str = "openai",
    topic: Optional[str] = None,
    on_progress=None,
    debug_dir: Optional[str] = None,
    translation_context: Optional[dict] = None,
) -> List[dict]:
    """
    Polish raw Vietnamese subtitle entries using English source context.

    On failure, returns *vi_entries* unchanged (fallback to raw translation).
    """
    load_env()
    normalize_topic(topic or os.getenv("TRANSLATION_TOPIC"))

    if len(source_entries) != len(vi_entries):
        vi_entries = _align_vi_entries_to_source(source_entries, vi_entries)
        if len(source_entries) != len(vi_entries):
            raise ValueError(
                f"Source/VI cue count mismatch: {len(source_entries)} vs {len(vi_entries)}"
            )

    try:
        provider = resolve_vi_editor_provider(translation_engine)
        model = resolve_vi_editor_model(provider)
        batch_size = get_vi_editor_batch_size()
        context_window = get_vi_editor_context_window()
        temperature = get_vi_editor_temperature()

        source_texts = [e["text"].strip() for e in source_entries]
        vi_texts = [e["text"].strip() for e in vi_entries]
        n = len(source_entries)

        editable_indices = [
            i for i in range(n) if source_texts[i] or vi_texts[i]
        ]
        if not editable_indices:
            return list(vi_entries)

        edited_texts = list(vi_texts)
        done = 0
        total = len(editable_indices)
        batch_num = 0

        for start in range(0, len(editable_indices), batch_size):
            batch_indices = editable_indices[start:start + batch_size]
            batch_num += 1
            batch_tag = f"batch_{batch_num:03d}"
            first_cue = batch_indices[0] + 1
            last_cue = batch_indices[-1] + 1
            print(
                f"  [VI Editor] editing cues {first_cue}-{last_cue} "
                f"({len(batch_indices)} lines)..."
            )

            results, stats = _edit_batch_with_retry(
                provider=provider,
                model=model,
                batch_indices=batch_indices,
                source_texts=source_texts,
                vi_texts=vi_texts,
                context_window=context_window,
                temperature=temperature,
                debug_dir=debug_dir,
                batch_tag=batch_tag,
                translation_context=translation_context,
            )
            _log_batch_metrics(
                provider=provider,
                model=model,
                batch_size=batch_size,
                input_count=len(batch_indices),
                output_count=len(results),
                retry_count=stats.get("retry_count", 0),
                usage=stats.get("usage") or {},
                fallback_mode=stats.get("fallback_mode", "batch"),
            )

            for entry_idx, new_text in zip(batch_indices, results):
                edited_texts[entry_idx] = new_text

            done += len(batch_indices)
            if on_progress:
                percent = min(99, int(100 * done / max(total, 1)))
                on_progress(f"Editing Vietnamese subtitles... {done}/{total}", percent)

        return [
            {**source_entries[i], "text": edited_texts[i]} for i in range(n)
        ]

    except Exception as exc:
        print(f"[VI Editor] Failed, using raw translations: {exc}")
        return _align_vi_entries_to_source(source_entries, vi_entries)


def edit_vi_srt_file(
    source_srt_path: str,
    vi_srt_path: str,
    output_srt_path: str,
    *,
    translation_engine: str = "openai",
    topic: Optional[str] = None,
    on_progress=None,
    debug_dir: Optional[str] = None,
    translation_context: Optional[dict] = None,
) -> str:
    """Read source + raw VI SRT files, run editor pass, write *output_srt_path*."""
    from .utils import parse_srt, write_srt_entries

    with open(source_srt_path, encoding="utf-8") as f:
        source_entries = parse_srt(f.read())
    with open(vi_srt_path, encoding="utf-8") as f:
        vi_entries = parse_srt(f.read())

    edited = edit_vi_srt_entries(
        source_entries,
        vi_entries,
        translation_engine=translation_engine,
        topic=topic,
        on_progress=on_progress,
        debug_dir=debug_dir,
        translation_context=translation_context,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_srt_path)), exist_ok=True)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        write_srt_entries(edited, file=f)
    return output_srt_path