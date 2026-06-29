"""
Vietnamese subtitle compression pass.

Shortens cues that are too long for their display window, especially fast cues.
Runs after the editor pass and before readability/timing.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

from .config import (
    VI_COMPRESSION_ENABLED,
    VI_COMPRESSION_FAST_MAX_DURATION,
    VI_COMPRESSION_FAST_MAX_WORDS,
    VI_COMPRESSION_MAX_CPS,
    VI_COMPRESSION_MEDIUM_MAX_DURATION,
    VI_COMPRESSION_MEDIUM_MAX_WORDS,
    VI_COMPRESSION_MIN_SHORTEN_RATIO,
    VI_COMPRESSION_TRIGGER_CPS,
    get_openai_model,
    llm_chat_kwargs,
    llm_temperature,
    load_env,
)
from .openai_chat import create_chat_completion

_COMPRESSION_SYSTEM_PROMPT = """You compress Vietnamese subtitles so viewers can read them within very short on-screen windows.

Your job is NOT to re-translate word-for-word. Choose the shortest natural Vietnamese phrase that preserves the core meaning.

Compression is not just cutting words — fix awkward phrasing while shortening.

Duration rules:
* If cue duration < 0.8s: usually 1–6 words only.
* If cue duration 0.8s–1.5s: usually under 10 words.
* Remove filler openers like "Ý tôi là", "có thể tôi cũng chỉ", "được cho những người đó" when the cue is fast.
* Preserve core meaning, not every word.
* Keep names, numbers, and key terms accurate.
* Do not merge, split, skip, or reorder cues.
* Return JSON only.

Investment / business phrasing (prefer natural spoken Vietnamese):
* "produce/do something" → "tạo ra giá trị" or "tạo ra gì" when context fits.
* "paying more than the last guy" → "mua lại giá cao hơn".
* "make your own coin" → "tự tạo đồng riêng" (avoid awkward mixes like "nghĩ ra coin riêng").
* "have all of it" / hoarding → "giữ hết" reads more naturally than "có hết" in investing talk.
* Prefer "đồng" over literal "coin" when talking about cryptocurrency in Vietnamese.

Examples:

LONG: Ý tôi là, có thể tôi cũng chỉ bán được cho những người đó, nhưng
SHORT: Có thể vẫn có người mua lại, nhưng

LONG: Nó phải tạo ra được gì đó chứ.
SHORT: Phải tạo ra gì chứ.

LONG: Ý tôi là, có thể tôi sẽ gặp lại những người đó, nhưng
SHORT: Dù vẫn có người mua lại,

LONG: Tự nghĩ ra coin riêng đi.
SHORT: Tự tạo đồng riêng đi.

LONG: Phải làm gì chứ.
SHORT: Phải tạo ra gì chứ.

LONG: có người sau trả giá cao hơn người trước.
SHORT: người sau mua lại giá cao hơn.

LONG: Nếu tôi có hết, anh ta có thể tạo ra bí ẩn về nó.
SHORT: Nếu tôi giữ hết, nó vẫn chỉ là bí ẩn."""

_STRICT_RETRY_SUFFIX = (
    "\n\nSTRICT MODE: Previous result was still too long or awkward. "
    "Cut harder and pick the shortest natural Vietnamese. "
    "Remove filler. Fix mixed EN/VI phrasing (e.g. 'coin' → 'đồng', 'nghĩ ra coin' → 'tạo đồng riêng')."
)

_FILLER_PREFIX_RE = re.compile(
    r"^(?:Ý tôi là,?\s*|Tôi nghĩ là,?\s*|Có thể tôi cũng chỉ\s*)",
    re.IGNORECASE,
)

_FILLER_FRAGMENT_RES = [
    re.compile(r"Ý tôi là", re.IGNORECASE),
    re.compile(r"có thể tôi cũng chỉ", re.IGNORECASE),
    re.compile(r"được cho những người đó", re.IGNORECASE),
    re.compile(r"có thể tôi sẽ", re.IGNORECASE),
]

_EXACT_REPLACEMENTS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"^Nó phải tạo ra được gì đó chứ\.?$",
            re.IGNORECASE,
        ),
        "Phải tạo ra gì chứ.",
    ),
    (
        re.compile(
            r"^Nó phải làm được gì đó chứ\.?$",
            re.IGNORECASE,
        ),
        "Phải tạo ra gì chứ.",
    ),
    (
        re.compile(
            r"^Phải làm được gì chứ\.?$",
            re.IGNORECASE,
        ),
        "Phải tạo ra gì chứ.",
    ),
    (
        re.compile(
            r"^Phải làm gì chứ\.?$",
            re.IGNORECASE,
        ),
        "Phải tạo ra gì chứ.",
    ),
    (
        re.compile(
            r"^Tự nghĩ ra coin riêng đi\.?$",
            re.IGNORECASE,
        ),
        "Tự tạo đồng riêng đi.",
    ),
    (
        re.compile(
            r"^Tự làm cái của riêng mình đi\.?$",
            re.IGNORECASE,
        ),
        "Tự tạo đồng riêng đi.",
    ),
    (
        re.compile(
            r"^có người sau trả giá cao hơn người trước\.?$",
            re.IGNORECASE,
        ),
        "Người sau mua lại giá cao hơn.",
    ),
    (
        re.compile(
            r"^Nếu tôi có hết, anh ta có thể tạo ra bí ẩn về nó\.?$",
            re.IGNORECASE,
        ),
        "Nếu tôi giữ hết, nó vẫn chỉ là bí ẩn.",
    ),
    (
        re.compile(
            r"^Nếu tôi có tất cả, anh ấy có thể tạo ra một bí ẩn\.?$",
            re.IGNORECASE,
        ),
        "Nếu tôi giữ hết, nó vẫn chỉ là bí ẩn.",
    ),
    (
        re.compile(
            r"^Nếu tôi có tất cả, anh ấy tạo ra bí ẩn$",
            re.IGNORECASE,
        ),
        "Nếu tôi giữ hết, nó vẫn chỉ là bí ẩn.",
    ),
]

_AWKWARD_FRAGMENT_RES = [
    re.compile(r"nghĩ ra coin", re.IGNORECASE),
    re.compile(r"trả giá cao hơn người trước", re.IGNORECASE),
    re.compile(r"Phải làm gì chứ", re.IGNORECASE),
]


def _parse_ts(ts: str) -> float:
    ts = ts.strip()
    time_part, millis_str = ts.split(",")
    h, m, s = time_part.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(millis_str) / 1000.0


def _char_count(text: str) -> int:
    return len(re.sub(r"\s+", " ", text).strip())


def _word_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return len(text.split())


def _cps(text: str, duration: float) -> float:
    if duration <= 0:
        return 0.0
    return _char_count(text) / duration


def _target_word_max(duration: float) -> int:
    if duration < VI_COMPRESSION_FAST_MAX_DURATION:
        return VI_COMPRESSION_FAST_MAX_WORDS
    if duration < VI_COMPRESSION_MEDIUM_MAX_DURATION:
        return VI_COMPRESSION_MEDIUM_MAX_WORDS
    return max(VI_COMPRESSION_MEDIUM_MAX_WORDS, int(duration * 12))


def _has_awkward_phrasing(text: str) -> bool:
    return any(pattern.search(text) for pattern in _AWKWARD_FRAGMENT_RES)


def _apply_rule_compression(text: str) -> str:
    updated = text.strip()
    changed = True
    while changed:
        changed = False
        for pattern, replacement in _EXACT_REPLACEMENTS:
            if pattern.fullmatch(updated):
                updated = replacement
                changed = True
                break
    updated = _FILLER_PREFIX_RE.sub("", updated)
    updated = re.sub(r"\s+", " ", updated).strip()
    return updated


def _is_sufficiently_compressed(text: str, duration: float) -> bool:
    if not text.strip() or duration <= 0:
        return True
    words = _word_count(text)
    cps = _cps(text, duration)
    if duration < VI_COMPRESSION_FAST_MAX_DURATION:
        return words <= VI_COMPRESSION_FAST_MAX_WORDS
    if duration < VI_COMPRESSION_MEDIUM_MAX_DURATION:
        return words <= VI_COMPRESSION_MEDIUM_MAX_WORDS and cps <= VI_COMPRESSION_MAX_CPS
    return cps <= VI_COMPRESSION_TRIGGER_CPS


def _needs_compression(text: str, duration: float) -> bool:
    return not _is_sufficiently_compressed(text, duration)


def _has_filler_fragments(text: str) -> bool:
    return any(pattern.search(text) for pattern in _FILLER_FRAGMENT_RES)


def _compression_acceptable(
    original: str,
    candidate: str,
    duration: float,
    *,
    allow_rephrase: bool = False,
) -> bool:
    if not candidate.strip():
        return False
    new_cps = _cps(candidate, duration)
    if new_cps > VI_COMPRESSION_MAX_CPS:
        return False
    if duration < 2.0 and _has_filler_fragments(candidate):
        return False
    if _has_awkward_phrasing(candidate) and not _has_awkward_phrasing(original):
        return False
    if allow_rephrase and candidate.strip() != original.strip():
        return _is_sufficiently_compressed(candidate, duration) or not _has_awkward_phrasing(
            candidate
        )
    if not _meaningfully_shorter(original, candidate):
        return False
    return _is_sufficiently_compressed(candidate, duration)


def _meaningfully_shorter(old: str, new: str) -> bool:
    old_chars = _char_count(old)
    new_chars = _char_count(new)
    if new_chars >= old_chars:
        return False
    if old_chars == 0:
        return bool(new_chars)
    ratio = (old_chars - new_chars) / old_chars
    return ratio >= VI_COMPRESSION_MIN_SHORTEN_RATIO or _word_count(new) < _word_count(old)


def _build_user_prompt(
    batch: List[Tuple[int, str, float]],
    *,
    strict: bool = False,
) -> str:
    lines = []
    for local_i, (entry_idx, text, duration) in enumerate(batch, start=1):
        words = _word_count(text)
        target_words = _target_word_max(duration)
        lines.append(
            f"[{local_i}] duration={duration:.2f}s | words={words} | "
            f"target≤{target_words} words | text: {text}"
        )
    n = len(batch)
    prompt = (
        f"Compress these {n} Vietnamese subtitle lines to fit their duration.\n"
        "Pick the shortest natural Vietnamese phrase — fix awkward wording, not just cut words.\n"
        "Return exactly one shorter line per input line, same order.\n\n"
        + "\n".join(lines)
        + f'\n\nRespond with JSON: {{"items": [{{"index": 1, "text_vi": "..."}}, ...]}} '
        f"containing exactly {n} objects."
    )
    if strict:
        prompt += _STRICT_RETRY_SUFFIX
    return prompt


def _parse_compression_response(content: str, expected_count: int) -> List[str]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)
    items = data.get("items", data.get("subtitles"))
    if isinstance(items, list) and items and isinstance(items[0], str):
        if len(items) != expected_count:
            raise ValueError(f"Expected {expected_count} strings, got {len(items)}")
        return [str(v).strip() for v in items]

    if not isinstance(items, list):
        raise ValueError("Compression response missing items array")
    if len(items) != expected_count:
        raise ValueError(f"Expected {expected_count} items, got {len(items)}")

    by_index: Dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each compression item must be an object")
        index = item.get("index")
        text_vi = str(item.get("text_vi", "")).strip()
        if not isinstance(index, int) or index < 1 or index > expected_count:
            raise ValueError(f"Invalid item index {index}")
        if index in by_index:
            raise ValueError(f"Duplicate item index {index}")
        by_index[index] = text_vi

    missing = [i for i in range(1, expected_count + 1) if i not in by_index]
    if missing:
        raise ValueError(f"Missing item indices: {missing}")
    return [by_index[i] for i in range(1, expected_count + 1)]


def _call_openai_compress(
    batch: List[Tuple[int, str, float]],
    *,
    strict: bool = False,
) -> List[str]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found for VI compression pass.")

    client = OpenAI(api_key=api_key)
    model = get_openai_model()
    user_prompt = _build_user_prompt(batch, strict=strict)
    response = create_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": _COMPRESSION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=llm_temperature(0.2 if not strict else 0.1),
        response_format={"type": "json_object"},
        **llm_chat_kwargs(),
    )
    content = response.choices[0].message.content or ""
    return _parse_compression_response(content, len(batch))


def _compress_batch(
    batch: List[Tuple[int, str, float]],
    originals: List[str],
    durations: List[float],
) -> List[Optional[str]]:
    """Return compressed texts aligned to batch, or None to keep original."""
    try:
        candidates = _call_openai_compress(batch, strict=False)
    except Exception as exc:
        print(f"  [VI Compression] batch failed: {exc}")
        return [None] * len(batch)

    results: List[Optional[str]] = []
    retry_batch: List[Tuple[int, str, float]] = []
    retry_positions: List[int] = []

    for pos, ((entry_idx, _text, duration), original, candidate) in enumerate(
        zip(batch, originals, candidates)
    ):
        candidate = candidate.strip()
        if not candidate:
            results.append(None)
            continue

        if _compression_acceptable(
            original, candidate, duration, allow_rephrase=_has_awkward_phrasing(original)
        ):
            results.append(candidate)
            continue

        needs_retry = (
            _cps(candidate, duration) > VI_COMPRESSION_MAX_CPS
            or (
                not _meaningfully_shorter(original, candidate)
                and not _has_awkward_phrasing(original)
            )
            or (duration < 2.0 and _has_filler_fragments(candidate))
            or _has_awkward_phrasing(candidate)
        )
        if needs_retry:
            retry_batch.append((entry_idx, original, duration))
            retry_positions.append(pos)
            results.append(None)
            continue

        results.append(candidate)

    if retry_batch:
        try:
            retry_results = _call_openai_compress(retry_batch, strict=True)
            for pos, candidate, (entry_idx, original, duration) in zip(
                retry_positions, retry_results, retry_batch
            ):
                candidate = candidate.strip()
                if not candidate:
                    continue
                if _compression_acceptable(
                    original,
                    candidate,
                    duration,
                    allow_rephrase=_has_awkward_phrasing(original),
                ):
                    results[pos] = candidate
        except Exception as exc:
            print(f"  [VI Compression] strict retry failed: {exc}")

    return results


def compress_vi_entries(entries: List[dict]) -> List[dict]:
    """Compress overlong Vietnamese cues in *entries* (timing unchanged)."""
    from .config import load_env

    load_env()
    if not VI_COMPRESSION_ENABLED:
        print("[VI Compression] disabled")
        return list(entries)

    n = len(entries)
    durations = [
        _parse_ts(e["end_str"]) - _parse_ts(e["start_str"]) for e in entries
    ]
    texts = [e.get("text", "").strip() for e in entries]

    pending_indices: List[int] = []
    for i, text in enumerate(texts):
        if not text:
            continue
        rule_text = _apply_rule_compression(text)
        if rule_text != text:
            texts[i] = rule_text
            text = rule_text
        if _needs_compression(text, durations[i]) or _has_awkward_phrasing(text):
            pending_indices.append(i)

    if not pending_indices:
        print(f"[VI Compression] total={n} | changed=0")
        return [{**entry, "text": texts[i]} for i, entry in enumerate(entries)]

    changed = 0
    batch_size = 20
    for start in range(0, len(pending_indices), batch_size):
        chunk_indices = pending_indices[start:start + batch_size]
        batch = [(i, texts[i], durations[i]) for i in chunk_indices]
        originals = [texts[i] for i in chunk_indices]
        compressed = _compress_batch(batch, originals, durations)

        for entry_idx, new_text in zip(chunk_indices, compressed):
            if not new_text:
                continue
            old_text = texts[entry_idx]
            if new_text != old_text:
                changed += 1
                print(
                    f"  [VI Compression] cue {entry_idx + 1} | "
                    f"dur={durations[entry_idx]:.2f}s | "
                    f"cps {_cps(old_text, durations[entry_idx]):.1f}→"
                    f"{_cps(new_text, durations[entry_idx]):.1f}"
                )
                print(f"    before: {old_text}")
                print(f"    after:  {new_text}")
                texts[entry_idx] = new_text

    still_high = sum(
        1
        for i in range(n)
        if texts[i] and _cps(texts[i], durations[i]) > VI_COMPRESSION_MAX_CPS
    )
    print(
        f"[VI Compression] total={n} | changed={changed} | "
        f"still_above_max_cps={still_high}"
    )

    return [{**entry, "text": texts[i]} for i, entry in enumerate(entries)]


def compress_vi_srt_file(
    input_srt_path: str,
    output_srt_path: Optional[str] = None,
) -> str:
    """Compress an SRT file in place or to *output_srt_path*."""
    from .utils import parse_srt, write_srt_entries

    with open(input_srt_path, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    compressed = compress_vi_entries(entries)
    out_path = output_srt_path or input_srt_path

    if output_srt_path is None:
        srt_dir = os.path.dirname(os.path.abspath(input_srt_path))
        fd, tmp_path = tempfile.mkstemp(suffix=".srt", dir=srt_dir)
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                write_srt_entries(compressed, file=f)
            os.replace(tmp_path, input_srt_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return input_srt_path

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        write_srt_entries(compressed, file=f)
    return out_path
