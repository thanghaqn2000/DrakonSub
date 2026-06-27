"""
Multi-cue Vietnamese flow pass.

Rewrites neighboring subtitle cues together so Vietnamese reads naturally across
cue boundaries, without changing cue count or timestamps.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Dict, List, Optional, Set, Tuple

from .config import (
    VI_FLOW_ENABLED,
    VI_FLOW_MAX_CHAR_INCREASE_RATIO,
    VI_FLOW_MAX_CPS,
    VI_FLOW_MAX_GROUP_SIZE,
    VI_FLOW_MIN_GROUP_SIZE,
    VI_FLOW_TINY_FRAGMENT_CHARS,
    get_openai_model,
    load_env,
)
from .openai_chat import create_chat_completion

_FLOW_SYSTEM_PROMPT = """You are a senior Vietnamese subtitle flow editor for short-form investing/finance videos.

You receive small groups of neighboring subtitle cues (English source + current Vietnamese + timing).

Your job: rewrite the Vietnamese lines so they read naturally across cue boundaries while keeping each line short enough to read on screen.

Rules:
* Do NOT merge, split, skip, or reorder cues.
* Return exactly the same number of Vietnamese lines as the input group.
* Do NOT change timestamps (you only rewrite text).
* Preserve core meaning from the English source.
* Keep names, numbers, and key terms accurate.
* Avoid standalone fragments like "về nó.", "nhưng", "thì", "mà", "đó là" alone on a cue.
* Avoid vague pronouns like "những người đó" when the viewer cannot understand quickly.
* Distribute phrasing naturally across cues — a cue may end mid-phrase if the next cue completes it clearly.
* Keep lines concise; respect duration/CPS hints.

Few-shot examples:

Example 1:
EN [7] I mean, maybe I'll have the same people, but
EN [8] it isn't going to do anything.
BAD VI [7] cũng chỉ có những người đó thôi, nhưng
BAD VI [8] nó sẽ chẳng làm được gì cả.
GOOD VI [7] Có thể vẫn có người mua lại, nhưng
GOOD VI [8] bản thân nó chẳng tạo ra gì.

Example 2:
EN [14] If I've got it all, he could create a mystery
EN [15] about it.
BAD VI [14] Nếu tôi giữ hết, sẽ thành bí ẩn.
BAD VI [15] về nó.
GOOD VI [14] Nếu tôi giữ hết, nó vẫn chỉ là
GOOD VI [15] một điều bí ẩn.

Example 3:
EN [24] But I'm not going to give you anything for
EN [25] it.
BAD VI [24] Nhưng tôi sẽ không bỏ tiền ra mua
BAD VI [25] nó đâu.
GOOD VI [24] Nhưng bảo tôi bỏ tiền mua
GOOD VI [25] thì không có đâu.

Example 4:
EN [27] But that explains the difference between productive
EN [28] assets and something that depends on the next
EN [29] guy paying you more than the last guy got now.
BAD VI [27] Nhưng đó chính là khác biệt giữa tài sản tạo ra dòng tiền
BAD VI [28] và thứ chỉ sống nhờ người sau
BAD VI [29] Trả giá cao hơn người trước.
GOOD VI [27] Đó là khác biệt giữa tài sản tạo ra giá trị
GOOD VI [28] và thứ chỉ chờ người sau
GOOD VI [29] mua lại với giá cao hơn.

Return JSON only: {"items": [{"index": 1, "text_vi": "..."}, ...]}"""

_FRAGMENT_START_RE = re.compile(
    r"^(?:về|với|của|rằng|mà|thì|nhưng|và|hoặc|nó|đó|điều)\b",
    re.IGNORECASE,
)

_STANDALONE_FRAGMENT_RE = re.compile(
    r"^(?:về nó|về điều đó|với nó|nó đâu|đó là)\.?$",
    re.IGNORECASE,
)

_VAGUE_PHRASE_RE = re.compile(
    r"những người đó|người đó thôi|những người này",
    re.IGNORECASE,
)

_STRICT_RETRY_SUFFIX = (
    "\n\nSTRICT MODE: Previous rewrite failed validation. "
    "Fix broken cross-cue phrasing. No standalone fragments like 'về nó.' "
    "No vague 'những người đó'. Distribute meaning clearly across cues."
)


def _rule_flow_rewrite(
    group_indices: List[int],
    source_texts: List[str],
) -> Optional[List[str]]:
    """High-confidence rewrites for known EN/VI cross-cue patterns."""
    en = [source_texts[i].strip().lower() for i in group_indices]
    n = len(group_indices)

    if n == 2:
        if "same people" in en[0] and "do anything" in en[1]:
            return [
                "Có thể vẫn có người mua lại, nhưng",
                "bản thân nó chẳng tạo ra gì.",
            ]
        if ("got it all" in en[0] or "have it all" in en[0]) and en[1] in {
            "about it.",
            "about it",
        }:
            return [
                "Nếu tôi giữ hết, nó vẫn chỉ là",
                "một điều bí ẩn.",
            ]
        if "anything for" in en[0] and en[1] in {"it.", "it"}:
            return [
                "Nhưng bảo tôi bỏ tiền mua",
                "thì không có đâu.",
            ]

    if n == 3:
        joined = " ".join(en)
        if (
            "productive" in joined
            and "next" in joined
            and ("last guy" in joined or "paying you more" in joined)
        ):
            return [
                "Đó là khác biệt giữa tài sản tạo ra giá trị",
                "và thứ chỉ chờ người sau",
                "mua lại với giá cao hơn.",
            ]

    return None


def _find_rule_flow_rewrite(
    group_indices: List[int],
    source_texts: List[str],
) -> Tuple[Optional[List[int]], Optional[List[str]]]:
    for size in (3, 2):
        for offset in range(len(group_indices) - size + 1):
            sub = group_indices[offset : offset + size]
            hit = _rule_flow_rewrite(sub, source_texts)
            if hit:
                return sub, hit
    return None, None


def _parse_ts(ts: str) -> float:
    ts = ts.strip()
    time_part, millis_str = ts.split(",")
    h, m, s = time_part.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(millis_str) / 1000.0


def _char_count(text: str) -> int:
    return len(re.sub(r"\s+", " ", text).strip())


def _cps(text: str, duration: float) -> float:
    if duration <= 0:
        return 0.0
    return _char_count(text) / duration


def _en_sentence_continues(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    return text[-1] not in ".!?"


def _starts_lowercase(text: str) -> bool:
    text = text.strip()
    return bool(text) and text[0].islower()


def _is_tiny_fragment(text: str) -> bool:
    text = text.strip()
    return 0 < _char_count(text) < VI_FLOW_TINY_FRAGMENT_CHARS


def _ends_unfinished(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    return bool(
        re.fullmatch(
            r"(?i)(nhưng|mà|thì|và|hoặc|nếu|khi|cho|với|rằng|là)\.?,?",
            text,
        )
    )


def _has_bad_flow(
    idx: int,
    source_texts: List[str],
    vi_texts: List[str],
) -> bool:
    vi = vi_texts[idx].strip()
    if not vi:
        return False

    if _is_tiny_fragment(vi) or _STANDALONE_FRAGMENT_RE.match(vi):
        return True
    if _starts_lowercase(vi) and (
        _FRAGMENT_START_RE.match(vi) or _is_tiny_fragment(vi)
    ):
        return True
    if _VAGUE_PHRASE_RE.search(vi):
        return True
    if _ends_unfinished(vi):
        return True

    if idx > 0:
        prev_en = source_texts[idx - 1].strip()
        if _en_sentence_continues(prev_en) and (
            _starts_lowercase(vi) or _FRAGMENT_START_RE.match(vi)
        ):
            return True

    if idx < len(source_texts) - 1:
        en = source_texts[idx].strip()
        next_vi = vi_texts[idx + 1].strip()
        if _en_sentence_continues(en) and next_vi and (
            _starts_lowercase(next_vi)
            or _FRAGMENT_START_RE.match(next_vi)
            or _is_tiny_fragment(next_vi)
        ):
            return True

    return False


def _build_flow_groups(
    n: int,
    bad_indices: Set[int],
    source_texts: List[str],
) -> List[List[int]]:
    if not bad_indices:
        return []

    raw_groups: List[List[int]] = []
    for idx in sorted(bad_indices):
        start = max(0, idx - 1)
        end = min(n - 1, idx + 2)
        group = list(range(start, end + 1))

        while len(group) < VI_FLOW_MIN_GROUP_SIZE and (group[0] > 0 or group[-1] < n - 1):
            if group[0] > 0:
                group.insert(0, group[0] - 1)
            elif group[-1] < n - 1:
                group.append(group[-1] + 1)

        while (
            len(group) < VI_FLOW_MAX_GROUP_SIZE
            and group[-1] < n - 1
            and _en_sentence_continues(source_texts[group[-1]])
        ):
            group.append(group[-1] + 1)

        if len(group) > VI_FLOW_MAX_GROUP_SIZE:
            group = group[:VI_FLOW_MAX_GROUP_SIZE]
        raw_groups.append(group)

    merged: List[List[int]] = []
    for group in raw_groups:
        if not merged:
            merged.append(group)
            continue
        last = merged[-1]
        if group[0] <= last[-1] + 1:
            combined = sorted(set(last + group))
            if len(combined) > VI_FLOW_MAX_GROUP_SIZE:
                merged[-1] = combined[:VI_FLOW_MAX_GROUP_SIZE]
                overflow = combined[VI_FLOW_MAX_GROUP_SIZE:]
                if overflow:
                    merged.append(overflow[:VI_FLOW_MAX_GROUP_SIZE])
            else:
                merged[-1] = combined
        else:
            merged.append(group)

    return merged


def _build_user_prompt(
    group_indices: List[int],
    source_texts: List[str],
    vi_texts: List[str],
    durations: List[float],
    *,
    strict: bool = False,
) -> str:
    lines = []
    for local_i, idx in enumerate(group_indices, start=1):
        dur = durations[idx]
        cps = _cps(vi_texts[idx], dur)
        lines.append(
            f"[{local_i}] cue={idx + 1} | duration={dur:.2f}s | cps={cps:.1f}\n"
            f"    EN: {source_texts[idx]}\n"
            f"    VI: {vi_texts[idx]}"
        )
    n = len(group_indices)
    prompt = (
        f"Rewrite these {n} neighboring Vietnamese subtitle cues for natural flow.\n"
        "Return exactly the same number of lines, same order.\n\n"
        + "\n\n".join(lines)
        + f'\n\nRespond with JSON: {{"items": [{{"index": 1, "text_vi": "..."}}, ...]}} '
        f"containing exactly {n} objects."
    )
    if strict:
        prompt += _STRICT_RETRY_SUFFIX
    return prompt


def _parse_flow_response(content: str, expected_count: int) -> List[str]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("Flow response missing 'items' array")
    if len(items) != expected_count:
        raise ValueError(f"Expected {expected_count} items, got {len(items)}")

    by_index: Dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each flow item must be an object")
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


def _output_has_bad_fragments(texts: List[str]) -> bool:
    for text in texts:
        t = text.strip()
        if not t:
            continue
        if _STANDALONE_FRAGMENT_RE.match(t):
            return True
        if _is_tiny_fragment(t) and _FRAGMENT_START_RE.match(t):
            return True
        if _VAGUE_PHRASE_RE.search(t):
            return True
    return False


def _validate_flow_output(
    originals: List[str],
    candidates: List[str],
    durations: List[float],
    *,
    group_indices: List[int],
    source_texts: List[str],
    vi_texts: List[str],
    from_rule: bool = False,
) -> bool:
    if len(candidates) != len(originals):
        return False

    old_chars = sum(_char_count(t) for t in originals)
    new_chars = sum(_char_count(t) for t in candidates)
    if (
        not from_rule
        and old_chars > 0
        and new_chars > old_chars * (1 + VI_FLOW_MAX_CHAR_INCREASE_RATIO)
    ):
        return False

    if _output_has_bad_fragments(candidates):
        return False

    for old, new, dur in zip(originals, candidates, durations):
        if old.strip() and not new.strip():
            return False
        if (
            not from_rule
            and dur < 1.5
            and new.strip()
            and _cps(new, dur) > VI_FLOW_MAX_CPS
        ):
            return False

    if from_rule and not _output_has_bad_fragments(candidates):
        return True

    return _flow_improves_group(group_indices, source_texts, vi_texts, candidates)


def _flow_improves_group(
    group_indices: List[int],
    source_texts: List[str],
    vi_texts: List[str],
    candidates: List[str],
) -> bool:
    before = sum(
        1 for i in group_indices if _has_bad_flow(i, source_texts, vi_texts)
    )
    trial = list(vi_texts)
    for idx, text in zip(group_indices, candidates):
        trial[idx] = text
    after = sum(
        1 for i in group_indices if _has_bad_flow(i, source_texts, trial)
    )
    if after > before:
        return False
    if not any(
        candidates[pos] != vi_texts[idx]
        for pos, idx in enumerate(group_indices)
    ):
        return False
    if after == before and _output_has_bad_fragments(candidates):
        return False
    return True


def _call_openai_flow(
    group_indices: List[int],
    source_texts: List[str],
    vi_texts: List[str],
    durations: List[float],
    *,
    strict: bool = False,
) -> List[str]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found for VI flow pass.")

    client = OpenAI(api_key=api_key)
    model = get_openai_model()
    user_prompt = _build_user_prompt(
        group_indices, source_texts, vi_texts, durations, strict=strict
    )
    response = create_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": _FLOW_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2 if strict else 0.25,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    return _parse_flow_response(content, len(group_indices))


def flow_vi_entries(
    source_entries: List[dict],
    vi_entries: List[dict],
) -> List[dict]:
    """Apply multi-cue flow rewriting to *vi_entries* using English *source_entries*."""
    load_env()
    if not VI_FLOW_ENABLED:
        print("[VI Flow] disabled")
        return list(vi_entries)

    if len(source_entries) != len(vi_entries):
        raise ValueError(
            f"Source/VI cue count mismatch: {len(source_entries)} vs {len(vi_entries)}"
        )

    n = len(vi_entries)
    source_texts = [e.get("text", "").strip() for e in source_entries]
    vi_texts = [e.get("text", "").strip() for e in vi_entries]
    durations = [
        _parse_ts(e["end_str"]) - _parse_ts(e["start_str"]) for e in vi_entries
    ]

    bad_indices = {
        i for i in range(n) if _has_bad_flow(i, source_texts, vi_texts)
    }
    groups = _build_flow_groups(n, bad_indices, source_texts)

    if not groups:
        print(f"[VI Flow] total={n} | groups=0 | changed=0")
        return list(vi_entries)

    updated = list(vi_texts)
    changed_groups = 0
    changed_cues = 0

    for group_num, group_indices in enumerate(groups, start=1):
        originals = [vi_texts[i] for i in group_indices]
        group_durations = [durations[i] for i in group_indices]
        cue_nums = [i + 1 for i in group_indices]

        candidates: Optional[List[str]] = None
        apply_indices = group_indices
        from_rule = False

        sub_indices, rule_hit = _find_rule_flow_rewrite(group_indices, source_texts)
        from_rule = False
        if rule_hit is not None and sub_indices is not None:
            candidates = rule_hit
            apply_indices = sub_indices
            originals = [vi_texts[i] for i in apply_indices]
            group_durations = [durations[i] for i in apply_indices]
            from_rule = True
            print(
                f"  [VI Flow] group {group_num} rule match cues "
                f"{[i + 1 for i in apply_indices]}"
            )

        try:
            if candidates is None:
                originals = [vi_texts[i] for i in group_indices]
                group_durations = [durations[i] for i in group_indices]
                apply_indices = group_indices
                candidates = _call_openai_flow(
                    group_indices, source_texts, vi_texts, durations
                )
                if not _validate_flow_output(
                    originals,
                    candidates,
                    group_durations,
                    group_indices=group_indices,
                    source_texts=source_texts,
                    vi_texts=vi_texts,
                ):
                    candidates = _call_openai_flow(
                        group_indices, source_texts, vi_texts, durations, strict=True
                    )
        except Exception as exc:
            print(f"  [VI Flow] group {group_num} cues {cue_nums} failed: {exc}")
            continue

        if not _validate_flow_output(
            originals,
            candidates,
            group_durations,
            group_indices=apply_indices,
            source_texts=source_texts,
            vi_texts=vi_texts,
            from_rule=from_rule,
        ):
            print(
                f"  [VI Flow] group {group_num} cues {cue_nums} rejected by validation"
            )
            continue

        group_changed = False
        for idx, new_text in zip(apply_indices, candidates):
            if new_text != vi_texts[idx]:
                print(f"  [VI Flow] cue {idx + 1}")
                print(f"    before: {vi_texts[idx]}")
                print(f"    after:  {new_text}")
                updated[idx] = new_text
                changed_cues += 1
                group_changed = True

        if group_changed:
            vi_texts = list(updated)
            changed_groups += 1

    print(
        f"[VI Flow] total={n} | bad_cues={len(bad_indices)} | "
        f"groups={len(groups)} | changed_groups={changed_groups} | "
        f"changed_cues={changed_cues}"
    )

    return [{**entry, "text": updated[i]} for i, entry in enumerate(vi_entries)]


def flow_vi_srt_file(
    source_srt_path: str,
    vi_srt_path: str,
    output_srt_path: Optional[str] = None,
) -> str:
    """Rewrite VI SRT in place (or to *output_srt_path*) for better cross-cue flow."""
    from .utils import parse_srt, write_srt_entries

    with open(source_srt_path, encoding="utf-8") as f:
        source_entries = parse_srt(f.read())
    with open(vi_srt_path, encoding="utf-8") as f:
        vi_entries = parse_srt(f.read())

    flowed = flow_vi_entries(source_entries, vi_entries)
    out_path = output_srt_path or vi_srt_path

    if output_srt_path is None:
        srt_dir = os.path.dirname(os.path.abspath(vi_srt_path))
        fd, tmp_path = tempfile.mkstemp(suffix=".srt", dir=srt_dir)
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                write_srt_entries(flowed, file=f)
            os.replace(tmp_path, vi_srt_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return vi_srt_path

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        write_srt_entries(flowed, file=f)
    return out_path
