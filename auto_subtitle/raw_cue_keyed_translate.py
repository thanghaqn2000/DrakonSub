"""Cue-keyed raw translation — one VI string per source cue_index, strict JSON mapping."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from .config import get_openai_model, get_translation_batch_size, llm_temperature
from .openai_chat import create_chat_completion
from .openai_translate import _build_openai_raw_system_prompt, _parse_ts, _validate_translation_output
from .raw_llm_response_cache import raw_llm_complete
from .translation_topics import normalize_topic

_CUE_KEYED_SYSTEM_EXTRA = """
CUE-KEYED TRANSLATION MODE (strict):
- Translate each subtitle cue independently.
- Neighboring cues are context ONLY (pronouns, fragments) — never import their meaning.
- Do not merge, split, or reorder cues.
- Do not add information not present in the source cue.
- If the source cue is a short fragment, keep the Vietnamese as a fragment.
- Return JSON only with cue_index matching each input cue exactly.
"""

_CUE_KEYED_BATCH_SIZE = 8


def _strip_json_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return content


def parse_cue_keyed_translations(
    content: str,
    expected_cue_indexes: List[int],
) -> Dict[int, str]:
    """Parse and validate cue_index-keyed translation JSON."""
    data = json.loads(_strip_json_fence(content))
    items = data.get("translations")
    if not isinstance(items, list):
        raise ValueError("Response missing 'translations' array")

    expected = set(expected_cue_indexes)
    by_index: Dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each translation must be an object")
        cue_index = item.get("cue_index")
        if not isinstance(cue_index, int):
            raise ValueError(f"Invalid cue_index {cue_index!r}")
        if cue_index in by_index:
            raise ValueError(f"Duplicate cue_index {cue_index}")
        vi = str(item.get("vi", item.get("text", ""))).strip()
        by_index[cue_index] = vi

    extra = set(by_index) - expected
    if extra:
        raise ValueError(f"Unexpected cue_index values: {sorted(extra)}")
    missing = expected - set(by_index)
    if missing:
        raise ValueError(f"Missing cue_index values: {sorted(missing)}")
    return by_index


def _build_cue_keyed_user_prompt(
    cue_specs: List[dict],
    target_lang: str,
    *,
    repair: bool = False,
) -> str:
    lines = [
        f"Translate exactly {len(cue_specs)} English subtitle cues into natural {target_lang}.",
        "",
        "Rules:",
        "- Return exactly one Vietnamese string per cue_index.",
        "- Do not merge cues or move meaning between cue_index values.",
        "- Use prev_source/next_source only to resolve pronouns or incomplete phrases.",
        "- Keep fragments short if the source is a fragment.",
        "",
        "cues:",
    ]
    for spec in cue_specs:
        parts = [
            f"cue_index={spec['cue_index']}",
            f"start={spec.get('start', '')}",
            f"end={spec.get('end', '')}",
            f"source={json.dumps(spec['source'], ensure_ascii=False)}",
        ]
        if spec.get("prev_source"):
            parts.append(f"prev_source={json.dumps(spec['prev_source'], ensure_ascii=False)}")
        if spec.get("next_source"):
            parts.append(f"next_source={json.dumps(spec['next_source'], ensure_ascii=False)}")
        lines.append("- " + " | ".join(parts))

    lines.extend(
        [
            "",
            "Output JSON only:",
            '{"translations": [{"cue_index": <int>, "vi": "<vietnamese>"}, ...]}',
            f"Must include cue_index for all: {[s['cue_index'] for s in cue_specs]}",
        ]
    )
    if repair:
        lines.append(
            "REPAIR: previous response failed validation. "
            "Return all cue_index values with non-empty vi for non-empty source."
        )
    return "\n".join(lines)


def _cue_specs_for_batch(
    entries: List[dict],
    entry_indexes: List[int],
) -> List[dict]:
    texts = [e.get("text", "").strip() for e in entries]
    specs: List[dict] = []
    for entry_idx in entry_indexes:
        cue_index = entry_idx + 1
        prev_source = texts[entry_idx - 1] if entry_idx > 0 else ""
        next_source = texts[entry_idx + 1] if entry_idx + 1 < len(texts) else ""
        specs.append(
            {
                "cue_index": cue_index,
                "start": entries[entry_idx].get("start_str", ""),
                "end": entries[entry_idx].get("end_str", ""),
                "source": texts[entry_idx],
                "prev_source": prev_source,
                "next_source": next_source,
            }
        )
    return specs


def _call_cue_keyed_batch(
    client,
    model: str,
    entries: List[dict],
    entry_indexes: List[int],
    target_lang: str,
    topic: str,
    *,
    repair: bool = False,
) -> Dict[int, str]:
    specs = _cue_specs_for_batch(entries, entry_indexes)
    cue_indexes = [s["cue_index"] for s in specs]
    sources = [s["source"] for s in specs]

    system = _build_openai_raw_system_prompt(topic) + "\n\n" + _CUE_KEYED_SYSTEM_EXTRA
    user = _build_cue_keyed_user_prompt(specs, target_lang, repair=repair)

    content = raw_llm_complete(
        client,
        model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        llm_task_type="raw_cue_keyed_translate",
        batch_indices=cue_indexes,
        source_texts=sources,
        repair=repair,
        temperature=0.25 if repair else 0.35,
        response_format={"type": "json_object"},
    )
    parsed = parse_cue_keyed_translations(
        content,
        cue_indexes,
    )
    ordered = [parsed[i] for i in cue_indexes]
    _validate_translation_output(ordered, sources)
    return parsed


def translate_single_cue_keyed(
    client,
    model: str,
    entries: List[dict],
    entry_idx: int,
    target_lang: str,
    topic: str,
) -> str:
    """Repair path: translate one cue with local context only."""
    parsed = _call_cue_keyed_batch(
        client,
        model,
        entries,
        [entry_idx],
        target_lang,
        topic,
    )
    return parsed[entry_idx + 1]


def translate_srt_entries_cue_keyed_openai(
    entries: List[dict],
    target_lang: str = "vi",
    model: Optional[str] = None,
    topic: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, object]]:
    """Translate all entries using strict cue_index-keyed batches."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    topic = normalize_topic(topic or os.environ.get("TRANSLATION_TOPIC"))
    client = OpenAI(api_key=api_key)
    model = model or get_openai_model()
    batch_size = min(batch_size or _CUE_KEYED_BATCH_SIZE, get_translation_batch_size())

    non_empty_entry_indexes = [
        i for i, e in enumerate(entries) if e.get("text", "").strip()
    ]
    translated: Dict[int, str] = {i: entries[i].get("text", "") for i in range(len(entries))}
    stats = {"batches": 0, "single_cue_fallbacks": 0, "batch_failures": 0}

    for start in range(0, len(non_empty_entry_indexes), batch_size):
        batch_entry_idxs = non_empty_entry_indexes[start : start + batch_size]
        first = batch_entry_idxs[0] + 1
        last = batch_entry_idxs[-1] + 1
        print(f"  [cue_keyed] translating cues {first}-{last} ({len(batch_entry_idxs)} cues)…")
        stats["batches"] += 1
        try:
            parsed = _call_cue_keyed_batch(
                client,
                model,
                entries,
                batch_entry_idxs,
                target_lang,
                topic,
            )
        except Exception as exc:
            print(f"  [cue_keyed] batch failed ({exc}), per-cue fallback…")
            stats["batch_failures"] += 1
            parsed = {}
            for entry_idx in batch_entry_idxs:
                try:
                    parsed[entry_idx + 1] = translate_single_cue_keyed(
                        client, model, entries, entry_idx, target_lang, topic
                    )
                    stats["single_cue_fallbacks"] += 1
                except Exception as exc2:
                    print(f"  [cue_keyed] cue {entry_idx + 1} failed ({exc2}), keeping EN")
                    parsed[entry_idx + 1] = entries[entry_idx].get("text", "").strip()

        for entry_idx in batch_entry_idxs:
            translated[entry_idx] = parsed.get(entry_idx + 1, entries[entry_idx].get("text", ""))

    result = [{**entry, "text": translated[i]} for i, entry in enumerate(entries)]
    return result, stats
