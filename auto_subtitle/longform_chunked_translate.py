"""Long-form chunked raw translation with local context and chunk verification."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import get_openai_model
from .openai_translate import _build_openai_raw_system_prompt, _validate_translation_output
from .raw_llm_response_cache import raw_llm_complete
from .translation_topics import normalize_topic

_SENTENCE_END_CHARS = frozenset(".!?…")
_DEFAULT_TARGET_CHUNK_SIZE = 12
_DEFAULT_MIN_CHUNK_SIZE = 6
_DEFAULT_MAX_CHUNK_SIZE = 18
_OVERLAP_CONTEXT_CUES = 2
_REPORT_PATH = Path("artifacts/translation_quality_review/longform_chunked_strategy_v1.json")
_REPORT_MD_PATH = Path("artifacts/translation_quality_review/longform_chunked_strategy_v1.md")


@dataclass
class LongformChunk:
    chunk_id: str
    entry_indexes: List[int]
    prev_context_indexes: List[int]
    next_context_indexes: List[int]
    previous_chunk_summary_en: str = ""
    previous_chunk_summary_vi: str = ""


def _strip_json_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return content


def _cue_text(entry: dict) -> str:
    return str(entry.get("text", "")).strip()


def _is_sentence_end(text: str) -> bool:
    inner = text.strip().rstrip("\"')}]")
    return bool(inner and inner[-1] in _SENTENCE_END_CHARS)


def _extractive_chunk_summary(entries: List[dict], entry_indexes: List[int]) -> Tuple[str, str]:
    texts = [_cue_text(entries[i]) for i in entry_indexes if _cue_text(entries[i])]
    if not texts:
        return "", ""
    first = texts[0]
    second = texts[1] if len(texts) > 1 else ""
    summary_en = first if not second else f"{first} {second}"
    if len(summary_en) > 220:
        summary_en = summary_en[:217].rstrip() + "..."
    # If the model does not return a VI summary, keep an empty or mirrored short placeholder.
    return summary_en, ""


def build_longform_chunks(
    entries: List[dict],
    *,
    target_chunk_size: int = _DEFAULT_TARGET_CHUNK_SIZE,
    min_chunk_size: int = _DEFAULT_MIN_CHUNK_SIZE,
    max_chunk_size: int = _DEFAULT_MAX_CHUNK_SIZE,
) -> List[LongformChunk]:
    non_empty = [i for i, entry in enumerate(entries) if _cue_text(entry)]
    if not non_empty:
        return []

    chunks: List[LongformChunk] = []
    start = 0
    while start < len(non_empty):
        remaining = len(non_empty) - start
        size = min(target_chunk_size, remaining)
        size = max(min_chunk_size if remaining > min_chunk_size else remaining, size)
        size = min(size, max_chunk_size)
        if remaining <= max_chunk_size:
            size = remaining

        chosen_end = start + size
        lower_bound = min(start + min_chunk_size, len(non_empty))
        upper_bound = min(start + max_chunk_size, len(non_empty))
        for probe in range(lower_bound, upper_bound):
            idx = non_empty[probe - 1]
            if _is_sentence_end(_cue_text(entries[idx])):
                chosen_end = probe
                if probe >= start + target_chunk_size:
                    break
        entry_indexes = non_empty[start:chosen_end]
        prev_context = non_empty[max(0, start - _OVERLAP_CONTEXT_CUES):start]
        next_context = non_empty[chosen_end:chosen_end + _OVERLAP_CONTEXT_CUES]
        chunks.append(
            LongformChunk(
                chunk_id=f"chunk_{len(chunks) + 1:03d}",
                entry_indexes=entry_indexes,
                prev_context_indexes=prev_context,
                next_context_indexes=next_context,
            )
        )
        start = chosen_end
    return chunks


def parse_longform_chunk_response(
    content: str,
    *,
    expected_cue_indexes: List[int],
    fallback_by_index: Optional[Dict[int, str]] = None,
) -> dict:
    data = json.loads(_strip_json_fence(content))
    items = data.get("translations")
    if not isinstance(items, list):
        raise ValueError("Response missing 'translations' array")

    expected = set(expected_cue_indexes)
    translated: Dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each translation must be an object")
        cue_index = item.get("cue_index")
        if not isinstance(cue_index, int):
            raise ValueError(f"Invalid cue_index {cue_index!r}")
        if cue_index in translated:
            raise ValueError(f"Duplicate cue_index {cue_index}")
        vi = str(item.get("vi", item.get("text", ""))).strip()
        if not vi and fallback_by_index:
            vi = str(fallback_by_index.get(cue_index, "")).strip()
        if not vi:
            raise ValueError(f"Empty vi for cue_index {cue_index}")
        translated[cue_index] = vi

    extra = set(translated) - expected
    if extra:
        raise ValueError(f"Unexpected cue_index values: {sorted(extra)}")
    missing = expected - set(translated)
    if missing:
        raise ValueError(f"Missing cue_index values: {sorted(missing)}")

    return {
        "translations": translated,
        "chunk_summary_en": str(data.get("chunk_summary_en", "")).strip(),
        "chunk_summary_vi": str(data.get("chunk_summary_vi", "")).strip(),
    }


def _parse_verifier_response(content: str, cue_indexes: List[int]) -> dict:
    data = json.loads(_strip_json_fence(content))
    status = str(data.get("chunk_status", "pass")).strip().lower()
    if status not in {"pass", "needs_repair"}:
        raise ValueError(f"Invalid chunk_status: {status}")
    bad_cues = data.get("bad_cues") or []
    valid_bad = []
    allowed = set(cue_indexes)
    for cue in bad_cues:
        if not isinstance(cue, dict):
            continue
        cue_index = cue.get("cue_index")
        if cue_index not in allowed:
            continue
        valid_bad.append(
            {
                "cue_index": cue_index,
                "error": str(cue.get("error", "")).strip(),
                "severity": str(cue.get("severity", "low")).strip().lower(),
                "reason": str(cue.get("reason", "")).strip(),
            }
        )
    return {"chunk_status": status, "bad_cues": valid_bad}


def _chunk_specs(entries: List[dict], chunk: LongformChunk) -> List[dict]:
    specs = []
    for entry_idx in chunk.entry_indexes:
        specs.append(
            {
                "cue_index": entry_idx + 1,
                "start": entries[entry_idx].get("start_str", ""),
                "end": entries[entry_idx].get("end_str", ""),
                "source": _cue_text(entries[entry_idx]),
            }
        )
    return specs


def _context_lines(entries: List[dict], indexes: List[int]) -> List[str]:
    lines: List[str] = []
    for entry_idx in indexes:
        text = _cue_text(entries[entry_idx])
        if not text:
            continue
        lines.append(f"- cue_index={entry_idx + 1} | source={json.dumps(text, ensure_ascii=False)}")
    return lines or ["- (none)"]


def _build_longform_chunk_user_prompt(
    entries: List[dict],
    chunk: LongformChunk,
    target_lang: str,
    *,
    repair: bool = False,
) -> str:
    cue_specs = _chunk_specs(entries, chunk)
    lines = [
        f"Translate this subtitle chunk cue-by-cue into natural {target_lang}.",
        "",
        "Rules:",
        "- Return strict JSON keyed by cue_index.",
        "- Use previous/next context only to understand fragments.",
        "- Do not move meaning across cue_index values.",
        "- Do not import information from outside this chunk.",
        "- If a cue is a fragment, translate it as a fragment.",
        "- Preserve rhetorical repetition when source repeats.",
        "- If uncertain, stay literal rather than inventing context.",
        "",
        f"chunk_id={chunk.chunk_id}",
        f"previous_chunk_summary_en={json.dumps(chunk.previous_chunk_summary_en, ensure_ascii=False)}",
        f"previous_chunk_summary_vi={json.dumps(chunk.previous_chunk_summary_vi, ensure_ascii=False)}",
        "",
        "previous_context:",
        *_context_lines(entries, chunk.prev_context_indexes),
        "",
        "current_chunk:",
    ]
    for spec in cue_specs:
        lines.append(
            "- "
            + " | ".join(
                [
                    f"cue_index={spec['cue_index']}",
                    f"start={spec['start']}",
                    f"end={spec['end']}",
                    f"source={json.dumps(spec['source'], ensure_ascii=False)}",
                ]
            )
        )
    lines.extend(
        [
            "",
            "next_context:",
            *_context_lines(entries, chunk.next_context_indexes),
            "",
            "Output JSON only:",
            '{"translations": [{"cue_index": <int>, "vi": "<vietnamese>", "confidence": "high|medium|low", "notes": ""}], "chunk_summary_vi": "...", "chunk_summary_en": "..."}',
            f"Must include cue_index for all: {[spec['cue_index'] for spec in cue_specs]}",
        ]
    )
    if repair:
        lines.append(
            "REPAIR MODE: previous chunk output had semantic drift, hallucination, alignment bleed, or missing cue mapping."
        )
    return "\n".join(lines)


def _build_verifier_user_prompt(entries: List[dict], chunk: LongformChunk, vi_map: Dict[int, str]) -> str:
    lines = [
        "Verify this translated subtitle chunk using LOCAL context only.",
        "Detect cue meaning assigned to wrong cue, hallucinated entity/event, far-context bleed, unrelated fragment expansion, or unsupported repeated translation.",
        "Only mark needs_repair when there is a real semantic or alignment problem.",
        "",
        f"chunk_id={chunk.chunk_id}",
        "source_cues:",
    ]
    for entry_idx in chunk.entry_indexes:
        lines.append(
            "- "
            + " | ".join(
                [
                    f"cue_index={entry_idx + 1}",
                    f"source={json.dumps(_cue_text(entries[entry_idx]), ensure_ascii=False)}",
                    f"vi={json.dumps(vi_map.get(entry_idx + 1, ''), ensure_ascii=False)}",
                ]
            )
        )
    lines.extend(
        [
            "",
            'Return JSON only: {"chunk_status":"pass|needs_repair","bad_cues":[{"cue_index":41,"error":"semantic_drift|alignment|hallucination|neighbor_bleed","severity":"low|medium|high","reason":"..."}]}',
        ]
    )
    return "\n".join(lines)


def _write_strategy_report(stats: dict) -> None:
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# Longform Chunked Strategy v1",
        "",
        f"- chunk_count: {stats.get('chunk_count', 0)}",
        f"- chunk_size_stats: {stats.get('chunk_size_stats', {})}",
        f"- raw_llm_calls: {stats.get('raw_llm_calls', 0)}",
        f"- verifier_calls: {stats.get('verifier_calls', 0)}",
        f"- repair_calls: {stats.get('repair_calls', 0)}",
        f"- cache_hits: {stats.get('cache_hits', 0)}",
        f"- cache_misses: {stats.get('cache_misses', 0)}",
        "",
        "## Chunks",
        "",
    ]
    for chunk in stats.get("chunks", []):
        md_lines.append(
            f"- {chunk['chunk_id']}: cues {chunk['cue_start']}-{chunk['cue_end']} | "
            f"verifier={chunk['verifier_status']} | bad_cues={chunk['bad_cue_count']} | "
            f"repair_attempted={chunk['repair_attempted']}"
        )
    _REPORT_MD_PATH.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def translate_srt_entries_longform_chunked_openai(
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
    del batch_size, polish, translation_context, strict_cue_count
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    client = OpenAI(api_key=api_key)
    model = model or get_openai_model()
    topic = normalize_topic(topic or os.environ.get("TRANSLATION_TOPIC"))
    system_prompt = _build_openai_raw_system_prompt(topic)
    verifier_system = (
        "You are a strict subtitle alignment verifier. Use only local chunk evidence. "
        "Do not guess. Return JSON only."
    )

    chunks = build_longform_chunks(entries)
    translated_by_index: Dict[int, str] = {i: entry.get("text", "") for i, entry in enumerate(entries)}
    previous_summary_en = ""
    previous_summary_vi = ""
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_translation_mode": "longform_chunked",
        "chunk_count": len(chunks),
        "chunk_size_stats": {
            "min": min((len(chunk.entry_indexes) for chunk in chunks), default=0),
            "max": max((len(chunk.entry_indexes) for chunk in chunks), default=0),
            "avg": round(
                sum(len(chunk.entry_indexes) for chunk in chunks) / max(len(chunks), 1), 2
            ),
        },
        "raw_llm_calls": 0,
        "verifier_calls": 0,
        "repair_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "chunks": [],
    }

    for chunk in chunks:
        chunk.previous_chunk_summary_en = previous_summary_en
        chunk.previous_chunk_summary_vi = previous_summary_vi
        cue_indexes = [idx + 1 for idx in chunk.entry_indexes]
        source_texts = [_cue_text(entries[idx]) for idx in chunk.entry_indexes]
        prompt = _build_longform_chunk_user_prompt(entries, chunk, target_lang)
        cache_scope = {
            "chunk_id": chunk.chunk_id,
            "chunk_cue_indices": cue_indexes,
            "previous_chunk_summary_hash": chunk.previous_chunk_summary_en,
            "llm_task_type": "longform_chunk_translate",
        }
        content = raw_llm_complete(
            client,
            model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            llm_task_type="longform_chunk_translate",
            batch_indices=cue_indexes,
            source_texts=source_texts,
            temperature=0.2,
            response_format={"type": "json_object"},
            cache_scope=cache_scope,
        )
        stats["raw_llm_calls"] += 1
        parsed = parse_longform_chunk_response(
            content,
            expected_cue_indexes=cue_indexes,
            fallback_by_index={cue_index: text for cue_index, text in zip(cue_indexes, source_texts)},
        )
        ordered = [parsed["translations"][cue_index] for cue_index in cue_indexes]
        _validate_translation_output(ordered, source_texts)

        verify_prompt = _build_verifier_user_prompt(entries, chunk, parsed["translations"])
        verify_content = raw_llm_complete(
            client,
            model,
            messages=[
                {"role": "system", "content": verifier_system},
                {"role": "user", "content": verify_prompt},
            ],
            llm_task_type="longform_chunk_verify",
            batch_indices=cue_indexes,
            source_texts=source_texts,
            temperature=0.0,
            response_format={"type": "json_object"},
            cache_scope={
                "chunk_id": chunk.chunk_id,
                "chunk_cue_indices": cue_indexes,
                "llm_task_type": "longform_chunk_verify",
            },
        )
        stats["verifier_calls"] += 1
        verify = _parse_verifier_response(verify_content, cue_indexes)

        repair_attempted = False
        accepted_repairs = 0
        medium_or_high = [
            bad for bad in verify["bad_cues"] if bad.get("severity") in {"medium", "high"}
        ]
        if verify["chunk_status"] == "needs_repair" and medium_or_high:
            repair_attempted = True
            stats["repair_calls"] += 1
            repair_prompt = _build_longform_chunk_user_prompt(entries, chunk, target_lang, repair=True)
            repair_content = raw_llm_complete(
                client,
                model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": repair_prompt},
                ],
                llm_task_type="longform_chunk_repair",
                batch_indices=cue_indexes,
                source_texts=source_texts,
                repair=True,
                temperature=0.1,
                response_format={"type": "json_object"},
                cache_scope={
                    "chunk_id": chunk.chunk_id,
                    "chunk_cue_indices": cue_indexes,
                    "llm_task_type": "longform_chunk_repair",
                },
            )
            repaired = parse_longform_chunk_response(
                repair_content,
                expected_cue_indexes=cue_indexes,
                fallback_by_index={cue_index: text for cue_index, text in zip(cue_indexes, source_texts)},
            )
            repaired_verify_prompt = _build_verifier_user_prompt(entries, chunk, repaired["translations"])
            repaired_verify_content = raw_llm_complete(
                client,
                model,
                messages=[
                    {"role": "system", "content": verifier_system},
                    {"role": "user", "content": repaired_verify_prompt},
                ],
                llm_task_type="longform_chunk_verify",
                batch_indices=cue_indexes,
                source_texts=source_texts,
                temperature=0.0,
                response_format={"type": "json_object"},
                cache_scope={
                    "chunk_id": chunk.chunk_id,
                    "chunk_cue_indices": cue_indexes,
                    "llm_task_type": "longform_chunk_verify_after_repair",
                },
            )
            stats["verifier_calls"] += 1
            repaired_verify = _parse_verifier_response(repaired_verify_content, cue_indexes)
            if repaired_verify["chunk_status"] == "pass" or len(repaired_verify["bad_cues"]) < len(verify["bad_cues"]):
                parsed = repaired
                verify = repaired_verify
                accepted_repairs = 1

        for entry_idx in chunk.entry_indexes:
            translated_by_index[entry_idx] = parsed["translations"][entry_idx + 1]

        summary_en = parsed["chunk_summary_en"].strip()
        summary_vi = parsed["chunk_summary_vi"].strip()
        if not summary_en:
            summary_en, fallback_vi = _extractive_chunk_summary(entries, chunk.entry_indexes)
            if not summary_vi:
                summary_vi = fallback_vi
        previous_summary_en = summary_en
        previous_summary_vi = summary_vi

        stats["chunks"].append(
            {
                "chunk_id": chunk.chunk_id,
                "cue_start": cue_indexes[0],
                "cue_end": cue_indexes[-1],
                "chunk_size": len(cue_indexes),
                "verifier_status": verify["chunk_status"],
                "bad_cue_count": len(verify["bad_cues"]),
                "repair_attempted": repair_attempted,
                "repair_accepted": accepted_repairs,
                "dominant_failure_layer": "raw_translation" if verify["bad_cues"] else "none",
                "notes": "; ".join(
                    f"{bad['cue_index']}:{bad['error']}" for bad in verify["bad_cues"][:5]
                ),
            }
        )

    result = [{**entry, "text": translated_by_index[i]} for i, entry in enumerate(entries)]

    try:
        _write_strategy_report(stats)
    except Exception:
        pass

    return result
