"""Deterministic raw-stage LLM response cache for benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    benchmark_deterministic_enabled,
    get_openai_model,
    get_raw_translation_mode,
    llm_chat_kwargs,
    llm_temperature,
)
from .openai_chat import create_chat_completion

RAW_LLM_PROMPT_VERSION = "raw_v1"
CACHE_ROOT = Path("artifacts/multi_sample_benchmark/raw_llm_response_cache")
REPORT_PATH = Path("artifacts/translation_quality_review/raw_llm_response_cache_report.json")

_sample_id: ContextVar[Optional[str]] = ContextVar("raw_llm_sample_id", default=None)
_source_hash: ContextVar[str] = ContextVar("raw_llm_source_hash", default="")
_video_context_hash: ContextVar[str] = ContextVar("raw_llm_vctx_hash", default="")
_run_id: ContextVar[str] = ContextVar("raw_llm_run_id", default="")
_call_log: ContextVar[List[dict]] = ContextVar("raw_llm_call_log", default=[])


def raw_llm_cache_enabled() -> bool:
    raw = (os.getenv("RAW_LLM_RESPONSE_CACHE") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return False


def set_raw_llm_context(
    *,
    sample_id: Optional[str] = None,
    source_hash: str = "",
    video_context_hash: str = "",
    run_id: str = "",
) -> None:
    if sample_id is not None:
        _sample_id.set(sample_id)
    if source_hash:
        _source_hash.set(source_hash)
    if video_context_hash:
        _video_context_hash.set(video_context_hash)
    if run_id:
        _run_id.set(run_id)


def reset_call_log() -> None:
    _call_log.set([])


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _hash_messages(messages: List[dict]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return _hash_text(payload)


def _batch_source_hash(batch_indices: List[int], source_texts: Optional[List[str]]) -> str:
    if not source_texts:
        return ""
    return _hash_text("\n".join(source_texts))


def _normalize_cache_scope(cache_scope: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not cache_scope:
        return {}
    normalized: Dict[str, Any] = {}
    for key, value in sorted(cache_scope.items()):
        if isinstance(value, list):
            normalized[key] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def build_cache_key(
    *,
    llm_task_type: str,
    model: str,
    messages: List[dict],
    batch_indices: Optional[List[int]] = None,
    repair: bool = False,
    engine: str = "openai",
    cache_scope: Optional[Dict[str, Any]] = None,
) -> str:
    raw_mode = get_raw_translation_mode()
    normalized_scope = _normalize_cache_scope(cache_scope)
    parts = [
        _source_hash.get() or "no_source",
        _sample_id.get() or "no_sample",
        llm_task_type,
        engine,
        model.replace("/", "_"),
        RAW_LLM_PROMPT_VERSION,
        raw_mode,
        "det" if benchmark_deterministic_enabled() else "nondet",
        "repair" if repair else "base",
        _video_context_hash.get() or "no_vctx",
        _hash_messages(messages),
        ",".join(str(i) for i in (batch_indices or [])),
        json.dumps(normalized_scope, ensure_ascii=False, sort_keys=True),
    ]
    return _hash_text("|".join(parts))


def _cache_path(engine: str, model: str, raw_mode: str, cache_key: str) -> Path:
    return (
        CACHE_ROOT
        / engine
        / model.replace("/", "_")
        / raw_mode
        / f"{cache_key}.json"
    )


def _log_call(entry: dict) -> None:
    log = list(_call_log.get())
    log.append(entry)
    _call_log.set(log)


def raw_llm_complete(
    client,
    model: str,
    messages: List[dict],
    *,
    llm_task_type: str,
    batch_indices: Optional[List[int]] = None,
    source_texts: Optional[List[str]] = None,
    repair: bool = False,
    temperature: Optional[float] = None,
    cache_scope: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> str:
    """Call OpenAI for raw translation tasks with optional response cache."""
    engine = (os.getenv("TRANSLATION_ENGINE") or "openai").strip().lower()
    temp = llm_temperature(temperature if temperature is not None else 0.35)
    chat_extra = llm_chat_kwargs()
    cache_key = build_cache_key(
        llm_task_type=llm_task_type,
        model=model,
        messages=messages,
        batch_indices=batch_indices,
        repair=repair,
        engine=engine,
        cache_scope=cache_scope,
    )
    normalized_scope = _normalize_cache_scope(cache_scope)
    raw_mode = get_raw_translation_mode()
    path = _cache_path(engine, model, raw_mode, cache_key)

    use_cache = raw_llm_cache_enabled() and benchmark_deterministic_enabled()
    if use_cache and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        content = (data.get("response") or {}).get("raw_text", "")
        _log_call(
            {
                "run_id": _run_id.get(),
                "sample_id": _sample_id.get(),
                "llm_task_type": llm_task_type,
                "batch_indices": batch_indices or [],
                "cache_key": cache_key,
                "cache_hit": True,
                "cache_written": False,
                "cache_path": str(path),
                "cache_scope": normalized_scope,
            }
        )
        return content

    response = create_chat_completion(
        client,
        model,
        messages,
        temperature=temp,
        **chat_extra,
        **kwargs,
    )
    content = response.choices[0].message.content or ""

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "cache_key": cache_key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine": engine,
            "model": model,
            "raw_translation_mode": raw_mode,
            "llm_task_type": llm_task_type,
            "sample_id": _sample_id.get(),
            "batch_indices": batch_indices or [],
            "prompt_hash": _hash_messages(messages),
            "input_hash": _batch_source_hash(batch_indices or [], source_texts),
            "video_context_hash": _video_context_hash.get(),
            "cache_scope": normalized_scope,
            "request": {
                "messages_or_prompt_hash": _hash_messages(messages),
                "temperature": temp,
                "top_p": chat_extra.get("top_p", 1),
                "seed": chat_extra.get("seed"),
            },
            "response": {"raw_text": content, "parsed_json": None},
            "validation": {"parsed_ok": True, "cue_index_mapping_ok": True},
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _log_call(
            {
                "run_id": _run_id.get(),
                "sample_id": _sample_id.get(),
                "llm_task_type": llm_task_type,
                "batch_indices": batch_indices or [],
                "cache_key": cache_key,
                "cache_hit": False,
                "cache_written": True,
                "cache_path": str(path),
                "cache_scope": normalized_scope,
            }
        )
    return content


def flush_cache_report(*, enabled: Optional[bool] = None) -> Path:
    calls = _call_log.get()
    hits = sum(1 for c in calls if c.get("cache_hit"))
    misses = sum(1 for c in calls if not c.get("cache_hit"))
    writes = sum(1 for c in calls if c.get("cache_written"))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "enabled": enabled if enabled is not None else raw_llm_cache_enabled(),
        "calls": calls,
        "summary": {
            "total_calls": len(calls),
            "hits": hits,
            "misses": misses,
            "writes": writes,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {"runs": []}
    if REPORT_PATH.exists():
        try:
            existing = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    run_entry = {
        "run_id": _run_id.get() or datetime.now(timezone.utc).isoformat(),
        "calls": calls,
        "summary": report["summary"],
    }
    runs = [r for r in existing.get("runs", []) if r.get("run_id") != run_entry["run_id"]]
    runs.append(run_entry)
    existing["runs"] = runs
    existing["generated_at"] = report["generated_at"]
    existing["enabled"] = report["enabled"]
    existing["summary"] = {
        "total_calls": sum(r["summary"]["total_calls"] for r in runs),
        "hits": sum(r["summary"]["hits"] for r in runs),
        "misses": sum(r["summary"]["misses"] for r in runs),
        "writes": sum(r["summary"]["writes"] for r in runs),
    }
    REPORT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return REPORT_PATH
