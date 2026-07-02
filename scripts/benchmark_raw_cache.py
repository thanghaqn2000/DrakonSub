"""Raw translation cache helpers for multi-sample benchmark."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROMPT_VERSION = "raw_v1"
ROOT = Path(__file__).resolve().parents[1]
RAW_CACHE_ROOT = ROOT / "artifacts" / "multi_sample_benchmark" / "raw_cache"
FIXTURE_RAW_CACHE_ROOT = ROOT / "tests" / "fixtures" / "benchmark_raw"
INTEGRITY_REPORT_PATH = ROOT / "artifacts" / "multi_sample_benchmark" / "raw_cache_integrity_report.json"


def _model_name(engine: str) -> str:
    from auto_subtitle.config import get_gemini_model, get_openai_model

    engine = (engine or "openai").strip().lower()
    if engine == "gemini":
        return get_gemini_model()
    return get_openai_model()


def source_hash(source_path: Path) -> str:
    return hashlib.sha256(source_path.read_bytes()).hexdigest()[:16]


def cache_key(
    source_path: Path,
    engine: str,
    raw_translation_mode: Optional[str] = None,
) -> str:
    model = _model_name(engine).replace("/", "_")
    mode = (raw_translation_mode or "grouped").strip().lower()
    return f"{source_hash(source_path)}_{PROMPT_VERSION}_{model}_{mode}"


def legacy_cache_key(source_path: Path, engine: str) -> str:
    """Pre-mode-suffix cache key for backward-compatible lookup."""
    model = _model_name(engine).replace("/", "_")
    return f"{source_hash(source_path)}_{PROMPT_VERSION}_{model}"


def cache_dir(sample_id: str, engine: str, key: str) -> Path:
    return RAW_CACHE_ROOT / sample_id / engine / key


def cache_vi_raw_path(sample_id: str, engine: str, key: str) -> Path:
    return cache_dir(sample_id, engine, key) / "vi_raw.srt"


def cache_meta_path(sample_id: str, engine: str, key: str) -> Path:
    return cache_dir(sample_id, engine, key) / "cache_meta.json"


def _cached_vi_raw_at(root: Path, sample_id: str, engine: str, key: str) -> Optional[Path]:
    path = root / sample_id / engine / key / "vi_raw.srt"
    return path if path.exists() else None


def resolve_cached_vi_raw(
    sample_id: str,
    source_path: Path,
    engine: str,
    raw_translation_mode: Optional[str] = None,
) -> tuple[Optional[Path], str, str]:
    """Return (path, cache_key_used, lookup_status)."""
    mode = (raw_translation_mode or "grouped").strip().lower()
    primary = cache_key(source_path, engine, mode)
    for root in (RAW_CACHE_ROOT, FIXTURE_RAW_CACHE_ROOT):
        found = _cached_vi_raw_at(root, sample_id, engine, primary)
        if found:
            return found, primary, "hit_primary_key"

    legacy = legacy_cache_key(source_path, engine)
    if legacy != primary:
        for root in (RAW_CACHE_ROOT, FIXTURE_RAW_CACHE_ROOT):
            found = _cached_vi_raw_at(root, sample_id, engine, legacy)
            if found:
                return found, legacy, "hit_legacy_key"

    return None, primary, "miss"


def load_cached_vi_raw(
    sample_id: str,
    source_path: Path,
    engine: str,
    raw_translation_mode: Optional[str] = None,
) -> Optional[Path]:
    path, _, status = resolve_cached_vi_raw(
        sample_id, source_path, engine, raw_translation_mode
    )
    return path if status.startswith("hit") else None


def save_cached_vi_raw(
    sample_id: str,
    source_path: Path,
    engine: str,
    vi_raw_path: Path,
    *,
    raw_translation_mode: Optional[str] = None,
    deterministic: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mode = (raw_translation_mode or "grouped").strip().lower()
    key = cache_key(source_path, engine, mode)
    dest_dir = cache_dir(sample_id, engine, key)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "vi_raw.srt"
    shutil.copy2(vi_raw_path, dest)
    meta = {
        "sample": sample_id,
        "engine": engine,
        "cache_key": key,
        "source_hash": source_hash(source_path),
        "model": _model_name(engine),
        "prompt_version": PROMPT_VERSION,
        "raw_translation_mode": mode,
        "deterministic": deterministic,
        "written_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    cache_meta_path(sample_id, engine, key).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def diagnose_cache(
    sample_id: str,
    source_path: Path,
    engine: str,
    raw_translation_mode: Optional[str] = None,
    *,
    manifest_cache_key: Optional[str] = None,
) -> Dict[str, Any]:
    mode = (raw_translation_mode or "grouped").strip().lower()
    expected = cache_key(source_path, engine, mode)
    path, key_used, lookup = resolve_cached_vi_raw(
        sample_id, source_path, engine, mode
    )
    meta_path = cache_meta_path(sample_id, engine, expected)
    legacy = legacy_cache_key(source_path, engine)
    legacy_path = cache_vi_raw_path(sample_id, engine, legacy)

    miss_reason = ""
    cache_hit = path is not None
    if not cache_hit:
        if not source_path.exists():
            miss_reason = "source_hash_mismatch"
        elif meta_path.exists() and not cache_vi_raw_path(sample_id, engine, expected).exists():
            miss_reason = "raw_file_missing"
        elif legacy_path.exists() and expected != legacy:
            miss_reason = "mode_mismatch"
        elif manifest_cache_key and manifest_cache_key != expected:
            miss_reason = "cache_key_mismatch"
        else:
            miss_reason = "raw_file_missing"

    return {
        "sample_id": sample_id,
        "expected_cache_key": expected,
        "manifest_cache_key": manifest_cache_key or "",
        "cache_hit": cache_hit,
        "miss_reason": "" if cache_hit else miss_reason,
        "raw_cache_path": str(path) if path else str(cache_vi_raw_path(sample_id, engine, expected)),
        "raw_cache_exists": bool(path and path.exists()),
        "manifest_exists": meta_path.exists(),
        "source_hash": source_hash(source_path) if source_path.exists() else "",
        "engine": engine,
        "model": _model_name(engine),
        "raw_translation_mode": mode,
        "deterministic": None,
        "prompt_version": PROMPT_VERSION,
        "lookup_status": lookup,
        "fix_applied": "",
    }


def write_integrity_report(rows: List[Dict[str, Any]]) -> Path:
    INTEGRITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_hit_count": sum(1 for r in rows if r.get("cache_hit")),
        "cache_miss_count": sum(1 for r in rows if not r.get("cache_hit")),
        "samples": rows,
    }
    INTEGRITY_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return INTEGRITY_REPORT_PATH


def mode_type_for(
    *,
    reuse_raw: bool,
    used_raw_cache: bool,
    fresh_translate: bool,
) -> str:
    if reuse_raw or used_raw_cache:
        return "cached_raw_pipeline_regression"
    return "fresh_translate_end_to_end"


def score_interpretation(mode_type: str) -> str:
    if mode_type == "cached_raw_pipeline_regression":
        return "pipeline_quality_on_fixed_raw_translation"
    return "end_to_end_quality_includes_translation_variance"
