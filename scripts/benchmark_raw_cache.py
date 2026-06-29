"""Raw translation cache helpers for multi-sample benchmark."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

PROMPT_VERSION = "raw_v1"
RAW_CACHE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "multi_sample_benchmark" / "raw_cache"


def _model_name(engine: str) -> str:
    from auto_subtitle.config import get_gemini_model, get_openai_model

    engine = (engine or "openai").strip().lower()
    if engine == "gemini":
        return get_gemini_model()
    return get_openai_model()


def source_hash(source_path: Path) -> str:
    return hashlib.sha256(source_path.read_bytes()).hexdigest()[:16]


def cache_key(source_path: Path, engine: str) -> str:
    model = _model_name(engine).replace("/", "_")
    return f"{source_hash(source_path)}_{PROMPT_VERSION}_{model}"


def cache_dir(sample_id: str, engine: str, key: str) -> Path:
    return RAW_CACHE_ROOT / sample_id / engine / key


def cache_vi_raw_path(sample_id: str, engine: str, key: str) -> Path:
    return cache_dir(sample_id, engine, key) / "vi_raw.srt"


def cache_meta_path(sample_id: str, engine: str, key: str) -> Path:
    return cache_dir(sample_id, engine, key) / "cache_meta.json"


def load_cached_vi_raw(
    sample_id: str,
    source_path: Path,
    engine: str,
) -> Optional[Path]:
    key = cache_key(source_path, engine)
    path = cache_vi_raw_path(sample_id, engine, key)
    return path if path.exists() else None


def save_cached_vi_raw(
    sample_id: str,
    source_path: Path,
    engine: str,
    vi_raw_path: Path,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    key = cache_key(source_path, engine)
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
        **(extra or {}),
    }
    cache_meta_path(sample_id, engine, key).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


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
