"""Project paths and environment configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Verified with v1/chat/completions + response_format json_object (best first).
OPENAI_CHAT_MODELS = (
    "gpt-5.5-2026-04-23",
    "gpt-5.5",
    "o3",
    "gpt-4.1-2025-04-14",
    "gpt-4.1",
    "gpt-4o-2024-11-20",
    "gpt-4o",
    "o4-mini",
)

DEFAULT_OPENAI_MODEL = "gpt-5.5-2026-04-23"
DEFAULT_TRANSLATION_BATCH_SIZE = 30
DEFAULT_TRANSLATION_ENGINE = "openai"
SUPPORTED_TRANSLATION_ENGINES = ("openai", "gemini", "google")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

DEFAULT_VI_EDITOR_BATCH_SIZE = 30
DEFAULT_VI_EDITOR_CONTEXT_WINDOW = 5
DEFAULT_VI_EDITOR_TEMPERATURE = 0.3
VI_EDITOR_ENABLED = True
VI_EDITOR_PROVIDER = "auto"
VI_EDITOR_SAVE_DEBUG = True
EN_DOMAIN_CORRECTION_ENABLED = True
EN_DOMAIN_CORRECTION_MODE = "rules"
EN_DOMAIN_CORRECTION_SAVE_DEBUG = True

# Vietnamese subtitle compression (after editor, before readability).
VI_COMPRESSION_ENABLED = True
VI_COMPRESSION_FAST_MAX_DURATION = 0.8
VI_COMPRESSION_FAST_MAX_WORDS = 6
VI_COMPRESSION_MEDIUM_MAX_DURATION = 1.5
VI_COMPRESSION_MEDIUM_MAX_WORDS = 10
VI_COMPRESSION_TRIGGER_CPS = 22.0
VI_COMPRESSION_MAX_CPS = 24.0
VI_COMPRESSION_MIN_SHORTEN_RATIO = 0.12

# Multi-cue Vietnamese flow (after compression, before timing).
VI_FLOW_ENABLED = True
VI_FLOW_MIN_GROUP_SIZE = 2
VI_FLOW_MAX_GROUP_SIZE = 4
VI_FLOW_MAX_CHAR_INCREASE_RATIO = 0.10
VI_FLOW_MAX_CPS = 28.0
VI_FLOW_TINY_FRAGMENT_CHARS = 8
VI_FLOW_SAVE_DEBUG = True

# Final timing normalization (remove cue overlaps after timing optimizer).
TIMING_NORMALIZE_MIN_GAP = 0.05
TIMING_NORMALIZE_MIN_DURATION = 0.45

# General translation intelligence (context, meaning units, QA, repair).
TRANSLATION_INTELLIGENCE_ENABLED = True
TRANSLATION_INTELLIGENCE_SAVE_DEBUG = True

# Sample-specific OpenAI editor few-shots from prior experiments — off in production.
VI_EDITOR_OPENAI_FEW_SHOT_ENABLED = False

# Maximum consecutive cues per phrase group sent to GPT.
# Smaller = more groups (more API calls), larger = more context per call.
DEFAULT_MAX_CUES_PER_GROUP = 6
RAW_TRANSLATION_MODES = (
    "grouped",
    "cue_keyed",
    "hybrid_guarded",
    "span_guarded",
    "span_guarded_conservative",
    "span_guarded_tiered",
    "longform_chunked",
)
DEFAULT_RAW_TRANSLATION_MODE = "grouped"
POST_RAW_OVERLAP_GUARD_ENABLED = True

# Models that fail on chat/completions — map to the closest supported tier.
OPENAI_MODEL_ALIASES = {
    "gpt-5.5-pro": DEFAULT_OPENAI_MODEL,
    "chatgpt-4o-latest": "gpt-4o",
}


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def get_openai_model() -> str:
    load_env()
    raw = (os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL).strip()
    return OPENAI_MODEL_ALIASES.get(raw, raw)


def get_translation_batch_size() -> int:
    load_env()
    value = int(os.getenv("TRANSLATION_BATCH_SIZE", str(DEFAULT_TRANSLATION_BATCH_SIZE)))
    return max(1, min(value, 50))


def get_translation_engine() -> str:
    load_env()
    raw = (os.getenv("TRANSLATION_ENGINE") or DEFAULT_TRANSLATION_ENGINE).strip().lower()
    if raw not in SUPPORTED_TRANSLATION_ENGINES:
        valid = ", ".join(SUPPORTED_TRANSLATION_ENGINES)
        raise ValueError(
            f"Unsupported TRANSLATION_ENGINE '{raw}'. Choose one of: {valid}"
        )
    return raw


def get_gemini_model() -> str:
    load_env()
    return (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()


def get_phrase_group_max_cues() -> int:
    load_env()
    value = int(os.getenv("PHRASE_GROUP_MAX_CUES", str(DEFAULT_MAX_CUES_PER_GROUP)))
    return max(1, min(value, 20))


def get_raw_translation_mode() -> str:
    load_env()
    raw = (os.getenv("RAW_TRANSLATION_MODE") or DEFAULT_RAW_TRANSLATION_MODE).strip().lower()
    if raw not in RAW_TRANSLATION_MODES:
        valid = ", ".join(RAW_TRANSLATION_MODES)
        raise ValueError(
            f"Unsupported RAW_TRANSLATION_MODE '{raw}'. Choose one of: {valid}"
        )
    return raw


def translation_polish_enabled() -> bool:
    """OpenAI inline polish is replaced by the shared VI editor pass."""
    return False


def vi_editor_enabled() -> bool:
    return VI_EDITOR_ENABLED


def get_vi_editor_provider() -> str:
    return VI_EDITOR_PROVIDER


def resolve_vi_editor_provider(translation_engine: str) -> str:
    if translation_engine == "google":
        return "google"
    if VI_EDITOR_PROVIDER == "auto":
        return translation_engine
    return VI_EDITOR_PROVIDER


def get_vi_editor_model() -> str:
    return "auto"


def resolve_vi_editor_model(provider: str) -> str:
    if provider == "openai":
        return get_openai_model()
    return get_gemini_model()


def get_vi_editor_batch_size() -> int:
    return DEFAULT_VI_EDITOR_BATCH_SIZE


def get_vi_editor_context_window() -> int:
    return DEFAULT_VI_EDITOR_CONTEXT_WINDOW


def benchmark_deterministic_enabled() -> bool:
    load_env()
    raw = (os.getenv("BENCHMARK_DETERMINISTIC") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def llm_temperature(default: float) -> float:
    if benchmark_deterministic_enabled():
        return 0.0
    return default


def llm_chat_kwargs() -> dict:
    """Extra OpenAI chat params for benchmark deterministic mode."""
    if not benchmark_deterministic_enabled():
        return {}
    kwargs: dict = {"top_p": 1}
    seed_raw = (os.getenv("BENCHMARK_LLM_SEED") or "42").strip()
    try:
        kwargs["seed"] = int(seed_raw)
    except ValueError:
        pass
    return kwargs


def get_vi_editor_temperature() -> float:
    return llm_temperature(DEFAULT_VI_EDITOR_TEMPERATURE)


def vi_editor_save_debug() -> bool:
    return VI_EDITOR_SAVE_DEBUG


def en_domain_correction_enabled() -> bool:
    return EN_DOMAIN_CORRECTION_ENABLED


def en_domain_correction_mode() -> str:
    return EN_DOMAIN_CORRECTION_MODE


def en_domain_correction_save_debug() -> bool:
    return EN_DOMAIN_CORRECTION_SAVE_DEBUG


def vi_compression_enabled() -> bool:
    return VI_COMPRESSION_ENABLED


def vi_flow_enabled() -> bool:
    return VI_FLOW_ENABLED


def translation_intelligence_enabled() -> bool:
    return TRANSLATION_INTELLIGENCE_ENABLED


def post_raw_overlap_guard_enabled() -> bool:
    load_env()
    raw = (os.getenv("POST_RAW_OVERLAP_GUARD_ENABLED") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return POST_RAW_OVERLAP_GUARD_ENABLED


def translation_intelligence_save_debug() -> bool:
    return TRANSLATION_INTELLIGENCE_SAVE_DEBUG


def vi_editor_openai_few_shot_enabled() -> bool:
    return VI_EDITOR_OPENAI_FEW_SHOT_ENABLED
