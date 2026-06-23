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


def translation_polish_enabled() -> bool:
    load_env()
    raw = (os.getenv("OPENAI_TRANSLATION_POLISH") or "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}
