from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Tuple, TypeVar

T = TypeVar("T")

_GEMINI_KEY_ENV_NAMES = tuple(f"GEMINI_API_KEY_{index}" for index in range(1, 5))
_LEGACY_GEMINI_KEY_ENV_NAMES = ("GEMINI_API_KEY",)
_QUOTA_MARKERS = (
    "quota",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "resource exhausted",
    "too many requests",
)


class GeminiQuotaError(RuntimeError):
    """Gemini rate limit or quota exhausted for the current API key."""


def load_gemini_api_keys() -> List[str]:
    """Load Gemini keys in order: GEMINI_API_KEY_1..4, then legacy GEMINI_API_KEY."""
    keys: List[str] = []
    seen: set[str] = set()

    def _append(raw: str) -> None:
        value = (raw or "").strip()
        if value and value not in seen:
            keys.append(value)
            seen.add(value)

    for env_name in _GEMINI_KEY_ENV_NAMES:
        _append(os.getenv(env_name, ""))

    for env_name in _LEGACY_GEMINI_KEY_ENV_NAMES:
        _append(os.getenv(env_name, ""))

    return keys


def gemini_configured() -> bool:
    return bool(load_gemini_api_keys())


def is_gemini_key_rotatable_error(exc: BaseException) -> bool:
    if isinstance(exc, GeminiQuotaError):
        return True

    message = str(exc).lower()
    if any(marker in message for marker in _QUOTA_MARKERS):
        return True

    if isinstance(exc, OSError) and getattr(exc, "code", None) == 429:
        return True

    http_code = getattr(exc, "code", None)
    if http_code == 429:
        return True

    match = re.search(r"http\s+(\d+)", message)
    if match and match.group(1) == "429":
        return True

    return False


def call_gemini_with_key_rotation(
    operation: Callable[[str], T],
    *,
    api_keys: Optional[List[str]] = None,
    action: str = "Gemini request",
) -> T:
    keys = api_keys or load_gemini_api_keys()
    if not keys:
        raise ValueError(
            "No Gemini API keys configured. Set GEMINI_API_KEY_1..4 in .env "
            "when TRANSLATION_ENGINE=gemini."
        )

    last_error: Optional[BaseException] = None
    for index, api_key in enumerate(keys):
        try:
            return operation(api_key)
        except Exception as exc:
            if is_gemini_key_rotatable_error(exc) and index < len(keys) - 1:
                print(
                    f"[Gemini] {action}: key #{index + 1} hit quota/rate limit, "
                    f"rotating to key #{index + 2}..."
                )
                last_error = exc
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{action} failed without a usable Gemini API key.")


def resolve_gemini_model_for_keys(
    api_keys: List[str],
    requested_model: str,
    resolver: Callable[[str, str], str],
) -> Tuple[str, str]:
    if not api_keys:
        raise ValueError("No Gemini API keys configured.")

    last_error: Optional[BaseException] = None
    for index, api_key in enumerate(api_keys):
        try:
            return resolver(api_key, requested_model), api_key
        except Exception as exc:
            if is_gemini_key_rotatable_error(exc) and index < len(api_keys) - 1:
                print(
                    f"[Gemini] resolve model: key #{index + 1} hit quota/rate limit, "
                    f"rotating to key #{index + 2}..."
                )
                last_error = exc
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not resolve Gemini model with configured API keys.")
