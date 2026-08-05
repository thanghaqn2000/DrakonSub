from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from auto_subtitle.config import load_env

DEFAULT_SAYDI_SAMPLE = "liam-warm-thoughtful-and-determined-7XOKiK112QRZRSLbCfMc"
DEFAULT_SAYDI_SPEED = 1.0
SAYDI_SPEED_MIN = 0.5
SAYDI_SPEED_MAX = 2.0
SAYDI_SAMPLE_MAX_LEN = 200
SAYDI_VOICE_PRESETS: tuple[tuple[str, str], ...] = (
    ("Liam", "liam-warm-thoughtful-and-determined-7XOKiK112QRZRSLbCfMc"),
    ("Quyến rũ", "quyen-ru-f966mdF5njWREvreUG07"),
    ("Truyền cảm hứng", "ng-n-nguy-n-inspirational-narrator-DvG3I1kDzdBY3u4EzYh6"),
    ("Moncellence", "moncellence-clear-and-masculine-Jez3JdhBInQTvlAvDOWR"),
    ("Nữ hot TikTok", "cuppy-vi"),
    ("Minh Quân rõ ràng, nhanh", "hn-minh-qu-n-6854cb56138047a75b2b2163"),
    ("Tony Hoang rõ ràng, nhanh", "tony-hoang-K7ewtjKRNtwwt3lKQ6M0"),
)
SAYDI_SAMPLE_INVALID_MESSAGE = (
    "Giọng đọc Saydi không hợp lệ. Vui lòng kiểm tra mã giọng/sample."
)
SAYDI_SPEED_INVALID_MESSAGE = (
    f"Tốc độ đọc Saydi phải nằm trong khoảng từ {SAYDI_SPEED_MIN} đến {SAYDI_SPEED_MAX}."
)


class SaydiConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SaydiConfig:
    api_url: str
    token: str
    sample: str
    speed: float
    output_format: str
    timeout_seconds: int
    lang: str
    model: str | None = None


def list_saydi_voice_presets() -> list[dict[str, str]]:
    return [{"label": label, "sample": sample} for label, sample in SAYDI_VOICE_PRESETS]


def validate_saydi_sample(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > SAYDI_SAMPLE_MAX_LEN:
        raise SaydiConfigError(SAYDI_SAMPLE_INVALID_MESSAGE)
    if any(ord(ch) < 32 for ch in cleaned):
        raise SaydiConfigError(SAYDI_SAMPLE_INVALID_MESSAGE)
    return cleaned


def resolve_saydi_sample(sample_override: str | None = None) -> str:
    override = validate_saydi_sample(sample_override)
    if override is not None:
        return override
    return (os.getenv("SAYDI_TTS_SAMPLE") or DEFAULT_SAYDI_SAMPLE).strip()


def validate_saydi_speed(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        speed = float(value)
    except (TypeError, ValueError) as exc:
        raise SaydiConfigError(SAYDI_SPEED_INVALID_MESSAGE) from exc
    if not SAYDI_SPEED_MIN <= speed <= SAYDI_SPEED_MAX:
        raise SaydiConfigError(SAYDI_SPEED_INVALID_MESSAGE)
    return speed


def resolve_saydi_speed(speed_override: float | str | None = None) -> float:
    override = validate_saydi_speed(speed_override)
    if override is not None:
        return override
    env_raw = (os.getenv("SAYDI_TTS_SPEED") or "").strip()
    if env_raw:
        env_speed = validate_saydi_speed(env_raw)
        if env_speed is not None:
            return env_speed
    return DEFAULT_SAYDI_SPEED


def load_saydi_config(
    *,
    sample_override: str | None = None,
    speed_override: float | str | None = None,
) -> SaydiConfig:
    load_env()
    output_format = (os.getenv("SAYDI_TTS_OUTPUT_FORMAT") or "wav").strip().lower()
    if output_format not in {"wav", "mp3", "flac", "ogg"}:
        output_format = "wav"
    model = (os.getenv("SAYDI_TTS_MODEL") or "").strip() or None
    return SaydiConfig(
        api_url=(os.getenv("SAYDI_TTS_API_URL") or "https://api.voice.saydi.ai/tts").strip(),
        token=(os.getenv("SAYDI_TTS_API_TOKEN") or "").strip(),
        sample=resolve_saydi_sample(sample_override),
        speed=resolve_saydi_speed(speed_override),
        output_format=output_format,
        timeout_seconds=int(os.getenv("SAYDI_TTS_TIMEOUT_SECONDS") or "120"),
        lang=(os.getenv("SAYDI_TTS_LANG") or "vi").strip(),
        model=model,
    )


def build_saydi_request_payload(text: str, config: SaydiConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "sample": config.sample,
        "output_format": config.output_format,
        "lang": config.lang,
        "speed": config.speed,
    }
    if config.model:
        payload["model"] = config.model
    return payload


def synthesize_to_file(
    text: str,
    output_path: Path,
    *,
    config: Optional[SaydiConfig] = None,
) -> dict[str, Any]:
    cfg = config or load_saydi_config()
    if not cfg.token:
        raise RuntimeError("SAYDI_TTS_API_TOKEN is not configured")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_saydi_request_payload(text, cfg)
    request = urllib.request.Request(
        cfg.api_url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.token}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout_seconds) as response:
            audio = response.read()
            content_type = str(response.headers.get("Content-Type") or "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Saydi TTS HTTP {exc.code}: {detail[:300]}") from exc
    except Exception as exc:
        raise RuntimeError(f"Saydi TTS request failed: {exc}") from exc

    if not audio:
        raise RuntimeError("Saydi TTS returned empty audio body")

    output_path.write_bytes(audio)
    return {
        "output_path": str(output_path),
        "size_bytes": len(audio),
        "content_type": content_type,
    }
