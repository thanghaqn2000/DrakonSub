"""Analyze English transcript for topic, glossary, and translation guidance — one model call."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    PROJECT_ROOT,
    benchmark_deterministic_enabled,
    get_gemini_model,
    get_openai_model,
    get_translation_engine,
    llm_chat_kwargs,
    llm_temperature,
)

_FIXTURE_CONTEXT_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "video_context"

_MAX_TRANSCRIPT_CHARS = 12_000

_SYSTEM_PROMPT = """You analyze English video subtitles to help translate them into Vietnamese for general adult viewers (20+, non-specialists).

Audience & style:
- Neutral, concise, clear, natural Vietnamese — not teen slang, not academic.
- Target viewers are beginners who still need easy, spoken-language subtitles.
- Prefer simple everyday Vietnamese over textbook or jargon-heavy wording.

Terminology policy:
- Keep common useful terms when viewers already know them.
- For hard technical terms, suggest beginner-friendly Vietnamese explanations.
- Example style (general rule, not a fixed replacement list):
  "productive assets" → prefer "tài sản tạo ra giá trị" over stiff calques like "tài sản sản xuất".

Rules:
- Extract information ONLY from the provided transcript.
- Do not invent facts, names, or terms not supported by the transcript.
- key_terms must appear in the transcript OR be essential to understand it.
- suggested_vi must follow the terminology policy above.
- possible_asr_risks: flag phrases that look like speech recognition errors.
- translation_warnings: general cautions (idioms, pronouns, fragmented cues, tone).

Return JSON only with this schema:
{
  "detected_topic": "string",
  "confidence": 0.0,
  "audience_level": "general_beginner",
  "translation_style": "string",
  "tone": "string",
  "short_summary": "string",
  "key_terms": [
    {"source": "EN term/phrase from transcript", "suggested_vi": "...", "plain_explanation": "...", "confidence": 0.0}
  ],
  "named_entities": ["..."],
  "possible_asr_risks": ["..."],
  "translation_warnings": ["..."]
}"""


def _build_transcript(entries: List[dict]) -> str:
    lines = []
    for i, entry in enumerate(entries, start=1):
        text = entry.get("text", "").strip()
        if text:
            lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def _default_context(
    user_topic: str,
    audience_level: str,
    style: str,
) -> Dict[str, Any]:
    topic = user_topic if user_topic and user_topic != "auto" else "general"
    return {
        "detected_topic": topic,
        "confidence": 0.0,
        "audience_level": audience_level,
        "translation_style": style,
        "tone": "neutral",
        "short_summary": "",
        "key_terms": [],
        "named_entities": [],
        "possible_asr_risks": [],
        "translation_warnings": [],
    }


def _parse_context_json(content: str) -> Dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Video context response must be a JSON object")
    return data


def _call_openai_context(user_prompt: str) -> str:
    from openai import OpenAI

    from .openai_chat import create_chat_completion

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY required for video context analysis")

    client = OpenAI(api_key=api_key)
    model = get_openai_model()
    response = create_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=llm_temperature(0.2),
        response_format={"type": "json_object"},
        **llm_chat_kwargs(),
    )
    return response.choices[0].message.content or ""


def _call_gemini_context(user_prompt: str) -> str:
    from .gemini_translate import _call_gemini_json

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY required for video context analysis")

    content, _ = _call_gemini_json(
        api_key,
        get_gemini_model(),
        _SYSTEM_PROMPT,
        user_prompt,
        temperature=llm_temperature(0.2),
    )
    return content


def _context_cache_path(entries: List[dict]) -> Path:
    import hashlib

    blob = _build_transcript(entries).encode("utf-8")
    key = hashlib.sha256(blob).hexdigest()[:16]
    return _FIXTURE_CONTEXT_ROOT / f"{key}.json"


def analyze_video_context(
    entries: List[dict],
    *,
    user_topic: str = "auto",
    audience_level: str = "general_beginner",
    style: str = "simple_vietnamese_subtitle",
    engine: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One model call to extract dynamic video context from English SRT entries.
    Falls back to defaults if the API call fails.
    In benchmark deterministic mode, reuse cached context keyed by transcript hash.
    """
    cache_path = _context_cache_path(entries)
    if benchmark_deterministic_enabled() and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    transcript = _build_transcript(entries)
    if not transcript.strip():
        return _default_context(user_topic, audience_level, style)

    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:_MAX_TRANSCRIPT_CHARS] + "\n... [truncated]"

    style_desc = (
        "simple, natural Vietnamese subtitles for general adults"
        if style == "simple_vietnamese_subtitle"
        else style
    )

    user_prompt = (
        f"user_topic hint: {user_topic}\n"
        f"audience_level: {audience_level}\n"
        f"translation_style: {style_desc}\n\n"
        "Transcript:\n"
        f"{transcript}"
    )

    engine = (engine or get_translation_engine()).strip().lower()
    try:
        if engine == "gemini":
            raw = _call_gemini_context(user_prompt)
        else:
            raw = _call_openai_context(user_prompt)
        data = _parse_context_json(raw)
    except Exception as exc:
        print(f"[Video Context] analysis failed ({exc}), using defaults")
        data = _default_context(user_topic, audience_level, style_desc)

    data.setdefault("detected_topic", user_topic if user_topic != "auto" else "general")
    data.setdefault("confidence", 0.0)
    data.setdefault("audience_level", audience_level)
    data.setdefault("translation_style", style_desc)
    data.setdefault("tone", "neutral")
    data.setdefault("short_summary", "")
    data.setdefault("key_terms", [])
    data.setdefault("named_entities", [])
    data.setdefault("possible_asr_risks", [])
    data.setdefault("translation_warnings", [])
    if benchmark_deterministic_enabled():
        save_video_context(cache_path, data)
    return data


def save_video_context(path: str | Path, context: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
