"""Fix English loanwords mis-transcribed as Vietnamese phonetics in VI subtitles."""

import json
import os
import re
from typing import Callable, List, Optional

from .config import get_openai_model
from .openai_chat import create_chat_completion

# Wrong form → correct form (longer phrases first when applied)
LOANWORD_REPLACEMENTS = [
    ("việt súp", "vietsub"),
    ("việt sub", "vietsub"),
    ("việt xúp", "vietsub"),
    ("việt sup", "vietsub"),
    ("vi đeo", "video"),
    ("vi deo", "video"),
    ("phây búc", "Facebook"),
    ("phây búk", "Facebook"),
    ("tích tốc", "TikTok"),
    ("tích tok", "TikTok"),
    ("du túp", "YouTube"),
    ("du tube", "YouTube"),
    ("gúc gol", "Google"),
    ("gúc gồ", "Google"),
    ("in sta gram", "Instagram"),
    ("in stagram", "Instagram"),
    ("ây ai", "AI"),
    ("a i", "AI"),
    ("on lai", "online"),
    ("onlai", "online"),
    ("up đết", "update"),
    ("up date", "update"),
    ("phít bách", "feedback"),
    ("com men", "comment"),
    ("pho lâu", "follow"),
    ("mi ting", "meeting"),
    ("ma kết ting", "marketing"),
    ("phít", "fix"),
    ("fích", "fix"),
    ("tune", "tool"),
    ("tún", "tool"),
    ("tul", "tool"),
    ("pích", "pitch"),
    ("slai", "slide"),
    ("đê mô", "demo"),
    ("pốt", "post"),
    ("laiv", "live"),
    ("reels", "reels"),
    ("reel", "reel"),
    ("startup", "startup"),
    ("website", "website"),
    ("web sai", "website"),
    ("email", "email"),
    ("i meo", "email"),
    ("link", "link"),
    ("ling", "link"),
    ("clic", "click"),
    ("subscribe", "subscribe"),
    ("channel", "channel"),
    ("views", "views"),
    ("viu", "views"),
    ("trend", "trend"),
    ("brand", "brand"),
    ("brần", "brand"),
    ("feature", "feature"),
    ("phía chơ", "feature"),
    ("bug", "bug"),
    ("bắc", "bug"),
    ("test", "test"),
    ("tét", "test"),
    ("stream", "stream"),
]

_SORTED_REPLACEMENTS = sorted(
    LOANWORD_REPLACEMENTS, key=lambda pair: len(pair[0]), reverse=True
)

SYSTEM_PROMPT = """You fix Vietnamese video subtitles where English loanwords were mis-transcribed as Vietnamese phonetics.

Rules (strict):
- Input is Vietnamese speech subtitles. Speakers often mix in English words.
- Restore mis-transcribed English loanwords to standard English spelling (e.g. việt súp → vietsub, phít → fix, tune → tool, vi đeo → video).
- Keep all Vietnamese text unchanged — do not paraphrase, shorten, or improve wording.
- One input segment → exactly one output segment, same order.
- Do not merge or split segments.
- Keep names, numbers, and proper nouns accurate.
- Return JSON only, no markdown."""


def apply_dictionary_fix(text: str) -> str:
    result = text
    for wrong, correct in _SORTED_REPLACEMENTS:
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        result = pattern.sub(correct, result)
    return result


def fix_entries_dictionary(entries: List[dict]) -> List[dict]:
    return [
        {**entry, "text": apply_dictionary_fix(entry["text"])}
        for entry in entries
    ]


def _build_user_prompt(segments: List[str]) -> str:
    lines = [f"[{i + 1}] {text}" for i, text in enumerate(segments)]
    n = len(segments)
    return (
        f"Fix English loanwords in these {n} Vietnamese subtitle segments.\n\n"
        + "\n".join(lines)
        + f'\n\nRespond with JSON: {{"segments": ["...", ...]}} '
        f"containing exactly {n} strings in the same order."
    )


def _parse_segments(content: str, expected_count: int) -> List[str]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)
    segments = data.get("segments", data if isinstance(data, list) else None)
    if not isinstance(segments, list):
        raise ValueError("OpenAI response missing 'segments' array")

    if len(segments) != expected_count:
        raise ValueError(f"Expected {expected_count} segments, got {len(segments)}")

    return [str(s).strip() for s in segments]


def _call_openai_fix(client, model: str, segments: List[str]) -> List[str]:
    response = create_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(segments)},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return _parse_segments(response.choices[0].message.content or "", len(segments))


def _fix_batch(client, model: str, segments: List[str]) -> List[str]:
    try:
        return _call_openai_fix(client, model, segments)
    except ValueError:
        if len(segments) == 1:
            raise
        mid = len(segments) // 2
        return _fix_batch(client, model, segments[:mid]) + _fix_batch(
            client, model, segments[mid:]
        )


def fix_entries_openai(
    entries: List[dict],
    model: Optional[str] = None,
    batch_size: int = 15,
) -> List[dict]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  Skipping OpenAI loanword fix: OPENAI_API_KEY not set")
        return entries

    client = OpenAI(api_key=api_key)
    model = model or get_openai_model()
    fixed: List[Optional[dict]] = [None] * len(entries)
    pending_indices: List[int] = []
    pending_texts: List[str] = []

    def flush_batch() -> None:
        nonlocal pending_indices, pending_texts
        if not pending_texts:
            return

        print(
            f"  Fixing loanwords segments {pending_indices[0] + 1}-{pending_indices[-1] + 1}..."
        )
        batch_fixed = _fix_batch(client, model, pending_texts)
        for idx, text in zip(pending_indices, batch_fixed):
            fixed[idx] = {**entries[idx], "text": text}
        pending_indices = []
        pending_texts = []

    for i, entry in enumerate(entries):
        text = entry["text"].strip()
        if not text:
            fixed[i] = {**entry, "text": text}
            continue
        pending_indices.append(i)
        pending_texts.append(text)
        if len(pending_texts) >= batch_size:
            flush_batch()

    if pending_texts:
        flush_batch()

    return fixed


ProgressCallback = Callable[[str, int], None]


def fix_vi_srt_entries(
    entries: List[dict],
    use_openai: bool = True,
    on_progress: Optional[ProgressCallback] = None,
) -> List[dict]:
    if on_progress:
        on_progress("Fixing English loanwords (dictionary)...", 55)

    entries = fix_entries_dictionary(entries)

    if use_openai and os.environ.get("OPENAI_API_KEY"):
        if on_progress:
            on_progress("Checking English terms with AI...", 60)
        entries = fix_entries_openai(entries)

    if on_progress:
        on_progress("Loanword fix complete", 65)

    return entries
