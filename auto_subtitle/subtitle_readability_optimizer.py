"""
Vietnamese Subtitle Readability Optimizer
==========================================
Shortens Vietnamese subtitle text so the viewer can finish reading each cue
within its original display window, without touching timing.

Two-stage pipeline:
  1. Rule-based replacements  — instant, zero cost, handles common verbose patterns.
  2. OpenAI rewrite (opt-in)  — only for cues that are still above *hard_cps*
                                 after the rule pass.

Config (env variables, all optional):
  ENABLE_VI_READABILITY_OPTIMIZER   true / false          (default: true)
  VI_READABILITY_USE_OPENAI         true / false          (default: false)
  VI_READABILITY_TARGET_CPS         float                 (default: 17)
  VI_READABILITY_MAX_CPS            float                 (default: 22)
  VI_READABILITY_HARD_CPS           float                 (default: 26)
"""

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_TARGET_CPS: float = 17.0
_DEFAULT_MAX_CPS: float = 22.0
_DEFAULT_HARD_CPS: float = 26.0


@dataclass
class ReadabilityConfig:
    """
    Tunable parameters for the readability optimizer.

    The two CPS thresholds define three zones:
      cps ≤ max_cps   → leave text unchanged
      cps ≤ hard_cps  → apply rule-based shortening
      cps > hard_cps  → rule-based + optional OpenAI rewrite
    """

    enabled: bool = True
    target_cps: float = _DEFAULT_TARGET_CPS
    max_cps: float = _DEFAULT_MAX_CPS
    hard_cps: float = _DEFAULT_HARD_CPS
    use_openai: bool = False


def load_readability_config() -> ReadabilityConfig:
    """
    Build a ReadabilityConfig from environment variables.

    Falls back to conservative defaults when variables are absent.
    """
    from .config import load_env
    load_env()

    def _bool(key: str, default: bool) -> bool:
        raw = os.getenv(key, "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    def _float(key: str, default: float) -> float:
        raw = os.getenv(key, "").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    return ReadabilityConfig(
        enabled=_bool("ENABLE_VI_READABILITY_OPTIMIZER", True),
        use_openai=_bool("VI_READABILITY_USE_OPENAI", False),
        target_cps=_float("VI_READABILITY_TARGET_CPS", _DEFAULT_TARGET_CPS),
        max_cps=_float("VI_READABILITY_MAX_CPS", _DEFAULT_MAX_CPS),
        hard_cps=_float("VI_READABILITY_HARD_CPS", _DEFAULT_HARD_CPS),
    )


# ---------------------------------------------------------------------------
# Rule-based replacement table
# ---------------------------------------------------------------------------
# Order matters: longer / more-specific patterns must come first.
# Each tuple is (verbose_phrase, concise_phrase).  Case-sensitive.

_RULE_REPLACEMENTS: List[Tuple[str, str]] = [
    # --- User-provided examples ---
    ("trở nên đắt đỏ hơn", "đắt hơn"),
    ("nhà đầu tư thường rút lui khỏi", "nhà đầu tư rút khỏi"),
    ("các tài sản có mức độ rủi ro cao", "tài sản rủi ro"),
    ("điều này thực sự rất quan trọng", "điều này rất quan trọng"),
    ("bạn có thể thấy rằng", "có thể thấy"),
    ("chúng ta có thể thấy rằng", "có thể thấy"),
    ("có thể thấy rằng", "có thể thấy"),
    # --- Filler / verbose qualifiers ---
    ("thực sự rất", "rất"),
    ("thực sự là", "thực sự"),
    ("thực ra là", "thực ra"),
    ("đương nhiên là", "đương nhiên"),
    ("rõ ràng là", "rõ ràng"),
    ("một cách rõ ràng", "rõ ràng"),
    ("đặc biệt là", "đặc biệt"),
    ("ví dụ như", "ví dụ"),
    ("chẳng hạn như", "chẳng hạn"),
    ("đó là lý do tại sao", "đó là lý do"),
    ("đây là lý do tại sao", "đây là lý do"),
    ("điều này có nghĩa là", "tức là"),
    ("điều đó có nghĩa là", "tức là"),
    ("trong khi đó", "trong khi"),
    ("trở nên đắt hơn", "đắt hơn"),
    ("rút lui khỏi", "rút khỏi"),
    ("tham gia vào", "tham gia"),
    ("đầu tư vào", "đầu tư"),
    ("giảm xuống", "giảm"),
    ("tăng lên", "tăng"),
    ("đi lên", "tăng"),
    ("đi xuống", "giảm"),
    # --- Over-formal structures ---
    ("được gọi là", "là"),
    ("được biết đến là", "là"),
    ("có tên là", "là"),
    ("bao gồm cả", "bao gồm"),
    ("liên quan đến", "liên quan"),
    ("dẫn đến việc", "dẫn đến"),
    ("dẫn đến kết quả là", "dẫn đến"),
    ("có thể dẫn đến", "có thể gây"),
    ("cần phải", "cần"),
    ("phải cần", "cần"),
    ("tiếp tục tăng", "tăng tiếp"),
    ("tiếp tục giảm", "giảm tiếp"),
]

# Compile regex patterns once at import time for performance.
_COMPILED_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(re.escape(verbose), re.IGNORECASE), concise)
    for verbose, concise in _RULE_REPLACEMENTS
]

# Clean up multiple spaces left after replacements.
_MULTI_SPACE_RE = re.compile(r" {2,}")


def _apply_rules(text: str) -> str:
    """
    Apply the rule-based replacement table to *text*.

    Applies all patterns in order, collapses excess whitespace, and strips.
    """
    for pattern, replacement in _COMPILED_RULES:
        text = pattern.sub(replacement, text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _char_count(text: str) -> int:
    """Count readable characters, collapsing internal whitespace."""
    return len(re.sub(r"\s+", " ", text).strip())


def _parse_ts(ts: str) -> float:
    """Convert SRT timestamp ``HH:MM:SS,mmm`` to seconds."""
    ts = ts.strip()
    time_part, millis_str = ts.split(",")
    h, m, s = time_part.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(millis_str) / 1000.0


def _cps(text: str, duration: float) -> float:
    """Characters per second, safe against zero-division."""
    if duration <= 0:
        return 0.0
    return _char_count(text) / duration


def _is_debug() -> bool:
    """Return True when DRAKONSUB_DEBUG is truthy."""
    return os.getenv("DRAKONSUB_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# OpenAI rewrite pass
# ---------------------------------------------------------------------------

_OPENAI_SYSTEM_PROMPT = (
    "You are a Vietnamese subtitle editor for short-form social media videos. "
    "Your only job is to make each subtitle shorter so Facebook/Reels viewers can "
    "read it quickly within a tight display window.\n\n"
    "Rules:\n"
    "- Preserve core meaning exactly.\n"
    "- Do NOT add any new information.\n"
    "- Keep numbers, names, technical terms (Bitcoin, Fed, ETF, lãi suất, lạm phát, crypto, …) unchanged.\n"
    "- Remove filler words and overly formal phrasing.\n"
    "- Prefer everyday spoken Vietnamese over literary Vietnamese.\n"
    "- The result must still sound natural, not robotic.\n"
    "- Return exactly the same number of subtitles in the same order."
)


def _openai_rewrite_batch(
    texts: List[str],
    max_chars_per_line: List[int],
) -> List[str]:
    """
    Call OpenAI to shorten a batch of Vietnamese subtitle lines.

    *max_chars_per_line* tells the model the target character budget for each
    line so it knows how aggressively to shorten.  Returns a list of the same
    length as *texts*.
    """
    import os
    from openai import OpenAI

    from .config import get_openai_model, llm_chat_kwargs, llm_temperature
    from .openai_chat import create_chat_completion

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return texts  # skip silently if no key

    client = OpenAI(api_key=api_key)
    model = get_openai_model()

    n = len(texts)
    lines = [
        f"[{i + 1}] (max {max_chars_per_line[i]} chars) {text}"
        for i, text in enumerate(texts)
    ]
    user_prompt = (
        f"Shorten these {n} Vietnamese subtitles to fit within the character budget shown. "
        "Keep meaning, preserve all names and numbers.\n\n"
        + "\n".join(lines)
        + f'\n\nRespond with JSON: {{"subtitles": ["...", ...]}} '
        f"containing exactly {n} strings in the same order."
    )

    try:
        response = create_chat_completion(
            client,
            model,
            messages=[
                {"role": "system", "content": _OPENAI_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=llm_temperature(0.3),
            response_format={"type": "json_object"},
            **llm_chat_kwargs(),
        )
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        data = json.loads(content)
        values = data.get("subtitles", [])
        if isinstance(values, list) and len(values) == n:
            return [str(v).strip() for v in values]
    except Exception as exc:
        print(f"  [Readability] OpenAI rewrite failed ({exc}), keeping rule-based result.")

    return texts  # fall back to rule-based output on any error


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------

def _optimize_entries(
    entries: List[dict],
    cfg: ReadabilityConfig,
) -> List[dict]:
    """
    Apply readability optimisation to a list of SRT entry dicts.

    Returns a new list of the same length; original entries are not mutated.
    """
    n = len(entries)
    debug = _is_debug()

    # Pre-compute durations and initial CPS.
    durations = []
    orig_cps_list = []
    for e in entries:
        dur = _parse_ts(e["end_str"]) - _parse_ts(e["start_str"])
        durations.append(dur)
        orig_cps_list.append(_cps(e["text"], dur))

    # -----------------------------------------------------------------------
    # Stage 1 — rule-based pass
    # -----------------------------------------------------------------------
    texts_after_rules = []
    rule_changed = []

    for i, entry in enumerate(entries):
        orig_text = entry["text"]
        if orig_cps_list[i] <= cfg.max_cps:
            texts_after_rules.append(orig_text)
            rule_changed.append(False)
            continue

        new_text = _apply_rules(orig_text)
        texts_after_rules.append(new_text)
        rule_changed.append(new_text != orig_text)

    # -----------------------------------------------------------------------
    # Stage 2 — OpenAI rewrite for cues still above hard_cps
    # -----------------------------------------------------------------------
    texts_final = list(texts_after_rules)

    if cfg.use_openai:
        hard_indices = [
            i for i, text in enumerate(texts_after_rules)
            if _cps(text, durations[i]) > cfg.hard_cps
        ]

        if hard_indices:
            print(
                f"  [Readability] Sending {len(hard_indices)} high-CPS cues to OpenAI for rewrite…"
            )
            batch_texts = [texts_after_rules[i] for i in hard_indices]
            batch_budgets = [
                max(1, int(durations[i] * cfg.target_cps)) for i in hard_indices
            ]
            rewritten = _openai_rewrite_batch(batch_texts, batch_budgets)
            for pos, idx in enumerate(hard_indices):
                candidate = str(rewritten[pos]).strip()
                current = texts_after_rules[idx]
                if (
                    candidate
                    and _char_count(candidate) <= batch_budgets[pos]
                    and _cps(candidate, durations[idx]) <= _cps(current, durations[idx])
                ):
                    texts_final[idx] = candidate

    # -----------------------------------------------------------------------
    # Summary logging
    # -----------------------------------------------------------------------
    cues_above_max_before = sum(1 for c in orig_cps_list if c > cfg.max_cps)
    cues_shortened = sum(1 for i in range(n) if texts_final[i] != entries[i]["text"])
    cues_above_max_after = sum(
        1 for i in range(n) if _cps(texts_final[i], durations[i]) > cfg.max_cps
    )

    valid = [(orig_cps_list[i], _cps(texts_final[i], durations[i])) for i in range(n) if durations[i] > 0]
    avg_before = sum(b for b, _ in valid) / len(valid) if valid else 0.0
    avg_after = sum(a for _, a in valid) / len(valid) if valid else 0.0

    print(
        f"\n[Readability Optimizer] total={n} | above_max_cps_before={cues_above_max_before} "
        f"| shortened={cues_shortened} | above_max_cps_after={cues_above_max_after} "
        f"| avg_cps {avg_before:.1f}→{avg_after:.1f}"
    )

    if debug:
        examples = [
            i for i in range(n)
            if texts_final[i] != entries[i]["text"]
        ][:3]
        if examples:
            print("  [Readability] Examples:")
            for i in examples:
                print(f"    [{i + 1}] before: {entries[i]['text']}")
                print(f"    [{i + 1}]  after: {texts_final[i]}")

    # -----------------------------------------------------------------------
    # Build result
    # -----------------------------------------------------------------------
    return [
        {**entry, "text": texts_final[i]}
        for i, entry in enumerate(entries)
    ]


# ---------------------------------------------------------------------------
# Public API — entry-list level
# ---------------------------------------------------------------------------

def optimize_vietnamese_subtitle_readability(
    entries: List[dict],
    config: Optional[ReadabilityConfig] = None,
    on_progress=None,
) -> List[dict]:
    """
    Shorten Vietnamese subtitle text in *entries* to improve readability.

    For each cue whose characters-per-second exceeds *config.max_cps*, the
    text is shortened first via rule-based replacements, then optionally via
    an OpenAI rewrite if it still exceeds *config.hard_cps*.

    Timing (``start_str`` / ``end_str``) and entry count are never changed.
    Returns a new list; *entries* is not mutated.
    """
    if config is None:
        config = load_readability_config()

    if not config.enabled:
        return entries

    return _optimize_entries(entries, config)


# ---------------------------------------------------------------------------
# Public API — file level
# ---------------------------------------------------------------------------

def _persist_readability_artifact(src_path: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    shutil.copy2(src_path, dest_path)


def optimize_readability_file(
    input_srt_path: str,
    output_srt_path: Optional[str] = None,
    config: Optional[ReadabilityConfig] = None,
    on_progress=None,
    *,
    before_artifact_path: Optional[str] = None,
    after_artifact_path: Optional[str] = None,
) -> str:
    """
    Apply readability optimisation to an SRT file.

    If *output_srt_path* is ``None``, the input file is replaced in-place
    using an atomic temp-file swap.  Returns the path of the written file.

    Optional debug artifacts:
      before_artifact_path  copy input before processing
      after_artifact_path   copy output after processing
    """
    from .utils import parse_srt, write_srt_entries

    if config is None:
        config = load_readability_config()

    if before_artifact_path:
        _persist_readability_artifact(input_srt_path, before_artifact_path)

    if not config.enabled:
        if after_artifact_path:
            _persist_readability_artifact(input_srt_path, after_artifact_path)
        return input_srt_path

    with open(input_srt_path, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    optimized = _optimize_entries(entries, config)

    in_place = output_srt_path is None
    if in_place:
        srt_dir = os.path.dirname(os.path.abspath(input_srt_path))
        fd, tmp_path = tempfile.mkstemp(suffix=".srt", dir=srt_dir)
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                write_srt_entries(optimized, file=f)
            if after_artifact_path:
                _persist_readability_artifact(tmp_path, after_artifact_path)
            os.replace(tmp_path, input_srt_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return input_srt_path

    with open(output_srt_path, "w", encoding="utf-8") as f:
        write_srt_entries(optimized, file=f)
    if after_artifact_path:
        _persist_readability_artifact(output_srt_path, after_artifact_path)
    return output_srt_path
