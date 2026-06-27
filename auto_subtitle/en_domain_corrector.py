"""
Rule-based English ASR domain correction for investing/economics niche.

Runs after transcription and before translation. No LLM calls.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import (
    en_domain_correction_enabled,
    en_domain_correction_mode,
    en_domain_correction_save_debug,
    load_env,
)

# Terms that signal investing/Buffett/Bitcoin context in the full clip transcript.
_INVESTING_CONTEXT_TERMS: Sequence[str] = (
    "bitcoin",
    "buffett",
    "warren",
    "berkshire",
    "asset",
    "productive asset",
    "satoshi",
    "coin",
    "invest",
    "stock",
    "shareholder",
    "munger",
    "hathaway",
    "rental",
    "productive",
)


@dataclass(frozen=True)
class DomainCorrectionRule:
    """A single regex-based ASR correction rule."""

    name: str
    pattern: str
    replacement: str
    requires_context: bool = False
    confidence: str = "high"  # high = safe proper-noun fix; medium = needs context guard


# Longer / more specific patterns first.
_DOMAIN_RULES: List[DomainCorrectionRule] = [
    # high confidence: Buffett/Bitcoin joke misheard as "buffer coin"
    DomainCorrectionRule(
        name="call_a_buffer_coin",
        pattern=r"\bcall\s+a\s+buffer\s+coin\b",
        replacement="call it Buffett Coin",
        requires_context=True,
        confidence="high",
    ),
    DomainCorrectionRule(
        name="call_a_buffet_coin",
        pattern=r"\bcall\s+a\s+buffet\s+coin\b",
        replacement="call it Buffett Coin",
        requires_context=True,
        confidence="high",
    ),
    DomainCorrectionRule(
        name="buffer_coin",
        pattern=r"\bbuffer\s+coin\b",
        replacement="Buffett Coin",
        requires_context=True,
        confidence="high",
    ),
    DomainCorrectionRule(
        name="buffet_coin",
        pattern=r"\bbuffet\s+coin\b",
        replacement="Buffett Coin",
        requires_context=True,
        confidence="high",
    ),
    DomainCorrectionRule(
        name="warren_buffet",
        pattern=r"\bWarren\s+buffet\b",
        replacement="Warren Buffett",
        confidence="high",
    ),
    DomainCorrectionRule(
        name="berkshire_half_away",
        pattern=r"\bBerkshire\s+half\s+away\b",
        replacement="Berkshire Hathaway",
        confidence="high",
    ),
    DomainCorrectionRule(
        name="berkshire_hath_a_way",
        pattern=r"\bBerkshire\s+hath\s+a\s+way\b",
        replacement="Berkshire Hathaway",
        confidence="high",
    ),
    DomainCorrectionRule(
        name="bit_coin",
        pattern=r"\bbit\s+coin\b",
        replacement="Bitcoin",
        confidence="high",
    ),
    DomainCorrectionRule(
        name="satoshi_nakamotto",
        pattern=r"\bSatoshi\s+Nakamotto\b",
        replacement="Satoshi Nakamoto",
        confidence="high",
    ),
    DomainCorrectionRule(
        name="satoshi_nakamoto_misspell",
        pattern=r"\bSatoshi\s+Nakamato\b",
        replacement="Satoshi Nakamoto",
        confidence="high",
    ),
    DomainCorrectionRule(
        name="charlie_monger",
        pattern=r"\bCharlie\s+monger\b",
        replacement="Charlie Munger",
        confidence="high",
    ),
    DomainCorrectionRule(
        name="share_holders",
        pattern=r"\bshare\s+holders\b",
        replacement="shareholders",
        requires_context=True,
        confidence="medium",
    ),
    DomainCorrectionRule(
        name="stock_holder",
        pattern=r"\bstock\s+holder\b",
        replacement="shareholder",
        requires_context=True,
        confidence="medium",
    ),
    DomainCorrectionRule(
        name="bitcoins_general",
        pattern=r"\bbitcoins\b",
        replacement="Bitcoin",
        requires_context=True,
        confidence="medium",
    ),
    DomainCorrectionRule(
        name="cashflow",
        pattern=r"\bcashflow\b",
        replacement="cash flow",
        confidence="high",
    ),
]


def _has_investing_context(full_text: str) -> bool:
    lowered = full_text.lower()
    return any(term in lowered for term in _INVESTING_CONTEXT_TERMS)


def _apply_rule(text: str, rule: DomainCorrectionRule, *, has_context: bool) -> str:
    if rule.requires_context and not has_context:
        return text
    return re.sub(rule.pattern, rule.replacement, text, flags=re.IGNORECASE)


def correct_en_domain_entries(
    entries: List[dict],
    *,
    mode: Optional[str] = None,
) -> List[dict]:
    """
    Return a new entry list with rule-based domain corrections applied.

    Cue count, order, and timestamps are never changed.
    """
    load_env()
    if not en_domain_correction_enabled():
        print("[EN Domain Correction] disabled")
        return list(entries)

    mode = (mode or en_domain_correction_mode()).strip().lower()
    if mode != "rules":
        print(f"[EN Domain Correction] unsupported mode '{mode}', skipping")
        return list(entries)

    full_text = "\n".join(entry.get("text", "") for entry in entries)
    has_context = _has_investing_context(full_text)
    print(
        f"[EN Domain Correction] enabled mode={mode} "
        f"investing_context={'yes' if has_context else 'no'}"
    )

    corrected: List[dict] = []
    change_count = 0

    for i, entry in enumerate(entries):
        original = entry.get("text", "")
        updated = original
        applied_rules: List[str] = []

        for rule in _DOMAIN_RULES:
            new_text = _apply_rule(updated, rule, has_context=has_context)
            if new_text != updated:
                applied_rules.append(rule.name)
                updated = new_text

        if updated != original:
            change_count += 1
            print(
                f"  [EN Domain Correction] cue {i + 1} "
                f"rules={','.join(applied_rules)}"
            )
            print(f"    before: {original}")
            print(f"    after:  {updated}")

        corrected.append({**entry, "text": updated})

    print(f"[EN Domain Correction] changed_cues={change_count}/{len(entries)}")
    return corrected


def correct_en_domain_srt_file(
    input_srt_path: str,
    output_srt_path: Optional[str] = None,
    *,
    debug_dir: Optional[str] = None,
) -> str:
    """
    Apply domain correction to an English SRT file.

    When *output_srt_path* is None, replaces *input_srt_path* in place.
    """
    from .utils import parse_srt, write_srt_entries

    with open(input_srt_path, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    corrected = correct_en_domain_entries(entries)
    out_path = output_srt_path or input_srt_path

    if en_domain_correction_save_debug() and debug_dir:
        log_path = os.path.join(debug_dir, "en_domain_correction_log.json")
        changes = []
        for i, (before, after) in enumerate(zip(entries, corrected)):
            if before["text"] != after["text"]:
                changes.append(
                    {
                        "cue": i + 1,
                        "before": before["text"],
                        "after": after["text"],
                    }
                )
        os.makedirs(debug_dir, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "enabled": en_domain_correction_enabled(),
                    "mode": en_domain_correction_mode(),
                    "changed_cues": len(changes),
                    "changes": changes,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        write_srt_entries(corrected, file=f)
    return out_path
