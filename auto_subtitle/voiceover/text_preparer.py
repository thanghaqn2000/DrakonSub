from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from auto_subtitle.translation_topics import normalize_topic

from .srt_parser import VoiceoverCue

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"([.!?,;:])\1+")
_CLAUSE_SPLIT_RE = re.compile(r"\s*,\s*|\s*;\s*|\s*:\s*")

_CATHOLIC_TERMS = (
    "Chúa Giêsu Kitô",
    "Chúa Giêsu",
    "Thiên Chúa",
    "Chúa Thánh Thần",
    "Tin Mừng",
    "Kinh Thánh",
    "đức tin",
    "ân sủng",
    "ơn cứu độ",
    "tội lỗi",
    "sám hối",
    "hoán cải",
    "môn đệ",
    "tông đồ",
    "Thánh lễ",
    "bài giảng",
    "lời cầu nguyện",
)


@dataclass(frozen=True)
class PreparedVoiceoverCue:
    index: int
    start_ms: int
    end_ms: int
    original_text: str
    prepared_text: str
    original_char_count: int
    prepared_char_count: int
    target_char_count: int
    reduction_ratio: float
    status: str
    warnings: list[str]


def _normalize_text(text: str) -> str:
    value = (text or "").strip()
    value = _WHITESPACE_RE.sub(" ", value)
    value = _PUNCT_RE.sub(r"\1", value)
    value = value.replace("...", ".")
    return value.strip()


def _prefer_shorter_form(text: str) -> str:
    replacements = (
        ("chúng ta được mời gọi", "ta được mời gọi"),
        ("chúng ta đang", "ta đang"),
        ("chúng ta", "ta"),
        ("vẫn cần", "cần"),
        ("thật ", ""),
        ("rất ", ""),
    )
    result = text
    for old, new in replacements:
        result = result.replace(old, new)
    return _normalize_text(result)


def _shorten_text(text: str, target_char_count: int) -> tuple[str, list[str]]:
    warnings: list[str] = []
    normalized = _prefer_shorter_form(text)
    if len(normalized) <= target_char_count:
        return normalized, warnings

    clauses = [part.strip() for part in _CLAUSE_SPLIT_RE.split(normalized) if part.strip()]
    preserved = [part for part in clauses if any(term in part for term in _CATHOLIC_TERMS)]
    chosen: list[str] = []
    for part in preserved + clauses:
        if part in chosen:
            continue
        candidate = ", ".join(chosen + [part]).strip(", ")
        if not chosen or len(candidate) <= target_char_count:
            chosen.append(part)
        if len(", ".join(chosen)) >= target_char_count:
            break

    shortened = ", ".join(chosen).strip(", ")
    if not shortened:
        shortened = normalized[:target_char_count].rstrip(",;: ")
    shortened = _normalize_text(shortened)
    if len(shortened) > target_char_count:
        warnings.append("text_exceeds_target_after_preparation")
    return shortened, warnings


def prepare_voiceover_cues(
    cues: list[VoiceoverCue],
    *,
    topic: str = "catholic",
    max_chars_per_second: float = 13.0,
) -> list[PreparedVoiceoverCue]:
    normalize_topic(topic)
    prepared: list[PreparedVoiceoverCue] = []
    for cue in cues:
        original_text = _normalize_text(cue.text)
        target_char_count = max(20, int((cue.duration_ms / 1000.0) * max_chars_per_second))
        if not original_text:
            prepared.append(
                PreparedVoiceoverCue(
                    index=cue.index,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    original_text=cue.text,
                    prepared_text="",
                    original_char_count=len(cue.text.strip()),
                    prepared_char_count=0,
                    target_char_count=target_char_count,
                    reduction_ratio=1.0,
                    status="empty_or_invalid",
                    warnings=["empty_text"],
                )
            )
            continue

        if len(original_text) <= target_char_count:
            prepared.append(
                PreparedVoiceoverCue(
                    index=cue.index,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    original_text=original_text,
                    prepared_text=original_text,
                    original_char_count=len(original_text),
                    prepared_char_count=len(original_text),
                    target_char_count=target_char_count,
                    reduction_ratio=0.0,
                    status="ok",
                    warnings=[],
                )
            )
            continue

        shortened_text, warnings = _shorten_text(original_text, target_char_count)
        original_char_count = len(original_text)
        prepared_char_count = len(shortened_text)
        reduction_ratio = 0.0 if original_char_count == 0 else max(
            0.0, (original_char_count - prepared_char_count) / original_char_count
        )

        if prepared_char_count > target_char_count:
            status = "too_long"
            if "text_exceeds_target_after_preparation" not in warnings:
                warnings.append("text_exceeds_target_after_preparation")
        elif prepared_char_count < original_char_count:
            status = "shortened"
        else:
            status = "ok"

        prepared.append(
            PreparedVoiceoverCue(
                index=cue.index,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                original_text=original_text,
                prepared_text=shortened_text,
                original_char_count=original_char_count,
                prepared_char_count=prepared_char_count,
                target_char_count=target_char_count,
                reduction_ratio=reduction_ratio,
                status=status,
                warnings=warnings,
            )
        )
    return prepared


def summarize_prepared_cues(cues: Iterable[PreparedVoiceoverCue]) -> dict:
    items = list(cues)
    return {
        "text_ok_count": sum(1 for item in items if item.status == "ok"),
        "text_shortened_count": sum(1 for item in items if item.status == "shortened"),
        "text_too_long_count": sum(1 for item in items if item.status == "too_long"),
        "total_original_chars": sum(item.original_char_count for item in items),
        "total_prepared_chars": sum(item.prepared_char_count for item in items),
        "average_reduction_ratio": (
            round(sum(item.reduction_ratio for item in items) / len(items), 4) if items else 0.0
        ),
    }


def _ms_to_srt_timestamp(value: int) -> str:
    hours, rem = divmod(value, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_prepared_srt(cues: list[PreparedVoiceoverCue], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for cue in cues:
        text = cue.prepared_text.strip() or " "
        blocks.append(
            f"{cue.index}\n"
            f"{_ms_to_srt_timestamp(cue.start_ms)} --> {_ms_to_srt_timestamp(cue.end_ms)}\n"
            f"{text}"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def prepared_to_voiceover_cues(cues: list[PreparedVoiceoverCue]) -> list[VoiceoverCue]:
    return [
        VoiceoverCue(
            index=cue.index,
            start_ms=cue.start_ms,
            end_ms=cue.end_ms,
            text=cue.prepared_text,
        )
        for cue in cues
    ]
