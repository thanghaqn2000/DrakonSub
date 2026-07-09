from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TIMING_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}$"
)


class VoiceoverSrtError(ValueError):
    pass


@dataclass(frozen=True)
class VoiceoverCue:
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


def parse_timestamp_to_ms(timestamp: str) -> int:
    """Parse SRT timestamp ``HH:MM:SS,mmm`` to milliseconds."""
    try:
        hours, minutes, rest = timestamp.strip().split(":")
        seconds, millis = rest.split(",")
        return (
            int(hours) * 3_600_000
            + int(minutes) * 60_000
            + int(seconds) * 1_000
            + int(millis)
        )
    except (ValueError, AttributeError) as exc:
        raise VoiceoverSrtError(f"Invalid SRT timestamp: {timestamp!r}") from exc


def parse_voiceover_srt(path: Path) -> list[VoiceoverCue]:
    if not path.exists():
        raise VoiceoverSrtError(f"Subtitle file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    cues: list[VoiceoverCue] = []
    blocks = re.split(r"\r?\n\r?\n", content)
    for block in blocks:
        lines = [line.rstrip("\r") for line in block.splitlines()]
        if len(lines) < 3:
            raise VoiceoverSrtError("Malformed SRT block")
        try:
            index = int(lines[0].strip())
        except ValueError as exc:
            raise VoiceoverSrtError("Malformed SRT index") from exc
        timing = lines[1].strip()
        if not _TIMING_RE.match(timing):
            raise VoiceoverSrtError(f"Malformed SRT timing: {timing!r}")
        start_str, end_str = [part.strip() for part in timing.split("-->")]
        start_ms = parse_timestamp_to_ms(start_str)
        end_ms = parse_timestamp_to_ms(end_str)
        if end_ms < start_ms:
            raise VoiceoverSrtError(f"Cue {index} end precedes start")
        text = "\n".join(lines[2:]).strip()
        cues.append(
            VoiceoverCue(index=index, start_ms=start_ms, end_ms=end_ms, text=text)
        )

    for pos, cue in enumerate(cues, start=1):
        if cue.index != pos:
            raise VoiceoverSrtError("SRT cue indices must be sequential starting at 1")
    return cues
