"""
Vietnamese Subtitle Timing Optimizer
=====================================
Adjusts SRT cue start/end times so each subtitle stays on screen long enough
for a viewer to read the Vietnamese text comfortably, while staying close to
the original speech timing.

Algorithm (single forward pass, conservative):
  1. Compute characters-per-second (CPS) for each cue.
  2. If CPS is within the acceptable range and the duration meets the minimum,
     leave the cue untouched.
  3. If the cue needs more display time:
       a. Shift start_time earlier  (≤ max_early_shift, hard-clamped by prev cue).
       b. Extend end_time later     (≤ max_extend,       hard-clamped by next cue).
  4. Never overlap an adjacent cue by more than max_overlap_allowed.

Config (all tunable via environment variables):
  ENABLE_VI_TIMING_OPTIMIZER   true/false  (default: true)
  VI_SUBTITLE_READING_SPEED    slow/normal/fast  (default: normal)
  DRAKONSUB_DEBUG              true/false  (per-cue detail when enabled)
"""

import os
import re
import tempfile
from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_READING_SPEED_MAP = {
    "slow": 14.0,
    "normal": 17.0,
    "fast": 20.0,
}

# Conservative defaults — match the spec exactly.
_DEFAULT_TARGET_CPS: float = 17.0
_DEFAULT_MAX_CPS: float = 22.0
_DEFAULT_MIN_DURATION: float = 1.2      # seconds
_DEFAULT_MAX_EARLY_SHIFT: float = 0.45  # seconds
_DEFAULT_MAX_EXTEND: float = 0.90       # seconds
_DEFAULT_MIN_GAP: float = 0.05          # not used in hard clamp; informational
_DEFAULT_MAX_OVERLAP_ALLOWED: float = 0.10  # seconds


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TimingConfig:
    """
    All tunable parameters for the timing optimizer.

    Defaults match the project spec (conservative, readable-first).
    """

    enabled: bool = True
    target_cps: float = _DEFAULT_TARGET_CPS
    max_cps: float = _DEFAULT_MAX_CPS
    min_duration: float = _DEFAULT_MIN_DURATION
    max_early_shift: float = _DEFAULT_MAX_EARLY_SHIFT
    max_extend: float = _DEFAULT_MAX_EXTEND
    min_gap: float = _DEFAULT_MIN_GAP
    max_overlap_allowed: float = _DEFAULT_MAX_OVERLAP_ALLOWED


def load_timing_config() -> TimingConfig:
    """
    Build a TimingConfig from environment variables.

    Reads ENABLE_VI_TIMING_OPTIMIZER and VI_SUBTITLE_READING_SPEED after
    loading the project .env file.
    """
    from .config import load_env
    load_env()

    raw_enabled = os.getenv("ENABLE_VI_TIMING_OPTIMIZER", "true").strip().lower()
    enabled = raw_enabled in {"1", "true", "yes", "on"}

    speed = os.getenv("VI_SUBTITLE_READING_SPEED", "normal").strip().lower()
    target_cps = _READING_SPEED_MAP.get(speed, _DEFAULT_TARGET_CPS)
    cps_tolerance = _DEFAULT_MAX_CPS - _DEFAULT_TARGET_CPS

    return TimingConfig(
        enabled=enabled,
        target_cps=target_cps,
        max_cps=target_cps + cps_tolerance,
    )


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts: str) -> float:
    """
    Convert an SRT timestamp string ``HH:MM:SS,mmm`` to seconds (float).
    """
    ts = ts.strip()
    time_part, millis_str = ts.split(",")
    h, m, s = time_part.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(millis_str) / 1000.0


def _format_ts(seconds: float) -> str:
    """
    Convert seconds (float) to an SRT timestamp string ``HH:MM:SS,mmm``.
    """
    if seconds < 0.0:
        seconds = 0.0
    ms = round(seconds * 1000)
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1_000
    ms -= s * 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _char_count(text: str) -> int:
    """Count readable characters, collapsing internal whitespace."""
    return len(re.sub(r"\s+", " ", text).strip())


def _is_debug() -> bool:
    """Return True when DRAKONSUB_DEBUG is set to a truthy value."""
    return os.getenv("DRAKONSUB_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------

def _optimize_entries(
    entries: List[dict],
    cfg: TimingConfig,
) -> List[dict]:
    """
    Apply timing optimisation to a list of SRT entry dicts.

    Each entry must have ``start_str``, ``end_str``, and ``text`` keys as
    produced by ``utils.parse_srt``.  Returns a new list; input is not
    mutated.

    Algorithm uses a single forward pass so that adjustments to cue *i*
    (specifically its adjusted end time) constrain cue *i+1*.
    """
    n = len(entries)
    if n == 0:
        return entries

    # Pre-parse timestamps to floats for arithmetic.
    starts = [_parse_ts(e["start_str"]) for e in entries]
    ends = [_parse_ts(e["end_str"]) for e in entries]

    new_starts = list(starts)
    new_ends = list(ends)

    debug = _is_debug()

    # Accumulators for summary stats.
    total_valid = 0
    sum_orig_cps = 0.0
    adjusted_count = 0

    for i, entry in enumerate(entries):
        text = entry.get("text", "")
        tlen = _char_count(text)
        orig_start = starts[i]
        orig_end = ends[i]
        orig_dur = orig_end - orig_start

        if tlen == 0 or orig_dur <= 0:
            continue

        total_valid += 1
        orig_cps = tlen / orig_dur
        sum_orig_cps += orig_cps

        required_dur = max(cfg.min_duration, tlen / cfg.target_cps)
        needs_more = orig_cps > cfg.max_cps or orig_dur < cfg.min_duration

        if not needs_more:
            continue

        deficit = required_dur - orig_dur
        if deficit <= 0:
            continue

        # ------------------------------------------------------------------
        # Step 1 — shift start_time earlier
        # ------------------------------------------------------------------
        # Use the *adjusted* end of the previous cue so cascades are safe.
        prev_adj_end = new_ends[i - 1] if i > 0 else 0.0

        # Hard floor: do not start earlier than (prev_adj_end - max_overlap_allowed)
        hard_floor = prev_adj_end - cfg.max_overlap_allowed
        # Soft floor: do not shift more than max_early_shift
        soft_floor = orig_start - cfg.max_early_shift
        # Earliest allowed new_start
        earliest = max(hard_floor, soft_floor)
        # How much we can actually pull back
        max_shift = max(0.0, orig_start - earliest)
        actual_shift = min(deficit, max_shift)

        ns = orig_start - actual_shift
        remaining = deficit - actual_shift

        # ------------------------------------------------------------------
        # Step 2 — extend end_time later
        # ------------------------------------------------------------------
        # Use the *original* start of the next cue (not yet adjusted).
        next_orig_start = starts[i + 1] if i + 1 < n else float("inf")

        # Hard ceiling: do not extend past (next_start + max_overlap_allowed)
        hard_ceil = next_orig_start + cfg.max_overlap_allowed
        # Soft ceiling: do not extend more than max_extend
        soft_ceil = orig_end + cfg.max_extend
        # Latest allowed new_end
        latest = min(hard_ceil, soft_ceil)
        # How much we can extend
        max_ext = max(0.0, latest - orig_end)
        actual_ext = min(remaining, max_ext)

        ne = orig_end + actual_ext

        # ------------------------------------------------------------------
        # Commit if anything changed
        # ------------------------------------------------------------------
        if actual_shift > 0.001 or actual_ext > 0.001:
            new_starts[i] = ns
            new_ends[i] = ne
            adjusted_count += 1

            if debug:
                new_dur = ne - ns
                new_cps = tlen / new_dur if new_dur > 0 else 0.0
                shift_parts = []
                if actual_shift > 0.001:
                    shift_parts.append(f"start -{actual_shift:.2f}s")
                if actual_ext > 0.001:
                    shift_parts.append(f"end +{actual_ext:.2f}s")
                reason_str = ", ".join(shift_parts) or "no room"
                trigger = "cps>max" if orig_cps > cfg.max_cps else "dur<min"
                print(
                    f"  [Optimizer] cue {i + 1:4d} | "
                    f"len={tlen:3d} | "
                    f"orig_dur={orig_dur:.2f}s new_dur={new_dur:.2f}s | "
                    f"cps {orig_cps:.1f}→{new_cps:.1f} | "
                    f"[{orig_start:.3f}→{ns:.3f}, {orig_end:.3f}→{ne:.3f}] | "
                    f"{trigger} | {reason_str}"
                )

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    sum_new_cps = 0.0
    still_above_max = 0
    for i, entry in enumerate(entries):
        tlen = _char_count(entry.get("text", ""))
        dur = new_ends[i] - new_starts[i]
        if tlen > 0 and dur > 0:
            cps = tlen / dur
            sum_new_cps += cps
            if cps > cfg.max_cps:
                still_above_max += 1

    avg_before = sum_orig_cps / total_valid if total_valid else 0.0
    avg_after = sum_new_cps / total_valid if total_valid else 0.0
    print(
        f"\n[Timing Optimizer] total={n} | adjusted={adjusted_count} "
        f"| avg_cps {avg_before:.1f}→{avg_after:.1f} "
        f"| still_above_max_cps={still_above_max}"
    )

    # Build result with updated timestamps.
    result = []
    for i, entry in enumerate(entries):
        new_entry = dict(entry)
        new_entry["start_str"] = _format_ts(new_starts[i])
        new_entry["end_str"] = _format_ts(new_ends[i])
        result.append(new_entry)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_srt_timing_file(
    input_srt_path: str,
    output_srt_path: Optional[str] = None,
    config: Optional[TimingConfig] = None,
    on_progress=None,
) -> str:
    """
    Optimise Vietnamese subtitle timing in an SRT file for comfortable reading.

    Reads *input_srt_path*, adjusts cue start/end times based on
    character-per-second reading speed, and writes the result.

    Parameters
    ----------
    input_srt_path:
        Path to the input ``.srt`` file (Vietnamese text, original timing).
    output_srt_path:
        Destination path.  If ``None``, the input file is replaced in-place
        using an atomic temp-file swap so the original is never half-written.
    config:
        ``TimingConfig`` instance.  When ``None``, config is loaded from
        environment variables via ``load_timing_config()``.
    on_progress:
        Optional progress callback ``(message: str, percent: int) -> None``.

    Returns
    -------
    str
        Path of the written file (``input_srt_path`` for in-place mode).
    """
    from .utils import parse_srt, write_srt_entries

    if config is None:
        config = load_timing_config()

    if not config.enabled:
        return input_srt_path

    with open(input_srt_path, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    optimized = _optimize_entries(entries, config)

    in_place = output_srt_path is None
    if in_place:
        # Write to a temp file in the same directory, then atomically replace.
        srt_dir = os.path.dirname(os.path.abspath(input_srt_path))
        fd, tmp_path = tempfile.mkstemp(suffix=".srt", dir=srt_dir)
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                write_srt_entries(optimized, file=f)
            os.replace(tmp_path, input_srt_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return input_srt_path

    with open(output_srt_path, "w", encoding="utf-8") as f:
        write_srt_entries(optimized, file=f)
    return output_srt_path
