import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.srt_audio.timing import (  # noqa: E402
    estimate_speech_ms,
    ms_to_srt_timestamp,
    parse_srt_timestamp_to_ms,
    plan_cascade_starts,
    validate_cue_timings,
)


class SrtAudioTimingTests(unittest.TestCase):
    def test_parse_and_format_roundtrip(self) -> None:
        ms = parse_srt_timestamp_to_ms("00:01:02,345")
        self.assertEqual(ms, 62_345)
        self.assertEqual(ms_to_srt_timestamp(ms), "00:01:02,345")

    def test_estimate_scales_with_speed(self) -> None:
        base = estimate_speech_ms("abcdefghij", chars_per_second=10.0, saydi_speed=1.0)
        fast = estimate_speech_ms("abcdefghij", chars_per_second=10.0, saydi_speed=2.0)
        self.assertEqual(base, 1_000)
        self.assertEqual(fast, 500)

    def test_validate_flags_overlap_empty_and_too_long(self) -> None:
        cues = [
            {"index": 1, "start": "00:00:00,000", "end": "00:00:01,000", "text": "A" * 40},
            {"index": 2, "start": "00:00:00,500", "end": "00:00:02,000", "text": "ok"},
            {"index": 3, "start": "00:00:03,000", "end": "00:00:02,000", "text": "  "},
        ]
        issues = validate_cue_timings(cues, chars_per_second=13.0, saydi_speed=1.0)
        self.assertIn("overlap_next", issues[0])
        self.assertIn("too_long", issues[0])
        self.assertIn("start_after_end", issues[2])
        self.assertIn("empty_text", issues[2])


class CascadePlacementTests(unittest.TestCase):
    def test_respects_intent_when_gap_available(self) -> None:
        starts = plan_cascade_starts(
            intent_starts_ms=[0, 4000],
            durations_ms=[3000, 1000],
            gap_ms=280,
            max_gap_ms=2000,
        )
        self.assertEqual(starts, [0, 4000])

    def test_pushes_next_when_overflow(self) -> None:
        starts = plan_cascade_starts(
            intent_starts_ms=[0, 3000],
            durations_ms=[5000, 2000],
            gap_ms=280,
            max_gap_ms=2000,
        )
        self.assertEqual(starts, [0, 5280])

    def test_pulls_next_when_silence_exceeds_max_gap(self) -> None:
        # cue0 ends 3000; intent1=6000 (3s leftover) → cap rest at 2s → start 5000
        starts = plan_cascade_starts(
            intent_starts_ms=[0, 6000],
            durations_ms=[3000, 1000],
            gap_ms=280,
            max_gap_ms=2000,
        )
        self.assertEqual(starts, [0, 5000])

    def test_cascade_chain(self) -> None:
        starts = plan_cascade_starts(
            intent_starts_ms=[3000, 11000, 17000],
            durations_ms=[9000, 7000, 1000],
            gap_ms=280,
            max_gap_ms=2000,
        )
        self.assertEqual(starts, [3000, 12280, 19560])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            plan_cascade_starts([0], [1, 2], gap_ms=280)

    def test_max_gap_below_min_raises(self) -> None:
        with self.assertRaises(ValueError):
            plan_cascade_starts([0, 1000], [100, 100], gap_ms=500, max_gap_ms=200)


if __name__ == "__main__":
    unittest.main()
