import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.voiceover.audio_builder import (  # noqa: E402
    SegmentManifest,
    build_segment_manifests,
    build_manifest_summary,
)
from auto_subtitle.voiceover.srt_parser import VoiceoverCue  # noqa: E402
from auto_subtitle.voiceover.timing_planner import (  # noqa: E402
    TimingPlan,
    plan_timing,
)


class SegmentManifestTests(unittest.TestCase):
    def test_tts_shorter_than_cue_is_ok(self) -> None:
        cues = [
            VoiceoverCue(index=1, start_ms=0, end_ms=5_000, text="A"),
        ]
        plans = plan_timing(cues, [3_000], video_duration_ms=20_000)
        manifests = build_segment_manifests(cues, [Path("segments/0001.wav")], plans)
        self.assertEqual(manifests[0].overflow_ms, 0)
        self.assertFalse(manifests[0].has_overflow)
        self.assertEqual(manifests[0].status, "ok")
        self.assertEqual(manifests[0].planned_start_ms, 0)
        self.assertEqual(manifests[0].planned_end_ms, 3_000)

    def test_tts_longer_than_cue_borrows_gap_when_available(self) -> None:
        cues = [
            VoiceoverCue(index=1, start_ms=1_000, end_ms=4_000, text="B"),
            VoiceoverCue(index=2, start_ms=6_500, end_ms=8_000, text="C"),
        ]
        plans = plan_timing(
            cues,
            [4_200, 1_000],
            video_duration_ms=20_000,
            min_gap_ms=120,
            max_borrow_after_ms=1_200,
        )
        self.assertEqual(plans[0].borrowed_gap_after_ms, 1_200)
        self.assertEqual(plans[0].status, "extended_into_gap")
        manifests = build_segment_manifests(
            cues,
            [Path("segments/0001.wav"), Path("segments/0002.wav")],
            plans,
        )
        self.assertEqual(manifests[0].planned_end_ms, 5_200)
        self.assertEqual(manifests[0].status, "extended_into_gap")

    def test_overflow_warning_when_borrowing_is_insufficient(self) -> None:
        cues = [
            VoiceoverCue(index=1, start_ms=0, end_ms=2_000, text="one"),
            VoiceoverCue(index=2, start_ms=2_500, end_ms=5_000, text="two"),
        ]
        plans = plan_timing(cues, [3_000, 1_000], video_duration_ms=20_000)
        self.assertEqual(plans[0].status, "overflow_warning")
        self.assertGreater(plans[0].overlap_next_ms, 0)

    def test_severe_overflow_when_remaining_overflow_is_large(self) -> None:
        cues = [
            VoiceoverCue(index=1, start_ms=0, end_ms=1_500, text="overflow"),
            VoiceoverCue(index=2, start_ms=2_000, end_ms=3_000, text="next"),
        ]
        plans = plan_timing(
            cues,
            [5_500, 1_000],
            video_duration_ms=20_000,
            severe_overflow_ms=2_000,
        )
        self.assertEqual(plans[0].status, "severe_overflow")
        self.assertGreater(plans[0].overflow_ms, 2_000)

    def test_back_to_back_cues_preserve_min_gap(self) -> None:
        cues = [
            VoiceoverCue(index=1, start_ms=0, end_ms=2_000, text="one"),
            VoiceoverCue(index=2, start_ms=2_050, end_ms=4_000, text="two"),
        ]
        plans = plan_timing(cues, [2_500, 1_000], video_duration_ms=20_000, min_gap_ms=120)
        self.assertEqual(plans[0].borrowed_gap_after_ms, 0)
        self.assertEqual(plans[0].status, "overflow_warning")

    def test_last_cue_can_extend_to_video_duration(self) -> None:
        cues = [
            VoiceoverCue(index=1, start_ms=9_000, end_ms=10_000, text="ending"),
        ]
        plans = plan_timing(cues, [1_800], video_duration_ms=10_900, min_gap_ms=120)
        self.assertEqual(plans[0].borrowed_gap_after_ms, 800)
        self.assertEqual(plans[0].status, "extended_into_gap")

    def test_manifest_summary_counts_are_correct(self) -> None:
        plans = [
            TimingPlan(
                index=1,
                text="one",
                original_start_ms=0,
                original_end_ms=2_000,
                cue_duration_ms=2_000,
                tts_duration_ms=1_500,
                planned_start_ms=0,
                planned_end_ms=1_500,
                overflow_ms=0,
                borrowed_gap_after_ms=0,
                overlap_next_ms=0,
                status="ok",
            ),
            TimingPlan(
                index=2,
                text="two",
                original_start_ms=2_500,
                original_end_ms=4_000,
                cue_duration_ms=1_500,
                tts_duration_ms=2_000,
                planned_start_ms=2_500,
                planned_end_ms=4_500,
                overflow_ms=500,
                borrowed_gap_after_ms=500,
                overlap_next_ms=0,
                status="extended_into_gap",
            ),
            TimingPlan(
                index=3,
                text="three",
                original_start_ms=5_000,
                original_end_ms=6_000,
                cue_duration_ms=1_000,
                tts_duration_ms=4_200,
                planned_start_ms=5_000,
                planned_end_ms=6_200,
                overflow_ms=3_200,
                borrowed_gap_after_ms=200,
                overlap_next_ms=2_000,
                status="severe_overflow",
            ),
        ]
        summary = build_manifest_summary(plans)
        self.assertEqual(summary["cue_count"], 3)
        self.assertEqual(summary["ok_count"], 1)
        self.assertEqual(summary["extended_count"], 1)
        self.assertEqual(summary["overflow_warning_count"], 0)
        self.assertEqual(summary["severe_overflow_count"], 1)
        self.assertEqual(summary["max_overflow_ms"], 3_200)
        self.assertEqual(summary["total_borrowed_gap_ms"], 700)
        self.assertIsInstance(plans[0], TimingPlan)
        self.assertIsInstance(
            build_segment_manifests(
                [VoiceoverCue(index=1, start_ms=0, end_ms=2_000, text="x")],
                [Path("segments/0001.wav")],
                [
                    TimingPlan(
                        index=1,
                        text="x",
                        original_start_ms=0,
                        original_end_ms=2_000,
                        cue_duration_ms=2_000,
                        tts_duration_ms=1_000,
                        planned_start_ms=0,
                        planned_end_ms=1_000,
                        overflow_ms=0,
                        borrowed_gap_after_ms=0,
                        overlap_next_ms=0,
                        status="ok",
                    )
                ],
            )[0],
            SegmentManifest,
        )


if __name__ == "__main__":
    unittest.main()
