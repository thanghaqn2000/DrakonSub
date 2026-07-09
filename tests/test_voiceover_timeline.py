import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.voiceover.audio_builder import (  # noqa: E402
    SegmentManifest,
    build_segment_manifests,
)
from auto_subtitle.voiceover.srt_parser import VoiceoverCue  # noqa: E402


class SegmentManifestTests(unittest.TestCase):
    def test_tts_shorter_than_cue_has_no_overflow(self) -> None:
        cues = [
            VoiceoverCue(index=1, start_ms=0, end_ms=5_000, text="A"),
        ]
        manifests = build_segment_manifests(
            cues,
            [Path("segments/0001.wav")],
            [3_000],
        )
        self.assertEqual(manifests[0].overflow_ms, 0)
        self.assertFalse(manifests[0].has_overflow)
        self.assertEqual(manifests[0].placement, "start_aligned")

    def test_tts_longer_than_cue_sets_overflow(self) -> None:
        cues = [
            VoiceoverCue(index=1, start_ms=1_000, end_ms=4_000, text="B"),
        ]
        manifests = build_segment_manifests(
            cues,
            [Path("segments/0001.wav")],
            [5_500],
        )
        self.assertEqual(manifests[0].cue_duration_ms, 3_000)
        self.assertEqual(manifests[0].tts_duration_ms, 5_500)
        self.assertEqual(manifests[0].overflow_ms, 2_500)
        self.assertTrue(manifests[0].has_overflow)

    def test_multiple_cues_align_paths(self) -> None:
        cues = [
            VoiceoverCue(index=1, start_ms=0, end_ms=2_000, text="one"),
            VoiceoverCue(index=2, start_ms=2_500, end_ms=5_000, text="two"),
        ]
        paths = [Path("segments/0001.wav"), Path("segments/0002.wav")]
        durations = [1_800, 2_000]
        manifests = build_segment_manifests(cues, paths, durations)
        self.assertEqual(len(manifests), 2)
        self.assertEqual(manifests[1].cue_start_ms, 2_500)
        self.assertIsInstance(manifests[0], SegmentManifest)


if __name__ == "__main__":
    unittest.main()
