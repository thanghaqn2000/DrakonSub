import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.srt_audio.cue_service import (  # noqa: E402
    annotate_cues,
    load_effective_cues,
    save_edited_cues,
)
from auto_subtitle.subtitle_edit_service import SubtitleCue, write_srt  # noqa: E402


class SrtAudioCueServiceTests(unittest.TestCase):
    def test_annotate_includes_estimated_ms_and_issues(self) -> None:
        cues = [
            SubtitleCue(1, "00:00:00,000", "00:00:01,000", "short"),
            SubtitleCue(2, "00:00:01,000", "00:00:02,000", "x" * 80),
        ]
        rows = annotate_cues(cues, chars_per_second=13.0, saydi_speed=1.0)
        self.assertEqual(rows[0]["issues"], [])
        self.assertIn("too_long", rows[1]["issues"])
        self.assertGreater(rows[1]["estimated_ms"], 1000)

    def test_save_rejects_count_mismatch_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            original = [
                SubtitleCue(1, "00:00:00,000", "00:00:01,000", "a"),
                SubtitleCue(2, "00:00:01,000", "00:00:02,000", "b"),
            ]
            write_srt(original, job_dir / "input.srt")
            with self.assertRaises(ValueError):
                save_edited_cues(
                    job_dir,
                    [{"index": 1, "start": "00:00:00,000", "end": "00:00:01,000", "text": "a"}],
                )
            save_edited_cues(
                job_dir,
                [
                    {"index": 1, "start": "00:00:00,000", "end": "00:00:00,800", "text": "A"},
                    {"index": 2, "start": "00:00:01,000", "end": "00:00:02,000", "text": "B"},
                ],
            )
            loaded = load_effective_cues(job_dir)
            self.assertEqual(loaded[0].text, "A")
            self.assertTrue((job_dir / "edited.srt").is_file())


if __name__ == "__main__":
    unittest.main()
