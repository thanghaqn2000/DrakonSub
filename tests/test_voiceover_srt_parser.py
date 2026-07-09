import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.voiceover.srt_parser import (  # noqa: E402
    VoiceoverCue,
    VoiceoverSrtError,
    parse_timestamp_to_ms,
    parse_voiceover_srt,
)


FIXTURE = ROOT / "tests" / "fixtures" / "final_voiceover_vi_minimal.srt"


class ParseTimestampTests(unittest.TestCase):
    def test_zero(self) -> None:
        self.assertEqual(parse_timestamp_to_ms("00:00:00,000"), 0)

    def test_mixed(self) -> None:
        self.assertEqual(parse_timestamp_to_ms("00:01:02,345"), 62_345)

    def test_invalid(self) -> None:
        with self.assertRaises(VoiceoverSrtError):
            parse_timestamp_to_ms("bad")


class ParseVoiceoverSrtTests(unittest.TestCase):
    def test_fixture_has_five_cues(self) -> None:
        cues = parse_voiceover_srt(FIXTURE)
        self.assertEqual(len(cues), 5)
        self.assertIsInstance(cues[0], VoiceoverCue)
        self.assertEqual(cues[0].index, 1)
        self.assertEqual(cues[0].start_ms, 0)
        self.assertEqual(cues[0].duration_ms, 6_000)
        self.assertIn("Tin Mừng", cues[0].text)

    def test_sequential_indices_required(self) -> None:
        bad = "2\n00:00:00,000 --> 00:00:01,000\nHi"
        path = Path("/tmp/drakonsub_bad_voiceover.srt")
        path.write_text(bad, encoding="utf-8")
        try:
            with self.assertRaises(VoiceoverSrtError):
                parse_voiceover_srt(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
