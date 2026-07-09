import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.voiceover.srt_parser import VoiceoverCue, parse_voiceover_srt  # noqa: E402

from auto_subtitle.voiceover.text_preparer import (  # noqa: E402
    PreparedVoiceoverCue,
    prepare_voiceover_cues,
    summarize_prepared_cues,
    write_prepared_srt,
)

_CLI_PATH = ROOT / "scripts" / "prototype_voiceover_from_srt.py"
_CLI_SPEC = importlib.util.spec_from_file_location("prototype_voiceover_from_srt", _CLI_PATH)
assert _CLI_SPEC and _CLI_SPEC.loader
prototype_mod = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(prototype_mod)


def _ms_to_srt_timestamp(value: int) -> str:
    hours, rem = divmod(value, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


class TextPreparationTests(unittest.TestCase):
    def test_whitespace_normalization(self) -> None:
        cue = VoiceoverCue(index=1, start_ms=0, end_ms=2_000, text="  Xin   chào...   bạn  ")
        prepared = prepare_voiceover_cues([cue], topic="catholic", max_chars_per_second=13)
        self.assertEqual(prepared[0].prepared_text, "Xin chào. bạn")

    def test_short_cue_remains_unchanged(self) -> None:
        cue = VoiceoverCue(index=1, start_ms=0, end_ms=4_000, text="Chúa luôn yêu thương chúng ta.")
        prepared = prepare_voiceover_cues([cue], topic="catholic", max_chars_per_second=20)
        self.assertEqual(prepared[0].prepared_text, "Chúa luôn yêu thương chúng ta.")
        self.assertEqual(prepared[0].status, "ok")

    def test_long_cue_gets_shortened_or_marked_too_long(self) -> None:
        cue = VoiceoverCue(
            index=1,
            start_ms=0,
            end_ms=2_000,
            text=(
                "Trong bài giảng này, chúng ta được mời gọi suy ngẫm thật dài dòng về đức tin, "
                "về đời sống cầu nguyện, và về cách đáp lại tiếng Chúa trong từng hoàn cảnh."
            ),
        )
        prepared = prepare_voiceover_cues([cue], topic="catholic", max_chars_per_second=10)
        self.assertLess(prepared[0].prepared_char_count, prepared[0].original_char_count)
        self.assertIn(prepared[0].status, {"shortened", "too_long"})

    def test_catholic_terms_are_preserved(self) -> None:
        cue = VoiceoverCue(
            index=1,
            start_ms=0,
            end_ms=3_000,
            text="Chúa Giêsu Kitô ban ân sủng và Tin Mừng cho các môn đệ.",
        )
        prepared = prepare_voiceover_cues([cue], topic="catholic", max_chars_per_second=12)
        self.assertIn("Chúa Giêsu Kitô", prepared[0].prepared_text)
        self.assertIn("ân sủng", prepared[0].prepared_text)
        self.assertIn("Tin Mừng", prepared[0].prepared_text)

    def test_prepared_srt_writing_preserves_timestamps(self) -> None:
        cues = [
            PreparedVoiceoverCue(
                index=1,
                start_ms=1_000,
                end_ms=3_000,
                original_text="A",
                prepared_text="B",
                original_char_count=1,
                prepared_char_count=1,
                target_char_count=20,
                reduction_ratio=0.0,
                status="ok",
                warnings=[],
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "prepared.srt"
            write_prepared_srt(cues, out_path)
            loaded = parse_voiceover_srt(out_path)
        self.assertEqual(loaded[0].start_ms, 1_000)
        self.assertEqual(loaded[0].end_ms, 3_000)
        self.assertEqual(loaded[0].text, "B")

    @patch.object(prototype_mod, "mux_video_with_audio")
    @patch.object(prototype_mod, "mix_audio_tracks")
    @patch.object(prototype_mod, "build_voiceover_track")
    @patch.object(prototype_mod, "probe_audio_duration_ms", side_effect=[1000])
    @patch.object(prototype_mod, "synthesize_to_file")
    @patch.object(prototype_mod, "load_saydi_config", return_value=object())
    @patch.object(prototype_mod, "video_has_audio_stream", return_value=True)
    @patch.object(prototype_mod, "probe_video_duration_ms", return_value=20_000)
    def test_cli_without_prepare_text_preserves_phase2_behavior(
        self,
        _mock_video_duration,
        _mock_has_audio,
        _mock_load_cfg,
        mock_synthesize,
        _mock_probe_audio,
        _mock_build_track,
        _mock_mix_audio,
        _mock_mux,
    ) -> None:
        cue = VoiceoverCue(index=1, start_ms=0, end_ms=2_000, text="Giữ nguyên câu này.")
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "input.srt"
            srt_path.write_text(
                f"1\n{_ms_to_srt_timestamp(0)} --> {_ms_to_srt_timestamp(2000)}\n{cue.text}\n",
                encoding="utf-8",
            )
            manifest = prototype_mod.run_prototype(
                input_video=Path("sample-video.mp4"),
                voiceover_srt=srt_path,
                output_video=Path(tmpdir) / "out.mp4",
                job_dir=None,
                original_volume=0.3,
                voice_volume=1.0,
                min_gap_ms=120,
                max_borrow_after_ms=1200,
                severe_overflow_ms=2000,
                prepare_text=False,
                voiceover_topic="catholic",
                max_chars_per_second=13,
                prepared_srt_output=None,
                force=True,
            )
        self.assertFalse(manifest["text_preparation"]["enabled"])
        self.assertEqual(manifest["segments"][0]["original_text"], cue.text)
        self.assertEqual(manifest["segments"][0]["prepared_text"], cue.text)
        self.assertEqual(mock_synthesize.call_args.args[0], cue.text)

    @patch.object(prototype_mod, "mux_video_with_audio")
    @patch.object(prototype_mod, "mix_audio_tracks")
    @patch.object(prototype_mod, "build_voiceover_track")
    @patch.object(prototype_mod, "probe_audio_duration_ms", side_effect=[1000])
    @patch.object(prototype_mod, "synthesize_to_file")
    @patch.object(prototype_mod, "load_saydi_config", return_value=object())
    @patch.object(prototype_mod, "video_has_audio_stream", return_value=True)
    @patch.object(prototype_mod, "probe_video_duration_ms", return_value=20_000)
    def test_cli_with_prepare_text_writes_prepared_srt(
        self,
        _mock_video_duration,
        _mock_has_audio,
        _mock_load_cfg,
        mock_synthesize,
        _mock_probe_audio,
        _mock_build_track,
        _mock_mix_audio,
        _mock_mux,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "stress.srt"
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n"
                "Chúng ta đang có một câu thật dài, thật nhiều ý và thật khó đọc nếu đem đọc nguyên văn.\n",
                encoding="utf-8",
            )
            prepared_srt = Path(tmpdir) / "prepared_voiceover.srt"
            manifest = prototype_mod.run_prototype(
                input_video=Path("sample-video.mp4"),
                voiceover_srt=srt_path,
                output_video=Path(tmpdir) / "out.mp4",
                job_dir=None,
                original_volume=0.3,
                voice_volume=1.0,
                min_gap_ms=120,
                max_borrow_after_ms=1200,
                severe_overflow_ms=2000,
                prepare_text=True,
                voiceover_topic="catholic",
                max_chars_per_second=13,
                prepared_srt_output=prepared_srt,
                force=True,
            )
            written = prepared_srt.read_text(encoding="utf-8")
            self.assertTrue(prepared_srt.exists())
            self.assertIn("00:00:00,000 --> 00:00:01,000", written)
            self.assertTrue(manifest["text_preparation"]["enabled"])
            self.assertIn("prepared_srt_output", manifest["text_preparation"])
            self.assertNotEqual(mock_synthesize.call_args.args[0], manifest["segments"][0]["original_text"])

    def test_preparation_summary_counts(self) -> None:
        cues = [
            PreparedVoiceoverCue(
                index=1,
                start_ms=0,
                end_ms=2_000,
                original_text="abc",
                prepared_text="abc",
                original_char_count=3,
                prepared_char_count=3,
                target_char_count=20,
                reduction_ratio=0.0,
                status="ok",
                warnings=[],
            ),
            PreparedVoiceoverCue(
                index=2,
                start_ms=2_000,
                end_ms=4_000,
                original_text="abcdef",
                prepared_text="abc",
                original_char_count=6,
                prepared_char_count=3,
                target_char_count=20,
                reduction_ratio=0.5,
                status="shortened",
                warnings=[],
            ),
        ]
        summary = summarize_prepared_cues(cues)
        self.assertEqual(summary["text_ok_count"], 1)
        self.assertEqual(summary["text_shortened_count"], 1)
        self.assertEqual(summary["text_too_long_count"], 0)
        self.assertEqual(summary["total_original_chars"], 9)
        self.assertEqual(summary["total_prepared_chars"], 6)


if __name__ == "__main__":
    unittest.main()
