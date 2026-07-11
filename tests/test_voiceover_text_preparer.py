import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.voiceover.audio_builder import build_voiceover_track  # noqa: E402
from auto_subtitle.voiceover.job_service import (  # noqa: E402
    VoiceoverJobOptions,
    run_voiceover_job,
)
from auto_subtitle.voiceover.srt_parser import VoiceoverCue, parse_voiceover_srt  # noqa: E402
from auto_subtitle.voiceover.text_preparer import (  # noqa: E402
    PreparedVoiceoverCue,
    prepare_voiceover_cues,
    resolve_prepare_text_mode,
    summarize_prepared_cues,
    write_prepared_srt,
)


def _ms_to_srt_timestamp(value: int) -> str:
    hours, rem = divmod(value, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _saydi_cfg() -> object:
    return type(
        "Cfg",
        (),
        {
            "token": "x",
            "sample": "sample-id",
            "speed": 1.0,
            "lang": "vi",
            "output_format": "wav",
        },
    )()


BUG_CUES = [
    (6, "Và nếu tôi nói điều đó với bạn, bạn cũng sẽ không tin."),
    (7, "Tôi không thể khiến những người trẻ hiểu cuộc đời ngắn ngủi, trôi qua nhanh đến thế nào."),
    (
        12,
        "Thật thú vị là với tất cả khoa học y tế của chúng ta, chúng ta chưa bao giờ vượt qua con số kỳ diệu đó.",
    ),
    (
        19,
        "Bây giờ là thời điểm thích hợp, những việc chúng ta nên làm, những lớp học nên tham gia, những cuốn sách nên đọc.",
    ),
    (24, "Tiền bạn nên cho đi, hãy cho đi ngay bây giờ."),
    (25, "Thời gian học tập, hãy làm ngay bây giờ."),
    (26, "Những người bạn nên làm chứng, hãy làm ngay bây giờ."),
    (
        34,
        "Và mỗi ngày, ngân hàng mang tên Thời Gian lại mở một tài khoản mới cho bạn và tôi.",
    ),
]


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

    def test_safe_mode_preserves_long_cue_and_marks_too_long(self) -> None:
        cue = VoiceoverCue(
            index=1,
            start_ms=0,
            end_ms=2_000,
            text=(
                "Trong bài giảng này, chúng ta được mời gọi suy ngẫm thật dài dòng về đức tin, "
                "về đời sống cầu nguyện, và về cách đáp lại tiếng Chúa trong từng hoàn cảnh."
            ),
        )
        prepared = prepare_voiceover_cues(
            [cue], topic="catholic", max_chars_per_second=10, mode="safe"
        )
        self.assertEqual(prepared[0].prepared_text, prepared[0].original_text)
        self.assertEqual(prepared[0].status, "too_long")
        self.assertEqual(prepared[0].reduction_ratio, 0.0)

    def test_aggressive_mode_can_shorten_long_cue(self) -> None:
        cue = VoiceoverCue(
            index=1,
            start_ms=0,
            end_ms=2_000,
            text=(
                "Trong bài giảng này, chúng ta được mời gọi suy ngẫm thật dài dòng về đức tin, "
                "về đời sống cầu nguyện, và về cách đáp lại tiếng Chúa trong từng hoàn cảnh."
            ),
        )
        prepared = prepare_voiceover_cues(
            [cue], topic="catholic", max_chars_per_second=10, mode="aggressive"
        )
        self.assertLess(prepared[0].prepared_char_count, prepared[0].original_char_count)
        self.assertIn(prepared[0].status, {"shortened", "too_long"})

    def test_prepare_text_true_resolves_to_safe_not_aggressive(self) -> None:
        self.assertEqual(resolve_prepare_text_mode(prepare_text=True), "safe")
        self.assertEqual(resolve_prepare_text_mode(prepare_text=False), "disabled")

    def test_bug_cues_preserved_in_safe_mode(self) -> None:
        cues = [
            VoiceoverCue(index=index, start_ms=0, end_ms=1_000, text=text)
            for index, text in BUG_CUES
        ]
        prepared = prepare_voiceover_cues(
            cues, topic="catholic", max_chars_per_second=13, mode="safe"
        )
        summary = summarize_prepared_cues(prepared)
        self.assertEqual(summary["text_shortened_count"], 0)
        for item, (_, expected) in zip(prepared, BUG_CUES):
            self.assertEqual(item.prepared_text, expected)
            self.assertEqual(item.prepared_text, item.original_text)

    def test_money_cue_not_truncated_for_saydi(self) -> None:
        text = "Tiền bạn nên cho đi, hãy cho đi ngay bây giờ."
        cue = VoiceoverCue(index=24, start_ms=0, end_ms=1_000, text=text)
        prepared = prepare_voiceover_cues(
            [cue], topic="catholic", max_chars_per_second=13, mode="safe"
        )
        self.assertEqual(prepared[0].prepared_text, text)
        self.assertNotEqual(prepared[0].prepared_text, "Tiền bạn nên cho đi")

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

    @patch("auto_subtitle.voiceover.job_service.mux_video_with_audio")
    @patch("auto_subtitle.voiceover.job_service.mix_audio_tracks")
    @patch("auto_subtitle.voiceover.job_service.build_voiceover_track")
    @patch("auto_subtitle.voiceover.job_service.probe_audio_duration_ms", side_effect=[1000])
    @patch("auto_subtitle.voiceover.job_service.synthesize_to_file")
    @patch("auto_subtitle.voiceover.job_service.load_saydi_config", return_value=_saydi_cfg())
    @patch("auto_subtitle.voiceover.job_service.video_has_audio_stream", return_value=True)
    @patch("auto_subtitle.voiceover.job_service.probe_video_duration_ms", return_value=20_000)
    def test_service_without_prepare_text_preserves_phase2_behavior(
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
            result = run_voiceover_job(
                VoiceoverJobOptions(
                    input_video=Path("sample-video.mp4"),
                    voiceover_srt=srt_path,
                    output_video=Path(tmpdir) / "out.mp4",
                    workdir=Path(tmpdir) / "job",
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
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["text_preparation"]["enabled"])
        self.assertEqual(manifest["text_preparation"]["prepare_text_mode"], "disabled")
        self.assertEqual(manifest["segments"][0]["original_text"], cue.text)
        self.assertEqual(manifest["segments"][0]["prepared_text"], cue.text)
        self.assertEqual(mock_synthesize.call_args.args[0], cue.text)

    @patch("auto_subtitle.voiceover.job_service.mux_video_with_audio")
    @patch("auto_subtitle.voiceover.job_service.mix_audio_tracks")
    @patch("auto_subtitle.voiceover.job_service.build_voiceover_track")
    @patch("auto_subtitle.voiceover.job_service.probe_audio_duration_ms", side_effect=[1000])
    @patch("auto_subtitle.voiceover.job_service.synthesize_to_file")
    @patch("auto_subtitle.voiceover.job_service.load_saydi_config", return_value=_saydi_cfg())
    @patch("auto_subtitle.voiceover.job_service.video_has_audio_stream", return_value=True)
    @patch("auto_subtitle.voiceover.job_service.probe_video_duration_ms", return_value=20_000)
    def test_service_with_prepare_text_preserves_full_text_for_saydi(
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
        full_text = (
            "Chúng ta đang có một câu thật dài, thật nhiều ý và thật khó đọc nếu đem đọc nguyên văn."
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "stress.srt"
            srt_path.write_text(
                f"1\n00:00:00,000 --> 00:00:01,000\n{full_text}\n",
                encoding="utf-8",
            )
            prepared_srt = Path(tmpdir) / "prepared_voiceover.srt"
            result = run_voiceover_job(
                VoiceoverJobOptions(
                    input_video=Path("sample-video.mp4"),
                    voiceover_srt=srt_path,
                    output_video=Path(tmpdir) / "out.mp4",
                    workdir=Path(tmpdir) / "job",
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
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            written = prepared_srt.read_text(encoding="utf-8")
            self.assertTrue(prepared_srt.exists())
            self.assertIn("00:00:00,000 --> 00:00:01,000", written)
            self.assertIn(full_text, written)
            self.assertTrue(manifest["text_preparation"]["enabled"])
            self.assertEqual(manifest["text_preparation"]["prepare_text_mode"], "safe")
            self.assertEqual(manifest["text_preparation"]["text_shortened_count"], 0)
            self.assertGreaterEqual(manifest["text_preparation"]["text_too_long_count"], 1)
            self.assertEqual(mock_synthesize.call_args.args[0], full_text)
            self.assertEqual(manifest["summary"]["text_shortened_count"], 0)

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
        self.assertEqual(summary["text_changed_count"], 1)
        self.assertEqual(summary["total_original_chars"], 9)
        self.assertEqual(summary["total_prepared_chars"], 6)

    def test_build_voiceover_track_does_not_atruncate(self) -> None:
        source = inspect.getsource(build_voiceover_track)
        self.assertNotIn("atruncate", source)
        self.assertIn("adelay", source)


if __name__ == "__main__":
    unittest.main()
