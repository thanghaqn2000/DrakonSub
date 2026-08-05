import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.srt_audio.job_service import (  # noqa: E402
    SrtAudioJobError,
    create_job_from_srt_bytes,
    run_synthesize_job,
)


class SrtAudioJobServiceTests(unittest.TestCase):
    def test_create_job_writes_input_and_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt = b"1\n00:00:00,000 --> 00:00:01,000\nXin chao\n"
            job_id, job_dir = create_job_from_srt_bytes(root, srt)
            self.assertTrue((job_dir / "input.srt").is_file())
            meta = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "ready")
            self.assertEqual(meta["cue_count"], 1)
            self.assertEqual(job_id, job_dir.name)

    def test_synthesize_allows_too_long_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt = ("1\n00:00:00,000 --> 00:00:01,000\n" + ("dai " * 40) + "\n").encode(
                "utf-8"
            )
            _, job_dir = create_job_from_srt_bytes(root, srt)
            with patch("auto_subtitle.srt_audio.job_service.convert_wav_to_mp3"):
                with patch("auto_subtitle.srt_audio.job_service.build_srt_audio_track") as mock_build:
                    with patch(
                        "auto_subtitle.srt_audio.job_service.probe_audio_duration_ms",
                        return_value=800,
                    ):
                        with patch("auto_subtitle.srt_audio.job_service.synthesize_to_file") as mock_tts:
                            with patch(
                                "auto_subtitle.srt_audio.job_service.load_saydi_config",
                                return_value=type(
                                    "Cfg",
                                    (),
                                    {
                                        "token": "x",
                                        "sample": "s",
                                        "speed": 1.0,
                                        "lang": "vi",
                                        "output_format": "wav",
                                    },
                                )(),
                            ):
                                mock_build.side_effect = (
                                    lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"RIFF")
                                )
                                result = run_synthesize_job(
                                    job_dir,
                                    saydi_sample=None,
                                    saydi_speed=1.0,
                                    output_format="wav",
                                    chars_per_second=13.0,
                                )
                                mock_tts.assert_called_once()
                                self.assertTrue(Path(result["output_wav"]).is_file())

    @patch("auto_subtitle.srt_audio.job_service.convert_wav_to_mp3")
    @patch("auto_subtitle.srt_audio.job_service.build_srt_audio_track")
    @patch("auto_subtitle.srt_audio.job_service.probe_audio_duration_ms", return_value=500)
    @patch("auto_subtitle.srt_audio.job_service.synthesize_to_file")
    @patch(
        "auto_subtitle.srt_audio.job_service.load_saydi_config",
        return_value=type("C", (), {"token": "t", "sample": "s", "speed": 1.0})(),
    )
    def test_synthesize_allows_overlap_and_cascades(
        self, _cfg, _tts, _probe, _mock_build, _mock_mp3
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt = (
                "1\n00:00:00,000 --> 00:00:01,500\nok\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\nok\n"
            ).encode("utf-8")
            _, job_dir = create_job_from_srt_bytes(root, srt)
            result = run_synthesize_job(
                job_dir,
                saydi_sample=None,
                saydi_speed=1.0,
                output_format="wav",
                chars_per_second=13.0,
            )
            self.assertTrue(result["output_wav"])
            _tts.assert_called()

    @patch("auto_subtitle.srt_audio.job_service.convert_wav_to_mp3")
    @patch("auto_subtitle.srt_audio.job_service.build_srt_audio_track")
    @patch("auto_subtitle.srt_audio.job_service.probe_audio_duration_ms", return_value=800)
    @patch("auto_subtitle.srt_audio.job_service.synthesize_to_file")
    @patch(
        "auto_subtitle.srt_audio.job_service.load_saydi_config",
        return_value=type(
            "Cfg",
            (),
            {
                "token": "x",
                "sample": "s",
                "speed": 1.0,
                "lang": "vi",
                "output_format": "wav",
            },
        )(),
    )
    def test_synthesize_builds_wav(
        self, _cfg, _tts, _probe, mock_build, mock_mp3
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt = b"1\n00:00:00,000 --> 00:00:02,000\nXin chao\n"
            _, job_dir = create_job_from_srt_bytes(root, srt)

            def _fake_build(
                *, segment_starts_ms, segment_paths, track_duration_ms, output_path
            ):
                Path(output_path).write_bytes(b"RIFF")

            mock_build.side_effect = _fake_build
            result = run_synthesize_job(
                job_dir,
                saydi_sample=None,
                saydi_speed=1.0,
                output_format="wav",
                chars_per_second=13.0,
            )
            self.assertTrue(Path(result["output_wav"]).is_file())
            mock_mp3.assert_not_called()

    def test_synthesize_cascades_and_rewrites_edited_srt(self) -> None:
        sample = (
            "1\n00:00:00,000 --> 00:00:02,000\nOne.\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\nTwo.\n\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, job_dir = create_job_from_srt_bytes(root, sample.encode("utf-8"))
            from auto_subtitle.srt_audio.cue_service import edited_srt_path
            from auto_subtitle.subtitle_edit_service import load_srt

            durations = {1: 3500, 2: 1000}
            starts_captured: dict = {}

            def fake_tts(text, path, config=None):
                path.write_bytes(b"RIFF....")

            def fake_probe(path):
                return durations[int(path.stem)]

            def fake_build(*, segment_starts_ms, segment_paths, track_duration_ms, output_path):
                starts_captured["starts"] = list(segment_starts_ms)
                starts_captured["track"] = track_duration_ms
                Path(output_path).write_bytes(b"wav")

            with patch(
                "auto_subtitle.srt_audio.job_service.load_saydi_config",
                return_value=type("C", (), {"token": "t", "sample": "s", "speed": 1.0})(),
            ):
                with patch(
                    "auto_subtitle.srt_audio.job_service.synthesize_to_file",
                    side_effect=fake_tts,
                ):
                    with patch(
                        "auto_subtitle.srt_audio.job_service.probe_audio_duration_ms",
                        side_effect=fake_probe,
                    ):
                        with patch(
                            "auto_subtitle.srt_audio.job_service.build_srt_audio_track",
                            side_effect=fake_build,
                        ):
                            run_synthesize_job(
                                job_dir,
                                saydi_sample="s",
                                saydi_speed=1.0,
                                output_format="wav",
                                cue_gap_ms=280,
                            )

            self.assertEqual(starts_captured["starts"], [0, 3780])
            cues = load_srt(edited_srt_path(job_dir))
            self.assertEqual(cues[0].start, "00:00:00,000")
            self.assertEqual(cues[0].end, "00:00:03,500")
            self.assertEqual(cues[1].start, "00:00:03,780")
            self.assertEqual(cues[1].end, "00:00:04,780")
            manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["cue_gap_ms"], 280)
            self.assertEqual(manifest["segments"][1]["shift_ms"], 1780)
            self.assertEqual(manifest["saydi_speed"], 1.0)


if __name__ == "__main__":
    unittest.main()
