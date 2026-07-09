import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.voiceover.job_service import (  # noqa: E402
    VoiceoverJobError,
    VoiceoverJobOptions,
    VoiceoverJobResult,
    run_voiceover_job,
)

_CLI_PATH = ROOT / "scripts" / "prototype_voiceover_from_srt.py"
_CLI_SPEC = importlib.util.spec_from_file_location("prototype_voiceover_from_srt", _CLI_PATH)
assert _CLI_SPEC and _CLI_SPEC.loader
prototype_mod = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(prototype_mod)


class VoiceoverJobServiceTests(unittest.TestCase):
    def test_missing_input_video_gives_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "input.srt"
            srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
            options = VoiceoverJobOptions(
                input_video=Path(tmpdir) / "missing.mp4",
                voiceover_srt=srt_path,
                output_video=Path(tmpdir) / "out.mp4",
                workdir=Path(tmpdir) / "job",
            )
            with self.assertRaises(VoiceoverJobError) as ctx:
                run_voiceover_job(options)
        self.assertIn("Input video not found", str(ctx.exception))

    def test_missing_srt_gives_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            options = VoiceoverJobOptions(
                input_video=input_video,
                voiceover_srt=Path(tmpdir) / "missing.srt",
                output_video=Path(tmpdir) / "out.mp4",
                workdir=Path(tmpdir) / "job",
            )
            with self.assertRaises(VoiceoverJobError) as ctx:
                run_voiceover_job(options)
        self.assertIn("Voiceover SRT not found", str(ctx.exception))

    def test_output_exists_without_force_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            srt_path = Path(tmpdir) / "input.srt"
            srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
            output_video = Path(tmpdir) / "out.mp4"
            output_video.write_bytes(b"exists")
            options = VoiceoverJobOptions(
                input_video=input_video,
                voiceover_srt=srt_path,
                output_video=output_video,
                workdir=Path(tmpdir) / "job",
                force=False,
            )
            with self.assertRaises(VoiceoverJobError) as ctx:
                run_voiceover_job(options)
        self.assertIn("Output already exists", str(ctx.exception))

    @patch("auto_subtitle.voiceover.job_service.mux_video_with_audio")
    @patch("auto_subtitle.voiceover.job_service.mix_audio_tracks")
    @patch("auto_subtitle.voiceover.job_service.build_voiceover_track")
    @patch("auto_subtitle.voiceover.job_service.probe_audio_duration_ms", side_effect=[1000])
    @patch("auto_subtitle.voiceover.job_service.synthesize_to_file")
    @patch("auto_subtitle.voiceover.job_service.load_saydi_config", return_value=type("Cfg", (), {"token": "x", "sample": "default-sample", "lang": "vi", "output_format": "wav"})())
    @patch("auto_subtitle.voiceover.job_service.video_has_audio_stream", return_value=True)
    @patch("auto_subtitle.voiceover.job_service.probe_video_duration_ms", return_value=20_000)
    def test_force_true_allows_overwrite(
        self,
        _mock_video_duration,
        _mock_has_audio,
        _mock_load_cfg,
        _mock_synthesize,
        _mock_probe_audio,
        _mock_build_track,
        _mock_mix_audio,
        _mock_mux,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            srt_path = Path(tmpdir) / "input.srt"
            srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
            output_video = Path(tmpdir) / "out.mp4"
            output_video.write_bytes(b"exists")
            options = VoiceoverJobOptions(
                input_video=input_video,
                voiceover_srt=srt_path,
                output_video=output_video,
                workdir=Path(tmpdir) / "job",
                force=True,
            )
            result = run_voiceover_job(options)
        self.assertIsInstance(result, VoiceoverJobResult)

    @patch("auto_subtitle.voiceover.job_service.mux_video_with_audio")
    @patch("auto_subtitle.voiceover.job_service.mix_audio_tracks")
    @patch("auto_subtitle.voiceover.job_service.build_voiceover_track")
    @patch("auto_subtitle.voiceover.job_service.probe_audio_duration_ms", side_effect=[1000])
    @patch("auto_subtitle.voiceover.job_service.synthesize_to_file")
    @patch("auto_subtitle.voiceover.job_service.load_saydi_config", return_value=type("Cfg", (), {"token": "x", "sample": "default-sample", "lang": "vi", "output_format": "wav"})())
    @patch("auto_subtitle.voiceover.job_service.video_has_audio_stream", return_value=True)
    @patch("auto_subtitle.voiceover.job_service.probe_video_duration_ms", return_value=20_000)
    def test_prepare_text_writes_prepared_srt(
        self,
        _mock_video_duration,
        _mock_has_audio,
        _mock_load_cfg,
        _mock_synthesize,
        _mock_probe_audio,
        _mock_build_track,
        _mock_mix_audio,
        _mock_mux,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            srt_path = Path(tmpdir) / "input.srt"
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nChúng ta đang có một câu thật dài, thật nhiều ý.\n",
                encoding="utf-8",
            )
            prepared_srt = Path(tmpdir) / "job" / "prepared_voiceover.srt"
            options = VoiceoverJobOptions(
                input_video=input_video,
                voiceover_srt=srt_path,
                output_video=Path(tmpdir) / "out.mp4",
                workdir=Path(tmpdir) / "job",
                prepare_text=True,
                prepared_srt_output=prepared_srt,
                force=True,
            )
            result = run_voiceover_job(options)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(prepared_srt.exists())
            self.assertEqual(result.prepared_srt_path, prepared_srt)
            self.assertTrue(manifest["options"]["prepare_text"])

    @patch("auto_subtitle.voiceover.job_service.mux_video_with_audio")
    @patch("auto_subtitle.voiceover.job_service.mix_audio_tracks")
    @patch("auto_subtitle.voiceover.job_service.build_voiceover_track")
    @patch("auto_subtitle.voiceover.job_service.probe_audio_duration_ms", side_effect=[1000])
    @patch("auto_subtitle.voiceover.job_service.synthesize_to_file")
    @patch("auto_subtitle.voiceover.job_service.load_saydi_config", return_value=type("Cfg", (), {"token": "x", "sample": "default-sample", "lang": "vi", "output_format": "wav"})())
    @patch("auto_subtitle.voiceover.job_service.video_has_audio_stream", return_value=True)
    @patch("auto_subtitle.voiceover.job_service.probe_video_duration_ms", return_value=20_000)
    def test_service_returns_result_and_manifest_contains_metadata(
        self,
        _mock_video_duration,
        _mock_has_audio,
        _mock_load_cfg,
        _mock_synthesize,
        _mock_probe_audio,
        _mock_build_track,
        _mock_mix_audio,
        _mock_mux,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            srt_path = Path(tmpdir) / "input.srt"
            srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
            options = VoiceoverJobOptions(
                input_video=input_video,
                voiceover_srt=srt_path,
                output_video=Path(tmpdir) / "out.mp4",
                workdir=Path(tmpdir) / "job",
                force=True,
            )
            result = run_voiceover_job(options)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertIsInstance(result, VoiceoverJobResult)
        self.assertEqual(manifest["job_type"], "voiceover")
        self.assertEqual(manifest["version"], 1)
        self.assertIn("options", manifest)

    @patch("auto_subtitle.voiceover.job_service.mux_video_with_audio")
    @patch("auto_subtitle.voiceover.job_service.mix_audio_tracks")
    @patch("auto_subtitle.voiceover.job_service.build_voiceover_track")
    @patch("auto_subtitle.voiceover.job_service.probe_audio_duration_ms", side_effect=[1000])
    @patch("auto_subtitle.voiceover.job_service.synthesize_to_file")
    @patch("auto_subtitle.voiceover.job_service.load_saydi_config")
    @patch("auto_subtitle.voiceover.job_service.video_has_audio_stream", return_value=True)
    @patch("auto_subtitle.voiceover.job_service.probe_video_duration_ms", return_value=20_000)
    def test_manifest_records_selected_saydi_sample(
        self,
        _mock_video_duration,
        _mock_has_audio,
        mock_load_cfg,
        _mock_synthesize,
        _mock_probe_audio,
        _mock_build_track,
        _mock_mix_audio,
        _mock_mux,
    ) -> None:
        mock_load_cfg.return_value = type(
            "Cfg",
            (),
            {"token": "x", "sample": "custom-sample-123", "lang": "vi", "output_format": "wav"},
        )()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = Path(tmpdir) / "input.mp4"
            input_video.write_bytes(b"fake")
            srt_path = Path(tmpdir) / "input.srt"
            srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
            options = VoiceoverJobOptions(
                input_video=input_video,
                voiceover_srt=srt_path,
                output_video=Path(tmpdir) / "out.mp4",
                workdir=Path(tmpdir) / "job",
                saydi_sample="custom-sample-123",
                force=True,
            )
            result = run_voiceover_job(options)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        mock_load_cfg.assert_called_once_with(sample_override="custom-sample-123")
        self.assertEqual(manifest["saydi_sample"], "custom-sample-123")
        self.assertEqual(manifest["tts_provider"], "saydi")
        self.assertNotIn("SAYDI_TTS_API_TOKEN", json.dumps(manifest))

    @patch.object(prototype_mod, "run_voiceover_job")
    def test_cli_delegates_to_service(self, mock_run_job) -> None:
        mock_run_job.return_value = VoiceoverJobResult(
            output_video=Path("/tmp/out.mp4"),
            manifest_path=Path("/tmp/manifest.json"),
            prepared_srt_path=None,
            cue_count=1,
            segment_count=1,
            summary={"cue_count": 1},
            warnings=[],
        )
        argv = [
            "prog",
            "--input-video",
            "in.mp4",
            "--voiceover-srt",
            "in.srt",
            "--output-video",
            "out.mp4",
            "--force",
        ]
        with patch.object(sys, "argv", argv):
            exit_code = prototype_mod.main()
        self.assertEqual(exit_code, 0)
        self.assertTrue(mock_run_job.called)


if __name__ == "__main__":
    unittest.main()
