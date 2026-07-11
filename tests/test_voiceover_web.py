import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from auto_subtitle import web
from auto_subtitle.voiceover.job_service import VoiceoverJobError, VoiceoverJobOptions, VoiceoverJobResult


class VoiceoverWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.jobs_root = Path(self.tmp) / "jobs"
        self.voiceover_root = Path(self.tmp) / "voiceover_jobs"
        self.jobs_root_patcher = patch.object(web, "JOBS_ROOT", self.jobs_root)
        self.voiceover_root_patcher = patch.object(web, "VOICEOVER_JOBS_ROOT", self.voiceover_root)
        self.jobs_root_patcher.start()
        self.voiceover_root_patcher.start()
        self.client = TestClient(web.app)

    def tearDown(self) -> None:
        self.jobs_root_patcher.stop()
        self.voiceover_root_patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _post_voiceover_job(self) -> dict:
        res = self.client.post(
            "/api/voiceover/jobs",
            files={
                "input_video": ("clip.mp4", b"fake-video", "video/mp4"),
                "voiceover_srt": (
                    "voice.srt",
                    b"1\n00:00:00,000 --> 00:00:01,000\nXin chao\n",
                    "application/x-subrip",
                ),
            },
            data={"prepare_text": "true", "voiceover_topic": "catholic"},
        )
        self.assertEqual(res.status_code, 200)
        return res.json()

    @patch("auto_subtitle.web.threading.Thread")
    def test_post_voiceover_job_returns_processing_quickly(self, mock_thread) -> None:
        mock_thread.return_value.start = MagicMock()
        data = self._post_voiceover_job()
        self.assertEqual(data["status"], "processing")
        self.assertIn("status_url", data)
        self.assertEqual(data["message"], "Đã bắt đầu tạo video thuyết minh.")
        mock_thread.assert_called_once()

        status_res = self.client.get(data["status_url"])
        self.assertEqual(status_res.status_code, 200)
        self.assertEqual(status_res.json()["status"], "processing")
        self.assertEqual(status_res.json()["stage"], "queued")

    @patch.object(web, "run_voiceover_job")
    def test_background_success_writes_completed_job_json(self, mock_run_voiceover_job) -> None:
        job_id = "bg-success"
        job_dir = self.voiceover_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / "input.mp4"
        srt_path = job_dir / "voiceover.srt"
        output_path = job_dir / "output_voiceover.mp4"
        manifest_path = job_dir / "manifest.json"
        input_path.write_bytes(b"video")
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
        web._write_voiceover_job_json(
            job_id,
            {
                "job_id": job_id,
                "status": "processing",
                "stage": "queued",
                "progress_percent": 0,
                "output_video": str(output_path),
                "manifest": str(manifest_path),
            },
        )

        def _complete(options, *, progress_callback=None):
            if progress_callback:
                progress_callback("generating_voice", 35)
            manifest_path.write_text(json.dumps({"summary": {"cue_count": 1}}), encoding="utf-8")
            output_path.write_bytes(b"video")
            return VoiceoverJobResult(
                output_video=output_path,
                manifest_path=manifest_path,
                prepared_srt_path=None,
                cue_count=1,
                segment_count=1,
                summary={"cue_count": 1, "severe_overflow_count": 0, "text_shortened_count": 0, "text_too_long_count": 0},
                warnings=[],
            )

        mock_run_voiceover_job.side_effect = _complete
        options = VoiceoverJobOptions(
            input_video=input_path,
            voiceover_srt=srt_path,
            output_video=output_path,
            workdir=job_dir,
            force=True,
        )
        web._run_voiceover_job_background(job_id, options)

        status_res = self.client.get(f"/api/voiceover/jobs/{job_id}")
        status = status_res.json()
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["progress_percent"], 100)
        self.assertTrue(status["output_ready"])
        self.assertTrue(status["manifest_ready"])

    @patch.object(web, "run_voiceover_job", side_effect=VoiceoverJobError("SAYDI failed badly"))
    def test_background_failure_writes_failed_job_json(self, _mock_run) -> None:
        job_id = "bg-failed"
        job_dir = self.voiceover_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / "input.mp4"
        srt_path = job_dir / "voiceover.srt"
        output_path = job_dir / "output_voiceover.mp4"
        manifest_path = job_dir / "manifest.json"
        input_path.write_bytes(b"video")
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
        web._write_voiceover_job_json(
            job_id,
            {
                "job_id": job_id,
                "status": "processing",
                "stage": "queued",
                "progress_percent": 0,
                "output_video": str(output_path),
                "manifest": str(manifest_path),
            },
        )
        options = VoiceoverJobOptions(
            input_video=input_path,
            voiceover_srt=srt_path,
            output_video=output_path,
            workdir=job_dir,
            force=True,
        )
        web._run_voiceover_job_background(job_id, options)

        status_res = self.client.get(f"/api/voiceover/jobs/{job_id}")
        status = status_res.json()
        self.assertEqual(status["status"], "failed")
        self.assertIn("SAYDI failed badly", status["error"])

    def test_get_output_before_ready_returns_409(self) -> None:
        job_dir = self.voiceover_root / "job-processing"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "job_id": "job-processing",
                    "status": "processing",
                    "stage": "generating_voice",
                    "progress_percent": 35,
                    "output_video": str(job_dir / "output_voiceover.mp4"),
                    "manifest": str(job_dir / "manifest.json"),
                }
            ),
            encoding="utf-8",
        )
        res = self.client.get("/api/voiceover/jobs/job-processing/output-video")
        self.assertEqual(res.status_code, 409)

    def test_get_manifest_before_ready_returns_409(self) -> None:
        job_dir = self.voiceover_root / "job-processing-manifest"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "job_id": "job-processing-manifest",
                    "status": "processing",
                    "stage": "mixing_audio",
                    "progress_percent": 80,
                    "output_video": str(job_dir / "output_voiceover.mp4"),
                    "manifest": str(job_dir / "manifest.json"),
                }
            ),
            encoding="utf-8",
        )
        res = self.client.get("/api/voiceover/jobs/job-processing-manifest/manifest")
        self.assertEqual(res.status_code, 409)

    def test_get_voiceover_job_status_completed(self) -> None:
        job_dir = self.voiceover_root / "job-status"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "output_voiceover.mp4").write_bytes(b"video")
        (job_dir / "manifest.json").write_text(json.dumps({"summary": {"cue_count": 4}}), encoding="utf-8")
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "job_id": "job-status",
                    "status": "completed",
                    "stage": "completed",
                    "progress_percent": 100,
                    "created_at": "now",
                    "updated_at": "now",
                    "error": None,
                    "input_video": "input.mp4",
                    "voiceover_srt": "voiceover.srt",
                    "output_video": str(job_dir / "output_voiceover.mp4"),
                    "manifest": str(job_dir / "manifest.json"),
                    "summary": {"cue_count": 4},
                }
            ),
            encoding="utf-8",
        )
        res = self.client.get("/api/voiceover/jobs/job-status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["progress_percent"], 100)
        self.assertTrue(data["output_ready"])

    def test_get_voiceover_manifest_returns_json(self) -> None:
        job_dir = self.voiceover_root / "job-manifest"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "manifest.json").write_text(json.dumps({"job_type": "voiceover"}), encoding="utf-8")
        (job_dir / "output_voiceover.mp4").write_bytes(b"video")
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "job_id": "job-manifest",
                    "status": "completed",
                    "stage": "completed",
                    "progress_percent": 100,
                    "output_video": str(job_dir / "output_voiceover.mp4"),
                    "manifest": str(job_dir / "manifest.json"),
                }
            ),
            encoding="utf-8",
        )
        res = self.client.get("/api/voiceover/jobs/job-manifest/manifest")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["job_type"], "voiceover")

    def test_get_voiceover_output_video_returns_file(self) -> None:
        job_dir = self.voiceover_root / "job-video"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "output_voiceover.mp4").write_bytes(b"fake-video")
        (job_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "job_id": "job-video",
                    "status": "completed",
                    "stage": "completed",
                    "progress_percent": 100,
                    "output_video": str(job_dir / "output_voiceover.mp4"),
                    "manifest": str(job_dir / "manifest.json"),
                }
            ),
            encoding="utf-8",
        )
        res = self.client.get("/api/voiceover/jobs/job-video/output-video")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content, b"fake-video")

    def test_completed_fallback_without_job_json(self) -> None:
        job_dir = self.voiceover_root / "job-fallback"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "manifest.json").write_text(json.dumps({"job_type": "voiceover"}), encoding="utf-8")
        (job_dir / "output_voiceover.mp4").write_bytes(b"fake-video")
        res = self.client.get("/api/voiceover/jobs/job-fallback")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "completed")
        self.assertTrue(res.json()["output_ready"])

    def test_missing_voiceover_job_returns_404(self) -> None:
        res = self.client.get("/api/voiceover/jobs/not-found")
        self.assertEqual(res.status_code, 404)

    def test_invalid_upload_returns_friendly_400(self) -> None:
        res = self.client.post(
            "/api/voiceover/jobs",
            files={
                "input_video": ("clip.txt", b"bad", "text/plain"),
                "voiceover_srt": (
                    "voice.srt",
                    b"1\n00:00:00,000 --> 00:00:01,000\nXin chao\n",
                    "application/x-subrip",
                ),
            },
        )
        self.assertEqual(res.status_code, 400)

    def _post_voiceover_from_video(self) -> dict:
        res = self.client.post(
            "/api/voiceover/jobs/from-video",
            files={"input_video": ("clip.mp4", b"fake-video", "video/mp4")},
            data={"prepare_text": "true", "voiceover_topic": "catholic"},
        )
        self.assertEqual(res.status_code, 200)
        return res.json()

    @patch("auto_subtitle.web.threading.Thread")
    def test_post_from_video_returns_processing_quickly(self, mock_thread) -> None:
        mock_thread.return_value.start = MagicMock()
        data = self._post_voiceover_from_video()
        self.assertEqual(data["status"], "processing")
        self.assertIn("status_url", data)
        mock_thread.assert_called_once()

        job_id = data["job_id"]
        job_dir = self.voiceover_root / job_id
        self.assertTrue((job_dir / "input.mp4").is_file())

    @patch.object(web, "run_voiceover_job")
    @patch("auto_subtitle.voiceover.from_video.translate_srt_file")
    @patch("auto_subtitle.voiceover.from_video.transcribe_to_srt")
    @patch("auto_subtitle.voiceover.from_video.extract_audio")
    def test_from_video_background_happy_path(
        self,
        mock_extract,
        mock_transcribe,
        mock_translate,
        mock_run_voiceover,
    ) -> None:
        job_id = "from-video-success"
        job_dir = self.voiceover_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / "input.mp4"
        source_srt = job_dir / "source.srt"
        voiceover_srt = job_dir / "voiceover.srt"
        output_path = job_dir / "output_voiceover.mp4"
        manifest_path = job_dir / "manifest.json"
        input_path.write_bytes(b"video")
        source_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        voiceover_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")

        def _fake_transcribe(audio, srt, config, on_progress=None):
            Path(srt).write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8"
            )

        mock_transcribe.side_effect = _fake_transcribe

        def _fake_translate(src, dst, config, on_progress=None, **kwargs):
            Path(dst).write_text(voiceover_srt.read_text(encoding="utf-8"), encoding="utf-8")
            return dst

        mock_translate.side_effect = _fake_translate

        def _complete(options, *, progress_callback=None):
            manifest_path.write_text(json.dumps({"summary": {"cue_count": 1}}), encoding="utf-8")
            output_path.write_bytes(b"video")
            return VoiceoverJobResult(
                output_video=output_path,
                manifest_path=manifest_path,
                prepared_srt_path=None,
                cue_count=1,
                segment_count=1,
                summary={"cue_count": 1, "severe_overflow_count": 0, "text_shortened_count": 0, "text_too_long_count": 0},
                warnings=[],
            )

        mock_run_voiceover.side_effect = _complete

        web._run_voiceover_from_video_background(
            job_id,
            input_path,
            job_dir,
            prepare_text=True,
            voiceover_topic="catholic",
            original_volume=0.3,
            voice_volume=1.0,
            max_chars_per_second=13.0,
            min_gap_ms=120,
            max_borrow_after_ms=1200,
            severe_overflow_ms=2000,
        )

        mock_extract.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_translate.assert_called_once()
        mock_run_voiceover.assert_called_once()
        self.assertEqual(mock_run_voiceover.call_args[0][0].voiceover_srt, voiceover_srt)

        status = self.client.get(f"/api/voiceover/jobs/{job_id}").json()
        self.assertEqual(status["status"], "completed")

    @patch("auto_subtitle.voiceover.from_video.transcribe_to_srt", side_effect=RuntimeError("transcribe boom"))
    @patch("auto_subtitle.voiceover.from_video.extract_audio")
    def test_from_video_transcription_failure(self, _mock_extract, _mock_transcribe) -> None:
        job_id = "from-video-transcribe-fail"
        job_dir = self.voiceover_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / "input.mp4"
        input_path.write_bytes(b"video")

        web._run_voiceover_from_video_background(
            job_id,
            input_path,
            job_dir,
            prepare_text=True,
            voiceover_topic="catholic",
            original_volume=0.3,
            voice_volume=1.0,
            max_chars_per_second=13.0,
            min_gap_ms=120,
            max_borrow_after_ms=1200,
            severe_overflow_ms=2000,
        )

        status = self.client.get(f"/api/voiceover/jobs/{job_id}").json()
        self.assertEqual(status["status"], "failed")
        self.assertIn("nhận diện", status["error"].lower())

    @patch("auto_subtitle.voiceover.from_video.translate_srt_file", side_effect=RuntimeError("translate boom"))
    @patch("auto_subtitle.voiceover.from_video.transcribe_to_srt")
    @patch("auto_subtitle.voiceover.from_video.extract_audio")
    def test_from_video_translation_failure(
        self, _mock_extract, _mock_transcribe, _mock_translate
    ) -> None:
        job_id = "from-video-translate-fail"
        job_dir = self.voiceover_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / "input.mp4"
        input_path.write_bytes(b"video")

        web._run_voiceover_from_video_background(
            job_id,
            input_path,
            job_dir,
            prepare_text=True,
            voiceover_topic="catholic",
            original_volume=0.3,
            voice_volume=1.0,
            max_chars_per_second=13.0,
            min_gap_ms=120,
            max_borrow_after_ms=1200,
            severe_overflow_ms=2000,
        )

        status = self.client.get(f"/api/voiceover/jobs/{job_id}").json()
        self.assertEqual(status["status"], "failed")
        self.assertIn("dịch", status["error"].lower())


class VoiceoverScriptJobWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.voiceover_root = Path(self.tmp) / "voiceover_jobs"
        self.voiceover_root_patcher = patch.object(web, "VOICEOVER_JOBS_ROOT", self.voiceover_root)
        self.voiceover_root_patcher.start()
        self.client = TestClient(web.app)

    def tearDown(self) -> None:
        self.voiceover_root_patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_script_job(self, job_id: str, **extra) -> Path:
        job_dir = self.voiceover_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input.mp4").write_bytes(b"video")
        payload = {
            "job_id": job_id,
            "job_type": "script",
            "status": "script_ready",
            "stage": "script_ready",
            "progress_percent": 50,
            "voiceover_topic": "catholic",
            "max_chars_per_second": 13.0,
        }
        payload.update(extra)
        web._write_voiceover_job_json(job_id, payload)
        return job_dir

    @patch("auto_subtitle.web.threading.Thread")
    def test_post_script_job_returns_quickly(self, mock_thread) -> None:
        mock_thread.return_value.start = MagicMock()
        res = self.client.post(
            "/api/voiceover/script-jobs/from-video",
            files={"input_video": ("clip.mp4", b"fake-video", "video/mp4")},
            data={"voiceover_topic": "catholic"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "processing")
        self.assertIn("/api/voiceover/script-jobs/", data["status_url"])
        mock_thread.assert_called_once()

    @patch("auto_subtitle.web.threading.Thread")
    def test_post_script_job_from_url_auto_detects_youtube(self, mock_thread) -> None:
        mock_thread.return_value.start = MagicMock()
        res = self.client.post(
            "/api/voiceover/script-jobs/from-url",
            json={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "voiceover_topic": "catholic",
                "max_chars_per_second": 13,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["provider"], "youtube")
        self.assertIn("/api/voiceover/script-jobs/", data["status_url"])
        status = self.client.get(data["status_url"]).json()
        self.assertEqual(status["status"], "processing")
        self.assertEqual(status["stage"], "downloading")
        self.assertEqual(status["url_provider"], "youtube")
        mock_thread.assert_called_once()

    @patch("auto_subtitle.web.threading.Thread")
    def test_post_script_job_from_url_auto_detects_facebook(self, mock_thread) -> None:
        mock_thread.return_value.start = MagicMock()
        res = self.client.post(
            "/api/voiceover/script-jobs/from-url",
            json={"url": "https://www.facebook.com/watch/?v=1234567890"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["provider"], "facebook")

    def test_post_script_job_from_url_rejects_unsupported_link(self) -> None:
        res = self.client.post(
            "/api/voiceover/script-jobs/from-url",
            json={"url": "https://example.com/video.mp4"},
        )
        self.assertEqual(res.status_code, 400)

    @patch("auto_subtitle.web.run_script_generation_job")
    def test_from_url_background_downloads_then_generates(self, mock_script_gen) -> None:
        job_id = "from-url-bg"
        job_dir = self._write_script_job(
            job_id,
            status="processing",
            stage="downloading",
            progress_percent=0,
            url_provider="youtube",
        )
        input_path = job_dir / "input.mp4"
        source_srt = job_dir / "source.srt"
        voiceover_srt = job_dir / "voiceover.srt"
        source_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
        voiceover_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")

        def _fake_download(url, output_dir, *, output_filename="input.mp4"):
            path = Path(output_dir) / output_filename
            path.write_bytes(b"downloaded-video")
            return {"path": str(path), "provider": "youtube", "title": "Demo"}

        with patch(
            "auto_subtitle.url_import_service.download_video_from_url",
            side_effect=_fake_download,
        ):
            mock_script_gen.return_value = (source_srt, voiceover_srt)
            web._run_script_generation_from_url_background(
                job_id,
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                job_dir,
                voiceover_topic="catholic",
            )

        mock_script_gen.assert_called_once()
        status = self.client.get(f"/api/voiceover/script-jobs/{job_id}").json()
        self.assertEqual(status["status"], "script_ready")
        self.assertEqual(status["url_provider"], "youtube")
        self.assertEqual(status["source_title"], "Demo")
        self.assertTrue(input_path.is_file())

    @patch("auto_subtitle.web.threading.Thread")
    def test_immediate_status_poll_after_script_job_create(self, mock_thread) -> None:
        mock_thread.return_value.start = MagicMock()
        res = self.client.post(
            "/api/voiceover/script-jobs/from-video",
            files={"input_video": ("clip.mp4", b"fake-video", "video/mp4")},
            data={"voiceover_topic": "catholic"},
        )
        self.assertEqual(res.status_code, 200)
        job_id = res.json()["job_id"]
        status_res = self.client.get(f"/api/voiceover/script-jobs/{job_id}")
        self.assertEqual(status_res.status_code, 200)
        self.assertEqual(status_res.json()["status"], "processing")

    @patch.object(web, "run_voiceover_job")
    @patch("auto_subtitle.web.run_script_generation_job")
    def test_script_generation_does_not_call_tts(
        self, mock_script_gen, mock_run_voiceover
    ) -> None:
        job_id = "script-gen"
        job_dir = self._write_script_job(job_id, status="processing", stage="queued")
        input_path = job_dir / "input.mp4"
        source_srt = job_dir / "source.srt"
        voiceover_srt = job_dir / "voiceover.srt"
        source_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
        voiceover_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
        mock_script_gen.return_value = (source_srt, voiceover_srt)

        web._run_script_generation_background(
            job_id, input_path, job_dir, voiceover_topic="catholic"
        )

        mock_run_voiceover.assert_not_called()
        status = self.client.get(f"/api/voiceover/script-jobs/{job_id}").json()
        self.assertEqual(status["status"], "script_ready")
        self.assertTrue(status["voiceover_srt_ready"])

    def test_get_cues_from_voiceover_srt(self) -> None:
        job_id = "cues-base"
        job_dir = self._write_script_job(job_id)
        (job_dir / "source.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello world\n", encoding="utf-8"
        )
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8"
        )
        res = self.client.get(f"/api/voiceover/script-jobs/{job_id}/cues")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["source"], "voiceover.srt")
        self.assertEqual(len(data["cues"]), 1)
        self.assertEqual(data["cues"][0]["text"], "Xin chao")
        self.assertEqual(data["cues"][0]["source_text"], "Hello world")

    def test_get_cues_prefers_edited_srt(self) -> None:
        job_id = "cues-edited"
        job_dir = self._write_script_job(job_id, edited_srt_ready=True)
        (job_dir / "source.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nOriginal English\n", encoding="utf-8"
        )
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        (job_dir / "edited_voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nDa sua\n", encoding="utf-8"
        )
        data = self.client.get(f"/api/voiceover/script-jobs/{job_id}/cues").json()
        self.assertEqual(data["source"], "edited_voiceover.srt")
        self.assertEqual(data["cues"][0]["text"], "Da sua")
        self.assertEqual(data["cues"][0]["source_text"], "Original English")

    def test_status_includes_srt_download_urls_when_ready(self) -> None:
        job_id = "download-urls"
        job_dir = self._write_script_job(job_id)
        (job_dir / "source.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8"
        )
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8"
        )
        status = self.client.get(f"/api/voiceover/script-jobs/{job_id}").json()
        self.assertEqual(
            status["voiceover_srt_download_url"],
            f"/api/voiceover/script-jobs/{job_id}/download/voiceover-srt",
        )
        self.assertEqual(
            status["source_srt_download_url"],
            f"/api/voiceover/script-jobs/{job_id}/download/source-srt",
        )

    def test_download_voiceover_srt_returns_file(self) -> None:
        job_id = "download-vo"
        job_dir = self._write_script_job(job_id)
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8"
        )
        res = self.client.get(f"/api/voiceover/script-jobs/{job_id}/download/voiceover-srt")
        self.assertEqual(res.status_code, 200)
        self.assertIn("application/x-subrip", res.headers.get("content-type", ""))
        self.assertIn("Xin chao", res.text)

    def test_download_voiceover_srt_prefers_edited(self) -> None:
        job_id = "download-edited"
        job_dir = self._write_script_job(job_id, edited_srt_ready=True)
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        (job_dir / "edited_voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nDa sua\n", encoding="utf-8"
        )
        res = self.client.get(f"/api/voiceover/script-jobs/{job_id}/download/voiceover-srt")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Da sua", res.text)
        self.assertNotIn("Goc", res.text)

    def test_download_source_srt_returns_file(self) -> None:
        job_id = "download-src"
        job_dir = self._write_script_job(job_id)
        (job_dir / "source.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello world\n", encoding="utf-8"
        )
        res = self.client.get(f"/api/voiceover/script-jobs/{job_id}/download/source-srt")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Hello world", res.text)

    def test_download_srt_rejects_when_not_ready(self) -> None:
        job_id = "download-not-ready"
        job_dir = self._write_script_job(job_id, status="processing", stage="generating_script")
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8"
        )
        res = self.client.get(f"/api/voiceover/script-jobs/{job_id}/download/voiceover-srt")
        self.assertEqual(res.status_code, 409)

    def test_put_cues_writes_edited_srt(self) -> None:
        job_id = "save-cues"
        job_dir = self._write_script_job(job_id)
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        res = self.client.put(
            f"/api/voiceover/script-jobs/{job_id}/cues",
            json={
                "cues": [
                    {
                        "index": 1,
                        "start": "00:00:00,000",
                        "end": "00:00:01,000",
                        "text": "Da chinh sua",
                    }
                ]
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue((job_dir / "edited_voiceover.srt").is_file())
        self.assertIn("Da chinh sua", (job_dir / "edited_voiceover.srt").read_text(encoding="utf-8"))

    def test_put_cues_only_persists_vietnamese_text(self) -> None:
        job_id = "save-cues-no-en"
        job_dir = self._write_script_job(job_id)
        (job_dir / "source.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nEnglish only in source\n", encoding="utf-8"
        )
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        res = self.client.put(
            f"/api/voiceover/script-jobs/{job_id}/cues",
            json={
                "cues": [
                    {
                        "index": 1,
                        "start": "00:00:00,000",
                        "end": "00:00:01,000",
                        "text": "Da chinh sua",
                        "source_text": "Should not be saved",
                    }
                ]
            },
        )
        self.assertEqual(res.status_code, 200)
        edited = (job_dir / "edited_voiceover.srt").read_text(encoding="utf-8")
        self.assertIn("Da chinh sua", edited)
        self.assertNotIn("Should not be saved", edited)
        self.assertNotIn("English only in source", edited)

    def test_put_cues_rejects_timing_change(self) -> None:
        job_id = "reject-timing"
        job_dir = self._write_script_job(job_id)
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        res = self.client.put(
            f"/api/voiceover/script-jobs/{job_id}/cues",
            json={
                "cues": [
                    {
                        "index": 1,
                        "start": "00:00:00,500",
                        "end": "00:00:01,000",
                        "text": "Goc",
                    }
                ]
            },
        )
        self.assertEqual(res.status_code, 400)

    @patch("auto_subtitle.voiceover.script_job.run_voiceover_job")
    def test_render_uses_edited_srt(self, mock_run_voiceover) -> None:
        from auto_subtitle.voiceover.script_job import ScriptRenderOptions, render_script_job

        job_id = "render-edited"
        job_dir = self._write_script_job(job_id)
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        edited = job_dir / "edited_voiceover.srt"
        edited.write_text("1\n00:00:00,000 --> 00:00:01,000\nEdited\n", encoding="utf-8")
        output_path = job_dir / "output_voiceover.mp4"
        manifest_path = job_dir / "manifest.json"

        def _complete(options, *, progress_callback=None):
            self.assertEqual(options.voiceover_srt, edited)
            self.assertEqual(options.original_volume, 0.18)
            manifest_path.write_text("{}", encoding="utf-8")
            output_path.write_bytes(b"video")
            return VoiceoverJobResult(
                output_video=output_path,
                manifest_path=manifest_path,
                prepared_srt_path=None,
                cue_count=1,
                segment_count=1,
                summary={"cue_count": 1},
                warnings=[],
            )

        mock_run_voiceover.side_effect = _complete
        render_script_job(job_dir, ScriptRenderOptions(original_volume=0.18, voice_volume=1.0))
        mock_run_voiceover.assert_called_once()

    @patch("auto_subtitle.web.threading.Thread")
    def test_render_endpoint_default_volume(self, mock_thread) -> None:
        mock_thread.return_value.start = MagicMock()
        job_id = "render-api"
        job_dir = self._write_script_job(job_id)
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        res = self.client.post(f"/api/voiceover/script-jobs/{job_id}/render", json={})
        self.assertEqual(res.status_code, 200)
        mock_thread.assert_called_once()
        thread_kwargs = mock_thread.call_args.kwargs["kwargs"]
        self.assertEqual(thread_kwargs["options"].original_volume, 0.18)
        self.assertEqual(thread_kwargs["options"].voice_volume, 1.0)

    @patch("auto_subtitle.web.threading.Thread")
    def test_render_allows_rerender_after_completed(self, mock_thread) -> None:
        mock_thread.return_value.start = MagicMock()
        job_id = "render-rerun"
        job_dir = self._write_script_job(
            job_id,
            status="completed",
            stage="completed",
            progress_percent=100,
        )
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        (job_dir / "output_voiceover.mp4").write_bytes(b"video")
        res = self.client.post(
            f"/api/voiceover/script-jobs/{job_id}/render",
            json={"saydi_speed": 1.4, "max_chars_per_second": 15},
        )
        self.assertEqual(res.status_code, 200)
        thread_kwargs = mock_thread.call_args.kwargs["kwargs"]
        self.assertEqual(thread_kwargs["options"].saydi_speed, 1.4)
        self.assertEqual(thread_kwargs["options"].max_chars_per_second, 15.0)

    def test_render_rejects_while_rendering(self) -> None:
        job_id = "render-busy"
        job_dir = self._write_script_job(job_id, status="rendering", stage="generating_voice")
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        res = self.client.post(f"/api/voiceover/script-jobs/{job_id}/render", json={})
        self.assertEqual(res.status_code, 409)
        self.assertIn("đang render", res.json()["detail"].lower())

    @patch.object(web, "load_saydi_config")
    def test_voiceover_config_endpoint_returns_defaults(self, mock_load_cfg) -> None:
        mock_load_cfg.return_value = type("Cfg", (), {"sample": "config-sample", "speed": 1.1})()
        res = self.client.get("/api/voiceover/config")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["default_original_volume"], 0.18)
        self.assertEqual(data["default_voice_volume"], 1.0)
        self.assertEqual(
            data["default_saydi_sample"],
            "liam-warm-thoughtful-and-determined-7XOKiK112QRZRSLbCfMc",
        )
        self.assertEqual(data["default_saydi_speed"], 1.1)
        self.assertEqual(
            data["saydi_voice_options"][0],
            {
                "label": "Liam",
                "sample": "liam-warm-thoughtful-and-determined-7XOKiK112QRZRSLbCfMc",
            },
        )
        self.assertEqual(len(data["saydi_voice_options"]), 4)
        self.assertNotIn("token", json.dumps(data).lower())

    @patch("auto_subtitle.web.threading.Thread")
    def test_render_endpoint_passes_saydi_sample(self, mock_thread) -> None:
        mock_thread.return_value.start = MagicMock()
        job_id = "render-saydi-sample"
        job_dir = self._write_script_job(job_id)
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        res = self.client.post(
            f"/api/voiceover/script-jobs/{job_id}/render",
            json={"saydi_sample": "custom-sample-123"},
        )
        self.assertEqual(res.status_code, 200)
        thread_kwargs = mock_thread.call_args.kwargs["kwargs"]
        self.assertEqual(thread_kwargs["options"].saydi_sample, "custom-sample-123")

    @patch("auto_subtitle.web.threading.Thread")
    def test_render_endpoint_passes_saydi_speed(self, mock_thread) -> None:
        mock_thread.return_value.start = MagicMock()
        job_id = "render-saydi-speed"
        job_dir = self._write_script_job(job_id)
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        res = self.client.post(
            f"/api/voiceover/script-jobs/{job_id}/render",
            json={"saydi_speed": 1.35},
        )
        self.assertEqual(res.status_code, 200)
        thread_kwargs = mock_thread.call_args.kwargs["kwargs"]
        self.assertEqual(thread_kwargs["options"].saydi_speed, 1.35)

    def test_render_rejects_invalid_saydi_speed(self) -> None:
        job_id = "render-bad-speed"
        job_dir = self._write_script_job(job_id)
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        res = self.client.post(
            f"/api/voiceover/script-jobs/{job_id}/render",
            json={"saydi_speed": 9},
        )
        self.assertEqual(res.status_code, 400)

    def test_render_rejects_invalid_saydi_sample(self) -> None:
        job_id = "render-bad-sample"
        job_dir = self._write_script_job(job_id)
        (job_dir / "voiceover.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nGoc\n", encoding="utf-8"
        )
        res = self.client.post(
            f"/api/voiceover/script-jobs/{job_id}/render",
            json={"saydi_sample": "bad\nsample"},
        )
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
