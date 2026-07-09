import json
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


if __name__ == "__main__":
    unittest.main()
