import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from auto_subtitle import web
from auto_subtitle.voiceover.job_service import VoiceoverJobError, VoiceoverJobResult


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

    @patch.object(web, "run_voiceover_job")
    def test_post_voiceover_job_completed(self, mock_run_voiceover_job) -> None:
        job_dir = self.voiceover_root / "job-1"
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = job_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"summary": {"cue_count": 1}}), encoding="utf-8")
        output_video = job_dir / "output_voiceover.mp4"
        output_video.write_bytes(b"video")
        prepared_srt = job_dir / "prepared_voiceover.srt"
        prepared_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
        mock_run_voiceover_job.return_value = VoiceoverJobResult(
            output_video=output_video,
            manifest_path=manifest_path,
            prepared_srt_path=prepared_srt,
            cue_count=1,
            segment_count=1,
            summary={"cue_count": 1},
            warnings=[],
        )
        res = self.client.post(
            "/api/voiceover/jobs",
            files={
                "input_video": ("clip.mp4", b"fake-video", "video/mp4"),
                "voiceover_srt": ("voice.srt", b"1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", "application/x-subrip"),
            },
            data={"prepare_text": "true", "voiceover_topic": "catholic"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "completed")
        self.assertTrue(data["output_video_url"].endswith("/output-video"))

    @patch.object(web, "run_voiceover_job", side_effect=VoiceoverJobError("SAYDI failed badly"))
    def test_post_voiceover_job_failed_sanitized(self, _mock_run_voiceover_job) -> None:
        res = self.client.post(
            "/api/voiceover/jobs",
            files={
                "input_video": ("clip.mp4", b"fake-video", "video/mp4"),
                "voiceover_srt": ("voice.srt", b"1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", "application/x-subrip"),
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "failed")
        self.assertIn("SAYDI failed badly", res.json()["error"])

    def test_get_voiceover_job_status_completed(self) -> None:
        job_dir = self.voiceover_root / "job-status"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "job_id": "job-status",
                    "status": "completed",
                    "created_at": "now",
                    "updated_at": "now",
                    "error": None,
                    "input_video": "input.mp4",
                    "voiceover_srt": "voiceover.srt",
                    "output_video": "output_voiceover.mp4",
                    "manifest": "manifest.json",
                    "summary": {"cue_count": 4},
                }
            ),
            encoding="utf-8",
        )
        res = self.client.get("/api/voiceover/jobs/job-status")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "completed")

    def test_get_voiceover_manifest_returns_json(self) -> None:
        job_dir = self.voiceover_root / "job-manifest"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "manifest.json").write_text(json.dumps({"job_type": "voiceover"}), encoding="utf-8")
        res = self.client.get("/api/voiceover/jobs/job-manifest/manifest")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["job_type"], "voiceover")

    def test_get_voiceover_output_video_returns_file(self) -> None:
        job_dir = self.voiceover_root / "job-video"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "output_voiceover.mp4").write_bytes(b"fake-video")
        res = self.client.get("/api/voiceover/jobs/job-video/output-video")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content, b"fake-video")

    def test_missing_voiceover_job_returns_404(self) -> None:
        res = self.client.get("/api/voiceover/jobs/not-found")
        self.assertEqual(res.status_code, 404)

    def test_invalid_upload_returns_friendly_400(self) -> None:
        res = self.client.post(
            "/api/voiceover/jobs",
            files={
                "input_video": ("clip.txt", b"bad", "text/plain"),
                "voiceover_srt": ("voice.srt", b"1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", "application/x-subrip"),
            },
        )
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
