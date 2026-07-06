import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from auto_subtitle import web
from auto_subtitle.web import Job, JobStatus, jobs, jobs_lock


class UrlImportWebFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        jobs.clear()
        self.tmp = tempfile.mkdtemp()
        self.jobs_root = Path(self.tmp)
        self.jobs_root_patcher = patch.object(web, "JOBS_ROOT", self.jobs_root)
        self.jobs_root_patcher.start()
        self.client = TestClient(web.app)
        self.job_id = "test-url-job-001"
        self.job_dir = self.jobs_root / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.input_path = self.job_dir / "input.mp4"
        self.input_path.write_bytes(b"fake-mp4")

    def tearDown(self) -> None:
        jobs.clear()
        self.jobs_root_patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch.object(web, "_run_job")
    @patch.object(web, "_run_url_import_job")
    def test_from_url_does_not_auto_start_pipeline(
        self, mock_import_job, mock_run_job
    ) -> None:
        res = self.client.post(
            "/api/jobs/from-url",
            json={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "selected_provider": "youtube",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "downloading")
        self.assertEqual(data["provider"], "youtube")
        deadline = time.time() + 1
        while mock_import_job.call_count == 0 and time.time() < deadline:
            time.sleep(0.01)
        mock_run_job.assert_not_called()
        mock_import_job.assert_called_once()

        job_id = data["job_id"]
        meta_path = self.jobs_root / job_id / web.JOB_META_FILENAME
        self.assertTrue(meta_path.is_file())
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["source"], "url")
        self.assertEqual(meta["status"], JobStatus.DOWNLOADING.value)

    def test_provider_mismatch_rejected(self) -> None:
        res = self.client.post(
            "/api/jobs/from-url",
            json={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "selected_provider": "facebook",
            },
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("không khớp", res.json()["detail"].lower())

    def test_input_video_download_when_present(self) -> None:
        with jobs_lock:
            jobs[self.job_id] = Job(
                id=self.job_id,
                status=JobStatus.DOWNLOADED,
                output_name="clip",
                input_path=str(self.input_path),
            )
        res = self.client.get(f"/api/jobs/{self.job_id}/input-video")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content, b"fake-mp4")

    def test_input_video_rehydrates_when_memory_missing(self) -> None:
        meta = {
            "job_id": self.job_id,
            "source": "url",
            "provider": "facebook",
            "status": "downloaded",
            "input_filename": "input.mp4",
            "output_name": "clip",
            "source_language": "en",
            "topic": "economics",
            "translation_engine": "openai",
        }
        (self.job_dir / web.JOB_META_FILENAME).write_text(
            json.dumps(meta), encoding="utf-8"
        )
        res = self.client.get(f"/api/jobs/{self.job_id}/input-video")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content, b"fake-mp4")

    def test_input_video_404_when_missing(self) -> None:
        with jobs_lock:
            jobs[self.job_id] = Job(
                id=self.job_id,
                status=JobStatus.DOWNLOADED,
                output_name="clip",
                input_path=None,
            )
        res = self.client.get(f"/api/jobs/{self.job_id}/input-video")
        self.assertEqual(res.status_code, 404)

    @patch.object(web, "_run_job")
    def test_process_starts_pipeline_for_downloaded_job(self, mock_run_job) -> None:
        with jobs_lock:
            jobs[self.job_id] = Job(
                id=self.job_id,
                status=JobStatus.DOWNLOADED,
                output_name="clip",
                input_path=str(self.input_path),
            )
        res = self.client.post(f"/api/jobs/{self.job_id}/process")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "processing")
        import time

        time.sleep(0.3)
        mock_run_job.assert_called_once_with(self.job_id)

    @patch.object(web, "_run_job")
    def test_process_rehydrates_from_disk_when_memory_missing(self, mock_run_job) -> None:
        meta = {
            "job_id": self.job_id,
            "source": "url",
            "provider": "youtube",
            "status": "downloaded",
            "input_filename": "input.mp4",
            "output_name": "clip",
            "source_language": "en",
            "topic": "economics",
            "translation_engine": "openai",
        }
        (self.job_dir / web.JOB_META_FILENAME).write_text(
            json.dumps(meta), encoding="utf-8"
        )

        res = self.client.post(f"/api/jobs/{self.job_id}/process")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "processing")
        import time

        time.sleep(0.3)
        mock_run_job.assert_called_once_with(self.job_id)
        with jobs_lock:
            self.assertIn(self.job_id, jobs)

    def test_process_invalid_job_id_returns_friendly_404(self) -> None:
        res = self.client.post("/api/jobs/not-a-real-job/process")
        self.assertEqual(res.status_code, 404)

    def test_process_missing_input_returns_friendly_404(self) -> None:
        empty_job_id = "test-url-job-empty"
        empty_dir = self.jobs_root / empty_job_id
        empty_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "job_id": empty_job_id,
            "source": "url",
            "provider": "youtube",
            "status": "downloaded",
            "input_filename": "input.mp4",
            "output_name": "clip",
            "source_language": "en",
            "topic": "economics",
            "translation_engine": "openai",
        }
        (empty_dir / web.JOB_META_FILENAME).write_text(
            json.dumps(meta), encoding="utf-8"
        )
        res = self.client.post(f"/api/jobs/{empty_job_id}/process")
        self.assertEqual(res.status_code, 404)
        self.assertIn("tải lại", res.json()["detail"].lower())

    def test_get_job_rehydrates_from_disk(self) -> None:
        meta = {
            "job_id": self.job_id,
            "source": "url",
            "provider": "facebook",
            "status": "downloaded",
            "input_filename": "input.mp4",
            "output_name": "clip",
            "source_language": "en",
            "topic": "economics",
            "translation_engine": "openai",
        }
        (self.job_dir / web.JOB_META_FILENAME).write_text(
            json.dumps(meta), encoding="utf-8"
        )
        res = self.client.get(f"/api/jobs/{self.job_id}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "downloaded")
        self.assertTrue(data["can_process_subtitles"])


if __name__ == "__main__":
    unittest.main()
