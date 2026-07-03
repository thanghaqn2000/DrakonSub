import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from auto_subtitle import web
from auto_subtitle.web import Job, JobStatus, jobs, jobs_lock


class UrlImportWebFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        jobs.clear()
        self.client = TestClient(web.app)
        self.tmp = tempfile.mkdtemp()
        self.job_id = "test-url-job-001"
        self.job_dir = Path(self.tmp) / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.input_path = self.job_dir / "input.mp4"
        self.input_path.write_bytes(b"fake-mp4")

    def tearDown(self) -> None:
        jobs.clear()

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
        mock_run_job.assert_not_called()
        mock_import_job.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
