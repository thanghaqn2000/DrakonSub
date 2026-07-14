import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from auto_subtitle import web


class SrtAudioWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "srt-audio-jobs"
        self.root_patcher = patch.object(web, "SRT_AUDIO_JOBS_ROOT", self.root)
        self.root_patcher.start()
        self.client = TestClient(web.app)

    def tearDown(self) -> None:
        self.root_patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _create_job(self, text: Optional[bytes] = None) -> dict:
        srt = text or b"1\n00:00:00,000 --> 00:00:02,000\nXin chao\n"
        res = self.client.post(
            "/api/srt-audio/jobs",
            files={"srt_file": ("clip.srt", srt, "application/x-subrip")},
        )
        self.assertEqual(res.status_code, 200, res.text)
        return res.json()

    def test_create_and_get_cues(self) -> None:
        data = self._create_job()
        job_id = data["job_id"]
        status = self.client.get(f"/api/srt-audio/jobs/{job_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "ready")
        cues = self.client.get(f"/api/srt-audio/jobs/{job_id}/cues")
        self.assertEqual(cues.status_code, 200)
        body = cues.json()
        self.assertEqual(len(body["cues"]), 1)
        self.assertFalse(body["has_blocking_issues"])

    def test_put_cues_and_reject_bad_synthesize(self) -> None:
        data = self._create_job()
        job_id = data["job_id"]
        put = self.client.put(
            f"/api/srt-audio/jobs/{job_id}/cues",
            json={
                "saydi_speed": 1.0,
                "cues": [
                    {
                        "index": 1,
                        "start": "00:00:00,000",
                        "end": "00:00:01,000",
                        "text": "x" * 80,
                    }
                ],
            },
        )
        self.assertEqual(put.status_code, 200)
        self.assertTrue(put.json()["has_blocking_issues"])
        synth = self.client.post(
            f"/api/srt-audio/jobs/{job_id}/synthesize",
            json={"output_format": "wav", "saydi_speed": 1.0},
        )
        self.assertEqual(synth.status_code, 400)

    @patch("auto_subtitle.web.threading.Thread")
    def test_synthesize_starts_background(self, mock_thread) -> None:
        mock_thread.return_value.start = MagicMock()
        data = self._create_job()
        job_id = data["job_id"]
        synth = self.client.post(
            f"/api/srt-audio/jobs/{job_id}/synthesize",
            json={"output_format": "wav", "saydi_speed": 1.0},
        )
        self.assertEqual(synth.status_code, 200)
        self.assertEqual(synth.json()["status"], "processing")
        mock_thread.assert_called_once()

    def test_audio_download_after_completed(self) -> None:
        data = self._create_job()
        job_id = data["job_id"]
        job_dir = self.root / job_id
        (job_dir / "output.wav").write_bytes(b"RIFF")
        meta = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        meta["status"] = "completed"
        meta["output_format"] = "wav"
        (job_dir / "job.json").write_text(json.dumps(meta), encoding="utf-8")
        res = self.client.get(f"/api/srt-audio/jobs/{job_id}/audio?format=wav")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content, b"RIFF")


if __name__ == "__main__":
    unittest.main()
