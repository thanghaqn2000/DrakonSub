import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from auto_subtitle import web


class HealthEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.jobs_root = Path(self.tmp) / "jobs"
        self.jobs_root_patcher = patch.object(web, "JOBS_ROOT", self.jobs_root)
        self.jobs_root_patcher.start()
        self.client = TestClient(web.app)

    def tearDown(self) -> None:
        self.jobs_root_patcher.stop()

    def test_health_returns_ok_without_secrets(self) -> None:
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["jobs_root_writable"])
        self.assertIn("translation_engine", data)
        self.assertIn("openai_configured", data)
        self.assertNotIn("OPENAI_API_KEY", res.text)


if __name__ == "__main__":
    unittest.main()
