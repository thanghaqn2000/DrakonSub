import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from auto_subtitle import web
from auto_subtitle.gemini_translate import (
    check_gemini_capability,
    classify_gemini_error,
    reset_gemini_capability_cache,
)


class ClassifyGeminiErrorTests(unittest.TestCase):
    def test_location_restriction(self) -> None:
        text = (
            'Gemini listModels HTTP 400: {"error":{"code":400,'
            '"message":"User location is not supported for the API use.",'
            '"status":"FAILED_PRECONDITION"}}'
        )
        self.assertEqual(classify_gemini_error(text), "location_restricted")

    def test_auth_error(self) -> None:
        self.assertEqual(classify_gemini_error("HTTP 401 unauthorized"), "auth_error")
        self.assertEqual(classify_gemini_error("HTTP 403 forbidden"), "auth_error")

    def test_quota_error(self) -> None:
        self.assertEqual(classify_gemini_error("quota exceeded 429"), "quota_error")


class GeminiCapabilityProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_gemini_capability_cache()

    def tearDown(self) -> None:
        reset_gemini_capability_cache()

    @patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False)
    def test_not_configured_when_key_missing(self) -> None:
        result = check_gemini_capability(force=True)
        self.assertFalse(result["configured"])
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "not_configured")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
    @patch("auto_subtitle.gemini_translate._list_gemini_models")
    def test_available_when_probe_succeeds(self, mock_list) -> None:
        mock_list.return_value = ["gemini-2.5-flash"]
        result = check_gemini_capability(force=True)
        self.assertTrue(result["configured"])
        self.assertTrue(result["available"])
        self.assertIsNone(result["reason"])
        mock_list.assert_called_once_with("test-key", timeout=8)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
    @patch("auto_subtitle.gemini_translate._list_gemini_models")
    def test_location_restricted_when_probe_fails(self, mock_list) -> None:
        mock_list.side_effect = RuntimeError(
            'Gemini listModels HTTP 400: {"error":{"message":'
            '"User location is not supported for the API use.",'
            '"status":"FAILED_PRECONDITION"}}'
        )
        result = check_gemini_capability(force=True)
        self.assertTrue(result["configured"])
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "location_restricted")
        self.assertIn("location", result["message"].lower())


class DefaultsAndJobGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_gemini_capability_cache()
        self.tmp = tempfile.mkdtemp()
        self.jobs_root = Path(self.tmp) / "jobs"
        self.jobs_root_patcher = patch.object(web, "JOBS_ROOT", self.jobs_root)
        self.jobs_root_patcher.start()
        self.client = TestClient(web.app)

    def tearDown(self) -> None:
        self.client.close()
        self.jobs_root_patcher.stop()
        reset_gemini_capability_cache()

    @patch("auto_subtitle.web.check_gemini_capability")
    def test_defaults_includes_engine_capabilities(self, mock_cap) -> None:
        mock_cap.return_value = {
            "configured": True,
            "available": False,
            "reason": "location_restricted",
            "message": (
                "Gemini đang bị Google chặn theo location của server Azure hiện tại. "
                "Tạm thời hãy dùng Google hoặc OpenAI."
            ),
        }
        res = self.client.get("/api/defaults")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("engine_capabilities", data)
        self.assertEqual(
            data["engine_capabilities"]["gemini"]["reason"],
            "location_restricted",
        )
        self.assertFalse(data["engine_capabilities"]["gemini"]["available"])

    @patch("auto_subtitle.web.check_gemini_capability")
    def test_parse_rejects_unavailable_gemini(self, mock_cap) -> None:
        mock_cap.return_value = {
            "configured": True,
            "available": False,
            "reason": "location_restricted",
            "message": (
                "Gemini hiện chưa khả dụng trên server này do giới hạn location từ Google. "
                "Vui lòng chọn Google hoặc OpenAI."
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            web._parse_job_creation_fields(
                output_name="gemini-blocked",
                topic="economics",
                source_language="en",
                translation_engine="gemini",
            )
        self.assertIn("Gemini hiện chưa khả dụng", str(ctx.exception))
        mock_cap.assert_called()

    @patch("auto_subtitle.web.check_gemini_capability")
    def test_parse_allows_google_when_gemini_unavailable(self, mock_cap) -> None:
        mock_cap.return_value = {
            "configured": True,
            "available": False,
            "reason": "location_restricted",
            "message": "Gemini unavailable",
        }
        fields = web._parse_job_creation_fields(
            output_name="google-ok",
            topic="economics",
            source_language="en",
            translation_engine="google",
        )
        self.assertEqual(fields["translation_engine"], "google")
        mock_cap.assert_not_called()


class ResolveGeminiModelSkipListTests(unittest.TestCase):
    @patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-flash"}, clear=False)
    @patch("auto_subtitle.gemini_translate._list_gemini_models")
    def test_skips_list_models_when_model_configured(self, mock_list) -> None:
        from auto_subtitle.gemini_translate import _resolve_gemini_model

        model = _resolve_gemini_model("test-key", "gemini-2.5-flash")
        self.assertEqual(model, "gemini-2.5-flash")
        mock_list.assert_not_called()


if __name__ == "__main__":
    unittest.main()
