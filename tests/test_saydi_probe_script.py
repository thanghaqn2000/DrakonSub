import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PROBE_PATH = ROOT / "scripts" / "probe_saydi_tts.py"
_SPEC = importlib.util.spec_from_file_location("probe_saydi_tts", _PROBE_PATH)
assert _SPEC and _SPEC.loader
probe_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe_mod)


class ClassifySaydiErrorTests(unittest.TestCase):
    def test_auth_error(self) -> None:
        self.assertEqual(probe_mod.classify_saydi_error("HTTP 401 unauthorized", 401), "auth_error")

    def test_quota_error(self) -> None:
        self.assertEqual(probe_mod.classify_saydi_error("quota exceeded", 402), "quota_error")

    def test_network_error(self) -> None:
        self.assertEqual(probe_mod.classify_saydi_error("connection timed out"), "network_error")


class ProbeSaydiConfigTests(unittest.TestCase):
    @patch.object(probe_mod, "load_env")
    @patch.dict(os.environ, {"SAYDI_TTS_API_TOKEN": ""}, clear=True)
    def test_missing_token_is_not_configured(self, _mock_load_env) -> None:
        result = probe_mod.probe_saydi_tts(dry_run=True)
        self.assertFalse(result["configured"])
        self.assertFalse(result["available"])
        self.assertEqual(result["status"], "not_configured")

    @patch.object(probe_mod, "load_env")
    @patch.dict(
        os.environ,
        {
            "SAYDI_TTS_API_TOKEN": "test-token",
            "SAYDI_TTS_API_URL": "https://api.voice.saydi.ai/tts",
            "SAYDI_TTS_SAMPLE": "sample-voice",
            "SAYDI_TTS_OUTPUT_FORMAT": "wav",
        },
        clear=True,
    )
    @patch.object(probe_mod.urllib.request, "urlopen")
    def test_success_writes_audio_file(self, mock_urlopen, _mock_load_env) -> None:
        class FakeResponse:
            headers = {"Content-Type": "audio/wav"}

            def read(self) -> bytes:
                return b"RIFF\x00\x00\x00\x00WAVE"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        mock_urlopen.return_value = FakeResponse()

        with patch.object(probe_mod, "_probe_audio_duration", return_value=1.5):
            result = probe_mod.probe_saydi_tts(
                output_dir=Path("/tmp/drakonsub_saydi_probe_test"),
            )

        self.assertTrue(result["configured"])
        self.assertTrue(result["available"])
        self.assertEqual(result["status"], "ok")
        self.assertTrue(Path(result["output_file"]).is_file())
        self.assertGreater(result["output_size_bytes"], 0)
        mock_urlopen.assert_called_once()
        request = mock_urlopen.call_args[0][0]
        self.assertTrue(request.headers["Authorization"].startswith("Bearer "))
        self.assertIn("test-token", request.headers["Authorization"])


if __name__ == "__main__":
    unittest.main()
