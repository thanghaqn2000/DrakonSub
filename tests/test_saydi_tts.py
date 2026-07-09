import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_subtitle.voiceover.saydi_tts import (
    SaydiConfigError,
    build_saydi_request_payload,
    load_saydi_config,
    resolve_saydi_sample,
    synthesize_to_file,
    validate_saydi_sample,
)


class SaydiSampleValidationTests(unittest.TestCase):
    def test_empty_or_whitespace_returns_none(self) -> None:
        self.assertIsNone(validate_saydi_sample(None))
        self.assertIsNone(validate_saydi_sample(""))
        self.assertIsNone(validate_saydi_sample("   "))

    def test_rejects_control_characters(self) -> None:
        with self.assertRaises(SaydiConfigError):
            validate_saydi_sample("bad\nsample")

    def test_rejects_too_long_value(self) -> None:
        with self.assertRaises(SaydiConfigError):
            validate_saydi_sample("x" * 201)

    def test_accepts_hyphenated_slug(self) -> None:
        self.assertEqual(validate_saydi_sample(" ng-c-huy-n-2-0-abc "), "ng-c-huy-n-2-0-abc")


class SaydiConfigTests(unittest.TestCase):
    @patch("auto_subtitle.voiceover.saydi_tts.load_env")
    @patch.dict(
        os.environ,
        {
            "SAYDI_TTS_API_TOKEN": "secret-token",
            "SAYDI_TTS_SAMPLE": "env-sample",
            "SAYDI_TTS_LANG": "vi",
            "SAYDI_TTS_OUTPUT_FORMAT": "wav",
        },
        clear=True,
    )
    def test_load_config_uses_env_sample_by_default(self, _mock_load_env) -> None:
        cfg = load_saydi_config()
        self.assertEqual(cfg.sample, "env-sample")
        self.assertEqual(cfg.lang, "vi")

    @patch("auto_subtitle.voiceover.saydi_tts.load_env")
    @patch.dict(
        os.environ,
        {
            "SAYDI_TTS_API_TOKEN": "secret-token",
            "SAYDI_TTS_SAMPLE": "env-sample",
        },
        clear=True,
    )
    def test_sample_override_changes_config(self, _mock_load_env) -> None:
        cfg = load_saydi_config(sample_override="custom-sample-123")
        self.assertEqual(cfg.sample, "custom-sample-123")

    @patch("auto_subtitle.voiceover.saydi_tts.load_env")
    @patch.dict(
        os.environ,
        {
            "SAYDI_TTS_API_TOKEN": "secret-token",
            "SAYDI_TTS_SAMPLE": "env-sample",
        },
        clear=True,
    )
    def test_blank_override_falls_back_to_env(self, _mock_load_env) -> None:
        self.assertEqual(resolve_saydi_sample(""), "env-sample")
        self.assertEqual(load_saydi_config(sample_override="").sample, "env-sample")


class SaydiRequestPayloadTests(unittest.TestCase):
    @patch("auto_subtitle.voiceover.saydi_tts.load_env")
    @patch.dict(
        os.environ,
        {
            "SAYDI_TTS_API_TOKEN": "secret-token",
            "SAYDI_TTS_SAMPLE": "env-sample",
        },
        clear=True,
    )
    @patch("auto_subtitle.voiceover.saydi_tts.urllib.request.urlopen")
    def test_synthesize_sends_sample_in_json_body_not_token(self, mock_urlopen, _mock_load_env) -> None:
        class FakeResponse:
            headers = {"Content-Type": "audio/wav"}

            def read(self) -> bytes:
                return b"RIFF"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        mock_urlopen.return_value = FakeResponse()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "seg.wav"
            cfg = load_saydi_config(sample_override="custom-sample-123")
            synthesize_to_file("Xin chao", output_path, config=cfg)

        request = mock_urlopen.call_args[0][0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["sample"], "custom-sample-123")
        self.assertEqual(payload["text"], "Xin chao")
        self.assertNotIn("token", payload)
        self.assertTrue(request.headers["Authorization"].startswith("Bearer "))
        self.assertIn("secret-token", request.headers["Authorization"])

    def test_build_payload_shape(self) -> None:
        from auto_subtitle.voiceover.saydi_tts import SaydiConfig

        payload = build_saydi_request_payload(
            "hello",
            SaydiConfig(
                api_url="https://example.test/tts",
                token="t",
                sample="voice-1",
                output_format="wav",
                timeout_seconds=30,
                lang="vi",
            ),
        )
        self.assertEqual(
            payload,
            {
                "text": "hello",
                "sample": "voice-1",
                "output_format": "wav",
                "lang": "vi",
            },
        )


if __name__ == "__main__":
    unittest.main()
