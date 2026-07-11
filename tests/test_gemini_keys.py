import os
import unittest
from unittest.mock import patch

from auto_subtitle.gemini_keys import (
    GeminiQuotaError,
    call_gemini_with_key_rotation,
    gemini_configured,
    is_gemini_key_rotatable_error,
    load_gemini_api_keys,
    resolve_gemini_model_for_keys,
)


class GeminiKeysTests(unittest.TestCase):
    def test_load_gemini_api_keys_prefers_numbered_keys(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY_1": "key-1",
                "GEMINI_API_KEY_2": "key-2",
                "GEMINI_API_KEY_3": "key-3",
                "GEMINI_API_KEY_4": "key-4",
                "GEMINI_API_KEY": "legacy-1",
            },
            clear=False,
        ):
            self.assertEqual(
                load_gemini_api_keys(),
                ["key-1", "key-2", "key-3", "key-4", "legacy-1"],
            )
            self.assertTrue(gemini_configured())

    def test_load_gemini_api_keys_deduplicates(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY_1": "same-key",
                "GEMINI_API_KEY_2": "same-key",
                "GEMINI_API_KEY_3": "",
                "GEMINI_API_KEY_4": "",
                "GEMINI_API_KEY": "same-key",
            },
            clear=False,
        ):
            self.assertEqual(load_gemini_api_keys(), ["same-key"])

    def test_is_gemini_key_rotatable_error(self) -> None:
        self.assertTrue(is_gemini_key_rotatable_error(GeminiQuotaError("quota exceeded")))
        self.assertTrue(
            is_gemini_key_rotatable_error(RuntimeError("Gemini API HTTP 429: quota"))
        )
        self.assertTrue(
            is_gemini_key_rotatable_error(
                RuntimeError("RESOURCE_EXHAUSTED free_tier_requests")
            )
        )
        self.assertFalse(is_gemini_key_rotatable_error(ValueError("bad json")))

    def test_call_gemini_with_key_rotation_switches_on_quota(self) -> None:
        calls: list[str] = []

        def _operation(api_key: str) -> str:
            calls.append(api_key)
            if api_key == "key-1":
                raise GeminiQuotaError("quota exceeded")
            return "ok"

        result = call_gemini_with_key_rotation(
            _operation,
            api_keys=["key-1", "key-2"],
            action="test",
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["key-1", "key-2"])

    def test_call_gemini_with_key_rotation_raises_when_all_keys_exhausted(self) -> None:
        with self.assertRaises(GeminiQuotaError):
            call_gemini_with_key_rotation(
                lambda _key: (_ for _ in ()).throw(GeminiQuotaError("quota")),
                api_keys=["key-1", "key-2"],
                action="test",
            )

    def test_resolve_gemini_model_for_keys_rotates(self) -> None:
        calls: list[str] = []

        def _resolver(api_key: str, model: str) -> str:
            calls.append(api_key)
            if api_key == "key-1":
                raise GeminiQuotaError("quota")
            return model

        resolved, used_key = resolve_gemini_model_for_keys(
            ["key-1", "key-2"],
            "gemini-2.5-flash",
            _resolver,
        )
        self.assertEqual(resolved, "gemini-2.5-flash")
        self.assertEqual(used_key, "key-2")
        self.assertEqual(calls, ["key-1", "key-2"])


if __name__ == "__main__":
    unittest.main()
