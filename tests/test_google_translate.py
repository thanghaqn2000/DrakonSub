import unittest
from unittest.mock import MagicMock, patch

from auto_subtitle.utils import (
    assert_vietnamese_translation_applied,
    translate_srt_entries,
)


class GoogleTranslateEngineTests(unittest.TestCase):
    @patch("auto_subtitle.google_translate.GoogleTranslator")
    def test_translate_srt_entries_google_routes(self, mock_translator_cls):
        mock_translator = MagicMock()
        mock_translator.translate_batch.return_value = [
            "Tôi có một người bạn",
            "nhưng đang lên.",
        ]
        mock_translator_cls.return_value = mock_translator

        entries = [
            {"start_str": "00:00:00,000", "end_str": "00:00:02,000", "text": "I have a friend"},
            {"start_str": "00:00:02,000", "end_str": "00:00:04,000", "text": "but was coming up."},
        ]
        result = translate_srt_entries(entries, target_lang="vi", engine="google")
        self.assertEqual(result[0]["text"], "Tôi có một người bạn")
        self.assertEqual(result[1]["text"], "nhưng đang lên.")

    def test_assert_vietnamese_translation_applied_raises_on_english_fallback(self):
        entries = [
            {"text": "Hello world"},
            {"text": "How are you"},
        ]
        with self.assertRaises(RuntimeError):
            assert_vietnamese_translation_applied(entries, entries, target_lang="vi")

    def test_assert_vietnamese_translation_applied_passes_when_translated(self):
        source = [{"text": "Hello"}, {"text": "World"}]
        translated = [{"text": "Xin chào"}, {"text": "Thế giới"}]
        assert_vietnamese_translation_applied(source, translated, target_lang="vi")


if __name__ == "__main__":
    unittest.main()
