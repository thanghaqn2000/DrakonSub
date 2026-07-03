import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_subtitle.url_import_service import (
    FACEBOOK_DOWNLOAD_FAIL_MESSAGE,
    FACEBOOK_UNSUPPORTED_MESSAGE,
    PROVIDER_MISMATCH_MESSAGE,
    UrlImportError,
    cleanup_partial_downloads,
    detect_provider,
    download_video_from_url,
    validate_url_with_selected_provider,
    validate_video_url,
)


class UrlImportServiceTests(unittest.TestCase):
    def test_valid_youtube_watch(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(validate_video_url(url), url)
        self.assertEqual(detect_provider(url), "youtube")

    def test_valid_youtu_be(self) -> None:
        url = "https://youtu.be/dQw4w9WgXcQ"
        self.assertEqual(detect_provider(url), "youtube")

    def test_valid_youtube_shorts(self) -> None:
        url = "https://www.youtube.com/shorts/abc123"
        self.assertEqual(detect_provider(url), "youtube")

    def test_valid_facebook_video(self) -> None:
        url = "https://www.facebook.com/user/videos/1234567890/"
        self.assertEqual(detect_provider(url), "facebook")

    def test_valid_facebook_reel(self) -> None:
        url = "https://www.facebook.com/reel/1234567890"
        self.assertEqual(detect_provider(url), "facebook")

    def test_valid_fb_watch(self) -> None:
        url = "https://fb.watch/abc123/"
        self.assertEqual(detect_provider(url), "facebook")

    def test_valid_facebook_watch_query(self) -> None:
        url = "https://www.facebook.com/watch/?v=123456789"
        self.assertEqual(detect_provider(url), "facebook")

    def test_valid_m_facebook_watch_query(self) -> None:
        url = "https://m.facebook.com/watch/?v=123456789"
        self.assertEqual(detect_provider(url), "facebook")

    def test_valid_facebook_share_video(self) -> None:
        url = "https://www.facebook.com/share/v/abc123/"
        self.assertEqual(detect_provider(url), "facebook")

    def test_valid_facebook_share_reel(self) -> None:
        url = "https://www.facebook.com/share/r/abc123/"
        self.assertEqual(detect_provider(url), "facebook")

    def test_invalid_facebook_profile(self) -> None:
        with self.assertRaises(UrlImportError) as ctx:
            validate_video_url("https://www.facebook.com/profile.php?id=123")
        self.assertEqual(str(ctx.exception), FACEBOOK_UNSUPPORTED_MESSAGE)

    def test_invalid_facebook_post(self) -> None:
        with self.assertRaises(UrlImportError) as ctx:
            validate_video_url("https://www.facebook.com/somepage/posts/123")
        self.assertEqual(str(ctx.exception), FACEBOOK_UNSUPPORTED_MESSAGE)

    def test_invalid_facebook_photo(self) -> None:
        with self.assertRaises(UrlImportError) as ctx:
            validate_video_url("https://www.facebook.com/photo/?fbid=123")
        self.assertEqual(str(ctx.exception), FACEBOOK_UNSUPPORTED_MESSAGE)

    def test_invalid_unsupported_domain(self) -> None:
        with self.assertRaises(UrlImportError):
            validate_video_url("https://example.com/video.mp4")

    def test_invalid_localhost(self) -> None:
        with self.assertRaises(UrlImportError):
            validate_video_url("http://localhost/video")

    def test_invalid_private_ip(self) -> None:
        with self.assertRaises(UrlImportError):
            validate_video_url("http://192.168.1.10/video")

    def test_invalid_missing_scheme(self) -> None:
        with self.assertRaises(UrlImportError):
            validate_video_url("www.youtube.com/watch?v=abc")

    def test_invalid_empty(self) -> None:
        with self.assertRaises(UrlImportError):
            validate_video_url("")

    def test_youtube_selected_with_youtube_url(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        safe, provider = validate_url_with_selected_provider(url, "youtube")
        self.assertEqual(safe, url)
        self.assertEqual(provider, "youtube")

    def test_facebook_selected_with_facebook_url(self) -> None:
        url = "https://www.facebook.com/user/videos/1234567890/"
        safe, provider = validate_url_with_selected_provider(url, "facebook")
        self.assertEqual(safe, url)
        self.assertEqual(provider, "facebook")

    def test_youtube_selected_with_facebook_url_rejected(self) -> None:
        url = "https://www.facebook.com/user/videos/1234567890/"
        with self.assertRaises(UrlImportError) as ctx:
            validate_url_with_selected_provider(url, "youtube")
        self.assertEqual(str(ctx.exception), PROVIDER_MISMATCH_MESSAGE)

    def test_facebook_selected_with_youtube_url_rejected(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        with self.assertRaises(UrlImportError) as ctx:
            validate_url_with_selected_provider(url, "facebook")
        self.assertEqual(str(ctx.exception), PROVIDER_MISMATCH_MESSAGE)

    @patch("yt_dlp.YoutubeDL")
    def test_facebook_download_failure_maps_friendly_error(self, mock_ydl) -> None:
        instance = mock_ydl.return_value.__enter__.return_value
        instance.extract_info.side_effect = RuntimeError("login required")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UrlImportError) as ctx:
                download_video_from_url(
                    "https://www.facebook.com/user/videos/1234567890/",
                    tmp,
                )
            self.assertEqual(str(ctx.exception), FACEBOOK_DOWNLOAD_FAIL_MESSAGE)
            self.assertFalse(list(Path(tmp).glob("input.*")))

    def test_cleanup_partial_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / "input.mp4.part"
            partial.write_bytes(b"x")
            cleanup_partial_downloads(root)
            self.assertFalse(partial.exists())


if __name__ == "__main__":
    unittest.main()
