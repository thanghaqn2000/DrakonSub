import unittest

from auto_subtitle.url_import_service import (
    UrlImportError,
    detect_provider,
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


if __name__ == "__main__":
    unittest.main()
