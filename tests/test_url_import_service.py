import tempfile
import unittest
from pathlib import Path
import subprocess
from typing import Optional, Tuple
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
    def _probe_codecs(self, path: Path) -> Tuple[Optional[str], Optional[str]]:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,codec_type",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        import json

        data = json.loads(result.stdout)
        video_codec = None
        audio_codec = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_codec = stream.get("codec_name")
            elif stream.get("codec_type") == "audio":
                audio_codec = stream.get("codec_name")
        return video_codec, audio_codec

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

    def test_invalid_youtube_homepage(self) -> None:
        with self.assertRaises(UrlImportError):
            validate_video_url("https://www.youtube.com/")

    def test_invalid_youtube_channel(self) -> None:
        with self.assertRaises(UrlImportError):
            validate_video_url("https://www.youtube.com/@somechannel")

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

    def test_valid_facebook_reel_alphanumeric(self) -> None:
        url = "https://www.facebook.com/reel/1AbcDefGhIj"
        self.assertEqual(detect_provider(url), "facebook")

    def test_valid_facebook_share_with_query(self) -> None:
        url = "https://www.facebook.com/share/r/abc123/?mibextid=xxxxx"
        self.assertEqual(detect_provider(url), "facebook")

    def test_facebook_cannot_parse_data_maps_friendly(self) -> None:
        from auto_subtitle.url_import_service import _map_download_error

        mapped = _map_download_error(RuntimeError("Cannot parse data"), "facebook")
        self.assertEqual(str(mapped), FACEBOOK_DOWNLOAD_FAIL_MESSAGE)

    def test_user_reported_facebook_reel(self) -> None:
        url = "https://www.facebook.com/reel/1400852521880565"
        safe, provider = validate_url_with_selected_provider(url, "facebook")
        self.assertEqual(safe, url)
        self.assertEqual(provider, "facebook")

    @patch("yt_dlp.YoutubeDL")
    def test_facebook_download_failure_maps_friendly_error(self, mock_ydl) -> None:
        instance = mock_ydl.return_value.__enter__.return_value
        instance.extract_info.side_effect = RuntimeError("login required")

        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "input.mp4.part"
            partial.write_bytes(b"x")
            with self.assertRaises(UrlImportError) as ctx:
                download_video_from_url(
                    "https://www.facebook.com/user/videos/1234567890/",
                    tmp,
                )
            self.assertEqual(str(ctx.exception), FACEBOOK_DOWNLOAD_FAIL_MESSAGE)
            self.assertFalse(partial.exists())
            self.assertFalse(list(Path(tmp).glob("input.*")))

    @patch("yt_dlp.YoutubeDL")
    def test_facebook_download_normalizes_to_quicktime_friendly_mp4(self, mock_ydl) -> None:
        instance = mock_ydl.return_value.__enter__.return_value
        instance.extract_info.return_value = {"duration": 10, "title": "fb clip"}

        def _write_vp9_sample(_urls) -> None:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=160x120:d=1",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=1",
                    "-c:v",
                    "libvpx-vp9",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(Path(tmp) / "input.mp4"),
                ],
                check=True,
                capture_output=True,
            )

        instance.download.side_effect = _write_vp9_sample

        with tempfile.TemporaryDirectory() as tmp:
            result = download_video_from_url(
                "https://www.facebook.com/reel/1400852521880565",
                tmp,
            )

            self.assertEqual(Path(result["path"]).suffix.lower(), ".mp4")
            video_codec, audio_codec = self._probe_codecs(Path(result["path"]))
            self.assertEqual(video_codec, "h264")
            self.assertEqual(audio_codec, "aac")

    def test_cleanup_partial_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / "input.mp4.part"
            partial.write_bytes(b"x")
            cleanup_partial_downloads(root)
            self.assertFalse(partial.exists())


if __name__ == "__main__":
    unittest.main()
