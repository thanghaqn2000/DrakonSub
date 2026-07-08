import os
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
    YOUTUBE_BOT_BLOCK_MESSAGE,
    _build_ydl_opts,
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

    @patch.dict("os.environ", {}, clear=False)
    @patch("auto_subtitle.url_import_service._browser_cookie_source", return_value=("chrome",))
    def test_build_youtube_opts_falls_back_to_browser_cookies_when_no_file(
        self,
        mock_browser_cookie_source,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opts = _build_ydl_opts(Path(tmp), "youtube", use_browser_cookies=True)
        self.assertEqual(opts["cookiesfrombrowser"], ("chrome",))
        mock_browser_cookie_source.assert_called_once()

    @patch.dict("os.environ", {}, clear=False)
    def test_build_youtube_opts_keeps_local_default_clients_without_server_cookie_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opts = _build_ydl_opts(Path(tmp), "youtube")
        self.assertEqual(
            opts["extractor_args"],
            {"youtube": {"player_client": ["android", "web"]}},
        )
        self.assertNotIn("remote_components", opts)

    @patch("yt_dlp.YoutubeDL")
    @patch("auto_subtitle.url_import_service._browser_cookie_source", return_value=("chrome",))
    @patch(
        "auto_subtitle.url_import_service._normalize_downloaded_video",
        side_effect=lambda _out_dir, downloaded: downloaded,
    )
    def test_youtube_retries_with_browser_cookies_after_reload_error(
        self,
        _mock_normalize_downloaded_video,
        _mock_browser_cookie_source,
        mock_ydl,
    ) -> None:
        first_cm = unittest.mock.MagicMock()
        second_cm = unittest.mock.MagicMock()
        first = first_cm.__enter__.return_value
        second = second_cm.__enter__.return_value
        mock_ydl.side_effect = [first_cm, second_cm]

        first.extract_info.side_effect = RuntimeError("The page needs to be reloaded.")
        second.extract_info.return_value = {"duration": 10, "title": "zoo"}

        def _write_mp4(_urls) -> None:
            path = Path(tmpdir) / "input.mp4"
            path.write_bytes(b"mp4")

        second.download.side_effect = _write_mp4

        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_video_from_url(
                "https://www.youtube.com/watch?v=jNQXAC9IVRw",
                tmpdir,
            )

        self.assertEqual(result["provider"], "youtube")
        self.assertEqual(mock_ydl.call_count, 2)
        first_opts = mock_ydl.call_args_list[0].args[0]
        second_opts = mock_ydl.call_args_list[1].args[0]
        self.assertNotIn("cookiesfrombrowser", first_opts)
        self.assertEqual(second_opts["cookiesfrombrowser"], ("chrome",))

    @patch("auto_subtitle.url_import_service._try_youtube_external_cascade", return_value=None)
    @patch("yt_dlp.YoutubeDL")
    @patch("auto_subtitle.url_import_service._browser_cookie_source", return_value=None)
    def test_youtube_reload_error_maps_to_friendly_bot_message_without_cookie_fallback(
        self,
        _mock_browser_cookie_source,
        mock_ydl,
        _mock_external_cascade,
    ) -> None:
        instance = mock_ydl.return_value.__enter__.return_value
        instance.extract_info.side_effect = RuntimeError("The page needs to be reloaded.")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UrlImportError) as ctx:
                download_video_from_url(
                    "https://www.youtube.com/watch?v=jNQXAC9IVRw",
                    tmp,
                )
        self.assertEqual(str(ctx.exception), YOUTUBE_BOT_BLOCK_MESSAGE)

    def test_cleanup_partial_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / "input.mp4.part"
            partial.write_bytes(b"x")
            cleanup_partial_downloads(root)
            self.assertFalse(partial.exists())

    @patch.dict(
        os.environ,
        {
            "VIDEO_DOWNLOAD_API_KEY": "vda_test",
            "TUNELIO_API_KEY": "tnl_test",
            "CAPTAPI_API_KEY": "capt_test",
        },
        clear=False,
    )
    @patch("auto_subtitle.url_import_service._download_youtube_with_ytdlp")
    @patch("auto_subtitle.url_import_service._download_youtube_via_external")
    def test_youtube_tries_tunelio_then_captapi_before_ytdlp(
        self,
        mock_external,
        mock_ytdlp,
    ) -> None:
        from auto_subtitle.youtube_external_download import ExternalCreditsExhaustedError

        mock_external.side_effect = [
            ExternalCreditsExhaustedError("video-download-api credits exhausted"),
            ExternalCreditsExhaustedError("tunelio credits exhausted"),
            {
                "path": "/tmp/input.mp4",
                "provider": "youtube",
                "title": "demo",
                "duration": 10,
                "filesize": 123,
                "external_provider": "captapi",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = download_video_from_url(
                "https://www.youtube.com/watch?v=jNQXAC9IVRw",
                tmp,
            )
        self.assertEqual(result["external_provider"], "captapi")
        self.assertEqual(mock_external.call_count, 3)
        providers_called = [
            call.kwargs["external_provider"] for call in mock_external.call_args_list
        ]
        self.assertEqual(
            providers_called,
            ["video-download-api", "tunelio", "captapi"],
        )
        mock_ytdlp.assert_not_called()

    @patch.dict(
        os.environ,
        {
            "VIDEO_DOWNLOAD_API_KEY": "vda_test",
            "TUNELIO_API_KEY": "tnl_test",
            "CAPTAPI_API_KEY": "capt_test",
        },
        clear=False,
    )
    @patch("auto_subtitle.url_import_service._download_youtube_with_ytdlp")
    @patch("auto_subtitle.url_import_service._download_youtube_via_external")
    def test_youtube_falls_back_to_ytdlp_when_external_credits_exhausted(
        self,
        mock_external,
        mock_ytdlp,
    ) -> None:
        from auto_subtitle.youtube_external_download import ExternalCreditsExhaustedError

        mock_external.side_effect = [
            ExternalCreditsExhaustedError("video-download-api credits exhausted"),
            ExternalCreditsExhaustedError("tunelio credits exhausted"),
            ExternalCreditsExhaustedError("captapi credits exhausted"),
        ]
        mock_ytdlp.return_value = {
            "path": "/tmp/input.mp4",
            "provider": "youtube",
            "title": "demo",
            "duration": 10,
            "filesize": 123,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = download_video_from_url(
                "https://www.youtube.com/watch?v=jNQXAC9IVRw",
                tmp,
            )
        self.assertEqual(result["provider"], "youtube")
        self.assertEqual(mock_external.call_count, 3)
        providers_called = [
            call.kwargs["external_provider"] for call in mock_external.call_args_list
        ]
        self.assertEqual(
            providers_called,
            ["video-download-api", "tunelio", "captapi"],
        )
        mock_ytdlp.assert_called_once()


if __name__ == "__main__":
    unittest.main()
