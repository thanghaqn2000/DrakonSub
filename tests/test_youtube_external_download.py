"""Tests for third-party YouTube download API POC."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from auto_subtitle.youtube_external_download import (
    ExternalCreditsExhaustedError,
    probe_provider,
    resolve_video_download_api,
    resolve_captapi,
    resolve_tunelio,
    youtube_external_provider_chain,
)


class VideoDownloadApiResolveTests(unittest.TestCase):
    @patch.dict(os.environ, {"VIDEO_DOWNLOAD_API_KEY_1": "vda_test"}, clear=False)
    @patch("auto_subtitle.youtube_external_download._request_json")
    @patch("auto_subtitle.youtube_external_download.time.sleep")
    def test_resolve_video_download_api_returns_ready_url(
        self,
        _mock_sleep,
        mock_request,
    ) -> None:
        mock_request.return_value = {
            "success": True,
            "url": "https://worker.savenow.to/api/v2/download/token",
            "filename": "Me at the zoo.mp4",
            "title": "Me at the zoo",
            "format": "720",
        }
        result = resolve_video_download_api("https://www.youtube.com/watch?v=jNQXAC9IVRw")
        self.assertEqual(result.provider, "video-download-api")
        self.assertIn("savenow.to", result.download_url)
        self.assertEqual(result.title, "Me at the zoo")

    @patch.dict(
        os.environ,
        {
            "VIDEO_DOWNLOAD_API_KEY_1": "vda_1",
            "VIDEO_DOWNLOAD_API_KEY_2": "vda_2",
        },
        clear=False,
    )
    @patch("auto_subtitle.youtube_external_download._request_json")
    @patch("auto_subtitle.youtube_external_download.time.sleep")
    def test_resolve_video_download_api_tries_next_key_after_credit_error(
        self,
        _mock_sleep,
        mock_request,
    ) -> None:
        def side_effect(url, **kwargs):
            if "apikey=vda_1" in url:
                raise ExternalCreditsExhaustedError("credits exhausted")
            return {
                "success": True,
                "url": "https://worker.savenow.to/api/v2/download/token",
                "filename": "Me at the zoo.mp4",
                "title": "Me at the zoo",
                "format": "720",
            }

        mock_request.side_effect = side_effect
        result = resolve_video_download_api("https://www.youtube.com/watch?v=jNQXAC9IVRw")
        self.assertEqual(result.provider, "video-download-api")
        self.assertIn("savenow.to", result.download_url)


class CaptapiResolveTests(unittest.TestCase):
    @patch.dict(os.environ, {"CAPTAPI_API_KEY_1": "capt_live_test"}, clear=False)
    @patch("auto_subtitle.youtube_external_download._request_json")
    def test_resolve_captapi_returns_download_url(self, mock_request) -> None:
        mock_request.return_value = {
            "success": True,
            "cached": False,
            "creditsUsed": 3,
            "data": {
                "title": "Me at the zoo",
                "downloadUrl": "https://redirector.googlevideo.com/videoplayback?id=abc",
                "approxDurationMs": "19000",
            },
        }
        result = resolve_captapi("https://www.youtube.com/watch?v=jNQXAC9IVRw")
        self.assertEqual(result.provider, "captapi")
        self.assertIn("googlevideo.com", result.download_url)
        self.assertEqual(result.title, "Me at the zoo")
        self.assertEqual(result.duration_seconds, 19)
        self.assertEqual(result.credits_used, 3)

    @patch.dict(
        os.environ,
        {
            "CAPTAPI_API_KEY_1": "capt_1",
            "CAPTAPI_API_KEY_2": "capt_2",
        },
        clear=False,
    )
    @patch("auto_subtitle.youtube_external_download._request_json")
    def test_resolve_captapi_tries_next_key_after_credit_error(self, mock_request) -> None:
        def side_effect(url, **kwargs):
            auth = (kwargs.get("headers") or {}).get("Authorization", "")
            if auth == "Bearer capt_1":
                raise ExternalCreditsExhaustedError("credits exhausted")
            return {
                "success": True,
                "cached": False,
                "creditsUsed": 3,
                "data": {
                    "title": "Me at the zoo",
                    "downloadUrl": "https://redirector.googlevideo.com/videoplayback?id=abc",
                    "approxDurationMs": "19000",
                },
            }

        mock_request.side_effect = side_effect
        result = resolve_captapi("https://www.youtube.com/watch?v=jNQXAC9IVRw")
        self.assertEqual(result.provider, "captapi")
        self.assertIn("googlevideo.com", result.download_url)


class TunelioResolveTests(unittest.TestCase):
    @patch.dict(os.environ, {"TUNELIO_API_KEY": "tnl_test"}, clear=False)
    @patch("auto_subtitle.youtube_external_download._request_json")
    def test_resolve_tunelio_returns_tunnel_url(self, mock_request) -> None:
        def side_effect(url, **kwargs):
            if "/info?" in url:
                return {
                    "title": "Me at the zoo",
                    "duration_seconds": 19,
                    "formats": [{"quality": "240p"}, {"quality": "144p"}],
                }
            return {
                "url": "https://tunelio.dev/tunnel/signed",
                "filename": "sample.mp4",
                "quality": "240p",
                "status": "ok",
                "file_size": 12345,
            }

        mock_request.side_effect = side_effect
        result = resolve_tunelio("https://youtu.be/jNQXAC9IVRw", quality="720p")
        self.assertEqual(result.provider, "tunelio")
        self.assertEqual(result.download_url, "https://tunelio.dev/tunnel/signed")
        self.assertEqual(result.file_size_bytes, 12345)
        self.assertEqual(result.duration_seconds, 19)


class ProbeProviderTests(unittest.TestCase):
    @patch.dict(os.environ, {"CAPTAPI_API_KEY_1": "capt_live_test"}, clear=False)
    @patch("auto_subtitle.youtube_external_download.probe_download_head")
    @patch("auto_subtitle.youtube_external_download.resolve_captapi")
    def test_probe_provider_success(
        self,
        mock_resolve,
        mock_probe_head,
    ) -> None:
        from auto_subtitle.youtube_external_download import ExternalResolveResult

        mock_resolve.return_value = ExternalResolveResult(
            provider="captapi",
            youtube_url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
            download_url="https://example.com/file.mp4",
            title="Me at the zoo",
            duration_seconds=19,
            file_size_bytes=None,
            cached=False,
            credits_used=3,
            raw={},
        )
        mock_probe_head.return_value = 65536

        probe = probe_provider("captapi", "https://www.youtube.com/watch?v=jNQXAC9IVRw")
        self.assertTrue(probe.ok)
        self.assertEqual(probe.stage, "done")
        self.assertEqual(probe.bytes_downloaded, 65536)

    @patch.dict(os.environ, {}, clear=True)
    def test_probe_provider_missing_key(self) -> None:
        os.environ.pop("CAPTAPI_API_KEY", None)
        os.environ.pop("CAPTAPI_API_KEY_1", None)
        probe = probe_provider("captapi", "https://www.youtube.com/watch?v=jNQXAC9IVRw")
        self.assertFalse(probe.ok)
        self.assertEqual(probe.stage, "resolve")
        self.assertIn("CAPTAPI_API_KEY_1..4", probe.error or "")

    @patch.dict(
        os.environ,
        {
            "VIDEO_DOWNLOAD_API_KEY_1": "vda_test",
            "TUNELIO_API_KEY": "tnl_test",
            "CAPTAPI_API_KEY_1": "capt_test",
        },
        clear=False,
    )
    def test_provider_chain_order(self) -> None:
        self.assertEqual(
            youtube_external_provider_chain(),
            ["video-download-api", "tunelio", "captapi"],
        )


if __name__ == "__main__":
    unittest.main()
