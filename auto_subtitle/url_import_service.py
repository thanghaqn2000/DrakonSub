"""Safe public URL video import for YouTube and Facebook."""

from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

MAX_DURATION_SECONDS = 30 * 60
MAX_FILE_BYTES = 500 * 1024 * 1024

YOUTUBE_HOSTS = frozenset(
    {"www.youtube.com", "youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
)
FACEBOOK_HOSTS = frozenset(
    {
        "www.facebook.com",
        "facebook.com",
        "m.facebook.com",
        "web.facebook.com",
        "fb.watch",
        "www.fb.watch",
    }
)

FACEBOOK_PATH_HINTS = ("/videos/", "/video/", "/reel/", "/reels/", "/watch", "/share/")


PROVIDER_MISMATCH_MESSAGE = "Link không khớp với nguồn đã chọn."


class UrlImportError(ValueError):
    """User-facing URL import failure."""


def _normalize_host(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    return host


def _host_is_blocked(host: str) -> bool:
    if not host:
        return True
    if host in {"localhost", "0.0.0.0"} or host.endswith(".local"):
        return True
    if host == "127.0.0.1" or host.startswith("127."):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return bool(
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        )
    except ValueError:
        return False


def detect_provider(url: str) -> str:
    """Return 'youtube' or 'facebook' for a supported public video URL."""
    host = _normalize_host(url)
    if host in YOUTUBE_HOSTS:
        return "youtube"
    if host in FACEBOOK_HOSTS:
        path = (urlparse(url.strip()).path or "").lower()
        if host.startswith("fb.watch") or host == "fb.watch":
            return "facebook"
        if any(hint in path for hint in FACEBOOK_PATH_HINTS):
            return "facebook"
        if re.search(r"/reel[s]?/", path):
            return "facebook"
        raise UrlImportError(
            "Link không được hỗ trợ. Vui lòng dùng link YouTube hoặc Facebook công khai."
        )
    raise UrlImportError(
        "Link không được hỗ trợ. Vui lòng dùng link YouTube hoặc Facebook công khai."
    )


def validate_video_url(url: str) -> str:
    """Validate URL scheme, host safety, and provider support. Returns stripped URL."""
    raw = (url or "").strip()
    if not raw:
        raise UrlImportError("Vui lòng nhập link video.")

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise UrlImportError(
            "Link không được hỗ trợ. Vui lòng dùng link YouTube hoặc Facebook công khai."
        )

    host = _normalize_host(raw)
    if _host_is_blocked(host):
        raise UrlImportError(
            "Link không được hỗ trợ. Vui lòng dùng link YouTube hoặc Facebook công khai."
        )

    detect_provider(raw)
    return raw


def validate_url_with_selected_provider(url: str, selected_provider: str) -> tuple[str, str]:
    """Validate URL and ensure it matches the user-selected source (youtube/facebook)."""
    safe_url = validate_video_url(url)
    detected = detect_provider(safe_url)
    selected = (selected_provider or "youtube").strip().lower()
    if selected not in ("youtube", "facebook"):
        raise UrlImportError("Nguồn video không hợp lệ.")
    if detected != selected:
        raise UrlImportError(PROVIDER_MISMATCH_MESSAGE)
    return safe_url, detected


def _map_download_error(exc: Exception) -> UrlImportError:
    text = str(exc).lower()
    if "private" in text or "login" in text or "sign in" in text or "cookies" in text:
        return UrlImportError(
            "Không thể tải video này. Video có thể riêng tư, bị giới hạn hoặc cần đăng nhập."
        )
    if "unsupported url" in text or "unsupported" in text:
        return UrlImportError(
            "Link không được hỗ trợ. Vui lòng dùng link YouTube hoặc Facebook công khai."
        )
    return UrlImportError(
        "Tải video thất bại. Vui lòng thử lại hoặc tải file video trực tiếp."
    )


def _resolve_downloaded_path(output_dir: Path, requested: Path) -> Path:
    if requested.is_file():
        return requested
    candidates = sorted(output_dir.glob("input.*"))
    if not candidates:
        raise UrlImportError(
            "Tải video thất bại. Vui lòng thử lại hoặc tải file video trực tiếp."
        )
    return candidates[0]


def download_video_from_url(
    url: str,
    output_dir: str | Path,
    *,
    output_filename: str = "input.mp4",
) -> Dict[str, Any]:
    """
    Download a public video into *output_dir* using yt-dlp.

    Returns metadata dict with local path and provider.
    """
    import yt_dlp

    safe_url = validate_video_url(url)
    provider = detect_provider(safe_url)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / output_filename
    if target.exists():
        target.unlink()

    ydl_opts = {
        "format": "bv*+ba/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "input.%(ext)s"),
        "noplaylist": True,
        "max_filesize": MAX_FILE_BYTES,
        "socket_timeout": 30,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": False,
        # YouTube often 403s default web client; android+web fallback is yt-dlp recommended.
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(safe_url, download=False)
            duration = info.get("duration") or 0
            if duration and duration > MAX_DURATION_SECONDS:
                raise UrlImportError(
                    "Video quá dài hoặc quá nặng so với giới hạn hiện tại."
                )
            ydl.download([safe_url])
    except UrlImportError:
        raise
    except Exception as exc:
        raise _map_download_error(exc) from exc

    downloaded = _resolve_downloaded_path(out_dir, target)
    if downloaded.suffix.lower() != ".mp4":
        final_path = out_dir / "input.mp4"
        if final_path.exists():
            final_path.unlink()
        downloaded.rename(final_path)
        downloaded = final_path

    size = downloaded.stat().st_size
    if size > MAX_FILE_BYTES:
        downloaded.unlink(missing_ok=True)
        raise UrlImportError(
            "Video quá dài hoặc quá nặng so với giới hạn hiện tại."
        )

    title = ""
    try:
        title = str(info.get("title") or "").strip()
    except Exception:
        pass

    return {
        "path": str(downloaded),
        "provider": provider,
        "title": title,
        "duration": duration if duration else None,
        "filesize": size,
    }
