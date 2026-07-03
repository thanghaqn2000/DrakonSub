"""Safe public URL video import for YouTube and Facebook."""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

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

FACEBOOK_UNSUPPORTED_PATH_RE = re.compile(
    r"(?:^|/)"
    r"(?:profile\.php|posts?/|photo/?|photos/|groups/|events/|marketplace/|people/)"
    r"(?:/|$|\?)",
    re.IGNORECASE,
)

PROVIDER_MISMATCH_MESSAGE = "Link không khớp với nguồn đã chọn."
FACEBOOK_UNSUPPORTED_MESSAGE = (
    "Link Facebook này chưa được hỗ trợ. Vui lòng dùng link video, reel hoặc fb.watch công khai."
)
FACEBOOK_DOWNLOAD_FAIL_MESSAGE = (
    "Không thể tải video Facebook này. Video có thể riêng tư, bị giới hạn hoặc cần đăng nhập."
)
GENERIC_UNSUPPORTED_MESSAGE = (
    "Link không được hỗ trợ. Vui lòng dùng link YouTube hoặc Facebook công khai."
)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class UrlImportError(ValueError):
    """User-facing URL import failure."""


def _normalize_host(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    return host


def _is_fb_watch_host(host: str) -> bool:
    return host in {"fb.watch", "www.fb.watch"} or host.endswith(".fb.watch")


def _is_facebook_video_url(url: str) -> bool:
    """Return True when URL looks like a supported public Facebook video/reel/watch link."""
    parsed = urlparse(url.strip())
    host = _normalize_host(url)
    if host not in FACEBOOK_HOSTS and not _is_fb_watch_host(host):
        return False

    if _is_fb_watch_host(host):
        return bool((parsed.path or "").strip("/"))

    path = (parsed.path or "").lower()
    if FACEBOOK_UNSUPPORTED_PATH_RE.search(path):
        return False

    if re.search(r"/videos/\d+", path):
        return True
    if re.search(r"/reel[s]?/\d+", path):
        return True
    if re.search(r"/share/[vr]/[^/]+", path):
        return True
    if "/video/" in path or path.endswith("/video"):
        return True
    if path.rstrip("/") == "/watch":
        query = parse_qs(parsed.query or "")
        return bool(query.get("v", [""])[0].isdigit())
    return False


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
    if host in FACEBOOK_HOSTS or _is_fb_watch_host(host):
        if _is_facebook_video_url(url):
            return "facebook"
        raise UrlImportError(FACEBOOK_UNSUPPORTED_MESSAGE)
    raise UrlImportError(GENERIC_UNSUPPORTED_MESSAGE)


def validate_video_url(url: str) -> str:
    """Validate URL scheme, host safety, and provider support. Returns stripped URL."""
    raw = (url or "").strip()
    if not raw:
        raise UrlImportError("Vui lòng nhập link video.")

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise UrlImportError(GENERIC_UNSUPPORTED_MESSAGE)

    host = _normalize_host(raw)
    if _host_is_blocked(host):
        raise UrlImportError(GENERIC_UNSUPPORTED_MESSAGE)

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


def _map_download_error(exc: Exception, provider: Optional[str] = None) -> UrlImportError:
    text = str(exc).lower()
    restricted = any(
        token in text
        for token in (
            "private",
            "login",
            "sign in",
            "cookies",
            "restricted",
            "not available",
            "confirm your age",
            "age-restricted",
        )
    )
    if provider == "facebook":
        if restricted:
            return UrlImportError(FACEBOOK_DOWNLOAD_FAIL_MESSAGE)
        if "unsupported url" in text or "no video" in text:
            return UrlImportError(FACEBOOK_UNSUPPORTED_MESSAGE)
        return UrlImportError(FACEBOOK_DOWNLOAD_FAIL_MESSAGE)

    if restricted:
        return UrlImportError(
            "Không thể tải video này. Video có thể riêng tư, bị giới hạn hoặc cần đăng nhập."
        )
    if "unsupported url" in text or "unsupported" in text:
        return UrlImportError(GENERIC_UNSUPPORTED_MESSAGE)
    return UrlImportError(
        "Tải video thất bại. Vui lòng thử lại hoặc tải file video trực tiếp."
    )


def cleanup_partial_downloads(output_dir: str | Path) -> None:
    """Remove incomplete downloaded files from a failed import attempt."""
    out_dir = Path(output_dir)
    if not out_dir.is_dir():
        return
    for pattern in ("input.*", "*.part", "*.ytdl"):
        for path in out_dir.glob(pattern):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _resolve_downloaded_path(output_dir: Path, requested: Path) -> Path:
    if requested.is_file():
        return requested
    candidates = sorted(output_dir.glob("input.*"))
    if not candidates:
        raise UrlImportError(
            "Tải video thất bại. Vui lòng thử lại hoặc tải file video trực tiếp."
        )
    return candidates[0]


def _build_ydl_opts(out_dir: Path, provider: str) -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "format": "bv*+ba/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "input.%(ext)s"),
        "noplaylist": True,
        "max_filesize": MAX_FILE_BYTES,
        "socket_timeout": 30,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": False,
        "http_headers": {"User-Agent": _DEFAULT_USER_AGENT},
    }
    if provider == "youtube":
        opts["extractor_args"] = {"youtube": {"player_client": ["android", "web"]}}
    return opts


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
    cleanup_partial_downloads(out_dir)

    ydl_opts = _build_ydl_opts(out_dir, provider)
    info: Dict[str, Any] = {}
    duration = 0

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(safe_url, download=False) or {}
            duration = info.get("duration") or 0
            if duration and duration > MAX_DURATION_SECONDS:
                raise UrlImportError(
                    "Video quá dài hoặc quá nặng so với giới hạn hiện tại."
                )
            ydl.download([safe_url])
    except UrlImportError:
        cleanup_partial_downloads(out_dir)
        raise
    except Exception as exc:
        cleanup_partial_downloads(out_dir)
        raise _map_download_error(exc, provider) from exc

    downloaded = _resolve_downloaded_path(out_dir, target)
    if downloaded.suffix.lower() != ".mp4":
        final_path = out_dir / "input.mp4"
        if final_path.exists():
            final_path.unlink()
        downloaded.rename(final_path)
        downloaded = final_path

    size = downloaded.stat().st_size
    if size <= 0:
        cleanup_partial_downloads(out_dir)
        raise _map_download_error(RuntimeError("empty file"), provider)
    if size > MAX_FILE_BYTES:
        cleanup_partial_downloads(out_dir)
        raise UrlImportError(
            "Video quá dài hoặc quá nặng so với giới hạn hiện tại."
        )

    title = str(info.get("title") or "").strip()

    return {
        "path": str(downloaded),
        "provider": provider,
        "title": title,
        "duration": duration if duration else None,
        "filesize": size,
    }
