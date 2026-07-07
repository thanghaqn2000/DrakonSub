"""Safe public URL video import for YouTube and Facebook."""

from __future__ import annotations

import ipaddress
import os
import platform
import re
import subprocess
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
YOUTUBE_BOT_BLOCK_MESSAGE = (
    "YouTube chặn tải từ server này. Vui lòng tải file video trực tiếp hoặc liên hệ admin cấu hình cookies."
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


def _is_youtube_video_url(url: str) -> bool:
    """Return True when URL looks like a direct YouTube video link."""
    parsed = urlparse(url.strip())
    host = _normalize_host(url)
    path = (parsed.path or "").lower()
    if host in {"youtu.be", "www.youtu.be"} and path.strip("/"):
        return True
    if path == "/watch":
        query = parse_qs(parsed.query or "")
        return bool(query.get("v", [""])[0])
    if path.startswith(("/shorts/", "/embed/")):
        return True
    return False


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
    if re.search(r"/reel[s]?/[^/?#]+", path, re.IGNORECASE):
        return True
    if re.search(r"/share/[vr]/[^/?#]+", path, re.IGNORECASE):
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
        if _is_youtube_video_url(url):
            return "youtube"
        raise UrlImportError(GENERIC_UNSUPPORTED_MESSAGE)
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
            "not a bot",
            "confirm you're not a bot",
            "the page needs to be reloaded",
        )
    )
    if provider == "youtube" and (
        "not a bot" in text
        or "sign in to confirm" in text
        or "the page needs to be reloaded" in text
    ):
        return UrlImportError(YOUTUBE_BOT_BLOCK_MESSAGE)
    if provider == "facebook":
        if restricted or "cannot parse data" in text:
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


def _probe_media_codecs(path: Path) -> tuple[Optional[str], Optional[str]]:
    import ffmpeg

    probe = ffmpeg.probe(str(path))
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video" and not video_codec:
            video_codec = stream.get("codec_name")
        elif stream.get("codec_type") == "audio" and not audio_codec:
            audio_codec = stream.get("codec_name")
    return video_codec, audio_codec


def _needs_quicktime_compatible_mp4(path: Path) -> bool:
    if path.suffix.lower() != ".mp4":
        return True
    try:
        video_codec, audio_codec = _probe_media_codecs(path)
    except Exception:
        return True
    if video_codec != "h264":
        return True
    if audio_codec and audio_codec != "aac":
        return True
    return False


def _transcode_to_quicktime_compatible_mp4(source: Path, target: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _normalize_downloaded_video(output_dir: Path, downloaded: Path) -> Path:
    final_path = output_dir / "input.mp4"
    if not _needs_quicktime_compatible_mp4(downloaded):
        if downloaded == final_path:
            return downloaded
        if final_path.exists():
            final_path.unlink()
        downloaded.rename(final_path)
        return final_path

    compat_path = output_dir / "input.compat.mp4"
    compat_path.unlink(missing_ok=True)
    try:
        _transcode_to_quicktime_compatible_mp4(downloaded, compat_path)
    except (subprocess.CalledProcessError, OSError) as exc:
        compat_path.unlink(missing_ok=True)
        raise UrlImportError(
            "Tải video thất bại. Vui lòng thử lại hoặc tải file video trực tiếp."
        ) from exc

    if final_path.exists():
        final_path.unlink()
    compat_path.rename(final_path)
    if downloaded != final_path:
        downloaded.unlink(missing_ok=True)
    return final_path


def _youtube_cookie_file() -> Optional[str]:
    raw = (os.getenv("YT_DLP_COOKIES_FILE") or "").strip()
    if not raw:
        return None
    src = Path(raw)
    if not src.is_file():
        return None
    # yt-dlp rewrites cookie files on exit; RO mounts break downloads.
    dest = Path("/tmp/drakonsub-youtube-cookies.txt")
    try:
        import shutil

        shutil.copy2(src, dest)
        return str(dest)
    except OSError:
        return str(src) if os.access(src, os.W_OK) else None


def _browser_cookie_source() -> Optional[tuple[str]]:
    system = platform.system().lower()
    home = Path.home()
    candidates = []
    if system == "darwin":
        candidates = [
            (home / "Library/Application Support/Google/Chrome", ("chrome",)),
            (home / "Library/Application Support/Chromium", ("chromium",)),
            (home / "Library/Application Support/Microsoft Edge", ("edge",)),
            (home / "Library/Safari", ("safari",)),
        ]
    elif system == "windows":
        local = Path(os.getenv("LOCALAPPDATA", ""))
        appdata = Path(os.getenv("APPDATA", ""))
        candidates = [
            (local / "Google/Chrome/User Data", ("chrome",)),
            (local / "Chromium/User Data", ("chromium",)),
            (local / "Microsoft/Edge/User Data", ("edge",)),
            (appdata / "Mozilla/Firefox/Profiles", ("firefox",)),
        ]
    else:
        candidates = [
            (home / ".config/google-chrome", ("chrome",)),
            (home / ".config/chromium", ("chromium",)),
            (home / ".mozilla/firefox", ("firefox",)),
        ]

    for path, source in candidates:
        if path.exists():
            return source
    return None


def _build_ydl_opts(
    out_dir: Path,
    provider: str,
    *,
    use_browser_cookies: bool = False,
) -> Dict[str, Any]:
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
    cookie_file = _youtube_cookie_file()
    has_server_cookie_file = bool(cookie_file)
    if cookie_file:
        opts["cookiefile"] = cookie_file
    elif provider == "youtube" and use_browser_cookies:
        browser_cookie_source = _browser_cookie_source()
        if browser_cookie_source:
            opts["cookiesfrombrowser"] = browser_cookie_source
    if provider == "youtube":
        if has_server_cookie_file:
            opts["remote_components"] = ["ejs:github"]
            opts["extractor_args"] = {
                "youtube": {
                    "player_client": ["tv", "web"],
                    "player_skip": ["webpage"],
                }
            }
        else:
            opts["extractor_args"] = {"youtube": {"player_client": ["android", "web"]}}
    return opts


def _youtube_needs_browser_cookie_retry(exc: Exception, ydl_opts: Dict[str, Any]) -> bool:
    if "cookiefile" in ydl_opts or "cookiesfrombrowser" in ydl_opts:
        return False
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "the page needs to be reloaded",
            "not a bot",
            "sign in to confirm",
            "confirm you're not a bot",
        )
    )


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
        retry_succeeded = False
        if provider == "youtube" and _youtube_needs_browser_cookie_retry(exc, ydl_opts):
            cleanup_partial_downloads(out_dir)
            retry_opts = _build_ydl_opts(
                out_dir,
                provider,
                use_browser_cookies=True,
            )
            if "cookiesfrombrowser" in retry_opts:
                try:
                    with yt_dlp.YoutubeDL(retry_opts) as ydl:
                        info = ydl.extract_info(safe_url, download=False) or {}
                        duration = info.get("duration") or 0
                        if duration and duration > MAX_DURATION_SECONDS:
                            raise UrlImportError(
                                "Video quá dài hoặc quá nặng so với giới hạn hiện tại."
                            )
                        ydl.download([safe_url])
                    retry_succeeded = True
                except UrlImportError:
                    cleanup_partial_downloads(out_dir)
                    raise
                except Exception as retry_exc:
                    cleanup_partial_downloads(out_dir)
                    raise _map_download_error(retry_exc, provider) from retry_exc
            else:
                cleanup_partial_downloads(out_dir)
                raise _map_download_error(exc, provider) from exc
        if not retry_succeeded:
            cleanup_partial_downloads(out_dir)
            raise _map_download_error(exc, provider) from exc

    downloaded = _resolve_downloaded_path(out_dir, target)
    downloaded = _normalize_downloaded_video(out_dir, downloaded)

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
