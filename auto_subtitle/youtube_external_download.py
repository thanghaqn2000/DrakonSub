"""Resolve YouTube downloads via third-party APIs."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class ExternalResolveResult:
    provider: str
    youtube_url: str
    download_url: str
    title: Optional[str]
    duration_seconds: Optional[int]
    file_size_bytes: Optional[int]
    cached: Optional[bool]
    credits_used: Optional[int]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class ExternalDownloadProbe:
    provider: str
    youtube_url: str
    ok: bool
    stage: str
    error: Optional[str] = None
    resolve_ms: Optional[int] = None
    download_ms: Optional[int] = None
    bytes_downloaded: Optional[int] = None
    title: Optional[str] = None
    duration_seconds: Optional[int] = None


class ExternalDownloadError(RuntimeError):
    """Third-party YouTube download API failure."""


class ExternalCreditsExhaustedError(ExternalDownloadError):
    """Provider has no credits left; caller may try the next provider."""


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _assert_safe_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ExternalDownloadError(f"Unsafe external URL scheme: {parsed.scheme or '<empty>'}")
    return url


def _is_credits_exhausted(http_code: int, detail: str, payload: Optional[Dict[str, Any]] = None) -> bool:
    text = detail.lower()
    if http_code == 402:
        return True
    if payload:
        error_code = str(payload.get("error") or payload.get("code") or "").lower()
        message = str(payload.get("message") or payload.get("detail") or "").lower()
        if error_code in ("insufficient_credits", "payment_required"):
            return True
        if "insufficient" in message and "credit" in message:
            return True
        if "out of credits" in message or "not enough credits" in message:
            return True
    markers = (
        "insufficient_credits",
        "insufficient credits",
        "out of credits",
        "not enough credits",
        "no credits",
    )
    return any(marker in text for marker in markers)


def _request_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    safe_url = _assert_safe_url(url)
    req = urllib.request.Request(
        safe_url,
        headers={
            "Accept": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
            **(headers or {}),
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        payload: Optional[Dict[str, Any]] = None
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = None
        if _is_credits_exhausted(exc.code, detail, payload):
            raise ExternalCreditsExhaustedError(
                f"HTTP {exc.code} credits exhausted from {urllib.parse.urlparse(url).netloc}: {detail[:400]}"
            ) from exc
        raise ExternalDownloadError(
            f"HTTP {exc.code} from {urllib.parse.urlparse(url).netloc}: {detail[:400]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ExternalDownloadError(f"Network error: {exc}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ExternalDownloadError(f"Invalid JSON response: {body[:200]}") from exc
    if not isinstance(payload, dict):
        raise ExternalDownloadError("Expected JSON object response")
    return payload


def _captapi_keys() -> List[str]:
    keys: List[str] = []
    for index in range(1, 5):
        key = (os.getenv(f"CAPTAPI_API_KEY_{index}") or "").strip()
        if key:
            keys.append(key)

    legacy = (os.getenv("CAPTAPI_API_KEY") or "").strip()
    if legacy and legacy not in keys:
        keys.append(legacy)
    return keys


def _video_download_api_keys() -> List[str]:
    keys: List[str] = []
    for index in range(1, 5):
        key = (os.getenv(f"VIDEO_DOWNLOAD_API_KEY_{index}") or "").strip()
        if key:
            keys.append(key)

    # Backward compatibility for older local setups.
    legacy = (os.getenv("VIDEO_DOWNLOAD_API_KEY") or "").strip()
    if legacy and legacy not in keys:
        keys.append(legacy)
    return keys


def _tunelio_key() -> Optional[str]:
    return (os.getenv("TUNELIO_API_KEY") or "").strip() or None


def resolve_video_download_api(
    youtube_url: str,
    *,
    format_code: str = "720",
    worker_prepare: bool = True,
    poll_timeout_seconds: int = 90,
) -> ExternalResolveResult:
    api_keys = _video_download_api_keys()
    if not api_keys:
        raise ExternalDownloadError("VIDEO_DOWNLOAD_API_KEY_1..4 are not set")

    last_credit_error: Optional[ExternalCreditsExhaustedError] = None
    last_error: Optional[ExternalDownloadError] = None

    for api_key in api_keys:
        query = urllib.parse.urlencode(
            {
                "format": format_code,
                "url": youtube_url,
                "apikey": api_key,
                "worker_prepare": "1" if worker_prepare else "0",
            }
        )
        try:
            payload = _request_json(
                f"https://p.savenow.to/api/v2/download?{query}",
                timeout=90,
            )
        except ExternalCreditsExhaustedError as exc:
            last_credit_error = exc
            continue
        except ExternalDownloadError as exc:
            last_error = exc
            break

        if not payload.get("success"):
            last_error = ExternalDownloadError(f"Video Download API error: {payload}")
            break

        started = time.time()
        while not payload.get("url"):
            progress_url = _assert_safe_url(str(payload.get("progress_url") or "").strip())
            if not progress_url:
                raise ExternalDownloadError(
                    f"Video Download API response missing url/progress_url: {payload}"
                )
            if time.time() - started > poll_timeout_seconds:
                raise ExternalDownloadError("Video Download API progress polling timed out")
            time.sleep(2)
            try:
                payload = _request_json(progress_url, timeout=60)
            except ExternalCreditsExhaustedError as exc:
                last_credit_error = exc
                payload = {}
                break

            if _is_credits_exhausted(0, json.dumps(payload), payload):
                last_credit_error = ExternalCreditsExhaustedError(
                    f"Video Download API credits exhausted: {payload}"
                )
                payload = {}
                break

        if not payload:
            continue

        download_url = _assert_safe_url(str(payload.get("url") or "").strip())
        if not download_url:
            last_error = ExternalDownloadError(
                f"Video Download API missing download url: {payload}"
            )
            break

        return ExternalResolveResult(
            provider="video-download-api",
            youtube_url=youtube_url,
            download_url=download_url,
            title=(payload.get("title") or payload.get("filename") or None),
            duration_seconds=None,
            file_size_bytes=None,
            cached=None,
            credits_used=None,
            raw=payload,
        )

    if last_credit_error is not None:
        raise last_credit_error
    if last_error is not None:
        raise last_error
    raise ExternalDownloadError("Video Download API request failed")


def resolve_captapi(youtube_url: str) -> ExternalResolveResult:
    api_keys = _captapi_keys()
    if not api_keys:
        raise ExternalDownloadError("CAPTAPI_API_KEY_1..4 are not set")

    last_credit_error: Optional[ExternalCreditsExhaustedError] = None
    last_error: Optional[ExternalDownloadError] = None

    for api_key in api_keys:
        query = urllib.parse.urlencode({"url": youtube_url})
        try:
            payload = _request_json(
                f"https://api.captapi.com/v1/youtube/video-download?{query}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        except ExternalCreditsExhaustedError as exc:
            last_credit_error = exc
            continue
        except ExternalDownloadError as exc:
            last_error = exc
            break

        if not payload.get("success"):
            last_error = ExternalDownloadError(f"Captapi error: {payload}")
            break

        data = payload.get("data") or {}
        download_url = data.get("downloadUrl") or data.get("url")
        if not download_url:
            last_error = ExternalDownloadError(
                f"Captapi response missing downloadUrl: {payload}"
            )
            break

        duration_ms = _safe_int(data.get("approxDurationMs") or data.get("durationMs"))
        duration_seconds = int(duration_ms / 1000) if duration_ms is not None else None
        download_url = _assert_safe_url(str(download_url))

        return ExternalResolveResult(
            provider="captapi",
            youtube_url=youtube_url,
            download_url=str(download_url),
            title=(data.get("title") or None),
            duration_seconds=duration_seconds,
            file_size_bytes=data.get("sizeBytes"),
            cached=payload.get("cached"),
            credits_used=payload.get("creditsUsed"),
            raw=payload,
        )

    if last_credit_error is not None:
        raise last_credit_error
    if last_error is not None:
        raise last_error
    raise ExternalDownloadError("Captapi request failed")


def _tunelio_quality_preference() -> List[str]:
    return ["720p", "480p", "360p", "240p", "144p"]


def info_tunelio(youtube_url: str) -> Dict[str, Any]:
    api_key = _tunelio_key()
    if not api_key:
        raise ExternalDownloadError("TUNELIO_API_KEY is not set")
    query = urllib.parse.urlencode({"url": youtube_url})
    return _request_json(
        f"https://tunelio.dev/info?{query}",
        headers={"Authorization": f"Bearer {api_key}"},
    )


def _pick_tunelio_quality(info: Dict[str, Any], preferred: str = "720p") -> str:
    formats = info.get("formats") or []
    available = {
        str(item.get("quality") or "").lower()
        for item in formats
        if isinstance(item, dict) and item.get("quality")
    }
    if preferred.lower() in available:
        return preferred.lower()
    for quality in _tunelio_quality_preference():
        if quality in available:
            return quality
    if available:
        def _quality_rank(value: str) -> int:
            digits = "".join(ch for ch in value if ch.isdigit())
            return int(digits) if digits else -1

        return max(available, key=_quality_rank)
    raise ExternalDownloadError(f"Tunelio info has no video formats: {info}")


def resolve_tunelio(youtube_url: str, *, quality: str = "720p") -> ExternalResolveResult:
    api_key = _tunelio_key()
    if not api_key:
        raise ExternalDownloadError("TUNELIO_API_KEY is not set")

    info = info_tunelio(youtube_url)
    chosen_quality = _pick_tunelio_quality(info, preferred=quality)

    query = urllib.parse.urlencode({"url": youtube_url, "quality": chosen_quality})
    payload = _request_json(
        f"https://tunelio.dev/create?{query}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if payload.get("error"):
        error_code = str(payload.get("error"))
        if _is_credits_exhausted(0, error_code, payload):
            raise ExternalCreditsExhaustedError(
                f"Tunelio credits exhausted: {payload.get('message')}"
            )
        raise ExternalDownloadError(
            f"Tunelio error {error_code}: {payload.get('message')}"
        )

    download_url = payload.get("url")
    status = str(payload.get("status") or "").lower()
    if not download_url or status not in ("ok", ""):
        raise ExternalDownloadError(f"Tunelio response missing download url: {payload}")

    return ExternalResolveResult(
        provider="tunelio",
        youtube_url=youtube_url,
        download_url=_assert_safe_url(str(download_url)),
        title=(payload.get("filename") or info.get("title") or None),
        duration_seconds=_safe_int(info.get("duration_seconds")),
        file_size_bytes=payload.get("file_size"),
        cached=None,
        credits_used=None,
        raw=payload,
    )


def probe_download_head(
    download_url: str,
    *,
    max_bytes: int = 1024 * 1024,
    use_range: bool = True,
) -> int:
    """Download up to max_bytes to verify the resolved URL is reachable."""
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if use_range:
        headers["Range"] = f"bytes=0-{max(0, max_bytes - 1)}"
    req = urllib.request.Request(
        download_url,
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            chunk = resp.read(max_bytes)
    except urllib.error.HTTPError as exc:
        if exc.code not in (200, 206):
            detail = exc.read().decode("utf-8", errors="replace")
            raise ExternalDownloadError(
                f"Download probe HTTP {exc.code}: {detail[:300]}"
            ) from exc
        chunk = exc.read(max_bytes)
    except urllib.error.URLError as exc:
        raise ExternalDownloadError(f"Download probe network error: {exc}") from exc

    if not chunk:
        raise ExternalDownloadError("Download probe returned empty body")
    return len(chunk)


def probe_provider(
    provider: str,
    youtube_url: str,
    *,
    tunelio_quality: str = "720p",
    max_probe_bytes: int = 1024 * 1024,
) -> ExternalDownloadProbe:
    started = time.perf_counter()
    try:
        if provider == "video-download-api":
            resolved = resolve_video_download_api(youtube_url)
        elif provider == "captapi":
            resolved = resolve_captapi(youtube_url)
        elif provider == "tunelio":
            resolved = resolve_tunelio(youtube_url, quality=tunelio_quality)
        else:
            raise ExternalDownloadError(f"Unknown provider: {provider}")
    except ExternalDownloadError as exc:
        return ExternalDownloadProbe(
            provider=provider,
            youtube_url=youtube_url,
            ok=False,
            stage="resolve",
            error=str(exc),
            resolve_ms=int((time.perf_counter() - started) * 1000),
        )

    resolve_ms = int((time.perf_counter() - started) * 1000)
    dl_started = time.perf_counter()
    try:
        nbytes = probe_download_head(
            resolved.download_url,
            max_bytes=max_probe_bytes,
            use_range=(provider != "tunelio"),
        )
    except ExternalDownloadError as exc:
        return ExternalDownloadProbe(
            provider=provider,
            youtube_url=youtube_url,
            ok=False,
            stage="download_probe",
            error=str(exc),
            resolve_ms=resolve_ms,
            title=resolved.title,
            duration_seconds=resolved.duration_seconds,
        )

    return ExternalDownloadProbe(
        provider=provider,
        youtube_url=youtube_url,
        ok=True,
        stage="done",
        resolve_ms=resolve_ms,
        download_ms=int((time.perf_counter() - dl_started) * 1000),
        bytes_downloaded=nbytes,
        title=resolved.title,
        duration_seconds=resolved.duration_seconds,
    )


def youtube_external_provider_chain() -> List[str]:
    """Default YouTube download order: Video Download API -> Tunelio -> Captapi."""
    chain: List[str] = []
    if _video_download_api_keys():
        chain.append("video-download-api")
    if _tunelio_key():
        chain.append("tunelio")
    if _captapi_keys():
        chain.append("captapi")
    return chain


def available_providers() -> List[str]:
    return youtube_external_provider_chain()


def download_youtube_via_external_api(
    youtube_url: str,
    output_dir: str | Path,
    *,
    provider: str,
    output_filename: str = "input.mp4",
    tunelio_quality: str = "720p",
) -> Dict[str, Any]:
    """Resolve via external API and save the file into output_dir."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / output_filename

    if provider == "video-download-api":
        resolved = resolve_video_download_api(youtube_url)
    elif provider == "captapi":
        resolved = resolve_captapi(youtube_url)
    elif provider == "tunelio":
        resolved = resolve_tunelio(youtube_url, quality=tunelio_quality)
    else:
        raise ExternalDownloadError(f"Unknown provider: {provider}")

    req = urllib.request.Request(
        resolved.download_url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, target.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ExternalDownloadError(
            f"External download HTTP {exc.code}: {detail[:300]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ExternalDownloadError(f"External download network error: {exc}") from exc

    size = target.stat().st_size
    if size <= 0:
        target.unlink(missing_ok=True)
        raise ExternalDownloadError("External download produced empty file")

    return {
        "path": str(target),
        "provider": "youtube",
        "title": resolved.title or "",
        "duration": resolved.duration_seconds,
        "filesize": size,
        "external_provider": provider,
    }
