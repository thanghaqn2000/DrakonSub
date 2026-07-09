#!/usr/bin/env python3
"""Probe Saydi TTS connectivity for DrakonSub voiceover Phase 0."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.config import load_env  # noqa: E402

DEFAULT_PROBE_TEXT = "Xin chào, đây là bài kiểm tra giọng thuyết minh cho DrakonSub."
DEFAULT_OUTPUT_DIR = ROOT / "voiceover_probe_output"


def classify_saydi_error(detail: str, http_code: Optional[int] = None) -> str:
    text = (detail or "").lower()
    if http_code in (401, 403) or "unauthorized" in text or "forbidden" in text:
        return "auth_error"
    if http_code == 402 or "quota" in text or "credit" in text or "payment" in text:
        return "quota_error"
    if http_code == 422 or "validation" in text:
        return "invalid_response"
    if (
        "timeout" in text
        or "timed out" in text
        or "name or service not known" in text
        or "connection" in text
        or "network" in text
    ):
        return "network_error"
    return "unknown"


def _probe_config() -> Dict[str, Any]:
    load_env()
    return {
        "api_url": (os.getenv("SAYDI_TTS_API_URL") or "https://api.voice.saydi.ai/tts").strip(),
        "token": (os.getenv("SAYDI_TTS_API_TOKEN") or "").strip(),
        "sample": (
            os.getenv("SAYDI_TTS_SAMPLE") or "ng-c-huy-n-2-0-69140efab3d5d05406bafb22"
        ).strip(),
        "output_format": (os.getenv("SAYDI_TTS_OUTPUT_FORMAT") or "wav").strip().lower(),
        "timeout_seconds": int(os.getenv("SAYDI_TTS_TIMEOUT_SECONDS") or "120"),
        "lang": (os.getenv("SAYDI_TTS_LANG") or "vi").strip(),
    }


def probe_saydi_tts(
    *,
    text: str = DEFAULT_PROBE_TEXT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
) -> Dict[str, Any]:
    cfg = _probe_config()
    result: Dict[str, Any] = {
        "configured": bool(cfg["token"]),
        "available": False,
        "status": "not_configured",
        "output_file": "",
        "output_size_bytes": 0,
        "content_type": "",
        "duration_seconds": None,
        "error": "",
    }

    if not cfg["token"]:
        return result

    if dry_run:
        result["status"] = "dry_run"
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    ext = cfg["output_format"] if cfg["output_format"] in {"wav", "mp3", "flac", "ogg"} else "wav"
    output_path = output_dir / f"saydi_probe.{ext}"

    payload = {
        "text": text,
        "sample": cfg["sample"],
        "output_format": ext,
        "lang": cfg["lang"],
    }
    request = urllib.request.Request(
        cfg["api_url"],
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['token']}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=cfg["timeout_seconds"]) as response:
            audio = response.read()
            content_type = str(response.headers.get("Content-Type") or "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        status = classify_saydi_error(detail, exc.code)
        result["status"] = status
        result["error"] = f"HTTP {exc.code}: {detail[:300]}"
        return result
    except Exception as exc:
        result["status"] = classify_saydi_error(str(exc))
        result["error"] = str(exc)[:300]
        return result

    if not audio:
        result["status"] = "invalid_response"
        result["error"] = "Saydi returned empty audio body"
        return result

    output_path.write_bytes(audio)
    result["available"] = True
    result["status"] = "ok"
    result["output_file"] = str(output_path)
    result["output_size_bytes"] = len(audio)
    result["content_type"] = content_type
    result["duration_seconds"] = _probe_audio_duration(output_path)
    return result


def _probe_audio_duration(path: Path) -> Optional[float]:
    try:
        import subprocess

        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return float(proc.stdout.strip())
    except Exception:
        return None


def _print_report(result: Dict[str, Any]) -> None:
    print(f"configured={str(result['configured']).lower()}")
    print(f"available={str(result['available']).lower()}")
    print(f"status={result['status']}")
    if result.get("output_file"):
        print(f"output_file={result['output_file']}")
    if result.get("output_size_bytes"):
        print(f"output_size_bytes={result['output_size_bytes']}")
    if result.get("content_type"):
        print(f"content_type={result['content_type']}")
    if result.get("duration_seconds") is not None:
        print(f"duration_seconds={result['duration_seconds']}")
    if result.get("error"):
        print(f"error={result['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_PROBE_TEXT, help="Vietnamese probe text")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated probe audio",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check env configuration without calling Saydi",
    )
    args = parser.parse_args()

    result = probe_saydi_tts(
        text=args.text,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
    )
    _print_report(result)

    if not result["configured"]:
        return 2
    if result["status"] == "dry_run":
        return 0
    return 0 if result["available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
