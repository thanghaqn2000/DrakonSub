#!/usr/bin/env python3
"""Benchmark third-party YouTube download APIs for DrakonSub POC."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_subtitle.youtube_external_download import (  # noqa: E402
    ExternalDownloadProbe,
    available_providers,
    probe_provider,
)

DEFAULT_URLS = [
    "https://www.youtube.com/watch?v=jNQXAC9IVRw",
    "https://www.youtube.com/watch?v=LqVZ-wT0UbM",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
]


def _format_probe(probe: ExternalDownloadProbe) -> str:
    status = "PASS" if probe.ok else f"FAIL ({probe.stage})"
    parts = [
        f"- provider={probe.provider}",
        f"url={probe.youtube_url}",
        f"status={status}",
    ]
    if probe.title:
        parts.append(f"title={probe.title!r}")
    if probe.resolve_ms is not None:
        parts.append(f"resolve_ms={probe.resolve_ms}")
    if probe.download_ms is not None:
        parts.append(f"download_ms={probe.download_ms}")
    if probe.bytes_downloaded is not None:
        parts.append(f"bytes={probe.bytes_downloaded}")
    if probe.error:
        parts.append(f"error={probe.error}")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        default="captapi,tunelio",
        help="Comma-separated providers to test (captapi,tunelio)",
    )
    parser.add_argument(
        "--urls",
        default=",".join(DEFAULT_URLS),
        help="Comma-separated YouTube URLs",
    )
    parser.add_argument(
        "--tunelio-quality",
        default="720p",
        help="Tunelio quality parameter (default: 720p)",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "debug" / "poc_external_youtube_download_report.md"),
        help="Markdown report output path",
    )
    args = parser.parse_args()

    requested = [p.strip() for p in args.providers.split(",") if p.strip()]
    configured = set(available_providers())
    missing_keys = [p for p in requested if p not in configured]
    if missing_keys:
        print("Missing API keys for:", ", ".join(missing_keys))
        print("Set env vars:")
        if "captapi" in missing_keys:
            print("  export CAPTAPI_API_KEY=capt_live_...")
        if "tunelio" in missing_keys:
            print("  export TUNELIO_API_KEY=tnl_...")
        print("\nSignup:")
        print("  Captapi: https://captapi.com/dashboard/api-keys")
        print("  Tunelio: https://tunelio.dev/")
        return 2

    providers = [p for p in requested if p in configured]
    urls = [u.strip() for u in args.urls.split(",") if u.strip()]

    probes: list[ExternalDownloadProbe] = []
    for provider in providers:
        for url in urls:
            print(f"Testing {provider} -> {url}")
            probes.append(
                probe_provider(
                    provider,
                    url,
                    tunelio_quality=args.tunelio_quality,
                )
            )

    passed = sum(1 for p in probes if p.ok)
    total = len(probes)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "# POC External YouTube Download APIs",
        "",
        f"- generated_at: {now}",
        f"- providers: {', '.join(providers)}",
        f"- urls_tested: {len(urls)}",
        f"- success_rate: {passed}/{total} ({(passed / total * 100) if total else 0:.1f}%)",
        "",
        "## Results",
        "",
    ]
    for probe in probes:
        lines.append(_format_probe(probe))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Captapi: GET https://api.captapi.com/v1/youtube/video-download",
            "- Tunelio: GET https://tunelio.dev/create",
            "- This POC only verifies resolve + first 1MB download probe.",
            "- Add keys via CAPTAPI_API_KEY / TUNELIO_API_KEY before running.",
            "",
        ]
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"Success: {passed}/{total}")
    print(f"Report: {report_path}")
    print(json.dumps([p.__dict__ for p in probes], ensure_ascii=False, indent=2))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
