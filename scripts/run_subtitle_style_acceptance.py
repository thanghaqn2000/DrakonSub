#!/usr/bin/env python3
"""Automated render acceptance for Subtitle Style Controls (real video output)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_subtitle.subtitle_renderer import (  # noqa: E402
    SubtitleRenderStyle,
    _load_font,
    burn_subtitles,
)

SRT = """1
00:00:00,500 --> 00:00:01,800
STYLE TEST 123
"""


def _write_srt(path: Path) -> None:
    path.write_text(SRT, encoding="utf-8")


def _extract_frame(video: Path, out_png: Path, t: float = 1.0) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(out_png),
        ],
        check=True,
        capture_output=True,
    )


def _mean_rgb_alpha(png: Path, box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    from PIL import Image

    x0, y0, x1, y1 = box
    with Image.open(png) as img:
        rgba = img.convert("RGBA")
        pixels = [rgba.getpixel((x, y)) for y in range(y0, y1) for x in range(x0, x1)]
    if not pixels:
        return 0.0, 0.0, 0.0, 0.0
    n = len(pixels)
    return (
        sum(p[0] for p in pixels) / n,
        sum(p[1] for p in pixels) / n,
        sum(p[2] for p in pixels) / n,
        sum(p[3] for p in pixels) / n,
    )


def _render(layout: dict, video_in: Path, srt: Path, out: Path) -> None:
    style = SubtitleRenderStyle.from_dict(layout)
    style.reference_height = 480
    burn_subtitles(str(video_in), str(srt), str(out), style)


def main() -> int:
    artifacts = ROOT / "artifacts" / "subtitle_style_acceptance"
    artifacts.mkdir(parents=True, exist_ok=True)

    video_in = Path(tempfile.gettempdir()) / "drakonsub_style_test.mp4"
    if not video_in.is_file():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=#3366cc:s=640x480:d=2",
                "-pix_fmt",
                "yuv420p",
                str(video_in),
            ],
            check=True,
            capture_output=True,
        )

    srt = artifacts / "test.srt"
    _write_srt(srt)

    base_layout = {
        "mode": "rounded",
        "x_ratio": 0.5,
        "y_ratio": 0.78,
        "width_ratio": 0.86,
        "font_size": 40,
        "text_color": "#9333EA",
        "background_color": "#FFFFFF",
        "background_opacity": 1.0,
        "background_visible": True,
        "border_radius": 18,
        "padding_x": 28,
        "padding_y": 16,
        "font_family": "arial_bold",
    }

    results: dict = {"video_in": str(video_in), "artifacts_dir": str(artifacts)}

    # A — solid default
    out_a = artifacts / "A_solid_default.mp4"
    _render(base_layout, video_in, srt, out_a)
    frame_a = artifacts / "A_frame.png"
    _extract_frame(out_a, frame_a)
    # subtitle box top-left corner (avoid purple text pixels)
    solid_stats = _mean_rgb_alpha(frame_a, (135, 325, 175, 345))
    results["A_solid_background"] = {
        "mean_rgba": solid_stats,
        "pass": solid_stats[3] > 250 and solid_stats[0] > 240 and solid_stats[1] > 240 and solid_stats[2] > 240,
    }

    # B — background OFF
    layout_off = {**base_layout, "background_visible": False}
    out_b = artifacts / "B_bg_off.mp4"
    _render(layout_off, video_in, srt, out_b)
    frame_b = artifacts / "B_frame.png"
    _extract_frame(out_b, frame_b)
    off_stats = _mean_rgb_alpha(frame_b, (200, 330, 440, 420))
    results["B_background_off"] = {
        "mean_rgba": off_stats,
        "pass": off_stats[2] > 100 and off_stats[0] < 200,  # blue video bleeds through
    }

    # C — background ON again
    out_c = artifacts / "C_bg_on.mp4"
    _render(base_layout, video_in, srt, out_c)
    frame_c = artifacts / "C_frame.png"
    _extract_frame(out_c, frame_c)
    on_stats = _mean_rgb_alpha(frame_c, (135, 325, 175, 345))
    results["C_background_on"] = {
        "mean_rgba": on_stats,
        "pass": on_stats[3] > 250 and on_stats[0] > 240 and on_stats[1] > 240 and on_stats[2] > 240,
    }

    # D/E — fonts
    for key, font, label in (
        ("D_comfortaa", "comfortaa", "Comfortaa"),
        ("E_montserrat", "montserrat_alternates", "Montserrat Alternates"),
    ):
        layout_font = {**base_layout, "font_family": font}
        out = artifacts / f"{key}.mp4"
        _render(layout_font, video_in, srt, out)
        frame = artifacts / f"{key}_frame.png"
        _extract_frame(out, frame)
        font_obj, font_path = _load_font(40, font)
        results[key] = {
            "font_family": font,
            "font_path": font_path,
            "font_file_exists": bool(font_path and os.path.isfile(font_path)),
            "output": str(out),
            "pass": bool(font_path and os.path.isfile(font_path)),
        }

    # Compare font renders differ from arial
    frame_arial = artifacts / "A_frame.png"
    frame_d = artifacts / "D_comfortaa_frame.png"
    from PIL import Image
    import hashlib

    def _hash(path: Path) -> str:
        with Image.open(path) as img:
            return hashlib.md5(img.tobytes()).hexdigest()

    results["font_visual_diff"] = {
        "arial_vs_comfortaa_differ": _hash(frame_arial) != _hash(frame_d),
    }
    results["D_comfortaa"]["visual_diff_from_arial"] = results["font_visual_diff"][
        "arial_vs_comfortaa_differ"
    ]
    results["D_comfortaa"]["pass"] = (
        results["D_comfortaa"]["pass"] and results["font_visual_diff"]["arial_vs_comfortaa_differ"]
    )

    frame_e = artifacts / "E_montserrat_frame.png"
    results["E_montserrat"]["visual_diff_from_arial"] = _hash(frame_arial) != _hash(frame_e)
    results["E_montserrat"]["pass"] = (
        results["E_montserrat"]["pass"]
        and results["E_montserrat"]["visual_diff_from_arial"]
    )

    all_pass = all(
        results[k].get("pass")
        for k in (
            "A_solid_background",
            "B_background_off",
            "C_background_on",
            "D_comfortaa",
            "E_montserrat",
        )
    )
    results["all_pass"] = all_pass

    report_path = artifacts / "acceptance_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
