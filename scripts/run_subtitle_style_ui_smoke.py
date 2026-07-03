#!/usr/bin/env python3
"""API-level UI smoke: layout save + Render lại through web endpoints."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from auto_subtitle import web  # noqa: E402

VI_SRT = """1
00:00:00,500 --> 00:00:01,800
STYLE TEST 123
"""

BASE_LAYOUT = {
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


def _ensure_video(path: Path) -> None:
    if path.is_file():
        return
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
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _frame_hash(video: Path, t: float = 1.0) -> str:
    png = video.with_suffix(".smoke.png")
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(png)],
        check=True,
        capture_output=True,
    )
    from PIL import Image

    with Image.open(png) as img:
        return hashlib.md5(img.tobytes()).hexdigest()


def _wait_done(client: TestClient, job_id: str, timeout: float = 120.0) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload.get("status") == "done":
            return payload
        if payload.get("status") == "error":
            raise RuntimeError(payload.get("error") or "job failed")
        time.sleep(0.25)
    raise TimeoutError(f"job {job_id} did not finish")


def _put_layout(client: TestClient, job_id: str, layout: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.put(f"/api/jobs/{job_id}/layout", json=layout)
    if resp.status_code != 200:
        raise RuntimeError(f"PUT layout failed: {resp.status_code} {resp.text}")
    return resp.json()["layout"]


def _render(client: TestClient, job_id: str) -> None:
    resp = client.post(f"/api/jobs/{job_id}/render")
    if resp.status_code != 200:
        raise RuntimeError(f"POST render failed: {resp.status_code} {resp.text}")
    _wait_done(client, job_id)


def _seed_job(jobs_root: Path, video_src: Path) -> str:
    job_id = f"ui-smoke-{uuid.uuid4().hex[:8]}"
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    input_path = job_dir / "input.mp4"
    subprocess.run(["cp", str(video_src), str(input_path)], check=True)
    (job_dir / "source.srt").write_text(VI_SRT, encoding="utf-8")
    (job_dir / "vi_final.srt").write_text(VI_SRT, encoding="utf-8")
    layout_path = job_dir / "layout.json"
    layout_path.write_text(json.dumps(BASE_LAYOUT, indent=2), encoding="utf-8")
    output_path = job_dir / "demo.mp4"
    subprocess.run(["cp", str(input_path), str(output_path)], check=True)

    web.jobs[job_id] = web.Job(
        id=job_id,
        status=web.JobStatus.DONE,
        output_name="demo",
        input_path=str(input_path),
        source_srt_path=str(job_dir / "source.srt"),
        srt_path=str(job_dir / "vi_final.srt"),
        layout_path=str(layout_path),
        layout_saved=True,
        output_path=str(output_path),
        message="Complete",
        progress=100,
    )
    return job_id


def main() -> int:
    artifacts = ROOT / "artifacts" / "subtitle_style_ui_smoke"
    artifacts.mkdir(parents=True, exist_ok=True)

    video_src = Path(tempfile.gettempdir()) / "drakonsub_style_test.mp4"
    _ensure_video(video_src)

    tmp = tempfile.TemporaryDirectory()
    jobs_root = Path(tmp.name)
    original_root = web.JOBS_ROOT
    original_jobs = web.jobs
    web.JOBS_ROOT = jobs_root
    web.jobs = {}

    results: Dict[str, Any] = {"artifacts_dir": str(artifacts)}

    try:
        client = TestClient(web.app)
        job_id = _seed_job(jobs_root, video_src)
        results["job_id"] = job_id
        layout_path = jobs_root / job_id / "layout.json"
        output_path = jobs_root / job_id / "demo.mp4"

        # Test 3 — subtitles API (Chỉnh phụ đề modal data)
        sub_resp = client.get(f"/api/jobs/{job_id}/subtitles")
        results["subtitle_modal"] = {
            "status_code": sub_resp.status_code,
            "pass": sub_resp.status_code == 200 and "subtitles" in sub_resp.json(),
        }

        baseline_hash = _frame_hash(output_path)

        # Test 1 — background OFF
        layout_off = {**BASE_LAYOUT, "background_visible": False}
        saved_off = _put_layout(client, job_id, layout_off)
        on_disk_off = json.loads(layout_path.read_text(encoding="utf-8"))
        _render(client, job_id)
        hash_off = _frame_hash(output_path)
        results["background_off"] = {
            "layout_saved": saved_off.get("background_visible") is False,
            "layout_json": on_disk_off.get("background_visible"),
            "frame_hash": hash_off,
            "pass": saved_off.get("background_visible") is False and hash_off != baseline_hash,
        }

        # background ON
        layout_on = {**BASE_LAYOUT, "background_visible": True}
        saved_on = _put_layout(client, job_id, layout_on)
        on_disk_on = json.loads(layout_path.read_text(encoding="utf-8"))
        _render(client, job_id)
        hash_on = _frame_hash(output_path)
        results["background_on"] = {
            "layout_saved": saved_on.get("background_visible") is True,
            "layout_json": on_disk_on.get("background_visible"),
            "frame_hash": hash_on,
            "pass": saved_on.get("background_visible") is True and hash_on != hash_off,
        }

        # Test 2 — Comfortaa
        layout_comfortaa = {**BASE_LAYOUT, "font_family": "comfortaa"}
        saved_c = _put_layout(client, job_id, layout_comfortaa)
        on_disk_c = json.loads(layout_path.read_text(encoding="utf-8"))
        _render(client, job_id)
        hash_c = _frame_hash(output_path)
        results["comfortaa"] = {
            "layout_font": saved_c.get("font_family"),
            "layout_json_font": on_disk_c.get("font_family"),
            "frame_hash": hash_c,
            "pass": saved_c.get("font_family") == "comfortaa" and hash_c != hash_on,
        }

        # Montserrat Alternates
        layout_m = {**BASE_LAYOUT, "font_family": "montserrat_alternates"}
        saved_m = _put_layout(client, job_id, layout_m)
        on_disk_m = json.loads(layout_path.read_text(encoding="utf-8"))
        _render(client, job_id)
        hash_m = _frame_hash(output_path)
        results["montserrat_alternates"] = {
            "layout_font": saved_m.get("font_family"),
            "layout_json_font": on_disk_m.get("font_family"),
            "frame_hash": hash_m,
            "pass": saved_m.get("font_family") == "montserrat_alternates" and hash_m != hash_c,
        }

        results["evidence"] = {
            "layout_path": str(layout_path),
            "final_layout": on_disk_m,
            "frame_hashes": {
                "baseline": baseline_hash,
                "bg_off": hash_off,
                "bg_on": hash_on,
                "comfortaa": hash_c,
                "montserrat": hash_m,
            },
        }

        results["all_pass"] = all(
            results[k].get("pass")
            for k in (
                "subtitle_modal",
                "background_off",
                "background_on",
                "comfortaa",
                "montserrat_alternates",
            )
        )

        report_path = artifacts / "ui_smoke_report.json"
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        return 0 if results["all_pass"] else 1
    finally:
        web.JOBS_ROOT = original_root
        web.jobs = original_jobs
        tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
