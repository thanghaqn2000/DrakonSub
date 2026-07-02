#!/usr/bin/env python3
"""Build aggregate cue-shift diagnosis from benchmark sample artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_subtitle.cue_shift_detector import build_aggregate_diagnosis, diagnose_sample
from auto_subtitle.utils import parse_srt

OUT_ROOT = ROOT / "artifacts" / "multi_sample_benchmark"


def _load_sample_diagnosis(sample_dir: Path) -> dict:
    cached = sample_dir / "cue_shift_diagnosis_sample.json"
    if cached.exists():
        data = json.loads(cached.read_text(encoding="utf-8"))
        if data.get("sample") in (None, "job"):
            data["sample"] = sample_dir.name
        return data

    meaning_units = []
    mu_path = sample_dir / "meaning_units.json"
    if mu_path.exists():
        data = json.loads(mu_path.read_text(encoding="utf-8"))
        meaning_units = data.get("units") or data if isinstance(data, list) else []

    video_context = {}
    vc_path = sample_dir / "video_context.json"
    if vc_path.exists():
        video_context = json.loads(vc_path.read_text(encoding="utf-8"))

    en_path = sample_dir / "source.srt"
    vi_path = sample_dir / "vi_after_final_repair.srt"
    if not en_path.exists() or not vi_path.exists():
        return {"sample": sample_dir.name, "has_local_cue_shift": False, "shift_windows": []}

    source_entries = parse_srt(en_path.read_text(encoding="utf-8"))
    vi_entries = parse_srt(vi_path.read_text(encoding="utf-8"))
    return diagnose_sample(
        sample_dir.name, source_entries, vi_entries, meaning_units, video_context
    )


def main() -> None:
    samples = []
    for sample_dir in sorted(OUT_ROOT.iterdir()):
        if not sample_dir.is_dir():
            continue
        if sample_dir.name in ("deferred",):
            continue
        samples.append(_load_sample_diagnosis(sample_dir))

    report = build_aggregate_diagnosis(samples)
    out_path = OUT_ROOT / "cue_shift_diagnosis.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({report['summary']['sample_count_with_shift']} samples with shift)")


if __name__ == "__main__":
    main()
