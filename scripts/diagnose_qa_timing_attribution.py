#!/usr/bin/env python3
"""QA + timing attribution for translation quality review."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from auto_subtitle.qa_calibration import classify_cue_attribution  # noqa: E402
from auto_subtitle.subtitle_timing_optimizer import _parse_ts  # noqa: E402
from auto_subtitle.translation_quality_analyzer import analyze_translation_quality  # noqa: E402
from auto_subtitle.utils import parse_srt  # noqa: E402
from auto_subtitle.vi_compression import _cps  # noqa: E402

SAMPLES = ["raise_price_17", "no_rush_19", "buffett_bitcoin_29", "outsider_36"]
BENCH_ROOT = ROOT / "artifacts" / "multi_sample_benchmark" / "pipeline_regression"
OUT_ROOT = ROOT / "artifacts" / "translation_quality_review"

STAGES = {
    "vi_raw": "vi_raw.srt",
    "vi_after_editor": "vi_after_editor.srt",
    "vi_after_flow": "vi_after_flow.srt",
    "vi_final": "final_vi.srt",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    entries = parse_srt(path.read_text(encoding="utf-8"))
    return {i + 1: (e.get("text") or "").strip() for i, e in enumerate(entries)}


def _duration(entries: List[dict], cue_index: int) -> float:
    if cue_index < 1 or cue_index > len(entries):
        return 0.0
    e = entries[cue_index - 1]
    return max(0.01, _parse_ts(e.get("end_str", "0")) - _parse_ts(e.get("start_str", "0")))


def _stage_where_error(errors: List[str], stages: Dict[str, str], cue_index: int) -> str:
    if not errors:
        return "none"
    raw = stages.get("vi_raw", "")
    ed = stages.get("vi_after_editor", "")
    fl = stages.get("vi_after_flow", "")
    fn = stages.get("vi_final", "")
    if raw != ed:
        return "editor"
    if ed != fl:
        return "flow"
    if fl != fn:
        return "compression_or_timing"
    return "raw"


def _analyze_sample(sample_id: str, work_dir: Path) -> List[dict]:
    source_path = work_dir / "source_corrected.srt"
    if not source_path.exists():
        source_path = work_dir / "source.srt"
    source_entries = parse_srt(source_path.read_text(encoding="utf-8"))
    source_en = _load_map(source_path)
    stage_maps = {k: _load_map(work_dir / fname) for k, fname in STAGES.items()}
    final_entries = parse_srt((work_dir / "final_vi.srt").read_text(encoding="utf-8"))

    quality_path = work_dir / "translation_quality_report.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    else:
        quality = analyze_translation_quality(source_entries, final_entries)
    rows: List[dict] = []

    for a in quality.get("cue_assessments") or []:
        if not a.get("is_risky"):
            continue
        cid = a["cue_index"]
        dur = _duration(source_entries, cid)
        vi_final = stage_maps["vi_final"].get(cid, a.get("vi", ""))
        cps_val = round(_cps(vi_final, dur), 1) if vi_final and dur else ""
        errors = list(a.get("detected_translation_errors") or [])
        cal_notes = list(a.get("calibration_notes") or [])
        attr = classify_cue_attribution(errors, cal_notes, float(cps_val or 0), dur)
        stages_at = {k: stage_maps[k].get(cid, "") for k in stage_maps}
        start, end = "", ""
        if 0 < cid <= len(source_entries):
            e = source_entries[cid - 1]
            start, end = e.get("start_str", ""), e.get("end_str", "")

        rows.append(
            {
                "sample_id": sample_id,
                "cue_index": cid,
                "start_time": start,
                "end_time": end,
                "duration": round(dur, 3),
                "source_en": source_en.get(cid, a.get("en", "")),
                "vi_raw": stages_at["vi_raw"],
                "vi_after_editor": stages_at["vi_after_editor"],
                "vi_after_flow": stages_at["vi_after_flow"],
                "vi_final": vi_final,
                "auto_error_families": ";".join(errors),
                "cps": cps_val,
                "chars_per_line": max((len(ln) for ln in vi_final.splitlines()), default=len(vi_final)),
                "reading_time_estimate": round(len(vi_final.replace("\n", "")) / max(cps_val or 1, 1), 2)
                if vi_final
                else "",
                "quality_penalty": sum(
                    1 for _ in errors
                ),
                "stage_where_error_appears": _stage_where_error(errors, stages_at, cid),
                **attr,
            }
        )
    return rows


def _no_rush_focus(rows: List[dict], quality_score: int) -> dict:
    nr = [r for r in rows if r["sample_id"] == "no_rush_19"]
    fp = [r for r in nr if r.get("is_qa_false_positive_candidate")]
    true_err = [r for r in nr if not r.get("is_qa_false_positive_candidate") and r.get("auto_error_families")]
    return {
        "quality_score": quality_score,
        "risky_cue_count": len(nr),
        "qa_false_positive_candidates": len(fp),
        "true_errors_remaining": len(true_err),
        "root_cause_summary": [
            "Primary blockers: CPS on short fragments + semantic cue 12/17 if still flagged",
            "Calibration downgrades rhetorical build-it repeats and fragment CPS",
            "take no money → kiếm tiền classified minor_nuance_loss when applicable",
        ],
        "focus_cues": {r["cue_index"]: r for r in nr if r["cue_index"] in (1, 4, 12, 17)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="append", default=[])
    args = parser.parse_args()
    samples = args.sample or SAMPLES

    all_rows: List[dict] = []
    report: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "samples": {},
        "calibration_rules": [
            "fragment CPS downgrade (<0.8s, fragment, continues sentence)",
            "repeated_meaning downgrade when EN rhetorical repeat",
            "minor_nuance_loss for take no money vs kiếm tiền",
        ],
    }

    for sid in samples:
        work = BENCH_ROOT / sid
        if not work.exists():
            print(f"[qa_attribution] SKIP {sid}: no artifacts")
            continue
        rows = _analyze_sample(sid, work)
        all_rows.extend(rows)
        qpath = work / "translation_quality_report.json"
        qscore = None
        if qpath.exists():
            qscore = json.loads(qpath.read_text()).get("quality_score")
        report["samples"][sid] = {
            "quality_score": qscore,
            "risky_rows": len(rows),
            "no_rush_focus": _no_rush_focus(rows, qscore or 0) if sid == "no_rush_19" else None,
        }
        print(f"[qa_attribution] {sid}: {len(rows)} risky cues, score={qscore}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUT_ROOT / "qa_timing_attribution_v1.json"
    csv_path = OUT_ROOT / "qa_timing_attribution_v1.csv"
    md_path = OUT_ROOT / "qa_timing_attribution_v1.md"

    report["cue_rows"] = all_rows
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if all_rows:
        fields = list(all_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)

    lines = ["# QA Timing Attribution v1", "", f"Generated: {report['generated_at']}", ""]
    for sid, meta in report.get("samples", {}).items():
        lines.append(f"## {sid}")
        lines.append(f"- quality_score: **{meta.get('quality_score')}**")
        lines.append(f"- risky_rows: **{meta.get('risky_rows')}**")
        if meta.get("no_rush_focus"):
            for c in meta["no_rush_focus"].get("root_cause_summary", []):
                lines.append(f"- {c}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[qa_attribution] Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
