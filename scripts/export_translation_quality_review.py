#!/usr/bin/env python3
"""Export cue-level translation quality review artifacts for SA human scoring."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from auto_subtitle.config import get_openai_model  # noqa: E402
from auto_subtitle.subtitle_timing_optimizer import _parse_ts  # noqa: E402
from auto_subtitle.utils import parse_srt  # noqa: E402
from auto_subtitle.vi_compression import _cps  # noqa: E402

MANIFEST = ROOT / "scripts" / "benchmark_samples.json"
OUT_ROOT = ROOT / "artifacts" / "translation_quality_review"
WORK_ROOT = OUT_ROOT / "_work"

_benchmark_path = ROOT / "scripts" / "run_multi_sample_benchmark.py"
_spec = importlib.util.spec_from_file_location("run_multi_sample_benchmark", _benchmark_path)
assert _spec and _spec.loader
_benchmark = importlib.util.module_from_spec(_spec)
sys.modules["run_multi_sample_benchmark"] = _benchmark
_spec.loader.exec_module(_benchmark)

_audit_path = ROOT / "scripts" / "run_cue_mapping_audit.py"
_aspec = importlib.util.spec_from_file_location("run_cue_mapping_audit", _audit_path)
assert _aspec and _aspec.loader
_audit = importlib.util.module_from_spec(_aspec)
sys.modules["run_cue_mapping_audit"] = _audit
_aspec.loader.exec_module(_audit)

BenchmarkFlags = _benchmark.BenchmarkFlags
_prepare_sample_debug = _benchmark._prepare_sample_debug
_sample_result = _benchmark._sample_result
_run_pipeline_stages = _audit._run_pipeline_stages
_build_reports = _audit._build_reports

STAGE_FILES = {
    "vi_raw": "vi_raw.srt",
    "vi_after_editor": "vi_after_editor.srt",
    "vi_after_flow": "vi_after_flow.srt",
    "vi_final": "final_vi.srt",
}

REVIEW_FIELDS = [
    "sample_id",
    "cue_index",
    "start_time",
    "end_time",
    "source_en",
    "vi_raw",
    "vi_after_editor",
    "vi_after_flow",
    "vi_final",
    "auto_quality_score",
    "auto_error_families",
    "cps",
    "chars_per_line",
    "semantic_risk",
    "readability_risk",
    "SA_semantic_score",
    "SA_naturalness_score",
    "SA_readability_score",
    "SA_context_score",
    "SA_timing_score",
    "SA_overall_score",
    "SA_comment",
    "recommended_fix_layer",
]

SA_EMPTY_FIELDS = {
    "SA_semantic_score": "",
    "SA_naturalness_score": "",
    "SA_readability_score": "",
    "SA_context_score": "",
    "SA_timing_score": "",
    "SA_overall_score": "",
    "SA_comment": "",
    "recommended_fix_layer": "",
}


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _load_stage_map(work_dir: Path, filename: str) -> Dict[int, str]:
    path = work_dir / filename
    if not path.exists():
        return {}
    entries = parse_srt(path.read_text(encoding="utf-8"))
    return {i + 1: (e.get("text") or "").strip() for i, e in enumerate(entries)}


def _timing(entries: List[dict], cue_index: int) -> tuple[str, str]:
    if cue_index < 1 or cue_index > len(entries):
        return "", ""
    e = entries[cue_index - 1]
    return e.get("start_str", ""), e.get("end_str", "")


def _duration(entries: List[dict], cue_index: int) -> float:
    if cue_index < 1 or cue_index > len(entries):
        return 0.0
    e = entries[cue_index - 1]
    try:
        return max(0.01, _parse_ts(e.get("end_str", "0")) - _parse_ts(e.get("start_str", "0")))
    except Exception:
        return 0.0


def _chars_per_line(text: str) -> int:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        lines = [(text or "").strip()]
    return max((len(ln) for ln in lines), default=0)


def _infer_fix_layer(
    errors: List[str],
    vi_raw: str,
    vi_editor: str,
    vi_flow: str,
    vi_final: str,
) -> str:
    if not errors:
        return ""
    semantic = {
        "semantic_drift_error",
        "semantic_alignment_error",
        "missing_or_empty_cue_error",
        "literal_translation_error",
        "domain_term_error",
    }
    readability = {"readability_cps_error", "over_compression_error"}
    flowish = {"cue_flow_error", "repeated_meaning_error"}

    err_set = set(errors)
    if err_set & semantic and vi_raw == vi_editor == vi_flow:
        return "raw_translation"
    if err_set & flowish and vi_flow != vi_editor:
        return "flow"
    if vi_editor != vi_raw and err_set & (flowish | {"unnatural_vietnamese_error"}):
        return "editor"
    if err_set & readability:
        return "compression" if vi_final != vi_flow else "timing"
    if err_set & semantic:
        return "raw_translation"
    return "unknown"


def _build_rows(
    sample_id: str,
    work_dir: Path,
    quality_report: dict,
    auto_quality_score: Optional[int],
) -> tuple[List[dict], List[str]]:
    source_entries = parse_srt((work_dir / "source.srt").read_text(encoding="utf-8"))
    source_en = _load_stage_map(work_dir, "source_corrected.srt") or _load_stage_map(
        work_dir, "source.srt"
    )
    stage_maps = {key: _load_stage_map(work_dir, fname) for key, fname in STAGE_FILES.items()}
    missing_stages = [k for k, m in stage_maps.items() if not m and k != "vi_raw"]

    assessments = {
        a["cue_index"]: a for a in quality_report.get("cue_assessments") or []
    }
    n = len(source_entries)
    rows: List[dict] = []

    for cue_index in range(1, n + 1):
        a = assessments.get(cue_index, {})
        vi_final = stage_maps["vi_final"].get(cue_index, "")
        dur = _duration(source_entries, cue_index)
        cps_val = round(_cps(vi_final, dur), 1) if vi_final and dur else ""
        errors = list(a.get("detected_translation_errors") or [])
        start, end = _timing(source_entries, cue_index)
        vi_raw = stage_maps["vi_raw"].get(cue_index, "")
        vi_ed = stage_maps["vi_after_editor"].get(cue_index, "")
        vi_fl = stage_maps["vi_after_flow"].get(cue_index, "")

        row = {
            "sample_id": sample_id,
            "cue_index": cue_index,
            "start_time": start,
            "end_time": end,
            "source_en": source_en.get(cue_index, a.get("en", "")),
            "vi_raw": vi_raw,
            "vi_after_editor": vi_ed,
            "vi_after_flow": vi_fl,
            "vi_final": vi_final,
            "auto_quality_score": auto_quality_score if auto_quality_score is not None else "",
            "auto_error_families": ";".join(errors),
            "cps": cps_val,
            "chars_per_line": _chars_per_line(vi_final),
            "semantic_risk": bool(
                set(errors)
                & {
                    "semantic_drift_error",
                    "semantic_alignment_error",
                    "missing_or_empty_cue_error",
                }
            ),
            "readability_risk": "readability_cps_error" in errors,
            **SA_EMPTY_FIELDS,
            "recommended_fix_layer": _infer_fix_layer(errors, vi_raw, vi_ed, vi_fl, vi_final),
        }
        rows.append(row)

    notes = []
    if missing_stages:
        notes.append(f"missing stages: {', '.join(missing_stages)}")
    if not stage_maps["vi_raw"]:
        notes.append("vi_raw missing — used empty")
    return rows, notes


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: List[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"meta": meta, "cues": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_md(path: Path, sample_id: str, rows: List[dict], meta: dict) -> None:
    lines = [
        f"# Translation Quality Review — {sample_id}",
        "",
        f"- Mode: **{meta.get('mode')}**",
        f"- Engine: **{meta.get('engine')}**",
        f"- Model: **{meta.get('model')}**",
        f"- Auto quality score: **{meta.get('auto_quality_score')}**",
        f"- Cues: **{len(rows)}**",
        "",
    ]
    if meta.get("notes"):
        lines.append("## Notes")
        for note in meta["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    for row in rows:
        cid = row["cue_index"]
        lines.extend(
            [
                f"## Cue {cid}",
                f"Time: {row['start_time']} --> {row['end_time']}",
                "",
                "EN:",
                f"> {row['source_en']}",
                "",
                "VI raw:",
                f"> {row['vi_raw'] or '—'}",
                "",
                "VI after editor:",
                f"> {row['vi_after_editor'] or '—'}",
                "",
                "VI after flow:",
                f"> {row['vi_after_flow'] or '—'}",
                "",
                "VI final:",
                f"> {row['vi_final'] or '—'}",
                "",
                "Auto QA:",
                f"- score (sample): {row['auto_quality_score']}",
                f"- errors: {row['auto_error_families'] or '—'}",
                f"- cps: {row['cps']}",
                f"- semantic_risk: {row['semantic_risk']}",
                f"- readability_risk: {row['readability_risk']}",
                f"- suggested fix layer: {row['recommended_fix_layer'] or '—'}",
                "",
                "SA scores:",
                "- semantic:",
                "- naturalness:",
                "- readability:",
                "- context:",
                "- timing:",
                "- overall:",
                "",
                "SA comment:",
                "",
                "Recommended fix layer:",
                "",
                "---",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _flags_for_mode(mode: str, engine: str, use_raw_cache: bool) -> BenchmarkFlags:
    if mode == "pipeline_regression":
        return BenchmarkFlags(
            requested_engine=engine,
            use_raw_cache=use_raw_cache,
            reuse_raw_manifest=True,
            deterministic=True,
        )
    if mode == "end_to_end":
        return BenchmarkFlags(
            requested_engine=engine,
            force_fresh=True,
            deterministic=True,
        )
    raise ValueError(f"Unknown mode: {mode}")


def _export_mode(
    mode: str,
    samples: List[dict],
    jobs_root: Path,
    engine: str,
    use_raw_cache: bool,
) -> dict:
    os.environ["TRANSLATION_ENGINE"] = engine
    os.environ["BENCHMARK_DETERMINISTIC"] = "1"
    pass_flags = _flags_for_mode(mode, engine, use_raw_cache)
    index_samples: List[dict] = []
    total_cues = 0
    all_notes: List[str] = []

    for sample in samples:
        sid = sample["id"]
        work_dir = WORK_ROOT / mode / sid
        export_dir = OUT_ROOT / mode / sid
        print(f"\n[ReviewExport] {mode} / {sid}")

        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)

        source, reuse_raw, _, _ = _prepare_sample_debug(
            sample, jobs_root, work_dir, flags=pass_flags
        )
        if source is None:
            print(f"[ReviewExport] SKIP {sid}: missing source.srt")
            all_notes.append(f"{sid}: missing source")
            continue

        run_meta = _run_pipeline_stages(
            work_dir,
            job_source=source,
            reuse_raw=reuse_raw,
            translation_engine=engine,
        )
        _build_reports(work_dir, run_meta)
        metrics = _sample_result(work_dir)

        quality_path = work_dir / "translation_quality_report.json"
        quality_report = {}
        if quality_path.exists():
            quality_report = json.loads(quality_path.read_text(encoding="utf-8"))

        rows, notes = _build_rows(
            sid,
            work_dir,
            quality_report,
            metrics.get("quality_score"),
        )
        meta = {
            "sample_id": sid,
            "mode": mode,
            "engine": engine,
            "model": get_openai_model(),
            "auto_quality_score": metrics.get("quality_score"),
            "risky_cue_count": metrics.get("risky_cue_count"),
            "notes": notes,
        }

        export_dir.mkdir(parents=True, exist_ok=True)
        csv_path = export_dir / "review.csv"
        md_path = export_dir / "review.md"
        json_path = export_dir / "review.json"
        _write_csv(csv_path, rows)
        _write_md(md_path, sid, rows, meta)
        _write_json(json_path, rows, meta)

        total_cues += len(rows)
        all_notes.extend(notes)
        index_samples.append(
            {
                "sample_id": sid,
                "cue_count": len(rows),
                "review_csv": str(csv_path.relative_to(ROOT)),
                "review_md": str(md_path.relative_to(ROOT)),
                "review_json": str(json_path.relative_to(ROOT)),
                "auto_quality_score": metrics.get("quality_score"),
                "risky_cue_count": metrics.get("risky_cue_count"),
                "notes": notes,
            }
        )
        print(
            f"[ReviewExport] {sid}: {len(rows)} cues, "
            f"auto_score={metrics.get('quality_score')}, risky={metrics.get('risky_cue_count')}"
        )

    return {
        "mode": mode,
        "samples": index_samples,
        "summary": {
            "sample_count": len(index_samples),
            "total_cues": total_cues,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engine": engine,
            "model": get_openai_model(),
            "notes": all_notes,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SA translation quality review artifacts")
    parser.add_argument("--engine", default="openai")
    parser.add_argument(
        "--mode",
        choices=("pipeline_regression", "end_to_end"),
        default="end_to_end",
    )
    parser.add_argument("--both-modes", action="store_true")
    parser.add_argument("--use-raw-cache", action="store_true")
    parser.add_argument(
        "--samples",
        default="raise_price_17,no_rush_19,buffett_bitcoin_29,outsider_36",
        help="Comma-separated sample ids",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    jobs_root = Path(manifest["jobs_root"])
    wanted = {s.strip() for s in args.samples.split(",") if s.strip()}
    samples = [s for s in manifest["samples"] if s["id"] in wanted]
    if not samples:
        print("No matching samples", file=sys.stderr)
        return 1

    modes = ["pipeline_regression", "end_to_end"] if args.both_modes else [args.mode]
    runs = []
    for mode in modes:
        use_cache = args.use_raw_cache or mode == "pipeline_regression"
        runs.append(_export_mode(mode, samples, jobs_root, args.engine.strip().lower(), use_cache))

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }
    if len(runs) == 1:
        index["mode"] = runs[0]["mode"]
        index["samples"] = runs[0]["samples"]
        index["summary"] = runs[0]["summary"]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = OUT_ROOT / "quality_review_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ReviewExport] Wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
