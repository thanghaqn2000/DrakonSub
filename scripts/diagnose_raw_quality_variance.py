#!/usr/bin/env python3
"""Diagnose raw translation quality variance across repeated fresh e2e runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from auto_subtitle.utils import parse_srt  # noqa: E402
from scripts.benchmark_raw_cache import (  # noqa: E402
    RAW_CACHE_ROOT,
    legacy_cache_key,
    resolve_cached_vi_raw,
)

BENCHMARK_SCRIPT = ROOT / "scripts" / "run_multi_sample_benchmark.py"
E2E_ROOT = ROOT / "artifacts" / "multi_sample_benchmark" / "end_to_end"
OUT_ROOT = ROOT / "artifacts" / "translation_quality_review"
RUNS_ROOT = OUT_ROOT / "raw_variance_runs"
DEFAULT_SAMPLES = [
    "raise_price_17",
    "no_rush_19",
    "buffett_bitcoin_29",
    "outsider_36",
]
STAGES = ("vi_raw", "vi_after_editor", "vi_after_flow", "final_vi")
NO_RUSH_FOCUS = (1, 4, 6, 12, 17, 18)
OUTSIDER_FOCUS = (11, 12, 24, 27)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    return {i + 1: (e.get("text") or "").strip() for i, e in enumerate(parse_srt(path.read_text()))}


def _run_fresh_e2e(
    engine: str, raw_mode: str, deterministic: bool, raw_llm_cache: bool, run_idx: int
) -> int:
    import os

    os.environ["BENCHMARK_RUN_ID"] = f"run_{run_idx}"
    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--engine",
        engine,
        "--fresh",
        "--benchmark-mode",
        "end_to_end",
        "--raw-translation-mode",
        raw_mode,
    ]
    if deterministic:
        cmd.append("--deterministic")
    if raw_llm_cache:
        cmd.append("--raw-llm-cache")
    print(f"[variance] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def _snapshot_run(run_idx: int, samples: List[str]) -> Path:
    run_dir = RUNS_ROOT / f"run_{run_idx}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    report_src = E2E_ROOT / "benchmark_report.json"
    if report_src.exists():
        shutil.copy2(report_src, run_dir / "benchmark_report.json")
    for sid in samples:
        src = E2E_ROOT / sid
        if src.exists():
            shutil.copytree(src, run_dir / sid)
    return run_dir


def _sample_scores(run_dir: Path, sid: str) -> Dict[str, Any]:
    report_path = run_dir / "benchmark_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text())
    for item in report.get("samples", []):
        if item.get("id") == sid:
            m = item.get("metrics") or {}
            return {
                "quality_score": m.get("quality_score"),
                "risky_cue_count": m.get("risky_cue_count"),
                "semantic_alignment_errors": m.get("semantic_alignment_errors"),
                "confirmed_error_counts": m.get("confirmed_error_counts") or {},
            }
    return {}


def _vi_raw_hash(run_dir: Path, sid: str) -> str:
    p = run_dir / sid / "vi_raw.srt"
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _video_context_hash(sample_dir: Path) -> str:
    p = sample_dir / "video_context.json"
    return _file_hash(p)


def _span_summary(sample_dir: Path) -> Dict[str, Any]:
    p = sample_dir / "span_guarded_tiered_report.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        summary = data.get("summary") or data
        return {
            "classifier_calls": summary.get("classifier_calls"),
            "span_high": summary.get("span_high"),
            "repair_rejected": summary.get("repair_rejected"),
            "skipped_good_windows": summary.get("skipped_good_windows"),
        }
    except Exception:
        return {}


def _first_variance_stage(run_dirs: List[Path], sid: str, cue: int) -> str:
    prev = None
    for stage in STAGES:
        texts = []
        for rd in run_dirs:
            t = _load_map(rd / sid / f"{stage}.srt").get(cue, "")
            texts.append(t)
        if len(set(texts)) > 1:
            return stage
        prev = stage
    q_scores = [_sample_scores(rd, sid).get("quality_score") for rd in run_dirs]
    if len(set(q_scores)) > 1:
        return "qa_scoring"
    return "none"


def _legacy_known_good_raw(sid: str, engine: str, source_path: Path) -> Optional[str]:
    path, _, status = resolve_cached_vi_raw(sid, source_path, engine, "grouped")
    if status.startswith("hit"):
        return _load_map(path).get(1, "")[:80]
    legacy = RAW_CACHE_ROOT / sid / engine / legacy_cache_key(source_path, engine) / "vi_raw.srt"
    if legacy.exists():
        return _load_map(legacy).get(1, "")[:80]
    return None


def _build_cue_rows(
    samples: List[str],
    run_dirs: List[Path],
    jobs_manifest: dict,
) -> List[dict]:
    rows: List[dict] = []
    jobs_root = Path(jobs_manifest["jobs_root"])
    for sid in samples:
        job_id = next(s["job_id"] for s in jobs_manifest["samples"] if s["id"] == sid)
        source_path = jobs_root / job_id / "source.srt"
        source_entries = parse_srt(source_path.read_text()) if source_path.exists() else []
        n_cues = len(source_entries)
        focus = set(NO_RUSH_FOCUS if sid == "no_rush_19" else OUTSIDER_FOCUS if sid == "outsider_36" else range(1, n_cues + 1))
        if sid not in ("no_rush_19", "outsider_36"):
            focus = set(range(1, min(n_cues, 20) + 1))

        for cue in sorted(focus):
            if cue > n_cues:
                continue
            en = source_entries[cue - 1].get("text", "").strip()
            raw_by_run = [_load_map(rd / sid / "vi_raw.srt").get(cue, "") for rd in run_dirs]
            final_by_run = [_load_map(rd / sid / "final_vi.srt").get(cue, "") for rd in run_dirs]
            scores = [_sample_scores(rd, sid).get("quality_score") for rd in run_dirs]
            row = {
                "sample_id": sid,
                "cue_index": cue,
                "source_en": en,
                "run_1_raw": raw_by_run[0] if len(raw_by_run) > 0 else "",
                "run_2_raw": raw_by_run[1] if len(raw_by_run) > 1 else "",
                "run_3_raw": raw_by_run[2] if len(raw_by_run) > 2 else "",
                "run_1_final": final_by_run[0] if len(final_by_run) > 0 else "",
                "run_2_final": final_by_run[1] if len(final_by_run) > 1 else "",
                "run_3_final": final_by_run[2] if len(final_by_run) > 2 else "",
                "raw_text_changed": len(set(raw_by_run)) > 1,
                "final_text_changed": len(set(final_by_run)) > 1,
                "quality_score_by_run": scores,
                "stage_where_variance_first_appears": _first_variance_stage(run_dirs, sid, cue),
                "likely_variance_source": "raw_llm_translation"
                if len(set(raw_by_run)) > 1
                else "unknown",
            }
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--engine", default="openai")
    parser.add_argument("--raw-translation-mode", default="span_guarded_tiered")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--raw-llm-cache", action="store_true", default=False)
    parser.add_argument("--samples", default=",".join(DEFAULT_SAMPLES))
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    samples = [s.strip() for s in args.samples.split(",") if s.strip()]
    manifest = json.loads((ROOT / "scripts" / "benchmark_samples.json").read_text())

    run_dirs: List[Path] = []
    if not args.skip_benchmark:
        for i in range(1, args.runs + 1):
            code = _run_fresh_e2e(
                args.engine,
                args.raw_translation_mode,
                args.deterministic,
                args.raw_llm_cache,
                i,
            )
            if code != 0:
                print(f"[variance] run {i} benchmark exit {code}")
            run_dirs.append(_snapshot_run(i, samples))
    else:
        for i in range(1, args.runs + 1):
            rd = RUNS_ROOT / f"run_{i}"
            if rd.exists():
                run_dirs.append(rd)

    if len(run_dirs) < 2:
        print("[variance] need at least 2 run snapshots")
        return 1

    score_table: Dict[str, List[Optional[int]]] = {sid: [] for sid in samples}
    per_sample: Dict[str, Any] = {}
    for sid in samples:
        scores = [_sample_scores(rd, sid).get("quality_score") for rd in run_dirs]
        score_table[sid] = scores
        vc_hashes = [_video_context_hash(rd / sid) for rd in run_dirs]
        span_summaries = [_span_summary(rd / sid) for rd in run_dirs]
        per_sample[sid] = {
            "scores": scores,
            "min": min(s for s in scores if s is not None) if scores else None,
            "max": max(s for s in scores if s is not None) if scores else None,
            "range": (max(s for s in scores if s is not None) - min(s for s in scores if s is not None))
            if scores and all(s is not None for s in scores)
            else None,
            "vi_raw_hashes": [_vi_raw_hash(rd, sid) for rd in run_dirs],
            "vi_raw_stable": len(set(_vi_raw_hash(rd, sid) for rd in run_dirs if _vi_raw_hash(rd, sid))) <= 1,
            "video_context_stable": len(set(h for h in vc_hashes if h)) <= 1,
            "video_context_hashes": vc_hashes,
            "span_classifier_stable": span_summaries == span_summaries[:1] * len(span_summaries),
            "span_summaries": span_summaries,
        }

    cue_rows = _build_cue_rows(samples, run_dirs, manifest)
    provider_variance = any(
        per_sample[s].get("range") and per_sample[s]["range"] > 0 for s in samples
    )

    report: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "runs": len(run_dirs),
        "engine": args.engine,
        "raw_translation_mode": args.raw_translation_mode,
        "deterministic": args.deterministic,
        "raw_llm_cache": args.raw_llm_cache,
        "provider_level_variance_detected": provider_variance,
        "score_table": score_table,
        "per_sample": per_sample,
        "cue_rows": cue_rows,
        "recommendation": "Option A — Raw response cache by prompt hash for benchmark deterministic mode",
        "targets": {
            "no_rush_19": ">=85 in 2/3",
            "outsider_36": ">=80 in 2/3",
            "raise_price_17": ">=70 in 2/3",
            "buffett_bitcoin_29": ">=90 in 2/3",
        },
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUT_ROOT / "raw_quality_variance_v1.json"
    md_path = OUT_ROOT / "raw_quality_variance_v1.md"
    csv_path = OUT_ROOT / "raw_quality_variance_v1.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Raw Quality Variance v1", "", f"Generated: {report['generated_at']}", ""]
    lines.append("## Score table")
    lines.append("| sample | run scores | range |")
    lines.append("|--------|------------|-------|")
    for sid, scores in score_table.items():
        r = per_sample[sid].get("range")
        lines.append(f"| {sid} | {scores} | {r} |")
    lines.append("")
    lines.append(f"**provider_level_variance_detected**: {provider_variance}")
    lines.append(f"**recommendation**: {report['recommendation']}")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    if cue_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cue_rows[0].keys()))
            w.writeheader()
            w.writerows(cue_rows)

    print(f"[variance] Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
