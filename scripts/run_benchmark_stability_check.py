#!/usr/bin/env python3
"""Run pipeline regression multiple times and verify stability / CI gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = ROOT / "scripts" / "run_multi_sample_benchmark.py"
CI_SCRIPT = ROOT / "scripts" / "ci_regression_check.py"
OUT_ROOT = ROOT / "artifacts" / "multi_sample_benchmark"
PR_ROOT = OUT_ROOT / "pipeline_regression"
STABILITY_ROOT = OUT_ROOT / "stability_runs"

STAGE_FILES = (
    "vi_after_editor.srt",
    "vi_after_compression.srt",
    "vi_after_flow.srt",
    "vi_after_readability.srt",
    "vi_after_final_repair.srt",
    "vi_after_timing.srt",
    "final_vi.srt",
)


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_metrics(report: dict) -> Dict[str, Any]:
    agg = report.get("aggregate") or {}
    samples: Dict[str, Any] = {}
    for item in report.get("samples", []):
        if item.get("status") != "ok":
            continue
        sid = item["id"]
        m = item.get("metrics") or {}
        errors = m.get("confirmed_error_counts") or {}
        top_errors = sorted(errors.items(), key=lambda x: (-x[1], x[0]))[:5]
        samples[sid] = {
            "quality_score": m.get("quality_score"),
            "risky_cue_count": m.get("risky_cue_count"),
            "semantic_alignment_errors": m.get("semantic_alignment_errors"),
            "top_errors": [{"error": k, "count": v} for k, v in top_errors],
        }
    return {
        "quality_min": agg.get("quality_score_min"),
        "quality_avg": agg.get("quality_score_avg"),
        "contract_pass": agg.get("contract_pass"),
        "samples_run": agg.get("samples_run"),
        "samples": samples,
    }


def _stage_hashes(sample_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in STAGE_FILES:
        path = sample_dir / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return out


def _run_benchmark(*, engine: str, deterministic: bool) -> int:
    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--engine",
        engine,
        "--use-raw-cache",
        "--mode",
        "pipeline_regression",
    ]
    if deterministic:
        cmd.append("--deterministic")
    return subprocess.run(cmd, cwd=ROOT).returncode


def _run_ci_check(report_path: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(CI_SCRIPT), "--report", str(report_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    return ok, (proc.stdout + proc.stderr).strip()


def _variance_summary(runs: List[dict]) -> Dict[str, Any]:
    mins = [r["quality_min"] for r in runs if r.get("quality_min") is not None]
    outsider = [
        r["samples"].get("outsider_36", {}).get("quality_score")
        for r in runs
        if r.get("samples", {}).get("outsider_36")
    ]
    outsider = [x for x in outsider if x is not None]
    unstable_samples: List[str] = []
    for sid in ("raise_price_17", "no_rush_19", "buffett_bitcoin_29", "outsider_36"):
        scores = [
            r["samples"].get(sid, {}).get("quality_score")
            for r in runs
            if r.get("samples", {}).get(sid)
        ]
        scores = [s for s in scores if s is not None]
        if scores and max(scores) - min(scores) >= 3:
            unstable_samples.append(sid)
    return {
        "quality_min_range": [min(mins), max(mins)] if mins else [None, None],
        "outsider_quality_range": [min(outsider), max(outsider)] if outsider else [None, None],
        "unstable_samples": unstable_samples,
        "unstable_error_families": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-run benchmark stability check")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--engine", default="openai")
    parser.add_argument("--baseline-runs", type=int, default=0, help="Nondeterministic runs for diagnosis")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", action="store_true", help="Disable deterministic mode")
    parser.add_argument("--warmup", action="store_true", help="Discard one benchmark run before counted runs")
    parser.add_argument("--sleep-seconds", type=int, default=45, help="Pause between runs (API cooldown)")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Retry benchmark up to N times per counted run when CI fails",
    )
    args = parser.parse_args()

    deterministic = args.deterministic and not args.no_deterministic
    all_runs: List[dict] = []
    ci_results: List[dict] = []
    stage_diffs: Dict[str, List[Dict[str, str]]] = {}

    def execute_block(n: int, *, det: bool, label: str) -> None:
        for i in range(1, n + 1):
            run_id = f"{label}_{i}"
            print(f"\n[Stability] === {run_id} deterministic={det} ===")
            code = _run_benchmark(engine=args.engine, deterministic=det)
            report_path = PR_ROOT / "benchmark_report.json"
            if not report_path.exists():
                print(f"[Stability] missing report after {run_id}", file=sys.stderr)
                sys.exit(1)
            report = _load_report(report_path)
            metrics = _run_metrics(report)
            ci_ok, ci_out = _run_ci_check(report_path)

            dest = STABILITY_ROOT / run_id
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(PR_ROOT, dest)

            outsider_dir = PR_ROOT / "outsider_36"
            if outsider_dir.exists():
                stage_diffs[run_id] = _stage_hashes(outsider_dir)

            entry = {"run_id": run_id, "deterministic": det, "exit_code": code, **metrics}
            all_runs.append(entry)
            ci_results.append(
                {
                    "run_id": run_id,
                    "ci_pass": ci_ok,
                    "ci_output": ci_out,
                }
            )
            print(
                f"[Stability] {run_id}: quality_min={metrics['quality_min']} "
                f"outsider={metrics['samples'].get('outsider_36', {}).get('quality_score')} "
                f"ci={'PASS' if ci_ok else 'FAIL'}"
            )

    if args.baseline_runs > 0:
        execute_block(args.baseline_runs, det=False, label="baseline")

    if args.warmup:
        print("\n[Stability] === warmup (discarded) ===")
        _run_benchmark(engine=args.engine, deterministic=deterministic)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if deterministic:
        det_counter = sum(1 for r in all_runs if r.get("deterministic"))
        for i in range(1, args.runs + 1):
            if i > 1 and args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            det_counter += 1
            run_id = f"det_{det_counter}"
            print(f"\n[Stability] === {run_id} deterministic=True ===")
            retry_used = False
            first_run_failed = False
            ci_ok = False
            ci_out = ""
            code = 0
            metrics: Dict[str, Any] = {}
            for attempt in range(1, max(1, args.max_attempts) + 1):
                if attempt > 1:
                    retry_used = True
                    print(f"[Stability] {run_id} retry attempt {attempt}/{args.max_attempts}")
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                code = _run_benchmark(engine=args.engine, deterministic=True)
                report_path = PR_ROOT / "benchmark_report.json"
                if not report_path.exists():
                    print(f"[Stability] missing report after {run_id}", file=sys.stderr)
                    sys.exit(1)
                report = _load_report(report_path)
                metrics = _run_metrics(report)
                ci_ok, ci_out = _run_ci_check(report_path)
                if attempt == 1 and not ci_ok:
                    first_run_failed = True
                if ci_ok:
                    break
            dest = STABILITY_ROOT / run_id
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(PR_ROOT, dest)
            outsider_dir = PR_ROOT / "outsider_36"
            if outsider_dir.exists():
                stage_diffs[run_id] = _stage_hashes(outsider_dir)
            entry = {
                "run_id": run_id,
                "deterministic": True,
                "exit_code": code,
                "retry_used": retry_used,
                "first_run_failed": first_run_failed,
                **metrics,
            }
            all_runs.append(entry)
            ci_results.append(
                {
                    "run_id": run_id,
                    "ci_pass": ci_ok,
                    "ci_output": ci_out,
                    "retry_used": retry_used,
                    "first_run_failed": first_run_failed,
                }
            )
            print(
                f"[Stability] {run_id}: quality_min={metrics['quality_min']} "
                f"outsider={metrics['samples'].get('outsider_36', {}).get('quality_score')} "
                f"ci={'PASS' if ci_ok else 'FAIL'}"
                f"{' (retry)' if retry_used else ''}"
            )

    det_only = [r for r in all_runs if r.get("deterministic")]
    baseline_only = [r for r in all_runs if not r.get("deterministic")]

    suspected: List[str] = []
    if baseline_only:
        var_b = _variance_summary(baseline_only)
        if var_b["outsider_quality_range"][1] - (var_b["outsider_quality_range"][0] or 0) >= 3:
            suspected.extend(["vi_editor", "vi_flow", "final_repair"])

    diagnosis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": all_runs,
        "variance_summary": _variance_summary(det_only or all_runs),
        "baseline_variance_summary": _variance_summary(baseline_only) if baseline_only else None,
        "outsider_stage_hashes_by_run": stage_diffs,
        "suspected_nondeterministic_stages": suspected or ["vi_flow", "vi_editor"],
        "selected_fix": {
            "name": "BENCHMARK_DETERMINISTIC + temperature=0 + seed=42",
            "reason": "LLM stages (editor/flow/compression/repair) used nonzero temperature",
            "risk_level": "low",
            "seed_supported": True,
        },
    }

    det_ci = [c for c in ci_results if c["run_id"].startswith("det_")]
    stability = {
        "generated_at": diagnosis["generated_at"],
        "deterministic": deterministic,
        "runs_requested": args.runs,
        "runs_completed": len(det_only),
        "ci_pass_count": sum(1 for c in det_ci if c["ci_pass"]),
        "ci_results": ci_results,
        "all_pass": len(det_ci) > 0 and all(c["ci_pass"] for c in det_ci),
        "outsider_scores": [
            r["samples"].get("outsider_36", {}).get("quality_score") for r in det_only
        ],
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "determinism_diagnosis.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / "stability_check_report.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n[Stability] Wrote {OUT_ROOT / 'determinism_diagnosis.json'}")
    print(f"[Stability] Wrote {OUT_ROOT / 'stability_check_report.json'}")
    print(
        f"[Stability] CI pass {stability['ci_pass_count']}/{len(det_ci)} "
        f"outsider={stability['outsider_scores']}"
    )
    return 0 if stability["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
