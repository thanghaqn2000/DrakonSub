#!/usr/bin/env python3
"""Run pipeline audit across multiple drakonsub job samples and aggregate results."""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import importlib.util

_audit_path = ROOT / "scripts" / "run_cue_mapping_audit.py"
_spec = importlib.util.spec_from_file_location("run_cue_mapping_audit", _audit_path)
assert _spec and _spec.loader
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)

JOB_SOURCE = _audit.JOB_SOURCE
_build_reports = _audit._build_reports
_run_pipeline_stages = _audit._run_pipeline_stages

MANIFEST = ROOT / "scripts" / "benchmark_samples.json"
OUT_ROOT = ROOT / "artifacts" / "multi_sample_benchmark"
DEBUG_BASELINE = ROOT / "debug"


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sample_result(debug_dir: Path) -> Dict[str, Any]:
    contract = {}
    quality = {}
    cps = {}
    contract_path = debug_dir / "pipeline_contract_report.json"
    quality_path = debug_dir / "translation_quality_report.json"
    cps_path = debug_dir / "cps_diagnosis_report.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if cps_path.exists():
        cps = json.loads(cps_path.read_text(encoding="utf-8"))

    pre_timing = quality.get("pre_timing_after_repair") or {}
    return {
        "pipeline_contract_status": contract.get("pipeline_contract_status"),
        "post_final_repair_text_lock_status": contract.get(
            "post_final_repair_text_lock_status"
        ),
        "source_cue_count": contract.get("source_cue_count"),
        "final_cue_count": (contract.get("stage_cue_counts") or {}).get("final_vi"),
        "quality_score": quality.get("quality_score"),
        "score_band": quality.get("score_band"),
        "risky_cue_count": quality.get("summary", {}).get("risky_cue_count"),
        "semantic_alignment_errors": (
            quality.get("semantic_alignment", {})
            .get("summary", {})
            .get("semantic_alignment_errors")
        ),
        "confirmed_error_counts": quality.get("summary", {}).get(
            "confirmed_error_counts", {}
        ),
        "pre_timing_quality_score": pre_timing.get("quality_score"),
        "pre_timing_risky_cue_count": pre_timing.get("risky_cue_count"),
        "cps_error_count_pre_timing": cps.get("cps_error_count_pre_timing"),
        "cps_error_count_post_timing": cps.get("cps_error_count_post_timing"),
        "human_review_needed": quality.get("human_review_needed", False),
    }


def _prepare_sample_debug(
    sample: dict,
    jobs_root: Path,
    out_dir: Path,
    *,
    reuse_raw: bool,
) -> Path | None:
    job_dir = jobs_root / sample["job_id"]
    source = job_dir / "source.srt"
    if not source.exists():
        return None

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copy2(source, out_dir / "source.srt")

    if reuse_raw:
        job_raw = job_dir / "vi_raw.srt"
        baseline_raw = DEBUG_BASELINE / "vi_raw.srt"
        if sample["id"] == "buffett_bitcoin_29" and baseline_raw.exists():
            shutil.copy2(baseline_raw, out_dir / "vi_raw.srt")
        elif job_raw.exists():
            shutil.copy2(job_raw, out_dir / "vi_raw.srt")

    return source


def _write_summary(report: dict, path: Path) -> None:
    lines = [
        "# Multi-sample Benchmark Summary",
        "",
        f"Generated: {report.get('generated_at')}",
        f"Engine: {report.get('translation_engine')}",
        "",
        "## Aggregate",
        "",
        f"- Samples run: **{report['aggregate']['samples_run']}**",
        f"- Contract pass: **{report['aggregate']['contract_pass']}** / {report['aggregate']['samples_run']}",
        f"- Quality score min/avg/max: **{report['aggregate']['quality_score_min']}** / "
        f"**{report['aggregate']['quality_score_avg']:.1f}** / "
        f"**{report['aggregate']['quality_score_max']}**",
        f"- Any semantic alignment errors: **{report['aggregate']['any_semantic_errors']}**",
        "",
        "## Per sample",
        "",
        "| Sample | Cues | Contract | Quality | Risky | Semantic err | CPS post |",
        "|--------|------|----------|---------|-------|--------------|----------|",
    ]
    for item in report.get("samples", []):
        if item.get("status") != "ok":
            lines.append(
                f"| {item['id']} | — | **{item['status']}** | — | — | — | — |"
            )
            continue
        m = item["metrics"]
        lines.append(
            f"| {item['id']} | {m.get('source_cue_count')} | "
            f"{m.get('pipeline_contract_status')} | {m.get('quality_score')} | "
            f"{m.get('risky_cue_count')} | {m.get('semantic_alignment_errors')} | "
            f"{m.get('cps_error_count_post_timing')} |"
        )
    lines.extend(["", "## Deferred", ""])
    for d in report.get("deferred", []):
        lines.append(f"- **{d['id']}** ({d.get('cue_count')} cues): {d.get('reason')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import os

    manifest = _load_manifest()
    jobs_root = Path(manifest["jobs_root"])
    only = None
    if "--sample" in sys.argv:
        idx = sys.argv.index("--sample")
        if idx + 1 < len(sys.argv):
            only = sys.argv[idx + 1]

    samples: List[dict] = manifest["samples"]
    if only:
        samples = [s for s in samples if s["id"] == only]
        if not samples:
            print(f"Unknown sample id: {only}", file=sys.stderr)
            return 1

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []
    engine = os.getenv("TRANSLATION_ENGINE", "openai").strip().lower()

    for sample in samples:
        sid = sample["id"]
        out_dir = OUT_ROOT / sid
        print(f"\n[Benchmark] === {sid} ({sample.get('cue_count')} cues) ===")
        source = _prepare_sample_debug(
            sample,
            jobs_root,
            out_dir,
            reuse_raw=bool(sample.get("reuse_raw")),
        )
        if source is None:
            print(f"[Benchmark] SKIP {sid}: missing source.srt")
            results.append({"id": sid, "status": "missing_source", "job_id": sample["job_id"]})
            continue

        t0 = time.time()
        try:
            run_meta = _run_pipeline_stages(
                out_dir,
                job_source=source,
                reuse_raw=bool(sample.get("reuse_raw")),
            )
            _build_reports(out_dir, run_meta)
            metrics = _sample_result(out_dir)
            status = "ok"
            if metrics.get("pipeline_contract_status") != "pass":
                status = "contract_fail"
        except Exception as exc:
            metrics = {"error": str(exc)}
            status = "error"
            print(f"[Benchmark] ERROR {sid}: {exc}")

        elapsed = round(time.time() - t0, 1)
        entry = {
            "id": sid,
            "job_id": sample["job_id"],
            "status": status,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "artifact_dir": str(out_dir),
        }
        results.append(entry)
        if status == "ok":
            print(
                f"[Benchmark] {sid}: contract={metrics.get('pipeline_contract_status')} "
                f"quality={metrics.get('quality_score')} "
                f"risky={metrics.get('risky_cue_count')} "
                f"semantic={metrics.get('semantic_alignment_errors')} "
                f"({elapsed}s)"
            )

    ok = [r for r in results if r.get("status") == "ok"]
    scores = [r["metrics"]["quality_score"] for r in ok if r["metrics"].get("quality_score") is not None]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "translation_engine": engine,
        "manifest": str(MANIFEST),
        "samples": results,
        "deferred": manifest.get("deferred", []),
        "aggregate": {
            "samples_run": len(results),
            "samples_ok": len(ok),
            "contract_pass": sum(
                1 for r in ok if r["metrics"].get("pipeline_contract_status") == "pass"
            ),
            "quality_score_min": min(scores) if scores else None,
            "quality_score_max": max(scores) if scores else None,
            "quality_score_avg": sum(scores) / len(scores) if scores else None,
            "any_semantic_errors": any(
                (r["metrics"].get("semantic_alignment_errors") or 0) > 0 for r in ok
            ),
            "total_risky_cues": sum(r["metrics"].get("risky_cue_count") or 0 for r in ok),
        },
    }

    report_path = OUT_ROOT / "benchmark_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary(report, OUT_ROOT / "benchmark_summary.md")

    import subprocess

    diag_script = ROOT / "scripts" / "build_cross_sample_diagnosis.py"
    if diag_script.exists():
        subprocess.run([sys.executable, str(diag_script)], check=False, cwd=str(ROOT))

    print(f"\n[Benchmark] Done → {report_path}")
    return 0 if all(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
