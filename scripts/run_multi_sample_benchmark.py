#!/usr/bin/env python3
"""Run pipeline audit across multiple drakonsub job samples and aggregate results."""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import importlib.util

from scripts.benchmark_raw_cache import (
    cache_key as raw_cache_key,
    load_cached_vi_raw,
    mode_type_for,
    save_cached_vi_raw,
    score_interpretation,
)

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


@dataclass
class BenchmarkFlags:
    requested_engine: str
    force_fresh: bool = False
    reuse_raw_manifest: bool = False
    cache_raw: bool = False
    use_raw_cache: bool = False
    only_sample: Optional[str] = None
    benchmark_mode: str = "default"  # default | pipeline_regression | end_to_end


def _parse_flags(argv: List[str]) -> BenchmarkFlags:
    import os

    engine = os.getenv("TRANSLATION_ENGINE", "openai").strip().lower()
    if "--engine" in argv:
        idx = argv.index("--engine")
        if idx + 1 < len(argv):
            engine = argv[idx + 1].strip().lower()
    only = None
    if "--sample" in argv:
        idx = argv.index("--sample")
        if idx + 1 < len(argv):
            only = argv[idx + 1]
    mode = "default"
    if "--both-modes" in argv:
        mode = "both"
    elif "--benchmark-mode" in argv or "--mode" in argv:
        flag = "--benchmark-mode" if "--benchmark-mode" in argv else "--mode"
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            mode = argv[idx + 1].strip().lower()
    return BenchmarkFlags(
        requested_engine=engine,
        force_fresh="--fresh" in argv,
        reuse_raw_manifest="--reuse-raw" in argv,
        cache_raw="--cache-raw" in argv,
        use_raw_cache="--use-raw-cache" in argv,
        only_sample=only,
        benchmark_mode=mode,
    )


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
    base: Dict[str, Any] = {
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
    shift_path = debug_dir / "cue_shift_repair_report.json"
    shift_diag_path = debug_dir / "cue_shift_diagnosis_sample.json"
    if shift_diag_path.exists():
        shift_diag = json.loads(shift_diag_path.read_text(encoding="utf-8"))
        base.update(
            {
                "local_cue_shift_windows_detected": shift_diag.get(
                    "local_cue_shift_windows_detected",
                    len(shift_diag.get("shift_windows") or []),
                ),
                "local_cue_shift_windows_repaired": shift_diag.get(
                    "local_cue_shift_windows_repaired", 0
                ),
                "window_repairs_accepted": shift_diag.get("window_repairs_accepted", 0),
                "window_repairs_rejected": shift_diag.get("window_repairs_rejected", 0),
            }
        )
    elif shift_path.exists():
        shift_rep = json.loads(shift_path.read_text(encoding="utf-8"))
        base.update(
            {
                "local_cue_shift_windows_detected": shift_rep.get("windows_detected", 0),
                "local_cue_shift_windows_repaired": shift_rep.get("windows_requested", 0),
                "window_repairs_accepted": shift_rep.get("window_repairs_accepted", 0),
                "window_repairs_rejected": shift_rep.get("window_repairs_rejected", 0),
            }
        )
    return base


def _prepare_sample_debug(
    sample: dict,
    jobs_root: Path,
    out_dir: Path,
    *,
    flags: BenchmarkFlags,
) -> Tuple[Optional[Path], bool, bool, Optional[str]]:
    job_dir = jobs_root / sample["job_id"]
    source = job_dir / "source.srt"
    if not source.exists():
        return None, False, False, None

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copy2(source, out_dir / "source.srt")

    sid = sample["id"]
    reuse_raw = False
    used_raw_cache = False
    key: Optional[str] = None

    if flags.force_fresh:
        reuse_raw = False
    elif flags.reuse_raw_manifest or sample.get("reuse_raw"):
        job_raw = job_dir / "vi_raw.srt"
        baseline_raw = DEBUG_BASELINE / "vi_raw.srt"
        if sample["id"] == "buffett_bitcoin_29" and baseline_raw.exists():
            shutil.copy2(baseline_raw, out_dir / "vi_raw.srt")
            reuse_raw = True
        elif job_raw.exists():
            shutil.copy2(job_raw, out_dir / "vi_raw.srt")
            reuse_raw = True

    if not reuse_raw and not flags.force_fresh and flags.use_raw_cache:
        cached = load_cached_vi_raw(sid, source, flags.requested_engine)
        if cached:
            shutil.copy2(cached, out_dir / "vi_raw.srt")
            reuse_raw = True
            used_raw_cache = True
            key = raw_cache_key(source, flags.requested_engine)

    return source, reuse_raw, used_raw_cache, key


def _write_summary(report: dict, path: Path) -> None:
    lines = [
        "# Multi-sample Benchmark Summary",
        "",
        f"Generated: {report.get('generated_at')}",
        f"Benchmark mode: **{report.get('benchmark_mode', 'default')}**",
        f"Engine requested: **{report.get('translation_engine_requested')}** "
        f"(status: {report.get('benchmark_engine_status')})",
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
        "| Sample | Mode type | Contract | Quality | Risky | Semantic |",
        "|--------|-----------|----------|---------|-------|----------|",
    ]
    for item in report.get("samples", []):
        if item.get("status") != "ok":
            lines.append(
                f"| {item['id']} | — | **{item['status']}** | — | — | — |"
            )
            continue
        m = item["metrics"]
        bm = item.get("benchmark_mode", {})
        lines.append(
            f"| {item['id']} | {bm.get('mode_type', '—')} | "
            f"{m.get('pipeline_contract_status')} | {m.get('quality_score')} | "
            f"{m.get('risky_cue_count')} | {m.get('semantic_alignment_errors')} |"
        )
    lines.extend(["", "## Deferred", ""])
    for d in report.get("deferred", []):
        lines.append(f"- **{d['id']}** ({d.get('cue_count')} cues): {d.get('reason')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_benchmark_pass(
    samples: List[dict],
    jobs_root: Path,
    out_root: Path,
    flags: BenchmarkFlags,
    *,
    pass_name: str,
    pass_flags: BenchmarkFlags,
) -> Tuple[dict, int]:
    import os

    os.environ["TRANSLATION_ENGINE"] = flags.requested_engine
    results: List[dict] = []
    engine_samples: List[dict] = []
    engine_valid = True

    for sample in samples:
        sid = sample["id"]
        out_dir = out_root / sid
        print(
            f"\n[Benchmark:{pass_name}] === {sid} ({sample.get('cue_count')} cues) "
            f"engine={pass_flags.requested_engine} ==="
        )
        source, reuse_raw, used_raw_cache, cache_key = _prepare_sample_debug(
            sample, jobs_root, out_dir, flags=pass_flags
        )
        if source is None:
            print(f"[Benchmark] SKIP {sid}: missing source.srt")
            results.append({"id": sid, "status": "missing_source", "job_id": sample["job_id"]})
            continue

        mode_type = mode_type_for(
            reuse_raw=reuse_raw,
            used_raw_cache=used_raw_cache,
            fresh_translate=not reuse_raw,
        )
        mode = "error"
        effective = pass_flags.requested_engine
        t0 = time.time()
        try:
            run_meta = _run_pipeline_stages(
                out_dir,
                job_source=source,
                reuse_raw=reuse_raw,
                translation_engine=pass_flags.requested_engine,
            )
            _build_reports(out_dir, run_meta)
            metrics = _sample_result(out_dir)

            if pass_flags.cache_raw and not used_raw_cache:
                vi_raw = out_dir / "vi_raw.srt"
                if vi_raw.exists():
                    save_cached_vi_raw(
                        sid,
                        source,
                        pass_flags.requested_engine,
                        vi_raw,
                        extra={"benchmark_pass": pass_name},
                    )
                    if not cache_key:
                        cache_key = raw_cache_key(source, pass_flags.requested_engine)

            mode = run_meta.get("translation_mode", "fresh_translate")
            effective = run_meta.get("translation_engine_effective", pass_flags.requested_engine)
            is_valid = mode in ("reuse_raw", "cached_raw") or effective == pass_flags.requested_engine
            if not is_valid:
                engine_valid = False
            engine_samples.append(
                {
                    "sample": sid,
                    "mode": mode,
                    "mode_type": mode_type,
                    "translation_engine_requested": pass_flags.requested_engine,
                    "translation_engine_effective": effective,
                    "raw_translation_cache_key": cache_key,
                    "raw_translation_cached": used_raw_cache or pass_flags.cache_raw,
                    "is_valid": is_valid,
                }
            )
            status = "ok"
            if metrics.get("pipeline_contract_status") != "pass":
                status = "contract_fail"
        except Exception as exc:
            metrics = {"error": str(exc)}
            status = "error"
            print(f"[Benchmark] ERROR {sid}: {exc}")
            mode_type = "unknown"
            cache_key = None

        elapsed = round(time.time() - t0, 1)
        benchmark_mode_meta = {
            "mode": mode if status == "ok" else "error",
            "mode_type": mode_type,
            "translation_engine_requested": pass_flags.requested_engine,
            "translation_engine_effective": (
                effective if status == "ok" else pass_flags.requested_engine
            ),
            "raw_translation_cache_key": cache_key,
            "raw_translation_cached": used_raw_cache,
            "score_interpretation": score_interpretation(mode_type),
        }
        entry = {
            "id": sid,
            "job_id": sample["job_id"],
            "status": status,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "benchmark_mode": benchmark_mode_meta,
            "artifact_dir": str(out_dir),
        }
        results.append(entry)
        if status == "ok":
            print(
                f"[Benchmark:{pass_name}] {sid}: mode_type={mode_type} "
                f"quality={metrics.get('quality_score')} "
                f"semantic={metrics.get('semantic_alignment_errors')} ({elapsed}s)"
            )

    ok = [r for r in results if r.get("status") == "ok"]
    scores = [
        r["metrics"]["quality_score"]
        for r in ok
        if r["metrics"].get("quality_score") is not None
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_mode": pass_name,
        "translation_engine_requested": pass_flags.requested_engine,
        "translation_engine_env": os.getenv("TRANSLATION_ENGINE", "openai"),
        "benchmark_engine_status": "pass" if engine_valid else "invalid_engine_config",
        "manifest": str(MANIFEST),
        "samples": results,
        "deferred": _load_manifest().get("deferred", []),
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
    return report, 0 if all(r.get("status") == "ok" for r in results) else 1


def _write_artifacts(report: dict, out_root: Path, engine_samples: List[dict]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "benchmark_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    engine_report = {
        "requested_engine": report.get("translation_engine_requested"),
        "env_translation_engine": report.get("translation_engine_env"),
        "samples": engine_samples,
        "benchmark_engine_status": report.get("benchmark_engine_status"),
        "benchmark_mode": report.get("benchmark_mode"),
    }
    (out_root / "engine_selection_report.json").write_text(
        json.dumps(engine_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary(report, out_root / "benchmark_summary.md")


def _run_diagnosis_scripts() -> None:
    import subprocess

    for name in (
        "build_cross_sample_diagnosis.py",
        "build_cue_shift_diagnosis.py",
        "build_no_rush_diagnosis.py",
        "build_outsider_diagnosis.py",
    ):
        script = ROOT / "scripts" / name
        if script.exists():
            subprocess.run([sys.executable, str(script)], check=False, cwd=str(ROOT))


def main() -> int:
    import os

    flags = _parse_flags(sys.argv)
    from auto_subtitle.config import SUPPORTED_TRANSLATION_ENGINES

    if flags.requested_engine not in SUPPORTED_TRANSLATION_ENGINES:
        print(f"Unsupported engine: {flags.requested_engine}", file=sys.stderr)
        return 1

    manifest = _load_manifest()
    jobs_root = Path(manifest["jobs_root"])
    samples: List[dict] = manifest["samples"]
    if flags.only_sample:
        samples = [s for s in samples if s["id"] == flags.only_sample]
        if not samples:
            print(f"Unknown sample id: {flags.only_sample}", file=sys.stderr)
            return 1

    exit_code = 0
    engine_samples_all: List[dict] = []

    if flags.benchmark_mode in ("both", "end_to_end"):
        ef = BenchmarkFlags(
            requested_engine=flags.requested_engine,
            force_fresh=True,
            cache_raw=True,
            only_sample=flags.only_sample,
        )
        e2e_root = OUT_ROOT / "end_to_end"
        report, code = _run_benchmark_pass(
            samples, jobs_root, e2e_root, flags, pass_name="end_to_end", pass_flags=ef
        )
        _write_artifacts(report, e2e_root, [])
        exit_code = max(exit_code, code)

    if flags.benchmark_mode in ("both", "pipeline_regression"):
        pf = BenchmarkFlags(
            requested_engine=flags.requested_engine,
            use_raw_cache=True,
            reuse_raw_manifest=True,
            only_sample=flags.only_sample,
        )
        pr_root = OUT_ROOT / "pipeline_regression"
        report, code = _run_benchmark_pass(
            samples, jobs_root, pr_root, flags, pass_name="pipeline_regression", pass_flags=pf
        )
        engine_samples_all = [
            s for item in report.get("samples", []) for s in [{}]
        ]
        # collect engine samples from pass - re-read from report samples
        for item in report.get("samples", []):
            bm = item.get("benchmark_mode", {})
            engine_samples_all.append(
                {
                    "sample": item["id"],
                    "mode": bm.get("mode"),
                    "mode_type": bm.get("mode_type"),
                    "translation_engine_requested": flags.requested_engine,
                    "translation_engine_effective": bm.get("translation_engine_effective"),
                    "raw_translation_cache_key": bm.get("raw_translation_cache_key"),
                    "raw_translation_cached": bm.get("raw_translation_cached"),
                    "is_valid": True,
                }
            )
        _write_artifacts(report, pr_root, engine_samples_all)
        shutil.copy2(pr_root / "benchmark_report.json", OUT_ROOT / "benchmark_report.json")
        shutil.copy2(pr_root / "benchmark_summary.md", OUT_ROOT / "benchmark_summary.md")
        shutil.copy2(
            pr_root / "engine_selection_report.json", OUT_ROOT / "engine_selection_report.json"
        )
        for sid in [s["id"] for s in samples]:
            src = pr_root / sid
            dst = OUT_ROOT / sid
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        exit_code = max(exit_code, code)

    if flags.benchmark_mode == "end_to_end":
        shutil.copy2(e2e_root / "benchmark_report.json", OUT_ROOT / "benchmark_report.json")
        shutil.copy2(e2e_root / "benchmark_summary.md", OUT_ROOT / "benchmark_summary.md")
        for sid in [s["id"] for s in samples]:
            src = e2e_root / sid
            dst = OUT_ROOT / sid
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

    if flags.benchmark_mode == "default":
        pass_flags = BenchmarkFlags(
            requested_engine=flags.requested_engine,
            force_fresh=flags.force_fresh,
            reuse_raw_manifest=flags.reuse_raw_manifest,
            cache_raw=flags.cache_raw,
            use_raw_cache=flags.use_raw_cache,
            only_sample=flags.only_sample,
        )
        report, code = _run_benchmark_pass(
            samples,
            jobs_root,
            OUT_ROOT,
            flags,
            pass_name="default",
            pass_flags=pass_flags,
        )
        engine_samples_all = []
        for item in report.get("samples", []):
            bm = item.get("benchmark_mode", {})
            engine_samples_all.append(
                {
                    "sample": item["id"],
                    "mode": bm.get("mode"),
                    "mode_type": bm.get("mode_type"),
                    "translation_engine_requested": flags.requested_engine,
                    "translation_engine_effective": bm.get("translation_engine_effective"),
                    "raw_translation_cache_key": bm.get("raw_translation_cache_key"),
                    "raw_translation_cached": bm.get("raw_translation_cached"),
                    "is_valid": True,
                }
            )
        _write_artifacts(report, OUT_ROOT, engine_samples_all)
        exit_code = max(exit_code, code)

    _run_diagnosis_scripts()
    print(f"\n[Benchmark] Done → {OUT_ROOT / 'benchmark_report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
