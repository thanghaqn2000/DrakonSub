#!/usr/bin/env python3
"""Fail CI/local runs when multi-sample benchmark guardrails are not met."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT / "artifacts" / "multi_sample_benchmark" / "pipeline_regression" / "benchmark_report.json"
)
DEFAULT_ENGINE_REPORT = ROOT / "artifacts" / "multi_sample_benchmark" / "engine_selection_report.json"

DEFAULT_GUARDRAILS: Dict[str, Any] = {
    "contract_pass_required": 4,
    "contract_pass_total": 4,
    "quality_min": 70,
    "quality_avg": 80,
    "sample_quality_min": {
        "raise_price_17": 70,
        "no_rush_19": 70,
        "buffett_bitcoin_29": 80,
        "outsider_36": 75,
    },
    "semantic_alignment_errors_max": 3,
    "missing_or_empty_cue_errors_max": 0,
    "post_final_repair_text_lock_required": "pass",
    "engine_selection_required": "pass",
}

SOFT_GUARDRAILS: Dict[str, Any] = {
    **DEFAULT_GUARDRAILS,
    "quality_min": 60,
    "quality_avg": 70,
    "sample_quality_min": {
        "raise_price_17": 65,
        "no_rush_19": 65,
        "buffett_bitcoin_29": 70,
        "outsider_36": 65,
    },
    "semantic_alignment_errors_max": 6,
}


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _confirmed_missing_empty(metrics: dict) -> int:
    counts = metrics.get("confirmed_error_counts") or {}
    return int(counts.get("missing_or_empty_cue_error", 0))


def evaluate_report(
    report: dict,
    engine_report: Optional[dict],
    guardrails: Dict[str, Any],
) -> List[str]:
    failures: List[str] = []
    agg = report.get("aggregate") or {}
    samples = [s for s in report.get("samples", []) if s.get("status") == "ok"]

    contract_pass = int(agg.get("contract_pass") or 0)
    samples_run = int(agg.get("samples_run") or len(report.get("samples") or []))
    required_pass = int(guardrails.get("contract_pass_required", 4))
    required_total = int(guardrails.get("contract_pass_total", samples_run or 4))

    if contract_pass < required_pass or samples_run < required_total:
        failures.append(
            f"contract pass {contract_pass}/{samples_run} < {required_pass}/{required_total}"
        )

    qmin = agg.get("quality_score_min")
    qavg = agg.get("quality_score_avg")
    if qmin is not None and qmin < guardrails["quality_min"]:
        failures.append(f"quality_min {qmin} < {guardrails['quality_min']}")
    if qavg is not None and qavg < guardrails["quality_avg"]:
        failures.append(f"quality_avg {qavg:.1f} < {guardrails['quality_avg']}")

    per_sample_min = guardrails.get("sample_quality_min") or {}
    for sample in samples:
        sid = sample.get("id")
        if sid not in per_sample_min:
            continue
        m = sample.get("metrics") or {}
        score = m.get("quality_score")
        floor = per_sample_min[sid]
        if score is None:
            failures.append(f"{sid} missing quality_score")
        elif score < floor:
            failures.append(f"{sid} quality {score} < {floor}")

        lock = m.get("post_final_repair_text_lock_status")
        if guardrails.get("post_final_repair_text_lock_required") == "pass" and lock != "pass":
            failures.append(f"{sid} post_final_repair_text_lock {lock!r} != pass")

        contract = m.get("pipeline_contract_status")
        if contract != "pass":
            failures.append(f"{sid} pipeline_contract_status {contract!r} != pass")

        missing_empty = _confirmed_missing_empty(m)
        if missing_empty > guardrails["missing_or_empty_cue_errors_max"]:
            failures.append(
                f"{sid} missing_or_empty_cue_errors {missing_empty} > "
                f"{guardrails['missing_or_empty_cue_errors_max']}"
            )

    total_semantic = sum(
        int((s.get("metrics") or {}).get("semantic_alignment_errors") or 0) for s in samples
    )
    if total_semantic > guardrails["semantic_alignment_errors_max"]:
        failures.append(
            f"semantic_alignment_errors_total {total_semantic} > "
            f"{guardrails['semantic_alignment_errors_max']}"
        )

    engine_status = report.get("benchmark_engine_status")
    if engine_report:
        engine_status = engine_report.get("benchmark_engine_status", engine_status)
    if guardrails.get("engine_selection_required") == "pass" and engine_status != "pass":
        failures.append(f"benchmark_engine_status {engine_status!r} != pass")

    for sample in report.get("samples", []):
        if sample.get("status") not in ("ok",):
            failures.append(f"{sample.get('id')} benchmark status {sample.get('status')}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check multi-sample benchmark guardrails")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path to benchmark_report.json",
    )
    parser.add_argument(
        "--engine-report",
        type=Path,
        default=DEFAULT_ENGINE_REPORT,
        help="Path to engine_selection_report.json (optional)",
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Use softer guardrails for end-to-end / variance reports",
    )
    args = parser.parse_args()

    guardrails = SOFT_GUARDRAILS if args.soft else DEFAULT_GUARDRAILS
    report = _load_json(args.report)
    engine_report = None
    engine_path = args.engine_report
    if engine_path.exists():
        engine_report = _load_json(engine_path)
    elif (args.report.parent / "engine_selection_report.json").exists():
        engine_report = _load_json(args.report.parent / "engine_selection_report.json")

    failures = evaluate_report(report, engine_report, guardrails)
    mode = report.get("benchmark_mode", "unknown")
    agg = report.get("aggregate") or {}

    if failures:
        print("FAIL regression gate")
        for item in failures:
            print(f"- {item}")
        print(
            f"\n(mode={mode}, quality_min={agg.get('quality_score_min')}, "
            f"quality_avg={agg.get('quality_score_avg')})"
        )
        return 1

    print("PASS regression gate")
    print(
        f"mode={mode} contract={agg.get('contract_pass')}/{agg.get('samples_run')} "
        f"quality_min={agg.get('quality_score_min')} quality_avg={agg.get('quality_score_avg')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
