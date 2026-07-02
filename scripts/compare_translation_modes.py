#!/usr/bin/env python3
"""Compare translation modes cue-by-cue and produce failure attribution report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from auto_subtitle.raw_hybrid_guarded_translate import analyze_with_severity  # noqa: E402
from auto_subtitle.subtitle_timing_optimizer import _parse_ts  # noqa: E402
from auto_subtitle.utils import parse_srt  # noqa: E402
from auto_subtitle.vi_compression import _cps  # noqa: E402

MANIFEST = ROOT / "scripts" / "benchmark_samples.json"
BENCH_ROOT = ROOT / "artifacts" / "multi_sample_benchmark"
OUT_ROOT = ROOT / "artifacts" / "translation_quality_review"
MODE_RUNS = OUT_ROOT / "mode_runs"

SAMPLES = ["raise_price_17", "no_rush_19", "buffett_bitcoin_29", "outsider_36"]
RAW_MODES = ("grouped", "cue_keyed", "hybrid_guarded", "span_guarded", "span_guarded_conservative", "span_guarded_tiered")

FAILURE_LAYERS = (
    "raw_alignment",
    "raw_literalness",
    "raw_hallucination",
    "editor_overedit",
    "flow_repeated_meaning",
    "compression_loss",
    "timing_cps",
    "qa_false_positive",
    "source_segmentation_fragment",
    "unknown",
)

NEXT_ACTIONS = (
    "guard_severity_tuning",
    "raw_prompt_polish",
    "editor_flow_constraint",
    "qa_calibration",
    "timing_readability",
    "mixed",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _srt_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    entries = parse_srt(path.read_text(encoding="utf-8"))
    return {i + 1: (e.get("text") or "").strip() for i, e in enumerate(entries)}


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


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _overlap(a: str, b: str) -> float:
    aw = set(re.findall(r"\w+", (a or "").lower(), flags=re.UNICODE))
    bw = set(re.findall(r"\w+", (b or "").lower(), flags=re.UNICODE))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / max(len(aw), len(bw))


def _mode_sample_dir(mode: str, sample_id: str) -> Path:
    if mode == "hybrid_guarded":
        snap = MODE_RUNS / "hybrid_guarded" / sample_id
        if snap.exists():
            return snap
        return BENCH_ROOT / "end_to_end" / sample_id
    if mode in ("span_guarded", "span_guarded_conservative", "span_guarded_tiered"):
        snap = MODE_RUNS / mode / sample_id
        if snap.exists():
            return snap
        if mode == "span_guarded":
            return BENCH_ROOT / "end_to_end" / sample_id
        return MODE_RUNS / mode / sample_id
    if mode == "pipeline_regression":
        return BENCH_ROOT / "pipeline_regression" / sample_id
    return MODE_RUNS / mode / sample_id


def _ensure_hybrid_snapshot() -> None:
    dest_root = MODE_RUNS / "hybrid_guarded"
    if dest_root.exists():
        return
    src_root = BENCH_ROOT / "end_to_end"
    if not src_root.exists():
        return
    dest_root.mkdir(parents=True, exist_ok=True)
    for sid in SAMPLES:
        src = src_root / sid
        if src.exists():
            shutil.copytree(src, dest_root / sid, dirs_exist_ok=True)


def _run_mode_benchmark(raw_mode: str, engine: str = "openai") -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_multi_sample_benchmark.py"),
        "--engine",
        engine,
        "--fresh",
        "--cache-raw",
        "--mode",
        "end_to_end",
        "--raw-translation-mode",
        raw_mode,
        "--deterministic",
    ]
    print(f"[compare_modes] Running e2e benchmark raw_mode={raw_mode} ...")
    subprocess.run(cmd, cwd=ROOT, check=True)
    dest_root = MODE_RUNS / raw_mode
    dest_root.mkdir(parents=True, exist_ok=True)
    src_root = BENCH_ROOT / "end_to_end"
    for sid in SAMPLES:
        src = src_root / sid
        if src.exists():
            shutil.copytree(src, dest_root / sid, dirs_exist_ok=True)
    if (src_root / "benchmark_report.json").exists():
        shutil.copy2(src_root / "benchmark_report.json", dest_root / "benchmark_report.json")


def _load_mode_sample(mode: str, sample_id: str) -> Optional[dict]:
    work = _mode_sample_dir(mode, sample_id)
    if not work.exists():
        return None
    source_path = work / "source.srt"
    if not source_path.exists():
        source_path = work / "source_corrected.srt"
    if not source_path.exists():
        return None
    source_entries = parse_srt(source_path.read_text(encoding="utf-8"))
    source_en = _srt_map(work / "source_corrected.srt") or _srt_map(work / "source.srt")
    vi_raw = _srt_map(work / "vi_raw.srt")
    vi_editor = _srt_map(work / "vi_after_editor.srt")
    vi_flow = _srt_map(work / "vi_after_flow.srt")
    vi_final = _srt_map(work / "final_vi.srt")

    quality_report: dict = {}
    qr_path = work / "translation_quality_report.json"
    if qr_path.exists():
        quality_report = json.loads(qr_path.read_text(encoding="utf-8"))

    hybrid_report: dict = {}
    hr_path = work / "hybrid_guarded_report.json"
    if hr_path.exists():
        hybrid_report = json.loads(hr_path.read_text(encoding="utf-8"))

    return {
        "work_dir": str(work),
        "source_entries": source_entries,
        "source_en": source_en,
        "vi_raw": vi_raw,
        "vi_editor": vi_editor,
        "vi_flow": vi_flow,
        "vi_final": vi_final,
        "quality_report": quality_report,
        "quality_score": quality_report.get("quality_score"),
        "hybrid_report": hybrid_report,
    }


def _assessment_map(quality_report: dict) -> Dict[int, dict]:
    return {a["cue_index"]: a for a in quality_report.get("cue_assessments") or []}


def _guard_flags_for_cue(
    mode: str, cue_index: int, grouped_data: Optional[dict], hybrid_data: Optional[dict]
) -> List[str]:
    if mode == "hybrid_guarded" and hybrid_data:
        hr = hybrid_data.get("hybrid_report") or {}
        for item in (hr.get("before") or {}).get("flags") or []:
            if item.get("cue_index") == cue_index:
                return list(item.get("flags") or [])
        for rep in hr.get("repairs") or []:
            if rep.get("cue_index") == cue_index:
                return [rep.get("repair_type", "repair")]
    if mode == "grouped" and grouped_data:
        vi_entries = []
        for i, e in enumerate(grouped_data["source_entries"], start=1):
            vi_entries.append({"text": grouped_data["vi_raw"].get(i, "")})
        analysis = analyze_with_severity(grouped_data["source_entries"], vi_entries)
        for item in analysis.get("flags") or []:
            if item.get("cue_index") == cue_index:
                return list(item.get("flags") or [])
    return []


def _repair_info(mode: str, cue_index: int, hybrid_data: Optional[dict]) -> dict:
    if mode != "hybrid_guarded" or not hybrid_data:
        return {}
    hr = hybrid_data.get("hybrid_report") or {}
    for rep in hr.get("repairs") or []:
        if rep.get("cue_index") == cue_index:
            return rep
    return {}


def _infer_failure_layer(
    errors: List[str],
    source_flags: List[str],
    vi_raw: str,
    vi_editor: str,
    vi_flow: str,
    vi_final: str,
    cps_val: float,
    duration: float,
) -> str:
    err = set(errors or [])
    if "readability_cps_error" in err and duration < 2.5 and cps_val > 18:
        return "timing_cps"
    if err & {"semantic_alignment_error", "semantic_drift_error"}:
        if vi_raw and vi_editor == vi_flow == vi_final and vi_raw == vi_final:
            return "raw_alignment"
        if vi_editor != vi_raw:
            return "editor_overedit"
        return "raw_alignment"
    if err & {"repeated_meaning_error", "cue_flow_error"}:
        if vi_flow != vi_editor or vi_final != vi_flow:
            return "flow_repeated_meaning"
        return "flow_repeated_meaning"
    if err & {"readability_cps_error", "over_compression_error"}:
        return "compression_loss" if vi_final != vi_flow else "timing_cps"
    if source_flags and not err:
        return "qa_false_positive"
    if source_flags and err & {"split_term_across_cues_error", "cue_fragmentation_error"}:
        return "source_segmentation_fragment"
    if err & {"literal_translation_error", "domain_term_error"}:
        return "raw_literalness"
    return "unknown"


def _risk_signature(
    en: str,
    vi: str,
    errors: List[str],
    cps_val: float,
    duration: float,
    source_flags: List[str],
) -> List[str]:
    sig: List[str] = []
    en_wc = _word_count(en)
    vi_wc = _word_count(vi)
    if en_wc <= 4 and vi_wc >= en_wc * 2 + 2:
        sig.append("short_source_long_vi")
    if duration < 2.5 and cps_val > 18:
        sig.append("high_cps_short_duration")
    if "repeated_meaning_error" in errors:
        sig.append("rhetorical_repeat")
    if "split_term_across_cues_error" in (source_flags or []) or "cue_fragmentation_error" in (
        source_flags or []
    ):
        sig.append("fragment_continuation")
    if _overlap(en, vi) < 0.08 and en_wc >= 4:
        sig.append("low_semantic_overlap")
    if re.search(r"\b(bitcoin|buffett|berkshire|purchasing agent|bribe)\b", en, re.I):
        sig.append("proper_noun_or_domain_term")
    if "semantic_alignment_error" in errors and en_wc >= 5:
        bleed_terms = set(re.findall(r"\w+", en.lower())) & set(re.findall(r"\w+", vi.lower()))
        if len(bleed_terms) >= 3 and _overlap(en, vi) < 0.35:
            sig.append("neighbor_concept_bleed")
    return sig


def _stage_where_degraded(
    vi_raw: str, vi_editor: str, vi_flow: str, vi_final: str, errors: List[str]
) -> str:
    if not errors:
        return "none"
    if vi_raw != vi_editor:
        return "editor"
    if vi_editor != vi_flow:
        return "flow"
    if vi_flow != vi_final:
        return "compression_or_timing"
    if errors:
        return "raw"
    return "unknown"


def _human_verdict_no_rush(cue_index: int, en: str, vi: str, errors: List[str]) -> str:
    en_l = (en or "").lower()
    vi_l = (vi or "").lower()
    if cue_index == 17 and "take no" in en_l and "xây dựng" in vi_l:
        return "true_error"
    if cue_index == 12 and "know what i mean" in en_l and "phàn nàn" in vi_l:
        return "true_error"
    if "readability_cps_error" in errors and _word_count(en) <= 5:
        return "uncertain"
    if "repeated_meaning_error" in errors and "build it" in en_l:
        return "qa_false_positive"
    if errors:
        return "uncertain"
    return "uncertain"


def _recommended_action(layer: str, signatures: List[str]) -> str:
    if layer in ("raw_alignment", "raw_literalness", "raw_hallucination"):
        if "neighbor_concept_bleed" in signatures:
            return "guard_severity_tuning"
        return "raw_prompt_polish"
    if layer in ("editor_overedit", "flow_repeated_meaning"):
        return "editor_flow_constraint"
    if layer in ("timing_cps", "compression_loss"):
        return "timing_readability"
    if layer == "qa_false_positive":
        return "qa_calibration"
    if layer == "source_segmentation_fragment":
        return "mixed"
    return "mixed"


def _best_mode_for_cue(mode_finals: Dict[str, str], mode_errors: Dict[str, List[str]]) -> Tuple[str, str]:
    scored: List[Tuple[str, int]] = []
    for mode, final in mode_finals.items():
        if not final:
            continue
        penalty = len(mode_errors.get(mode) or []) * 10
        scored.append((mode, 100 - penalty))
    if not scored:
        return "", ""
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0]
    reason = f"fewest_auto_errors among available finals"
    return best, reason


def _build_cue_rows(mode_data: Dict[str, Dict[str, dict]]) -> List[dict]:
    rows: List[dict] = []
    for sample_id in SAMPLES:
        base = mode_data.get("grouped") or mode_data.get("hybrid_guarded") or {}
        sample_base = base.get(sample_id)
        if not sample_base:
            continue
        n = len(sample_base["source_entries"])
        grouped = mode_data.get("grouped", {}).get(sample_id)
        cue_keyed = mode_data.get("cue_keyed", {}).get(sample_id)
        hybrid = mode_data.get("hybrid_guarded", {}).get(sample_id)
        pipeline = mode_data.get("pipeline_regression", {}).get(sample_id)

        for cue_index in range(1, n + 1):
            en = sample_base["source_en"].get(cue_index, "")
            dur = _duration(sample_base["source_entries"], cue_index)

            def _pack(mode: str, data: Optional[dict], field: str) -> str:
                if not data:
                    return ""
                return data[field].get(cue_index, "")

            def _errors(mode: str, data: Optional[dict]) -> List[str]:
                if not data:
                    return []
                a = _assessment_map(data["quality_report"]).get(cue_index, {})
                return list(a.get("detected_translation_errors") or [])

            def _cps_for(data: Optional[dict], field: str) -> Any:
                if not data:
                    return ""
                text = data[field].get(cue_index, "")
                return round(_cps(text, dur), 1) if text and dur else ""

            mode_errors = {m: _errors(m, mode_data.get(m, {}).get(sample_id)) for m in RAW_MODES}
            mode_errors["pipeline_regression"] = _errors(
                "pipeline_regression", mode_data.get("pipeline_regression", {}).get(sample_id)
            )

            ref_data = hybrid or grouped or sample_base
            ref_assess = _assessment_map(ref_data["quality_report"]).get(cue_index, {})
            source_flags = list(ref_assess.get("source_risk_flags") or [])
            errors_union: Set[str] = set()
            for errs in mode_errors.values():
                errors_union.update(errs)

            vi_final_hybrid = _pack("hybrid", hybrid, "vi_final")
            layer = _infer_failure_layer(
                list(errors_union),
                source_flags,
                _pack("grouped", grouped, "vi_raw") or _pack("hybrid", hybrid, "vi_raw"),
                _pack("hybrid", hybrid, "vi_editor"),
                _pack("hybrid", hybrid, "vi_flow"),
                vi_final_hybrid,
                float(_cps_for(hybrid, "vi_final") or 0),
                dur,
            )
            signatures = _risk_signature(
                en,
                vi_final_hybrid,
                list(errors_union),
                float(_cps_for(hybrid, "vi_final") or 0),
                dur,
                source_flags,
            )

            mode_finals = {
                "grouped": _pack("grouped", grouped, "vi_final"),
                "cue_keyed": _pack("cue_keyed", cue_keyed, "vi_final"),
                "hybrid_guarded": vi_final_hybrid,
                "pipeline_regression": _pack("pipeline", pipeline, "vi_final"),
            }
            best_raw, raw_reason = _best_mode_for_cue(
                {
                    "grouped": _pack("grouped", grouped, "vi_raw"),
                    "cue_keyed": _pack("cue_keyed", cue_keyed, "vi_raw"),
                    "hybrid_guarded": _pack("hybrid", hybrid, "vi_raw"),
                },
                {k: mode_errors.get(k, []) for k in RAW_MODES},
            )
            best_final, final_reason = _best_mode_for_cue(mode_finals, mode_errors)

            row = {
                "sample_id": sample_id,
                "cue_index": cue_index,
                "source_en": en,
                "grouped_raw": _pack("grouped", grouped, "vi_raw"),
                "cue_keyed_raw": _pack("cue_keyed", cue_keyed, "vi_raw"),
                "hybrid_raw": _pack("hybrid", hybrid, "vi_raw"),
                "grouped_final": mode_finals["grouped"],
                "cue_keyed_final": mode_finals["cue_keyed"],
                "hybrid_final": mode_finals["hybrid_guarded"],
                "pipeline_final": mode_finals["pipeline_regression"],
                "auto_errors_by_mode": mode_errors,
                "quality_score_by_mode": {
                    m: (mode_data.get(m, {}).get(sample_id) or {}).get("quality_score")
                    for m in list(RAW_MODES) + ["pipeline_regression"]
                },
                "cps_by_mode": {
                    "grouped_final": _cps_for(grouped, "vi_final"),
                    "cue_keyed_final": _cps_for(cue_keyed, "vi_final"),
                    "hybrid_final": _cps_for(hybrid, "vi_final"),
                    "pipeline_final": _cps_for(pipeline, "vi_final"),
                },
                "chars_per_line_by_mode": {
                    "hybrid_final": _chars_per_line(vi_final_hybrid),
                },
                "guard_flags_by_mode": {
                    "grouped": _guard_flags_for_cue("grouped", cue_index, grouped, None),
                    "hybrid_guarded": _guard_flags_for_cue(
                        "hybrid_guarded", cue_index, grouped, hybrid
                    ),
                },
                "repair_applied_by_mode": {
                    "hybrid_guarded": _repair_info("hybrid_guarded", cue_index, hybrid),
                },
                "stage_where_degraded": _stage_where_degraded(
                    _pack("hybrid", hybrid, "vi_raw"),
                    _pack("hybrid", hybrid, "vi_editor"),
                    _pack("hybrid", hybrid, "vi_flow"),
                    vi_final_hybrid,
                    list(errors_union),
                ),
                "likely_failure_layer": layer,
                "risk_signature": signatures,
                "recommended_next_action": _recommended_action(layer, signatures),
                "best_raw_mode": best_raw,
                "best_final_mode": best_final,
                "best_mode_reason": final_reason or raw_reason,
                "can_auto_route": bool(signatures) and not signatures == ["fragment_continuation"],
            }
            if sample_id == "no_rush_19" and errors_union:
                row["human_likely_verdict"] = _human_verdict_no_rush(
                    cue_index, en, vi_final_hybrid, list(errors_union)
                )
            rows.append(row)
    return rows


def _focused_no_rush(rows: List[dict]) -> dict:
    sample_rows = [r for r in rows if r["sample_id"] == "no_rush_19"]
    risky = [r for r in sample_rows if any(r["auto_errors_by_mode"].values())]
    error_counts = Counter()
    for r in risky:
        for errs in r["auto_errors_by_mode"].values():
            error_counts.update(errs)

    take_no_cues = [
        r
        for r in sample_rows
        if "take no" in (r["source_en"] or "").lower()
        or "take no money" in (r["source_en"] or "").lower()
    ]

    return {
        "quality_by_mode": {
            m: next(
                (r["quality_score_by_mode"].get(m) for r in sample_rows if r["cue_index"] == 1),
                None,
            )
            for m in list(RAW_MODES) + ["pipeline_regression"]
        },
        "top_error_families": error_counts.most_common(),
        "risky_cue_count": len(risky),
        "worst_cues": sorted(
            risky,
            key=lambda r: sum(len(v) for v in r["auto_errors_by_mode"].values()),
            reverse=True,
        )[:8],
        "take_no_money_analysis": take_no_cues,
        "conclusions": [
            "Primary failures are readability_cps + repeated_meaning + semantic_alignment, not pure neighbor bleed.",
            "Cue 17 VI borrows meaning from cue 18 ('build it') — raw_alignment / cue_shift class error.",
            "CPS flags on short cues may be timing/readability, not semantic.",
            "Editor/flow rarely improve no_rush; pipeline_regression stays ~46 on hybrid raw.",
        ],
    }


def _focused_raise_price(rows: List[dict], mode_data: dict) -> dict:
    sample_rows = [r for r in rows if r["sample_id"] == "raise_price_17"]
    focus_idxs = {5, 8, 9, 10}
    focus = [r for r in sample_rows if r["cue_index"] in focus_idxs]

    hybrid = mode_data.get("hybrid_guarded", {}).get("raise_price_17") or {}
    grouped = mode_data.get("grouped", {}).get("raise_price_17") or {}
    repairs_failed: List[dict] = []
    repairs_worse: List[dict] = []

    if grouped and hybrid:
        vi_g = [dict(e) for e in grouped["source_entries"]]
        for i, e in enumerate(grouped["source_entries"], start=1):
            vi_g[i - 1] = {"text": grouped["vi_raw"].get(i, "")}
        analysis = analyze_with_severity(grouped["source_entries"], vi_g)
        high_idxs = {f["cue_index"] for f in analysis.get("high_flags") or []}
        for idx in focus_idxs:
            if idx not in high_idxs:
                repairs_failed.append(
                    {
                        "cue_index": idx,
                        "reason": "severity below HIGH — no hybrid repair triggered",
                        "flags": [
                            f
                            for f in analysis.get("flags") or []
                            if f.get("cue_index") == idx
                        ],
                    }
                )

    for r in sample_rows:
        rep = r.get("repair_applied_by_mode", {}).get("hybrid_guarded") or {}
        if rep and rep.get("before") and rep.get("after"):
            if rep["before"] != rep["after"]:
                g_final = r.get("grouped_final", "")
                h_final = r.get("hybrid_final", "")
                if g_final and h_final and r["cue_index"] in focus_idxs:
                    repairs_worse.append(
                        {
                            "cue_index": r["cue_index"],
                            "before": rep.get("before"),
                            "after": rep.get("after"),
                            "grouped_final": g_final,
                            "hybrid_final": h_final,
                        }
                    )

    return {
        "e2e_scores": {
            m: (mode_data.get(m, {}).get("raise_price_17") or {}).get("quality_score")
            for m in RAW_MODES
        },
        "focus_cues": focus,
        "hybrid_repair_not_triggered": repairs_failed,
        "hybrid_repair_made_worse": repairs_worse,
        "conclusions": [
            "cue_keyed e2e 73 vs hybrid 51: grouped baseline likely weak on alignment cues 5/8-10.",
            "Hybrid guard may miss MEDIUM alignment on fragmented bribery/title-insurance span.",
            "Pipeline on hybrid raw scores 87 — downstream can recover some grouped raw issues.",
        ],
    }


def _qa_false_positive_candidates(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        if r["likely_failure_layer"] == "qa_false_positive":
            out.append(
                {
                    "sample_id": r["sample_id"],
                    "cue_index": r["cue_index"],
                    "source_en": r["source_en"],
                    "hybrid_final": r["hybrid_final"],
                    "source_flags": r.get("guard_flags_by_mode"),
                }
            )
        if r.get("human_likely_verdict") == "qa_false_positive":
            out.append(
                {
                    "sample_id": r["sample_id"],
                    "cue_index": r["cue_index"],
                    "source_en": r["source_en"],
                    "verdict": "qa_false_positive",
                }
            )
    return out


def _mode_decision_matrix(rows: List[dict]) -> List[dict]:
    matrix: List[dict] = []
    for r in rows:
        if not any(r["auto_errors_by_mode"].values()) and not r["risk_signature"]:
            continue
        matrix.append(
            {
                "sample_id": r["sample_id"],
                "cue_index": r["cue_index"],
                "best_raw_mode": r["best_raw_mode"],
                "best_final_mode": r["best_final_mode"],
                "reason": r["best_mode_reason"],
                "risk_signature": r["risk_signature"],
                "can_auto_route": r["can_auto_route"],
                "recommended_next_action": r["recommended_next_action"],
            }
        )
    return matrix


def _write_csv(path: Path, rows: List[dict]) -> None:
    flat_fields = [
        "sample_id",
        "cue_index",
        "source_en",
        "grouped_raw",
        "cue_keyed_raw",
        "hybrid_raw",
        "grouped_final",
        "cue_keyed_final",
        "hybrid_final",
        "pipeline_final",
        "likely_failure_layer",
        "stage_where_degraded",
        "best_raw_mode",
        "best_final_mode",
        "recommended_next_action",
        "risk_signature",
        "human_likely_verdict",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["risk_signature"] = ";".join(r.get("risk_signature") or [])
            out["human_likely_verdict"] = r.get("human_likely_verdict", "")
            w.writerow(out)


def _write_md(path: Path, report: dict) -> None:
    lines = [
        "# Failure Attribution v1",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Mode scores (e2e)",
        "",
    ]
    for mode, scores in report.get("mode_e2e_scores", {}).items():
        lines.append(f"### {mode}")
        for sid, sc in scores.items():
            lines.append(f"- {sid}: **{sc}**")
        lines.append("")

    lines.extend(["## no_rush_19 focus", ""])
    nr = report.get("no_rush_focus") or {}
    for c in nr.get("conclusions") or []:
        lines.append(f"- {c}")
    lines.append("")

    lines.extend(["## raise_price_17 focus", ""])
    rp = report.get("raise_price_focus") or {}
    for c in rp.get("conclusions") or []:
        lines.append(f"- {c}")
    lines.append("")

    lines.extend(["## Recommended next phase", ""])
    for item in report.get("recommended_next_phase") or []:
        lines.append(f"- {item}")
    lines.append("")

    lines.extend(["## v1b status", "", report.get("v1b_recommendation", ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare translation modes for failure attribution")
    parser.add_argument("--run-missing", action="store_true", help="Run grouped/cue_keyed e2e benchmarks")
    parser.add_argument("--engine", default="openai")
    args = parser.parse_args()

    _ensure_hybrid_snapshot()

    for mode in ("grouped", "cue_keyed"):
        missing = any(not _mode_sample_dir(mode, sid).exists() for sid in SAMPLES)
        if args.run_missing and missing:
            _run_mode_benchmark(mode, engine=args.engine)

    all_modes = list(RAW_MODES) + ["pipeline_regression"]
    mode_data: Dict[str, Dict[str, dict]] = {m: {} for m in all_modes}
    for mode in all_modes:
        for sid in SAMPLES:
            loaded = _load_mode_sample(mode, sid)
            if loaded:
                mode_data[mode][sid] = loaded

    rows = _build_cue_rows(mode_data)
    no_rush = _focused_no_rush(rows)
    raise_price = _focused_raise_price(rows, mode_data)
    matrix = _mode_decision_matrix(rows)
    qa_fp = _qa_false_positive_candidates(rows)

    mode_e2e_scores = {
        m: {sid: (mode_data.get(m, {}).get(sid) or {}).get("quality_score") for sid in SAMPLES}
        for m in RAW_MODES
    }

    report = {
        "generated_at": _utc_now(),
        "modes_compared": all_modes,
        "artifact_roots": {m: str(_mode_sample_dir(m, "SAMPLE")) for m in all_modes},
        "mode_e2e_scores": mode_e2e_scores,
        "cue_rows": rows,
        "no_rush_focus": no_rush,
        "raise_price_focus": raise_price,
        "mode_decision_matrix": matrix,
        "qa_false_positive_candidates": qa_fp,
        "hybrid_repair_failed_to_trigger": raise_price.get("hybrid_repair_not_triggered"),
        "hybrid_repair_made_worse": raise_price.get("hybrid_repair_made_worse"),
        "recommended_next_phase": [
            "1. guard_severity_tuning — neighbor bleed + fragment span on raise_price 8-10",
            "2. qa_calibration — repeated_meaning on rhetorical 'build it' pairs in no_rush",
            "3. raw_prompt_polish — cue 17 'take no money' polarity",
            "4. timing_readability — short-cue CPS after timing pass",
        ],
        "v1b_recommendation": "PAUSE v1b tuning until guard flags fragmented multi-cue spans; continue hybrid for natural samples, not global default.",
    }

    json_path = OUT_ROOT / "failure_attribution_v1.json"
    md_path = OUT_ROOT / "failure_attribution_v1.md"
    csv_path = OUT_ROOT / "failure_attribution_v1.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_md(md_path, report)

    print(f"[compare_modes] Wrote {json_path}")
    print(f"[compare_modes] Wrote {md_path}")
    print(f"[compare_modes] Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
