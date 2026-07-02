#!/usr/bin/env python3
"""Build SA review slice v1 from exported translation quality review artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = ROOT / "artifacts" / "translation_quality_review"
OUT_DIR = REVIEW_ROOT
SLICE_ID = "sa_review_slice_v1"

SAMPLE_QUOTAS = {
    "raise_price_17": {"total": 10, "risky": 6, "medium": 2, "good": 2},
    "outsider_36": {"total": 10, "risky": 6, "raw_final_diff": 2, "good": 2},
    "no_rush_19": {"total": 6, "risky": 3, "medium": 2, "good": 1},
    "buffett_bitcoin_29": {"total": 6, "risky": 2, "medium": 2, "good": 2},
}

CSV_FIELDS = [
    "slice_id",
    "sample_id",
    "mode",
    "cue_index",
    "start_time",
    "end_time",
    "selection_reason",
    "source_en",
    "e2e_vi_raw",
    "e2e_vi_after_editor",
    "e2e_vi_after_flow",
    "e2e_vi_final",
    "pipeline_vi_raw",
    "pipeline_vi_after_editor",
    "pipeline_vi_after_flow",
    "pipeline_vi_final",
    "auto_error_families",
    "recommended_fix_layer",
    "SA_semantic_score",
    "SA_naturalness_score",
    "SA_readability_score",
    "SA_context_score",
    "SA_timing_score",
    "SA_overall_score",
    "SA_comment",
    "SA_correct_fix_layer",
]

SA_EMPTY = {f: "" for f in CSV_FIELDS if f.startswith("SA_")}


def _load_review(mode: str, sample_id: str) -> Dict[int, dict]:
    path = REVIEW_ROOT / mode / sample_id / "review.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(c["cue_index"]): c for c in data.get("cues") or []}


def _risk_score(cue: dict) -> int:
    errors = (cue.get("auto_error_families") or "").strip()
    score = len([e for e in errors.split(";") if e]) * 3
    if cue.get("semantic_risk"):
        score += 5
    if cue.get("readability_risk"):
        score += 2
    return score


def _raw_final_diff(cue: dict) -> int:
    raw = (cue.get("vi_raw") or "").strip()
    final = (cue.get("vi_final") or "").strip()
    if not raw or not final:
        return 0
    if raw == final:
        return 0
    return abs(len(final) - len(raw)) + (10 if raw != final else 0)


def _is_good(cue: dict) -> bool:
    return _risk_score(cue) == 0 and not (cue.get("auto_error_families") or "").strip()


def _pick(
    candidates: List[dict],
    *,
    count: int,
    reason: str,
    used: Set[int],
) -> List[dict]:
    picked: List[dict] = []
    for cue in candidates:
        idx = int(cue["cue_index"])
        if idx in used:
            continue
        picked.append({**cue, "selection_reason": reason})
        used.add(idx)
        if len(picked) >= count:
            break
    return picked


def _select_sample(sample_id: str, e2e: Dict[int, dict], quota: dict) -> List[dict]:
    cues = list(e2e.values())
    used: Set[int] = set()
    selected: List[dict] = []

    risky_sorted = sorted(cues, key=lambda c: (-_risk_score(c), c["cue_index"]))
    risky_n = quota.get("risky", 0)
    selected.extend(_pick(risky_sorted, count=risky_n, reason="risky_e2e", used=used))

    if quota.get("raw_final_diff"):
        diff_sorted = sorted(
            cues,
            key=lambda c: (-_raw_final_diff(c), -_risk_score(c), c["cue_index"]),
        )
        selected.extend(
            _pick(
                diff_sorted,
                count=quota["raw_final_diff"],
                reason="raw_final_diff_e2e",
                used=used,
            )
        )

    medium_n = quota.get("medium", 0)
    if medium_n:
        medium_sorted = sorted(
            [c for c in cues if _risk_score(c) > 0 and int(c["cue_index"]) not in used],
            key=lambda c: (_risk_score(c), c["cue_index"]),
        )
        selected.extend(
            _pick(medium_sorted, count=medium_n, reason="medium_e2e", used=used)
        )

    good_n = quota.get("good", 0)
    if good_n:
        good_sorted = sorted(
            [c for c in cues if _is_good(c)],
            key=lambda c: c["cue_index"],
        )
        if len(good_sorted) < good_n:
            good_sorted = sorted(cues, key=lambda c: (_risk_score(c), c["cue_index"]))
        selected.extend(_pick(good_sorted, count=good_n, reason="good_e2e", used=used))

    total = quota["total"]
    if len(selected) < total:
        remaining = sorted(cues, key=lambda c: c["cue_index"])
        selected.extend(
            _pick(remaining, count=total - len(selected), reason="fill_quota", used=used)
        )

    return selected[:total]


def _merge_row(e2e: dict, pipeline: Optional[dict], selection_reason: str) -> dict:
    p = pipeline or {}
    return {
        "slice_id": SLICE_ID,
        "sample_id": e2e["sample_id"],
        "mode": "end_to_end",
        "cue_index": e2e["cue_index"],
        "start_time": e2e.get("start_time", ""),
        "end_time": e2e.get("end_time", ""),
        "selection_reason": selection_reason,
        "source_en": e2e.get("source_en", ""),
        "e2e_vi_raw": e2e.get("vi_raw", ""),
        "e2e_vi_after_editor": e2e.get("vi_after_editor", ""),
        "e2e_vi_after_flow": e2e.get("vi_after_flow", ""),
        "e2e_vi_final": e2e.get("vi_final", ""),
        "pipeline_vi_raw": p.get("vi_raw", ""),
        "pipeline_vi_after_editor": p.get("vi_after_editor", ""),
        "pipeline_vi_after_flow": p.get("vi_after_flow", ""),
        "pipeline_vi_final": p.get("vi_final", ""),
        "auto_error_families": e2e.get("auto_error_families", ""),
        "recommended_fix_layer": e2e.get("recommended_fix_layer", ""),
        **SA_EMPTY,
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: List[dict]) -> None:
    lines = [
        f"# SA Review Slice — {SLICE_ID}",
        "",
        f"Total cues: **{len(rows)}**",
        "",
        "Score each cue 1–10 per rubric. Fields left blank for SA.",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Slice cue {i} — {row['sample_id']} cue {row['cue_index']}",
                "",
                f"- mode: **end_to_end** (primary)",
                f"- time: {row['start_time']} --> {row['end_time']}",
                f"- selection: **{row['selection_reason']}**",
                f"- auto errors: {row['auto_error_families'] or '—'}",
                f"- suggested fix layer: {row['recommended_fix_layer'] or '—'}",
                "",
                "**EN source:**",
                f"> {row['source_en']}",
                "",
                "**VI raw (e2e):**",
                f"> {row['e2e_vi_raw'] or '—'}",
                "",
                "**VI after editor (e2e):**",
                f"> {row['e2e_vi_after_editor'] or '—'}",
                "",
                "**VI after flow (e2e):**",
                f"> {row['e2e_vi_after_flow'] or '—'}",
                "",
                "**VI final (e2e):**",
                f"> {row['e2e_vi_final'] or '—'}",
                "",
                "**Pipeline final (same cue, cached raw):**",
                f"> {row['pipeline_vi_final'] or '—'}",
                "",
                "**Pipeline raw (same cue):**",
                f"> {row['pipeline_vi_raw'] or '—'}",
                "",
                "**SA scoring:**",
                "- semantic:",
                "- naturalness:",
                "- readability:",
                "- context:",
                "- timing:",
                "- overall:",
                "- comment:",
                "- correct_fix_layer:",
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_slice() -> dict:
    rows: List[dict] = []
    per_sample: Dict[str, int] = {}
    missing_pipeline: List[str] = []

    for sample_id, quota in SAMPLE_QUOTAS.items():
        e2e = _load_review("end_to_end", sample_id)
        pipeline = _load_review("pipeline_regression", sample_id)
        if not e2e:
            raise FileNotFoundError(f"Missing end_to_end review for {sample_id}")
        selected = _select_sample(sample_id, e2e, quota)
        for item in selected:
            idx = int(item["cue_index"])
            p = pipeline.get(idx)
            if not p:
                missing_pipeline.append(f"{sample_id}#{idx}")
            rows.append(_merge_row(item, p, item["selection_reason"]))
        per_sample[sample_id] = len(selected)

    rows.sort(key=lambda r: (r["sample_id"], int(r["cue_index"])))

    meta = {
        "slice_id": SLICE_ID,
        "total_cues": len(rows),
        "per_sample": per_sample,
        "selection_strategy": SAMPLE_QUOTAS,
        "missing_pipeline_comparison": missing_pipeline,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{SLICE_ID}.json"
    csv_path = OUT_DIR / f"{SLICE_ID}.csv"
    md_path = OUT_DIR / f"{SLICE_ID}.md"

    json_path.write_text(
        json.dumps({"meta": meta, "cues": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, rows)
    _write_md(md_path, rows)

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "md": str(md_path),
        **meta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SA review slice v1")
    parser.parse_args()
    result = build_slice()
    print(f"[ReviewSlice] cues={result['total_cues']} per_sample={result['per_sample']}")
    print(f"[ReviewSlice] wrote {result['md']}")
    if result["missing_pipeline_comparison"]:
        print(
            f"[ReviewSlice] missing pipeline: {len(result['missing_pipeline_comparison'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
