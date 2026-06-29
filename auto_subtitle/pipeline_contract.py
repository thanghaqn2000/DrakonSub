"""Pipeline contract enforcement: cue-count, alignment, artifact integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .utils import parse_srt, write_srt_entries


class PipelineContractError(RuntimeError):
    """Raised when a hard pipeline contract violation cannot be recovered."""


def compute_cue_text_hash(entries: List[dict]) -> str:
    """Stable hash over cue text by index (timestamps ignored)."""
    parts = []
    for i, entry in enumerate(entries, start=1):
        text = (entry.get("text") or "").strip()
        parts.append(f"{i}:{text}")
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _non_empty_source_indexes(source_entries: List[dict]) -> List[int]:
    return [
        i + 1
        for i, e in enumerate(source_entries)
        if (e.get("text") or "").strip()
    ]


def validate_translation_output(
    source_entries: List[dict],
    vi_entries: List[dict],
) -> Dict[str, Any]:
    """Validate translation output against source cue contract."""
    errors: List[str] = []
    warnings: List[str] = []
    missing_or_empty: List[int] = []

    n_src = len(source_entries)
    n_vi = len(vi_entries)

    if n_src != n_vi:
        errors.append(f"cue_count_mismatch: source={n_src} vi={n_vi}")

    limit = min(n_src, n_vi)
    for i in range(limit):
        src_text = (source_entries[i].get("text") or "").strip()
        vi_text = (vi_entries[i].get("text") or "").strip()
        if src_text and not vi_text:
            missing_or_empty.append(i + 1)
            errors.append(f"missing_or_empty_cue_error at cue {i + 1}")

    if n_vi > n_src:
        errors.append(f"duplicate_or_extra_cues: vi has {n_vi - n_src} extra cue(s)")

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "missing_or_empty_cue_errors": missing_or_empty,
        "source_cue_count": n_src,
        "vi_cue_count": n_vi,
    }


def align_vi_entries_to_source(
    source_entries: List[dict],
    vi_entries: List[dict],
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Align VI cues to source indexes/timestamps.

  Uses timestamp match first, then text-order fallback for unmatched slots.
  Does not insert silent empty placeholders when source cue is non-empty.
    """
    meta: Dict[str, Any] = {
        "alignment_applied": False,
        "timestamp_matches": 0,
        "order_fallback_assignments": 0,
        "missing_or_empty_cue_errors": [],
        "recovery_failed": False,
        "detail": "",
    }

    if len(source_entries) == len(vi_entries):
        aligned = [
            {**source_entries[i], "text": vi_entries[i].get("text", "")}
            for i in range(len(source_entries))
        ]
        for i, src in enumerate(source_entries):
            vi = vi_entries[i]
            if (
                src.get("start_str") != vi.get("start_str")
                or src.get("end_str") != vi.get("end_str")
            ):
                meta["alignment_applied"] = True
                break
        return aligned, meta

    meta["alignment_applied"] = True
    vi_by_time = {(e["start_str"], e["end_str"]): e for e in vi_entries}
    used_vi_ids: set[int] = set()
    slots: List[Optional[dict]] = []

    for source in source_entries:
        key = (source["start_str"], source["end_str"])
        if key in vi_by_time:
            vi = vi_by_time[key]
            slots.append({**source, "text": vi.get("text", "")})
            used_vi_ids.add(id(vi))
            meta["timestamp_matches"] += 1
        else:
            slots.append(None)

    remaining = [e for e in vi_entries if id(e) not in used_vi_ids]
    rem_idx = 0
    for i, slot in enumerate(slots):
        if slot is not None:
            continue
        src_text = (source_entries[i].get("text") or "").strip()
        if rem_idx < len(remaining):
            slots[i] = {**source_entries[i], "text": remaining[rem_idx].get("text", "")}
            meta["order_fallback_assignments"] += 1
            rem_idx += 1
        elif src_text:
            meta["missing_or_empty_cue_errors"].append(i + 1)
            meta["recovery_failed"] = True
            slots[i] = {**source_entries[i], "text": ""}
        else:
            slots[i] = {**source_entries[i], "text": ""}

    if rem_idx < len(remaining):
        meta["recovery_failed"] = True
        meta["detail"] = (
            f"{len(remaining) - rem_idx} translated cue(s) could not be assigned"
        )

    aligned = [s for s in slots if s is not None]
    if len(aligned) != len(source_entries):
        meta["recovery_failed"] = True

    return aligned, meta


def enforce_translation_contract(
    source_entries: List[dict],
    vi_entries: List[dict],
    *,
    retry_translate: Optional[Callable[[], List[dict]]] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Validate translation output; align or retry once before hard fail.

    Returns (working_entries, contract_meta).
    Raises PipelineContractError when contract cannot be satisfied.
    """
    report: Dict[str, Any] = {
        "alignment_applied": False,
        "translation_retry_applied": False,
        "missing_or_empty_cue_errors": [],
        "errors": [],
        "warnings": [],
    }

    def _attempt(entries: List[dict]) -> Tuple[List[dict], Dict[str, Any]]:
        validation = validate_translation_output(source_entries, entries)
        if validation["is_valid"] and len(entries) == len(source_entries):
            aligned = [
                {**source_entries[i], "text": entries[i].get("text", "")}
                for i in range(len(source_entries))
            ]
            return aligned, {"alignment_applied": False, "validation": validation}

        aligned, align_meta = align_vi_entries_to_source(source_entries, entries)
        validation2 = validate_translation_output(source_entries, aligned)
        align_meta["validation"] = validation2
        return aligned, align_meta

    working, align_meta = _attempt(vi_entries)
    report["alignment_applied"] = align_meta.get("alignment_applied", False)
    report["missing_or_empty_cue_errors"] = align_meta.get(
        "missing_or_empty_cue_errors", []
    )

    validation = align_meta.get("validation") or validate_translation_output(
        source_entries, working
    )
    if validation["is_valid"] and not align_meta.get("recovery_failed"):
        report["errors"] = validation.get("errors", [])
        return working, report

    if retry_translate is not None:
        report["translation_retry_applied"] = True
        retried = retry_translate()
        working, align_meta = _attempt(retried)
        report["alignment_applied"] = (
            report["alignment_applied"] or align_meta.get("alignment_applied", False)
        )
        report["missing_or_empty_cue_errors"] = align_meta.get(
            "missing_or_empty_cue_errors", []
        )
        validation = align_meta.get("validation") or validate_translation_output(
            source_entries, working
        )

    if not validation["is_valid"] or align_meta.get("recovery_failed"):
        report["errors"] = validation.get("errors", [])
        if align_meta.get("detail"):
            report["errors"].append(align_meta["detail"])
        raise PipelineContractError(
            "Translation contract failed after alignment/retry: "
            + "; ".join(report["errors"])
        )

    report["errors"] = validation.get("errors", [])
    return working, report


def verify_post_final_repair_text_lock(
    before_entries: List[dict],
    after_entries: List[dict],
) -> Dict[str, Any]:
    """Detect cue text changes after final semantic repair."""
    violations: List[dict] = []
    n = min(len(before_entries), len(after_entries))
    for i in range(n):
        before_text = (before_entries[i].get("text") or "").strip()
        after_text = (after_entries[i].get("text") or "").strip()
        if before_text != after_text:
            violations.append(
                {
                    "cue_index": i + 1,
                    "before": before_text,
                    "after": after_text,
                }
            )

    if len(before_entries) != len(after_entries):
        violations.append(
            {
                "cue_index": None,
                "before": f"count={len(before_entries)}",
                "after": f"count={len(after_entries)}",
                "issue": "cue_count_changed",
            }
        )

    changed = bool(violations)
    return {
        "post_final_repair_text_changed": changed,
        "post_final_repair_text_lock_status": "fail" if changed else "pass",
        "violations": violations,
        "before_hash": compute_cue_text_hash(before_entries),
        "after_hash": compute_cue_text_hash(after_entries),
    }


def count_stage_artifacts(artifact_dir: Path) -> Dict[str, Any]:
    """Read cue counts from known stage artifact files."""
    stage_files = {
        "source": "source.srt",
        "source_corrected": "source_corrected.srt",
        "vi_raw": "vi_raw.srt",
        "vi_raw_aligned": "vi_raw_aligned.srt",
        "vi_after_preliminary_repair": "vi_after_preliminary_repair.srt",
        "vi_after_editor": "vi_after_editor.srt",
        "vi_after_compression": "vi_after_compression.srt",
        "vi_after_flow": "vi_after_flow.srt",
        "vi_after_readability": "vi_after_readability.srt",
        "vi_after_final_repair": "vi_after_final_repair.srt",
        "vi_after_timing": "vi_after_timing.srt",
        "final_vi": "final_vi.srt",
    }
    counts: Dict[str, Any] = {}
    for stage, filename in stage_files.items():
        path = artifact_dir / filename
        if not path.exists():
            counts[stage] = "not_run"
            continue
        entries = parse_srt(path.read_text(encoding="utf-8"))
        counts[stage] = len(entries)
    return counts


def detect_cue_count_changed_by_stage(
    stage_counts: Dict[str, Any],
    *,
    baseline_stage: str = "source_corrected",
) -> List[dict]:
    """List stages where cue count diverges from baseline."""
    baseline = stage_counts.get(baseline_stage)
    if not isinstance(baseline, int):
        return []
    changes: List[dict] = []
    for stage, count in stage_counts.items():
        if stage == baseline_stage or not isinstance(count, int):
            continue
        if count != baseline:
            changes.append(
                {"stage": stage, "count": count, "expected": baseline}
            )
    return changes


def build_pipeline_contract_report(
    *,
    source_cue_count: int,
    vi_raw_cue_count: int,
    vi_raw_aligned_cue_count: Optional[int],
    alignment_applied: bool,
    translation_retry_applied: bool,
    stage_cue_counts: Dict[str, Any],
    post_final_repair_text_changed: bool,
    post_final_repair_text_lock_status: str,
    errors: List[str],
    warnings: List[str],
    missing_or_empty_cue_errors: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Assemble the pipeline contract report dict."""
    cue_changes = detect_cue_count_changed_by_stage(stage_cue_counts)
    contract_ok = (
        not errors
        and post_final_repair_text_lock_status == "pass"
        and not cue_changes
        and not missing_or_empty_cue_errors
    )
    return {
        "source_cue_count": source_cue_count,
        "vi_raw_cue_count": vi_raw_cue_count,
        "vi_raw_aligned_cue_count": vi_raw_aligned_cue_count,
        "alignment_applied": alignment_applied,
        "translation_retry_applied": translation_retry_applied,
        "missing_or_empty_cue_errors": missing_or_empty_cue_errors or [],
        "stage_cue_counts": stage_cue_counts,
        "cue_count_changed_by_stage": cue_changes,
        "post_final_repair_text_changed": post_final_repair_text_changed,
        "post_final_repair_text_lock_status": post_final_repair_text_lock_status,
        "pipeline_contract_status": "pass" if contract_ok else "fail",
        "errors": errors,
        "warnings": warnings,
    }


def save_pipeline_contract_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text_lock_violation(path: Path, lock_report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock_report, ensure_ascii=False, indent=2), encoding="utf-8")


def save_srt_entries(path: Path, entries: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        write_srt_entries(entries, file=f)
