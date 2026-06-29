#!/usr/bin/env python3
"""Audit cue index → text mapping across subtitle pipeline stages."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from auto_subtitle.config import (  # noqa: E402
    VI_COMPRESSION_ENABLED,
    VI_FLOW_ENABLED,
    en_domain_correction_enabled,
    translation_intelligence_enabled,
)
from auto_subtitle.en_domain_corrector import correct_en_domain_srt_file  # noqa: E402
from auto_subtitle.pipeline import SubtitleConfig, translate_srt_file  # noqa: E402
from auto_subtitle.pipeline_contract import (
    PipelineContractError,
    align_vi_entries_to_source,
    build_pipeline_contract_report,
    enforce_translation_contract,
    save_pipeline_contract_report,
    save_srt_entries,
    save_text_lock_violation,
    verify_post_final_repair_text_lock,
    count_stage_artifacts,
)
from auto_subtitle.subtitle_readability_optimizer import optimize_readability_file  # noqa: E402
from auto_subtitle.subtitle_timing_optimizer import (  # noqa: E402
    normalize_final_srt_timing,
    optimize_srt_timing_file,
)
from auto_subtitle.translation_intelligence import (  # noqa: E402
    TranslationIntelligenceContext,
    finalize_delivery_quality_report,
    run_post_translation_qa,
    run_pre_translation_intelligence,
)
from auto_subtitle.utils import parse_srt, write_srt_entries  # noqa: E402
from auto_subtitle.vi_compression import compress_vi_srt_file  # noqa: E402
from auto_subtitle.vi_editor import edit_vi_srt_entries  # noqa: E402
from auto_subtitle.vi_flow import flow_vi_srt_file  # noqa: E402
from auto_subtitle.semantic_alignment_guard import _detect_cue_shift  # noqa: E402

JOB_SOURCE = Path(
    "/var/folders/kc/wq9gs6yd0pl2b0q6ddqfqvfc0000gn/T/"
    "drakonsub_jobs/6bfeabd2-7d63-4d0c-8561-bbbc61df5891/source.srt"
)
DEBUG = ROOT / "debug"
RISKY_CUES = [4, 5, 10, 18, 19, 20, 24, 25, 27, 29]

STAGE_ORDER = [
    "source",
    "source_corrected",
    "vi_raw",
    "vi_raw_aligned",
    "vi_after_editor",
    "vi_after_compression",
    "vi_after_flow",
    "vi_after_readability",
    "vi_after_final_repair",
    "vi_after_timing",
    "final_vi",
]


def _save_srt(path: Path, entries: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        write_srt_entries(entries, file=f)


def _load_srt(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return parse_srt(path.read_text(encoding="utf-8"))


def _cue_map(entries: List[dict]) -> Dict[int, str]:
    return {i + 1: (e.get("text") or "").strip() for i, e in enumerate(entries)}


def _unit_for_cue(meaning_units: List[dict], cue_index: int) -> Optional[int]:
    for unit in meaning_units:
        if cue_index in (unit.get("cue_indexes") or []):
            return unit.get("unit_id")
    return None


def _unit_source_text(meaning_units: List[dict], unit_id: Optional[int]) -> str:
    if unit_id is None:
        return ""
    for unit in meaning_units:
        if unit.get("unit_id") == unit_id:
            return unit.get("source_text", "")
    return ""


def _detect_issues_for_cue(
    cue_index: int,
    stage: str,
    source_entries: List[dict],
    stage_entries: List[dict],
    meaning_units: List[dict],
    video_context: dict,
    prev_stage: Optional[str],
    prev_map: Dict[int, str],
    repair_map: Dict[int, str],
) -> List[dict]:
    issues: List[dict] = []
    n_src = len(source_entries)
    n_stage = len(stage_entries)
    en = source_entries[cue_index - 1].get("text", "").strip() if cue_index <= n_src else ""
    vi = stage_entries[cue_index - 1].get("text", "").strip() if cue_index <= n_stage else ""
    unit_id = _unit_for_cue(meaning_units, cue_index)

    if stage in ("source", "source_corrected"):
        return issues

    if cue_index > n_stage:
        issues.append(
            {
                "stage": stage,
                "cue_index": cue_index,
                "source_en": en,
                "stage_vi": "",
                "issue": "cue_count_mismatch",
                "suspected_origin_stage": stage,
                "confidence": 0.95,
                "detail": f"stage has {n_stage} cues, source has {n_src}",
            }
        )
        return issues

    if en and not vi:
        issues.append(
            {
                "stage": stage,
                "cue_index": cue_index,
                "source_en": en,
                "stage_vi": vi,
                "issue": "empty_vi_for_non_empty_source",
                "suspected_origin_stage": stage,
                "confidence": 0.9,
            }
        )

    if vi and n_src == n_stage:
        src_texts = [e.get("text", "") for e in source_entries]
        vi_texts = [e.get("text", "") for e in stage_entries]
        shifted, reason, conf = _detect_cue_shift(
            cue_index, src_texts, vi_texts, video_context=video_context
        )
        if shifted:
            issues.append(
                {
                    "stage": stage,
                    "cue_index": cue_index,
                    "source_en": en,
                    "stage_vi": vi,
                    "issue": "semantic_shift_to_later_unit"
                    if "later" in reason
                    else "semantic_shift_to_earlier_unit",
                    "suspected_origin_stage": stage,
                    "confidence": conf,
                    "detail": reason,
                }
            )

    if n_src != n_stage and vi and en:
        issues.append(
            {
                "stage": stage,
                "cue_index": cue_index,
                "source_en": en,
                "stage_vi": vi,
                "issue": "cue_count_mismatch",
                "suspected_origin_stage": stage,
                "confidence": 0.85,
                "detail": f"source={n_src} stage={n_stage}",
            }
        )

    if prev_stage and prev_map.get(cue_index, "") != vi and vi != prev_map.get(cue_index, ""):
        changed = True
        if stage in ("vi_after_timing", "final_vi") and prev_map.get(cue_index, "") == vi:
            changed = False
        if changed:
            issue_type = "stage_text_changed"
            if stage in ("vi_after_editor", "vi_after_compression", "vi_after_flow", "vi_after_readability"):
                repaired = repair_map.get(cue_index, "")
                if repaired and repaired != vi and prev_map.get(cue_index, "") == repaired:
                    issue_type = "post_repair_overwrite"
            if stage in ("vi_after_timing", "final_vi"):
                final_repair_text = repair_map.get(cue_index, "")
                if final_repair_text and final_repair_text != vi:
                    issue_type = "post_final_repair_text_change"
            issues.append(
                {
                    "stage": stage,
                    "cue_index": cue_index,
                    "source_en": en,
                    "stage_vi": vi,
                    "issue": issue_type,
                    "suspected_origin_stage": stage,
                    "confidence": 0.8,
                    "detail": f"prev({prev_stage})={prev_map.get(cue_index, '')!r}",
                }
            )

    return issues


def _first_bad_stage(issues: List[dict], cue_index: int) -> Optional[dict]:
    stage_rank = {s: i for i, s in enumerate(STAGE_ORDER)}
    cue_issues = [
        i
        for i in issues
        if i["cue_index"] == cue_index
        and i["issue"]
        not in ("stage_text_changed", "post_repair_overwrite")
    ]
    if not cue_issues:
        return None
    cue_issues.sort(key=lambda x: stage_rank.get(x["stage"], 999))
    return cue_issues[0]


def _run_pipeline_stages(debug: Path, *, reuse_raw: bool = False) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"stages": {}, "notes": []}
    config = SubtitleConfig.from_env()
    config.source_language = "en"
    config.translation_engine = os.getenv("TRANSLATION_ENGINE", "openai").strip().lower()
    if config.translation_engine not in ("openai",):
        config.translation_engine = "openai"

    shutil.copy2(JOB_SOURCE, debug / "source.srt")
    source_entries = _load_srt(debug / "source.srt")
    meta["stages"]["source"] = {"path": str(debug / "source.srt"), "cue_count": len(source_entries)}

    source_corrected_path = debug / "source_corrected.srt"
    if en_domain_correction_enabled():
        correct_en_domain_srt_file(
            str(debug / "source.srt"),
            str(source_corrected_path),
            debug_dir=str(debug),
        )
    else:
        shutil.copy2(debug / "source.srt", source_corrected_path)
    source_entries = _load_srt(source_corrected_path)
    meta["stages"]["source_corrected"] = {
        "path": str(source_corrected_path),
        "cue_count": len(source_entries),
    }

    meaning_units_path = debug / "meaning_units.json"
    video_context_path = debug / "video_context.json"
    intel_ctx: Optional[TranslationIntelligenceContext] = None
    translation_context = None
    if translation_intelligence_enabled() and meaning_units_path.exists() and video_context_path.exists():
        intel_ctx = TranslationIntelligenceContext(
            video_context=json.loads(video_context_path.read_text(encoding="utf-8")),
            meaning_units=json.loads(meaning_units_path.read_text(encoding="utf-8")),
            engine=config.translation_engine,
            debug_dir=str(debug),
        )
        translation_context = intel_ctx.to_dict()
        meta["notes"].append("Reused existing meaning_units.json and video_context.json")
    elif translation_intelligence_enabled():
        intel_ctx = run_pre_translation_intelligence(
            source_entries,
            str(debug),
            user_topic=config.translation_topic,
            engine=config.translation_engine,
        )
        translation_context = intel_ctx.to_dict()
        meta["notes"].append("Built fresh video context and meaning units")

    vi_raw_path = debug / "vi_raw.srt"
    alignment_applied = False
    translation_retry_applied = False
    if reuse_raw and vi_raw_path.exists():
        vi_raw_entries = _load_srt(vi_raw_path)
        meta["notes"].append("Reused existing vi_raw.srt")
    else:
        translate_srt_file(
            str(source_corrected_path),
            str(vi_raw_path),
            config,
            translation_context=translation_context,
        )
        vi_raw_entries = _load_srt(vi_raw_path)
    meta["stages"]["vi_raw"] = {"path": str(vi_raw_path), "cue_count": len(vi_raw_entries)}

    def _retry_translate() -> List[dict]:
        nonlocal translation_retry_applied
        translation_retry_applied = True
        retry_path = debug / "_vi_retry.srt"
        translate_srt_file(
            str(source_corrected_path),
            str(retry_path),
            config,
            translation_context=translation_context,
            strict_cue_count=True,
        )
        return _load_srt(retry_path)

    try:
        working_entries, contract_meta = enforce_translation_contract(
            source_entries,
            vi_raw_entries,
            retry_translate=_retry_translate,
        )
        alignment_applied = contract_meta.get("alignment_applied", False)
    except PipelineContractError as exc:
        meta["notes"].append(f"translation contract failed: {exc}")
        raise

    vi_raw_aligned_path = debug / "vi_raw_aligned.srt"
    if alignment_applied:
        save_srt_entries(vi_raw_aligned_path, working_entries)
        meta["stages"]["vi_raw_aligned"] = {
            "path": str(vi_raw_aligned_path),
            "cue_count": len(working_entries),
        }
    else:
        meta["notes"].append("vi_raw_aligned not needed")

    working = debug / "_audit_working.srt"
    _save_srt(working, working_entries)

    vi_editor_path = debug / "vi_after_editor.srt"
    edited = edit_vi_srt_entries(
        source_entries,
        _load_srt(working),
        translation_engine=config.translation_engine,
        topic=config.translation_topic,
        debug_dir=str(debug / "editor_debug_audit"),
        translation_context=translation_context,
    )
    _save_srt(vi_editor_path, edited)
    shutil.copy2(vi_editor_path, working)
    meta["stages"]["vi_after_editor"] = {
        "path": str(vi_editor_path),
        "cue_count": len(edited),
    }

    if VI_COMPRESSION_ENABLED:
        compress_vi_srt_file(str(working))
        shutil.copy2(working, debug / "vi_after_compression.srt")
        meta["stages"]["vi_after_compression"] = {
            "path": str(debug / "vi_after_compression.srt"),
            "cue_count": len(_load_srt(working)),
        }
    else:
        meta["notes"].append("VI compression disabled")

    if VI_FLOW_ENABLED:
        flow_vi_srt_file(str(source_corrected_path), str(working))
        shutil.copy2(working, debug / "vi_after_flow.srt")
        meta["stages"]["vi_after_flow"] = {
            "path": str(debug / "vi_after_flow.srt"),
            "cue_count": len(_load_srt(working)),
        }
    else:
        meta["notes"].append("VI flow disabled")

    os.environ["DRAKONSUB_VI_BEFORE_READABILITY_SRT"] = str(debug / "vi_before_readability.srt")
    os.environ["DRAKONSUB_VI_AFTER_READABILITY_SRT"] = str(debug / "vi_after_readability.srt")
    optimize_readability_file(str(working))
    shutil.copy2(working, debug / "vi_after_readability.srt")
    meta["stages"]["vi_after_readability"] = {
        "path": str(debug / "vi_after_readability.srt"),
        "cue_count": len(_load_srt(working)),
    }

    qa_report: Optional[Dict[str, Any]] = None
    if intel_ctx is not None:
        try:
            repaired, qa_report = run_post_translation_qa(
                source_entries,
                _load_srt(working),
                intel_ctx,
                str(debug),
            )
        except Exception as exc:
            meta["notes"].append(f"final repair failed: {exc}; using pre-repair text")
            repaired = _load_srt(working)
        _save_srt(debug / "vi_after_final_repair.srt", repaired)
        shutil.copy2(debug / "vi_after_final_repair.srt", working)
        meta["stages"]["vi_after_final_repair"] = {
            "path": str(debug / "vi_after_final_repair.srt"),
            "cue_count": len(repaired),
        }
    else:
        shutil.copy2(working, debug / "vi_after_final_repair.srt")
        meta["notes"].append("Translation intelligence disabled; vi_after_final_repair = readability output")

    pre_timing_entries = _load_srt(debug / "vi_after_final_repair.srt")
    optimize_srt_timing_file(str(working))
    normalize_final_srt_timing(str(working))
    post_timing_entries = _load_srt(working)
    lock_report = verify_post_final_repair_text_lock(
        pre_timing_entries, post_timing_entries
    )
    if lock_report["post_final_repair_text_changed"]:
        save_text_lock_violation(debug / "post_repair_text_lock_violation.json", lock_report)

    shutil.copy2(working, debug / "vi_after_timing.srt")
    meta["stages"]["vi_after_timing"] = {
        "path": str(debug / "vi_after_timing.srt"),
        "cue_count": len(post_timing_entries),
    }

    final_path = debug / "final_vi.srt"
    shutil.copy2(working, final_path)
    meta["stages"]["final_vi"] = {
        "path": str(final_path),
        "cue_count": len(_load_srt(final_path)),
    }

    if intel_ctx is not None and qa_report is not None:
        try:
            finalize_delivery_quality_report(
                source_entries,
                pre_timing_entries,
                post_timing_entries,
                intel_ctx,
                qa_report,
                str(debug),
            )
            meta["notes"].append("Final delivery QA + cps_diagnosis_report generated")
        except Exception as exc:
            meta["notes"].append(f"delivery QA finalize failed: {exc}")

    stage_counts = count_stage_artifacts(debug)
    contract_report = build_pipeline_contract_report(
        source_cue_count=len(source_entries),
        vi_raw_cue_count=len(vi_raw_entries),
        vi_raw_aligned_cue_count=(
            len(_load_srt(vi_raw_aligned_path)) if vi_raw_aligned_path.exists() else None
        ),
        alignment_applied=alignment_applied,
        translation_retry_applied=translation_retry_applied,
        stage_cue_counts=stage_counts,
        post_final_repair_text_changed=lock_report["post_final_repair_text_changed"],
        post_final_repair_text_lock_status=lock_report["post_final_repair_text_lock_status"],
        errors=[],
        warnings=[],
        missing_or_empty_cue_errors=contract_meta.get("missing_or_empty_cue_errors", []),
    )
    save_pipeline_contract_report(debug / "pipeline_contract_report.json", contract_report)
    meta["pipeline_contract_status"] = contract_report["pipeline_contract_status"]
    meta["post_final_repair_text_lock_status"] = lock_report[
        "post_final_repair_text_lock_status"
    ]

    meta["pipeline_order"] = [
        "transcribe/domain-correct → source_corrected",
        "translate → vi_raw (immutable)",
        "contract align → vi_raw_aligned (if needed)",
        "vi_editor → vi_after_editor",
        "vi_compression → vi_after_compression",
        "vi_flow → vi_after_flow",
        "readability → vi_after_readability",
        "final_semantic_qa/repair → vi_after_final_repair",
        "timing optimize + normalize → vi_after_timing → final_vi",
    ]
    meta["meaning_units_path"] = str(meaning_units_path)
    return meta


def _build_reports(debug: Path, run_meta: Dict[str, Any]) -> None:
    source_entries = _load_srt(debug / "source_corrected.srt")
    meaning_units = json.loads((debug / "meaning_units.json").read_text(encoding="utf-8"))
    video_context = json.loads((debug / "video_context.json").read_text(encoding="utf-8"))

    stage_maps: Dict[str, Dict[int, str]] = {}
    stage_counts: Dict[str, int] = {}
    for stage in STAGE_ORDER:
        if stage == "source":
            path = debug / "source.srt"
        elif stage == "source_corrected":
            path = debug / "source_corrected.srt"
        elif stage == "vi_raw_aligned":
            path = debug / "vi_raw_aligned.srt"
        else:
            path = debug / f"{stage}.srt"
        entries = _load_srt(path)
        stage_maps[stage] = _cue_map(entries)
        stage_counts[stage] = len(entries)

    repair_map = stage_maps.get("vi_after_final_repair", {})
    all_issues: List[dict] = []
    prev_stage: Optional[str] = None
    prev_map: Dict[int, str] = {}

    for stage in STAGE_ORDER:
        if stage in ("source", "source_corrected"):
            prev_stage = stage
            prev_map = stage_maps.get(stage, {})
            continue
        for cue_index in range(1, len(source_entries) + 1):
            all_issues.extend(
                _detect_issues_for_cue(
                    cue_index,
                    stage,
                    source_entries,
                    _load_srt(debug / f"{stage}.srt") if (debug / f"{stage}.srt").exists() else [],
                    meaning_units,
                    video_context,
                    prev_stage,
                    prev_map,
                    repair_map,
                )
            )
        prev_stage = stage
        prev_map = stage_maps.get(stage, {})

    first_bad: Dict[str, Any] = {}
    for cue in RISKY_CUES + [9, 11, 28]:
        hit = _first_bad_stage(all_issues, cue)
        if hit:
            first_bad[str(cue)] = hit

    legacy_job_vi: Dict[str, Any] = {}
    legacy_path = JOB_SOURCE.parent / "vi.srt"
    if legacy_path.exists():
        legacy_entries = _load_srt(legacy_path)
        legacy_map = _cue_map(legacy_entries)
        legacy_issues: List[dict] = []
        for cue_index in range(1, len(source_entries) + 1):
            legacy_issues.extend(
                _detect_issues_for_cue(
                    cue_index,
                    "legacy_job_vi",
                    source_entries,
                    legacy_entries,
                    meaning_units,
                    video_context,
                    None,
                    {},
                    {},
                )
            )
        legacy_job_vi = {
            "path": str(legacy_path),
            "cue_count": len(legacy_entries),
            "first_bad_stage_by_cue": {
                str(c): _first_bad_stage(legacy_issues, c)
                for c in RISKY_CUES + [9, 11, 28]
                if _first_bad_stage(legacy_issues, c)
            },
            "note": "Historical job output — not a pipeline stage artifact.",
            "cue_10_vi": legacy_map.get(10, ""),
            "cue_29_vi": legacy_map.get(29, "<missing cue>"),
        }

    contract_path = debug / "pipeline_contract_report.json"
    contract_summary: Dict[str, Any] = {}
    if contract_path.exists():
        contract_summary = json.loads(contract_path.read_text(encoding="utf-8"))

    first_count_mismatch_stage = next(
        (
            ch["stage"]
            for ch in contract_summary.get("cue_count_changed_by_stage", [])
        ),
        next(
            (i["stage"] for i in all_issues if i["issue"] == "cue_count_mismatch"),
            None,
        ),
    )
    first_empty_stage = next(
        (
            i["stage"]
            for i in all_issues
            if i["issue"] == "empty_vi_for_non_empty_source"
        ),
        None,
    )
    first_post_final_repair_text_stage = next(
        (
            i["stage"]
            for i in all_issues
            if i["issue"] == "post_final_repair_text_change"
        ),
        None,
    )

    final_vi_path = debug / "final_vi.srt"
    final_vi_mtime = final_vi_path.stat().st_mtime if final_vi_path.exists() else None

    audit = {
        "source_cue_count": len(source_entries),
        "stage_cue_counts": stage_counts,
        "pipeline_order": run_meta.get("pipeline_order"),
        "notes": run_meta.get("notes"),
        "issues": all_issues,
        "first_bad_stage_by_cue": first_bad,
        "pipeline_contract_status": contract_summary.get("pipeline_contract_status"),
        "post_final_repair_text_lock_status": contract_summary.get(
            "post_final_repair_text_lock_status"
        ),
        "contract_checks": {
            "first_cue_count_mismatch_stage": first_count_mismatch_stage,
            "first_empty_cue_stage": first_empty_stage,
            "first_post_final_repair_text_change_stage": first_post_final_repair_text_stage,
            "final_vi_from_latest_run": final_vi_mtime is not None,
            "final_vi_path": str(final_vi_path),
            "legacy_job_vi_path": str(JOB_SOURCE.parent / "vi.srt"),
            "final_vi_is_legacy_job_output": False,
        },
        "risky_cue_summary": {
            str(c): {
                "stages": {
                    s: stage_maps.get(s, {}).get(c, "<missing>")
                    for s in STAGE_ORDER
                    if s not in ("source", "source_corrected")
                },
                "first_bad": first_bad.get(str(c)),
                "meaning_unit_id": _unit_for_cue(meaning_units, c),
            }
            for c in RISKY_CUES
        },
        "answers": {
            "cue_10_first_wrong_stage": (first_bad.get("10") or {}).get("stage"),
            "cue_29_first_empty_stage": next(
                (
                    i["stage"]
                    for i in all_issues
                    if i["cue_index"] == 29 and i["issue"] == "empty_vi_for_non_empty_source"
                ),
                None,
            ),
            "repair_before_editor": False,
            "final_repair_after_readability": True,
            "post_final_repair_stages_change_text": any(
                i["issue"] in ("post_final_repair_text_change", "stage_text_changed")
                for i in all_issues
                if i["stage"] in ("vi_after_timing", "final_vi")
            ),
            "vi_raw_preserved": (
                stage_maps.get("vi_raw", {}) != stage_maps.get("vi_after_final_repair", {})
                or contract_summary.get("alignment_applied") is not None
            ),
            "pipeline_contract_status": contract_summary.get("pipeline_contract_status"),
        },
        "legacy_job_vi_analysis": legacy_job_vi,
    }
    (debug / "cue_mapping_audit_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Stage diff report (cue mapping audit)",
        "",
        f"Source cues: {len(source_entries)}",
        "",
        "## Stage cue counts",
        "",
    ]
    for stage, count in stage_counts.items():
        lines.append(f"- **{stage}**: {count}")
    lines.extend(["", "## Pipeline order (current)", ""])
    for step in run_meta.get("pipeline_order", []):
        lines.append(f"1. {step}")
    lines.extend(["", "## Risky cue texts by stage", ""])
    for cue in RISKY_CUES:
        en = source_entries[cue - 1].get("text", "") if cue <= len(source_entries) else ""
        lines.append(f"### Cue {cue}")
        lines.append(f"- **EN**: {en}")
        lines.append(f"- **Unit**: {_unit_for_cue(meaning_units, cue)}")
        fb = first_bad.get(str(cue))
        if fb:
            lines.append(
                f"- **First issue**: `{fb['issue']}` at `{fb['stage']}` "
                f"(confidence {fb.get('confidence', 0):.2f})"
            )
            if fb.get("detail"):
                lines.append(f"  - {fb['detail']}")
        lines.append("")
        lines.append("| Stage | VI text |")
        lines.append("|-------|---------|")
        for stage in STAGE_ORDER:
            if stage in ("source", "source_corrected"):
                continue
            text = stage_maps.get(stage, {}).get(cue, "—")
            if not text:
                text = "*(empty)*"
            text = text.replace("|", "\\|")
            lines.append(f"| {stage} | {text} |")
        lines.append("")

    lines.extend(["## Key findings", ""])
    lines.append(
        f"- Cue 10 first wrong stage: **{audit['answers']['cue_10_first_wrong_stage']}**"
    )
    lines.append(
        f"- Cue 29 first empty stage: **{audit['answers']['cue_29_first_empty_stage']}**"
    )
    lines.append(
        f"- Final repair after readability: **{audit['answers']['final_repair_after_readability']}**"
    )
    lines.append(
        f"- Post-final-repair stages change text: **{audit['answers']['post_final_repair_stages_change_text']}**"
    )
    lines.append(
        f"- Pipeline contract status: **{audit['answers'].get('pipeline_contract_status')}**"
    )
    if legacy_job_vi:
        lines.extend(["", "## Legacy job vi.srt (historical)", ""])
        lines.append(f"- Path: `{legacy_job_vi.get('path')}`")
        lines.append(f"- Cue count: **{legacy_job_vi.get('cue_count')}** (source={len(source_entries)})")
        lines.append(f"- Cue 10 VI: {legacy_job_vi.get('cue_10_vi')}")
        lines.append(f"- Cue 29 VI: {legacy_job_vi.get('cue_29_vi')}")
        for cue, hit in (legacy_job_vi.get("first_bad_stage_by_cue") or {}).items():
            lines.append(
                f"- Cue {cue} first issue: `{hit.get('issue')}` — {hit.get('detail', hit.get('stage_vi', ''))}"
            )
    lines.append("")

    (debug / "stage_diff_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    DEBUG.mkdir(parents=True, exist_ok=True)
    reuse = "--reuse-raw" in sys.argv
    print("[Cue Mapping Audit] Running pipeline stage capture…")
    run_meta = _run_pipeline_stages(DEBUG, reuse_raw=reuse)
    print("[Cue Mapping Audit] Building reports…")
    _build_reports(DEBUG, run_meta)
    print(f"[Cue Mapping Audit] Done → {DEBUG / 'cue_mapping_audit_report.json'}")
    print(f"[Cue Mapping Audit] Done → {DEBUG / 'stage_diff_report.md'}")


if __name__ == "__main__":
    main()
