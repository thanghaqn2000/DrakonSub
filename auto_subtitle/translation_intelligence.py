"""Orchestrate pre/post translation intelligence: context, units, QA, repair."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .meaning_unit_builder import build_meaning_units, save_meaning_units
from .semantic_alignment_guard import analyze_semantic_alignment, save_alignment_report
from .translation_error_taxonomy import get_error_type
from .translation_quality_analyzer import analyze_translation_quality, save_quality_report
from .cps_diagnosis import (
    build_cps_diagnosis_report,
    merge_delivery_quality_report,
    save_cps_diagnosis_report,
)
from .repair_unit_selector import select_units_for_repair
from .cue_shift_detector import detect_local_shift_windows, diagnose_sample
from .cue_shift_repair import repair_cue_shift_windows
from .unit_repair_redistributor import (
    apply_unit_rewrite_repairs,
    parse_unit_rewrite_response,
)
from .utils import write_srt_entries
from .video_context_analyzer import analyze_video_context, save_video_context

_UNIT_REWRITE_SYSTEM = """You are a Vietnamese subtitle meaning repair editor for general adult viewers.

Your job is to rewrite the FULL MEANING of each English unit into one clean Vietnamese passage.
- Do NOT return per-cue text or cue indexes.
- Do NOT drop ideas from the English source unit.
- Do NOT add generic explanations not in the source.
- Remove awkward standalone fragments; write one flowing natural Vietnamese unit.
- Use video context, glossary, and ASR risk notes when helpful.
- Style: neutral, concise, clear, natural — easy for general Vietnamese adults.

Return JSON only:
{"units": [{"unit_id": 1, "unit_translation": "one clean Vietnamese version of the full unit", "notes": "optional short reason"}, ...]}"""

_REPAIR_INSTRUCTIONS: Dict[str, str] = {
    "cue_flow_error": (
        "Remove standalone weak fragments. Write one coherent Vietnamese unit that reads "
        "naturally when split across cues later."
    ),
    "repeated_meaning_error": (
        "Do not repeat the same question or phrase. Say each idea once in the unit."
    ),
    "semantic_alignment_error": (
        "Match English source meaning and order. Do not shift ideas or invent generic text."
    ),
    "semantic_drift_error": (
        "Re-align to source meaning. Do not add or drop key ideas."
    ),
    "readability_cps_error": (
        "Prefer shorter, simpler Vietnamese while keeping full meaning in the unit."
    ),
    "literal_translation_error": (
        "Rewrite for natural spoken Vietnamese while keeping meaning."
    ),
    "missing_or_empty_cue_error": (
        "Ensure the unit translation covers every non-empty source cue's meaning."
    ),
    "possible_asr_term_unresolved": (
        "ASR RISK: flagged source phrase may be a transcription error. Infer likely intent "
        "from context if confident; otherwise keep neutral literal wording."
    ),
}


@dataclass
class TranslationIntelligenceContext:
    video_context: Dict[str, Any]
    meaning_units: List[dict]
    engine: str = "openai"
    debug_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_context": self.video_context,
            "meaning_units": self.meaning_units,
            "engine": self.engine,
        }


def run_pre_translation_intelligence(
    source_entries: List[dict],
    debug_dir: str,
    *,
    user_topic: str = "auto",
    audience_level: str = "general_beginner",
    style: str = "simple_vietnamese_subtitle",
    engine: str = "openai",
) -> TranslationIntelligenceContext:
    """Video context + meaning units; saves debug artifacts."""
    out = Path(debug_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[Translation Intelligence] Analyzing video context…")
    video_context = analyze_video_context(
        source_entries,
        user_topic=user_topic,
        audience_level=audience_level,
        style=style,
        engine=engine,
    )
    save_video_context(out / "video_context.json", video_context)

    print("[Translation Intelligence] Building meaning units…")
    meaning_units = build_meaning_units(source_entries)
    save_meaning_units(out / "meaning_units.json", meaning_units)

    return TranslationIntelligenceContext(
        video_context=video_context,
        meaning_units=meaning_units,
        engine=engine,
        debug_dir=debug_dir,
    )


def _write_subset_srt(path: Path, entries: List[dict], cue_indexes: List[int], texts: Dict[int, str]) -> None:
    subset = []
    for idx in cue_indexes:
        if 0 < idx <= len(entries):
            e = dict(entries[idx - 1])
            e["text"] = texts.get(idx, e.get("text", ""))
            subset.append(e)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        write_srt_entries(subset, file=f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _asr_risks_for_unit(
    unit: dict,
    source_entries: List[dict],
    video_context: Dict[str, Any],
) -> List[str]:
    cues = unit.get("cue_indexes") or []
    unit_source = " ".join(
        source_entries[c - 1].get("text", "") for c in cues if 0 < c <= len(source_entries)
    ).lower()
    found = []
    for risk in video_context.get("possible_asr_risks") or []:
        phrase = str(risk).strip()
        if phrase and phrase.lower() in unit_source:
            found.append(phrase)
    return found


def _build_unit_rewrite_requests(
    risky_units: List[dict],
    source_entries: List[dict],
    vi_entries: List[dict],
    video_context: Dict[str, Any],
    recommendations: List[dict],
) -> List[dict]:
    """Structured request payloads for debug logging."""
    rec_by_unit = {r["unit_id"]: r for r in recommendations if r.get("unit_id")}
    requests = []
    for unit in risky_units:
        uid = unit["unit_id"]
        cues = unit["cue_indexes"]
        rec = rec_by_unit.get(uid, {})
        error_types = unit.get("detected_translation_errors") or rec.get("error_types") or []
        requests.append(
            {
                "unit_id": uid,
                "cue_indexes": cues,
                "source_unit_text": unit.get("source_text", ""),
                "current_vi_cues": {
                    str(c): vi_entries[c - 1].get("text", "") if c <= len(vi_entries) else ""
                    for c in cues
                },
                "error_types": error_types,
                "asr_risks": _asr_risks_for_unit(unit, source_entries, video_context),
            }
        )
    return requests


def _build_unit_rewrite_prompt(
    risky_units: List[dict],
    source_entries: List[dict],
    vi_entries: List[dict],
    video_context: Dict[str, Any],
    recommendations: List[dict],
) -> str:
    rec_by_unit = {r["unit_id"]: r for r in recommendations if r.get("unit_id")}

    blocks = []
    for unit in risky_units:
        uid = unit["unit_id"]
        cues = unit["cue_indexes"]
        rec = rec_by_unit.get(uid, {})
        error_types = unit.get("detected_translation_errors") or rec.get("error_types") or []
        strategies = []
        for eid in error_types:
            if eid in _REPAIR_INSTRUCTIONS:
                strategies.append(_REPAIR_INSTRUCTIONS[eid])
            else:
                try:
                    strategies.append(f"{eid}: {get_error_type(eid).fix_strategy}")
                except KeyError:
                    pass

        asr_risks = _asr_risks_for_unit(unit, source_entries, video_context)

        cue_lines = []
        for c in cues:
            en = source_entries[c - 1].get("text", "") if c <= len(source_entries) else ""
            vi = vi_entries[c - 1].get("text", "") if c <= len(vi_entries) else ""
            cue_lines.append(f"  [{c}] EN: {en}\n       VI: {vi}")

        asr_block = ""
        if asr_risks:
            asr_block = (
                "ASR risks in this unit (may be transcription errors — infer from context, "
                "do not force a fix if uncertain):\n"
                + "\n".join(f"  - {r}" for r in asr_risks)
                + "\n"
            )

        blocks.append(
            f"Unit {uid}\n"
            f"English source (full unit): {unit.get('source_text', '')}\n"
            f"Current Vietnamese cues (for reference only — do not copy fragments):\n"
            + "\n".join(cue_lines)
            + f"\nConfirmed errors: {', '.join(error_types)}\n"
            f"Guidance:\n"
            + "\n".join(f"  - {s}" for s in strategies)
            + f"\n{asr_block}"
            + "Write ONE clean Vietnamese unit_translation covering the full English meaning."
        )

    glossary = json.dumps(video_context.get("key_terms") or [], ensure_ascii=False, indent=2)
    asr_global = video_context.get("possible_asr_risks") or []
    asr_global_block = ""
    if asr_global:
        asr_global_block = (
            "Global possible ASR risks for this video:\n"
            + "\n".join(f"- {r}" for r in asr_global[:8])
            + "\n\n"
        )
    return (
        f"Video context summary: {video_context.get('short_summary', '')}\n"
        f"Tone: {video_context.get('tone', '')}\n"
        f"Style: {video_context.get('translation_style', '')}\n\n"
        f"{asr_global_block}"
        f"Glossary:\n{glossary}\n\n"
        "Rewrite these units (unit_translation only, no per-cue output):\n\n"
        + "\n\n".join(blocks)
    )


def _call_unit_rewrite_model(prompt: str, engine: str) -> str:
    if engine == "gemini":
        from .gemini_translate import _call_gemini_json
        from .config import get_gemini_model

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        content, _ = _call_gemini_json(
            api_key, get_gemini_model(), _UNIT_REWRITE_SYSTEM, prompt, temperature=llm_temperature(0.25)
        )
        return content

    from openai import OpenAI

    from .config import get_openai_model, llm_chat_kwargs, llm_temperature
    from .openai_chat import create_chat_completion

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = create_chat_completion(
        client,
        get_openai_model(),
        messages=[
            {"role": "system", "content": _UNIT_REWRITE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=llm_temperature(0.25),
        response_format={"type": "json_object"},
        **llm_chat_kwargs(),
    )
    return response.choices[0].message.content or ""


def repair_risky_units(
    source_entries: List[dict],
    vi_entries: List[dict],
    intel_ctx: TranslationIntelligenceContext,
    quality_report: Dict[str, Any],
    *,
    debug_dir: Optional[str] = None,
) -> tuple[List[dict], Dict[str, Any]]:
    """Unit-level rewrite + deterministic redistribution with validation."""
    meta: Dict[str, Any] = {
        "applied": False,
        "skipped_reason": None,
        "units_requested": 0,
        "repair_mode": "unit_rewrite_redistribute",
        "repair_selection": None,
        "unit_rewrite_requests": None,
        "unit_rewrite_responses": None,
        "redistribution_report": None,
        "vi_cue_fluency_report": None,
        "contract_validation": None,
        "accepted_repairs": {},
        "rejected_repairs": {},
        "rejected_units": {},
        "human_review_units": [],
    }

    risky_units = quality_report.get("risky_units") or []
    recommendations = [
        r for r in quality_report.get("repair_recommendations") or []
        if r.get("action") == "repair_unit"
    ]

    if not risky_units or not recommendations:
        meta["skipped_reason"] = "no_confirmed_errors"
        return list(vi_entries), meta

    if quality_report.get("quality_score", 100) >= 85 and len(risky_units) <= 1:
        meta["skipped_reason"] = "quality_acceptable"
        print("[Translation Intelligence] Quality acceptable; skipping repair call")
        return list(vi_entries), meta

    selection = select_units_for_repair(
        risky_units,
        cue_assessments=quality_report.get("cue_assessments"),
    )
    units_to_repair = selection["selected_units"]
    meta["repair_selection"] = selection
    meta["units_requested"] = len(units_to_repair)

    if not units_to_repair:
        meta["skipped_reason"] = "no_units_selected"
        return list(vi_entries), meta

    print(
        f"[Translation Intelligence] Unit rewrite + redistribute for "
        f"{len(units_to_repair)} unit(s) [ids: {selection['selected_unit_ids']}]…"
    )

    rewrite_requests = _build_unit_rewrite_requests(
        units_to_repair,
        source_entries,
        vi_entries,
        intel_ctx.video_context,
        recommendations,
    )
    meta["unit_rewrite_requests"] = rewrite_requests

    prompt = _build_unit_rewrite_prompt(
        units_to_repair,
        source_entries,
        vi_entries,
        intel_ctx.video_context,
        recommendations,
    )

    if debug_dir:
        _save_json(Path(debug_dir) / "unit_rewrite_requests.json", rewrite_requests)
        _save_json(Path(debug_dir) / "repair_selection_report.json", selection)

    try:
        raw = _call_unit_rewrite_model(prompt, intel_ctx.engine)
        meta["raw_response"] = raw
        unit_rewrites = parse_unit_rewrite_response(raw)
        meta["unit_rewrite_responses"] = unit_rewrites
        if debug_dir:
            _save_json(Path(debug_dir) / "unit_rewrite_responses.json", unit_rewrites)
    except Exception as exc:
        meta["skipped_reason"] = f"unit_rewrite_failed: {exc}"
        print(f"[Translation Intelligence] Unit rewrite failed ({exc})")
        return list(vi_entries), meta

    if not unit_rewrites:
        meta["skipped_reason"] = "empty_unit_rewrite_response"
        return list(vi_entries), meta

    result, apply_meta = apply_unit_rewrite_repairs(
        source_entries,
        vi_entries,
        unit_rewrites,
        units_to_repair,
        intel_ctx.meaning_units,
        intel_ctx.video_context,
    )
    meta["contract_validation"] = apply_meta.get("contract")
    meta["redistribution_report"] = apply_meta.get("redistribution_reports", [])
    meta["vi_cue_fluency_report"] = apply_meta.get("vi_fluency_reports", [])
    meta["accepted_repairs"] = apply_meta.get("accepted", {})
    meta["rejected_repairs"] = apply_meta.get("rejected", {})
    meta["rejected_units"] = apply_meta.get("rejected_units", {})
    meta["human_review_units"] = apply_meta.get("human_review_units", [])
    meta["applied"] = apply_meta.get("applied", False)

    if debug_dir:
        _save_json(
            Path(debug_dir) / "redistribution_report.json",
            meta["redistribution_report"],
        )
        _save_json(
            Path(debug_dir) / "vi_cue_fluency_report.json",
            meta["vi_cue_fluency_report"],
        )

    if not meta["applied"]:
        meta["skipped_reason"] = meta.get("skipped_reason") or "no_repairs_accepted"

    return result, meta


def _write_repair_diff(path: Path, repair_meta: Dict[str, Any], report_before: Dict, report_after: Dict) -> None:
    lines = [
        "# Repair applied diff",
        "",
        f"- Repair applied: **{repair_meta.get('applied')}**",
        f"- Skip reason: {repair_meta.get('skipped_reason') or 'n/a'}",
        f"- Units requested: {repair_meta.get('units_requested', 0)}",
        f"- Quality before: {report_before.get('quality_score')} ({report_before.get('score_band')})",
        f"- Quality after: {report_after.get('quality_score')} ({report_after.get('score_band')})",
        f"- Risky cues before: {report_before.get('summary', {}).get('risky_cue_count', len(report_before.get('risky_cues', [])))}",
        f"- Risky cues after: {report_after.get('summary', {}).get('risky_cue_count', len(report_after.get('risky_cues', [])))}",
        "",
        "## Cue text changes",
    ]
    updates = repair_meta.get("accepted_repairs") or repair_meta.get("cues_updated") or {}
    if not updates:
        lines.append("- (no cue text changes accepted)")
    else:
        for idx in sorted(updates, key=lambda x: int(x) if str(x).isdigit() else x):
            item = updates[idx]
            if isinstance(item, dict):
                lines.append(f"\n### Cue {idx}")
                lines.append(f"- Before: {item.get('before', '')}")
                lines.append(f"- After: {item.get('after', '')}")
    rejected = repair_meta.get("rejected_repairs") or {}
    if rejected:
        lines.append("\n## Rejected repairs")
        for idx in sorted(rejected, key=int):
            item = rejected[idx]
            lines.append(f"- Cue {idx}: {item.get('reason')} | proposed: {item.get('proposed', '')}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_post_translation_qa(
    source_entries: List[dict],
    vi_entries: List[dict],
    intel_ctx: TranslationIntelligenceContext,
    debug_dir: str,
) -> tuple[List[dict], Dict[str, Any]]:
    """Analyze quality, optionally repair risky units, save debug artifacts."""
    out = Path(debug_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_before = analyze_translation_quality(
        source_entries,
        vi_entries,
        intel_ctx.video_context,
        intel_ctx.meaning_units,
    )
    alignment_before = analyze_semantic_alignment(
        source_entries, vi_entries, intel_ctx.meaning_units, intel_ctx.video_context
    )
    save_alignment_report(out / "semantic_alignment_report_before_repair.json", alignment_before)
    _save_json(out / "qa_before_repair.json", report_before)

    risky_unit_cues_before = sorted(
        {rc["cue_index"] for rc in report_before.get("risky_cues", [])}
    )
    _save_json(out / "risky_units_before_repair.json", report_before.get("risky_units", []))
    if risky_unit_cues_before:
        _write_subset_srt(
            out / "risky_units_before_repair.srt",
            vi_entries,
            risky_unit_cues_before,
            {i: vi_entries[i - 1].get("text", "") for i in risky_unit_cues_before},
        )

    repaired, repair_meta = repair_risky_units(
        source_entries, vi_entries, intel_ctx, report_before, debug_dir=str(out)
    )

    shift_windows = detect_local_shift_windows(
        source_entries,
        repaired,
        intel_ctx.meaning_units,
        intel_ctx.video_context,
    )
    shift_diagnosis = diagnose_sample(
        "job",
        source_entries,
        repaired,
        intel_ctx.meaning_units,
        intel_ctx.video_context,
    )
    shift_diagnosis["shift_windows"] = shift_windows
    _save_json(out / "cue_shift_diagnosis_sample.json", shift_diagnosis)

    if shift_windows:
        print(
            f"[Translation Intelligence] Cue-shift window repair for "
            f"{len(shift_windows)} window(s)…"
        )
        repaired, shift_meta = repair_cue_shift_windows(
            source_entries,
            repaired,
            engine=intel_ctx.engine,
            meaning_units=intel_ctx.meaning_units,
            video_context=intel_ctx.video_context,
            windows=shift_windows,
            debug_dir=str(out),
        )
        repair_meta["cue_shift_repair"] = shift_meta
        shift_diagnosis["local_cue_shift_windows_detected"] = len(shift_windows)
        shift_diagnosis["local_cue_shift_windows_repaired"] = shift_meta.get(
            "windows_requested", 0
        )
        shift_diagnosis["window_repairs_accepted"] = shift_meta.get(
            "window_repairs_accepted", 0
        )
        shift_diagnosis["window_repairs_rejected"] = shift_meta.get(
            "window_repairs_rejected", 0
        )
    else:
        repair_meta["cue_shift_repair"] = {"skipped_reason": "no_shift_windows"}
        shift_diagnosis["local_cue_shift_windows_detected"] = 0
        shift_diagnosis["local_cue_shift_windows_repaired"] = 0
        shift_diagnosis["window_repairs_accepted"] = 0
        shift_diagnosis["window_repairs_rejected"] = 0

    _save_json(out / "cue_shift_repair_report.json", repair_meta.get("cue_shift_repair") or {})
    _save_json(out / "cue_shift_diagnosis_sample.json", shift_diagnosis)

    report_after = analyze_translation_quality(
        source_entries,
        repaired,
        intel_ctx.video_context,
        intel_ctx.meaning_units,
    )
    alignment_after = analyze_semantic_alignment(
        source_entries, repaired, intel_ctx.meaning_units, intel_ctx.video_context
    )
    save_alignment_report(out / "semantic_alignment_report_after_repair.json", alignment_after)
    _save_json(out / "qa_after_repair.json", report_after)

    _save_json(out / "repair_contract_validation.json", repair_meta.get("contract_validation") or {})
    if repair_meta.get("repair_selection") is not None:
        _save_json(out / "repair_selection_report.json", repair_meta["repair_selection"])
    if repair_meta.get("unit_rewrite_requests") is not None:
        _save_json(out / "unit_rewrite_requests.json", repair_meta["unit_rewrite_requests"])
    if repair_meta.get("unit_rewrite_responses") is not None:
        _save_json(out / "unit_rewrite_responses.json", repair_meta["unit_rewrite_responses"])
    if repair_meta.get("redistribution_report") is not None:
        _save_json(out / "redistribution_report.json", repair_meta["redistribution_report"])
    if repair_meta.get("vi_cue_fluency_report") is not None:
        _save_json(out / "vi_cue_fluency_report.json", repair_meta["vi_cue_fluency_report"])
    _save_json(out / "accepted_repairs.json", repair_meta.get("accepted_repairs") or {})
    _save_json(out / "rejected_repairs.json", repair_meta.get("rejected_repairs") or {})

    risky_unit_cues_after = sorted(
        {rc["cue_index"] for rc in report_after.get("risky_cues", [])}
    )
    _save_json(out / "risky_units_after_repair.json", report_after.get("risky_units", []))
    if risky_unit_cues_after:
        _write_subset_srt(
            out / "risky_units_after_repair.srt",
            repaired,
            risky_unit_cues_after,
            {i: repaired[i - 1].get("text", "") for i in risky_unit_cues_after},
        )
    else:
        (out / "risky_units_after_repair.srt").write_text("", encoding="utf-8")

    _write_repair_diff(out / "repair_applied_diff.md", repair_meta, report_before, report_after)

    combined_report = {
        **report_after,
        "before_repair": {
            "quality_score": report_before.get("quality_score"),
            "score_band": report_before.get("score_band"),
            "risky_cue_count": report_before.get("summary", {}).get(
                "risky_cue_count", len(report_before.get("risky_cues", []))
            ),
            "source_risk_summary": report_before.get("source_risk_summary", {}),
        },
        "repair": repair_meta,
    }
    contract_val = repair_meta.get("contract_validation") or {}
    if not contract_val.get("valid", True):
        combined_report["human_review_needed"] = True
    if alignment_after.get("human_review_needed"):
        combined_report["human_review_needed"] = True
    if repair_meta.get("human_review_units"):
        combined_report["human_review_needed"] = True
    save_quality_report(out / "translation_quality_report.json", combined_report)

    with open(out / "final_vi.srt", "w", encoding="utf-8") as f:
        write_srt_entries(repaired, file=f)

    if combined_report.get("human_review_needed"):
        print("[Translation Intelligence] Human review recommended for some cues")

    return repaired, combined_report


def finalize_delivery_quality_report(
    source_entries: List[dict],
    pre_timing_entries: List[dict],
    post_timing_entries: List[dict],
    intel_ctx: TranslationIntelligenceContext,
    pre_timing_report: Dict[str, Any],
    debug_dir: str,
) -> Dict[str, Any]:
    """
    Re-score delivery subtitles after timing-only stage and emit CPS diagnosis.

    Viewer-facing quality uses post-timing entries; pre-timing CPS warnings that
    timing resolves are documented in cps_diagnosis_report.json.
    """
    out = Path(debug_dir)
    cps_report = build_cps_diagnosis_report(
        source_entries,
        pre_timing_entries,
        post_timing_entries,
        intel_ctx.meaning_units,
        intel_ctx.video_context,
    )
    save_cps_diagnosis_report(out / "cps_diagnosis_report.json", cps_report)

    post_timing_report = analyze_translation_quality(
        source_entries,
        post_timing_entries,
        intel_ctx.video_context,
        intel_ctx.meaning_units,
    )
    merged = merge_delivery_quality_report(
        pre_timing_report, post_timing_report, cps_report
    )
  # CPS-only issues fixed by timing should not force human review.
    if merged.get("summary", {}).get("confirmed_error_counts") == {}:
        merged["human_review_needed"] = bool(
            merged.get("semantic_alignment", {}).get("alignment_error_count", 0)
            or merged.get("asr_risks", {}).get("unresolved_cue_count", 0)
        )
    save_quality_report(out / "translation_quality_report.json", merged)
    return merged
