"""Window-level repair for local cue-shift misalignment."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cue_shift_detector import detect_local_shift_windows
from .semantic_alignment_guard import apply_validated_repairs, parse_repair_response_structured

_WINDOW_REPAIR_SYSTEM = """You are a Vietnamese subtitle alignment repair editor.

Fix LOCAL CUE MISALIGNMENT: each Vietnamese subtitle line must match the English line at the SAME cue index.
- Output one Vietnamese line per cue index in the window.
- Preserve the speaker's meaning from each English cue; do not shift ideas across cues.
- Natural spoken Vietnamese for on-screen subtitles.
- Do NOT change cue count. Do NOT leave empty text for non-empty English cues.
- Do NOT repeat the same phrase across adjacent cues unless English does.

Return JSON only:
{"windows": [{"window_id": 10000, "cues": [{"cue_index": 8, "text": "..."}, {"cue_index": 9, "text": "..."}]}]}"""


def _glossary_block(video_context: Optional[Dict[str, Any]]) -> str:
    lines = []
    for item in (video_context or {}).get("key_terms") or []:
        src = item.get("source", "")
        sug = item.get("suggested_vi", "")
        if src:
            lines.append(f"- {src} → {sug or '(use natural Vietnamese)'}")
    return "\n".join(lines) if lines else "(none)"


def _build_window_repair_prompt(
    windows: List[dict],
    source_entries: List[dict],
    vi_entries: List[dict],
    video_context: Optional[Dict[str, Any]],
) -> str:
    blocks: List[str] = []
    for w in windows:
        wid = w["window_id"]
        indexes = w["cue_indexes"]
        lo = max(1, min(indexes) - 1)
        hi = min(len(source_entries), max(indexes) + 1)
        ctx_before = ""
        if lo < min(indexes):
            ctx_before = (
                f"Context cue {lo} EN: {source_entries[lo - 1].get('text', '')}\n"
                f"Context cue {lo} VI: {vi_entries[lo - 1].get('text', '')}\n"
            )
        ctx_after = ""
        if hi > max(indexes):
            ctx_after = (
                f"Context cue {hi} EN: {source_entries[hi - 1].get('text', '')}\n"
                f"Context cue {hi} VI: {vi_entries[hi - 1].get('text', '')}\n"
            )
        en_lines = []
        vi_lines = []
        for idx in indexes:
            en_lines.append(f"{idx}: {source_entries[idx - 1].get('text', '')}")
            vi_lines.append(f"{idx}: {vi_entries[idx - 1].get('text', '')}")
        blocks.append(
            f"### Window {wid} (pattern: {w.get('pattern', '')})\n"
            f"REQUIRED cue indexes (return ONLY these): {indexes}\n"
            f"{ctx_before}"
            f"English cues to align (one VI line each):\n"
            + "\n".join(en_lines)
            + "\n\nCurrent misaligned Vietnamese:\n"
            + "\n".join(vi_lines)
            + f"\n{ctx_after}"
        )
    return (
        f"Video summary: {(video_context or {}).get('short_summary', '')}\n"
        f"Glossary:\n{_glossary_block(video_context)}\n\n"
        "Repair these windows — each VI line must match its EN cue at the same index:\n\n"
        + "\n\n".join(blocks)
    )


def _call_window_repair_model(prompt: str, engine: str) -> str:
    if engine == "gemini":
        from .gemini_translate import _call_gemini_json
        from .config import get_gemini_model

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        content, _ = _call_gemini_json(
            api_key, get_gemini_model(), _WINDOW_REPAIR_SYSTEM, prompt, temperature=0.2
        )
        return content

    from openai import OpenAI

    from .config import get_openai_model
    from .openai_chat import create_chat_completion

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = create_chat_completion(
        client,
        get_openai_model(),
        messages=[
            {"role": "system", "content": _WINDOW_REPAIR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _parse_window_repair_response(content: str) -> List[dict]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    data = json.loads(content)
    repair_units: List[dict] = []
    for w in data.get("windows") or []:
        wid = w.get("window_id")
        cues = w.get("cues") or []
        if wid is None or not cues:
            continue
        repair_units.append({"unit_id": wid, "cues": cues})
    return repair_units


def repair_cue_shift_windows(
    source_entries: List[dict],
    vi_entries: List[dict],
    *,
    engine: str,
    meaning_units: Optional[List[dict]] = None,
    video_context: Optional[Dict[str, Any]] = None,
    windows: Optional[List[dict]] = None,
    debug_dir: Optional[str] = None,
    max_windows: int = 4,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Detect (if needed) and repair local cue-shift windows."""
    meta: Dict[str, Any] = {
        "applied": False,
        "skipped_reason": None,
        "windows_detected": 0,
        "windows_requested": 0,
        "windows_repaired": 0,
        "window_repairs_accepted": 0,
        "window_repairs_rejected": 0,
        "accepted": {},
        "rejected": {},
        "windows": [],
    }

    if windows is None:
        windows = detect_local_shift_windows(
            source_entries, vi_entries, meaning_units, video_context
        )
    meta["windows_detected"] = len(windows)
    meta["windows"] = windows

    if not windows:
        meta["skipped_reason"] = "no_shift_windows"
        return list(vi_entries), meta

    to_repair = sorted(windows, key=lambda w: -w.get("confidence", 0))[:max_windows]
    meta["windows_requested"] = len(to_repair)

    expected_units = [
        {"unit_id": w["window_id"], "cue_indexes": w["cue_indexes"]} for w in to_repair
    ]
    prompt = _build_window_repair_prompt(
        to_repair, source_entries, vi_entries, video_context
    )

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        (Path(debug_dir) / "cue_shift_window_repair_prompt.txt").write_text(
            prompt, encoding="utf-8"
        )

    try:
        raw = _call_window_repair_model(prompt, engine)
        meta["raw_response"] = raw
        if debug_dir:
            (Path(debug_dir) / "cue_shift_window_repair_response.json").write_text(
                raw, encoding="utf-8"
            )
        repair_units = _parse_window_repair_response(raw)
    except Exception as exc:
        meta["skipped_reason"] = f"window_repair_failed: {exc}"
        print(f"[Cue Shift Repair] Model call failed ({exc})")
        return list(vi_entries), meta

    if not repair_units:
        meta["skipped_reason"] = "empty_window_repair_response"
        return list(vi_entries), meta

    expected_by_id = {u["unit_id"]: u for u in expected_units}
    filtered_units: List[dict] = []
    for unit in repair_units:
        uid = unit.get("unit_id")
        expected = expected_by_id.get(uid)
        if not expected:
            continue
        allowed = set(expected.get("cue_indexes") or [])
        cues = [
            c
            for c in (unit.get("cues") or [])
            if isinstance(c.get("cue_index"), int) and c["cue_index"] in allowed
        ]
        if len(cues) == len(allowed):
            filtered_units.append({"unit_id": uid, "cues": cues})
    repair_units = filtered_units

    if not repair_units:
        meta["skipped_reason"] = "window_repair_index_mismatch"
        return list(vi_entries), meta

    result, apply_meta = apply_validated_repairs(
        source_entries,
        vi_entries,
        repair_units,
        expected_units,
        meaning_units,
        video_context,
    )
    meta["contract"] = apply_meta.get("contract")
    meta["accepted"] = apply_meta.get("accepted", {})
    meta["rejected"] = apply_meta.get("rejected", {})
    meta["window_repairs_accepted"] = len(meta["accepted"])
    meta["window_repairs_rejected"] = len(meta["rejected"])
    meta["windows_repaired"] = len(
        {apply_meta.get("accepted", {}).get(k, {}).get("unit_id") for k in meta["accepted"]}
    ) if meta["accepted"] else 0
    meta["applied"] = bool(meta["accepted"])
    if not meta["applied"]:
        meta["skipped_reason"] = meta.get("skipped_reason") or "no_window_repairs_accepted"

    return result, meta
