#!/usr/bin/env python3
"""Diagnostic: compare Gemini vs OpenAI subtitle pipelines on Buffett sample."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_subtitle.config import load_env  # noqa: E402
from auto_subtitle.en_domain_corrector import correct_en_domain_srt_file  # noqa: E402
from auto_subtitle.pipeline import SubtitleConfig, translate_srt_file  # noqa: E402
from auto_subtitle.subtitle_readability_optimizer import optimize_readability_file  # noqa: E402
from auto_subtitle.subtitle_timing_optimizer import (  # noqa: E402
    normalize_final_srt_timing,
    optimize_srt_timing_file,
)
from auto_subtitle.utils import parse_srt  # noqa: E402
from auto_subtitle.vi_compression import compress_vi_srt_file  # noqa: E402
from auto_subtitle.vi_editor import edit_vi_srt_file  # noqa: E402
from auto_subtitle.vi_flow import flow_vi_srt_file  # noqa: E402

JOB_SOURCE = Path(
    "/var/folders/kc/wq9gs6yd0pl2b0q6ddqfqvfc0000gn/T/"
    "drakonsub_jobs/6bfeabd2-7d63-4d0c-8561-bbbc61df5891/source.srt"
)
OUT = ROOT / "debug" / "provider_comparison"

TRACK_CASES = {
    "B_cue7_8": [7, 8],
    "C_cue14_15": [14, 15],
    "D_cue24_25": [24, 25],
    "E_cue27_29": [27, 28, 29],
}


def _cue_map(entries: list[dict]) -> dict[int, str]:
    return {i + 1: e.get("text", "").strip() for i, e in enumerate(entries)}


def _save_stage(path: Path, entries: list[dict]) -> None:
    from auto_subtitle.utils import write_srt_entries

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        write_srt_entries(entries, file=f)


def run_provider(provider: str, engine: str) -> dict:
    prefix = provider
    provider_dir = OUT / provider
    provider_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TRANSLATION_ENGINE"] = engine
    os.environ["ENABLE_VI_READABILITY_OPTIMIZER"] = "true"
    os.environ["ENABLE_VI_TIMING_OPTIMIZER"] = "true"
    if engine == "openai":
        os.environ["OPENAI_MODEL"] = "gpt-4o-mini"

    load_env()

    source = provider_dir / f"{prefix}_source.srt"
    shutil.copy2(JOB_SOURCE, source)
    correct_en_domain_srt_file(str(source), str(source), debug_dir=str(provider_dir))

    stages: dict[str, Path] = {}
    working = provider_dir / "working.srt"

    config = SubtitleConfig.from_env()
    config.translation_engine = engine
    config.source_language = "en"

    translate_srt_file(str(source), str(working), config)
    stages["raw_translation"] = provider_dir / f"{prefix}_raw_translation.srt"
    shutil.copy2(working, stages["raw_translation"])

    edit_vi_srt_file(
        str(source),
        str(working),
        str(provider_dir / f"{prefix}_after_editor.srt"),
        translation_engine=engine,
        topic=config.translation_topic,
        debug_dir=str(provider_dir),
    )
    shutil.copy2(working, stages.setdefault("after_editor", provider_dir / f"{prefix}_after_editor.srt"))

    compress_vi_srt_file(str(working))
    stages["after_compression"] = provider_dir / f"{prefix}_after_compression.srt"
    shutil.copy2(working, stages["after_compression"])

    flow_vi_srt_file(str(source), str(working))
    stages["after_flow"] = provider_dir / f"{prefix}_after_flow.srt"
    shutil.copy2(working, stages["after_flow"])

    # No final QA pass in codebase — copy flow output as placeholder
    stages["after_final_qa"] = provider_dir / f"{prefix}_after_final_qa.srt"
    shutil.copy2(working, stages["after_final_qa"])

    optimize_readability_file(str(working))
    optimize_srt_timing_file(str(working))
    normalize_final_srt_timing(str(working))
    stages["final"] = provider_dir / f"{prefix}_final.srt"
    shutil.copy2(working, stages["final"])

    snapshots = {}
    for stage_name, path in stages.items():
        entries = parse_srt(path.read_text(encoding="utf-8"))
        snapshots[stage_name] = _cue_map(entries)

    empty_cues = {
        stage: [n for n, t in cmap.items() if not t]
        for stage, cmap in snapshots.items()
    }

    track = {}
    for case_id, cue_nums in TRACK_CASES.items():
        track[case_id] = {
            stage: {str(n): snapshots[stage].get(n, "") for n in cue_nums}
            for stage in snapshots
        }

    result = {
        "provider": provider,
        "engine": engine,
        "cue_count": len(parse_srt(stages["final"].read_text(encoding="utf-8"))),
        "empty_cues_by_stage": empty_cues,
        "tracked_cases": track,
        "artifact_paths": {k: str(v) for k, v in stages.items()},
    }

    with open(provider_dir / f"{prefix}_comparison.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def dump_prompts() -> None:
    from auto_subtitle.gemini_translate import (
        _GEMINI_SYSTEM_PROMPT,
        _DOMAIN_GLOSSARY,
        _build_gemini_system_prompt,
    )
    from auto_subtitle.openai_translate import _build_grouped_user_prompt
    from auto_subtitle.translation_topics import build_system_prompt
    from auto_subtitle.vi_compression import _COMPRESSION_SYSTEM_PROMPT
    from auto_subtitle.vi_editor import VI_EDITOR_SYSTEM_PROMPT
    from auto_subtitle.vi_flow import _FLOW_SYSTEM_PROMPT

    prompt_dir = OUT / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "openai_raw_system.txt": build_system_prompt("economics"),
        "gemini_raw_system.txt": _build_gemini_system_prompt("economics"),
        "gemini_raw_system_base.txt": _GEMINI_SYSTEM_PROMPT,
        "gemini_domain_glossary.txt": _DOMAIN_GLOSSARY,
        "vi_editor_system.txt": VI_EDITOR_SYSTEM_PROMPT,
        "vi_compression_system.txt": _COMPRESSION_SYSTEM_PROMPT,
        "vi_flow_system.txt": _FLOW_SYSTEM_PROMPT,
        "openai_raw_user_example.txt": _build_grouped_user_prompt(
            [[0, 1]], ["Hello world.", "How are you?"], "Vietnamese"
        ),
    }

    from auto_subtitle.gemini_translate import _build_grouped_user_prompt as gemini_user

    files["gemini_raw_user_example.txt"] = gemini_user(
        [[0, 1]],
        [
            "Earlier line.",
            "Hello world.",
            "How are you?",
            "Later line.",
            "Another later.",
        ],
        "Vietnamese",
    )

    for name, content in files.items():
        (prompt_dir / name).write_text(content, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dump_prompts()

    if not JOB_SOURCE.exists():
        print(f"Missing sample source: {JOB_SOURCE}", file=sys.stderr)
        sys.exit(1)

    results = {}
    if os.environ.get("SKIP_GEMINI_COMPARISON") != "1":
        try:
            results["gemini"] = run_provider("gemini", "gemini")
        except Exception as exc:
            print(f"Gemini comparison skipped/failed: {exc}", file=sys.stderr)
    results["openai"] = run_provider("openai", "openai")

    with open(OUT / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v["cue_count"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
