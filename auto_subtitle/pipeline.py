import json
import os
import shutil
import tempfile
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import ffmpeg
import whisper
from .config import get_translation_engine, load_env

from .utils import (
    write_srt,
    parse_srt,
    write_srt_entries,
    translate_srt_entries,
    assert_vietnamese_translation_applied,
    build_word_aligned_segments,
    str2bool,
)

load_env()

ProgressCallback = Callable[[str, int], None]

# Whisper load/transcribe is not safe across concurrent web job threads.
_WHISPER_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name) or default


@dataclass
class SubtitleConfig:
    model: str = "small"
    translate_to: str = "vi"
    translation_engine: str = "openai"
    subtitle_margin_bottom: float = 32.0   # % of video height (legacy compat)
    subtitle_font_size: int = 55
    subtitle_font_color: str = "#9333EA"
    subtitle_background_color: str = "#FFFFFF"
    subtitle_box_padding: int = 14         # used only in classic ASS mode
    subtitle_reference_height: int = 1920
    language: str = "auto"
    task: str = "transcribe"
    translation_topic: str = "economics"
    source_language: str = "en"            # "en" or "vi"
    vi_loanword_openai: bool = True
    # Rounded subtitle style (rounded mode only)
    subtitle_style_mode: str = "rounded"   # "rounded" | "classic"
    subtitle_border_radius: int = 18
    subtitle_padding_x: int = 28
    subtitle_padding_y: int = 16
    subtitle_text_safe_padding_y: int = 12
    subtitle_background_opacity: float = 1.0
    subtitle_max_width_ratio: float = 0.86
    subtitle_line_spacing: float = 1.15
    subtitle_bottom_margin_ratio: float = 0.11

    @classmethod
    def from_env(cls) -> "SubtitleConfig":
        from .translation_topics import DEFAULT_TOPIC, normalize_topic

        raw_bottom_ratio = os.getenv("SUBTITLE_BOTTOM_MARGIN_RATIO", "").strip()
        if raw_bottom_ratio:
            try:
                bottom_margin_ratio = float(raw_bottom_ratio)
            except ValueError:
                bottom_margin_ratio = _env_float("SUBTITLE_MARGIN_BOTTOM", 32.0) / 100.0
        else:
            bottom_margin_ratio = _env_float("SUBTITLE_MARGIN_BOTTOM", 32.0) / 100.0

        return cls(
            model=_env_str("WHISPER_MODEL", "small"),
            translation_engine=get_translation_engine(),
            subtitle_margin_bottom=_env_float("SUBTITLE_MARGIN_BOTTOM", 32.0),
            subtitle_font_size=_env_int("SUBTITLE_FONT_SIZE", 55),
            subtitle_font_color=_env_str("SUBTITLE_FONT_COLOR", "#9333EA"),
            subtitle_background_color=_env_str(
                "SUBTITLE_BACKGROUND_COLOR", "#FFFFFF"
            ),
            subtitle_box_padding=_env_int("SUBTITLE_BOX_PADDING", 14),
            subtitle_reference_height=_env_int("SUBTITLE_REFERENCE_HEIGHT", 1920),
            translation_topic=normalize_topic(
                _env_str("TRANSLATION_TOPIC", DEFAULT_TOPIC)
            ),
            vi_loanword_openai=str2bool(
                _env_str("VI_LOANWORD_OPENAI_FIX", "true")
            ),
            subtitle_style_mode=_env_str("SUBTITLE_STYLE_MODE", "rounded"),
            subtitle_border_radius=_env_int("SUBTITLE_BORDER_RADIUS", 18),
            subtitle_padding_x=_env_int("SUBTITLE_PADDING_X", 28),
            subtitle_padding_y=_env_int("SUBTITLE_PADDING_Y", 16),
            subtitle_text_safe_padding_y=_env_int("SUBTITLE_TEXT_SAFE_PADDING_Y", 12),
            subtitle_background_opacity=_env_float("SUBTITLE_BACKGROUND_OPACITY", 1.0),
            subtitle_max_width_ratio=_env_float("SUBTITLE_MAX_WIDTH_RATIO", 0.86),
            subtitle_line_spacing=_env_float("SUBTITLE_LINE_SPACING", 1.15),
            subtitle_bottom_margin_ratio=bottom_margin_ratio,
        )


def _report(on_progress: Optional[ProgressCallback], message: str, percent: int) -> None:
    if on_progress:
        on_progress(message, percent)


def extract_audio(video_path: str, audio_path: str) -> None:
    ffmpeg.input(video_path).output(
        audio_path,
        acodec="pcm_s16le",
        ac=1,
        ar="16k",
    ).run(quiet=True, overwrite_output=True)


def transcribe_to_srt(
    audio_path: str,
    srt_path: str,
    config: SubtitleConfig,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Transcribe audio to SRT using the backend appropriate for source_language."""
    if config.source_language == "vi":
        _transcribe_vi_to_srt(audio_path, srt_path, config, on_progress)
    else:
        _transcribe_en_to_srt(audio_path, srt_path, config, on_progress)


def _transcribe_en_to_srt(
    audio_path: str,
    srt_path: str,
    config: SubtitleConfig,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    transcribe_args = {"task": config.task}
    if config.model.endswith(".en"):
        transcribe_args["language"] = "en"
    elif config.language != "auto":
        transcribe_args["language"] = config.language

    _report(on_progress, "Loading Whisper model...", 15)
    with _WHISPER_LOCK:
        model = whisper.load_model(config.model)

        _report(on_progress, "Transcribing audio...", 20)
        warnings.filterwarnings("ignore")
        result = model.transcribe(audio_path, word_timestamps=True, **transcribe_args)
        warnings.filterwarnings("default")

    if not isinstance(result, dict):
        raise RuntimeError(
            "Whisper transcription returned an empty result. "
            "Retry the job; avoid running multiple transcriptions at once."
        )
    raw_segments = result.get("segments")
    if not isinstance(raw_segments, list):
        raise RuntimeError(
            "Whisper transcription returned no segments. "
            "Retry the job; avoid running multiple transcriptions at once."
        )

    segments = build_word_aligned_segments(raw_segments)
    with open(srt_path, "w", encoding="utf-8") as srt:
        write_srt(segments, file=srt)


def _transcribe_vi_to_srt(
    audio_path: str,
    srt_path: str,
    config: SubtitleConfig,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    from .phowhisper_transcribe import DEFAULT_VI_MODEL, transcribe_vi

    segments_raw = transcribe_vi(audio_path, model_name=DEFAULT_VI_MODEL, on_progress=on_progress)
    segments = build_word_aligned_segments(segments_raw)
    with open(srt_path, "w", encoding="utf-8") as srt:
        write_srt(segments, file=srt)


def _save_working_srt(path: str, entries: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        write_srt_entries(entries, file=f)


def _run_en_vi_pipeline(
    *,
    translation_source: str,
    source_entries: list,
    artifact_dir: str,
    work_dir: str,
    config: SubtitleConfig,
    on_progress: Optional[ProgressCallback],
    translation_context: Optional[dict],
    intel_ctx,
) -> str:
    """
    EN→VI subtitle pipeline with contract locks and explicit stage artifacts.

    Returns path to final working SRT (before burn).
    """
    from .config import (
        VI_COMPRESSION_ENABLED,
        VI_FLOW_ENABLED,
        translation_intelligence_enabled,
        vi_editor_enabled,
        vi_editor_save_debug,
    )
    from .pipeline_contract import (
        PipelineContractError,
        build_pipeline_contract_report,
        enforce_translation_contract,
        save_pipeline_contract_report,
        save_srt_entries,
        save_text_lock_violation,
        verify_post_final_repair_text_lock,
        count_stage_artifacts,
    )
    from .vi_compression import compress_vi_srt_file
    from .vi_editor import edit_vi_srt_file
    from .vi_flow import flow_vi_srt_file

    artifact = Path(artifact_dir)
    artifact.mkdir(parents=True, exist_ok=True)
    debug_dir = artifact / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    working_path = os.path.join(work_dir, "vi_working.srt")
    vi_raw_path = artifact / "vi_raw.srt"
    contract_errors: list[str] = []
    contract_warnings: list[str] = []
    alignment_applied = False
    translation_retry_applied = False
    missing_or_empty: list[int] = []

    # --- Translate → vi_raw (immutable) ---
    translate_srt_file(
        translation_source,
        str(vi_raw_path),
        config,
        on_progress,
        translation_context=translation_context,
    )
    with open(vi_raw_path, encoding="utf-8") as f:
        vi_raw_entries = parse_srt(f.read())

    from .config import get_raw_translation_mode, post_raw_overlap_guard_enabled

    if post_raw_overlap_guard_enabled():
        from .post_raw_overlap_guard import guard_post_raw_overlap

        vi_raw_entries, _overlap_report = guard_post_raw_overlap(
            source_entries,
            vi_raw_entries,
            topic=config.translation_topic,
            debug_dir=str(debug_dir),
        )
        save_srt_entries(vi_raw_path, vi_raw_entries)

    if get_raw_translation_mode() == "cue_keyed":
        from .raw_translation_alignment_guard import guard_and_repair_raw_translations

        vi_raw_entries, _guard_report = guard_and_repair_raw_translations(
            source_entries,
            vi_raw_entries,
            topic=config.translation_topic,
            debug_dir=str(debug_dir),
        )
        save_srt_entries(vi_raw_path, vi_raw_entries)
    # hybrid_guarded / span_guarded: guard+repair runs inside translate path

    def _retry_translate() -> list:
        nonlocal translation_retry_applied
        translation_retry_applied = True
        retry_path = os.path.join(work_dir, "vi_retry.srt")
        translate_srt_file(
            translation_source,
            retry_path,
            config,
            on_progress,
            translation_context=translation_context,
            strict_cue_count=True,
        )
        with open(retry_path, encoding="utf-8") as f:
            return parse_srt(f.read())

    try:
        working_entries, contract_meta = enforce_translation_contract(
            source_entries,
            vi_raw_entries,
            retry_translate=_retry_translate,
        )
        alignment_applied = contract_meta.get("alignment_applied", False)
        translation_retry_applied = contract_meta.get(
            "translation_retry_applied", translation_retry_applied
        )
        missing_or_empty = contract_meta.get("missing_or_empty_cue_errors", [])
    except PipelineContractError as exc:
        report = build_pipeline_contract_report(
            source_cue_count=len(source_entries),
            vi_raw_cue_count=len(vi_raw_entries),
            vi_raw_aligned_cue_count=None,
            alignment_applied=alignment_applied,
            translation_retry_applied=translation_retry_applied,
            stage_cue_counts=count_stage_artifacts(artifact),
            post_final_repair_text_changed=False,
            post_final_repair_text_lock_status="not_run",
            errors=[str(exc)],
            warnings=contract_warnings,
            missing_or_empty_cue_errors=missing_or_empty,
        )
        save_pipeline_contract_report(debug_dir / "pipeline_contract_report.json", report)
        raise

    vi_raw_aligned_path = artifact / "vi_raw_aligned.srt"
    if alignment_applied:
        save_srt_entries(vi_raw_aligned_path, working_entries)
    _save_working_srt(working_path, working_entries)

    # --- Editor ---
    if vi_editor_enabled():
        _report(on_progress, "Editing Vietnamese subtitles...", 62)
        vi_editor_path = artifact / "vi_after_editor.srt"
        edit_vi_srt_file(
            translation_source,
            working_path,
            str(vi_editor_path),
            translation_engine=config.translation_engine,
            topic=config.translation_topic,
            on_progress=on_progress,
            debug_dir=str(artifact) if vi_editor_save_debug() else None,
            translation_context=translation_context,
        )
        shutil.copy2(vi_editor_path, working_path)
    else:
        shutil.copy2(working_path, artifact / "vi_after_editor.srt")

    # --- Compression ---
    if VI_COMPRESSION_ENABLED:
        _report(on_progress, "Compressing Vietnamese subtitles...", 65)
        compress_vi_srt_file(working_path)
    shutil.copy2(working_path, artifact / "vi_after_compression.srt")

    # --- Flow ---
    if VI_FLOW_ENABLED:
        _report(on_progress, "Improving Vietnamese cue flow...", 67)
        flow_vi_srt_file(translation_source, working_path)
    shutil.copy2(working_path, artifact / "vi_after_flow.srt")

    # --- Readability (semantic; before final repair) ---
    from .subtitle_readability_optimizer import optimize_readability_file

    _report(on_progress, "Optimising subtitle readability...", 68)
    optimize_readability_file(
        working_path,
        before_artifact_path=str(artifact / "vi_before_readability.srt"),
        after_artifact_path=str(artifact / "vi_after_readability.srt"),
    )
    shutil.copy2(working_path, artifact / "vi_after_readability.srt")

    # --- Final semantic QA / repair (end of semantic pipeline) ---
    if intel_ctx is not None and translation_intelligence_enabled():
        from .translation_intelligence import run_post_translation_qa

        with open(working_path, encoding="utf-8") as f:
            vi_before_repair = parse_srt(f.read())
        repaired_entries, _qa_report = run_post_translation_qa(
            source_entries,
            vi_before_repair,
            intel_ctx,
            str(debug_dir),
        )
        _save_working_srt(working_path, repaired_entries)
        save_srt_entries(artifact / "vi_after_final_repair.srt", repaired_entries)
    else:
        shutil.copy2(working_path, artifact / "vi_after_final_repair.srt")

    with open(artifact / "vi_after_final_repair.srt", encoding="utf-8") as f:
        pre_timing_entries = parse_srt(f.read())
    pre_timing_hash_entries = list(pre_timing_entries)

    # --- Timing only (no text changes) ---
    from .subtitle_timing_optimizer import (
        normalize_final_srt_timing,
        optimize_srt_timing_file,
    )

    _report(on_progress, "Optimising subtitle timing...", 74)
    optimize_srt_timing_file(working_path)
    shutil.copy2(working_path, artifact / "vi_before_timing_normalize.srt")
    _report(on_progress, "Normalising subtitle timing...", 78)
    normalize_final_srt_timing(working_path)

    with open(working_path, encoding="utf-8") as f:
        post_timing_entries = parse_srt(f.read())

    lock_report = verify_post_final_repair_text_lock(
        pre_timing_hash_entries, post_timing_entries
    )
    if lock_report["post_final_repair_text_changed"]:
        save_text_lock_violation(
            debug_dir / "post_repair_text_lock_violation.json", lock_report
        )
        contract_errors.append("post_final_repair_text_lock_violation")

    shutil.copy2(working_path, artifact / "vi_after_timing.srt")
    shutil.copy2(working_path, artifact / "final_vi.srt")

    if intel_ctx is not None and translation_intelligence_enabled():
        qa_path = debug_dir / "translation_quality_report.json"
        if qa_path.exists():
            from .translation_intelligence import finalize_delivery_quality_report

            pre_report = json.loads(qa_path.read_text(encoding="utf-8"))
            finalize_delivery_quality_report(
                source_entries,
                pre_timing_hash_entries,
                post_timing_entries,
                intel_ctx,
                pre_report,
                str(debug_dir),
            )

    stage_counts = count_stage_artifacts(artifact)
    report = build_pipeline_contract_report(
        source_cue_count=len(source_entries),
        vi_raw_cue_count=len(vi_raw_entries),
        vi_raw_aligned_cue_count=(
            len(parse_srt(vi_raw_aligned_path.read_text(encoding="utf-8")))
            if vi_raw_aligned_path.exists()
            else None
        ),
        alignment_applied=alignment_applied,
        translation_retry_applied=translation_retry_applied,
        stage_cue_counts=stage_counts,
        post_final_repair_text_changed=lock_report["post_final_repair_text_changed"],
        post_final_repair_text_lock_status=lock_report[
            "post_final_repair_text_lock_status"
        ],
        errors=contract_errors,
        warnings=contract_warnings,
        missing_or_empty_cue_errors=missing_or_empty,
    )
    save_pipeline_contract_report(debug_dir / "pipeline_contract_report.json", report)
    return working_path


def translate_srt_file(
    srt_path: str,
    output_srt_path: str,
    config: SubtitleConfig,
    on_progress: Optional[ProgressCallback] = None,
    translation_context: Optional[dict] = None,
    *,
    strict_cue_count: bool = False,
) -> str:
    _report(on_progress, "Translating subtitles to Vietnamese...", 55)
    with open(srt_path, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    translated = translate_srt_entries(
        entries,
        target_lang=config.translate_to,
        engine=config.translation_engine,
        topic=config.translation_topic,
        translation_context=translation_context,
        strict_cue_count=strict_cue_count,
    )
    assert_vietnamese_translation_applied(entries, translated, target_lang=config.translate_to)

    with open(output_srt_path, "w", encoding="utf-8") as f:
        write_srt_entries(translated, file=f)

    return output_srt_path


def fix_vi_loanwords_file(
    srt_path: str,
    config: SubtitleConfig,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    from .vi_loanword_fix import fix_vi_srt_entries

    with open(srt_path, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    fixed = fix_vi_srt_entries(
        entries,
        use_openai=config.vi_loanword_openai,
        on_progress=on_progress,
    )

    with open(srt_path, "w", encoding="utf-8") as f:
        write_srt_entries(fixed, file=f)

    return srt_path


def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    config: SubtitleConfig,
    on_progress: Optional[ProgressCallback] = None,
    layout: Optional[dict] = None,
) -> str:
    from .subtitle_renderer import SubtitleRenderStyle, burn_subtitles as _render

    _report(on_progress, "Rendering video with subtitles...", 80)

    if layout:
        style = SubtitleRenderStyle.from_dict(layout)
        style.reference_height = config.subtitle_reference_height
        style.text_safe_padding_y = config.subtitle_text_safe_padding_y
        style.line_spacing = config.subtitle_line_spacing
        # Preserve configured legacy bottom margin unless layout explicitly sets it.
        if "bottom_margin_ratio" not in layout:
            style.bottom_margin_ratio = config.subtitle_bottom_margin_ratio
    else:
        style = SubtitleRenderStyle(
            mode=config.subtitle_style_mode,
            border_radius=config.subtitle_border_radius,
            padding_x=config.subtitle_padding_x,
            padding_y=config.subtitle_padding_y,
            text_safe_padding_y=config.subtitle_text_safe_padding_y,
            background_color=config.subtitle_background_color,
            background_opacity=config.subtitle_background_opacity,
            text_color=config.subtitle_font_color,
            font_size=config.subtitle_font_size,
            bottom_margin_ratio=config.subtitle_bottom_margin_ratio,
            max_width_ratio=config.subtitle_max_width_ratio,
            line_spacing=config.subtitle_line_spacing,
            reference_height=config.subtitle_reference_height,
        )

    _render(video_path, srt_path, output_path, style)
    _report(on_progress, "Done", 100)
    return output_path


def reburn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    layout: dict,
    config: Optional[SubtitleConfig] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    """Re-burn subtitles only (no transcribe/translate)."""
    config = config or SubtitleConfig.from_env()
    _report(on_progress, "Re-rendering subtitles...", 10)
    return burn_subtitles(
        video_path,
        srt_path,
        output_path,
        config,
        on_progress,
        layout=layout,
    )


def generate_vietsub(
    video_path: str,
    output_path: str,
    config: Optional[SubtitleConfig] = None,
    on_progress: Optional[ProgressCallback] = None,
    persist_srt_path: Optional[str] = None,
    persist_source_srt_path: Optional[str] = None,
    persist_layout_path: Optional[str] = None,
) -> str:
    """Full pipeline: transcribe → (translate if EN) → burn subtitles."""
    from .subtitle_renderer import layout_dict_from_config

    config = config or SubtitleConfig.from_env()
    work_dir = tempfile.mkdtemp(prefix="drakonsub_")

    _report(on_progress, "Extracting audio...", 5)
    audio_path = os.path.join(work_dir, "audio.wav")
    extract_audio(video_path, audio_path)

    srt_path = os.path.join(work_dir, "source.srt")
    transcribe_to_srt(audio_path, srt_path, config, on_progress)

    if config.source_language == "vi":
        if persist_source_srt_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(persist_source_srt_path)), exist_ok=True
            )
            shutil.copy2(srt_path, persist_source_srt_path)
        fix_vi_loanwords_file(srt_path, config, on_progress)
        final_srt = srt_path
    else:
        artifact_dir = (
            os.path.dirname(os.path.abspath(persist_srt_path))
            if persist_srt_path
            else work_dir
        )
        os.makedirs(artifact_dir, exist_ok=True)

        translation_source = srt_path
        from .config import (
            en_domain_correction_enabled,
            en_domain_correction_save_debug,
            translation_intelligence_enabled,
        )
        from .en_domain_corrector import correct_en_domain_srt_file

        shutil.copy2(srt_path, os.path.join(artifact_dir, "source.srt"))

        if en_domain_correction_enabled():
            _report(on_progress, "Correcting English domain terms...", 48)
            source_raw_path = os.path.join(artifact_dir, "source_raw.srt")
            source_corrected_path = os.path.join(artifact_dir, "source_corrected.srt")
            shutil.copy2(srt_path, source_raw_path)
            correct_en_domain_srt_file(
                srt_path,
                source_corrected_path,
                debug_dir=artifact_dir if en_domain_correction_save_debug() else None,
            )
            translation_source = source_corrected_path
        else:
            shutil.copy2(srt_path, os.path.join(artifact_dir, "source_raw.srt"))
            shutil.copy2(srt_path, os.path.join(artifact_dir, "source_corrected.srt"))

        if persist_source_srt_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(persist_source_srt_path)), exist_ok=True
            )
            shutil.copy2(translation_source, persist_source_srt_path)

        intel_ctx = None
        translation_context = None
        if translation_intelligence_enabled():
            from .translation_intelligence import run_pre_translation_intelligence

            with open(translation_source, encoding="utf-8") as f:
                source_entries_for_intel = parse_srt(f.read())
            intel_debug = str(Path(artifact_dir) / "debug")
            intel_ctx = run_pre_translation_intelligence(
                source_entries_for_intel,
                intel_debug,
                user_topic=config.translation_topic,
                engine=config.translation_engine,
            )
            translation_context = intel_ctx.to_dict()

        with open(translation_source, encoding="utf-8") as f:
            source_entries = parse_srt(f.read())

        final_srt = _run_en_vi_pipeline(
            translation_source=translation_source,
            source_entries=source_entries,
            artifact_dir=artifact_dir,
            work_dir=work_dir,
            config=config,
            on_progress=on_progress,
            translation_context=translation_context,
            intel_ctx=intel_ctx,
        )

    # Shorten verbose Vietnamese text to improve readability (in-place, opt-out via env).
    from .subtitle_readability_optimizer import optimize_readability_file

    if config.source_language == "vi":
        _report(on_progress, "Optimising subtitle readability...", 68)
        optimize_readability_file(final_srt)

        from .subtitle_timing_optimizer import (
            normalize_final_srt_timing,
            optimize_srt_timing_file,
        )
        _report(on_progress, "Optimising subtitle timing...", 74)
        optimize_srt_timing_file(final_srt)
        artifact_dir = (
            os.path.dirname(os.path.abspath(persist_srt_path))
            if persist_srt_path
            else work_dir
        )
        os.makedirs(artifact_dir, exist_ok=True)
        shutil.copy2(
            final_srt, os.path.join(artifact_dir, "vi_before_timing_normalize.srt")
        )
        _report(on_progress, "Normalising subtitle timing...", 78)
        normalize_final_srt_timing(final_srt)
        shutil.copy2(final_srt, os.path.join(artifact_dir, "vi_final.srt"))

    layout = layout_dict_from_config(config)

    if persist_srt_path:
        os.makedirs(os.path.dirname(os.path.abspath(persist_srt_path)), exist_ok=True)
        shutil.copy2(final_srt, persist_srt_path)

    if persist_layout_path:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(persist_layout_path)), exist_ok=True)
        with open(persist_layout_path, "w", encoding="utf-8") as fp:
            json.dump(layout, fp, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    return burn_subtitles(
        video_path, final_srt, output_path, config, on_progress, layout=layout
    )
