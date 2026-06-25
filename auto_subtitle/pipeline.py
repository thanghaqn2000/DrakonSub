import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import ffmpeg
import whisper
from .config import load_env

from .utils import (
    write_srt,
    parse_srt,
    write_srt_entries,
    translate_srt_entries,
    build_word_aligned_segments,
    str2bool,
)

load_env()

ProgressCallback = Callable[[str, int], None]


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
    subtitle_background_opacity: float = 0.92
    subtitle_max_width_ratio: float = 0.86
    subtitle_line_spacing: float = 1.15

    @classmethod
    def from_env(cls) -> "SubtitleConfig":
        from .translation_topics import DEFAULT_TOPIC, normalize_topic

        return cls(
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
            subtitle_background_opacity=_env_float("SUBTITLE_BACKGROUND_OPACITY", 0.92),
            subtitle_max_width_ratio=_env_float("SUBTITLE_MAX_WIDTH_RATIO", 0.86),
            subtitle_line_spacing=_env_float("SUBTITLE_LINE_SPACING", 1.15),
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
    model = whisper.load_model(config.model)

    _report(on_progress, "Transcribing audio...", 20)
    warnings.filterwarnings("ignore")
    result = model.transcribe(audio_path, word_timestamps=True, **transcribe_args)
    warnings.filterwarnings("default")

    segments = build_word_aligned_segments(result["segments"])
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


def translate_srt_file(
    srt_path: str,
    output_srt_path: str,
    config: SubtitleConfig,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    _report(on_progress, "Translating subtitles to Vietnamese...", 55)
    with open(srt_path, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    translated = translate_srt_entries(
        entries,
        target_lang=config.translate_to,
        engine=config.translation_engine,
        topic=config.translation_topic,
    )

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
) -> str:
    from .subtitle_renderer import SubtitleRenderStyle, burn_subtitles as _render

    _report(on_progress, "Rendering video with subtitles...", 80)

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
        bottom_margin_ratio=config.subtitle_margin_bottom / 100.0,
        max_width_ratio=config.subtitle_max_width_ratio,
        line_spacing=config.subtitle_line_spacing,
        reference_height=config.subtitle_reference_height,
    )

    _render(video_path, srt_path, output_path, style)
    _report(on_progress, "Done", 100)
    return output_path


def generate_vietsub(
    video_path: str,
    output_path: str,
    config: Optional[SubtitleConfig] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    """Full pipeline: transcribe → (translate if EN) → burn subtitles."""
    config = config or SubtitleConfig.from_env()
    work_dir = tempfile.mkdtemp(prefix="drakonsub_")

    _report(on_progress, "Extracting audio...", 5)
    audio_path = os.path.join(work_dir, "audio.wav")
    extract_audio(video_path, audio_path)

    srt_path = os.path.join(work_dir, "source.srt")
    transcribe_to_srt(audio_path, srt_path, config, on_progress)

    if config.source_language == "vi":
        fix_vi_loanwords_file(srt_path, config, on_progress)
        final_srt = srt_path
    else:
        final_srt = os.path.join(work_dir, "vi.srt")
        translate_srt_file(srt_path, final_srt, config, on_progress)

    # Shorten verbose Vietnamese text to improve readability (in-place, opt-out via env).
    from .subtitle_readability_optimizer import optimize_readability_file
    _report(on_progress, "Optimising subtitle readability...", 68)
    optimize_readability_file(final_srt)

    # Adjust cue timing for comfortable Vietnamese reading (in-place, opt-out via env).
    from .subtitle_timing_optimizer import optimize_srt_timing_file
    _report(on_progress, "Optimising subtitle timing...", 74)
    optimize_srt_timing_file(final_srt)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    return burn_subtitles(video_path, final_srt, output_path, config, on_progress)
