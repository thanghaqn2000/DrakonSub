import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import ffmpeg
import whisper
from dotenv import load_dotenv

from .utils import (
    write_srt,
    parse_srt,
    write_srt_entries,
    translate_srt_entries,
    prepare_burn_subtitles,
    build_word_aligned_segments,
)

load_dotenv()

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
    subtitle_margin_bottom: float = 32.0
    subtitle_font_size: int = 55
    subtitle_font_color: str = "#9333EA"
    subtitle_background_color: str = "#FFFFFF"
    subtitle_box_padding: int = 14
    subtitle_reference_height: int = 1920
    language: str = "auto"
    task: str = "transcribe"
    translation_topic: str = "economics"

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


def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    config: SubtitleConfig,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    _report(on_progress, "Rendering video with subtitles...", 80)
    ass_path = prepare_burn_subtitles(
        srt_path,
        video_path,
        config.subtitle_margin_bottom,
        config.subtitle_font_size,
        config.subtitle_font_color,
        config.subtitle_background_color,
        config.subtitle_box_padding,
        config.subtitle_reference_height,
    )

    video = ffmpeg.input(video_path)
    audio = video.audio
    ffmpeg.concat(
        video.filter("subtitles", ass_path),
        audio,
        v=1,
        a=1,
    ).output(output_path).run(quiet=True, overwrite_output=True)

    _report(on_progress, "Done", 100)
    return output_path


def generate_vietsub(
    video_path: str,
    output_path: str,
    config: Optional[SubtitleConfig] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    """Full pipeline: transcribe EN → translate VI → burn subtitles."""
    config = config or SubtitleConfig.from_env()
    work_dir = tempfile.mkdtemp(prefix="drakonsub_")

    _report(on_progress, "Extracting audio...", 5)
    audio_path = os.path.join(work_dir, "audio.wav")
    extract_audio(video_path, audio_path)

    en_srt = os.path.join(work_dir, "en.srt")
    transcribe_to_srt(audio_path, en_srt, config, on_progress)

    vi_srt = os.path.join(work_dir, "vi.srt")
    translate_srt_file(en_srt, vi_srt, config, on_progress)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    return burn_subtitles(video_path, vi_srt, output_path, config, on_progress)
