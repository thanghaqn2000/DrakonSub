"""Generate voiceover SRT from source video (extract → transcribe → translate)."""

from pathlib import Path
from typing import Callable, Optional

from ..pipeline import SubtitleConfig, extract_audio, transcribe_to_srt, translate_srt_file
from ..translation_topics import normalize_topic
from ..utils import parse_srt, write_srt_entries
from .job_service import VoiceoverJobError
from .srt_quality import (
    group_source_cues_for_voiceover,
    optimize_voiceover_srt_entries,
    voiceover_narration_translation_context,
    write_voiceover_srt_entries,
)

ProgressCallback = Callable[[str, int], None]


def build_voiceover_subtitle_config(voiceover_topic: str) -> SubtitleConfig:
    """SubtitleConfig for standalone transcribe/translate (not full Vietsub burn)."""
    config = SubtitleConfig.from_env()
    config.translation_topic = normalize_topic(voiceover_topic)
    config.source_language = "en"
    return config


def _group_source_srt_for_voiceover(source_srt: Path) -> None:
    """Rewrite source.srt with phrase-level cue grouping before translation."""
    entries = parse_srt(source_srt.read_text(encoding="utf-8"))
    grouped = group_source_cues_for_voiceover(entries)
    with open(source_srt, "w", encoding="utf-8") as handle:
        write_srt_entries(grouped, file=handle)


def _optimize_voiceover_srt(voiceover_srt: Path) -> None:
    """Compact, stretch CPS, drop empty cues, and reindex final voiceover.srt."""
    entries = parse_srt(voiceover_srt.read_text(encoding="utf-8"))
    optimized = optimize_voiceover_srt_entries(entries)
    write_voiceover_srt_entries(optimized, voiceover_srt)


def prepare_voiceover_srt_from_video(
    input_video: Path,
    job_dir: Path,
    voiceover_topic: str,
    on_progress: Optional[ProgressCallback] = None,
) -> tuple[Path, Path, Path]:
    """Extract audio, transcribe to source.srt, translate to voiceover.srt."""
    if not input_video.is_file():
        raise VoiceoverJobError(f"Input video not found: {input_video}")

    audio_path = job_dir / "audio.wav"
    source_srt = job_dir / "source.srt"
    voiceover_srt = job_dir / "voiceover.srt"
    config = build_voiceover_subtitle_config(voiceover_topic)

    def _report(stage: str, percent: int) -> None:
        if on_progress:
            on_progress(stage, percent)

    try:
        _report("extracting_audio", 10)
        extract_audio(str(input_video), str(audio_path))

        _report("transcribing", 25)
        transcribe_to_srt(str(audio_path), str(source_srt), config)

        _report("grouping_source_cues", 35)
        _group_source_srt_for_voiceover(source_srt)

        _report("translating_voiceover", 45)
        translate_srt_file(
            str(source_srt),
            str(voiceover_srt),
            config,
            translation_context=voiceover_narration_translation_context(),
        )

        _report("optimizing_voiceover_srt", 48)
        _optimize_voiceover_srt(voiceover_srt)
    except VoiceoverJobError:
        raise
    except Exception as exc:
        message = str(exc)
        if "transcrib" in message.lower():
            raise VoiceoverJobError("Không thể nhận diện lời nói trong video.") from exc
        if "translat" in message.lower():
            raise VoiceoverJobError("Không thể dịch phụ đề sang tiếng Việt.") from exc
        raise VoiceoverJobError(message) from exc

    if not voiceover_srt.is_file():
        raise VoiceoverJobError("Không tạo được file SRT thuyết minh.")

    return audio_path, source_srt, voiceover_srt


# Map run_voiceover_job internal stages to later progress range for video-only flow.
VOICEOVER_TTS_STAGE_FROM_VIDEO = {
    "starting": ("starting", 50),
    "preparing_text": ("preparing_text", 60),
    "generating_voice": ("generating_voice", 70),
    "mixing_audio": ("mixing_audio", 85),
}
