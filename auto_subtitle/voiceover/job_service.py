from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .audio_builder import (
    build_manifest_summary,
    build_segment_manifests,
    build_voiceover_track,
    manifests_to_dicts,
    probe_audio_duration_ms,
    probe_video_duration_ms,
)
from .audio_mixer import mix_audio_tracks, mux_video_with_audio, video_has_audio_stream
from .saydi_tts import load_saydi_config, synthesize_to_file
from .srt_parser import VoiceoverSrtError, parse_voiceover_srt
from .text_preparer import (
    prepare_voiceover_cues,
    prepared_to_voiceover_cues,
    summarize_prepared_cues,
    write_prepared_srt,
)
from .timing_planner import plan_timing


class VoiceoverJobError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceoverJobOptions:
    input_video: Path
    voiceover_srt: Path
    output_video: Path
    workdir: Path
    original_volume: float = 0.30
    voice_volume: float = 1.00
    prepare_text: bool = False
    voiceover_topic: str = "catholic"
    max_chars_per_second: float = 13.0
    prepared_srt_output: Path | None = None
    min_gap_ms: int = 120
    max_borrow_after_ms: int = 1200
    severe_overflow_ms: int = 2000
    force: bool = False


@dataclass(frozen=True)
class VoiceoverJobResult:
    output_video: Path
    manifest_path: Path
    prepared_srt_path: Path | None
    cue_count: int
    segment_count: int
    summary: dict
    warnings: list[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _segment_filename(index: int) -> str:
    return f"{index:04d}.wav"


def run_voiceover_job(options: VoiceoverJobOptions) -> VoiceoverJobResult:
    input_video = Path(options.input_video)
    voiceover_srt = Path(options.voiceover_srt)
    output_video = Path(options.output_video)
    workdir = Path(options.workdir)

    if not input_video.exists():
        raise VoiceoverJobError(f"Input video not found: {input_video}")
    if not voiceover_srt.exists():
        raise VoiceoverJobError(f"Voiceover SRT not found: {voiceover_srt}")
    if output_video.exists() and not options.force:
        raise VoiceoverJobError(f"Output already exists: {output_video}")

    workdir.mkdir(parents=True, exist_ok=True)

    try:
        cues = parse_voiceover_srt(voiceover_srt)
    except VoiceoverSrtError as exc:
        raise VoiceoverJobError(str(exc)) from exc
    if not cues:
        raise VoiceoverJobError(f"No cues found in {voiceover_srt}")

    prepared_cues = None
    tts_cues = cues
    prepared_srt_path = None
    text_summary = {
        "text_ok_count": len(cues),
        "text_shortened_count": 0,
        "text_too_long_count": 0,
        "total_original_chars": sum(len(cue.text) for cue in cues),
        "total_prepared_chars": sum(len(cue.text) for cue in cues),
        "average_reduction_ratio": 0.0,
    }

    if options.prepare_text:
        prepared_cues = prepare_voiceover_cues(
            cues,
            topic=options.voiceover_topic,
            max_chars_per_second=options.max_chars_per_second,
        )
        tts_cues = prepared_to_voiceover_cues(prepared_cues)
        text_summary = summarize_prepared_cues(prepared_cues)
        prepared_srt_path = options.prepared_srt_output or (workdir / "prepared_voiceover.srt")
        write_prepared_srt(prepared_cues, prepared_srt_path)

    try:
        video_duration_ms = probe_video_duration_ms(input_video)
        has_original_audio = video_has_audio_stream(input_video)
        saydi_config = load_saydi_config()
        if not getattr(saydi_config, "token", ""):
            raise VoiceoverJobError("SAYDI_TTS_API_TOKEN is not configured")

        segments_dir = workdir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        segment_paths: list[Path] = []
        tts_durations_ms: list[int] = []
        warnings: list[str] = []

        for cue in tts_cues:
            segment_path = segments_dir / _segment_filename(cue.index)
            synthesize_to_file(cue.text, segment_path, config=saydi_config)
            segment_paths.append(segment_path)
            tts_durations_ms.append(probe_audio_duration_ms(segment_path))

        timing_plans = plan_timing(
            tts_cues,
            tts_durations_ms,
            video_duration_ms=video_duration_ms,
            min_gap_ms=options.min_gap_ms,
            max_borrow_after_ms=options.max_borrow_after_ms,
            severe_overflow_ms=options.severe_overflow_ms,
        )
        manifests = build_segment_manifests(
            tts_cues,
            segment_paths,
            timing_plans,
            prepared_cues=prepared_cues,
        )
        summary = build_manifest_summary(timing_plans)
        summary.update(text_summary)

        voiceover_track = workdir / "voiceover_track.wav"
        build_voiceover_track(
            manifests=manifests,
            video_duration_ms=video_duration_ms,
            output_path=voiceover_track,
        )

        mixed_audio = workdir / "mixed_audio.wav"
        mix_audio_tracks(
            original_audio_source=input_video,
            voiceover_track=voiceover_track,
            output_path=mixed_audio,
            original_volume=options.original_volume,
            voice_volume=options.voice_volume,
            has_original_audio=has_original_audio,
        )

        mux_video_with_audio(
            input_video=input_video,
            mixed_audio=mixed_audio,
            output_video=output_video,
        )
    except VoiceoverJobError:
        raise
    except Exception as exc:
        message = str(exc)
        if "SAYDI_TTS_API_TOKEN" in message:
            raise VoiceoverJobError("SAYDI_TTS_API_TOKEN is not configured") from exc
        raise VoiceoverJobError(message) from exc

    manifest = {
        "job_type": "voiceover",
        "version": 1,
        "created_at": _utc_now(),
        "input_video": str(input_video.resolve()),
        "voiceover_srt": str(voiceover_srt.resolve()),
        "prepared_srt": str(prepared_srt_path.resolve()) if prepared_srt_path else None,
        "output_video": str(output_video.resolve()),
        "options": {
            "original_volume": options.original_volume,
            "voice_volume": options.voice_volume,
            "prepare_text": options.prepare_text,
            "voiceover_topic": options.voiceover_topic,
            "max_chars_per_second": options.max_chars_per_second,
            "min_gap_ms": options.min_gap_ms,
            "max_borrow_after_ms": options.max_borrow_after_ms,
            "severe_overflow_ms": options.severe_overflow_ms,
        },
        "video_duration_ms": video_duration_ms,
        "has_original_audio": has_original_audio,
        "text_preparation": {
            "enabled": options.prepare_text,
            "topic": options.voiceover_topic,
            "max_chars_per_second": options.max_chars_per_second,
            **({"prepared_srt_output": str(prepared_srt_path.resolve())} if prepared_srt_path else {}),
        },
        "timing_config": {
            "min_gap_ms": options.min_gap_ms,
            "max_borrow_after_ms": options.max_borrow_after_ms,
            "severe_overflow_ms": options.severe_overflow_ms,
        },
        "segments": manifests_to_dicts(manifests),
        "summary": summary,
        "artifacts": {
            "segments_dir": str((workdir / "segments").resolve()),
            "voiceover_track": str((workdir / "voiceover_track.wav").resolve()),
            "mixed_audio": str((workdir / "mixed_audio.wav").resolve()),
        },
        "overflow_warnings": summary["overflow_warning_count"] + summary["severe_overflow_count"],
    }

    manifest_path = workdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for segment in manifests:
        if segment.status in {"overflow_warning", "severe_overflow"}:
            warnings.append(f"cue_{segment.index}:{segment.status}")

    return VoiceoverJobResult(
        output_video=output_video,
        manifest_path=manifest_path,
        prepared_srt_path=prepared_srt_path,
        cue_count=len(cues),
        segment_count=len(manifests),
        summary=summary,
        warnings=warnings,
    )
