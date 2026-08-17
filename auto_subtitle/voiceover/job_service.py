from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .audio_builder import (
    build_manifest_summary,
    build_segment_manifests,
    build_voiceover_track,
    manifests_to_dicts,
    probe_audio_duration_ms,
    probe_video_duration_ms,
)
from .audio_mixer import mix_audio_tracks, mux_video_with_audio, video_has_audio_stream
from .saydi_tts import SaydiConfig, load_saydi_config, synthesize_to_file
from .srt_parser import VoiceoverSrtError, parse_voiceover_srt
from .text_preparer import (
    PrepareTextMode,
    prepare_voiceover_cues,
    prepared_to_voiceover_cues,
    resolve_prepare_text_mode,
    summarize_prepared_cues,
    write_prepared_srt,
)
from .timing_planner import cue_audio_budget_ms, plan_timing, suggest_saydi_speed_for_budget


def _saydi_config_with_speed(config: SaydiConfig | object, speed: float) -> SaydiConfig | object:
    if isinstance(config, SaydiConfig) or (is_dataclass(config) and not isinstance(config, type)):
        return replace(config, speed=speed)  # type: ignore[arg-type]
    return SimpleNamespace(
        api_url=getattr(config, "api_url", ""),
        token=getattr(config, "token", ""),
        sample=getattr(config, "sample", ""),
        speed=speed,
        output_format=getattr(config, "output_format", "wav"),
        timeout_seconds=getattr(config, "timeout_seconds", 120),
        lang=getattr(config, "lang", "vi"),
        model=getattr(config, "model", None),
    )


class VoiceoverJobError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceoverJobOptions:
    input_video: Path
    voiceover_srt: Path
    output_video: Path
    workdir: Path
    original_volume: float = 0.18
    voice_volume: float = 1.00
    prepare_text: bool = False
    prepare_text_mode: PrepareTextMode | None = None
    voiceover_topic: str = "catholic"
    max_chars_per_second: float = 13.0
    prepared_srt_output: Path | None = None
    min_gap_ms: int = 120
    max_borrow_after_ms: int = 1200
    severe_overflow_ms: int = 2000
    saydi_sample: str | None = None
    saydi_speed: float | None = None
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


def run_voiceover_job(
    options: VoiceoverJobOptions,
    *,
    progress_callback: Callable[[str, int], None] | None = None,
) -> VoiceoverJobResult:
    def _report(stage: str, percent: int) -> None:
        if progress_callback is not None:
            progress_callback(stage, percent)

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

    _report("starting", 5)

    prepared_cues = None
    tts_cues = cues
    prepared_srt_path = None
    prepare_mode = resolve_prepare_text_mode(
        prepare_text=options.prepare_text,
        prepare_text_mode=options.prepare_text_mode,
    )
    text_summary = {
        "text_ok_count": len(cues),
        "text_shortened_count": 0,
        "text_too_long_count": 0,
        "text_changed_count": 0,
        "total_original_chars": sum(len(cue.text) for cue in cues),
        "total_prepared_chars": sum(len(cue.text) for cue in cues),
        "average_reduction_ratio": 0.0,
    }

    if prepare_mode != "disabled":
        _report("preparing_text", 15)
        prepared_cues = prepare_voiceover_cues(
            cues,
            topic=options.voiceover_topic,
            max_chars_per_second=options.max_chars_per_second,
            mode=prepare_mode,
        )
        tts_cues = prepared_to_voiceover_cues(prepared_cues)
        text_summary = summarize_prepared_cues(prepared_cues)
        prepared_srt_path = options.prepared_srt_output or (workdir / "prepared_voiceover.srt")
        write_prepared_srt(prepared_cues, prepared_srt_path)

    try:
        video_duration_ms = probe_video_duration_ms(input_video)
        has_original_audio = video_has_audio_stream(input_video)
        saydi_config = load_saydi_config(
            sample_override=options.saydi_sample,
            speed_override=options.saydi_speed,
        )
        if not getattr(saydi_config, "token", ""):
            raise VoiceoverJobError("SAYDI_TTS_API_TOKEN is not configured")

        segments_dir = workdir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        _report("generating_voice", 35)

        segment_paths: list[Path] = []
        tts_durations_ms: list[int] = []
        saydi_speeds: list[float] = []
        warnings: list[str] = []
        base_speed = float(saydi_config.speed)

        for idx, cue in enumerate(tts_cues):
            segment_path = segments_dir / _segment_filename(cue.index)
            cue_config = saydi_config
            synthesize_to_file(cue.text, segment_path, config=cue_config)
            duration_ms = probe_audio_duration_ms(segment_path)

            is_last_cue = idx + 1 >= len(tts_cues)
            next_start_ms = tts_cues[idx + 1].start_ms if not is_last_cue else video_duration_ms
            budget_ms = cue_audio_budget_ms(
                cue,
                next_start_ms=next_start_ms,
                video_duration_ms=video_duration_ms,
                is_last_cue=is_last_cue,
                min_gap_ms=options.min_gap_ms,
                max_borrow_after_ms=options.max_borrow_after_ms,
            )
            suggested_speed = suggest_saydi_speed_for_budget(
                base_speed=base_speed,
                tts_duration_ms=duration_ms,
                budget_ms=budget_ms,
            )
            if suggested_speed > base_speed + 1e-9:
                cue_config = _saydi_config_with_speed(saydi_config, suggested_speed)
                synthesize_to_file(cue.text, segment_path, config=cue_config)
                duration_ms = probe_audio_duration_ms(segment_path)
                warnings.append(
                    f"cue_{cue.index}:raised_saydi_speed_to_{suggested_speed:.2f}"
                )

            segment_paths.append(segment_path)
            tts_durations_ms.append(duration_ms)
            saydi_speeds.append(float(getattr(cue_config, "speed", base_speed)))

        timing_plans = plan_timing(
            tts_cues,
            tts_durations_ms,
            video_duration_ms=video_duration_ms,
            min_gap_ms=options.min_gap_ms,
            max_borrow_after_ms=options.max_borrow_after_ms,
            severe_overflow_ms=options.severe_overflow_ms,
            resolve_overlaps=True,
        )
        manifests = build_segment_manifests(
            tts_cues,
            segment_paths,
            timing_plans,
            prepared_cues=prepared_cues,
            saydi_speeds=saydi_speeds,
        )
        summary = build_manifest_summary(timing_plans)
        summary.update(text_summary)
        summary["speed_raised_count"] = sum(
            1 for speed in saydi_speeds if speed > base_speed + 1e-9
        )
        summary["max_saydi_speed"] = max(saydi_speeds, default=base_speed)

        _report("mixing_audio", 80)

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
            "prepare_text_mode": prepare_mode,
            "voiceover_topic": options.voiceover_topic,
            "max_chars_per_second": options.max_chars_per_second,
            "min_gap_ms": options.min_gap_ms,
            "max_borrow_after_ms": options.max_borrow_after_ms,
            "severe_overflow_ms": options.severe_overflow_ms,
            "saydi_sample": saydi_config.sample,
            "saydi_speed": saydi_config.speed,
        },
        "tts_provider": "saydi",
        "saydi_sample": saydi_config.sample,
        "saydi_speed": saydi_config.speed,
        "tts_lang": saydi_config.lang,
        "tts_output_format": saydi_config.output_format,
        "video_duration_ms": video_duration_ms,
        "has_original_audio": has_original_audio,
        "source_srt_used_for_tts": str(voiceover_srt.resolve()),
        "text_preparation": {
            "enabled": prepare_mode != "disabled",
            "prepare_text_mode": prepare_mode,
            "topic": options.voiceover_topic,
            "max_chars_per_second": options.max_chars_per_second,
            "text_changed_count": text_summary.get("text_changed_count", 0),
            "text_shortened_count": text_summary.get("text_shortened_count", 0),
            "text_too_long_count": text_summary.get("text_too_long_count", 0),
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
        if segment.status in {"overflow_warning", "severe_overflow", "shifted_to_avoid_overlap"}:
            warnings.append(f"cue_{segment.index}:{segment.status}")
        if segment.shifted_ms > 0 and segment.status != "shifted_to_avoid_overlap":
            warnings.append(f"cue_{segment.index}:shifted_{segment.shifted_ms}ms")

    return VoiceoverJobResult(
        output_video=output_video,
        manifest_path=manifest_path,
        prepared_srt_path=prepared_srt_path,
        cue_count=len(cues),
        segment_count=len(manifests),
        summary=summary,
        warnings=warnings,
    )
