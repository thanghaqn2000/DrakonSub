import os
import re
import tempfile
from typing import Dict, Iterator, List, Optional, TextIO


def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}

    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(
            f"Expected one of {set(str2val.keys())}, got {string}")


def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(transcript: Iterator[dict], file: TextIO):
    for i, segment in enumerate(transcript, start=1):
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True)} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True)}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def build_word_aligned_segments(
    segments: List[dict],
    max_chars: int = 42,
    max_duration: float = 3.5,
) -> List[dict]:
    """Split Whisper output into shorter cues aligned to spoken words."""
    words = []
    for segment in segments:
        segment_words = segment.get("words") or []
        if segment_words:
            words.extend(segment_words)
        elif segment.get("text", "").strip():
            words.append({
                "word": segment["text"].strip(),
                "start": segment["start"],
                "end": segment["end"],
            })

    if not words:
        return segments

    cues = []
    chunk_words: List[str] = []
    chunk_start = None
    chunk_end = None

    def flush_chunk() -> None:
        nonlocal chunk_words, chunk_start, chunk_end
        if not chunk_words or chunk_start is None or chunk_end is None:
            return
        cues.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": " ".join(chunk_words).strip(),
        })
        chunk_words = []
        chunk_start = None
        chunk_end = None

    for word_info in words:
        token = word_info.get("word", "").strip()
        if not token:
            continue

        if chunk_start is None:
            chunk_start = word_info["start"]
        chunk_end = word_info["end"]
        chunk_words.append(token)

        text = " ".join(chunk_words)
        duration = chunk_end - chunk_start
        ends_sentence = token.endswith((".", "?", "!"))

        if len(text) >= max_chars or duration >= max_duration or ends_sentence:
            flush_chunk()

    flush_chunk()
    return cues or segments


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def hex_color_to_ass(color: str) -> str:
    """Convert #RRGGBB (or RRGGBB) to ASS &HBBGGRR format."""
    value = color.strip().lstrip("#")
    if len(value) != 6 or not re.fullmatch(r"[0-9A-Fa-f]{6}", value):
        raise ValueError(f"Expected hex color like #9333EA, got {color!r}")

    red = value[0:2]
    green = value[2:4]
    blue = value[4:6]
    return f"&H00{blue.upper()}{green.upper()}{red.upper()}"


def _parse_rotation_degrees(value) -> Optional[int]:
    """Parse rotation metadata to integer degrees, or None if invalid."""
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _extract_video_rotation_degrees(video_stream: dict) -> int:
    """
    Read rotation (degrees) from common ffprobe locations.

    Checks stream.rotation, side_data_list[].rotation, and tags.rotate.
    Returns 0 when no rotation metadata is present.
    """
    candidates = []

    if "rotation" in video_stream:
        candidates.append(video_stream["rotation"])

    for side_data in video_stream.get("side_data_list") or []:
        if "rotation" in side_data:
            candidates.append(side_data["rotation"])

    tags = video_stream.get("tags") or {}
    if "rotate" in tags:
        candidates.append(tags["rotate"])

    for raw in candidates:
        parsed = _parse_rotation_degrees(raw)
        if parsed is not None:
            return parsed

    return 0


def _rotation_requires_dimension_swap(rotation_degrees: int) -> bool:
    """True when display orientation swaps stored width/height (90° or 270°)."""
    normalized = abs(int(rotation_degrees)) % 360
    return normalized in (90, 270)


def normalize_video_rotation_degrees(rotation_degrees: int) -> int:
    """Normalize rotation metadata to 0–359 degrees."""
    return int(rotation_degrees) % 360


def ffmpeg_rotation_normalize_filter(rotation_degrees: int) -> Optional[str]:
    """
    FFmpeg filter that bakes display orientation into pixel data.

    Returns None when no rotation correction is needed (0°).
    Used before subtitle overlay so layout coords match the frame FFmpeg sees.
    """
    norm = normalize_video_rotation_degrees(rotation_degrees)
    if norm == 0:
        return None
    if norm == 90:
        return "transpose=2"
    if norm == 180:
        return "hflip,vflip"
    if norm == 270:
        return "transpose=1"
    return None


def _visual_orientation(display_width: int, display_height: int) -> str:
    if display_height > display_width:
        return "portrait"
    if display_width > display_height:
        return "landscape"
    return "square"


def probe_video_orientation(video_path: str) -> dict:
    """
    Probe stored/display dimensions and visual orientation for a video file.

    visual_orientation is derived from display dimensions (what a player shows).
    """
    display_w, display_h, stored_w, stored_h, rotation = get_video_display_size(
        video_path
    )
    return {
        "stored_width": stored_w,
        "stored_height": stored_h,
        "rotation": rotation,
        "display_width": display_w,
        "display_height": display_h,
        "visual_orientation": _visual_orientation(display_w, display_h),
    }


def video_stream_has_rotation_metadata(video_path: str) -> bool:
    """True when the video stream still carries non-zero rotation metadata."""
    import ffmpeg

    probe = ffmpeg.probe(video_path)
    video_stream = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    if normalize_video_rotation_degrees(_extract_video_rotation_degrees(video_stream)) != 0:
        return True

    for side_data in video_stream.get("side_data_list") or []:
        rot = side_data.get("rotation")
        if rot is not None and normalize_video_rotation_degrees(rot) != 0:
            return True

    rotate_tag = (video_stream.get("tags") or {}).get("rotate")
    if rotate_tag is not None and str(rotate_tag).strip() not in ("", "0"):
        try:
            if normalize_video_rotation_degrees(int(rotate_tag)) != 0:
                return True
        except (TypeError, ValueError):
            return True

    return False


def ffmpeg_strip_rotation_metadata_args() -> List[str]:
    """FFmpeg output args that prevent copying rotation / Display Matrix metadata."""
    return [
        "-map_metadata",
        "-1",
        "-metadata:s:v:0",
        "rotate=0",
    ]


def ffmpeg_remux_strip_rotation(input_path: str, output_path: str) -> List[str]:
    """Re-encode pass that strips rotation metadata when remux copy is insufficient."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-map_metadata",
        "-1",
        "-metadata:s:v:0",
        "rotate=0",
        "-c:a",
        "copy",
        output_path,
    ]


def ffmpeg_video_encode_args() -> List[str]:
    """Explicit H.264 encode settings for subtitle burn output."""
    return [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]


def validate_output_orientation(input_path: str, output_path: str) -> None:
    """
    Ensure output matches input visual display size and orientation.

    Raises RuntimeError when display dimensions or orientation differ.
    """
    inp = probe_video_orientation(input_path)
    out = probe_video_orientation(output_path)
    if (
        inp["display_width"] != out["display_width"]
        or inp["display_height"] != out["display_height"]
    ):
        raise RuntimeError(
            "Output display dimensions "
            f"{out['display_width']}x{out['display_height']} do not match input "
            f"{inp['display_width']}x{inp['display_height']}"
        )
    if inp["visual_orientation"] != out["visual_orientation"]:
        raise RuntimeError(
            "Output visual orientation "
            f"{out['visual_orientation']} does not match input "
            f"{inp['visual_orientation']}"
        )


def log_video_orientation_probe(label: str, video_path: str) -> dict:
    """Probe a video and print a structured orientation log line."""
    info = probe_video_orientation(video_path)
    print(
        f"[Renderer] {label}: "
        f"stored={info['stored_width']}x{info['stored_height']} | "
        f"rotation={info['rotation']}° | "
        f"display={info['display_width']}x{info['display_height']} | "
        f"visual={info['visual_orientation']}"
    )
    return info


def get_video_display_size(video_path: str) -> tuple[int, int, int, int, int]:
    """
    Return video dimensions for layout/rendering (display-oriented).

    Tuple: (display_width, display_height, stored_width, stored_height, rotation)
    """
    import ffmpeg

    probe = ffmpeg.probe(video_path)
    video_stream = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    stored_w = int(video_stream["width"])
    stored_h = int(video_stream["height"])
    rotation = _extract_video_rotation_degrees(video_stream)

    if _rotation_requires_dimension_swap(rotation):
        return stored_h, stored_w, stored_w, stored_h, rotation
    return stored_w, stored_h, stored_w, stored_h, rotation


def get_video_size(video_path: str) -> tuple[int, int]:
    """Return (width, height) in display orientation (rotation-aware)."""
    display_w, display_h, _, _, _ = get_video_display_size(video_path)
    return display_w, display_h


def scale_subtitle_metric(value: int, video_size: int, reference_size: int) -> int:
    """Scale a pixel metric so it looks the same across video resolutions."""
    if reference_size <= 0:
        return value
    return max(1, round(value * video_size / reference_size))


def srt_timestamp_to_ass(timestamp: str) -> str:
    """Convert SRT timestamp (HH:MM:SS,mmm) to ASS (H:MM:SS.cc)."""
    time_part, millis = timestamp.strip().split(",")
    centiseconds = int(millis) // 10
    hours, minutes, seconds = time_part.split(":")
    return f"{int(hours)}:{minutes}:{seconds}.{centiseconds:02d}"


def escape_ass_text(text: str) -> str:
    """Escape ASS override characters in subtitle dialogue text."""
    text = text.strip()
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    return text.replace("\n", "\\N")


def write_ass_for_burn(
    entries: List[dict],
    ass_path: str,
    video_path: str,
    margin_bottom_percent: float = 32.0,
    font_size: int = 55,
    font_color: str = "#9333EA",
    background_color: str = "#FFFFFF",
    box_padding: int = 14,
    reference_height: int = 1920,
) -> None:
    """Write an ASS subtitle file matched to the video resolution."""
    width, height = get_video_size(video_path)
    margin_v = int(height * margin_bottom_percent / 100)
    margin_h = int(width * 0.05)
    scaled_font_size = scale_subtitle_metric(font_size, height, reference_height)
    scaled_box_padding = scale_subtitle_metric(box_padding, height, reference_height)
    text_colour = hex_color_to_ass(font_color)
    box_colour = hex_color_to_ass(background_color)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{scaled_font_size},{text_colour},&H000000FF,{box_colour},&H00000000,1,0,0,0,100,100,0,0,3,{scaled_box_padding},0,2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header.rstrip()]
    for entry in entries:
        start = srt_timestamp_to_ass(entry["start_str"])
        end = srt_timestamp_to_ass(entry["end_str"])
        text = escape_ass_text(entry["text"])
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def prepare_burn_subtitles(
    srt_path: str,
    video_path: str,
    margin_bottom_percent: float = 32.0,
    font_size: int = 55,
    font_color: str = "#9333EA",
    background_color: str = "#FFFFFF",
    box_padding: int = 14,
    reference_height: int = 1920,
) -> str:
    """Convert SRT to a temporary ASS file positioned for the target video."""
    with open(srt_path, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    fd, ass_path = tempfile.mkstemp(
        prefix=f"{filename(video_path)}.",
        suffix=".burn.ass",
        dir=tempfile.gettempdir(),
    )
    os.close(fd)
    write_ass_for_burn(
        entries,
        ass_path,
        video_path,
        margin_bottom_percent,
        font_size,
        font_color,
        background_color,
        box_padding,
        reference_height,
    )
    return ass_path


def parse_srt(content: str) -> List[dict]:
    entries = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        if len(lines) < 3:
            lines.append("")
        start_str, end_str = lines[1].split(" --> ")
        entries.append({
            "start_str": start_str.strip(),
            "end_str": end_str.strip(),
            "text": "\n".join(lines[2:]).strip(),
        })
    return entries


def write_srt_entries(entries: List[dict], file: TextIO):
    for i, entry in enumerate(entries, start=1):
        text = entry["text"].strip().replace("-->", "->")
        if not text:
            text = " "
        print(
            f"{i}\n"
            f"{entry['start_str']} --> {entry['end_str']}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )


def translate_srt_entries(
    entries: List[dict],
    target_lang: str,
    source_lang: str = "en",
    engine: str = "openai",
    topic: Optional[str] = None,
) -> List[dict]:
    engine = (engine or "openai").strip().lower()

    if engine == "openai":
        from .openai_translate import translate_srt_entries_openai

        return translate_srt_entries_openai(
            entries, target_lang=target_lang, topic=topic
        )
    if engine == "gemini":
        from .gemini_translate import translate_srt_entries_gemini

        return translate_srt_entries_gemini(
            entries, target_lang=target_lang, topic=topic
        )

    raise ValueError(f"Unsupported translation engine: {engine}")


def export_translation_ab_srt(
    srt_path: str,
    output_dir: str,
    target_lang: str = "vi",
    topic: Optional[str] = None,
) -> Dict[str, str]:
    """
    Translate one source SRT with both engines for quick quality A/B comparison.

    Output files:
    - vi_openai.srt
    - vi_gemini.srt
    """
    with open(srt_path, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    os.makedirs(output_dir, exist_ok=True)
    outputs: Dict[str, str] = {}
    for engine in ("openai", "gemini"):
        translated = translate_srt_entries(
            entries=entries,
            target_lang=target_lang,
            source_lang="en",
            engine=engine,
            topic=topic,
        )
        out_path = os.path.join(output_dir, f"{target_lang}_{engine}.srt")
        with open(out_path, "w", encoding="utf-8") as out_fp:
            write_srt_entries(translated, file=out_fp)
        outputs[engine] = out_path

    return outputs
