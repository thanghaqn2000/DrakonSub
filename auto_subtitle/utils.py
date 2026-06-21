import os
import re
import tempfile
from typing import Iterator, List, Optional, TextIO


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


def get_video_size(video_path: str) -> tuple[int, int]:
    import ffmpeg

    probe = ffmpeg.probe(video_path)
    video_stream = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    return int(video_stream["width"]), int(video_stream["height"])


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
        if len(lines) < 3:
            continue
        start_str, end_str = lines[1].split(" --> ")
        entries.append({
            "start_str": start_str.strip(),
            "end_str": end_str.strip(),
            "text": "\n".join(lines[2:]).strip(),
        })
    return entries


def write_srt_entries(entries: List[dict], file: TextIO):
    for i, entry in enumerate(entries, start=1):
        print(
            f"{i}\n"
            f"{entry['start_str']} --> {entry['end_str']}\n"
            f"{entry['text'].strip().replace('-->', '->')}\n",
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
    if engine == "openai":
        from .openai_translate import translate_srt_entries_openai

        return translate_srt_entries_openai(
            entries, target_lang=target_lang, topic=topic
        )

    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translated = []

    for entry in entries:
        text = entry["text"].strip()
        if not text:
            translated.append({**entry, "text": text})
            continue
        translated.append({**entry, "text": translator.translate(text)})

    return translated
