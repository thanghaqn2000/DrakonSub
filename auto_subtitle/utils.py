import os
import re
from typing import Iterator, List, TextIO


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


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]


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
) -> List[dict]:
    if engine == "openai":
        from .openai_translate import translate_srt_entries_openai

        return translate_srt_entries_openai(entries, target_lang=target_lang)

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
