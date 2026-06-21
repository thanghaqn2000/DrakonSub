import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from dotenv import load_dotenv
from .utils import (
    filename,
    str2bool,
    write_srt,
    parse_srt,
    write_srt_entries,
    translate_srt_entries,
    prepare_burn_subtitles,
    build_word_aligned_segments,
)

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name) or default


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--model", default="small",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--output_srt", type=str2bool, default=False,
                        help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
    help="What is the origin language of the video? If unset, it is detected automatically.")
    parser.add_argument("--from_srt", type=str, default=None,
                        help="use an existing .srt file instead of transcribing")
    parser.add_argument("--translate_to", type=str, default=None,
                        help="translate each subtitle segment to this language code (e.g. vi)")
    parser.add_argument("--translation_engine", type=str, default="openai",
                        choices=["openai", "google"],
                        help="translation backend: openai (natural Vietnamese) or google")
    parser.add_argument("--subtitle_margin_bottom", type=float,
                        default=_env_float("SUBTITLE_MARGIN_BOTTOM", 32.0),
                        help="subtitle distance from bottom edge, as %% of video height (higher = further up)")
    parser.add_argument("--subtitle_font_size", type=int,
                        default=_env_int("SUBTITLE_FONT_SIZE", 55),
                        help="subtitle font size in pixels (env: SUBTITLE_FONT_SIZE)")
    parser.add_argument("--subtitle_font_color", type=str,
                        default=_env_str("SUBTITLE_FONT_COLOR", "#9333EA"),
                        help="subtitle text color as hex, e.g. #9333EA (env: SUBTITLE_FONT_COLOR)")
    parser.add_argument("--subtitle_background_color", type=str,
                        default=_env_str("SUBTITLE_BACKGROUND_COLOR", "#FFFFFF"),
                        help="subtitle background color as hex, e.g. #FFFFFF (env: SUBTITLE_BACKGROUND_COLOR)")
    parser.add_argument("--subtitle_box_padding", type=int,
                        default=_env_int("SUBTITLE_BOX_PADDING", 14),
                        help="padding inside subtitle background box in pixels (env: SUBTITLE_BOX_PADDING)")
    parser.add_argument("--subtitle_reference_height", type=int,
                        default=_env_int("SUBTITLE_REFERENCE_HEIGHT", 1920),
                        help="reference video height for font scaling, e.g. 1920 for vertical reels (env: SUBTITLE_REFERENCE_HEIGHT)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="extra suffix before extension, e.g. '_new' -> video.vi_new.mp4")
    parser.add_argument("--output_name", type=str, default=None,
                        help="custom output video basename without extension, e.g. vid-speech-vietsub")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    from_srt: str = args.pop("from_srt")
    translate_to: str = args.pop("translate_to")
    translation_engine: str = args.pop("translation_engine")
    subtitle_margin_bottom: float = args.pop("subtitle_margin_bottom")
    subtitle_font_size: int = args.pop("subtitle_font_size")
    subtitle_font_color: str = args.pop("subtitle_font_color")
    subtitle_background_color: str = args.pop("subtitle_background_color")
    subtitle_box_padding: int = args.pop("subtitle_box_padding")
    subtitle_reference_height: int = args.pop("subtitle_reference_height")
    output_suffix: str = args.pop("output_suffix")
    output_name: str = args.pop("output_name")

    os.makedirs(output_dir, exist_ok=True)

    videos = args.pop("video")

    if from_srt:
        subtitles = apply_translation(
            from_srt, videos, output_dir, translate_to, translation_engine, output_suffix
        )
    else:
        if model_name.endswith(".en"):
            warnings.warn(
                f"{model_name} is an English-only model, forcing English detection.")
            args["language"] = "en"
        elif language != "auto":
            args["language"] = language

        model = whisper.load_model(model_name)
        audios = get_audio(videos)
        subtitles = get_subtitles(
            audios,
            output_srt or srt_only,
            output_dir,
            lambda audio_path: model.transcribe(
                audio_path, word_timestamps=True, **args
            ),
        )
        subtitles = {
            path: maybe_translate_srt(
                srt_path,
                translate_to,
                output_dir,
                path,
                translation_engine,
                output_suffix,
                output_name,
                output_srt,
            )
            for path, srt_path in subtitles.items()
        }

    if srt_only:
        return

    for path, srt_path in subtitles.items():
        if output_name:
            out_path = os.path.join(output_dir, f"{output_name}.mp4")
        else:
            lang_suffix = f".{translate_to}" if translate_to else ""
            name_suffix = f"{lang_suffix}{output_suffix}" if output_suffix else lang_suffix
            out_path = os.path.join(output_dir, f"{filename(path)}{name_suffix}.mp4")

        print(f"Adding subtitles to {filename(path)}...")

        video = ffmpeg.input(path)
        audio = video.audio
        ass_path = prepare_burn_subtitles(
            srt_path,
            path,
            subtitle_margin_bottom,
            subtitle_font_size,
            subtitle_font_color,
            subtitle_background_color,
            subtitle_box_padding,
            subtitle_reference_height,
        )

        ffmpeg.concat(
            video.filter("subtitles", ass_path), audio, v=1, a=1
        ).output(out_path).run(quiet=True, overwrite_output=True)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")


def get_audio(paths):
    temp_dir = tempfile.gettempdir()

    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        ffmpeg.input(path).output(
            output_path,
            acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True, overwrite_output=True)

        audio_paths[path] = output_path

    return audio_paths


def get_subtitles(audio_paths: list, output_srt: bool, output_dir: str, transcribe: callable):
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")
        
        print(
            f"Generating subtitles for {filename(path)}... This might take a while."
        )

        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        segments = build_word_aligned_segments(result["segments"])

        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(segments, file=srt)

        subtitles_path[path] = srt_path

    return subtitles_path


def maybe_translate_srt(
    srt_path: str,
    translate_to: str,
    output_dir: str,
    video_path: str,
    translation_engine: str = "openai",
    output_suffix: str = "",
    output_name: str = None,
    output_srt: bool = False,
) -> str:
    if not translate_to:
        return srt_path

    engine_label = "OpenAI" if translation_engine == "openai" else "Google Translate"
    print(f"Translating subtitles to {translate_to} via {engine_label}...")
    with open(srt_path, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    translated = translate_srt_entries(
        entries, target_lang=translate_to, engine=translation_engine
    )

    srt_dir = output_dir if output_srt else tempfile.gettempdir()
    if output_name:
        out_path = os.path.join(srt_dir, f"{output_name}.srt")
    else:
        name_suffix = f".{translate_to}{output_suffix}"
        out_path = os.path.join(srt_dir, f"{filename(video_path)}{name_suffix}.srt")

    with open(out_path, "w", encoding="utf-8") as f:
        write_srt_entries(translated, file=f)

    return out_path


def apply_translation(
    from_srt: str,
    videos: list,
    output_dir: str,
    translate_to: str,
    translation_engine: str = "openai",
    output_suffix: str = "",
) -> dict:
    engine_label = "OpenAI" if translation_engine == "openai" else "Google Translate"
    print(f"Translating subtitles to {translate_to} via {engine_label}...")
    with open(from_srt, encoding="utf-8") as f:
        entries = parse_srt(f.read())

    translated = translate_srt_entries(
        entries, target_lang=translate_to, engine=translation_engine
    )
    subtitles = {}
    name_suffix = f".{translate_to}{output_suffix}"

    for path in videos:
        out_path = os.path.join(output_dir, f"{filename(path)}{name_suffix}.srt")
        with open(out_path, "w", encoding="utf-8") as f:
            write_srt_entries(translated, file=f)
        subtitles[path] = out_path

    return subtitles


def simple_main():
    import sys

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: auto_sub <video_path> [options]")
        print("Example: auto_sub /path/to/vid-speech.mp4")
        print("Output:  /path/to/vid-speech-vietsub.mp4")
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 1)

    video_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(video_path):
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    output_dir = os.path.dirname(video_path) or "."
    output_name = f"{filename(video_path)}-vietsub"

    sys.argv = [
        "auto_sub",
        video_path,
        "--translate_to",
        "vi",
        "--output_srt",
        "false",
        "-o",
        output_dir,
        "--output_name",
        output_name,
        *sys.argv[2:],
    ]
    main()


if __name__ == '__main__':
    main()
