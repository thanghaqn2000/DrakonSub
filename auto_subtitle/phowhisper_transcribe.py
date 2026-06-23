"""Vietnamese ASR using PhoWhisper (VinAI Research).

PhoWhisper is fine-tuned from Whisper on 844h of diverse Vietnamese audio.
It provides ~10x lower WER vs the generic multilingual Whisper-small for VI.

Model sizes (HuggingFace):
    vinai/PhoWhisper-small   — 244M params, fast, WER ~6.3% on VIVOS
    vinai/PhoWhisper-medium  — 769M params, balanced, WER ~5.0% on VIVOS
    vinai/PhoWhisper-large   — 1.55B params, highest accuracy, needs ~10GB VRAM
"""

import re
from typing import Callable, List, Optional

DEFAULT_VI_MODEL = "vinai/PhoWhisper-small"

# Whisper BPE emits token id 3197 as literal "unk" for undecodable audio.
WHISPER_UNK_TOKEN_ID = 3197
WHISPER_UNKNOWN_TOKEN_ID = 47774

# Matches standalone .unk / unk / UNKNOWN leaked from the tokenizer.
_PLACEHOLDER_TOKEN_RE = re.compile(
    r"(?<!\S)(?:\.unk|unk\.|unk|UNKNOWN)(?!\S)",
    re.IGNORECASE,
)

_pipeline = None
_pipeline_model_name = None

ProgressCallback = Callable[[str, int], None]


def _is_placeholder_token(text: str) -> bool:
    """True when a word-level chunk is a Whisper unknown-token artifact."""
    stripped = text.strip()
    if not stripped:
        return True
    normalized = stripped.strip(".,;:!?").lower()
    return normalized in {"unk", "unknown"}


def sanitize_vi_transcription_text(text: str) -> str:
    """Strip Whisper placeholder tokens from decoded transcription text."""
    cleaned = _PLACEHOLDER_TOKEN_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _generate_kwargs(pipe) -> dict:
    """Generation kwargs with Whisper unk placeholders suppressed."""
    suppressed = list(pipe.generation_config.suppress_tokens or [])
    for token_id in (WHISPER_UNK_TOKEN_ID, WHISPER_UNKNOWN_TOKEN_ID):
        if token_id not in suppressed:
            suppressed.append(token_id)
    return {
        "language": "vi",
        "task": "transcribe",
        "suppress_tokens": suppressed,
    }


def _pick_device() -> tuple:
    import torch

    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _get_pipeline(model_name: str, on_progress: Optional[ProgressCallback] = None):
    global _pipeline, _pipeline_model_name

    if _pipeline is not None and _pipeline_model_name == model_name:
        return _pipeline

    from transformers import pipeline as hf_pipeline

    if on_progress:
        on_progress("Loading PhoWhisper model (first run may download ~1GB)...", 15)

    device, dtype = _pick_device()
    try:
        _pipeline = hf_pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            dtype=dtype,
            device=device,
            model_kwargs={"attn_implementation": "eager"},
        )
    except Exception:
        if device != "cpu":
            if on_progress:
                on_progress("GPU memory full, falling back to CPU...", 15)
            _pipeline = hf_pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                dtype=dtype,
                device="cpu",
                model_kwargs={"attn_implementation": "eager"},
            )
        else:
            raise

    _pipeline_model_name = model_name
    return _pipeline


def _chunks_to_whisper_segments(chunks: List[dict], full_text: str) -> List[dict]:
    """Convert transformers pipeline word-chunk output to openai-whisper segment format."""
    if not chunks:
        return []

    words = []
    for chunk in chunks:
        start, end = chunk.get("timestamp") or (0.0, 0.0)
        if start is None:
            start = 0.0
        if end is None:
            end = start
        text = chunk["text"].strip()
        if not text or _is_placeholder_token(text):
            continue
        words.append(
            {
                "word": text,
                "start": float(start),
                "end": float(end),
            }
        )

    if not words:
        return []

    return [
        {
            "text": sanitize_vi_transcription_text(" ".join(w["word"] for w in words)),
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "words": words,
        }
    ]


def transcribe_vi(
    audio_path: str,
    model_name: str = DEFAULT_VI_MODEL,
    on_progress: Optional[ProgressCallback] = None,
) -> List[dict]:
    """Transcribe Vietnamese audio, returning segments compatible with
    build_word_aligned_segments().

    Uses Apple MPS / CUDA when available. Model is cached after first load.
    """
    device, _ = _pick_device()
    device_label = {"cuda:0": "GPU", "mps": "Apple GPU", "cpu": "CPU"}.get(device, device)

    pipe = _get_pipeline(model_name, on_progress)

    if on_progress:
        on_progress(
            f"Transcribing Vietnamese audio ({device_label}, có thể mất vài phút)...",
            25,
        )

    gen_kwargs = _generate_kwargs(pipe)

    try:
        result = pipe(
            audio_path,
            return_timestamps="word",
            generate_kwargs=gen_kwargs,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if on_progress:
            on_progress("GPU memory full, retrying on CPU...", 20)
        global _pipeline, _pipeline_model_name
        _pipeline = None
        _pipeline_model_name = None
        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            dtype=_pick_device()[1],
            device="cpu",
            model_kwargs={"attn_implementation": "eager"},
        )
        _pipeline = pipe
        _pipeline_model_name = model_name
        gen_kwargs = _generate_kwargs(pipe)
        result = pipe(
            audio_path,
            return_timestamps="word",
            generate_kwargs=gen_kwargs,
        )

    chunks = result.get("chunks", [])
    full_text = sanitize_vi_transcription_text(result.get("text", ""))

    if on_progress:
        on_progress("Transcription complete, preparing subtitles...", 70)

    return _chunks_to_whisper_segments(chunks, full_text)
