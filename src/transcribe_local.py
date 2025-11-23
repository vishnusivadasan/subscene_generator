"""
Local transcription module using faster-whisper.
Provides offline transcription without API costs.
"""

from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from utils import logger


# Global model cache to avoid reloading
_model_cache = {}


def get_model(model_size: str = "medium", device: str = "auto", compute_type: str = "auto"):
    """
    Get or create a cached Whisper model instance.

    Args:
        model_size: Model size (tiny, base, small, medium, large-v3)
        device: Device to use (auto, cuda, cpu)
        compute_type: Compute type (auto, float16, int8, etc.)

    Returns:
        WhisperModel instance
    """
    from faster_whisper import WhisperModel

    cache_key = f"{model_size}_{device}_{compute_type}"

    if cache_key not in _model_cache:
        # Auto-detect device if not specified
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {device}")

        # Auto-detect compute type based on device
        if compute_type == "auto":
            if device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"

        logger.info(f"Loading Whisper model: {model_size} (device={device}, compute_type={compute_type})")
        _model_cache[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        logger.info("Model loaded successfully")

    return _model_cache[cache_key]


def transcribe_audio_local(
    audio_path: str,
    model_size: str = "medium",
    device: str = "auto",
    language: str = "ja"
) -> Tuple[List[Dict[str, any]], None]:
    """
    Transcribe audio using local faster-whisper model.

    Args:
        audio_path: Path to the audio file (WAV)
        model_size: Model size (tiny, base, small, medium, large-v3)
        device: Device to use (auto, cuda, cpu)
        language: Source language code (default: ja for Japanese)

    Returns:
        Tuple of (list of transcription segments, None for compatibility)
    """
    model = get_model(model_size, device)

    logger.info(f"Transcribing with local Whisper ({model_size} model)...")

    # Transcribe with faster-whisper
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,  # Filter out silence
        vad_parameters=dict(
            min_silence_duration_ms=500,
        )
    )

    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Convert segments to our format
    all_segments = []

    # Wrap with tqdm for progress (faster-whisper provides duration info)
    for segment in tqdm(segments_iter, desc="Transcribing", unit="segment"):
        all_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    logger.info(f"Transcription complete. Total segments: {len(all_segments)}")

    # Return (segments, None) to match API function signature
    # The None is for chunks_info which isn't needed for local processing
    return all_segments, None


def transcribe_audio_local_translate(
    audio_path: str,
    model_size: str = "medium",
    device: str = "auto"
) -> Tuple[List[Dict[str, any]], None]:
    """
    Transcribe and translate audio to English using local faster-whisper model.

    Args:
        audio_path: Path to the audio file (WAV)
        model_size: Model size (tiny, base, small, medium, large-v3)
        device: Device to use (auto, cuda, cpu)

    Returns:
        Tuple of (list of translated segments in English, None for compatibility)
    """
    model = get_model(model_size, device)

    logger.info(f"Transcribing and translating with local Whisper ({model_size} model)...")

    # Transcribe with translation task
    segments_iter, info = model.transcribe(
        audio_path,
        task="translate",  # Translate to English
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        )
    )

    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Convert segments to our format
    all_segments = []

    for segment in tqdm(segments_iter, desc="Translating", unit="segment"):
        all_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    logger.info(f"Translation complete. Total segments: {len(all_segments)}")

    return all_segments, None
