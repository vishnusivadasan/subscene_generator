"""
Local transcription module using faster-whisper.
Provides offline transcription without API costs.
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from faster_whisper.vad import VadOptions
from faster_whisper.audio import decode_audio
from utils import logger


# Global model cache to avoid reloading
_model_cache = {}


class LanguageDetectionError(Exception):
    """Raised when detected language doesn't match expected language."""
    pass


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


def detect_language(
    audio_path: str,
    model_size: str = "medium",
    device: str = "auto",
    min_confidence: float = 0.7,
) -> Tuple[str, float]:
    """
    Detect the language of an audio file using VAD to find speech segments.

    Args:
        audio_path: Path to the audio file
        model_size: Model size (tiny, base, small, medium, large-v3)
        device: Device to use (auto, cuda, cpu)
        min_confidence: Minimum confidence threshold for detection

    Returns:
        Tuple of (language_code, probability)

    Raises:
        LanguageDetectionError: If no speech is detected in the audio
    """
    model = get_model(model_size, device)

    logger.info("Detecting language using VAD-based speech detection...")

    # Load audio as numpy array (16kHz mono)
    audio = decode_audio(audio_path, sampling_rate=16000)

    # Use VAD filter and multiple segments for robust detection
    # language_detection_segments=4 samples from different parts of the audio
    vad_options = VadOptions(
        threshold=0.5,
        min_silence_duration_ms=500,
    )
    language, probability, all_probs = model.detect_language(
        audio,
        vad_filter=True,
        vad_parameters=vad_options,
        language_detection_segments=4,  # Sample from multiple segments
        language_detection_threshold=0.5,
    )

    # Check if we got a valid detection
    if probability < 0.1:
        raise LanguageDetectionError(
            "Could not detect language - no speech found in audio. "
            "The audio may contain only music, silence, or sound effects."
        )

    logger.info(f"Detected language: {language} (confidence: {probability:.1%})")

    # Log top 3 language candidates for debugging
    top_langs = sorted(all_probs, key=lambda x: x[1], reverse=True)[:3]
    logger.debug(f"Top language candidates: {top_langs}")

    return language, probability


def transcribe_audio_local(
    audio_path: str,
    model_size: str = "medium",
    device: str = "auto",
    language: str = "ja",
    beam_size: int = 5,
    skip_language_check: bool = False,
    min_language_confidence: float = 0.7,
) -> Tuple[List[Dict[str, any]], None]:
    """
    Transcribe audio using local faster-whisper model.

    Args:
        audio_path: Path to the audio file (WAV)
        model_size: Model size (tiny, base, small, medium, large-v3)
        device: Device to use (auto, cuda, cpu)
        language: Source language code (default: ja for Japanese)
        beam_size: Beam size for decoding
        skip_language_check: Skip language detection check before transcription
        min_language_confidence: Minimum confidence for language detection

    Returns:
        Tuple of (list of transcription segments, None for compatibility)

    Raises:
        LanguageDetectionError: If detected language doesn't match expected language
    """
    model = get_model(model_size, device)

    # Check language before transcribing (unless skipped)
    if not skip_language_check:
        detected_lang, confidence = detect_language(
            audio_path, model_size, device, min_language_confidence
        )

        if detected_lang != language:
            raise LanguageDetectionError(
                f"Detected language '{detected_lang}' ({confidence:.1%} confidence) "
                f"does not match expected language '{language}'. Aborting transcription. "
                f"Use --skip-language-check to bypass this check."
            )

        logger.info(f"Language check passed: {detected_lang} ({confidence:.1%} confidence)")

    logger.info(f"Transcribing with local Whisper ({model_size} model)...")

    # Transcribe with faster-whisper
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=True,  # Filter out silence
        vad_parameters=dict(
            min_silence_duration_ms=500,
        )
    )

    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Convert segments to our format
    all_segments = []

    # Duration-based progress bar (shows percentage and time remaining)
    total_duration = info.duration
    bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]'

    with tqdm(total=total_duration, desc="Transcribing", unit="s", bar_format=bar_format) as pbar:
        last_end = 0.0
        for segment in segments_iter:
            all_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            # Update progress based on segment end time
            pbar.update(segment.end - last_end)
            last_end = segment.end

        # Ensure progress bar reaches 100%
        if last_end < total_duration:
            pbar.update(total_duration - last_end)

    logger.info(f"Transcription complete. Total segments: {len(all_segments)}")

    # Return (segments, None) to match API function signature
    # The None is for chunks_info which isn't needed for local processing
    return all_segments, None


def transcribe_audio_local_translate(
    audio_path: str,
    model_size: str = "medium",
    device: str = "auto",
    beam_size: int = 5,
    skip_language_check: bool = False,
    min_language_confidence: float = 0.7,
    expected_language: str = "ja",
) -> Tuple[List[Dict[str, any]], None]:
    """
    Transcribe and translate audio to English using local faster-whisper model.

    Args:
        audio_path: Path to the audio file (WAV)
        model_size: Model size (tiny, base, small, medium, large-v3)
        device: Device to use (auto, cuda, cpu)
        beam_size: Beam size for decoding
        skip_language_check: Skip language detection check before transcription
        min_language_confidence: Minimum confidence for language detection
        expected_language: Expected source language (default: ja for Japanese)

    Returns:
        Tuple of (list of translated segments in English, None for compatibility)

    Raises:
        LanguageDetectionError: If detected language doesn't match expected language
    """
    model = get_model(model_size, device)

    # Check language before transcribing (unless skipped)
    if not skip_language_check:
        detected_lang, confidence = detect_language(
            audio_path, model_size, device, min_language_confidence
        )

        if detected_lang != expected_language:
            raise LanguageDetectionError(
                f"Detected language '{detected_lang}' ({confidence:.1%} confidence) "
                f"does not match expected language '{expected_language}'. Aborting transcription. "
                f"Use --skip-language-check to bypass this check."
            )

        logger.info(f"Language check passed: {detected_lang} ({confidence:.1%} confidence)")

    logger.info(f"Transcribing and translating with local Whisper ({model_size} model)...")

    # Transcribe with translation task
    segments_iter, info = model.transcribe(
        audio_path,
        task="translate",  # Translate to English
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        )
    )

    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Convert segments to our format
    all_segments = []

    # Duration-based progress bar (shows percentage and time remaining)
    total_duration = info.duration
    bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]'

    with tqdm(total=total_duration, desc="Translating", unit="s", bar_format=bar_format) as pbar:
        last_end = 0.0
        for segment in segments_iter:
            all_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            # Update progress based on segment end time
            pbar.update(segment.end - last_end)
            last_end = segment.end

        # Ensure progress bar reaches 100%
        if last_end < total_duration:
            pbar.update(total_duration - last_end)

    logger.info(f"Translation complete. Total segments: {len(all_segments)}")

    return all_segments, None
