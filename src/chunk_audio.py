"""
Audio chunking module using ffmpeg.
Splits audio files into chunks for parallel processing.
"""

import subprocess
from typing import List, Dict
from tqdm import tqdm
from utils import logger, ensure_directory
from config import CHUNK_DURATION_MS


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds using ffprobe.

    Args:
        audio_path: Path to the audio file

    Returns:
        Duration in seconds
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def chunk_audio(audio_path: str) -> List[Dict[str, any]]:
    """
    Split audio file into chunks of specified duration using ffmpeg.
    Returns generator that yields chunks as they are created.

    Args:
        audio_path: Path to the WAV audio file

    Yields:
        Dictionary with chunk information:
        {
            "chunk_path": "audio/chunk_0.wav",
            "offset_seconds": 0.0
        }
    """
    logger.info(f"Loading audio file: {audio_path}")

    # Get audio duration
    total_duration_sec = get_audio_duration(audio_path)
    logger.info(f"Audio duration: {total_duration_sec:.2f} seconds")

    # Ensure audio directory exists
    ensure_directory("audio")

    chunk_duration_sec = CHUNK_DURATION_MS / 1000.0

    # Calculate total number of chunks
    total_chunks = int((total_duration_sec + chunk_duration_sec - 1) // chunk_duration_sec)
    logger.info(f"Will create {total_chunks} chunks (streaming to transcription as created)")

    chunk_start_sec = 0.0

    # Split audio into chunks using ffmpeg - yield each chunk immediately
    while chunk_start_sec < total_duration_sec:
        # Calculate chunk duration (or use remaining audio if less than chunk size)
        current_chunk_duration = min(chunk_duration_sec, total_duration_sec - chunk_start_sec)

        # Create chunk filename
        chunk_path = f"audio/chunk_{int(chunk_start_sec * 1000)}.wav"

        # Use ffmpeg to extract chunk with re-encoding to ensure compatibility
        command = [
            "ffmpeg",
            "-i", audio_path,
            "-ss", str(chunk_start_sec),
            "-t", str(current_chunk_duration),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            chunk_path,
            "-y"
        ]

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        # Yield chunk info immediately for transcription
        yield {
            "chunk_path": chunk_path,
            "offset_seconds": chunk_start_sec
        }

        chunk_start_sec += chunk_duration_sec


def cleanup_chunks(chunks_info: List[Dict[str, any]]) -> None:
    """
    Delete all chunk files after processing.

    Args:
        chunks_info: List of chunk information dictionaries
    """
    from utils import cleanup_files

    chunk_paths = [chunk["chunk_path"] for chunk in chunks_info]
    logger.info(f"Cleaning up {len(chunk_paths)} chunk files...")
    cleanup_files(*chunk_paths)
    logger.info("Chunk cleanup complete")
