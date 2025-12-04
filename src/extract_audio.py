"""
Audio extraction module using ffmpeg.
Extracts audio from video files and converts to the required format.
"""

import subprocess
from pathlib import Path
from utils import logger, ensure_directory, get_basename
from config import AUDIO_SETTINGS


def extract_audio(video_path: Path) -> str:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path object of the video file

    Returns:
        Path to the extracted WAV audio file

    Raises:
        RuntimeError: If ffmpeg extraction fails
    """
    # Ensure audio directory exists
    ensure_directory("audio")

    # Create output path
    basename = get_basename(video_path)
    output_path = f"audio/{basename}.wav"

    logger.info(f"Extracting audio from: {video_path.name}")

    # Build ffmpeg command
    # -i: input file
    # -vn: no video
    # -acodec pcm_s16le: PCM 16-bit little-endian codec
    # -ar 16000: audio sample rate 16 kHz
    # -ac 1: mono audio (1 channel)
    # -y: overwrite output file if exists
    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", AUDIO_SETTINGS["codec"],
        "-ar", str(AUDIO_SETTINGS["sample_rate"]),
        "-ac", str(AUDIO_SETTINGS["channels"]),
        output_path,
        "-y"
    ]

    try:
        # Run ffmpeg command
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        # Check stderr for critical errors (ffmpeg may exit 0 despite corrupt audio)
        stderr_output = result.stderr.decode() if result.stderr else ""
        critical_errors = [
            "Conversion failed",
            "Error while filtering",
            "Failed to inject frame into filter network",
            "Rematrix is needed between",
        ]
        for error in critical_errors:
            if error in stderr_output:
                logger.error(f"ffmpeg extraction failed (corrupt audio): {error}")
                # Clean up partial output file
                if Path(output_path).exists():
                    Path(output_path).unlink()
                raise RuntimeError(f"Corrupt audio stream: {error}")

        logger.info(f"Audio extracted successfully: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"ffmpeg extraction failed: {error_msg}")
        raise RuntimeError(f"Failed to extract audio from video: {error_msg}")

    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg: "
            "https://ffmpeg.org/download.html"
        )
