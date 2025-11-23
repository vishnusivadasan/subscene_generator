"""
Utility functions for the Whisper Subtitle Generator.
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)


def validate_video_path(video_path: str) -> Path:
    """
    Validate that the video file exists and is accessible.

    Args:
        video_path: Path to the video file (absolute or relative)

    Returns:
        Path object of the validated video file

    Raises:
        FileNotFoundError: If video file doesn't exist
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {video_path}")
    return path


def ensure_directory(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory: Path to the directory

    Returns:
        Path object of the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_basename(file_path: Path) -> str:
    """
    Get the basename of a file without extension.

    Args:
        file_path: Path object of the file

    Returns:
        Basename without extension
    """
    return file_path.stem


def format_timestamp_srt(seconds: float) -> str:
    """
    Format a timestamp in seconds to SRT format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def cleanup_files(*file_paths: str) -> None:
    """
    Delete files if they exist.

    Args:
        *file_paths: Variable number of file paths to delete
    """
    for file_path in file_paths:
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.debug(f"Deleted: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")
