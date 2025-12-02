"""
Utility functions for the Whisper Subtitle Generator.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple

# Supported video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'}

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


def validate_input_path(input_path: str) -> Tuple[Path, bool]:
    """
    Validate input path - can be file or directory.

    Args:
        input_path: Path to a video file or directory

    Returns:
        Tuple of (resolved_path, is_directory)

    Raises:
        FileNotFoundError: If path doesn't exist
    """
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    return path, path.is_dir()


def is_video_file(path: Path) -> bool:
    """
    Check if a file has a supported video extension.

    Args:
        path: Path to check

    Returns:
        True if file has a supported video extension
    """
    return path.suffix.lower() in VIDEO_EXTENSIONS


def find_video_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Find all video files in a directory.

    Args:
        directory: Directory to search
        recursive: If True, search subdirectories recursively

    Returns:
        Sorted list of video file paths
    """
    video_files = []
    if recursive:
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(directory.rglob(f"*{ext}"))
            # Also match uppercase extensions
            video_files.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(directory.glob(f"*{ext}"))
            video_files.extend(directory.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    return sorted(set(video_files))


def has_existing_srt(video_path: Path) -> bool:
    """
    Check if an SRT file already exists for this video.

    Args:
        video_path: Path to the video file

    Returns:
        True if .srt file exists alongside the video
    """
    srt_path = video_path.with_suffix('.srt')
    return srt_path.exists()


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
