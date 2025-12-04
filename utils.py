"""
Utility functions for the Whisper Subtitle Generator.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

# Supported video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'}

# Metadata file suffix
METADATA_SUFFIX = '.subscene.json'
METADATA_VERSION = 1

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
        # Single pass through all files, filter by extension in Python
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(path)
    else:
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(path)

    return sorted(video_files)


def scan_video_folder(directory: Path, recursive: bool = True) -> dict:
    """
    Scan a directory once and collect all video files, SRT files, and metadata.

    This is optimized for network filesystems - does a single directory traversal
    instead of multiple glob operations and individual file existence checks.

    Args:
        directory: Directory to search
        recursive: If True, search subdirectories recursively

    Returns:
        Dict with:
        - videos: List of video file paths
        - srt_set: Set of video stems that have .srt files
        - metadata: Dict mapping video path -> loaded metadata (or None)
    """
    video_files = []
    srt_stems = set()  # Store stems of files that have SRTs
    metadata_files = {}  # Map video base name -> metadata file path

    # Single pass through all files
    iterator = directory.rglob("*") if recursive else directory.iterdir()

    for path in iterator:
        if not path.is_file():
            continue

        suffix_lower = path.suffix.lower()

        # Collect video files
        if suffix_lower in VIDEO_EXTENSIONS:
            video_files.append(path)
        # Track SRT files by their stem (filename without extension)
        elif suffix_lower == '.srt':
            srt_stems.add(path.stem)
        # Track metadata files
        elif path.name.endswith(METADATA_SUFFIX):
            # Extract the video filename this metadata belongs to
            # e.g., "video.mp4.subscene.json" -> "video.mp4"
            video_name = path.name[:-len(METADATA_SUFFIX)]
            metadata_files[video_name] = path

    # Sort videos
    video_files = sorted(video_files)

    # Build set of video paths that have SRTs (for fast lookup)
    # A video "foo.mp4" has an SRT if "foo" is in srt_stems
    srt_set = set()
    for video_path in video_files:
        if video_path.stem in srt_stems:
            srt_set.add(video_path)

    # Load metadata for each video (reading files is unavoidable, but we know which exist)
    metadata = {}
    for video_path in video_files:
        meta_path = metadata_files.get(video_path.name)
        if meta_path:
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata[video_path] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load metadata for {video_path.name}: {e}")
                metadata[video_path] = None
        else:
            metadata[video_path] = None

    return {
        'videos': video_files,
        'srt_set': srt_set,
        'metadata': metadata
    }


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


def get_metadata_path(video_path: Path) -> Path:
    """
    Get the metadata file path for a video.

    Args:
        video_path: Path to the video file

    Returns:
        Path to the metadata JSON file
    """
    return video_path.parent / f"{video_path.name}{METADATA_SUFFIX}"


def load_metadata(video_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load metadata for a video if it exists.

    Args:
        video_path: Path to the video file

    Returns:
        Metadata dictionary or None if not found
    """
    metadata_path = get_metadata_path(video_path)
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load metadata for {video_path.name}: {e}")
        return None


def save_metadata(video_path: Path, metadata: Dict[str, Any]) -> None:
    """
    Save metadata for a video.

    Args:
        video_path: Path to the video file
        metadata: Metadata dictionary to save
    """
    metadata_path = get_metadata_path(video_path)
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.warning(f"Failed to save metadata for {video_path.name}: {e}")


def create_metadata(
    video_path: Path,
    detected_language: str,
    language_confidence: float,
    settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new metadata dictionary.

    Args:
        video_path: Path to the video file
        detected_language: Detected language code
        language_confidence: Confidence of language detection
        settings: Processing settings used

    Returns:
        Metadata dictionary
    """
    return {
        "version": METADATA_VERSION,
        "video_file": video_path.name,
        "detected_language": detected_language,
        "language_confidence": language_confidence,
        "detected_at": datetime.utcnow().isoformat() + "Z",
        "processed": False,
        "processed_at": None,
        "settings": settings,
        "srt_file": None
    }


def update_metadata_processed(video_path: Path, srt_filename: str) -> None:
    """
    Update metadata to mark a video as processed.

    Args:
        video_path: Path to the video file
        srt_filename: Name of the generated SRT file
    """
    metadata = load_metadata(video_path)
    if metadata:
        metadata["processed"] = True
        metadata["processed_at"] = datetime.utcnow().isoformat() + "Z"
        metadata["srt_file"] = srt_filename
        save_metadata(video_path, metadata)
