"""
SRT subtitle file generation module.
Formats transcription segments into proper SRT format.
"""

from typing import List, Dict
from pathlib import Path
from utils import logger, ensure_directory, format_timestamp_srt, get_basename


def create_srt(segments: List[Dict[str, any]], output_path: str) -> None:
    """
    Create an SRT subtitle file from transcription segments.

    Args:
        segments: List of segment dictionaries with start, end, and text
        output_path: Path where the SRT file will be saved
    """
    # Sort segments by start time (should already be sorted, but ensure)
    segments.sort(key=lambda x: x["start"])

    logger.info(f"Generating SRT file with {len(segments)} segments")

    # Build SRT content
    srt_content = []

    for index, segment in enumerate(segments, start=1):
        start_time = format_timestamp_srt(segment["start"])
        end_time = format_timestamp_srt(segment["end"])
        text = segment["text"]

        # SRT format:
        # 1
        # 00:00:05,120 --> 00:00:07,900
        # Translated text here
        #
        srt_block = f"{index}\n{start_time} --> {end_time}\n{text}\n"
        srt_content.append(srt_block)

    # Join all blocks with blank line separator
    final_srt = "\n".join(srt_content)

    # Write to file with UTF-8 encoding
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_srt)

    logger.info(f"SRT file created successfully: {output_path}")


def save_subtitles(segments: List[Dict[str, any]], video_path: Path) -> str:
    """
    Save transcription segments as SRT file in the same directory as the video.

    Args:
        segments: List of transcription segments
        video_path: Path object of the original video file

    Returns:
        Path to the created SRT file
    """
    # Get the directory of the input video
    video_directory = video_path.parent

    # Create output path in the same directory as the video
    basename = get_basename(video_path)
    output_path = video_directory / f"{basename}.srt"

    # Create SRT file
    create_srt(segments, str(output_path))

    return str(output_path)


def save_japanese_srt(segments: List[Dict[str, any]], video_path: Path) -> str:
    """
    Save Japanese transcription segments as .ja.srt file for caching.

    Args:
        segments: List of Japanese transcription segments
        video_path: Path object of the original video file

    Returns:
        Path to the created Japanese SRT file
    """
    # Get the directory of the input video
    video_directory = video_path.parent

    # Create output path with .ja.srt extension
    basename = get_basename(video_path)
    output_path = video_directory / f"{basename}.ja.srt"

    # Create SRT file
    create_srt(segments, str(output_path))

    logger.info(f"Japanese subtitles cached: {output_path}")
    return str(output_path)


def load_japanese_srt(video_path: Path) -> List[Dict[str, any]]:
    """
    Load cached Japanese subtitles from .ja.srt file.

    Args:
        video_path: Path object of the original video file

    Returns:
        List of segment dictionaries, or None if file doesn't exist
    """
    # Get the directory of the input video
    video_directory = video_path.parent

    # Check for .ja.srt file
    basename = get_basename(video_path)
    ja_srt_path = video_directory / f"{basename}.ja.srt"

    if not ja_srt_path.exists():
        return None

    logger.info(f"Found cached Japanese subtitles: {ja_srt_path}")

    # Parse SRT file
    segments = []
    try:
        with open(ja_srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by double newlines to get blocks
        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                # Line 0: sequence number (ignore)
                # Line 1: timestamps
                # Line 2+: text
                timestamp_line = lines[1]
                text = "\n".join(lines[2:])

                # Parse timestamps: "00:00:05,120 --> 00:00:07,900"
                if " --> " in timestamp_line:
                    start_str, end_str = timestamp_line.split(" --> ")

                    # Convert SRT timestamp to seconds
                    def srt_to_seconds(srt_time):
                        hours, minutes, rest = srt_time.split(":")
                        seconds, milliseconds = rest.split(",")
                        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0

                    start = srt_to_seconds(start_str)
                    end = srt_to_seconds(end_str)

                    segments.append({
                        "start": start,
                        "end": end,
                        "text": text
                    })

        logger.info(f"Loaded {len(segments)} segments from cached Japanese subtitles")
        return segments

    except Exception as e:
        logger.error(f"Failed to load Japanese SRT file: {e}")
        return None
