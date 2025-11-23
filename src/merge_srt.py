"""
SRT subtitle file generation module.
Formats transcription segments into proper SRT format.
"""

from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
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

    # Build SRT content with progress bar
    srt_content = []

    for index, segment in tqdm(enumerate(segments, start=1), total=len(segments), desc="Writing SRT", unit="segment"):
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
