#!/usr/bin/env python3
"""
Whisper Subtitle Generator - Main Entry Point

Processes video files to generate English subtitle files using OpenAI Whisper API.
"""

import argparse
import sys
from pathlib import Path

from utils import validate_video_path, logger, cleanup_files
from src.extract_audio import extract_audio
from src.chunk_audio import chunk_audio, cleanup_chunks
from src.transcribe import transcribe_audio
from src.merge_srt import save_subtitles


def main():
    """Main entry point for the subtitle generator."""

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate English subtitles from video files using Whisper API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py test_file.mp4
  python main.py input_videos/myvideo.mp4
  python main.py "/path/with spaces/video.mp4"
        """
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file (supports absolute and relative paths)"
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for transcription (default: from config or 4)"
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Validate video path
        logger.info("=" * 60)
        logger.info("Whisper Subtitle Generator")
        logger.info("=" * 60)

        video_path = validate_video_path(args.video_path)
        logger.info(f"Input video: {video_path}")

        # Step 1: Extract audio
        logger.info("\n[1/4] Extracting audio...")
        audio_path = extract_audio(video_path)

        # Step 2: Chunk audio
        logger.info("\n[2/4] Chunking audio...")
        chunks_info = chunk_audio(audio_path)

        # Step 3: Transcribe chunks in parallel
        logger.info("\n[3/4] Transcribing audio (this may take a while)...")
        segments = transcribe_audio(chunks_info, workers=args.workers)

        # Clean up chunk files
        cleanup_chunks(chunks_info)

        # Check if we got any segments
        if not segments:
            logger.error("No transcription segments were generated. Cannot create subtitle file.")
            sys.exit(1)

        # Step 4: Save subtitles
        logger.info("\n[4/4] Generating subtitle file...")
        srt_path = save_subtitles(segments, video_path)

        # Clean up extracted audio file
        cleanup_files(audio_path)

        # Success!
        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS! Subtitle file created:")
        logger.info(f"  {srt_path}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\n\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
