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
from src.transcribe import transcribe_audio, translate_segments, correct_translations
from src.merge_srt import save_subtitles
from config import ENABLE_CORRECTION


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
        help="Number of parallel workers for transcription only (default: from config or 4). Translation uses TRANSLATION_WORKERS from config."
    )

    parser.add_argument(
        "--no-correction",
        action="store_true",
        help="Disable GPT-4 correction step (faster but lower quality)"
    )

    parser.add_argument(
        "--with-correction",
        action="store_true",
        help="Enable GPT-4 correction step (overrides .env setting)"
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

        # Determine if correction should be enabled
        enable_correction = ENABLE_CORRECTION
        if args.with_correction:
            enable_correction = True
        elif args.no_correction:
            enable_correction = False

        # Determine total steps (combined chunking+transcription into one step)
        total_steps = 4 if enable_correction else 3

        # Step 1: Extract audio
        logger.info(f"\n[1/{total_steps}] Extracting audio...")
        audio_path = extract_audio(video_path)

        # Step 2: Concurrent chunking and transcription
        logger.info(f"\n[2/{total_steps}] Chunking and transcribing audio (concurrent processing)...")
        chunks_generator = chunk_audio(audio_path)
        segments, chunks_info = transcribe_audio(chunks_generator, workers=args.workers)

        # Clean up chunk files
        cleanup_chunks(chunks_info)

        # Check if we got any segments
        if not segments:
            logger.error("No transcription segments were generated. Cannot create subtitle file.")
            sys.exit(1)

        # Step 3: Translate to English using GPT-4
        logger.info(f"\n[3/{total_steps}] Translating to English with GPT-4...")
        segments = translate_segments(segments)  # Uses TRANSLATION_WORKERS from config

        # Step 4 (optional): Correct translations
        if enable_correction:
            logger.info(f"\n[4/{total_steps}] Correcting translations with GPT-4...")
            segments = correct_translations(segments)

        # Final step: Save subtitles
        logger.info(f"\n[{total_steps}/{total_steps}] Generating subtitle file...")
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
