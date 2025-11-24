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
from src.transcribe import transcribe_audio, transcribe_audio_translate, translate_segments, correct_translations
from src.merge_srt import save_subtitles, save_japanese_srt, load_japanese_srt
from config import ENABLE_CORRECTION, BULK_TRANSLATOR, FALLBACK_CHAIN, LOCAL_WHISPER_MODEL, LOCAL_WHISPER_DEVICE, OPENAI_API_KEY


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

    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Force re-transcription even if cached Japanese subtitles exist"
    )

    parser.add_argument(
        "--direct-whisper",
        action="store_true",
        help="Use Whisper's direct translation to English (faster, cheaper, but lower quality than GPT-4 translation)"
    )

    parser.add_argument(
        "--bulk-translator",
        type=str,
        choices=["openai", "google"],
        default=None,
        help="Bulk translation method: 'openai' (high quality, costs money) or 'google' (fast, free, lower quality). Overrides BULK_TRANSLATOR from config."
    )

    parser.add_argument(
        "--fallback-chain",
        type=str,
        default=None,
        help="Comma-separated fallback chain for failed translations (e.g., 'google,openai,untranslated'). Overrides FALLBACK_CHAIN from config."
    )

    parser.add_argument(
        "--local-whisper",
        action="store_true",
        help="Use local faster-whisper model instead of OpenAI API (no API costs, requires GPU/CPU)"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default=None,
        help="Whisper model size for local transcription (default: medium). Only used with --local-whisper."
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Device for local Whisper model (default: auto). Only used with --local-whisper."
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for local Whisper decoding (default: 5). Higher values may improve accuracy but are slower. Only used with --local-whisper."
    )

    parser.add_argument(
        "--skip-language-check",
        action="store_true",
        help="Skip Japanese language detection check before transcription. Only used with --local-whisper."
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

        # Determine local whisper settings
        use_local_whisper = args.local_whisper
        whisper_model = args.model if args.model else LOCAL_WHISPER_MODEL
        whisper_device = args.device if args.device else LOCAL_WHISPER_DEVICE

        # Check if OpenAI API key is required
        if not use_local_whisper and not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found. Use --local-whisper for offline transcription or set the API key in .env")
            sys.exit(1)

        if use_local_whisper:
            logger.info(f"Using local Whisper: model={whisper_model}, device={whisper_device}, beam_size={args.beam_size}")

        # Determine if correction should be enabled
        enable_correction = ENABLE_CORRECTION
        if args.with_correction:
            enable_correction = True
        elif args.no_correction:
            enable_correction = False

        # Direct whisper mode is incompatible with correction (already translated)
        if args.direct_whisper and enable_correction:
            logger.warning("--direct-whisper is incompatible with correction. Disabling correction.")
            enable_correction = False

        # Determine bulk translator (CLI overrides config)
        bulk_translator = args.bulk_translator if args.bulk_translator else BULK_TRANSLATOR

        # Determine fallback chain (CLI overrides config)
        fallback_chain = args.fallback_chain.split(',') if args.fallback_chain else FALLBACK_CHAIN

        # Determine total steps
        if args.direct_whisper or (use_local_whisper and args.direct_whisper):
            # Direct whisper: extract + translate = 2 steps
            total_steps = 2
        elif use_local_whisper:
            # Local whisper: extract + transcribe + translate (+ optional correction)
            # No chunking step needed for local whisper
            total_steps = 4 if enable_correction else 3
        else:
            # Normal mode: extract + transcribe + translate (+ optional correction)
            total_steps = 4 if enable_correction else 3

        # Check for cached Japanese subtitles (skip transcription if found)
        # Skip cache entirely when using direct whisper (no Japanese intermediate)
        segments = None
        if not args.force_transcribe and not args.direct_whisper:
            segments = load_japanese_srt(video_path)

        if segments:
            logger.info("Using cached Japanese subtitles (skip transcription)")
            logger.info("Use --force-transcribe to re-transcribe from audio")
        else:
            # Step 1: Extract audio
            logger.info(f"\n[1/{total_steps}] Extracting audio...")
            audio_path = extract_audio(video_path)

            # Step 2: Transcription/translation
            if use_local_whisper:
                # Local Whisper transcription (no chunking needed)
                from src.transcribe_local import (
                    transcribe_audio_local,
                    transcribe_audio_local_translate,
                    LanguageDetectionError
                )

                if args.direct_whisper:
                    # Direct translation to English
                    logger.info(f"\n[2/{total_steps}] Transcribing and translating with local Whisper ({whisper_model})...")
                    segments, _ = transcribe_audio_local_translate(
                        audio_path,
                        model_size=whisper_model,
                        device=whisper_device,
                        beam_size=args.beam_size,
                        skip_language_check=args.skip_language_check
                    )
                else:
                    # Transcribe to Japanese first
                    logger.info(f"\n[2/{total_steps}] Transcribing with local Whisper ({whisper_model})...")
                    segments, _ = transcribe_audio_local(
                        audio_path,
                        model_size=whisper_model,
                        device=whisper_device,
                        beam_size=args.beam_size,
                        skip_language_check=args.skip_language_check
                    )
                chunks_info = None  # No chunks to clean up
            elif args.direct_whisper:
                # Direct Whisper API translation (no GPT-4 needed)
                logger.info(f"\n[2/{total_steps}] Chunking and translating to English with Whisper API (concurrent processing)...")
                chunks_generator = chunk_audio(audio_path)
                segments, chunks_info = transcribe_audio_translate(chunks_generator, workers=args.workers)
            else:
                # Normal API mode: Transcribe to Japanese first
                logger.info(f"\n[2/{total_steps}] Chunking and transcribing audio (concurrent processing)...")
                chunks_generator = chunk_audio(audio_path)
                segments, chunks_info = transcribe_audio(chunks_generator, workers=args.workers)

            # Clean up chunk files (if any)
            if chunks_info:
                cleanup_chunks(chunks_info)

            # Check if we got any segments
            if not segments:
                logger.error("No transcription segments were generated. Cannot create subtitle file.")
                sys.exit(1)

            # Save Japanese subtitles for future use (only in normal mode)
            if not args.direct_whisper:
                save_japanese_srt(segments, video_path)

            # Clean up extracted audio file
            cleanup_files(audio_path)

        # Step 3: Translate to English (skip if using direct Whisper)
        if not args.direct_whisper:
            translator_name = bulk_translator.upper()
            logger.info(f"\n[3/{total_steps}] Translating to English with {translator_name}...")
            logger.info(f"Fallback chain: {' → '.join(fallback_chain)}")
            segments = translate_segments(segments, bulk_translator=bulk_translator, fallback_chain=fallback_chain)

        # Step 4 (optional): Correct translations (only in normal mode)
        if enable_correction and not args.direct_whisper:
            logger.info(f"\n[4/{total_steps}] Correcting translations with GPT-4...")
            segments = correct_translations(segments)

        # Final step: Save subtitles
        logger.info(f"\n[{total_steps}/{total_steps}] Generating subtitle file...")
        srt_path = save_subtitles(segments, video_path)

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
