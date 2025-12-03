#!/usr/bin/env python3
"""
Whisper Subtitle Generator - Main Entry Point

Processes video files to generate English subtitle files using OpenAI Whisper API.
"""

import argparse
import sys
from pathlib import Path

from utils import (
    validate_video_path, validate_input_path, find_video_files, has_existing_srt,
    logger, cleanup_files, load_metadata, save_metadata, create_metadata,
    update_metadata_processed
)
from src.extract_audio import extract_audio
from src.chunk_audio import chunk_audio, cleanup_chunks
from src.transcribe import transcribe_audio, transcribe_audio_translate, translate_segments, correct_translations
from src.merge_srt import save_subtitles, save_japanese_srt, load_japanese_srt
from config import ENABLE_CORRECTION, BULK_TRANSLATOR, FALLBACK_CHAIN, LOCAL_WHISPER_MODEL, LOCAL_WHISPER_DEVICE, OPENAI_API_KEY


def process_single_video(
    video_path: Path,
    args,
    use_local_whisper: bool,
    whisper_model: str,
    whisper_device: str,
    enable_correction: bool,
    bulk_translator: str,
    fallback_chain: list,
    source_language: str = None,
    pre_extracted_audio: str = None
) -> bool:
    """
    Process a single video file to generate subtitles.

    Args:
        video_path: Path to the video file
        args: Parsed command line arguments
        use_local_whisper: Whether to use local Whisper model
        whisper_model: Whisper model size
        whisper_device: Device for local Whisper
        enable_correction: Whether to enable GPT-4 correction
        bulk_translator: Bulk translation method
        fallback_chain: Fallback chain for translation
        source_language: Detected source language code (e.g., 'ja', 'zh', 'ko'). If None, will auto-detect.
        pre_extracted_audio: Path to pre-extracted audio file (from prefetch). If None, will extract.

    Returns:
        True on success, False on failure
    """
    try:
        logger.info(f"Processing: {video_path}")

        # Determine total steps
        if args.direct_whisper or (use_local_whisper and args.direct_whisper):
            total_steps = 2
        elif use_local_whisper:
            total_steps = 4 if enable_correction else 3
        else:
            total_steps = 4 if enable_correction else 3

        # Check for cached source subtitles
        segments = None
        if not args.force_transcribe and not args.direct_whisper:
            segments = load_japanese_srt(video_path)  # TODO: rename to load_source_srt

        if segments:
            logger.info("Using cached source subtitles (skip transcription)")
        else:
            # Step 1: Extract audio (skip if pre-extracted)
            if pre_extracted_audio:
                logger.info(f"\n[1/{total_steps}] Using pre-extracted audio...")
                audio_path = pre_extracted_audio
            else:
                logger.info(f"\n[1/{total_steps}] Extracting audio...")
                audio_path = extract_audio(video_path)

            # Step 2: Transcription/translation
            if use_local_whisper:
                from src.transcribe_local import (
                    transcribe_audio_local,
                    transcribe_audio_local_translate,
                    LanguageDetectionError
                )

                if args.direct_whisper:
                    logger.info(f"\n[2/{total_steps}] Transcribing and translating with local Whisper ({whisper_model})...")
                    segments, _ = transcribe_audio_local_translate(
                        audio_path,
                        model_size=whisper_model,
                        device=whisper_device,
                        beam_size=args.beam_size,
                        skip_language_check=True if source_language else args.skip_language_check,
                        expected_language=source_language if source_language else "ja"
                    )
                else:
                    logger.info(f"\n[2/{total_steps}] Transcribing with local Whisper ({whisper_model})...")
                    # Use detected language if provided, otherwise let it auto-detect
                    transcribe_lang = source_language if source_language else "ja"
                    segments, _ = transcribe_audio_local(
                        audio_path,
                        model_size=whisper_model,
                        device=whisper_device,
                        language=transcribe_lang,
                        beam_size=args.beam_size,
                        skip_language_check=True if source_language else args.skip_language_check
                    )
                chunks_info = None
            elif args.direct_whisper:
                logger.info(f"\n[2/{total_steps}] Chunking and translating to English with Whisper API...")
                chunks_generator = chunk_audio(audio_path)
                segments, chunks_info = transcribe_audio_translate(chunks_generator, workers=args.workers)
            else:
                logger.info(f"\n[2/{total_steps}] Chunking and transcribing audio...")
                chunks_generator = chunk_audio(audio_path)
                segments, chunks_info = transcribe_audio(chunks_generator, workers=args.workers)

            # Clean up chunk files
            if chunks_info:
                cleanup_chunks(chunks_info)

            if not segments:
                logger.error("No transcription segments were generated.")
                return False

            # Save Japanese subtitles for future use
            if not args.direct_whisper:
                save_japanese_srt(segments, video_path)

            # Clean up extracted audio file
            cleanup_files(audio_path)

        # Step 3: Translate to English
        if not args.direct_whisper:
            translator_name = bulk_translator.upper()
            logger.info(f"\n[3/{total_steps}] Translating to English with {translator_name}...")
            logger.info(f"Fallback chain: {' → '.join(fallback_chain)}")
            segments = translate_segments(segments, bulk_translator=bulk_translator, fallback_chain=fallback_chain)

        # Step 4 (optional): Correct translations
        if enable_correction and not args.direct_whisper:
            logger.info(f"\n[4/{total_steps}] Correcting translations with GPT-4...")
            segments = correct_translations(segments)

        # Final step: Save subtitles
        logger.info(f"\n[{total_steps}/{total_steps}] Generating subtitle file...")
        srt_path = save_subtitles(segments, video_path)

        logger.info(f"SUCCESS! Subtitle created: {srt_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}")
        return False


def prepare_video(
    video_path: Path,
    whisper_model: str,
    whisper_device: str,
    settings: dict,
    detect_language_func,
    get_model_func
) -> tuple:
    """
    Prepare a video for processing (can run in background thread).

    Checks metadata, extracts audio, and detects language.

    Args:
        video_path: Path to the video file
        whisper_model: Whisper model size for language detection
        whisper_device: Device for Whisper model
        settings: Settings dict for metadata
        detect_language_func: Language detection function
        get_model_func: Model loading function

    Returns:
        Tuple of (video_path, detected_lang, confidence, audio_path, skip_reason)
        - skip_reason is None if video should be processed
        - skip_reason is "english" if video is in English
        - skip_reason is "processed" if already processed
        - audio_path is None if skipped or if using cached metadata
    """
    try:
        # Check existing metadata first
        metadata = load_metadata(video_path)

        if metadata:
            # Check if already English (skip without re-detecting)
            if metadata.get("detected_language") == "en":
                return (video_path, "en", metadata.get("language_confidence", 0), None, "english")

            # Check if already processed and SRT exists
            if metadata.get("processed") and has_existing_srt(video_path):
                return (video_path, None, 0, None, "processed")

            # Have metadata but not processed - use cached language info
            detected_lang = metadata.get("detected_language")
            confidence = metadata.get("language_confidence", 0)
            return (video_path, detected_lang, confidence, None, None)

        # No metadata - need to detect language
        # Extract audio for language detection
        audio_path = extract_audio(video_path)

        # Detect language
        detected_lang, confidence = detect_language_func(
            audio_path,
            model_size=whisper_model,
            device=whisper_device
        )

        # Create and save metadata
        new_metadata = create_metadata(video_path, detected_lang, confidence, settings)
        save_metadata(video_path, new_metadata)

        if detected_lang == "en":
            cleanup_files(audio_path)
            return (video_path, "en", confidence, None, "english")

        # Return with audio path for processing
        return (video_path, detected_lang, confidence, audio_path, None)

    except Exception as e:
        logger.error(f"Error preparing {video_path.name}: {e}")
        return (video_path, None, 0, None, f"error: {e}")


def process_folder(
    folder_path: Path,
    args,
    use_local_whisper: bool,
    whisper_model: str,
    whisper_device: str,
    enable_correction: bool,
    bulk_translator: str,
    fallback_chain: list
) -> None:
    """
    Process all video files in a folder.

    Args:
        folder_path: Path to the folder
        args: Parsed command line arguments
        use_local_whisper: Whether to use local Whisper model
        whisper_model: Whisper model size
        whisper_device: Device for local Whisper
        enable_correction: Whether to enable GPT-4 correction
        bulk_translator: Bulk translation method
        fallback_chain: Fallback chain for translation
    """
    from src.transcribe_local import detect_language, get_model

    # Find all video files
    video_files = find_video_files(folder_path)

    if not video_files:
        logger.warning(f"No video files found in {folder_path}")
        return

    logger.info(f"Found {len(video_files)} video file(s) in {folder_path}")

    # Filter out files with existing SRT if requested
    if args.skip_existing:
        original_count = len(video_files)
        video_files = [v for v in video_files if not has_existing_srt(v)]
        skipped_existing = original_count - len(video_files)
        if skipped_existing > 0:
            logger.info(f"Skipping {skipped_existing} video(s) with existing .srt files")
    else:
        skipped_existing = 0

    if not video_files:
        logger.info("No videos to process (all have existing subtitles)")
        return

    # Build settings dict for metadata
    settings = {
        "whisper_model": whisper_model,
        "bulk_translator": bulk_translator,
        "with_correction": enable_correction,
        "direct_whisper": args.direct_whisper
    }

    # Tracking statistics
    processed = 0
    skipped_language = 0
    skipped_metadata = 0
    failed = 0
    skipped_files = []

    total = len(video_files)

    # Pre-load Whisper model for language detection (needed for prefetch threads)
    logger.info("Loading Whisper model for language detection...")
    get_model(whisper_model, whisper_device)

    # Check if prefetch is enabled
    use_prefetch = not getattr(args, 'no_prefetch', False) and total > 1

    if use_prefetch:
        from concurrent.futures import ThreadPoolExecutor
        logger.info("Pipeline mode: prefetching next video while processing")

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit first video for preparation
            pending_future = executor.submit(
                prepare_video, video_files[0], whisper_model, whisper_device,
                settings, detect_language, get_model
            )

            for idx, video_path in enumerate(video_files):
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"[{idx + 1}/{total}] Processing: {video_path.name}")
                logger.info("=" * 60)

                try:
                    # Get preparation result for current video
                    prep_result = pending_future.result()
                    _, detected_lang, confidence, audio_path, skip_reason = prep_result

                    # Submit NEXT video for preparation (if exists)
                    if idx + 1 < len(video_files):
                        pending_future = executor.submit(
                            prepare_video, video_files[idx + 1], whisper_model, whisper_device,
                            settings, detect_language, get_model
                        )

                    # Handle skip reasons
                    if skip_reason == "english":
                        logger.info(f"Skipping (already English): {detected_lang} ({confidence:.1%} confidence)")
                        skipped_language += 1
                        skipped_files.append((video_path.name, detected_lang, confidence))
                        continue
                    elif skip_reason == "processed":
                        logger.info(f"Skipping (metadata: already processed)")
                        skipped_metadata += 1
                        continue
                    elif skip_reason and skip_reason.startswith("error:"):
                        logger.error(f"Preparation failed: {skip_reason}")
                        failed += 1
                        continue

                    logger.info(f"Language: {detected_lang} ({confidence:.1%} confidence) - proceeding with transcription")

                    # Process the video with detected language and pre-extracted audio
                    success = process_single_video(
                        video_path,
                        args,
                        use_local_whisper,
                        whisper_model,
                        whisper_device,
                        enable_correction,
                        bulk_translator,
                        fallback_chain,
                        source_language=detected_lang,
                        pre_extracted_audio=audio_path
                    )

                    if success:
                        processed += 1
                        update_metadata_processed(video_path, video_path.with_suffix('.srt').name)
                    else:
                        failed += 1
                        # Clean up audio if processing failed
                        if audio_path:
                            cleanup_files(audio_path)

                except KeyboardInterrupt:
                    logger.info("\n\nProcess interrupted by user")
                    logger.info(f"Processed {processed} video(s) before interruption")
                    raise

                except Exception as e:
                    logger.error(f"Error processing {video_path}: {e}")
                    failed += 1

    else:
        # Sequential processing (no prefetch)
        for idx, video_path in enumerate(video_files, 1):
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"[{idx}/{total}] Processing: {video_path.name}")
            logger.info("=" * 60)

            try:
                # Prepare video (synchronous)
                _, detected_lang, confidence, audio_path, skip_reason = prepare_video(
                    video_path, whisper_model, whisper_device,
                    settings, detect_language, get_model
                )

                # Handle skip reasons
                if skip_reason == "english":
                    logger.info(f"Skipping (already English): {detected_lang} ({confidence:.1%} confidence)")
                    skipped_language += 1
                    skipped_files.append((video_path.name, detected_lang, confidence))
                    continue
                elif skip_reason == "processed":
                    logger.info(f"Skipping (metadata: already processed)")
                    skipped_metadata += 1
                    continue
                elif skip_reason and skip_reason.startswith("error:"):
                    logger.error(f"Preparation failed: {skip_reason}")
                    failed += 1
                    continue

                logger.info(f"Language: {detected_lang} ({confidence:.1%} confidence) - proceeding with transcription")

                # Process the video
                success = process_single_video(
                    video_path,
                    args,
                    use_local_whisper,
                    whisper_model,
                    whisper_device,
                    enable_correction,
                    bulk_translator,
                    fallback_chain,
                    source_language=detected_lang,
                    pre_extracted_audio=audio_path
                )

                if success:
                    processed += 1
                    update_metadata_processed(video_path, video_path.with_suffix('.srt').name)
                else:
                    failed += 1
                    if audio_path:
                        cleanup_files(audio_path)

            except KeyboardInterrupt:
                logger.info("\n\nProcess interrupted by user")
                logger.info(f"Processed {processed} video(s) before interruption")
                raise

            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                failed += 1

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Folder Processing Complete")
    logger.info("=" * 60)
    logger.info(f"Total files found: {total + skipped_existing}")
    logger.info(f"  - Processed successfully: {processed}")
    logger.info(f"  - Skipped (already English): {skipped_language}")
    logger.info(f"  - Skipped (already processed): {skipped_metadata}")
    logger.info(f"  - Skipped (SRT exists, --skip-existing): {skipped_existing}")
    logger.info(f"  - Failed: {failed}")

    if skipped_files:
        logger.info("")
        logger.info("Files skipped due to language:")
        for name, lang, conf in skipped_files:
            logger.info(f"  - {name}: {lang} ({conf:.1%})")

    logger.info("=" * 60)


def main():
    """Main entry point for the subtitle generator."""

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate English subtitles from video files using Whisper API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py test_file.mp4                    # Process single video
  python main.py /path/to/videos/                 # Process all videos in folder
  python main.py /path/to/videos/ --skip-existing # Skip videos with existing .srt
  python main.py "/path/with spaces/video.mp4"
        """
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a video file or folder containing videos (supports absolute and relative paths)"
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

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have .srt subtitle files (only applies to folder processing)"
    )

    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable background prefetching of next video (reduces memory usage, slower processing)"
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Validate input path (can be file or directory)
        logger.info("=" * 60)
        logger.info("Whisper Subtitle Generator")
        logger.info("=" * 60)

        input_path, is_directory = validate_input_path(args.input_path)

        # Determine local whisper settings
        use_local_whisper = args.local_whisper
        whisper_model = args.model if args.model else LOCAL_WHISPER_MODEL
        whisper_device = args.device if args.device else LOCAL_WHISPER_DEVICE

        # Folder processing requires local whisper for language detection
        if is_directory and not use_local_whisper:
            logger.info("Folder mode: enabling local Whisper for language detection")
            use_local_whisper = True

        # Check if OpenAI API key is required (for translation, not transcription in local mode)
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

        if is_directory:
            # Process folder
            logger.info(f"Input folder: {input_path}")
            process_folder(
                input_path,
                args,
                use_local_whisper,
                whisper_model,
                whisper_device,
                enable_correction,
                bulk_translator,
                fallback_chain
            )
        else:
            # Process single video
            logger.info(f"Input video: {input_path}")
            success = process_single_video(
                input_path,
                args,
                use_local_whisper,
                whisper_model,
                whisper_device,
                enable_correction,
                bulk_translator,
                fallback_chain
            )
            if not success:
                sys.exit(1)

            logger.info("\n" + "=" * 60)
            logger.info("Processing complete!")
            logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\n\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
