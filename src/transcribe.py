"""
Parallel transcription module using OpenAI Whisper API.
Handles concurrent API calls with retry logic and rate limit handling.
"""

import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import logger
from config import client, WORKERS, GPT_MODEL, ENABLE_CORRECTION, CORRECTION_BATCH_SIZE


def process_chunk(chunk_info: Dict[str, any]) -> List[Dict[str, any]]:
    """
    Process a single audio chunk with Whisper API.
    Includes retry logic with exponential backoff.

    Args:
        chunk_info: Dictionary with chunk_path and offset_seconds

    Returns:
        List of transcription segments with adjusted timestamps
    """
    chunk_path = chunk_info["chunk_path"]
    offset_seconds = chunk_info["offset_seconds"]

    max_retries = 3
    backoff_times = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s

    for attempt in range(max_retries):
        try:
            # Open audio file and send to Whisper API for transcription (original language)
            with open(chunk_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )

            # Extract segments from response
            segments = []
            if hasattr(response, 'segments') and response.segments:
                for seg in response.segments:
                    # Adjust timestamps by chunk offset
                    # Access attributes directly, not as dictionary
                    segments.append({
                        "start": seg.start + offset_seconds,
                        "end": seg.end + offset_seconds,
                        "text": seg.text.strip()
                    })

            return segments

        except Exception as e:
            error_message = str(e)

            # Check if it's a rate limit error (429)
            is_rate_limit = "429" in error_message or "rate_limit" in error_message.lower()

            if attempt < max_retries - 1:
                # Calculate backoff time
                backoff = backoff_times[attempt]

                if is_rate_limit:
                    logger.warning(f"[Worker] Retry #{attempt + 1} after rate limit")
                else:
                    logger.warning(
                        f"[Worker] Retry #{attempt + 1} after error: {error_message}"
                    )

                time.sleep(backoff)
            else:
                # Final failure after all retries
                logger.error(
                    f"[ERROR] Failed chunk at offset {offset_seconds:.2f}s "
                    f"after {max_retries} attempts: {error_message}"
                )
                return []

    return []


def transcribe_audio(chunks_info: List[Dict[str, any]], workers: int = None) -> List[Dict[str, any]]:
    """
    Transcribe audio chunks in parallel using ThreadPoolExecutor.

    Args:
        chunks_info: List of chunk information dictionaries
        workers: Number of parallel workers (overrides config if provided)

    Returns:
        List of all transcription segments sorted by start time
    """
    # Determine worker count
    if workers is None:
        workers = WORKERS

    logger.info(f"Starting parallel transcription with {workers} workers")
    logger.info(f"Total chunks to process: {len(chunks_info)}")

    all_segments = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {
            executor.submit(process_chunk, chunk): chunk
            for chunk in chunks_info
        }

        # Collect results as they complete with progress bar
        with tqdm(total=len(chunks_info), desc="Transcribing", unit="chunk") as pbar:
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    segments = future.result()
                    all_segments.extend(segments)
                except Exception as e:
                    logger.error(
                        f"Unexpected error processing chunk "
                        f"{chunk['chunk_path']}: {e}"
                    )
                pbar.update(1)

    # Sort all segments by start time
    all_segments.sort(key=lambda x: x["start"])

    logger.info(f"Transcription complete. Total segments: {len(all_segments)}")

    return all_segments


def translate_segment(segment: Dict[str, any]) -> Dict[str, any]:
    """
    Translate a single segment using GPT-4.

    Args:
        segment: Dictionary with start, end, and text (original language)

    Returns:
        Segment with translated English text
    """
    original_text = segment["text"]

    max_retries = 3
    backoff_times = [1, 2, 4]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional subtitle translator. Translate Japanese to natural English. "
                                   "Preserve tone, nuance, emotion, and intent. Do NOT censor or soften the meaning. "
                                   "Output only the translation, no explanations."
                    },
                    {
                        "role": "user",
                        "content": f'Translate this Japanese subtitle line to natural English:\n\n"{original_text}"'
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )

            translated_text = response.choices[0].message.content.strip()
            # Remove quotes if GPT added them
            if translated_text.startswith('"') and translated_text.endswith('"'):
                translated_text = translated_text[1:-1]

            return {
                "start": segment["start"],
                "end": segment["end"],
                "text": translated_text,
                "original": original_text  # Keep original for debugging
            }

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(backoff_times[attempt])
            else:
                logger.warning(f"Translation failed for segment, using original: {e}")
                return segment

    return segment


def translate_segments(segments: List[Dict[str, any]], workers: int = None) -> List[Dict[str, any]]:
    """
    Translate all segments using GPT-4 in parallel.

    Args:
        segments: List of segments with original language text
        workers: Number of parallel workers (overrides config if provided)

    Returns:
        List of segments with translated English text
    """
    if workers is None:
        workers = WORKERS

    logger.info(f"Starting parallel translation with {workers} workers")
    logger.info(f"Using model: {GPT_MODEL}")

    translated_segments = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all segments for translation
        future_to_segment = {
            executor.submit(translate_segment, segment): segment
            for segment in segments
        }

        # Collect results with progress bar
        with tqdm(total=len(segments), desc="Translating", unit="segment") as pbar:
            for future in as_completed(future_to_segment):
                try:
                    translated = future.result()
                    translated_segments.append(translated)
                except Exception as e:
                    segment = future_to_segment[future]
                    logger.error(f"Unexpected error translating segment: {e}")
                    translated_segments.append(segment)  # Keep original on error
                pbar.update(1)

    # Sort by start time
    translated_segments.sort(key=lambda x: x["start"])

    logger.info(f"Translation complete. Total segments: {len(translated_segments)}")

    return translated_segments


def correct_translations(segments: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Correct and improve translations using GPT-4 with batched context.

    Args:
        segments: List of translated segments

    Returns:
        List of corrected segments
    """
    logger.info(f"Starting translation correction (batch size: {CORRECTION_BATCH_SIZE})")

    corrected_segments = []

    # Process in batches
    for i in tqdm(range(0, len(segments), CORRECTION_BATCH_SIZE), desc="Correcting", unit="batch"):
        batch = segments[i:i + CORRECTION_BATCH_SIZE]

        # Create batch text
        batch_text = "\n".join([f"{idx+1}. {seg['text']}" for idx, seg in enumerate(batch)])

        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional subtitle editor. Clean up the following English subtitle lines "
                                   "so they are natural, accurate, and preserve the emotional meaning of the original Japanese. "
                                   "Fix mistranslations, unnatural phrasing, and missing nuance. Do not censor content. "
                                   "Return ONLY the corrected lines in the same numbered format."
                    },
                    {
                        "role": "user",
                        "content": f"Clean up these subtitle lines:\n\n{batch_text}"
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )

            corrected_text = response.choices[0].message.content.strip()

            # Parse corrected lines
            corrected_lines = []
            for line in corrected_text.split("\n"):
                line = line.strip()
                if line and ". " in line:
                    # Remove number prefix
                    text = line.split(". ", 1)[1] if ". " in line else line
                    corrected_lines.append(text)

            # Update segments with corrected text
            for idx, seg in enumerate(batch):
                if idx < len(corrected_lines):
                    seg["text"] = corrected_lines[idx]
                corrected_segments.append(seg)

        except Exception as e:
            logger.warning(f"Correction failed for batch, using uncorrected: {e}")
            corrected_segments.extend(batch)

    logger.info("Correction complete")

    return corrected_segments
