"""
Parallel transcription module using OpenAI Whisper API.
Handles concurrent API calls with retry logic and rate limit handling.
"""

import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import logger
from config import client, WORKERS


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

    logger.info(f"[Worker] Processing chunk offset {offset_seconds:.2f}s")

    max_retries = 3
    backoff_times = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s

    for attempt in range(max_retries):
        try:
            # Open audio file and send to Whisper API for translation to English
            with open(chunk_path, "rb") as audio_file:
                response = client.audio.translations.create(
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

            logger.info(f"[Worker] Completed chunk offset {offset_seconds:.2f}s")
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

        # Collect results as they complete
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

    # Sort all segments by start time
    all_segments.sort(key=lambda x: x["start"])

    logger.info(f"Transcription complete. Total segments: {len(all_segments)}")

    return all_segments
