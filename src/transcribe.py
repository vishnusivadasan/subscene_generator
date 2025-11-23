"""
Parallel transcription module using OpenAI Whisper API.
Handles concurrent API calls with retry logic and rate limit handling.
"""

import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import logger
from config import client, WORKERS, GPT_MODEL, ENABLE_CORRECTION, TRANSLATION_BATCH_MINUTES, MAX_SEGMENTS_PER_BATCH


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


def transcribe_audio(chunks_generator, total_chunks: int = None, workers: int = None) -> List[Dict[str, any]]:
    """
    Transcribe audio chunks in parallel using ThreadPoolExecutor.
    Accepts chunks from a generator and starts transcription immediately.

    Args:
        chunks_generator: Generator that yields chunk information dictionaries
        total_chunks: Total number of chunks for progress tracking (optional)
        workers: Number of parallel workers (overrides config if provided)

    Returns:
        List of all transcription segments sorted by start time
    """
    # Determine worker count
    if workers is None:
        workers = WORKERS

    logger.info(f"Starting concurrent chunking and transcription with {workers} workers")

    all_segments = []
    chunks_info = []  # Keep track of chunks for cleanup

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_chunk = {}

        # Progress bar (update total as chunks arrive if not provided)
        pbar = tqdm(desc="Transcribing", unit="chunk", total=total_chunks)

        # Submit chunks as they arrive from generator
        for chunk in chunks_generator:
            chunks_info.append(chunk)
            future = executor.submit(process_chunk, chunk)
            future_to_chunk[future] = chunk

            # Update total if we didn't know it upfront
            if total_chunks is None:
                pbar.total = len(chunks_info)
                pbar.refresh()

        # All chunks submitted, now collect results as they complete
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

        pbar.close()

    # Sort all segments by start time
    all_segments.sort(key=lambda x: x["start"])

    logger.info(f"Transcription complete. Total segments: {len(all_segments)}")

    return all_segments, chunks_info


def translate_batch(batch: List[Dict[str, any]], batch_num: int) -> List[Dict[str, any]]:
    """
    Translate a batch of segments using GPT-4 with full context.

    Args:
        batch: List of segments within a time window
        batch_num: Batch number for logging

    Returns:
        List of translated segments
    """
    max_retries = 3
    backoff_times = [1, 2, 4]

    # Format batch for GPT
    batch_text = []
    for idx, seg in enumerate(batch, 1):
        # Include timestamp for context
        timestamp = f"[{int(seg['start']//60):02d}:{int(seg['start']%60):02d}]"
        batch_text.append(f"{idx}. {timestamp} {seg['text']}")

    batch_input = "\n".join(batch_text)

    logger.debug(f"Batch {batch_num}: Translating {len(batch)} segments")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional subtitle translator. Translate Japanese subtitle lines to natural English.\n\n"
                                   "CRITICAL RULES:\n"
                                   "1. Return EXACTLY the same number of lines as the input\n"
                                   "2. Use the EXACT same numbered format: '1. ', '2. ', '3. ', etc.\n"
                                   "3. Do NOT skip any numbers\n"
                                   "4. Do NOT add commentary or explanations\n"
                                   "5. Preserve conversational flow, tone, nuance, and emotion\n"
                                   "6. Do NOT censor or soften meaning - translate all sexual, vulgar, and explicit language directly\n"
                                   "7. Do NOT use euphemisms - preserve crude/vulgar terms exactly as they appear in context\n\n"
                                   "Example format:\n"
                                   "Input:\n"
                                   "1. [00:05] こんにちは\n"
                                   "2. [00:08] 元気ですか\n\n"
                                   "Output:\n"
                                   "1. Hello\n"
                                   "2. How are you"
                    },
                    {
                        "role": "user",
                        "content": f"Translate these {len(batch)} Japanese subtitles to English:\n\n{batch_input}"
                    }
                ],
                temperature=0.3,
                max_tokens=4000
            )

            translated_text = response.choices[0].message.content.strip()

            # Check if response is empty or too short
            if not translated_text or len(translated_text) < 10:
                if attempt < max_retries - 1:
                    logger.warning(f"Batch {batch_num} attempt {attempt + 1}: GPT-4 returned empty/short response. Retrying...")
                    time.sleep(backoff_times[attempt])
                    continue
                else:
                    logger.error(f"Batch {batch_num}: GPT-4 returned empty response after {max_retries} attempts. Using originals.")
                    return batch

            # Multi-strategy parsing with fallbacks
            translated_lines = []

            # Strategy 1: Parse numbered format (preferred)
            for line in translated_text.split("\n"):
                line = line.strip()
                if line and ". " in line:
                    # Try to extract number and text
                    parts = line.split(". ", 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        translated_lines.append(parts[1].strip())

            # Strategy 2: If count mismatch, try aggressive line extraction
            if len(translated_lines) != len(batch):
                logger.warning(f"Batch {batch_num} attempt {attempt + 1}: Expected {len(batch)} lines, got {len(translated_lines)}. Using fallback parsing.")
                logger.debug(f"Batch {batch_num} raw response preview: {translated_text[:500]}...")

                translated_lines = []
                for line in translated_text.split("\n"):
                    line = line.strip()
                    # Skip empty lines, headers, and metadata
                    if not line or line.startswith("#") or line.lower().startswith("output") or line.lower().startswith("input"):
                        continue

                    # Remove timestamp markers like [00:05]
                    if "[" in line and "]" in line:
                        # Find last ] and take text after it
                        bracket_end = line.rfind("]")
                        if bracket_end != -1:
                            line = line[bracket_end + 1:].strip()

                    # Remove any number prefix if exists
                    if ". " in line:
                        parts = line.split(". ", 1)
                        if parts[0].strip().isdigit():
                            line = parts[1].strip()

                    if line:
                        translated_lines.append(line)

            # Strategy 3: If still severely mismatched and we have retries left, retry
            if len(translated_lines) < len(batch) * 0.5 and attempt < max_retries - 1:
                logger.warning(f"Batch {batch_num} attempt {attempt + 1}: Only got {len(translated_lines)}/{len(batch)} lines (< 50%). Retrying...")
                time.sleep(backoff_times[attempt])
                continue

            # Strategy 4: Truncate or pad if minor mismatch
            if len(translated_lines) < len(batch):
                logger.warning(f"Batch {batch_num}: Still short ({len(translated_lines)}/{len(batch)}). Padding with originals.")
                # Pad with original text for missing lines
                while len(translated_lines) < len(batch):
                    translated_lines.append(batch[len(translated_lines)]["text"])
            elif len(translated_lines) > len(batch):
                logger.warning(f"Batch {batch_num}: Too many lines ({len(translated_lines)}/{len(batch)}). Truncating.")
                translated_lines = translated_lines[:len(batch)]

            # Map translations back to segments
            result = []
            for idx, seg in enumerate(batch):
                result.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": translated_lines[idx],
                    "original": seg["text"]
                })

            return result

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Batch {batch_num} attempt {attempt + 1}: Exception occurred: {e}. Retrying...")
                time.sleep(backoff_times[attempt])
            else:
                logger.error(f"Batch {batch_num}: Translation failed after {max_retries} attempts: {e}. Using originals.")
                return batch

    return batch


def translate_segments(segments: List[Dict[str, any]], workers: int = None) -> List[Dict[str, any]]:
    """
    Translate all segments using GPT-4 with time-based batching for better context.

    Args:
        segments: List of segments with original language text
        workers: Number of parallel workers (overrides config if provided)

    Returns:
        List of segments with translated English text
    """
    if workers is None:
        workers = WORKERS

    # Create time-based batches
    batch_window_seconds = TRANSLATION_BATCH_MINUTES * 60
    batches = []
    current_batch = []
    batch_start_time = 0

    for segment in segments:
        # Check if we need to start a new batch
        if segment["start"] >= batch_start_time + batch_window_seconds and current_batch:
            batches.append(current_batch)
            current_batch = [segment]
            batch_start_time = segment["start"]
        else:
            if not current_batch:
                batch_start_time = segment["start"]
            current_batch.append(segment)

    # Add the last batch
    if current_batch:
        batches.append(current_batch)

    # Split oversized batches (dialogue-heavy scenes)
    final_batches = []
    for batch in batches:
        if len(batch) > MAX_SEGMENTS_PER_BATCH:
            # Split into smaller batches
            for i in range(0, len(batch), MAX_SEGMENTS_PER_BATCH):
                final_batches.append(batch[i:i + MAX_SEGMENTS_PER_BATCH])
        else:
            final_batches.append(batch)

    logger.info(f"Starting parallel translation with {workers} workers")
    logger.info(f"Using model: {GPT_MODEL}")
    logger.info(f"Batching: {len(final_batches)} batches ({TRANSLATION_BATCH_MINUTES}-minute windows, max {MAX_SEGMENTS_PER_BATCH} segments/batch)")

    batches = final_batches

    all_translated = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all batches for translation
        future_to_batch = {
            executor.submit(translate_batch, batch, idx): (batch, idx)
            for idx, batch in enumerate(batches, 1)
        }

        # Collect results with progress bar
        with tqdm(total=len(batches), desc="Translating", unit="batch") as pbar:
            for future in as_completed(future_to_batch):
                try:
                    translated_batch = future.result()
                    all_translated.extend(translated_batch)
                except Exception as e:
                    batch, batch_num = future_to_batch[future]
                    logger.error(f"Unexpected error translating batch {batch_num}: {e}")
                    all_translated.extend(batch)  # Keep originals on error
                pbar.update(1)

    # Sort by start time
    all_translated.sort(key=lambda x: x["start"])

    logger.info(f"Translation complete. Total segments: {len(all_translated)}")

    return all_translated


def correct_translations(segments: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Correct and improve translations using GPT-4 with time-based batching for context.

    Args:
        segments: List of translated segments

    Returns:
        List of corrected segments
    """
    # Use same batching strategy as translation (time-based windows)
    batch_window_seconds = TRANSLATION_BATCH_MINUTES * 60
    batches = []
    current_batch = []
    batch_start_time = 0

    for segment in segments:
        if segment["start"] >= batch_start_time + batch_window_seconds and current_batch:
            batches.append(current_batch)
            current_batch = [segment]
            batch_start_time = segment["start"]
        else:
            if not current_batch:
                batch_start_time = segment["start"]
            current_batch.append(segment)

    if current_batch:
        batches.append(current_batch)

    logger.info(f"Starting translation correction ({len(batches)} batches, {TRANSLATION_BATCH_MINUTES}-minute windows)")

    corrected_segments = []

    # Process batches sequentially with progress bar
    for batch_num, batch in enumerate(tqdm(batches, desc="Correcting", unit="batch"), 1):
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
                                   "Fix mistranslations, unnatural phrasing, and missing nuance. "
                                   "Maintain conversational flow and context between lines. "
                                   "Do NOT censor content - preserve all sexual, vulgar, and explicit language exactly. "
                                   "Do NOT soften crude or vulgar terms - maintain them as-is. "
                                   "Return ONLY the corrected lines in the same numbered format."
                    },
                    {
                        "role": "user",
                        "content": f"Clean up these subtitle lines:\n\n{batch_text}"
                    }
                ],
                temperature=0.3,
                max_tokens=4000
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
            logger.warning(f"Correction failed for batch {batch_num}, using uncorrected: {e}")
            corrected_segments.extend(batch)

    logger.info("Correction complete")

    return corrected_segments
