"""
Google Translate module for subtitle translation.
Uses deep-translator library for free translation with better compatibility.
"""

import time
from typing import List, Dict
from deep_translator import GoogleTranslator
from utils import logger
from config import GOOGLE_BUNDLE_SIZE


def translate_batch_google(batch: List[Dict[str, any]], batch_num: int) -> List[Dict[str, any]]:
    """
    Translate a batch of segments using Google Translate with bundling strategy.
    Bundles lines together to reduce API calls while avoiding rate limits.

    Args:
        batch: List of segments with 'start', 'end', 'text' keys
        batch_num: Batch number for logging

    Returns:
        List of translated segments with 'original' field added
    """
    translator = GoogleTranslator(source='ja', target='en')
    translated_segments = []
    bundle_size = GOOGLE_BUNDLE_SIZE

    logger.info(f"  Batch {batch_num}: Translating {len(batch)} segments (bundling {bundle_size} lines per API call)")

    # Process batch in bundles
    for i in range(0, len(batch), bundle_size):
        bundle = batch[i:i + bundle_size]

        # Bundle lines with newline separator
        bundled_text = "\n".join([seg["text"] for seg in bundle])

        try:
            # Translate bundled text
            result = translator.translate(bundled_text)
            translations = result.split('\n')

            # Verify we got the right number of translations
            if len(translations) != len(bundle):
                logger.warning(f"  Batch {batch_num}: Bundle translation line count mismatch "
                             f"(expected {len(bundle)}, got {len(translations)}). Falling back to line-by-line.")

                # Fallback to line-by-line for this bundle
                for seg in bundle:
                    try:
                        result = translator.translate(seg["text"])
                        translated_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": result,
                            "original": seg["text"]
                        })
                    except Exception as e:
                        logger.error(f"  Batch {batch_num}: Error translating line: {e}")
                        # Keep original Japanese text on error
                        translated_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"],
                            "original": seg["text"]
                        })
            else:
                # Successful bundle translation
                for seg, translation in zip(bundle, translations):
                    translated_segments.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": translation.strip(),
                        "original": seg["text"]
                    })

            # Small delay to avoid rate limiting
            time.sleep(0.2)

        except Exception as e:
            logger.error(f"  Batch {batch_num}: Error in bundle translation: {e}")

            # Fallback to line-by-line for this bundle
            for seg in bundle:
                try:
                    result = translator.translate(seg["text"])
                    translated_segments.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": result,
                        "original": seg["text"]
                    })
                    time.sleep(0.2)
                except Exception as line_error:
                    logger.error(f"  Batch {batch_num}: Error translating line: {line_error}")
                    # Keep original Japanese text on error
                    translated_segments.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                        "original": seg["text"]
                    })

    logger.info(f"  Batch {batch_num}: Translated {len(translated_segments)} segments successfully")
    return translated_segments


def translate_single_line_google(segment: Dict[str, any], line_num: int) -> Dict[str, any]:
    """
    Translate a single segment using Google Translate.
    Used as fallback when batch translation fails.

    Args:
        segment: Single segment with 'start', 'end', 'text' keys
        line_num: Line number for logging

    Returns:
        Translated segment with 'original' field added
    """
    translator = GoogleTranslator(source='ja', target='en')

    try:
        result = translator.translate(segment["text"])
        return {
            "start": segment["start"],
            "end": segment["end"],
            "text": result,
            "original": segment["text"]
        }
    except Exception as e:
        logger.error(f"  Line {line_num}: Google Translate error: {e}")
        # Return original on error (will trigger next fallback in chain)
        raise
