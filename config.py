"""
Configuration module for Whisper Subtitle Generator.
Loads environment variables and creates OpenAI client singleton.
"""

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Suppress verbose HTTP request logs from OpenAI/httpx (only show warnings/errors)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment. "
        "Please add it to your .env file."
    )

# Get worker count (default to 4 if not specified)
WORKERS = int(os.getenv("WORKERS", "4"))

# Translation worker count (separate from transcription workers)
# Lower to avoid GPT-4 rate limits
TRANSLATION_WORKERS = int(os.getenv("TRANSLATION_WORKERS", "4"))

# GPT translation settings
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")  # Use gpt-4o for best quality
ENABLE_CORRECTION = os.getenv("ENABLE_CORRECTION", "false").lower() == "true"
TRANSLATION_BATCH_MINUTES = int(os.getenv("TRANSLATION_BATCH_MINUTES", "5"))  # Group segments by time window
MAX_SEGMENTS_PER_BATCH = int(os.getenv("MAX_SEGMENTS_PER_BATCH", "60"))  # Split if batch exceeds this

# Translation provider settings
BULK_TRANSLATOR = os.getenv("BULK_TRANSLATOR", "google")  # "openai" or "google"
FALLBACK_CHAIN = os.getenv("FALLBACK_CHAIN", "google,openai,untranslated").split(",")  # Fallback order
GOOGLE_BUNDLE_SIZE = int(os.getenv("GOOGLE_BUNDLE_SIZE", "100"))  # Lines per Google API call

# Create singleton OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Audio chunk duration in milliseconds (80 seconds)
CHUNK_DURATION_MS = 80000

# Audio settings for extraction
AUDIO_SETTINGS = {
    "sample_rate": 16000,  # 16 kHz
    "channels": 1,          # mono
    "codec": "pcm_s16le"    # PCM 16-bit
}
