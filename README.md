# Whisper Subtitle Generator

Automatically generate English subtitles from video files using OpenAI's Whisper API or local Whisper models with parallel processing.

## Features

- **Dual Transcription Modes** - Use OpenAI Whisper API or local faster-whisper (offline, no API costs)
- **Parallel Processing** - Process multiple audio chunks simultaneously
- **Multiple Translation Backends** - GPT-4, Google Translate, or Whisper's built-in translation
- **Progress Bars** - Real-time progress indicators for each processing step
- **Japanese Subtitle Caching** - Skip re-transcription on subsequent runs
- **Robust Error Handling** - Retry logic with exponential backoff for API failures
- **Multi-hour Video Support** - Handles videos of any length
- **Accurate Timing** - Precise timestamp alignment across chunks

## Prerequisites

1. **Python 3.7+**
2. **ffmpeg** - Install from [ffmpeg.org](https://ffmpeg.org/download.html)
3. **OpenAI API Key** - Required for API mode (optional for local Whisper)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Using OpenAI Whisper API (requires API key)
python main.py video.mp4

# Using local Whisper (no API key needed for transcription)
python main.py video.mp4 --local-whisper

# Local Whisper with direct English translation (fastest, no translation step)
python main.py video.mp4 --local-whisper --direct-whisper
```

## CLI Options

| Option | Description |
|--------|-------------|
| `video_path` | Path to the video file (required) |
| `-w, --workers N` | Number of parallel workers for transcription (default: 4) |
| `--local-whisper` | Use local faster-whisper model instead of OpenAI API |
| `--model SIZE` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` (default: medium) |
| `--device DEVICE` | Device for local Whisper: `auto`, `cuda`, `cpu` (default: auto) |
| `--direct-whisper` | Use Whisper's built-in translation to English (skip GPT-4/Google) |
| `--bulk-translator` | Translation method: `openai` or `google` (default: google) |
| `--fallback-chain` | Comma-separated fallback order (default: `google,openai,untranslated`) |
| `--with-correction` | Enable GPT-4 correction pass for higher quality |
| `--no-correction` | Disable GPT-4 correction pass |
| `--force-transcribe` | Force re-transcription even if cached Japanese subtitles exist |

## Usage Examples

### Basic Usage (API Mode)
```bash
python main.py test_file.mp4
```

### Local Whisper (Offline Mode)
```bash
# Auto-detect GPU/CPU, use medium model
python main.py video.mp4 --local-whisper

# Use small model for faster processing
python main.py video.mp4 --local-whisper --model small

# Force CPU usage
python main.py video.mp4 --local-whisper --device cpu

# Use large model for best quality (requires ~3GB VRAM)
python main.py video.mp4 --local-whisper --model large-v3
```

### Direct Translation (Skip GPT-4/Google)
```bash
# API mode - use Whisper API's built-in translation
python main.py video.mp4 --direct-whisper

# Local mode - use local Whisper's built-in translation
python main.py video.mp4 --local-whisper --direct-whisper
```

### Translation Options
```bash
# Use OpenAI GPT-4 for translation (higher quality, costs money)
python main.py video.mp4 --bulk-translator openai

# Use Google Translate (free, fast)
python main.py video.mp4 --bulk-translator google

# Custom fallback chain
python main.py video.mp4 --fallback-chain "openai,google,untranslated"
```

### Quality Options
```bash
# Enable GPT-4 correction pass (best quality)
python main.py video.mp4 --with-correction

# Disable correction (faster)
python main.py video.mp4 --no-correction
```

### Performance Tuning
```bash
# More workers for faster transcription (API mode)
python main.py video.mp4 --workers 8

# Force re-transcription (ignore cache)
python main.py video.mp4 --force-transcribe
```

## Configuration

Create a `.env` file in the project root:

```bash
# OpenAI API Key (required for API mode, optional for local Whisper)
OPENAI_API_KEY=sk-your-key-here

# Transcription Settings
WORKERS=4                          # Parallel workers for API transcription

# Translation Settings
TRANSLATION_WORKERS=4              # Parallel workers for translation
BULK_TRANSLATOR=google             # Default: "google" or "openai"
FALLBACK_CHAIN=google,openai,untranslated  # Fallback order for failed translations
GPT_MODEL=gpt-4o-mini              # GPT model for translation/correction

# Batching Settings
TRANSLATION_BATCH_MINUTES=5        # Group segments by time window
MAX_SEGMENTS_PER_BATCH=60          # Max segments per translation batch
GOOGLE_BUNDLE_SIZE=100             # Lines per Google API call

# Correction Settings
ENABLE_CORRECTION=false            # Enable GPT-4 correction by default

# Local Whisper Settings
LOCAL_WHISPER_MODEL=medium         # Default model: tiny, base, small, medium, large-v3
LOCAL_WHISPER_DEVICE=auto          # Device: auto, cuda, cpu
```

## Local Whisper Model Sizes

| Model | Parameters | VRAM | Relative Speed | Quality |
|-------|-----------|------|----------------|---------|
| `tiny` | 39M | ~1GB | Fastest | Low |
| `base` | 74M | ~1GB | Fast | Fair |
| `small` | 244M | ~2GB | Medium | Good |
| `medium` | 769M | ~5GB | Slow | Great |
| `large-v3` | 1550M | ~10GB | Slowest | Best |

## Processing Pipeline

### Standard Mode (Japanese → English)
1. **Extract Audio** - Video → WAV (PCM 16-bit, mono, 16 kHz)
2. **Transcribe** - Audio → Japanese text (Whisper API or local)
3. **Translate** - Japanese → English (GPT-4 or Google Translate)
4. **Correction** (optional) - GPT-4 cleanup pass
5. **Generate SRT** - Create subtitle file

### Direct Whisper Mode
1. **Extract Audio** - Video → WAV
2. **Transcribe + Translate** - Audio → English text (Whisper's built-in translation)
3. **Generate SRT** - Create subtitle file

## Output

Subtitle files are saved in the same directory as the input video:

| Input | Output |
|-------|--------|
| `video.mp4` | `video.srt` (English), `video.ja.srt` (Japanese cache) |
| `/path/to/movie.mp4` | `/path/to/movie.srt` |

## Project Structure

```
subscene_generator/
├── src/
│   ├── extract_audio.py      # Audio extraction using ffmpeg
│   ├── chunk_audio.py        # Split audio into 80-second chunks
│   ├── transcribe.py         # OpenAI Whisper API transcription
│   ├── transcribe_local.py   # Local faster-whisper transcription
│   ├── translate_google.py   # Google Translate integration
│   └── merge_srt.py          # SRT file generation
├── main.py                   # CLI entry point
├── config.py                 # Configuration & settings
├── utils.py                  # Helper utilities
├── requirements.txt          # Python dependencies
└── .env                      # API keys & settings
```

## Error Handling

- **Rate Limits**: Automatic retry with exponential backoff (1s → 2s → 4s)
- **Failed Chunks**: Logged but processing continues
- **Translation Failures**: Falls back through the fallback chain
- **Missing ffmpeg**: Clear error message with installation instructions

## License

This project is provided as-is for educational and personal use.
