# Whisper Subtitle Generator

Automatically generate English subtitles from video files using OpenAI's Whisper API or local Whisper models with parallel processing.

## Features

- **Dual Transcription Modes** - Use OpenAI Whisper API or local faster-whisper (offline, no API costs)
- **Folder Processing** - Process entire directories of videos with automatic language detection
- **Smart Skip Logic** - Metadata tracking to skip already-processed or problematic files
- **Multiple Translation Backends** - GPT-4, Google Translate, or Whisper's built-in translation
- **Progress Bars** - Real-time progress indicators for each processing step
- **Parallel Processing** - Process multiple audio chunks simultaneously
- **Corrupt File Handling** - Automatically detect and skip videos with corrupt audio streams
- **Japanese Subtitle Caching** - Skip re-transcription on subsequent runs
- **Robust Error Handling** - Retry logic with exponential backoff for API failures
- **Multi-hour Video Support** - Handles videos of any length

## Prerequisites

1. **Python 3.8+**
2. **ffmpeg** - Install from [ffmpeg.org](https://ffmpeg.org/download.html)
3. **OpenAI API Key** - Required for API mode (optional for local Whisper with Google Translate)

## Installation

### Quick Setup (Recommended)

```bash
# Clone the repository and run setup script
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Check Python version (3.8+ required)
- Check/prompt for ffmpeg installation
- Create virtual environment
- Install all dependencies
- Optionally install PyTorch for GPU acceleration
- Create `.env` template file

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch for GPU acceleration
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Process a single video (local Whisper + Google Translate)
python main.py video.mp4 --local-whisper

# Process a folder of videos
python main.py /path/to/videos/ --local-whisper --skip-existing

# Local Whisper with direct English translation (fastest, no translation step)
python main.py video.mp4 --local-whisper --direct-whisper
```

## CLI Options

| Option | Description |
|--------|-------------|
| `path` | Path to video file or folder (required) |
| `-w, --workers N` | Number of parallel workers for transcription (default: 4) |
| `--local-whisper` | Use local faster-whisper model instead of OpenAI API |
| `--model SIZE` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` (default: medium) |
| `--device DEVICE` | Device for local Whisper: `auto`, `cuda`, `cpu` (default: auto) |
| `--beam-size N` | Beam size for local Whisper decoding (default: 5) |
| `--direct-whisper` | Use Whisper's built-in translation to English (skip GPT-4/Google) |
| `--bulk-translator` | Translation method: `openai` or `google` (default: google) |
| `--fallback-chain` | Comma-separated fallback order (default: `google,openai,untranslated`) |
| `--with-correction` | Enable GPT-4 correction pass for higher quality |
| `--no-correction` | Disable GPT-4 correction pass |
| `--skip-existing` | Skip videos that already have .srt files |
| `--no-prefetch` | Disable background prefetching of next video |
| `--force-transcribe` | Force re-transcription even if cached Japanese subtitles exist |

## Usage Examples

### Single Video Processing

```bash
# Basic usage with local Whisper
python main.py video.mp4 --local-whisper

# Use large model for best quality (requires ~10GB VRAM)
python main.py video.mp4 --local-whisper --model large-v3

# Force CPU usage
python main.py video.mp4 --local-whisper --device cpu

# Direct translation (fastest, skip separate translation step)
python main.py video.mp4 --local-whisper --direct-whisper
```

### Folder Processing

```bash
# Process all videos in a folder
python main.py /path/to/videos/ --local-whisper

# Skip already processed videos (have .srt files)
python main.py /path/to/videos/ --local-whisper --skip-existing

# High quality with large model
python main.py /path/to/videos/ --local-whisper --model large-v3 --beam-size 5
```

### Translation Options

```bash
# Use Google Translate (free, fast) - default
python main.py video.mp4 --local-whisper --bulk-translator google

# Use OpenAI GPT-4 for translation (higher quality, costs money)
python main.py video.mp4 --bulk-translator openai

# Custom fallback chain
python main.py video.mp4 --fallback-chain "openai,google,untranslated"
```

### API Mode (OpenAI Whisper)

```bash
# Using OpenAI Whisper API (requires API key in .env)
python main.py video.mp4

# With GPT-4 correction pass
python main.py video.mp4 --with-correction
```

## Folder Processing Features

When processing a folder, the tool provides:

### Automatic Language Detection
- Detects source language before processing
- Skips English videos automatically
- Supports any non-English source language

### Smart Skip Logic
Each video gets a `.subscene.json` metadata file tracking:
- Detected language and confidence
- Processing status
- Error information (for corrupt files)

Files are automatically skipped if:
- Already processed (has .srt file and marked processed)
- Detected as English
- Previously failed with corrupt audio
- Has `--skip-existing` flag and .srt exists

### Processing Summary
After folder processing, you'll see a summary:
```
Folder Processing Complete
============================================================
Total files found: 100
  - Processed successfully: 85
  - Skipped (already English): 5
  - Skipped (already processed): 7
  - Skipped (SRT exists, --skip-existing): 0
  - Skipped (previous error): 2
  - Failed: 1
```

### Corrupt File Handling
Videos with corrupt audio streams are:
- Detected automatically during processing
- Marked in metadata to skip on future runs
- Logged with error details

## Configuration

Create a `.env` file in the project root (or run `setup.sh` to generate template):

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

### Standard Mode (Source Language → English)
1. **Extract Audio** - Video → WAV (PCM 16-bit, mono, 16 kHz)
2. **Language Detection** - Detect source language (folder mode)
3. **Transcribe** - Audio → Source language text (Whisper API or local)
4. **Translate** - Source → English (GPT-4 or Google Translate)
5. **Correction** (optional) - GPT-4 cleanup pass
6. **Generate SRT** - Create subtitle file

### Direct Whisper Mode
1. **Extract Audio** - Video → WAV
2. **Transcribe + Translate** - Audio → English text (Whisper's built-in translation)
3. **Generate SRT** - Create subtitle file

## Output

Subtitle files are saved in the same directory as the input video:

| Input | Output |
|-------|--------|
| `video.mp4` | `video.srt` (English), `video.ja.srt` (Japanese cache) |
| `/path/to/movie.mp4` | `/path/to/movie.srt`, `/path/to/movie.mp4.subscene.json` |

### Metadata Files

For folder processing, each video gets a `.subscene.json` metadata file:
```json
{
  "version": 1,
  "video_file": "video.mp4",
  "detected_language": "ja",
  "language_confidence": 0.95,
  "processed": true,
  "processed_at": "2025-12-04T12:00:00Z",
  "srt_file": "video.srt"
}
```

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
├── setup.sh                  # Automated setup script
├── requirements.txt          # Python dependencies
└── .env                      # API keys & settings
```

## Error Handling

- **Rate Limits**: Automatic retry with exponential backoff (1s → 2s → 4s)
- **Failed Chunks**: Logged but processing continues
- **Translation Failures**: Falls back through the fallback chain
- **Corrupt Audio**: Detected, logged, and skipped on future runs
- **Missing ffmpeg**: Clear error message with installation instructions

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows - Download from https://ffmpeg.org/download.html
```

### CUDA/GPU not detected
1. Ensure NVIDIA drivers are installed
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

### Out of memory with large model
Use a smaller model or force CPU:
```bash
python main.py video.mp4 --local-whisper --model medium --device cpu
```

## License

This project is provided as-is for educational and personal use.
