# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based subtitle generator that uses OpenAI's Whisper API for transcription and GPT-4 for translation. The system processes videos through a multi-step pipeline with parallel processing for efficiency.

## Development Commands

### Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Application
```bash
# Basic usage (transcribe + translate with GPT-4)
python main.py test_file.mp4

# With custom worker count
python main.py test_file.mp4 --workers 8

# With GPT-4 correction pass for best quality
python main.py test_file.mp4 --with-correction

# Skip correction for faster processing
python main.py test_file.mp4 --no-correction
```

## Architecture

### Processing Pipeline
The application follows a sequential pipeline coordinated by `main.py`:

**Without Correction (4 steps - default):**
1. **Audio Extraction**: Video → WAV (PCM 16-bit, mono, 16 kHz) using ffmpeg
2. **Audio Chunking**: WAV → 80-second chunks using ffprobe/ffmpeg
3. **Parallel Transcription**: Chunks → Japanese text via Whisper API
4. **GPT-4 Translation**: Japanese text → English translation (parallel)
5. **SRT Generation**: Merge translated segments with timestamps

**With Correction (5 steps):**
- Same as above, but adds **GPT-4 Correction** step after translation
- Processes 30-line batches for context-aware improvements

### Key Benefits of Two-Stage Approach

✅ **Better Japanese transcription** - Whisper optimized for Japanese when not translating
✅ **Better translation quality** - GPT-4 understands nuance, slang, context
✅ **No censoring** - GPT-4 preserves NSFW content and emotional tone  
✅ **Fewer hallucinations** - GPT-4 correction step fixes Whisper mishears
✅ **Still cheap** - Translation costs minimal compared to Whisper

### Cost Analysis (1-hour video):
- Old: $0.36 (Whisper direct translation)
- New: ~$0.37 (3% increase for massive quality gain)
- With correction: ~$0.38 (5% increase)
