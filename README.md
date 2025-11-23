# Whisper Subtitle Generator

Automatically generate English subtitles from video files using OpenAI's Whisper API with parallel processing.

## Features

✅ **Parallel Processing** - Process multiple audio chunks simultaneously (default 4 workers)
✅ **Progress Bars** - Real-time progress indicators for each processing step
✅ **Automatic Translation** - Detects source language and translates to English
✅ **Robust Error Handling** - Retry logic with exponential backoff for API failures
✅ **Multi-hour Video Support** - Handles videos of any length
✅ **Accurate Timing** - Precise timestamp alignment across chunks
✅ **Clean SRT Output** - Standard subtitle format compatible with all video players

## Prerequisites

1. **Python 3.7+**
2. **ffmpeg** - Install from [ffmpeg.org](https://ffmpeg.org/download.html)
3. **OpenAI API Key** - Already configured in your `.env` file ✓

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py test_file.mp4
```

### With Custom Video Path

```bash
python main.py input_videos/myvideo.mp4
```

### With Custom Worker Count

```bash
python main.py test_file.mp4 --workers 8
```

### Paths with Spaces

```bash
python main.py "/path/with spaces/video.mp4"
```

## Project Structure

```
subtitle_maker/
├── src/
│   ├── extract_audio.py     # Audio extraction using ffmpeg
│   ├── chunk_audio.py       # Split audio into 80-second chunks
│   ├── transcribe.py        # Parallel Whisper API processing
│   └── merge_srt.py         # SRT file generation
├── input_videos/            # Place your videos here (optional)
├── audio/                   # Temporary audio files (auto-cleaned)
├── main.py                  # CLI entry point
├── config.py                # Configuration & OpenAI client
├── utils.py                 # Helper utilities
├── requirements.txt         # Python dependencies
└── .env                     # API key (already configured)
```

## How It Works

1. **Extract Audio** - Converts video to WAV (PCM 16-bit, mono, 16 kHz)
2. **Chunk Audio** - Splits into 80-second segments for parallel processing
3. **Transcribe** - Sends chunks to Whisper API in parallel with retry logic
4. **Merge** - Combines segments with correct timestamps into SRT file (saved next to video)
5. **Cleanup** - Removes temporary files automatically

## Configuration

Edit `.env` to customize settings:

```bash
OPENAI_API_KEY=your-key-here    # Required
WORKERS=4                        # Optional: Number of parallel workers
```

## Output

Subtitle files are saved in the same directory as the input video with a `.srt` extension.

Examples:
- Input: `test_file.mp4` → Output: `test_file.srt` (same directory)
- Input: `/home/user/Videos/movie.mp4` → Output: `/home/user/Videos/movie.srt`
- Input: `input_videos/myvideo.mp4` → Output: `input_videos/myvideo.srt`

## Error Handling

- **Rate Limits**: Automatic retry with exponential backoff (1s → 2s → 4s)
- **Failed Chunks**: Logged but processing continues (no crash)
- **Missing ffmpeg**: Clear error message with installation instructions

## Example Output

```
============================================================
Whisper Subtitle Generator
============================================================
[INFO] Input video: /path/to/test_file.mp4

[1/4] Extracting audio...
[INFO] Audio extracted successfully: audio/test_file.wav

[2/4] Chunking audio...
[INFO] Audio duration: 180.50 seconds
Creating chunks: 100%|██████████| 3/3 [00:00<00:00, 3.75chunk/s]
[INFO] Total chunks created: 3

[3/4] Transcribing audio (this may take a while)...
[INFO] Starting parallel transcription with 4 workers
Transcribing: 100%|██████████| 3/3 [00:09<00:00, 3.27s/chunk]
[INFO] Transcription complete. Total segments: 42

[4/4] Generating subtitle file...
Writing SRT: 100%|██████████| 42/42 [00:00<00:00]
[INFO] SRT file created successfully: /path/to/test_file.srt

============================================================
SUCCESS! Subtitle file created:
  /path/to/test_file.srt
============================================================
```

## License

This project is provided as-is for educational and personal use.
