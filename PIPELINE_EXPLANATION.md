# Whisper Subtitle Generator - Pipeline Explanation

## Overview

This application processes video files to generate English subtitle files (.srt) using OpenAI's Whisper API. It handles videos of any length by splitting them into chunks and processing them in parallel.

## Pipeline Flow

```
Video File → Audio Extraction → Chunking → Parallel Transcription → SRT Generation
```

## Detailed Process

### 1. Audio Extraction (`src/extract_audio.py`)

**Purpose**: Convert video to optimized audio format for Whisper API

**Process**:
- Uses `ffmpeg` command-line tool via subprocess
- Extracts audio stream from video file
- Converts to WAV format with specific settings:
  - **Codec**: PCM 16-bit little-endian (`pcm_s16le`)
  - **Sample Rate**: 16 kHz (optimal for speech recognition)
  - **Channels**: Mono (single channel)
- Saves to `audio/<video_name>.wav`

**Why these settings?**
- Whisper API performs best with 16 kHz mono audio
- PCM format provides lossless quality
- Mono reduces file size without losing speech quality

---

### 2. Audio Chunking (`src/chunk_audio.py`)

**Purpose**: Split large audio files into manageable chunks for parallel processing

**Process**:
- Gets audio duration using `ffprobe`
- Splits audio into 80-second (80,000 ms) chunks using `ffmpeg`
- Each chunk is saved as a separate WAV file
- Returns metadata for each chunk:
  ```python
  {
      "chunk_path": "audio/chunk_0.wav",
      "offset_seconds": 0.0  # Used for timestamp adjustment
  }
  ```

**Why 80 seconds?**
- Balances API call overhead with processing speed
- Whisper API has a 25 MB file size limit (80s ≈ 2-3 MB)
- Allows efficient parallel processing without overwhelming the API

**Cleanup**:
- Chunk files are automatically deleted after transcription completes

---

### 3. Parallel Transcription (`src/transcribe.py`)

**Purpose**: Transcribe audio chunks concurrently with robust error handling

**Architecture**:
- Uses `ThreadPoolExecutor` for parallel processing
- Default 4 workers (configurable via `.env` or CLI argument)
- Each worker processes chunks independently

**Worker Process** (`process_chunk`):

1. **Open audio chunk file**
2. **Call Whisper API**:
   ```python
   client.audio.transcriptions.create(
       model="whisper-1",
       file=audio_file,
       response_format="verbose_json",
       timestamp_granularities=["segment"]
   )
   ```
3. **Extract segments** from API response
4. **Adjust timestamps** by adding chunk offset:
   ```python
   segment["start"] += offset_seconds
   segment["end"] += offset_seconds
   ```
5. **Return segments**

**Retry Logic**:
- **Max Retries**: 3 attempts per chunk
- **Backoff Strategy**: Exponential (1s → 2s → 4s)
- **Handles**:
  - HTTP 429 rate limit errors
  - Network timeouts
  - Connection resets
  - Other transient failures

**Error Handling**:
- If a chunk fails after 3 attempts:
  - Logs error with chunk offset
  - Returns empty segment list
  - **Processing continues** (no crash)
- Successful chunks are collected and sorted by start time

**Logging Example**:
```
[Worker] Processing chunk offset 160.00s
[Worker] Retry #1 after rate limit
[Worker] Completed chunk offset 160.00s
```

---

### 4. SRT File Generation (`src/merge_srt.py`)

**Purpose**: Format transcription segments into standard SRT subtitle format

**Process**:

1. **Sort segments** by start time (ensures chronological order)
2. **Format each segment**:
   ```
   1
   00:00:05,120 --> 00:00:07,900
   Translated text here

   2
   00:00:10,500 --> 00:00:13,200
   Next subtitle segment
   ```

3. **Timestamp formatting**:
   - Convert seconds to `HH:MM:SS,mmm` format
   - Example: `125.5` seconds → `00:02:05,500`

4. **Write to file**:
   - Save to same directory as input video: `<video_directory>/<video_name>.srt`
   - Use UTF-8 encoding (supports international characters)

**SRT Format Rules**:
- Sequential numbering starting at 1
- Timestamp format: `HH:MM:SS,mmm` (comma, not period)
- Text on separate line
- Blank line between segments

---

## Configuration (`config.py`)

**Environment Variables** (from `.env`):
- `OPENAI_API_KEY` - Required for Whisper API access
- `WORKERS` - Optional, default is 4

**Singleton Pattern**:
- Creates a single `OpenAI()` client instance
- Shared across all worker threads (thread-safe)
- Reduces overhead and connection pooling

**Constants**:
- `CHUNK_DURATION_MS = 80000` (80 seconds)
- Audio settings (16 kHz, mono, PCM)

---

## Main Entry Point (`main.py`)

**CLI Interface**:
```bash
python main.py <video_path> [--workers N]
```

**Workflow**:
1. Validate video file exists
2. Extract audio → `audio/video.wav`
3. Chunk audio → List of chunk metadata
4. Transcribe chunks in parallel → List of segments
5. Cleanup chunk files
6. Generate SRT file → `<video_directory>/video.srt`
7. Cleanup audio file
8. Print success message with output path

**Error Handling**:
- Validates input path (supports spaces, relative/absolute)
- Graceful keyboard interrupt (Ctrl+C)
- Clear error messages for common issues
- Non-zero exit codes for failures

---

## Key Design Decisions

### 1. Parallel Processing
- **Why**: Multi-hour videos would take too long sequentially
- **Implementation**: ThreadPoolExecutor with configurable workers
- **Benefit**: 4x faster with 4 workers (network-bound, not CPU-bound)

### 2. Retry Logic
- **Why**: API calls can fail due to rate limits or network issues
- **Implementation**: Exponential backoff prevents API hammering
- **Benefit**: Robust handling of transient failures

### 3. Chunk Cleanup
- **Why**: Audio chunks can consume significant disk space
- **Implementation**: Delete immediately after transcription
- **Benefit**: Minimal disk usage during processing

### 4. Timestamp Adjustment
- **Why**: Each chunk is transcribed independently (starts at 0:00)
- **Implementation**: Add chunk offset to all timestamps
- **Benefit**: Accurate timing across entire video

### 5. Continue on Failure
- **Why**: One bad chunk shouldn't fail entire video
- **Implementation**: Workers return empty list on failure, processing continues
- **Benefit**: Partial results better than no results

---

## Performance Characteristics

**For a 1-hour video**:
- Audio extraction: ~10-30 seconds (depends on video codec)
- Chunking: ~5-10 seconds
- Transcription: ~3-8 minutes (with 4 workers)
- SRT generation: <1 second
- **Total**: ~4-9 minutes

**Scaling**:
- More workers → Faster (up to rate limit)
- Longer videos → Linear time increase
- Network speed affects API call latency

---

## Dependencies Explained

### `python-dotenv`
- Loads `.env` file for API key management
- Keeps secrets out of source code

### `openai>=1.0.0`
- Official OpenAI Python client
- Provides Whisper API access
- Handles authentication and request formatting

### `ffmpeg` / `ffprobe`
- Industry-standard multimedia framework
- Handles audio extraction and chunking via subprocess
- System binary (not a Python package)
- Must be installed separately on your system

---

## File Organization

```
subtitle_maker/
├── config.py              # Environment config & OpenAI client
├── utils.py               # Shared helper functions
├── main.py                # CLI entry point & workflow orchestration
├── src/
│   ├── extract_audio.py   # Video → Audio conversion
│   ├── chunk_audio.py     # Audio splitting & cleanup
│   ├── transcribe.py      # Parallel API calls with retry
│   └── merge_srt.py       # SRT formatting & file writing
├── audio/                 # Temporary audio files (auto-cleaned)
├── input_videos/          # Optional: organize input videos
└── .env                   # API key & configuration

Note: .srt subtitle files are saved in the same directory as the input video.
```

---

## Security & Best Practices

1. **API Key Security**: Stored in `.env` with restricted permissions (600)
2. **Error Logging**: No sensitive data in error messages
3. **Path Validation**: Prevents directory traversal attacks
4. **UTF-8 Encoding**: Proper handling of international characters
5. **Resource Cleanup**: Always delete temporary files
6. **Graceful Shutdown**: Handles Ctrl+C properly

---

## Limitations & Considerations

1. **API Costs**: Each chunk costs ~$0.006 (Whisper pricing)
2. **Rate Limits**: OpenAI enforces rate limits (handled with retries)
3. **Translation Only**: Always translates to English (as per requirements)
4. **Internet Required**: Cannot work offline
5. **ffmpeg Dependency**: Must be installed separately

---

## Future Enhancements (Not Implemented)

- Support for multiple output languages
- Speaker diarization (who said what)
- Progress bar for long videos
- Resume capability for interrupted processing
- Batch processing multiple videos
- GPU-accelerated local Whisper (no API costs)
