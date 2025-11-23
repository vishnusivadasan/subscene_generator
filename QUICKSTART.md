# Quick Start Guide

Get subtitles from your video in 3 simple steps!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have **ffmpeg** installed on your system:
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Step 2: Verify API Key

Your `.env` file should already contain:
```bash
OPENAI_API_KEY=your-key-here
```

Check it's configured:
```bash
cat .env
```

## Step 3: Run the Generator

### Test with the provided test file:
```bash
python main.py test_file.mp4
```

### Or with your own video:
```bash
python main.py path/to/your/video.mp4
```

### Advanced: Use more workers for faster processing
```bash
python main.py test_file.mp4 --workers 8
```

## Output

Your subtitle file will be created in the same directory as your video:
```
Input:  test_file.mp4
Output: test_file.srt (same folder)

Input:  /path/to/movie.mp4
Output: /path/to/movie.srt
```

## What Happens During Processing

1. **Extracting audio** - Converts video to WAV format (~10-30 sec)
2. **Chunking audio** - Splits into 80-second segments (~5-10 sec)
3. **Transcribing** - Parallel API calls to Whisper (~3-8 min per hour of video)
4. **Generating SRT** - Formats and saves subtitle file (<1 sec)

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg on your system (see Step 1)

### "OPENAI_API_KEY not found"
Check your `.env` file contains the API key

### "Rate limit exceeded"
The script will automatically retry. If it persists, reduce workers:
```bash
python main.py video.mp4 --workers 2
```

### Slow processing
Increase worker count (default is 4):
```bash
python main.py video.mp4 --workers 8
```

## Cost Estimation

OpenAI Whisper API pricing: ~$0.006 per minute of audio

Examples:
- 30-minute video ≈ $0.18
- 1-hour video ≈ $0.36
- 2-hour video ≈ $0.72

## Notes

- **Language**: Automatically detects source language and translates to English
- **Quality**: Uses Whisper's best model for high accuracy
- **Cleanup**: Temporary files are automatically deleted
- **Paths**: Supports both relative and absolute paths, including spaces

## Support

For issues or questions, check:
- `README.md` - Full documentation
- `PIPELINE_EXPLANATION.md` - Technical details
- OpenAI Whisper API docs: https://platform.openai.com/docs/guides/speech-to-text

---

Happy subtitle generating! 🎬
