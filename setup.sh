#!/bin/bash
#
# Whisper Subtitle Generator - Setup Script
# Creates virtual environment and installs all dependencies
#

set -e

echo "========================================"
echo "Whisper Subtitle Generator - Setup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    else
        echo -e "${RED}✗ Python 3.8+ required, found $PYTHON_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check ffmpeg
echo ""
echo "Checking ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    echo -e "${GREEN}✓ ffmpeg $FFMPEG_VERSION found${NC}"
else
    echo -e "${YELLOW}⚠ ffmpeg not found${NC}"
    echo ""
    echo "Please install ffmpeg:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS:         brew install ffmpeg"
    echo "  Windows:       Download from https://ffmpeg.org/download.html"
    echo ""
    read -p "Continue without ffmpeg? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check ffprobe
echo ""
echo "Checking ffprobe..."
if command -v ffprobe &> /dev/null; then
    echo -e "${GREEN}✓ ffprobe found${NC}"
else
    echo -e "${YELLOW}⚠ ffprobe not found (usually installed with ffmpeg)${NC}"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ venv directory already exists${NC}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    else
        echo "Using existing virtual environment"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check for NVIDIA GPU and offer torch installation
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    echo ""
    read -p "Install PyTorch for GPU acceleration? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing PyTorch with CUDA support..."
        pip install torch --index-url https://download.pytorch.org/whl/cu121
        echo -e "${GREEN}✓ PyTorch installed with CUDA support${NC}"
    fi
else
    echo -e "${YELLOW}No NVIDIA GPU detected (GPU acceleration unavailable)${NC}"
fi

# Create .env file if it doesn't exist
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << 'EOF'
# OpenAI API Key (required for API mode, optional for local Whisper with Google Translate)
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
EOF
    echo -e "${GREEN}✓ .env template created${NC}"
    echo -e "${YELLOW}  → Edit .env to add your OpenAI API key (if using API mode)${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Print success message
echo ""
echo "========================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "========================================"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "Usage examples:"
echo "  # Process a single video (local Whisper + Google Translate)"
echo "  python main.py video.mp4 --local-whisper"
echo ""
echo "  # Process a folder of videos"
echo "  python main.py /path/to/videos/ --local-whisper --skip-existing"
echo ""
echo "  # Use large model for best quality"
echo "  python main.py video.mp4 --local-whisper --model large-v3"
echo ""
echo "  # Direct Whisper translation (fastest)"
echo "  python main.py video.mp4 --local-whisper --direct-whisper"
echo ""
