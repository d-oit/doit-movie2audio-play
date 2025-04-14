#!/bin/bash

# Test script that skips TTS-dependent tests
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Running Tests Without TTS ==="
echo -e "${YELLOW}Note: TTS functionality will be skipped due to Python version compatibility${NC}"
echo "Current Python version: $(python --version)"
echo

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate || source .venv/Scripts/activate

# Install core dependencies (excluding TTS)
echo "Installing core dependencies..."
pip install torch torchvision torchaudio \
    transformers accelerate \
    Pillow opencv-python \
    soundfile pydub ffmpeg-python \
    moviepy pyannote.audio \
    pytest pytest-cov

# Run tests with TTS-related tests excluded
echo -e "\nRunning tests..."
pytest tests/ \
    -v \
    --ignore tests/test_narration_generator.py \
    -k "not test_tts and not test_narration" \
    --cov=. \
    --cov-report=term-missing

echo -e "\n${YELLOW}Note: TTS tests were skipped. Install Python 3.9-3.11 for full functionality.${NC}"
echo "See PYTHON_VERSION.md for setup instructions."