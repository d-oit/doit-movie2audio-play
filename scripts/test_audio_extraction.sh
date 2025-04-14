#!/bin/bash

# Test script for audio extraction functionality
set -euo pipefail

# Check required commands
command -v python >/dev/null 2>&1 || { echo >&2 "Python is required but not installed. Aborting."; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo >&2 "FFmpeg is required but not installed. Aborting."; exit 1; }

# Set up test environment
TEST_DIR="test_output/audio_extraction"
mkdir -p "$TEST_DIR"
TEST_VIDEO="test_data/sample.mp4"
TEST_OUTPUT="$TEST_DIR/extracted_audio.wav"

# Clean up from previous runs
rm -f "$TEST_OUTPUT"

echo "Running audio extraction tests..."

# Run the test
if ! python -m pytest tests/audio_extractor_test.py -v; then
    echo "Unit tests failed"
    exit 1
fi

# Integration test - actual extraction
echo "Running integration test..."
if ! python -c "from audio_extractor import AudioExtractor; AudioExtractor().extract_audio('$TEST_VIDEO', '$TEST_OUTPUT')"; then
    echo "Audio extraction failed"
    exit 1
fi

# Verify output
if [ ! -f "$TEST_OUTPUT" ]; then
    echo "Output file not created"
    exit 1
fi

echo "Audio extraction tests passed successfully"
exit 0