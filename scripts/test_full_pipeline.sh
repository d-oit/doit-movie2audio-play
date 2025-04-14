#!/bin/bash

# Test script for full movie audio processing pipeline
set -euo pipefail

# Check requirements
command -v python >/dev/null 2>&1 || { echo >&2 "Python is required but not installed. Aborting."; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo >&2 "FFmpeg is required but not installed. Aborting."; exit 1; }

# Setup test environment
TEST_DIR="test_output/full_pipeline"
mkdir -p "$TEST_DIR"
TEST_VIDEO="test_data/sample.mp4"
FINAL_OUTPUT="$TEST_DIR/final_audio.mp3"

# Clean up from previous runs
rm -rf "$TEST_DIR"/*

echo "Running full pipeline tests..."

# Test with local Whisper model
echo "Testing pipeline with local Whisper model..."
if ! python main_orchestrator.py "$TEST_VIDEO" --output "$FINAL_OUTPUT"; then
    echo "Pipeline failed with local model"
    exit 1
fi

# Verify output
if [ ! -f "$FINAL_OUTPUT" ]; then
    echo "Final output file not created"
    exit 1
fi

# Test with API if key provided
if [ -n "${WHISPER_API_KEY:-}" ]; then
    echo "Testing pipeline with Whisper API..."
    API_OUTPUT="$TEST_DIR/api_final.mp3"
    if ! python main_orchestrator.py "$TEST_VIDEO" --output "$API_OUTPUT" --api-key "$WHISPER_API_KEY"; then
        echo "Pipeline failed with API"
        exit 1
    fi
else
    echo "Skipping API test (no WHISPER_API_KEY set)"
fi

# Test without TTS if available
echo "Testing pipeline without TTS..."
NOTTS_OUTPUT="$TEST_DIR/no_tts_final.mp3"
if ! python main_orchestrator.py "$TEST_VIDEO" --output "$NOTTS_OUTPUT" --no-tts; then
    echo "Pipeline failed without TTS"
    exit 1
fi

echo "Full pipeline tests passed successfully"
exit 0