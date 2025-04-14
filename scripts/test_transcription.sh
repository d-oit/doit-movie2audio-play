#!/bin/bash

# Test script for Whisper transcription with enhanced error handling
set -euo pipefail

# Load environment variables from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "Error: .env file not found"
    exit 1
fi

# Default audio file path
DEFAULT_AUDIO="temp/audio.wav"
AUDIO_FILE="${1:-$DEFAULT_AUDIO}"

# Error handling function
handle_error() {
    echo "Error: $1" >&2
    exit 1
}

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    handle_error "Audio file not found: $AUDIO_FILE"
fi

echo "Testing transcription on: $AUDIO_FILE"

# Run transcription test
if ! python scripts/run_transcription_test.py "$AUDIO_FILE"; then
    handle_error "Transcription test failed"
fi

echo "Transcription test completed successfully"
exit 0