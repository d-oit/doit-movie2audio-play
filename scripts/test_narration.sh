#!/bin/bash

# Test script for narration generation
set -euo pipefail

# Check requirements
command -v python >/dev/null 2>&1 || { echo >&2 "Python is required but not installed. Aborting."; exit 1; }

# Setup test environment
TEST_DIR="test_output/narration"
mkdir -p "$TEST_DIR"
TEST_TEXT="This is a test narration text."
TEST_OUTPUT="$TEST_DIR/narration.mp3"

# Clean up from previous runs
rm -f "$TEST_OUTPUT"

echo "Running narration generation tests..."

# Run unit tests
if ! python -m pytest tests/narration_generator_test.py -v; then
    echo "Unit tests failed"
    exit 1
fi

# Test with gTTS fallback
echo "Testing narration generation with gTTS fallback..."
if ! python -c "from narration_generator import NarrationGenerator; \
    NarrationGenerator().generate('$TEST_TEXT', '$TEST_OUTPUT')"; then
    echo "Narration generation failed"
    exit 1
fi

# Verify output
if [ ! -f "$TEST_OUTPUT" ]; then
    echo "Output file not created"
    exit 1
fi

# Test with custom TTS if available
if [ -n "${TTS_SERVICE:-}" ]; then
    echo "Testing with custom TTS service..."
    CUSTOM_OUTPUT="$TEST_DIR/custom_narration.mp3"
    if ! python -c "from narration_generator import NarrationGenerator; \
        import os; \
        NarrationGenerator().generate('$TEST_TEXT', '$CUSTOM_OUTPUT', tts_service=os.getenv('TTS_SERVICE'))"; then
        echo "Custom TTS narration failed"
        exit 1
    fi
else
    echo "Skipping custom TTS test (no TTS_SERVICE configured)"
fi

echo "Narration generation tests passed successfully"
exit 0