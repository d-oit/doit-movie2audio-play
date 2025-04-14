#!/bin/bash

# Test script for audio editing operations
set -euo pipefail

# Check requirements
command -v python >/dev/null 2>&1 || { echo >&2 "Python is required but not installed. Aborting."; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo >&2 "FFmpeg is required but not installed. Aborting."; exit 1; }

# Setup test environment
TEST_DIR="test_output/audio_editing"
mkdir -p "$TEST_DIR"
TEST_AUDIO="test_data/sample.wav"
TEST_SCENES="test_data/scenes.json"

# Clean up from previous runs
rm -f "$TEST_DIR"/*

echo "Running audio editing tests..."

# Run unit tests
if ! python -m pytest tests/audio_editor_test.py -v; then
    echo "Unit tests failed"
    exit 1
fi

# Test trimming
echo "Testing audio trimming..."
TRIMMED_OUTPUT="$TEST_DIR/trimmed.wav"
if ! python -c "from audio_editor import AudioEditor; \
    from data_structures import Scene; \
    editor = AudioEditor(output_dir='$TEST_DIR'); \
    editor.trim_audio('$TEST_AUDIO', Scene(start=5, end=10))"; then
    echo "Audio trimming failed"
    exit 1
fi

# Test concatenation
echo "Testing audio concatenation..."
CONCAT_OUTPUT="$TEST_DIR/concatenated.wav"
if ! python -c "from audio_editor import AudioEditor; \
    editor = AudioEditor(output_dir='$TEST_DIR'); \
    editor.concatenate_audio(['$TEST_AUDIO', '$TEST_AUDIO'], '$CONCAT_OUTPUT')"; then
    echo "Audio concatenation failed"
    exit 1
fi

# Test mixing
echo "Testing audio mixing..."
MIXED_OUTPUT="$TEST_DIR/mixed.wav"
if ! python -c "from audio_editor import AudioEditor; \
    editor = AudioEditor(output_dir='$TEST_DIR'); \
    editor.mix_audio(['$TEST_AUDIO', '$TEST_AUDIO'], '$MIXED_OUTPUT')"; then
    echo "Audio mixing failed"
    exit 1
fi

# Test effects
echo "Testing audio effects..."
EFFECTS_OUTPUT="$TEST_DIR/effects.wav"
if ! python -c "from audio_editor import AudioEditor; \
    editor = AudioEditor(output_dir='$TEST_DIR'); \
    editor.apply_effects('$TEST_AUDIO', '$EFFECTS_OUTPUT', fade_in=1, fade_out=1, volume=0.8)"; then
    echo "Audio effects failed"
    exit 1
fi

echo "Audio editing tests passed successfully"
exit 0