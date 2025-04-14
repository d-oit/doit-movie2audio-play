#!/bin/bash

# Test script for the VAD-based audio description pipeline
set -euo pipefail

# Load environment variables if .env exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Default test video file (should be a short clip)
DEFAULT_VIDEO="test_data/test_clip.mp4"

# Error handling function
handle_error() {
    echo "Error: $1" >&2
    exit 1
}

# Help function
show_help() {
    echo "Usage: $0 [video_file] [options]"
    echo ""
    echo "Options:"
    echo "  --debug     Enable debug logging"
    echo "  --help      Show this help message"
    echo ""
    echo "If no video file is specified, will use: $DEFAULT_VIDEO"
}

# Parse command line arguments
VIDEO_FILE="$DEFAULT_VIDEO"
DEBUG_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        *)
            if [ -f "$1" ]; then
                VIDEO_FILE="$1"
            else
                handle_error "Video file not found: $1"
            fi
            shift
            ;;
    esac
done

# Ensure video file exists
[ -f "$VIDEO_FILE" ] || handle_error "Video file not found: $VIDEO_FILE"

# Create necessary directories
mkdir -p "${TEMP_DIR:-temp}"
mkdir -p "${OUTPUT_DIR:-output}"

echo "=== Testing VAD-based Audio Description Pipeline ==="
echo "Input video: $VIDEO_FILE"
echo "Environment Configuration:"
echo "- HUGGING_FACE_TOKEN: ${HUGGING_FACE_TOKEN:-(not set)}"
echo "- TTS_MODEL_PATH: ${TTS_MODEL_PATH:-tts_models/de/thorsten/vits}"
echo "- OUTPUT_DIR: ${OUTPUT_DIR:-output}"
echo "- TEMP_DIR: ${TEMP_DIR:-temp}"

# Run the pipeline
echo -e "\nStarting pipeline..."
if python main_orchestrator.py "$VIDEO_FILE" $DEBUG_FLAG; then
    echo -e "\nPipeline test completed successfully!"
    echo "Check the output directory for results."
else
    handle_error "Pipeline test failed"
fi