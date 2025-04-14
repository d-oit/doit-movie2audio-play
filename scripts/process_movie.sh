#!/bin/bash

# Main movie processing script with enhanced error handling
set -euo pipefail

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Configuration with fallbacks
DEFAULT_OUTPUT="${OUTPUT_DIR:-output}/final_audio.mp3"
WHISPER_MODEL="${WHISPER_MODEL_PATH:-base}"  # Use WHISPER_MODEL_PATH from .env
TTS_VOICE="${TTS_VOICE_ID:-tts_models/de/thorsten/vits}"  # Use TTS_VOICE_ID from .env

# Error handling function
handle_error() {
    echo "Error: $1" >&2
    echo "Processing failed on movie file: $INPUT_VIDEO" >&2
    exit 1
}

# Check requirements
check_requirements() {
    local missing=0
    for cmd in python ffmpeg; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "Error: Required command '$cmd' is not installed" >&2
            missing=1
        fi
    done
    [ "$missing" -eq 0 ] || handle_error "Missing required dependencies"
}

# Validate environment
validate_environment() {
    # Activate virtual environment if it exists
    if [ -d ".venv" ]; then
        echo "Activating virtual environment..."
        if [ -f ".venv/Scripts/activate" ]; then
            # Windows
            source .venv/Scripts/activate
        elif [ -f ".venv/bin/activate" ]; then
            # Unix/Linux/MacOS
            source .venv/bin/activate
        else
            handle_error "Virtual environment exists but activation script not found"
        fi
    else
        handle_error "Virtual environment '.venv' not found. Please create it first."
    fi

    echo "Installing/updating Python dependencies..."
    if ! pip install -r requirements.txt; then
        handle_error "Failed to install Python dependencies"
    fi

    # Verify critical dependencies (ffmpeg-python and moviepy)
    if ! python -c "import ffmpeg; from moviepy.editor import AudioFileClip" 2>/dev/null; then
        handle_error "Critical Python dependencies (ffmpeg-python, moviepy) are still missing after installation"
    fi

    # Check output directory
    if [ -n "$OUTPUT_FILE" ]; then
        local output_dir
        output_dir=$(dirname "$OUTPUT_FILE")
        if [ -n "$output_dir" ]; then
            mkdir -p "$output_dir" || handle_error "Cannot create output directory: $output_dir"
        fi
    fi
}

# Parse arguments
parse_arguments() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 <input_video> [--output output_file] [--debug] [--bg-reduction-db value] [--narration-adjust-db value]"
        exit 1
    fi

    INPUT_VIDEO="$1"
    shift
    OUTPUT_FILE="$DEFAULT_OUTPUT"
    DEBUG=0
    BG_REDUCTION=""
    NARRATION_ADJUST=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --debug)
                DEBUG=1
                shift
                ;;
            --bg-reduction-db)
                BG_REDUCTION="$2"
                shift 2
                ;;
            --narration-adjust-db)
                NARRATION_ADJUST="$2"
                shift 2
                ;;
            *)
                handle_error "Unknown argument: $1"
                ;;
        esac
    done
}

# Main processing
main() {
    parse_arguments "$@"
    check_requirements
    validate_environment

    echo "Processing movie file: $INPUT_VIDEO"
    echo "Output will be saved to: $OUTPUT_FILE"
    echo "Environment Configuration:"
    echo "- WHISPER_MODEL_PATH: $WHISPER_MODEL"
    echo "- TTS_VOICE_ID: $TTS_VOICE"
    echo "- OUTPUT_DIR: ${OUTPUT_DIR:-output}"
    echo "- TEMP_DIR: ${TEMP_DIR:-temp}"
    [ -z "${WHISPER_API_KEY:-}" ] || echo "Using Whisper API"
    [ -z "${NO_TTS:-}" ] || echo "TTS narration disabled"

    # Build command with all parameters
    CMD="python main_orchestrator.py \"$INPUT_VIDEO\" --output \"$OUTPUT_FILE\""
    [ "$DEBUG" -eq 1 ] && CMD="$CMD --debug"
    [ -n "$BG_REDUCTION" ] && CMD="$CMD --bg-reduction-db $BG_REDUCTION"
    [ -n "$NARRATION_ADJUST" ] && CMD="$CMD --narration-adjust-db $NARRATION_ADJUST"

    # Run the Python script with all parameters
    echo "Executing: $CMD"
    if ! eval "$CMD"; then
        handle_error "Processing failed during execution"
    fi

    echo "Successfully processed movie file"
    echo "Final output created at: $OUTPUT_FILE"
}

main "$@"
exit 0