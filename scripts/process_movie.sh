#!/bin/bash

# Main movie processing script with enhanced error handling
set -euo pipefail
# set -x # Debugging disabled

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Load default config file if it exists
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file $CONFIG_FILE not found, will use default settings"
fi

# Configuration with fallbacks
DEFAULT_OUTPUT="output/final_audio.mp3"

# Error handling function
handle_error() {
    echo "Error: $1" >&2
    echo "Processing failed on movie file: $INPUT_VIDEO" >&2
    exit 1
}

# Check requirements
check_requirements() {
    echo "Checking requirements..."
    local missing=0
    for cmd in python ffmpeg; do
        echo "Checking for command: $cmd"
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "Error: Required command '$cmd' is not installed" >&2
            missing=1
        else
            echo "Command '$cmd' found."
        fi
    done
    if [ "$missing" -ne 0 ]; then
        handle_error "Missing required dependencies"
    fi
    echo "Requirements check passed."
}

# Validate environment
validate_environment() {
    echo "Validating environment..."

    # Activate virtual environment if it exists
    echo "Checking for virtual environment..."
    if [ -d ".venv" ]; then
        echo "Found .venv directory. Attempting activation..."
        if [ -f ".venv/Scripts/activate" ]; then
            echo "Activating Windows virtual environment..."
            source .venv/Scripts/activate || handle_error "Failed to activate Windows virtual environment"
            echo "Windows virtual environment activated."
        elif [ -f ".venv/bin/activate" ]; then
            echo "Activating Unix virtual environment..."
            source .venv/bin/activate || handle_error "Failed to activate Unix virtual environment"
            echo "Unix virtual environment activated."
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
    echo "Python dependencies installed/updated."

    # Verify critical dependencies (ffmpeg-python and moviepy)
    echo "Verifying critical Python dependencies (ffmpeg-python, moviepy)..."
    if ! python -c "import ffmpeg; from moviepy.editor import AudioFileClip" >/dev/null 2>&1; then
        # Try importing individually for more specific error
        if ! python -c "import ffmpeg" >/dev/null 2>&1; then
             handle_error "Critical Python dependency 'ffmpeg-python' is missing or broken after installation"
        elif ! python -c "from moviepy.editor import AudioFileClip" >/dev/null 2>&1; then
             handle_error "Critical Python dependency 'moviepy' is missing or broken after installation"
        else
             handle_error "Critical Python dependencies check failed for an unknown reason"
        fi
    fi
    echo "Critical Python dependencies verified."

    # Check output directory (Simplified check, relies on main logic now)
    echo "Checking output directory existence (will be created later if needed)..."
    # The actual creation is handled in the main processing loop for batch mode
    # For single file mode, the python script might handle it, or we rely on the default 'output' folder created by load_config in python
    echo "Environment validation passed."
}

# Parse arguments
parse_arguments() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 <input_video> [--output output.mp3] [--debug] [--config config.yaml]"
        echo "       $0 --batch <input_folder> [--output-dir output] [--debug] [--config config.yaml]"
        exit 1
    fi

    # Check for batch mode
    if [ "$1" = "--batch" ]; then
        BATCH_MODE=1
        shift
        INPUT_FOLDER="$1"
        shift
    else
        BATCH_MODE=0
        INPUT_VIDEO="$1"
        shift # Move shift inside else block
    fi
    OUTPUT_FILE="$DEFAULT_OUTPUT" # Default for single file mode
    OUTPUT_DIR="${OUTPUT_DIR:-output}" # Default for batch mode, respects env var
    DEBUG=0
    BG_REDUCTION=""
    NARRATION_ADJUST=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --output)
                if [ "$BATCH_MODE" -eq 1 ]; then
                    handle_error "--output cannot be used with --batch mode. Use --output-dir instead."
                fi
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --output-dir)
                if [ "$BATCH_MODE" -eq 0 ]; then
                    handle_error "--output-dir can only be used with --batch mode."
                fi
                OUTPUT_DIR="$2"
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
                # Any remaining argument after batch/folder or single file is unknown
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

    if [ "$BATCH_MODE" -eq 1 ]; then
        # Process all MP4 files in the input folder
        echo "Processing all MP4 files in folder: $INPUT_FOLDER"
        echo "Output directory: ${OUTPUT_DIR:-output}"
        echo "Environment Configuration:"
        echo "- Config File: ${CONFIG_FILE}"
        echo "- Output Directory: ${OUTPUT_DIR:-output}"

        # Create output directory if it doesn't exist
        mkdir -p "${OUTPUT_DIR:-output}"

        # Process each MP4 file
        for video_file in "$INPUT_FOLDER"/*.mp4; do
            if [ ! -f "$video_file" ]; then
                continue  # No MP4 files found
            fi

            filename=$(basename -- "$video_file")
            output_file="${OUTPUT_DIR:-output}/${filename%.*}_described.mp3"

            echo "Processing: $video_file"
            echo "Output will be saved to: $output_file"

            # Build command with all parameters
            CMD="python -m src.main \"$video_file\" --output \"$output_file\""
            [ "$DEBUG" -eq 1 ] && CMD="$CMD --debug"
            [ -f "$CONFIG_FILE" ] && CMD="$CMD --config \"$CONFIG_FILE\""

            # Run the Python script with all parameters
            echo "Executing: $CMD"
            if ! eval "$CMD"; then
                echo "Warning: Processing failed for $video_file"
                continue
            fi

            echo "Successfully processed: $video_file"
            echo "Output created at: $output_file"
            echo "----------------------------------------"
        done
    else
        # Single file processing mode
        echo "Processing movie file: $INPUT_VIDEO"
        echo "Output will be saved to: $OUTPUT_FILE"
        echo "Environment Configuration:"
        echo "- Config File: ${CONFIG_FILE}"
        echo "- Output File: ${OUTPUT_FILE}"

        # Build command with all parameters
        CMD="python -m src.main \"$INPUT_VIDEO\" --output \"$OUTPUT_FILE\""
        [ "$DEBUG" -eq 1 ] && CMD="$CMD --debug"
        [ -f "$CONFIG_FILE" ] && CMD="$CMD --config \"$CONFIG_FILE\""

        # Run the Python script with all parameters
        echo "Executing: $CMD"
        if ! eval "$CMD"; then
            handle_error "Processing failed during execution"
        fi

        # Only show success message if output file exists and is not empty
        if [ -s "$OUTPUT_FILE" ]; then
            echo "Successfully processed movie file"
            echo "Final output created at: $OUTPUT_FILE"
        else
            handle_error "Processing failed: Output file was not created or is empty"
        fi
    fi
}

main "$@"
exit 0