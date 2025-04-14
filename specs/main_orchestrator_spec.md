# Specification: Main Orchestrator (Modified `main_orchestrator.py`)

## 1. Purpose
To orchestrate the end-to-end process of generating an audio-described movie track by coordinating the execution of various modules: audio extraction, VAD, scene description, narration synthesis, and audio mixing. It also handles configuration loading and command-line argument parsing.

## 2. Inputs
- Command Line Arguments:
    - `mp4_file` (str): Path to the input video file (required).
    - `--output` (str, optional): Path for the final output audio file. Overrides `.env`.
    - `--debug` (bool, optional): Flag to enable debug logging.
    - Potentially other CLI args to override specific `.env` settings (e.g., `--background-reduction-db`).
- Environment Variables (`.env` file):
    - `OUTPUT_DIR`, `TEMP_DIR`
    - `HUGGING_FACE_TOKEN` (for VAD)
    - `TTS_MODEL_PATH`, `TTS_CONFIG_PATH`, `TTS_VOICE_ID` (for TTS)
    - `BACKGROUND_VOLUME_REDUCTION_DB`, `NARRATION_VOLUME_ADJUST_DB` (for Mixer)
    - `LOG_LEVEL`, `LOG_FILE`
    - Potentially settings for the vision model if not hardcoded.

## 3. Outputs
- Side Effect: Creates the final mixed audio file in the specified output location.
- Side Effect: Creates temporary files (extracted audio, narration segments) in the temp directory.
- Side Effect: Logs progress and errors to console and/or log file.

## 4. Core Logic (Pseudocode)

```python
# Dependencies: All module dependencies (os, logging, argparse, dotenv, etc.)
#               plus the project's own modules.

import argparse
import logging
import os
import time
from dotenv import load_dotenv
from pathlib import Path

# Import functions from other modules
from audio_extractor import extract_audio_from_mp4 # Assuming this extracts to WAV/MP3
# from vad_analyzer import detect_non_dialogue_segments # New
# from scene_describer import generate_descriptions # New
# from narration_generator import synthesize_narrations # Adapted
# from audio_mixer import mix_audio # New
# from exceptions import ConfigurationError, ...

# --- Configuration & Setup Functions (Adapt existing) ---

def setup_logging(debug: bool = False):
    # (Keep existing logic, ensure it uses LOG_LEVEL/LOG_FILE from env)
    pass

def clean_env_value(value: str) -> str:
     # (Keep existing logic)
     pass

def load_config() -> dict:
    """Load configuration from .env file, including new settings."""
    if not load_dotenv():
        # Consider falling back to .env.template or raising error
        print("Warning: .env file not found. Using defaults or template.")
        # Or: raise ConfigurationError(".env file not found")

    config = {}
    # Required Vars
    required = ['OUTPUT_DIR', 'TEMP_DIR', 'BACKGROUND_VOLUME_REDUCTION_DB', 'NARRATION_VOLUME_ADJUST_DB']
    # Optional Vars (Defaults handled in modules or here)
    optional = ['HUGGING_FACE_TOKEN', 'TTS_MODEL_PATH', 'TTS_CONFIG_PATH', 'TTS_VOICE_ID', 'LOG_LEVEL', 'LOG_FILE']

    # (Adapt existing loading logic for required/optional vars)
    # Ensure numeric values are converted correctly (float for dB)
    try:
        config['BACKGROUND_VOLUME_REDUCTION_DB'] = float(os.getenv('BACKGROUND_VOLUME_REDUCTION_DB', 15.0))
        config['NARRATION_VOLUME_ADJUST_DB'] = float(os.getenv('NARRATION_VOLUME_ADJUST_DB', 0.0))
    except ValueError:
        raise ConfigurationError("Invalid numeric value for volume settings in .env")

    # Load other vars...
    config['HUGGING_FACE_TOKEN'] = os.getenv('HUGGING_FACE_TOKEN') # Can be None
    # ... load TTS config ...
    # ... load logging config ...

    # Create dirs if they don't exist
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(config['TEMP_DIR'], exist_ok=True)

    return config

def parse_args() -> argparse.Namespace:
    """Parse command line arguments, potentially adding new ones."""
    parser = argparse.ArgumentParser(description="Generate audio description for movie")
    parser.add_argument("mp4_file", help="Input MP4 file path")
    parser.add_argument("--output", help="Output file path for final audio mix (default: <OUTPUT_DIR>/<movie_name>_described.mp3)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    # Add overrides for key config values if desired
    parser.add_argument("--bg-reduction-db", type=float, help="Override background volume reduction (dB)")
    parser.add_argument("--narration-adjust-db", type=float, help="Override narration volume adjustment (dB)")

    return parser.parse_args()

# --- Main Orchestration Logic ---

def main():
    """Main entry point for the audio description generation process."""
    start_time = time.time()
    args = parse_args()
    setup_logging(args.debug) # Setup logging first
    logger = logging.getLogger(__name__)

    try:
        logger.info("--- Starting Audio Description Generation ---")
        config = load_config()

        # Override config with CLI args if provided
        if args.output:
            config['final_output_path'] = args.output
        else:
            # Default output path based on input filename
            input_filename = Path(args.mp4_file).stem
            config['final_output_path'] = os.path.join(config['OUTPUT_DIR'], f"{input_filename}_described.mp3")

        if args.bg_reduction_db is not None:
            config['BACKGROUND_VOLUME_REDUCTION_DB'] = args.bg_reduction_db
        if args.narration_adjust_db is not None:
            config['NARRATION_VOLUME_ADJUST_DB'] = args.narration_adjust_db

        logger.info(f"Input video: {args.mp4_file}")
        logger.info(f"Final output will be: {config['final_output_path']}")
        logger.info(f"Using Temp directory: {config['TEMP_DIR']}")

        # == Step 1: Extract Audio ==
        logger.info("Step 1: Extracting audio...")
        # Define expected audio path
        extracted_audio_filename = f"{Path(args.mp4_file).stem}_audio.wav" # Prefer WAV for processing
        extracted_audio_path = os.path.join(config['TEMP_DIR'], extracted_audio_filename)
        extract_audio_from_mp4(args.mp4_file, extracted_audio_path) # Assuming it extracts to WAV
        logger.info(f"Audio extracted to: {extracted_audio_path}")

        # == Step 2: Voice Activity Detection ==
        logger.info("Step 2: Detecting non-dialogue segments (VAD)...")
        non_dialogue_segments = detect_non_dialogue_segments(
            extracted_audio_path,
            hf_token=config.get('HUGGING_FACE_TOKEN')
        )
        logger.info(f"Found {len(non_dialogue_segments)} potential non-dialogue segments.")
        if not non_dialogue_segments:
             logger.warning("No non-dialogue segments found. No narration will be added.")
             # Optionally exit early or just proceed (will result in original audio)

        # == Step 3: Generate Scene Descriptions ==
        logger.info("Step 3: Generating scene descriptions...")
        # Pass relevant config if needed by the describer
        segments_with_descriptions = generate_descriptions(
            args.mp4_file,
            non_dialogue_segments,
            config=config # Pass full config for future use
        )
        logger.info("Scene description generation complete.")

        # == Step 4: Synthesize Narrations (TTS) ==
        logger.info("Step 4: Synthesizing narrations...")
        # Prepare TTS config subset
        tts_config = {
            'model_path': config.get('TTS_MODEL_PATH'),
            'config_path': config.get('TTS_CONFIG_PATH'),
            'voice_id': config.get('TTS_VOICE_ID')
            # Add language if needed by TTS module
        }
        narrated_segments = synthesize_narrations(
            segments_with_descriptions,
            tts_config,
            config['TEMP_DIR'] # Use TEMP_DIR as base for narration files
        )
        logger.info("Narration synthesis complete.")

        # == Step 5: Mix Audio ==
        logger.info("Step 5: Mixing final audio...")
        mix_audio(
            extracted_audio_path,
            narrated_segments,
            config['final_output_path'],
            background_volume_reduction_db=config['BACKGROUND_VOLUME_REDUCTION_DB'],
            narration_volume_adjust_db=config['NARRATION_VOLUME_ADJUST_DB']
        )
        logger.info(f"Final mixed audio saved to: {config['final_output_path']}")

        # == End ==
        duration = time.time() - start_time
        logger.info(f"--- Processing completed in {duration:.2f} seconds ---")

    except FileNotFoundError as e:
        logger.error(f"Error: Input file not found - {e}")
        exit(1)
    except ConfigurationError as e:
        logger.error(f"Error: Configuration problem - {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=args.debug) # Show traceback if debug
        exit(1)

if __name__ == "__main__":
    main()

```

## 5. Dependencies
- `argparse`, `logging`, `os`, `time`, `pathlib`
- `python-dotenv`
- All dependencies from the modules it orchestrates (`audio_extractor`, `vad_analyzer`, `scene_describer`, `narration_generator`, `audio_mixer`).

## 6. Configuration (`.env`)
- Consolidates configuration needed by all sub-modules. See individual specs and section 2 above.

## 7. Edge Cases
- Invalid command-line arguments.
- Missing `.env` file or critical variables within it.
- Errors raised by any of the sub-modules (should be caught and logged).
- No non-dialogue segments found by VAD.
- File system permission errors (reading input, writing output/temp files).
- Handling KeyboardInterrupt gracefully.

## 8. TDD Anchors
- `test_orchestrator_parses_args()`: Verify CLI arguments are parsed correctly.
- `test_orchestrator_loads_config()`: Verify `.env` loading and handling of required/optional/default values.
- `test_orchestrator_calls_modules_in_order()`: Use mocks to verify that `extract_audio`, `detect_non_dialogue_segments`, `generate_descriptions`, `synthesize_narrations`, and `mix_audio` are called sequentially with expected arguments.
- `test_orchestrator_handles_module_errors()`: Mock a sub-module to raise an exception and verify the orchestrator catches it, logs appropriately, and exits.
- `test_orchestrator_handles_no_non_dialogue_segments()`: Mock VAD to return an empty list and verify the process completes without error (or exits gracefully as designed).