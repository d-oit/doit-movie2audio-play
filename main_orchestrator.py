import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Import our modules
from vad_analyzer import detect_non_dialogue_segments
from scene_describer import SceneDescriber
from narration_generator import NarrationGenerator
from audio_mixer import AudioMixer
from audio_extractor import extract_audio_from_mp4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag and environment settings."""
    load_dotenv()
    
    log_level = logging.DEBUG if debug else os.getenv('LOG_LEVEL', 'INFO')
    log_file = os.getenv('LOG_FILE')
    
    # Set root logger level
    logging.getLogger().setLevel(log_level)
    
    # Add file handler if log file is specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            )
            logging.getLogger().addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not set up file logging to {log_file}: {e}")

def load_config() -> Dict:
    """Load configuration from environment variables."""
    if not load_dotenv():
        logger.warning(".env file not found. Using default values.")
    
    config = {
        'OUTPUT_DIR': os.getenv('OUTPUT_DIR', 'output'),
        'TEMP_DIR': os.getenv('TEMP_DIR', 'temp'),
        'HUGGING_FACE_TOKEN': os.getenv('HUGGING_FACE_TOKEN'),
        'TTS_MODEL_PATH': os.getenv('TTS_MODEL_PATH'),
        'TTS_CONFIG_PATH': os.getenv('TTS_CONFIG_PATH'),
        'BACKGROUND_VOLUME_REDUCTION_DB': float(os.getenv('BACKGROUND_VOLUME_REDUCTION_DB', -15.0)),
        'NARRATION_VOLUME_ADJUST_DB': float(os.getenv('NARRATION_VOLUME_ADJUST_DB', 0.0))
    }
    
    # Create necessary directories
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(config['TEMP_DIR'], exist_ok=True)
    os.makedirs(os.path.join(config['TEMP_DIR'], 'narrations'), exist_ok=True)
    
    return config

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate audio descriptions for non-dialogue movie scenes"
    )
    
    parser.add_argument(
        "mp4_file",
        help="Input MP4 file path"
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: <OUTPUT_DIR>/<input_name>_described.mp3)"
    )
    parser.add_argument(
        "--bg-reduction",
        type=float,
        help="Background volume reduction in dB (overrides env setting)"
    )
    parser.add_argument(
        "--narration-adjust",
        type=float,
        help="Narration volume adjustment in dB (overrides env setting)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def create_progress_bar(total_steps: int, desc: str = "Progress") -> tqdm:
    """Create a progress bar with consistent styling."""
    return tqdm(
        total=total_steps,
        desc=desc,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

# Add service instances to the signature
def process_movie(
    input_path: str,
    output_path: str,
    config: Dict,
    scene_describer: SceneDescriber,
    narrator: NarrationGenerator,
    mixer: AudioMixer,
    background_reduction_db: Optional[float] = None,
    narration_adjust_db: Optional[float] = None
) -> bool:
    """
    Process a movie file to add audio descriptions during non-dialogue segments.
    
    Args:
        input_path: Path to input video file
        output_path: Path for output audio file
        config: Configuration dictionary
        background_reduction_db: Optional override for background volume reduction
        narration_adjust_db: Optional override for narration volume adjustment
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # --- Initial Validation ---
        if not os.path.exists(input_path):
            logger.error(f"Input video file not found: {input_path}")
            raise FileNotFoundError(f"Input video file not found: {input_path}")
            
        # --- Setup ---
        # Create overall progress bar for main steps
        main_progress = create_progress_bar(5, "Overall Progress")
        main_progress.set_description("Step 1: Extracting audio")
        
        # --- Step 1: Extract Audio ---
        logger.info("Step 1: Extracting audio from video...")
        audio_path = os.path.join(config['TEMP_DIR'], 'extracted_audio.wav')
        extract_audio_from_mp4(input_path, audio_path)
        logger.info(f"Audio extracted to: {audio_path}")
        main_progress.update(1)
        
        # Step 2: Detect Non-Dialogue Segments
        main_progress.set_description("Step 2: Detecting non-dialogue segments")
        logger.info("Step 2: Detecting non-dialogue segments...")
        non_dialogue_segments = detect_non_dialogue_segments(
            audio_path,
            config['HUGGING_FACE_TOKEN']
        )
        logger.info(f"Found {len(non_dialogue_segments)} non-dialogue segments")
        main_progress.update(1)
        
        if not non_dialogue_segments:
            logger.warning("No non-dialogue segments found. No narration needed.")
            return True
        
        # Step 3: Generate Scene Descriptions
        main_progress.set_description("Step 3: Generating scene descriptions")
        logger.info("Step 3: Generating scene descriptions...")
        # scene_describer = SceneDescriber() # Removed: Use passed instance
        scenes_with_descriptions = scene_describer.generate_descriptions(
            input_path,
            non_dialogue_segments
        )
        main_progress.update(1)
        
        # Step 4: Generate Narrations
        main_progress.set_description("Step 4: Generating narrations")
        logger.info("Step 4: Generating narrations...")
        # narrator = NarrationGenerator() # Removed: Use passed instance
        narrated_segments = narrator.process_scenes(
            scenes_with_descriptions,
            progress_bar=True  # Enable progress bar in narrator
        )
        main_progress.update(1)
        
        # Step 5: Mix Audio
        main_progress.set_description("Step 5: Mixing audio")
        logger.info("Step 5: Mixing final audio...")
        # mixer = AudioMixer() # Removed: Use passed instance
        
        # Use provided values or fall back to config
        bg_reduction = background_reduction_db or config['BACKGROUND_VOLUME_REDUCTION_DB']
        narr_adjust = narration_adjust_db or config['NARRATION_VOLUME_ADJUST_DB']
        
        # Ensure narrator's TTS engine is initialized if needed (might be better done outside)
        # narrator.initialize_tts() # Consider if this is needed here or should be done once before calling process_movie

        success = mixer.mix_audio(
            audio_path,
            narrated_segments,
            output_path,
            bg_reduction,
            narr_adjust
        )
        
        if success:
            main_progress.update(1)
            main_progress.set_description("Complete!")
            logger.info(f"Processing complete. Final audio saved to: {output_path}")
            return True
        else:
            logger.error("Audio mixing failed")
            main_progress.set_description("Failed!")
            return False
        
        # Ensure progress bar is closed
        main_progress.close()
    except Exception as e:
        logger.error(f"Error processing movie: {e}", exc_info=True)
        return False

def main():
    """Main entry point."""
    start_time = time.time()
    
    # Parse arguments and load config
    args = parse_args()
    setup_logging(args.debug)
    config = load_config()
    
    try:
        # Validate input file
        if not os.path.exists(args.mp4_file):
            raise FileNotFoundError(f"Input file not found: {args.mp4_file}")
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            input_name = Path(args.mp4_file).stem
            output_path = os.path.join(config['OUTPUT_DIR'], f"{input_name}_described.mp3")
        
        # Initialize services once
        logger.info("Initializing services...")
        scene_describer = SceneDescriber()
        narrator = NarrationGenerator()
        # Explicitly initialize TTS engine here if needed by process_scenes
        # narrator.initialize_tts() # Uncomment if process_scenes requires pre-initialization
        mixer = AudioMixer()
        logger.info("Services initialized.")

        # Process the movie
        logger.info(f"Processing movie: {args.mp4_file}")
        logger.info(f"Output will be saved to: {output_path}")
        
        success = process_movie(
            args.mp4_file,
            output_path,
            config,
            scene_describer, # Pass initialized instance
            narrator,        # Pass initialized instance
            mixer,           # Pass initialized instance
            args.bg_reduction,
            args.narration_adjust
        )
        
        # Report results
        duration = time.time() - start_time
        if success:
            logger.info(f"Processing completed successfully in {duration:.1f} seconds")
            exit(0)
        else:
            logger.error(f"Processing failed after {duration:.1f} seconds")
            exit(1)
            
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()