import os
import logging
from typing import List, Dict, Optional, Union
import pyttsx3 # Replaced TTS.api
# import torch # No longer needed for pyttsx3
import soundfile as sf
from moviepy.editor import AudioFileClip
import moviepy.video.fx.all as vfx # Import effects
import tempfile
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NarrationGenerator:
    def __init__(self):
        """
        Initialize the Narration Generator using pyttsx3.
        """
        self.tts_engine = None
        logger.info("Initializing Narration Generator (using pyttsx3)")

    def initialize_tts(self):
        """Initialize the pyttsx3 engine."""
        if self.tts_engine is None:
            try:
                logger.info("Initializing pyttsx3 engine...")
                self.tts_engine = pyttsx3.init()
                # Optional: Configure properties like rate, volume, voice
                # self.tts_engine.setProperty('rate', 150)
                # self.tts_engine.setProperty('volume', 0.9)
                # voices = self.tts_engine.getProperty('voices')
                # self.tts_engine.setProperty('voice', voices[0].id) # Example: Set first available voice
                logger.info("pyttsx3 engine initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing pyttsx3 engine: {e}")
                self.tts_engine = None # Ensure engine is None if init fails
                raise RuntimeError(f"Failed to initialize pyttsx3 engine: {e}")

    def adjust_audio_duration(
        self,
        audio_path: str,
        target_duration: float,
        max_speed_change: float = 0.3
    ) -> str:
        """
        Adjust the duration of a generated narration using moviepy.

        Args:
            audio_path: Path to the audio file to adjust.
            target_duration: Desired duration in seconds.
            max_speed_change: Maximum allowed speed change (e.g., 0.3 = ±30%).

        Returns:
            Path to the adjusted audio file. Returns original path if no change needed.

        Raises:
            FileNotFoundError: If audio_path does not exist.
            ValueError: If target_duration is invalid or audio is empty.
            RuntimeError: For internal moviepy or I/O errors during processing.
        """
        # --- Input Validation ---
        if not os.path.exists(audio_path):
            logger.error(f"Input audio file not found for adjustment: {audio_path}")
            raise FileNotFoundError(f"Input audio file not found for adjustment: {audio_path}")
        if target_duration <= 0:
            logger.error(f"Invalid target duration: {target_duration}")
            raise ValueError("Target duration must be positive")

        audio = None
        adjusted_audio = None
        try:
            # --- Load and Calculate Speed ---
            logger.debug(f"Loading audio for adjustment: {audio_path}")
            audio = AudioFileClip(audio_path)
            current_duration = audio.duration

            if current_duration is None or current_duration <= 0:
                 raise ValueError("Audio file is empty or has invalid duration")

            speed_factor = current_duration / target_duration
            logger.debug(f"Current duration: {current_duration:.2f}s, Target: {target_duration:.2f}s, Initial Speed Factor: {speed_factor:.2f}")

            # --- Limit Speed Change ---
            if abs(1 - speed_factor) > max_speed_change:
                original_speed_factor = speed_factor
                speed_factor = (1 + max_speed_change) if speed_factor > 1 else (1 - max_speed_change)
                logger.warning(f"Required speed change ({original_speed_factor:.2f}x) exceeds limit ({max_speed_change*100:.0f}%). Clamping to {speed_factor:.2f}x.")

            # --- Apply Adjustment (if needed) ---
            if abs(speed_factor - 1.0) < 1e-6: # Check if speed_factor is effectively 1
                logger.debug("No significant speed adjustment needed.")
                return audio_path # Return original path

            # Create new temp file path
            base, ext = os.path.splitext(audio_path)
            adjusted_path = f"{base}_adjusted{ext}"
            logger.info(f"Adjusting speed by factor {1/speed_factor:.2f}. Output: {adjusted_path}")

            # Apply speed adjustment using fx
            adjusted_audio = audio.fx(vfx.speedx, factor=1/speed_factor)

            # Write the adjusted audio
            adjusted_audio.write_audiofile(adjusted_path, logger=None)
            logger.debug(f"Adjusted audio written to {adjusted_path}")

            return adjusted_path

        # --- Error Handling ---
        except (IOError, OSError) as e:
            logger.error(f"File I/O error during audio adjustment: {e}", exc_info=True)
            raise RuntimeError(f"File I/O error during adjustment for {audio_path}: {e}") from e
        except Exception as e: # Catch other potential moviepy errors
            logger.error(f"Unexpected error adjusting audio duration for {audio_path}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during adjustment for {audio_path}: {e}") from e
        # --- Cleanup ---
        finally:
            if adjusted_audio:
                try:
                    adjusted_audio.close()
                    logger.debug(f"Closed adjusted audio clip resource.")
                except Exception as ce:
                     logger.warning(f"Error closing adjusted audio clip: {ce}")
            if audio:
                try:
                    audio.close()
                    logger.debug(f"Closed original audio clip resource.")
                except Exception as ce:
                     logger.warning(f"Error closing original audio clip: {ce}")

    def generate_narration(
        self,
        text: str,
        output_path: str,
        target_duration: Optional[float] = None
    ) -> bool:
        """
        Generate narration audio for the given text.
        
        Args:
            text: Text to synthesize
            output_path: Where to save the audio file
            target_duration: Optional target duration for the narration
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.tts_engine is None:
            self.initialize_tts()
            
        try:
            # Log path details for debugging
            dir_to_create = os.path.dirname(output_path)
            logger.debug(f"Attempting to create directory for narration output. Full path: '{output_path}', Directory: '{dir_to_create}'")
            # Create output directory if it doesn't exist
            os.makedirs(dir_to_create, exist_ok=True)
            
            # Generate speech
            logger.debug(f"Generating narration for text: '{text}' to path: '{output_path}'")
            output_dir = os.path.dirname(output_path)
            logger.debug(f"Creating output directory: '{output_dir}'")
            os.makedirs(output_dir, exist_ok=True)
            logger.debug("Directory created or verified")

            # Use pyttsx3 to save to file
            logger.debug("Calling pyttsx3 save_to_file...")
            self.tts_engine.save_to_file(text, output_path)
            logger.debug("Calling pyttsx3 runAndWait...")
            self.tts_engine.runAndWait() # Blocks until speaking/saving is complete
            logger.debug(f"pyttsx3 saved narration to: {output_path}")
            
            # Adjust duration if target_duration is specified
            if target_duration is not None:
                output_path = self.adjust_audio_duration(output_path, target_duration)
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating narration: {e}")
            return False
    def process_scenes(
        self,
        scenes_with_descriptions: List[Dict],
        progress_bar: bool = False,
        base_temp_dir: Optional[str] = None # Add base temp dir argument
    ) -> List[Dict]:
        """
        Generate narrations for all scenes with descriptions.
        
        Args:
            scenes_with_descriptions: List of scene dictionaries with descriptions.
            progress_bar: Whether to show a progress bar (default: False).
            
        Returns:
            The input list, updated with paths to generated narration audio files.
        """
        total_scenes = len(scenes_with_descriptions)
        logger.info(f"Processing narrations for {total_scenes} scenes...")
        
        # Create progress bar if requested
        pbar = None
        if progress_bar:
            pbar = tqdm(
                total=total_scenes,
                desc="Generating narrations",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        
        for i, scene in enumerate(scenes_with_descriptions, 1):
            if not scene.get('description'):
                logger.warning(f"Scene {i} has no description, skipping narration")
                scene['narration_path'] = None
                if pbar:
                    pbar.update(1)
                continue
                
            # Create output filename based on scene timing
            # Determine base directory for narrations
            temp_dir_root = base_temp_dir or tempfile.gettempdir()
            narration_output_dir = os.path.join(temp_dir_root, "movie2audio_narrations")
            os.makedirs(narration_output_dir, exist_ok=True) # Ensure it exists

            # Create absolute output path
            output_filename = f"narration_{scene['start_time']:.2f}.wav"
            output_path = os.path.join(narration_output_dir, output_filename)
            
            # Generate narration
            logger.info(f"Generating narration {i}/{len(scenes_with_descriptions)}")
            success = self.generate_narration(
                text=scene['description'],
                output_path=output_path,
                target_duration=scene.get('duration')  # Use scene duration as target
            )
            
            if success:
                scene['narration_path'] = output_path
                logger.debug(f"Narration generated: {output_path}")
            else:
                scene['narration_path'] = None
                logger.warning(f"Failed to generate narration for scene {i}")
            
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
        
        return scenes_with_descriptions

def main():
    """CLI interface for testing."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate narrations for scene descriptions")
    parser.add_argument("--text", help="Test text to synthesize")
    parser.add_argument("--scenes", help="JSON file with scene descriptions")
    # Removed --model and --config arguments as they are not used by pyttsx3
    # parser.add_argument("--model", help="Path to TTS model")
    # parser.add_argument("--config", help="Path to model config")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        generator = NarrationGenerator() # Removed model/config args
        
        if args.text:
            # Test single text narration
            output_path = os.path.join("temp", "test_narration.wav")
            if generator.generate_narration(args.text, output_path):
                print(f"Narration generated: {output_path}")
        
        elif args.scenes:
            # Process scenes from JSON file
            import json
            with open(args.scenes, 'r') as f:
                scenes = json.load(f)
            
            results = generator.process_scenes(scenes)
            
            print("\nNarration Generation Results:")
            for i, scene in enumerate(results, 1):
                print(f"\nScene {i}:")
                print(f"Description: {scene['description']}")
                print(f"Narration: {scene.get('narration_path', 'Failed')}")
                
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()