import os
import sys
import logging
import yaml
import argparse
from typing import Dict, List, Optional
from datetime import timedelta
from pathlib import Path
from src.components import (
    VideoAnalyzer, Transcriber, SceneDetector,
    DescriptionGenerator, SpeechSynthesizer, AudioAssembler,
    Scene
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate audio descriptions for a video file")
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("--output", help="Path to output audio file", default=None)
    parser.add_argument("--config", help="Path to config file", default="config.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

# Load configuration
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Configure logging formatters from config
TERMINAL_FORMAT = CONFIG['logging']['progress_format']
FILE_FORMAT = CONFIG['logging']['component_format']

def initialize_logging(output_dir: str = "output") -> tuple[logging.Logger, logging.Logger, Dict[str, str]]:
    """
    Initialize the logging system with separate streams for progress and component logs.
    
    Args:
        output_dir: Directory to store component log files
    
    Returns:
        Tuple of (progress_logger, main_logger, component_log_files)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure progress logger for terminal output
    progress_logger = logging.getLogger("progress_reporter")
    progress_logger.setLevel(logging.INFO)
    terminal_handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    terminal_handler.setFormatter(logging.Formatter(TERMINAL_FORMAT))
    progress_logger.addHandler(terminal_handler)
    progress_logger.propagate = False
    
    # Configure component loggers
    component_log_files = {}
    component_names = ["mistral", "blip", "whisper", "tts", "ffmpeg", "main"]
    
    for component_name in component_names:
        log_file_path = os.path.join(output_dir, f"{component_name}.log")
        component_log_files[component_name] = log_file_path
        
        # Configure component logger
        logger = logging.getLogger(component_name)
        logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Add file handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        logger.addHandler(file_handler)
        logger.propagate = False
    
    # Get main logger
    main_logger = logging.getLogger("main")
    
    return progress_logger, main_logger, component_log_files

def report_progress(progress_logger: logging.Logger, message: str, percentage: Optional[float] = None):
    """Report progress to the terminal with optional completion percentage."""
    if percentage is not None:
        progress_logger.info(f"[{percentage:.1f}%] {message}")
    else:
        progress_logger.info(message)

def calculate_total_duration(input_video: str, processing_steps: List[str]) -> timedelta:
    """
    Calculate estimated total processing duration.
    
    Args:
        input_video: Path to input video file
        processing_steps: List of processing step names
    
    Returns:
        Estimated duration as timedelta
    """
    # TODO: Implement actual duration calculation based on video properties
    # For now, return a placeholder estimate
    return timedelta(minutes=len(processing_steps) * 5)

def filter_logo_text(text_content: str) -> str:
    """
    Filter out text associated with logos, watermarks, etc.
    
    Args:
        text_content: Text to filter
        
    Returns:
        Filtered text with logo-related content removed
    """
    # Get logo patterns from config
    logo_patterns = (
        CONFIG['logo_patterns']['studios'] +
        CONFIG['logo_patterns']['channels'] +
        [rf"{pattern}" for pattern in CONFIG['logo_patterns']['generic']]
    )
    
    filtered_text = text_content
    for pattern in logo_patterns:
        filtered_text = filtered_text.replace(pattern, "")
    
    return filtered_text.strip()

def main():
    """Main entry point for video-to-audio description processing."""
    try:
        # Initialize logging
        progress_logger, main_logger, log_files = initialize_logging()
        main_logger.info(f"Logging initialized. Component logs in: {log_files}")
        
        # Parse command line arguments
        args = parse_args()
        
        # Update config if custom config file provided
        global CONFIG
        if args.config != "config.yaml":
            CONFIG = load_config(args.config)
            
        # Set up logging level based on debug flag
        log_level = logging.DEBUG if args.debug else logging.INFO
        for logger_name in ["progress_reporter", "main", "ffmpeg", "whisper", "blip", "mistral", "tts"]:
            logging.getLogger(logger_name).setLevel(log_level)
            
        # Validate input video path
        input_video = args.input
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video not found: {input_video}")
            
        # Determine output path
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = CONFIG['paths'].get('output_dir', 'output')
            os.makedirs(output_dir, exist_ok=True)
            args.output = os.path.join(output_dir, "final_audio.mp3")
            
        processing_steps = list(CONFIG['components'].keys())
        
        # Calculate and display duration estimate
        try:
            total_duration = calculate_total_duration(input_video, processing_steps)
            report_progress(progress_logger, f"Estimated total duration: {total_duration}")
        except Exception as e:
            report_progress(progress_logger, f"WARN: Could not estimate duration: {e}")
            main_logger.warning(f"Duration estimation failed: {e}", exc_info=True)
        
        # Start processing
        report_progress(progress_logger, "Starting video processing...", 0)
        main_logger.info("Starting main processing workflow.")
        
        # Initialize components with their loggers and configs
        video_analyzer = VideoAnalyzer(
            logging.getLogger("ffmpeg"),
            CONFIG['components']['ffmpeg']
        )
        transcriber = Transcriber(
            logging.getLogger("whisper"),
            CONFIG['components']['whisper']
        )
        scene_detector = SceneDetector(
            logging.getLogger("blip"),
            CONFIG['components']['blip']
        )
        description_generator = DescriptionGenerator(
            logging.getLogger("mistral"),
            CONFIG['components']['mistral']
        )
        speech_synthesizer = SpeechSynthesizer(
            logging.getLogger("tts"),
            CONFIG['components']['tts']
        )
        audio_assembler = AudioAssembler(
            logging.getLogger("ffmpeg"),
            CONFIG['components']['ffmpeg']
        )

        # Step 1: Extract Audio
        report_progress(progress_logger, "Analyzing video...", 10)
        audio_path = video_analyzer.process(input_video)
        
        # Step 2: Generate Transcript
        report_progress(progress_logger, "Transcribing audio...", 30)
        transcript = transcriber.process(audio_path)
        
        # Step 3: Detect Scenes
        report_progress(progress_logger, "Detecting scenes...", 50)
        scenes = scene_detector.process(input_video)
        
        # Filter scene descriptions
        filtered_scenes = []
        for scene in scenes:
            if scene.text_description:
                filtered_description = filter_logo_text(scene.text_description)
                scene.text_description = filtered_description
            filtered_scenes.append(scene)
        
        # Step 4: Generate Descriptions
        report_progress(progress_logger, "Generating audio descriptions...", 70)
        raw_descriptions = description_generator.process(filtered_scenes, transcript)
        
        # Filter generated descriptions
        filtered_descriptions = [filter_logo_text(desc) for desc in raw_descriptions]
        
        # Step 5: Synthesize Speech
        report_progress(progress_logger, "Synthesizing speech...", 90)
        synthesized_audio = speech_synthesizer.process(filtered_descriptions)
        
        # Step 6: Final Assembly
        report_progress(progress_logger, "Assembling final audio...", 95)
        final_output = audio_assembler.process(
            audio_path, synthesized_audio, filtered_scenes)
            
        # Copy to specified output path if different
        if final_output != args.output:
            import shutil
            shutil.move(final_output, args.output)
            final_output = args.output
        report_progress(progress_logger, "Processing complete.", 100)
        main_logger.info(f"Main processing workflow finished successfully. "
                        f"Output saved to: {final_output}")
        
    except Exception as e:
        main_logger.error(f"An error occurred during processing: {e}", exc_info=True)
        report_progress(progress_logger,
                       f"ERROR: Processing failed. Check logs in 'output/' for details.")

if __name__ == "__main__":
    main()