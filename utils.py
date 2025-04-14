import os
import logging
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import torch
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PathManager:
    """Handles path operations and temporary file management."""
    
    @staticmethod
    def ensure_dirs(*paths: str) -> None:
        """Create directories if they don't exist."""
        for path in paths:
            os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def get_temp_path(filename: str, subdir: Optional[str] = None) -> str:
        """Get path in temp directory, optionally in a subdirectory."""
        base_temp = os.getenv('TEMP_DIR', 'temp')
        if subdir:
            path = os.path.join(base_temp, subdir, filename)
        else:
            path = os.path.join(base_temp, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    @staticmethod
    def get_output_path(filename: str) -> str:
        """Get path in output directory."""
        output_dir = os.getenv('OUTPUT_DIR', 'output')
        path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

class AudioUtils:
    """Audio processing utility functions."""
    
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 0.0
    
    @staticmethod
    def is_silent(segment: AudioSegment, silence_threshold: float = -50.0) -> bool:
        """Check if an audio segment is silent."""
        return segment.dBFS < silence_threshold
    
    @staticmethod
    def get_device() -> torch.device:
        """Get the appropriate device (CPU/GPU) for processing."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataUtils:
    """Data handling utility functions."""
    
    @staticmethod
    def save_json(data: Union[List, Dict], file_path: str) -> bool:
        """Save data to JSON file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Optional[Union[List, Dict]]:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return None

class ProcessUtils:
    """Process management utilities."""
    
    @staticmethod
    def check_dependencies() -> bool:
        """Check if all required dependencies are available."""
        try:
            # Check for required Python packages
            import torch
            import transformers
            import pyannote.audio
            import moviepy.editor
            import TTS
            
            # Check for GPU availability
            has_gpu = torch.cuda.is_available()
            if has_gpu:
                logger.info("GPU is available for processing")
            else:
                logger.warning("No GPU available, processing will be slower")
            
            # Check environment variables
            required_env = ['OUTPUT_DIR', 'TEMP_DIR']
            missing_env = [var for var in required_env if not os.getenv(var)]
            if missing_env:
                logger.warning(f"Missing environment variables: {missing_env}")
            
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return False

class VideoUtils:
    """Video processing utility functions."""
    
    @staticmethod
    def get_video_info(file_path: str) -> Dict[str, Any]:
        """Get video file information."""
        try:
            from moviepy.editor import VideoFileClip
            with VideoFileClip(file_path) as clip:
                return {
                    'duration': clip.duration,
                    'size': clip.size,
                    'fps': clip.fps,
                    'audio_fps': clip.audio.fps if clip.audio else None,
                    'has_audio': clip.audio is not None
                }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}

def main():
    """Test utility functions."""
    # Test environment
    ProcessUtils.check_dependencies()
    
    # Test path operations
    test_dirs = ['temp/test', 'output/test']
    PathManager.ensure_dirs(*test_dirs)
    
    # Test audio operations
    test_audio = PathManager.get_temp_path('test.wav')
    if os.path.exists(test_audio):
        duration = AudioUtils.get_audio_duration(test_audio)
        print(f"Test audio duration: {duration}s")
    
    # Test device detection
    device = AudioUtils.get_device()
    print(f"Processing device: {device}")

if __name__ == "__main__":
    main()