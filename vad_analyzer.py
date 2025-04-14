import os
import logging
import torch
import torchaudio
from typing import List, Tuple
from dotenv import load_dotenv
# Removed incorrect import: from silero_vad.utils_vad import get_speech_timestamps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file.
        
    Returns:
        Duration in seconds.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        RuntimeError: If file can't be loaded.
    """
    try:
        # Get duration using torchaudio
        metadata = torchaudio.info(file_path)
        return metadata.num_frames / metadata.sample_rate
    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")

def calculate_inverse_segments(speech_segments: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
    """
    Calculate non-speech segments based on speech segment timestamps.
    
    Args:
        speech_segments: List of (start_time, end_time) for speech segments.
        total_duration: Total duration of the audio in seconds.
        
    Returns:
        List of (start_time, end_time) tuples for non-speech segments.
    """
    non_speech = []
    current_time = 0.0
    
    # Sort segments by start time to ensure proper gap calculation
    speech_segments = sorted(speech_segments, key=lambda x: x[0])
    
    # Find gaps between speech segments
    for start, end in speech_segments:
        if start > current_time:
            non_speech.append((current_time, start))
        current_time = max(current_time, end)
    
    # Add final gap if needed
    if current_time < total_duration:
        non_speech.append((current_time, total_duration))
    
    return non_speech

def detect_non_dialogue_segments(audio_file_path: str, hf_token: str = None) -> List[Tuple[float, float]]:
    """
    Identifies non-dialogue segments in an audio file using Silero VAD.
    
    Args:
        audio_file_path: Path to the audio file.
        hf_token: Ignored (kept for backward compatibility).
        
    Returns:
        List of (start_time, end_time) tuples for non-dialogue segments.
        
    Raises:
        FileNotFoundError: If audio file not found.
        RuntimeError: If VAD processing fails.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    try:
        # Load Silero VAD model using torch.hub
        # force_reload=False assumes the model might be cached
        logger.info("Loading Silero VAD model...")
        # Correctly unpack the model and utilities from torch.hub.load
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      trust_repo=True)
        (get_speech_timestamps, _, _, _, _) = utils # Unpack specific utils needed
        
        # Get total audio duration
        total_duration = get_audio_duration(audio_file_path)
        if total_duration == 0:
            raise RuntimeError("Could not determine audio duration or audio file is empty")
        
        # Load audio file
        logger.info(f"Processing audio file: {audio_file_path}")
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        # Run VAD
        speech_timestamps = get_speech_timestamps(
            waveform,
            model,
            threshold=0.5,
            sampling_rate=sample_rate
        )
        
        # Convert timestamps to seconds
        speech_segments = [
            (ts['start'] / sample_rate, ts['end'] / sample_rate) 
            for ts in speech_timestamps
        ]
        logger.info(f"Found {len(speech_segments)} speech segments")
        
        # Calculate non-dialogue segments (inverse of speech segments)
        non_dialogue_segments = calculate_inverse_segments(speech_segments, total_duration)
        logger.info(f"Identified {len(non_dialogue_segments)} non-dialogue segments")
        
        # Filter out very short segments (less than 0.5 seconds)
        MIN_SEGMENT_DURATION = 0.5  # seconds
        non_dialogue_segments = [
            (start, end) for start, end in non_dialogue_segments
            if end - start >= MIN_SEGMENT_DURATION
        ]
        logger.info(f"After filtering short segments: {len(non_dialogue_segments)} segments remain")
        
        return non_dialogue_segments
        
    except Exception as e:
        logger.error(f"Error during VAD processing: {e}")
        raise RuntimeError(f"VAD processing failed: {str(e)}")

def main():
    """CLI interface for testing."""
    import argparse
    parser = argparse.ArgumentParser(description="Detect non-dialogue segments in audio file")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--token", help="Ignored (kept for backward compatibility)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        segments = detect_non_dialogue_segments(args.audio_file, args.token)
        print("\nNon-dialogue segments found:")
        for i, (start, end) in enumerate(segments, 1):
            duration = end - start
            print(f"{i}. {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
