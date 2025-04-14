from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict # Import Dict

@dataclass
class Scene:
    """Scene representation with timing, description, and narration info"""
    scene_id: int
    start_time: float  # seconds
    end_time: float    # seconds
    description: str   # AI or user-provided
    transcription: Optional[str] = None  # Raw transcription for this scene
    narration_text: Optional[str] = None  # AI or user-provided
    narration_audio_path: Optional[str] = None  # path to narration audio file

@dataclass
class AnalysisResult:
    """Analysis result containing scenes and non-language segments"""
    scenes: List[Scene]
    non_language_segments: List[Tuple[float, float]]  # (start, end) in seconds
    errors: List[str]
    full_transcription: Optional[str] = None  # Complete transcription of audio
    # Add the missing segments field used by scene_segmenter
    segments: Optional[List[Dict]] = None # Detailed transcription segments (e.g., from Whisper)