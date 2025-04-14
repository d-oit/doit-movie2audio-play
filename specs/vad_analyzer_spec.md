# Specification: VAD Analyzer (`vad_analyzer.py`)

## 1. Purpose
To analyze an audio file and identify segments that **do not** contain speech (non-dialogue segments), using the `pyannote.audio` library for Voice Activity Detection (VAD).

## 2. Inputs
- `audio_file_path` (str): Path to the input audio file (e.g., `.wav`, `.mp3`).
- `hf_token` (str, optional): Hugging Face authentication token, potentially needed for `pyannote.audio` models. To be loaded from `.env`.

## 3. Outputs
- `non_dialogue_segments` (List[Tuple[float, float]]): A list of tuples, where each tuple represents a non-dialogue segment with its start and end time in seconds. Example: `[(0.0, 5.3), (10.1, 15.0), ...]`.

## 4. Core Logic (Pseudocode)

```python
# Dependencies: pyannote.audio, torch, pydub (for duration)

from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import os

# TDD Anchor: test_vad_identifies_known_silent_segment()
# TDD Anchor: test_vad_ignores_known_dialogue_segment()

def get_audio_duration(file_path):
    """Helper to get audio duration in seconds."""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    except Exception as e:
        # Handle file loading errors
        print(f"Error loading audio for duration: {e}")
        return 0

def calculate_inverse_segments(speech_segments, total_duration):
    """Calculates non-speech segments based on speech segments."""
    non_speech = []
    current_time = 0.0
    for segment in speech_segments:
        start, end = segment.start, segment.end
        if start > current_time:
            # Add the gap before the current speech segment
            non_speech.append((current_time, start))
        current_time = max(current_time, end) # Move pointer to the end of the speech segment

    # Add the final gap if any
    if current_time < total_duration:
        non_speech.append((current_time, total_duration))

    return non_speech

def detect_non_dialogue_segments(audio_file_path: str, hf_token: str = None) -> list[tuple[float, float]]:
    """
    Identifies non-dialogue segments in an audio file using pyannote.audio VAD.

    Args:
        audio_file_path: Path to the audio file.
        hf_token: Hugging Face token (optional).

    Returns:
        List of (start_time, end_time) tuples for non-dialogue segments.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    try:
        # 1. Initialize pyannote.audio VAD pipeline
        #    Use authentication token if provided
        #    Model: "pyannote/voice-activity-detection"
        #    Consider device mapping (CPU/GPU)
        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=hf_token
        )
        # Move to GPU if available
        if torch.cuda.is_available():
             pipeline.to(torch.device("cuda"))

        # 2. Apply VAD pipeline to the audio file
        print(f"Applying VAD to {audio_file_path}...")
        vad_result = pipeline(audio_file_path)
        # vad_result is an Annotation object containing speech segments Timeline([Segment(start1, end1), ...])
        speech_segments = list(vad_result.itersegments())
        print(f"Detected {len(speech_segments)} speech segments.")


        # 3. Get total audio duration
        total_duration = get_audio_duration(audio_file_path)
        if total_duration == 0:
             raise ValueError("Could not determine audio duration.")

        # 4. Calculate inverse (non-dialogue) segments
        non_dialogue_segments = calculate_inverse_segments(speech_segments, total_duration)
        print(f"Calculated {len(non_dialogue_segments)} non-dialogue segments.")

        return non_dialogue_segments

    except ImportError:
        raise ImportError("pyannote.audio or torch not installed properly.")
    except Exception as e:
        # Catch potential model loading errors, runtime errors, etc.
        print(f"Error during VAD processing: {e}")
        raise RuntimeError(f"VAD processing failed: {e}")

```

## 5. Dependencies
- `pyannote.audio`
- `torch`
- `torchaudio` (potentially needed by pyannote)
- `pydub` (for getting audio duration easily)
- `python-dotenv` (for loading HF token)

## 6. Configuration (`.env`)
- `HUGGING_FACE_TOKEN` (Optional): Needed if the chosen `pyannote.audio` model requires authentication.

## 7. Edge Cases
- Very short audio files.
- Audio files with only silence or only speech.
- Corrupted audio files.
- Missing `HUGGING_FACE_TOKEN` if required by the model.
- Handling potential VAD model errors (download issues, compatibility).
- Very long audio files (memory usage, processing time). Consider chunking if necessary.

## 8. TDD Anchors
- `test_vad_identifies_known_silent_segment()`: Use an audio file with a known silent section and verify it's correctly identified as non-dialogue.
- `test_vad_ignores_known_dialogue_segment()`: Use an audio file with known speech and verify those segments are *not* in the non-dialogue output.
- `test_vad_handles_edge_silence()`: Test files starting/ending with silence.
- `test_vad_requires_hf_token_handling()`: Mock scenarios where token is needed/missing.