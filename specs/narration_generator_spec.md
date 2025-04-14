# Specification: Narration Generator (Adapted `narration_generator.py`)

## 1. Purpose
To take text descriptions generated for non-dialogue scenes and synthesize them into audible speech segments using a Text-to-Speech (TTS) engine (e.g., Coqui-AI).

## 2. Inputs
- `segments_with_descriptions` (List[Dict]): The output from the Scene Describer, containing segment timestamps and text descriptions.
  Example: `[{'start_time': 0.0, 'end_time': 5.3, 'description': 'A car drives down a street.'}, ...]`
- `tts_config` (Dict): Configuration for the TTS engine, including model path, config path, voice ID, etc. Loaded from `.env`.
- `output_dir` (str): Directory to save the generated narration audio files.

## 3. Outputs
- `narrated_segments` (List[Dict]): The input list, with each dictionary now *also* containing the path to the generated narration audio file.
  Example: `[{'start_time': 0.0, 'end_time': 5.3, 'description': '...', 'narration_path': 'temp/narration_0.00.wav'}, ...]`
- Side Effect: Creates individual `.wav` files for each narration segment in the specified `output_dir` (or a subdirectory like `temp/narrations`).

## 4. Core Logic (Pseudocode)

```python
# Dependencies: TTS engine library (e.g., TTS from Coqui-AI), os

import os
# Assuming a TTS library like Coqui's TTS is used
# from TTS.api import TTS # Example import

# TDD Anchor: test_tts_generates_audio_for_description()
# TDD Anchor: test_tts_handles_empty_description()
# TDD Anchor: test_tts_uses_correct_voice()

def initialize_tts_engine(tts_config: dict):
    """Initializes the TTS engine based on config."""
    try:
        # Example using Coqui TTS:
        # model_path = tts_config.get('model_path')
        # config_path = tts_config.get('config_path')
        # voice_id = tts_config.get('voice_id') # Or speaker_wav for voice cloning
        # device = "cuda" if torch.cuda.is_available() else "cpu"

        # tts = TTS(model_path=model_path, config_path=config_path, progress_bar=True).to(device)
        # return tts
        print("Initializing TTS engine (using placeholder)...")
        # Placeholder for actual TTS initialization
        class MockTTS:
            def tts_to_file(self, text, speaker, language, file_path):
                 print(f"  [Mock TTS] Synthesizing '{text}' to {file_path}")
                 # Create a dummy short silent file for testing structure
                 from pydub import AudioSegment
                 silence = AudioSegment.silent(duration=1000) # 1 second
                 silence.export(file_path, format="wav")
                 return file_path
        return MockTTS() # Return mock object for now
    except Exception as e:
        print(f"Error initializing TTS engine: {e}")
        raise RuntimeError(f"TTS engine initialization failed: {e}")

def generate_narration_output_path(base_dir: str, start_time: float) -> str:
    """Generates a unique filename for the narration segment."""
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)
    # Use start time for uniqueness, formatted to avoid issues
    filename = f"narration_{start_time:.2f}.wav".replace('.', '_')
    return os.path.join(base_dir, filename)

def synthesize_narrations(segments_with_descriptions: list[dict], tts_config: dict, output_dir: str) -> list[dict]:
    """
    Generates narration audio files for each segment description.

    Args:
        segments_with_descriptions: List of dictionaries with descriptions.
        tts_config: Dictionary with TTS configuration.
        output_dir: Base directory to save narration files.

    Returns:
        The input list, updated with 'narration_path' for each segment.
    """
    narrated_segments = []
    narration_temp_dir = os.path.join(output_dir, "narrations_temp") # Subdir for clarity

    try:
        tts_engine = initialize_tts_engine(tts_config)
        print(f"Synthesizing narrations for {len(segments_with_descriptions)} segments...")

        for i, segment in enumerate(segments_with_descriptions):
            text = segment.get("description", "")
            if not text or text == "No visual information available." or text == "Description generation failed.":
                print(f"  Segment {i+1} ({segment['start_time']:.2f}-{segment['end_time']:.2f}): Skipping narration (no valid description).")
                segment["narration_path"] = None # Indicate no narration generated
            else:
                output_path = generate_narration_output_path(narration_temp_dir, segment['start_time'])
                print(f"  Segment {i+1} ({segment['start_time']:.2f}-{segment['end_time']:.2f}): Synthesizing to {output_path}")

                # Use the TTS engine to generate the audio file
                # Example call structure (adjust based on actual TTS library):
                tts_engine.tts_to_file(
                    text=text,
                    # speaker=tts_config.get('voice_id'), # Or speaker_wav
                    # language=tts_config.get('language', 'en'), # Get language if needed
                    file_path=output_path
                )
                segment["narration_path"] = output_path

            narrated_segments.append(segment)

        print("Narration synthesis complete.")
        return narrated_segments

    except Exception as e:
        print(f"Error during narration synthesis: {e}")
        raise RuntimeError(f"Narration synthesis failed: {e}")

```

## 5. Dependencies
- TTS Engine Library (e.g., `TTS` from Coqui-AI, or others like `gTTS`, `pyttsx3`)
- `os`

## 6. Configuration (`.env`)
- `TTS_MODEL_PATH`
- `TTS_CONFIG_PATH` (if applicable)
- `TTS_VOICE_ID` or `TTS_SPEAKER_WAV`
- `TTS_LANGUAGE` (if required by the engine)
- `TEMP_DIR` (used as base for `narrations_temp`)

## 7. Edge Cases
- Empty description text.
- Very long description text (might exceed TTS limits or take long).
- TTS engine errors (model loading, synthesis failure).
- Invalid TTS configuration in `.env`.
- File system errors (cannot write to output directory).
- Placeholder descriptions from the previous step (e.g., "Description generation failed.").

## 8. TDD Anchors
- `test_tts_generates_audio_for_description()`: Provide a sample segment dictionary and verify a `.wav` file is created.
- `test_tts_handles_empty_description()`: Provide a segment with an empty description and verify `narration_path` is `None` and no file is created.
- `test_tts_uses_correct_voice()`: If using a library supporting multiple voices, verify the correct one is used (might require mocking the TTS call).
- `test_tts_handles_engine_failure()`: Mock the TTS engine to raise an error and verify the function handles it gracefully (e.g., raises its own error).