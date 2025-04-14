# Specification: Audio Mixer (`audio_mixer.py`)

## 1. Purpose
To combine the original movie audio track with the generated narration audio segments, overlaying the narration only during the specified non-dialogue time segments and adjusting the volume of the original audio during narration.

## 2. Inputs
- `original_audio_path` (str): Path to the full original audio track (e.g., `.wav`, `.mp3`).
- `narrated_segments` (List[Dict]): The list output from the Narration Generator, containing segment timestamps, descriptions (optional), and paths to the narration audio files (`narration_path`). `narration_path` can be `None` if no narration was generated for a segment.
  Example: `[{'start_time': 0.0, 'end_time': 5.3, 'narration_path': 'temp/narration_0.00.wav'}, {'start_time': 10.1, 'end_time': 15.0, 'narration_path': None}, ...]`
- `output_path` (str): The desired path for the final mixed audio file (e.g., `output/final_mix.mp3`).
- `background_volume_reduction_db` (float): How many dB to reduce the original audio volume during narration segments (e.g., 15.0 for a significant reduction). Loaded from `.env`.
- `narration_volume_adjust_db` (float): How many dB to adjust the narration volume relative to its original level (e.g., 0.0 for no change, +3.0 to make it louder). Loaded from `.env`.

## 3. Outputs
- Side Effect: Creates the final mixed audio file at the specified `output_path`.

## 4. Core Logic (Pseudocode)

```python
# Dependencies: pydub

from pydub import AudioSegment
import os

# TDD Anchor: test_mixing_overlays_narration_correctly()
# TDD Anchor: test_mixing_reduces_original_audio_volume_during_narration()
# TDD Anchor: test_mixing_handles_missing_narration_file()
# TDD Anchor: test_mixing_handles_narration_longer_than_segment()

def mix_audio(original_audio_path: str,
              narrated_segments: list[dict],
              output_path: str,
              background_volume_reduction_db: float = 15.0, # Default reduction
              narration_volume_adjust_db: float = 0.0): # Default no adjustment
    """
    Mixes narration audio into the original track during non-dialogue segments.

    Args:
        original_audio_path: Path to the original full audio.
        narrated_segments: List of dicts with segment times and narration paths.
        output_path: Path for the final output file.
        background_volume_reduction_db: dB reduction for original audio during narration.
        narration_volume_adjust_db: dB adjustment for narration audio.
    """
    if not os.path.exists(original_audio_path):
        raise FileNotFoundError(f"Original audio file not found: {original_audio_path}")

    try:
        print(f"Loading original audio: {original_audio_path}")
        original_audio = AudioSegment.from_file(original_audio_path)
        final_mix = original_audio # Start with the original

        print(f"Processing {len(narrated_segments)} segments for mixing...")
        for i, segment in enumerate(narrated_segments):
            start_time = segment.get('start_time')
            end_time = segment.get('end_time')
            narration_path = segment.get('narration_path')

            if narration_path is None or not os.path.exists(narration_path):
                print(f"  Segment {i+1} ({start_time:.2f}-{end_time:.2f}): No narration file found or specified. Skipping mix for this segment.")
                continue # Skip if no narration exists for this segment

            if start_time is None or end_time is None:
                 print(f"  Segment {i+1}: Missing start or end time. Skipping.")
                 continue

            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            segment_duration_ms = end_ms - start_ms

            if segment_duration_ms <= 0:
                 print(f"  Segment {i+1}: Invalid duration ({segment_duration_ms}ms). Skipping.")
                 continue

            print(f"  Segment {i+1} ({start_time:.2f}-{end_time:.2f}): Mixing narration from {narration_path}")

            try:
                # Load narration audio
                narration_audio = AudioSegment.from_file(narration_path)

                # Adjust narration volume
                narration_audio += narration_volume_adjust_db

                # --- Strategy: Modify the 'final_mix' in place ---

                # 1. Extract the original audio segment that will be modified
                original_segment_to_modify = final_mix[start_ms:end_ms]

                # 2. Reduce its volume
                quiet_original_segment = original_segment_to_modify - background_volume_reduction_db

                # 3. Overlay narration onto the quieted segment
                #    Handle narration longer/shorter than the gap:
                #    - If narration is longer, truncate it to fit the gap.
                #    - If narration is shorter, it will just fill the beginning.
                narration_to_overlay = narration_audio[:segment_duration_ms] # Truncate if longer
                mixed_segment = quiet_original_segment.overlay(narration_to_overlay)

                # 4. Splice the modified segment back into the final mix
                final_mix = final_mix[:start_ms] + mixed_segment + final_mix[end_ms:]

            except Exception as e:
                print(f"  Error processing segment {i+1} ({start_time:.2f}-{end_time:.2f}): {e}. Skipping mix for this segment.")
                # Continue with the next segment even if one fails

        # Export the final result
        print(f"Exporting final mixed audio to: {output_path}")
        output_format = os.path.splitext(output_path)[1][1:] or "mp3" # Default to mp3
        final_mix.export(output_path, format=output_format)
        print("Mixing complete.")

    except FileNotFoundError as e:
         raise e # Re-raise file not found
    except Exception as e:
        print(f"Error during audio mixing process: {e}")
        raise RuntimeError(f"Audio mixing failed: {e}")

```

## 5. Dependencies
- `pydub`

## 6. Configuration (`.env`)
- `BACKGROUND_VOLUME_REDUCTION_DB` (e.g., `15`)
- `NARRATION_VOLUME_ADJUST_DB` (e.g., `0`)

## 7. Edge Cases
- `narration_path` is `None` or points to a non-existent file.
- Narration audio is longer than the non-dialogue segment (current logic truncates).
- Narration audio is shorter than the non-dialogue segment (current logic overlays at start).
- Segments with zero or negative duration.
- Overlapping segments in the input list (shouldn't happen if VAD output is clean).
- Errors loading original or narration audio files (corruption, format).
- File system errors writing the final output file.

## 8. TDD Anchors
- `test_mixing_overlays_narration_correctly()`: Create a silent original track, add a known narration sound at a specific time, verify the output.
- `test_mixing_reduces_original_audio_volume_during_narration()`: Use an original track with constant sound, mix in narration, verify the original sound volume is lower during the narration segment.
- `test_mixing_handles_missing_narration_file()`: Provide a segment with `narration_path=None` or an invalid path, verify the original audio is unchanged for that segment.
- `test_mixing_handles_narration_longer_than_segment()`: Use narration audio longer than the target segment, verify the output contains only the truncated narration within the segment duration.
- `test_mixing_volume_adjustments()`: Verify the dB adjustments for background and narration are applied correctly.