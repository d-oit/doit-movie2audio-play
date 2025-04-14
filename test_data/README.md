# Test Data Directory

This directory should contain short video clips for testing the VAD-based audio description pipeline.

## Expected Files

- `test_clip.mp4`: A short video clip (recommended 30-60 seconds) containing both dialogue and non-dialogue segments.

## Test Clip Requirements

The test clip should include:
- Clear dialogue segments
- Clear non-dialogue segments (e.g., action scenes, scenic shots)
- A mix of audio levels
- Visual content that can be described by the scene describer

## Example Usage

1. Place your test clip in this directory
2. Run the test pipeline:
   ```bash
   ./scripts/test_pipeline.sh test_data/test_clip.mp4
   ```

## Notes

- Keep test clips short to speed up testing
- Test clips should be representative of the types of content you want to process
- Consider including clips that test edge cases:
  - Overlapping dialogue with background noise
  - Music with and without dialogue
  - Action sequences
  - Scene transitions
  - Different lighting conditions (for scene description)
  - Multiple speakers
  - Background ambient noise

## Processing Steps Tested

The test clip will exercise:
1. Voice Activity Detection (VAD)
2. Scene Description Generation
3. Text-to-Speech Narration
4. Audio Mixing and Volume Adjustment

Each step's output can be found in the respective directories under `temp/` during processing.