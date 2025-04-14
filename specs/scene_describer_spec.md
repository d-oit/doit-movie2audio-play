# Specification: Scene Describer (`scene_describer.py`)

## 1. Purpose
To analyze video frames corresponding to specific time segments (identified as non-dialogue by VAD) and generate concise textual descriptions of the visual content using an advanced AI vision-language model (e.g., BLIP).

## 2. Inputs
- `video_file_path` (str): Path to the input video file (e.g., `.mp4`).
- `non_dialogue_segments` (List[Tuple[float, float]]): A list of tuples representing non-dialogue segments (start\_time, end\_time) from the VAD analyzer.
- `config` (Dict): Configuration dictionary potentially containing model details or API keys if using an external service.

## 3. Outputs
- `segments_with_descriptions` (List[Dict]): A list of dictionaries, where each dictionary contains the original segment timestamps and the generated text description.
  Example: `[{'start_time': 0.0, 'end_time': 5.3, 'description': 'A car drives down a street.'}, ...]`

## 4. Core Logic (Pseudocode)

```python
# Dependencies: transformers, Pillow, torch, opencv-python, moviepy

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2 # Using opencv for frame extraction
from moviepy.editor import VideoFileClip # Alternative for frame extraction / easier seeking
import os

# TDD Anchor: test_scene_description_for_known_action_sequence()
# TDD Anchor: test_scene_description_handles_short_segments()
# TDD Anchor: test_description_model_loading()

def load_vision_model_and_processor(model_name="Salesforce/blip-image-captioning-large"):
    """Loads the BLIP model and processor."""
    try:
        print(f"Loading vision model: {model_name}")
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        # Move model to GPU if available
        if torch.cuda.is_available():
            model.to("cuda")
        print("Vision model loaded successfully.")
        return processor, model
    except Exception as e:
        print(f"Error loading vision model: {e}")
        raise RuntimeError(f"Failed to load vision model {model_name}: {e}")

def extract_representative_frame(video_path, timestamp_sec):
    """Extracts a single frame near the middle of the segment."""
    try:
        # Using moviepy for potentially easier frame seeking
        with VideoFileClip(video_path) as clip:
             # Ensure timestamp is within video duration
             timestamp_sec = min(timestamp_sec, clip.duration - 0.1) # Avoid going past the end
             timestamp_sec = max(timestamp_sec, 0) # Ensure non-negative
             frame = clip.get_frame(timestamp_sec)
        # Moviepy frame is numpy array (H, W, C), convert to PIL Image (RGB)
        return Image.fromarray(frame).convert("RGB")

        # --- OpenCV alternative (might be faster but seeking less precise) ---
        # cap = cv2.VideoCapture(video_path)
        # if not cap.isOpened():
        #     raise IOError(f"Cannot open video file: {video_path}")
        # frame_rate = cap.get(cv2.CAP_PROP_FPS)
        # frame_num = int(timestamp_sec * frame_rate)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        # success, frame = cap.read()
        # cap.release()
        # if success:
        #     # Convert BGR (OpenCV default) to RGB
        #     return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # else:
        #     return None
        # --- End OpenCV alternative ---

    except Exception as e:
        print(f"Error extracting frame at {timestamp_sec}s from {video_path}: {e}")
        return None # Return None if frame extraction fails

def generate_description_for_frame(frame: Image.Image, processor, model):
    """Generates a caption for a single PIL Image frame."""
    if frame is None:
        return "No visual information available."
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = processor(images=frame, return_tensors="pt").to(device)
        
        # Generate caption
        outputs = model.generate(**inputs, max_length=50) # Adjust max_length as needed
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"Error during caption generation: {e}")
        return "Description generation failed." # Return placeholder on error

def generate_descriptions(video_file_path: str, non_dialogue_segments: list[tuple[float, float]], config: dict = None) -> list[dict]:
    """
    Generates text descriptions for non-dialogue video segments.

    Args:
        video_file_path: Path to the video file.
        non_dialogue_segments: List of (start_time, end_time) tuples.
        config: Configuration dictionary (unused for now, but for future flexibility).

    Returns:
        List of dictionaries with segment times and descriptions.
    """
    if not os.path.exists(video_file_path):
        raise FileNotFoundError(f"Video file not found: {video_file_path}")

    segments_with_descriptions = []
    try:
        processor, model = load_vision_model_and_processor() # Load model once

        print(f"Generating descriptions for {len(non_dialogue_segments)} segments...")
        for i, (start_time, end_time) in enumerate(non_dialogue_segments):
            # Extract frame near the middle of the segment
            mid_time = start_time + (end_time - start_time) / 2
            frame = extract_representative_frame(video_file_path, mid_time)

            # Generate description
            description = generate_description_for_frame(frame, processor, model)
            print(f"  Segment {i+1} ({start_time:.2f}-{end_time:.2f}): {description}")

            segments_with_descriptions.append({
                "start_time": start_time,
                "end_time": end_time,
                "description": description
            })

        print("Description generation complete.")
        return segments_with_descriptions

    except Exception as e:
        print(f"Error in description generation process: {e}")
        # Optionally return partial results or raise a higher-level error
        raise RuntimeError(f"Scene description generation failed: {e}")

```

## 5. Dependencies
- `transformers` (for BLIP model)
- `torch`
- `Pillow` (PIL)
- `opencv-python` (cv2) or `moviepy` (for frame extraction - moviepy preferred for seeking)
- `accelerate` (often required by `transformers`)

## 6. Configuration (`.env`)
- Potentially `TRANSFORMERS_CACHE` to control model download location.
- API keys if using a cloud-based vision service instead of local BLIP.

## 7. Edge Cases
- Very short non-dialogue segments (might not have meaningful visual change).
- Segments with rapid scene changes (single frame might not be representative). Consider sampling multiple frames.
- Video file corruption or format issues.
- Vision model errors (loading, inference).
- Resource constraints (GPU memory for BLIP).
- Segments at the very beginning or end of the video.

## 8. TDD Anchors
- `test_scene_description_for_known_action_sequence()`: Use a video clip with a known, simple action (e.g., car driving) during a silent part and verify the generated description is relevant.
- `test_scene_description_handles_short_segments()`: Test with a very brief non-dialogue segment.
- `test_description_model_loading()`: Verify the BLIP model loads correctly (or fails gracefully if dependencies are missing).
- `test_frame_extraction_logic()`: Test the frame extraction at different timestamps.
- `test_description_handles_extraction_failure()`: Ensure a placeholder description is generated if frame extraction fails.