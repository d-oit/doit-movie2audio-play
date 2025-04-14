import os
import logging
import torch
from typing import List, Dict, Tuple, Optional
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from moviepy.editor import VideoFileClip
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SceneDescriber:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large"):
        """
        Initialize the scene describer with BLIP model.
        
        Args:
            model_name: Name/path of the BLIP model to use.
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Scene Describer (using device: {self.device})")
        # Don't load model immediately
    def load_model(self):
        """Load the BLIP model and processor."""
        if self.processor is None:
            logger.info(f"Loading BLIP model: {self.model_name}")
            try:
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
                self.model.to(self.device)
                logger.info("BLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading BLIP model: {e}")
                raise RuntimeError(f"Failed to load BLIP model: {e}")

    def extract_representative_frames(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        num_frames: int = 3
    ) -> List[Image.Image]:
        """
        Extract representative frames from a video segment.
        
        Args:
            video_path: Path to the video file.
            start_time: Start time of the segment (seconds).
            end_time: End time of the segment (seconds).
            num_frames: Number of frames to extract.
            
        Returns:
            List of PIL Image objects.
        """
        try:
            with VideoFileClip(video_path) as clip:
                # Ensure timestamps are within video duration
                duration = clip.duration
                start_time = max(0, min(start_time, duration))
                end_time = max(0, min(end_time, duration))
                
                if end_time <= start_time:
                    raise ValueError("Invalid time range")
                
                # Calculate frame timestamps
                segment_duration = end_time - start_time
                timestamps = [
                    start_time + (i * segment_duration / (num_frames - 1))
                    for i in range(num_frames)
                ]
                
                # Extract frames
                frames = []
                for ts in timestamps:
                    frame = clip.get_frame(ts)
                    # Convert from numpy array (H,W,C) to PIL Image
                    pil_frame = Image.fromarray(frame).convert('RGB')
                    frames.append(pil_frame)
                
                return frames
                
        except ValueError as ve:
            logger.error(f"Error extracting frames: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []

    def generate_description_for_frame(self, image: Image.Image) -> str:
        """
        Generate a description for a single frame using BLIP.
        
        Args:
            image: PIL Image to analyze.
            
        Returns:
            Generated description text.
        """
        # Load model lazily if not already loaded
        if self.processor is None:
            self.load_model()
            
        try:
            # Prepare image for the model
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate caption
            outputs = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=1.0,
                length_penalty=1.0
            )
            
            # Decode the generated caption
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return "Description generation failed"

    def generate_descriptions(
        self,
        video_file_path: str,
        non_dialogue_segments: List[Tuple[float, float]]
    ) -> List[Dict]:
        """
        Generate descriptions for all non-dialogue segments.
        
        Args:
            video_file_path: Path to the video file.
            non_dialogue_segments: List of (start_time, end_time) tuples.
            
        Returns:
            List of dictionaries containing segment info and descriptions.
        """
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file not found: {video_file_path}")
            
        descriptions = []
        total_segments = len(non_dialogue_segments)
        
        logger.info(f"Generating descriptions for {total_segments} segments...")
        
        for i, (start_time, end_time) in enumerate(non_dialogue_segments, 1):
            logger.info(f"Processing segment {i}/{total_segments} ({start_time:.2f}s - {end_time:.2f}s)")
            
            # Extract frames from this segment
            frames = self.extract_representative_frames(
                video_file_path,
                start_time,
                end_time
            )
            
            if not frames:
                logger.warning(f"No frames extracted for segment {i}")
                continue
            
            # Generate descriptions for each frame
            frame_descriptions = []
            for frame in frames:
                description = self.generate_description_for_frame(frame)
                if description and description != "Description generation failed":
                    frame_descriptions.append(description)
            
            # Combine frame descriptions into a single segment description
            if frame_descriptions:
                # Use the most informative description (usually the longest)
                main_description = max(frame_descriptions, key=len)
            else:
                main_description = "No visual information available"
            
            segment_info = {
                "start_time": start_time,
                "end_time": end_time,
                "description": main_description,
                "duration": end_time - start_time
            }
            
            descriptions.append(segment_info)
            logger.debug(f"Generated description for segment {i}: {main_description}")
        
        logger.info(f"Description generation complete. Processed {len(descriptions)} segments.")
        return descriptions

def main():
    """CLI interface for testing."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate descriptions for non-dialogue video segments")
    parser.add_argument("video_file", help="Path to video file")
    parser.add_argument("--segments", help="JSON file with non-dialogue segments (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # For testing, use a single segment if no segments file provided
        if args.segments:
            import json
            with open(args.segments, 'r') as f:
                segments = json.load(f)
        else:
            # Test with first 5 seconds
            segments = [(0, 5)]
        
        describer = SceneDescriber()
        results = describer.generate_descriptions(args.video_file, segments)
        
        print("\nGenerated Descriptions:")
        for i, result in enumerate(results, 1):
            print(f"\nSegment {i}:")
            print(f"Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
            print(f"Description: {result['description']}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()