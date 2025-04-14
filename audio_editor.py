import logging
from typing import List
import ffmpeg
from exceptions import AudioOverlayError, AudioMergeError
from data_structures import Scene
import os
import tempfile

def overlay_narration_on_scene(scene: Scene, original_audio_path: str, output_path: str) -> None:
    """
    Overlays narration audio onto scene audio.
    Raises AudioOverlayError on failure.
    """
    try:
        logging.info(f"Overlaying narration for scene {scene.scene_id}: {original_audio_path} + {scene.narration_audio_path} -> {output_path}")
        
        # Create filter complex for mixing audio streams
        input_orig = ffmpeg.input(original_audio_path)
        input_narr = ffmpeg.input(scene.narration_audio_path)
        
        mixed = ffmpeg.filter([input_orig.audio, input_narr.audio], 
                            'amix', 
                            inputs=2, 
                            duration='longest')
        
        stream = ffmpeg.output(mixed, output_path)
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        
        logging.info(f"Overlay complete for scene {scene.scene_id}, saved to {output_path}")
    except Exception as e:
        logging.error(f"Audio overlay failed for scene {scene.scene_id}: {e}", exc_info=True)
        raise AudioOverlayError(f"Failed to overlay narration: {e}")

def merge_scenes_to_final_mp3(scenes: List[Scene], output_mp3_path: str) -> None:
    """
    Concatenates all scenes into final MP3.
    Raises AudioMergeError on failure.
    """
    try:
        logging.info(f"Merging {len(scenes)} scenes into {output_mp3_path}")
        
        # Create temporary file list for concat
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            for scene in scenes:
                if not os.path.exists(scene.narration_audio_path):
                    logging.warning(f"Narration audio missing for scene {scene.scene_id}: {scene.narration_audio_path}")
                    continue
                f.write(f"file '{os.path.abspath(scene.narration_audio_path)}'\n")
            temp_list_path = f.name

        try:
            # Use ffmpeg concat demuxer
            stream = ffmpeg.input(temp_list_path, format='concat', safe=0)
            stream = ffmpeg.output(stream, output_mp3_path, acodec='libmp3lame')
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            
            logging.info(f"Final MP3 exported to {output_mp3_path}")
        finally:
            # Clean up temporary file
            os.unlink(temp_list_path)
            
    except Exception as e:
        logging.error(f"Audio merge failed: {e}", exc_info=True)
        raise AudioMergeError(f"Failed to merge scenes: {e}")