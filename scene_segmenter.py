import logging
from typing import List
from data_structures import Scene, AnalysisResult
from exceptions import SceneSegmentationError

def segment_scenes(analysis_result: AnalysisResult) -> List[Scene]:
    """
    Segments audio into scenes using non-language segments and timestamps.
    Returns list of Scene objects.
    Raises SceneSegmentationError on failure.
    """
    try:
        # If we have no scenes from analysis, throw an error
        if not analysis_result.scenes:
            raise SceneSegmentationError("No scenes found in analysis result")

        # Sort non-language segments by start time
        non_language_segments = sorted(analysis_result.non_language_segments, key=lambda x: x[0])
        
        # If there are no non-language segments, return the scenes as-is
        if not non_language_segments:
            logging.info("No non-language segments found, using scenes from analysis directly")
            return analysis_result.scenes

        # Combine nearby non-language segments (less than 2 seconds apart)
        merged_segments = []
        current_segment = list(non_language_segments[0])

        for segment in non_language_segments[1:]:
            if segment[0] - current_segment[1] < 2.0:  # 2 second threshold
                current_segment[1] = segment[1]
            else:
                merged_segments.append(tuple(current_segment))
                current_segment = list(segment)
        merged_segments.append(tuple(current_segment))

        # Create scenes based on non-language segments
        final_scenes = []
        scene_id = 0
        last_end = 0.0

        for segment in merged_segments:
            # Create a scene from last_end to current segment start if there's content
            if segment[0] - last_end > 1.0:  # Minimum 1 second scene duration
                # Find relevant descriptions from analysis
                descriptions = [
                    seg["text"] for seg in analysis_result.segments
                    if seg["start"] >= last_end and seg["end"] <= segment[0]
                ]
                
                scene = Scene(
                    scene_id=scene_id,
                    start_time=last_end,
                    end_time=segment[0],
                    description=" ".join(descriptions) if descriptions else "Scene content"
                )
                final_scenes.append(scene)
                scene_id += 1
            
            # Create a scene for the non-language segment
            scene = Scene(
                scene_id=scene_id,
                start_time=segment[0],
                end_time=segment[1],
                description="Non-language segment (music/effects)"
            )
            final_scenes.append(scene)
            scene_id += 1
            
            last_end = segment[1]

        # Add final scene if there's content after the last non-language segment
        if last_end < analysis_result.scenes[-1].end_time:
            descriptions = [
                seg["text"] for seg in analysis_result.segments
                if seg["start"] >= last_end
            ]
            
            scene = Scene(
                scene_id=scene_id,
                start_time=last_end,
                end_time=analysis_result.scenes[-1].end_time,
                description=" ".join(descriptions) if descriptions else "Final scene"
            )
            final_scenes.append(scene)

        logging.info(f"Segmented {len(final_scenes)} scenes")
        return final_scenes

    except Exception as e:
        logging.error(f"Scene segmentation failed: {e}", exc_info=True)
        raise SceneSegmentationError(f"Failed to segment scenes: {e}")