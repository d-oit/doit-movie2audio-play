import pytest
from scene_segmenter import segment_scenes
from exceptions import SceneSegmentationError
from data_structures import Scene, AnalysisResult

class TestSceneSegmenter:
    def test_segment_scenes_empty(self):
        """Test with empty non-language segments"""
        # Add a dummy scene representing the overall duration
        dummy_scene = Scene(scene_id=0, start_time=0.0, end_time=15.0, description="Full duration")
        analysis = AnalysisResult(scenes=[dummy_scene], non_language_segments=[], errors=[])
        result = segment_scenes(analysis)
        
        # If no non-language segments, it should return the original scenes list
        assert result == [dummy_scene]

    def test_segment_scenes_normal(self):
        """Test with normal non-language segments"""
        segments = [(2.0, 5.0), (8.0, 10.0)]
        dummy_scene = Scene(scene_id=0, start_time=0.0, end_time=15.0, description="Full duration")
        # Data for NORMAL test
        dummy_transcription_segments = [
            {"start": 0.5, "end": 1.5, "text": "Hello"},
            {"start": 6.0, "end": 7.0, "text": "World"},
            {"start": 11.0, "end": 14.0, "text": "Test"}
        ]
        analysis = AnalysisResult(scenes=[dummy_scene], non_language_segments=segments, errors=[], segments=dummy_transcription_segments) # Add segments
        result = segment_scenes(analysis)
        
        # Expected scenes: [0-2], [2-5 non-lang], [5-8], [8-10 non-lang], [10-15]
        assert len(result) == 5
        assert result[0].start_time == 0.0 and result[0].end_time == 2.0 and result[0].description == "Hello" # Scene 0-2 gets "Hello"
        assert result[1].start_time == 2.0 and result[1].end_time == 5.0 and "Non-language" in result[1].description
        assert result[2].start_time == 5.0 and result[2].end_time == 8.0 and result[2].description == "World" # Scene 5-8 gets "World"
        assert result[3].start_time == 8.0 and result[3].end_time == 10.0 and "Non-language" in result[3].description
        assert result[4].start_time == 10.0 and result[4].end_time == 15.0 and result[4].description == "Test" # Scene 10-15 gets "Test"

    def test_segment_scenes_overlapping(self):
        """Test with overlapping segments"""
        segments = [(2.0, 5.0), (4.0, 7.0)]
        dummy_scene = Scene(scene_id=0, start_time=0.0, end_time=15.0, description="Full duration")
        # Add dummy transcription segments for description generation
        # Data for OVERLAPPING test
        dummy_transcription_segments = [
            {"start": 0.5, "end": 1.5, "text": "Overlap"},
            {"start": 10.0, "end": 12.0, "text": "Test"}
        ]
        analysis = AnalysisResult(scenes=[dummy_scene], non_language_segments=segments, errors=[], segments=dummy_transcription_segments) # Add segments
        result = segment_scenes(analysis)
        
        # Segments merge to (2.0, 7.0)
        # Expected scenes: [0-2], [2-7 non-lang], [7-15]
        assert len(result) == 3
        assert result[0].start_time == 0.0 and result[0].end_time == 2.0 and result[0].description == "Overlap" # Scene 0-2 gets "Overlap"
        assert result[1].start_time == 2.0 and result[1].end_time == 7.0 and "Non-language" in result[1].description
        assert result[2].start_time == 7.0 and result[2].end_time == 15.0 and result[2].description == "Test" # Scene 7-15 gets "Test"

    def test_segment_scenes_invalid_input(self):
        """Test with invalid input data"""
        with pytest.raises(SceneSegmentationError):
            segment_scenes(None)

    def test_segment_scenes_malformed_segments(self):
        """Test with malformed segment data"""
        segments = [("invalid", "data")]
        dummy_scene = Scene(scene_id=0, start_time=0.0, end_time=15.0, description="Full duration")
        analysis = AnalysisResult(scenes=[dummy_scene], non_language_segments=segments, errors=[])
        with pytest.raises(SceneSegmentationError):
            segment_scenes(analysis)