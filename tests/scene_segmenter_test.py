import pytest
from unittest.mock import patch, MagicMock
from scene_segmenter import SceneSegmenter
from data_structures import Scene

class TestSceneSegmenter:
    @pytest.fixture
    def segmenter(self):
        return SceneSegmenter()

    def test_init_sets_default_threshold(self, segmenter):
        assert segmenter.silence_threshold == -30

    def test_segment_scenes_returns_empty_for_no_audio(self, segmenter):
        result = segmenter.segment_scenes([])
        assert result == []

    def test_segment_scenes_creates_single_scene_for_continuous_audio(self, segmenter):
        analysis = [{"text": "test", "start": 0, "end": 10}]
        result = segmenter.segment_scenes(analysis)
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 10

    def test_segment_scenes_splits_at_silence(self, segmenter):
        analysis = [
            {"text": "part1", "start": 0, "end": 5},
            {"text": "part2", "start": 8, "end": 10}
        ]
        result = segmenter.segment_scenes(analysis)
        assert len(result) == 2
        assert result[0].end == 5
        assert result[1].start == 8

    def test_segment_scenes_merges_short_gaps(self, segmenter):
        analysis = [
            {"text": "part1", "start": 0, "end": 5},
            {"text": "part2", "start": 5.5, "end": 10}
        ]
        result = segmenter.segment_scenes(analysis)
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 10

    def test_create_scene_objects_has_correct_content(self, segmenter):
        analysis = [{"text": "test", "start": 0, "end": 10}]
        result = segmenter.segment_scenes(analysis)
        assert isinstance(result[0], Scene)
        assert result[0].content == "test"