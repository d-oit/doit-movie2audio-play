import os
import logging
from unittest.mock import patch, MagicMock
import pytest

from src.components import (
    BaseComponent,
    VideoAnalyzer,
    Transcriber,
    SceneDetector,
    DescriptionGenerator,
    SpeechSynthesizer,
    AudioAssembler,
    Scene
)

@pytest.fixture
def mock_logger():
    """Fixture providing a mock logger."""
    logger = MagicMock(spec=logging.Logger)
    return logger

@pytest.fixture
def test_config():
    """Fixture providing test configuration for components."""
    return {
        'ffmpeg': {'threads': 4},
        'whisper': {'model': 'base', 'language': 'en'},
        'blip': {'model': 'blip2-opt-2.7b', 'frame_interval': 1.0},
        'mistral': {'model': 'mistral-7b', 'max_length': 100},
        'tts': {'voice': 'en-US-Neural2-F', 'rate': 1.0}
    }

def test_base_component_abstract():
    """Test that BaseComponent cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseComponent(MagicMock(), {})

def test_scene_dataclass():
    """Test Scene dataclass behavior."""
    scene = Scene(0.0, 10.0, "Test scene", "Raw text")
    assert scene.start_time == 0.0
    assert scene.end_time == 10.0
    assert scene.text_description == "Test scene"
    assert scene.raw_text == "Raw text"

def test_video_analyzer(mock_logger, test_config, tmp_path):
    """Test VideoAnalyzer component."""
    # Create test video file
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    
    analyzer = VideoAnalyzer(mock_logger, test_config['ffmpeg'])
    result = analyzer.process(str(video_path))
    
    # Verify output path construction
    assert str(video_path.parent / "extracted" / "test.wav") == result
    mock_logger.debug.assert_called()
    mock_logger.info.assert_called()

def test_transcriber(mock_logger, test_config, tmp_path):
    """Test Transcriber component."""
    # Create test audio file
    audio_path = tmp_path / "test.wav"
    audio_path.touch()
    
    transcriber = Transcriber(mock_logger, test_config['whisper'])
    result = transcriber.process(str(audio_path))
    
    # Verify placeholder transcript
    assert result == "Detected speech placeholder"
    mock_logger.debug.assert_called()

def test_scene_detector(mock_logger, test_config, tmp_path):
    """Test SceneDetector component."""
    # Create test video file
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    
    detector = SceneDetector(mock_logger, test_config['blip'])
    scenes = detector.process(str(video_path))
    
    # Verify placeholder scenes
    assert len(scenes) == 2
    assert scenes[0].start_time == 0.0
    assert scenes[0].end_time == 10.0
    mock_logger.info.assert_called_with("Detected 2 scenes")

def test_description_generator(mock_logger, test_config):
    """Test DescriptionGenerator component."""
    scenes = [
        Scene(0.0, 10.0, "Opening scene", "Raw opening"),
        Scene(10.0, 20.0, "Main scene", "Raw main")
    ]
    transcript = "Test transcript"
    
    generator = DescriptionGenerator(mock_logger, test_config['mistral'])
    descriptions = generator.process(scenes, transcript)
    
    # Verify placeholder descriptions
    assert len(descriptions) == 2
    assert "Description for scene 0" in descriptions[0]
    mock_logger.info.assert_called_with("Generated 2 descriptions")

def test_speech_synthesizer(mock_logger, test_config):
    """Test SpeechSynthesizer component."""
    descriptions = ["Test description 1", "Test description 2"]
    
    synthesizer = SpeechSynthesizer(mock_logger, test_config['tts'])
    result = synthesizer.process(descriptions)
    
    # Verify output path
    assert result.endswith("synthesized.wav")
    mock_logger.debug.assert_called()

def test_audio_assembler(mock_logger, test_config):
    """Test AudioAssembler component."""
    original_audio = "original.wav"
    synthesized_audio = "synthesized.wav"
    scenes = [Scene(0.0, 10.0, "Scene 1"), Scene(10.0, 20.0, "Scene 2")]
    
    assembler = AudioAssembler(mock_logger, test_config['ffmpeg'])
    result = assembler.process(original_audio, synthesized_audio, scenes)
    
    # Verify output path
    assert result.endswith("final_audio.mp3")
    mock_logger.info.assert_called()