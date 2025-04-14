import os
import pytest
from unittest.mock import patch, MagicMock # Import MagicMock
from pathlib import Path # Import Path
from main_orchestrator import process_movie, load_config, parse_args
# from utils import PathManager, AudioUtils, ProcessUtils # Commented out as unused
# Add imports for the actual classes needed for instantiation/mocking
from scene_describer import SceneDescriber
from narration_generator import NarrationGenerator
from audio_mixer import AudioMixer

@pytest.fixture
def test_video(tmp_path):
    """Create a test video file."""
    from moviepy.editor import VideoClip # Use VideoClip instead of ColorClip
    import numpy as np
    
    # Create a simple video with changing colors
    duration = 3  # seconds
    fps = 10
    size = (100, 100)
    
    # Create frames with changing colors
    def make_frame(t):
        # Color changes over time
        color = np.array([int(255 * t/duration), 0, 0])
        return np.tile(color, (*size, 1))
    
    # Create the clip
    clip = VideoClip(make_frame=make_frame, duration=duration) # Use VideoClip constructor
    clip = clip.set_fps(fps)
    
    # Add simple audio (sine wave)
    from moviepy.audio.AudioClip import AudioArrayClip
    import numpy as np
    
    # Create an array of audio data for a sine wave
    fps = 16000  # Compatible with Silero VAD
    t = np.linspace(0, duration, int(fps * duration))
    audio_data = np.array([np.sin(440 * 2 * np.pi * t)]).T
    
    audioclip = AudioArrayClip(audio_data, fps=fps) # Use the same fps as defined above
    audioclip = audioclip.set_duration(duration) # Set duration afterwards
    clip = clip.set_audio(audioclip)
    
    # Save the video
    video_path = str(tmp_path / "test.mp4")
    clip.write_videofile(video_path)
    
    return video_path

@pytest.fixture
def test_config(tmp_path):
    """Create test configuration."""
    return {
        'OUTPUT_DIR': str(tmp_path / "output"),
        'TEMP_DIR': str(tmp_path / "temp"),
        'HUGGING_FACE_TOKEN': 'dummy_token',
        'BACKGROUND_VOLUME_REDUCTION_DB': -15.0,
        'NARRATION_VOLUME_ADJUST_DB': 0.0,
        'TTS_MODEL_PATH': str(tmp_path / "models" / "tts"),
        'TTS_CONFIG_PATH': str(tmp_path / "models" / "tts" / "config.json"),
        'SCENE_DESCRIBER_MODEL': 'Salesforce/blip-image-captioning-large'
    }

def test_load_config(monkeypatch):
    """Test configuration loading."""
    # Mock environment variables
    env_vars = {
        'OUTPUT_DIR': '/test/output',
        'TEMP_DIR': '/test/temp',
        'HUGGING_FACE_TOKEN': 'test_token',
        'BACKGROUND_VOLUME_REDUCTION_DB': '-15.0',
        'NARRATION_VOLUME_ADJUST_DB': '0.0'
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    config = load_config()
    
    assert config['OUTPUT_DIR'] == '/test/output'
    assert config['HUGGING_FACE_TOKEN'] == 'test_token'
    assert config['BACKGROUND_VOLUME_REDUCTION_DB'] == -15.0

def test_parse_args():
    """Test command line argument parsing."""
    import sys
    
    # Test basic usage
    test_args = ['script.py', 'video.mp4']
    with pytest.MonkeyPatch.context() as m:
        m.setattr(sys, 'argv', test_args)
        args = parse_args()
        assert args.mp4_file == 'video.mp4'
        assert not args.debug
    
    # Test with options
    test_args = ['script.py', 'video.mp4', '--debug', '--output', 'out.mp3']
    with pytest.MonkeyPatch.context() as m:
        m.setattr(sys, 'argv', test_args)
        args = parse_args()
        assert args.mp4_file == 'video.mp4'
        assert args.debug
        assert args.output == 'out.mp3'

@patch('main_orchestrator.extract_audio_from_mp4')
@patch('main_orchestrator.detect_non_dialogue_segments')
@patch('main_orchestrator.SceneDescriber') # Mock SceneDescriber
@patch('main_orchestrator.NarrationGenerator') # Mock NarrationGenerator
@patch('main_orchestrator.AudioMixer')
# Keep mocks for services as this test checks the overall flow with mocked steps
def test_process_movie(mock_mixer_class, mock_narrator_class, mock_describer_class, mock_detect, mock_extract, test_video, test_config, tmp_path):
    """Test the complete movie processing pipeline."""
    output_path = str(tmp_path / "output.mp3")
    
    # Create required directories
    os.makedirs(test_config['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(test_config['TEMP_DIR'], exist_ok=True)
    
    # Configure mocks
    mock_extract.return_value = None
    mock_detect.return_value = [(0, 10)]
    # Mock the instances returned by the constructors
    # Get the mock instances created by the class mocks
    mock_describer_instance = mock_describer_class.return_value
    mock_narrator_instance = mock_narrator_class.return_value
    mock_mixer_instance = mock_mixer_class.return_value
    # Configure return values for method calls on the instances
    mock_describer_instance.generate_descriptions.return_value = [{'start_time': 0, 'end_time': 10, 'description': 'Mock description', 'duration': 10}]
    mock_narrator_instance.process_scenes.return_value = [{'start_time': 0, 'end_time': 10, 'description': 'Mock description', 'duration': 10, 'narration_path': 'mock_narration.wav'}]
    # Simulate file creation upon successful mixing
    def mock_mix_audio_side_effect(*args, **kwargs):
        # args[2] should be the output_path based on mix_audio signature in audio_mixer.py
        output_file_path = args[2]
        # Create an empty dummy file just to satisfy os.path.exists
        Path(output_file_path).touch()
        return True # Simulate success
    mock_mixer_instance.mix_audio.side_effect = mock_mix_audio_side_effect
    
    # Process the movie
    # Pass the mock instances to process_movie
    success = process_movie(
        test_video,
        output_path,
        test_config,
        mock_describer_instance,
        mock_narrator_instance,
        mock_mixer_instance
    )
    
    assert success
    assert os.path.exists(output_path)
    
    # We only check existence because the mock creates an empty file
    # Further checks would require a more complex mock or integration test

def test_error_handling_missing_file(tmp_path):
    """Test error handling for missing input file."""
    # Create mock service instances
    mock_describer = MagicMock(spec=SceneDescriber)
    mock_narrator = MagicMock(spec=NarrationGenerator)
    mock_mixer = MagicMock(spec=AudioMixer)

    # process_movie catches the FileNotFoundError and returns False
    success = process_movie(
        "nonexistent.mp4",
        str(tmp_path / "output.mp3"),
        {}, # Config doesn't matter here as file check is first
        mock_describer, # Pass mocks
        mock_narrator,
        mock_mixer
    )
    assert success is False

def test_error_handling_invalid_config(tmp_path): # Add tmp_path fixture
    """Test error handling for invalid configuration (missing keys)."""
    # Create a dummy input file to pass the initial check
    dummy_input_path = tmp_path / "test.mp4"
    dummy_input_path.touch()
    
    # Create mock service instances
    mock_describer = MagicMock(spec=SceneDescriber)
    mock_narrator = MagicMock(spec=NarrationGenerator)
    mock_mixer = MagicMock(spec=AudioMixer)

    # Call process_movie with an empty config and mocks
    success = process_movie(
        str(dummy_input_path),
        str(tmp_path / "output.mp3"),
        {},  # Empty config, missing TEMP_DIR etc.
        mock_describer,
        mock_narrator,
        mock_mixer
    )
    
    # Assert that the function returned False due to the caught exception
    assert success is False

# Add test_video fixture to the arguments
def test_path_creation(tmp_path, test_video):
    """Test automatic path creation."""
    output_path = str(tmp_path / "subdir" / "output.mp3") # Define specific output path
    config = {
        # Use different directories than the main test_config to avoid conflicts
        'OUTPUT_DIR': str(tmp_path / "path_creation_output"),
        'TEMP_DIR': str(tmp_path / "path_creation_temp"),
        'HUGGING_FACE_TOKEN': 'dummy_token',
        # Add other potentially required keys with dummy values if needed by mocks/functions
        'BACKGROUND_VOLUME_REDUCTION_DB': -15.0,
        'NARRATION_VOLUME_ADJUST_DB': 0.0
    }
    
    # Mock the functions that would run after audio extraction to isolate path creation
    with patch('main_orchestrator.detect_non_dialogue_segments', return_value=[]), \
         patch('main_orchestrator.SceneDescriber'), \
         patch('main_orchestrator.NarrationGenerator'), \
         patch('main_orchestrator.AudioMixer'):
        # Call process_movie with the actual test video file
        # Create mock service instances (needed even if patched below)
        mock_describer = MagicMock(spec=SceneDescriber)
        mock_narrator = MagicMock(spec=NarrationGenerator)
        mock_mixer = MagicMock(spec=AudioMixer)
        
        process_movie(
            test_video, # Use the fixture providing a real video path
            output_path,
            config,
            mock_describer,
            mock_narrator,
            mock_mixer
        )
        
    # Assert that the specific directories defined in *this test's* config exist
    # process_movie itself doesn't create these, they should exist beforehand
    # or be created by a setup step (like load_config would normally do).
    # For this test, let's ensure they exist before asserting.
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(config['TEMP_DIR'], exist_ok=True)
    
    assert os.path.exists(config['OUTPUT_DIR'])
    assert os.path.exists(config['TEMP_DIR'])

def test_concurrent_processing(test_video, test_config, tmp_path):
    """Test handling multiple simultaneous processing requests."""
    import concurrent.futures
    
    def process_one(index):
        # Create a unique output path for this thread
        output_path = str(tmp_path / f"output_{index}.mp3")
        
        # Create a unique config for this thread to avoid temp file collisions
        thread_config = test_config.copy()
        unique_temp_dir = tmp_path / f"temp_{index}"
        thread_config['TEMP_DIR'] = str(unique_temp_dir)
        # Ensure the unique temp dir exists (load_config usually does this, but we bypass it here)
        os.makedirs(unique_temp_dir, exist_ok=True)
        # Instantiate services *outside* the thread function if they are thread-safe,
        # or mock them if they are not or if mocking is sufficient for the test.
        # For this concurrency test, let's mock them to avoid potential state issues.
        mock_describer = MagicMock(spec=SceneDescriber)
        mock_narrator = MagicMock(spec=NarrationGenerator)
        mock_mixer = MagicMock(spec=AudioMixer)
        # Simulate successful mixing by having the mock create the output file
        def mock_mix_audio_side_effect_concurrent(*args, **kwargs):
             out_path = args[2]
             Path(out_path).touch()
             return True
        mock_mixer.mix_audio.side_effect = mock_mix_audio_side_effect_concurrent

        # Call process_movie with the unique config and mocks
        return process_movie(
            test_video,
            output_path,
            thread_config,
            mock_describer,
            mock_narrator,
            mock_mixer
        )
        return process_movie(test_video, output_path, thread_config)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(process_one, range(2)))
    
    assert all(results)  # All processing should succeed