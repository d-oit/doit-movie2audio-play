import os
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def test_env(tmp_path_factory, project_root):
    """Create a test environment with necessary directories and files."""
    # Create base test directory
    test_dir = tmp_path_factory.mktemp("test_env")
    
    # Create required subdirectories
    dirs = ["temp", "output", "temp/narrations"]
    for d in dirs:
        os.makedirs(test_dir / d, exist_ok=True)
    
    # Create a test .env file
    env_content = """
HUGGING_FACE_TOKEN=dummy_token
TTS_MODEL_PATH=tts_models/de/thorsten/vits
BACKGROUND_VOLUME_REDUCTION_DB=-15.0
NARRATION_VOLUME_ADJUST_DB=0.0
OUTPUT_DIR={}/output
TEMP_DIR={}/temp
    """.format(test_dir, test_dir)
    
    env_file = test_dir / ".env"
    env_file.write_text(env_content)
    
    # Return the test environment info
    return {
        "root": test_dir,
        "env_file": env_file,
        "output_dir": test_dir / "output",
        "temp_dir": test_dir / "temp"
    }

@pytest.fixture(scope="session")
def sample_audio(tmp_path_factory):
    """Create a sample audio file for testing."""
    from pydub import AudioSegment
    from pydub.generators import Sine
    
    # Create a 3-second audio file with alternating silence and tone
    duration = 3000  # 3 seconds
    silence = AudioSegment.silent(duration=1000)
    tone = Sine(440).to_audio_segment(duration=1000)
    
    audio = silence + tone + silence
    
    # Save the audio file
    audio_dir = tmp_path_factory.mktemp("audio")
    audio_path = audio_dir / "sample.wav"
    audio.export(str(audio_path), format="wav")
    
    return str(audio_path)

@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    """Create a sample video file for testing."""
    from moviepy.editor import ColorClip, AudioFileClip
    import numpy as np
    
    # Create a simple 3-second video
    duration = 3
    size = (320, 240)
    fps = 24
    
    # Create color frames
    def make_frame(t):
        # Create a frame with changing color
        color = [int(255 * t/duration), 100, 100]
        return np.tile(color, (*size, 1)).astype('uint8')
    
    # Create video clip
    clip = ColorClip(size=size, duration=duration, make_frame=make_frame)
    clip = clip.set_fps(fps)
    
    # Add audio (sine wave)
    def make_audio(t):
        return [np.sin(440 * 2 * np.pi * t)]
    
    clip = clip.set_audio(AudioFileClip(sample_audio))
    
    # Save the video
    video_dir = tmp_path_factory.mktemp("video")
    video_path = video_dir / "sample.mp4"
    clip.write_videofile(str(video_path))
    
    return str(video_path)

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )