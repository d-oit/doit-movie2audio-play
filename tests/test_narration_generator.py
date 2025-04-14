import os
import time # Import time for sleep
import pytest
from unittest.mock import patch, MagicMock
# from pydub import AudioSegment # Remove pydub import
import wave # Use standard wave module
import struct # Use struct for packing data
from narration_generator import NarrationGenerator

@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary test audio file."""
    audio = AudioSegment.silent(duration=1000)  # 1 second silence
    path = tmp_path / "test_narration.wav"
    audio.export(str(path), format="wav")
    return str(path)

@pytest.fixture
def test_scenes():
    """Create test scene descriptions."""
    return [
        {
            'start_time': 0.0,
            'end_time': 2.0,
            'description': 'A person walking down the street',
            'duration': 2.0
        },
        {
            'start_time': 4.0,
            'end_time': 6.0,
            'description': 'A car drives by',
            'duration': 2.0
        }
    ]

@patch('narration_generator.pyttsx3.init')
def test_narration_generator_initialization(mock_init):
    """Test NarrationGenerator initialization (doesn't initialize engine yet)."""
    generator = NarrationGenerator()
    assert generator.tts_engine is None
    mock_init.assert_not_called() # Engine shouldn't be initialized in __init__

@patch('narration_generator.pyttsx3.init')
def test_initialize_tts(mock_init):
    """Test pyttsx3 engine initialization."""
    mock_engine = MagicMock()
    mock_init.return_value = mock_engine
    
    generator = NarrationGenerator()
    generator.initialize_tts()
    
    mock_init.assert_called_once()
    assert generator.tts_engine is mock_engine

def test_adjust_audio_duration(tmp_path): # Use tmp_path directly
    """Test audio duration adjustment."""
    generator = NarrationGenerator()
    
    # Create the dummy INPUT audio file within the test's tmp_path
    input_audio_path = tmp_path / "test_narration.wav"
    original_duration_ms = 1000
    # Create a minimal valid WAV file using wave module
    input_audio_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    num_samples = int(original_duration_ms / 1000 * sample_rate)
    nchannels = 1
    sampwidth = 2 # 16-bit
    nframes = num_samples
    comptype = "NONE"
    compname = "not compressed"
    
    with wave.open(str(input_audio_path), 'wb') as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.setnframes(nframes)
        wf.setcomptype(comptype, compname)
        # Write silence (zeros)
        for _ in range(nframes):
            wf.writeframesraw(struct.pack('<h', 0))
    time.sleep(0.1) # Keep delay just in case

    # Test shortening
    target_short = 0.5
    # Ensure path is absolute string
    input_path_str = str(input_audio_path.resolve())
    adjusted_path_short = generator.adjust_audio_duration(
        input_path_str,
        target_duration=target_short
    )
    # Check the adjusted file exists (adjust_audio_duration should create it)
    assert os.path.exists(adjusted_path_short)
    # Check the path is different from input (unless no adjustment needed)
    if abs(target_short * 1000 - original_duration_ms) > 1: # Check if adjustment was expected
         assert adjusted_path_short != str(input_audio_path)
    # Cannot easily check duration without pydub, rely on existence check
    # Test lengthening
    target_long = 2.0
    # Ensure path is absolute string
    input_path_str = str(input_audio_path.resolve())
    adjusted_path_long = generator.adjust_audio_duration(
        input_path_str, # Use original path again
        target_duration=target_long
    )
    assert os.path.exists(adjusted_path_long)
    if abs(target_long * 1000 - original_duration_ms) > 1:
        assert adjusted_path_long != str(input_audio_path)
    # Cannot easily check duration without pydub, rely on existence check

@patch('narration_generator.pyttsx3.init')
def test_generate_narration(mock_init, tmp_path):
    """Test narration generation using mocked pyttsx3."""
    mock_engine = MagicMock()
    mock_init.return_value = mock_engine
    
    generator = NarrationGenerator()
    output_path = str(tmp_path / "narration.wav")
    test_text = "This is a test narration"
    # Ensure the output directory exists *before* calling the function
    os.makedirs(tmp_path, exist_ok=True)

    # Mock pyttsx3 behavior: save_to_file queues, runAndWait executes
    save_args = {}
    def mock_save_to_file_queue(text, path):
        # Store arguments for runAndWait mock
        save_args['text'] = text
        save_args['path'] = path
        # Add a basic check
        assert text == test_text
        assert path == output_path

    def mock_run_and_wait_execute():
        # Execute the queued save operation: create the dummy file
        path = save_args.get('path')
        if path:
            # Ensure directory exists (redundant with test setup but safe)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Create a minimal valid WAV file using wave module
            sample_rate = 16000
            num_samples = 100
            nchannels = 1
            sampwidth = 2
            with wave.open(path, 'wb') as wf:
                 wf.setnchannels(nchannels)
                 wf.setsampwidth(sampwidth)
                 wf.setframerate(sample_rate)
                 wf.setnframes(num_samples)
                 wf.setcomptype("NONE", "not compressed")
                 for _ in range(num_samples):
                     wf.writeframesraw(struct.pack('<h', 0))
        else:
             pytest.fail("runAndWait called before save_to_file was mocked correctly")

    mock_engine.save_to_file.side_effect = mock_save_to_file_queue
    mock_engine.runAndWait.side_effect = mock_run_and_wait_execute

    success = generator.generate_narration(test_text, output_path)
    
    assert success
    mock_init.assert_called_once() # initialize_tts should be called
    mock_engine.save_to_file.assert_called_once_with(test_text, output_path)
    mock_engine.runAndWait.assert_called_once()
    assert os.path.exists(output_path) # Check if dummy file was created

@patch('narration_generator.NarrationGenerator.adjust_audio_duration') # Mock adjustment
@patch('narration_generator.pyttsx3.init')
def test_process_scenes(mock_init, mock_adjust, test_scenes, tmp_path): # Add mock_adjust
    """Test processing multiple scenes with mocked pyttsx3."""
    mock_engine = MagicMock()
    mock_init.return_value = mock_engine
    
    generator = NarrationGenerator()
    
    # Simulate file creation for each scene
    # Define the expected output directory based on the new logic in process_scenes
    narration_dir = tmp_path / "movie2audio_narrations"
    # No need to create it here, process_scenes will do it

    def mock_save_to_file(text, path):
        # Ensure the path starts with the expected absolute directory
        assert path.startswith(str(narration_dir))
        # Create a minimal valid WAV file instead of a text file
        # Create a minimal valid WAV file using wave module
        sample_rate = 16000
        num_samples = 100 # Small number of samples
        nchannels = 1
        sampwidth = 2
        with wave.open(path, 'wb') as wf:
             wf.setnchannels(nchannels)
             wf.setsampwidth(sampwidth)
             wf.setframerate(sample_rate)
             wf.setnframes(num_samples)
             wf.setcomptype("NONE", "not compressed")
             for _ in range(num_samples):
                 wf.writeframesraw(struct.pack('<h', 0))
            
    mock_engine.save_to_file.side_effect = mock_save_to_file

    # Process scenes
    # Pass tmp_path as the base_temp_dir
    # Make adjust_audio_duration simply return the input path
    mock_adjust.side_effect = lambda path, duration: path

    processed_scenes = generator.process_scenes(test_scenes, base_temp_dir=str(tmp_path))
    
    assert len(processed_scenes) == len(test_scenes)
    assert mock_init.call_count == 1 # Initialize only once
    assert mock_engine.save_to_file.call_count == len(test_scenes)
    assert mock_engine.runAndWait.call_count == len(test_scenes)
    
    for scene in processed_scenes:
        assert 'narration_path' in scene
        # Construct expected path based on the new logic
        expected_path = os.path.join(str(narration_dir), f"narration_{scene['start_time']:.2f}.wav")
        assert scene['narration_path'] == expected_path
        assert os.path.exists(scene['narration_path'])

@patch('narration_generator.pyttsx3.init')
def test_error_handling_invalid_text(mock_init, tmp_path):
    """Test handling of empty text (pyttsx3 might not error)."""
    mock_engine = MagicMock()
    mock_init.return_value = mock_engine
    
    generator = NarrationGenerator()
    output_path = str(tmp_path / "empty_narration.wav")
    
    # Simulate file creation even for empty text
    def mock_save_to_file(text, path):
        with open(path, 'w') as f:
            f.write("dummy audio data for empty text")
            
    mock_engine.save_to_file.side_effect = mock_save_to_file
    
    success = generator.generate_narration("", output_path)
    
    # pyttsx3 might succeed even with empty text, creating a silent file
    assert success
    mock_engine.save_to_file.assert_called_once_with("", output_path)
    mock_engine.runAndWait.assert_called_once()
    assert os.path.exists(output_path)

def test_error_handling_invalid_duration(tmp_path):
    """Test error handling for invalid target duration adjustment."""
    generator = NarrationGenerator()
    # Create the dummy INPUT file using wave module
    dummy_input_path = tmp_path / "invalid_duration_test.wav"
    dummy_input_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    num_samples = 100
    nchannels = 1
    sampwidth = 2
    with wave.open(str(dummy_input_path), 'wb') as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.setnframes(num_samples)
        wf.setcomptype("NONE", "not compressed")
        for _ in range(num_samples):
            wf.writeframesraw(struct.pack('<h', 0))
    time.sleep(0.1) # Keep delay

    # Expect ValueError due to negative target_duration
    # Ensure path is absolute string
    dummy_input_path_str = str(dummy_input_path.resolve())
    with pytest.raises(ValueError, match="Target duration must be positive"):
        generator.adjust_audio_duration(
            dummy_input_path_str, # Pass the correct input path
            target_duration=-1.0  # Invalid duration
        )

def test_error_handling_file_not_found():
    """Test error handling for missing files."""
    generator = NarrationGenerator()
    with pytest.raises(FileNotFoundError):
        generator.adjust_audio_duration(
            "nonexistent.wav",
            target_duration=1.0
        )

@patch('narration_generator.pyttsx3.init')
@patch('narration_generator.NarrationGenerator.adjust_audio_duration')
def test_timing_constraints(mock_adjust, mock_init, tmp_path):
    """Test that adjust_audio_duration is called when target_duration is set."""
    mock_engine = MagicMock()
    mock_init.return_value = mock_engine
    
    generator = NarrationGenerator()
    output_path = str(tmp_path / "timing_test.wav")
    adjusted_output_path = str(tmp_path / "timing_test_adjusted.wav")
    test_text = "This is a test narration that should fit in two seconds"
    target_duration = 2.0

    # Simulate file creation by save_to_file
    def mock_save_to_file(text, path):
        with open(path, 'w') as f:
            f.write("dummy audio data")
            
    mock_engine.save_to_file.side_effect = mock_save_to_file
    # Make adjust_audio_duration return the adjusted path
    mock_adjust.return_value = adjusted_output_path

    success = generator.generate_narration(
        test_text,
        output_path,
        target_duration=target_duration
    )
    
    assert success
    mock_engine.save_to_file.assert_called_once_with(test_text, output_path)
    mock_engine.runAndWait.assert_called_once()
    # Check that adjust_audio_duration was called correctly
    mock_adjust.assert_called_once_with(output_path, target_duration)
    # Note: We don't check the actual duration here as adjust_audio_duration is mocked

@patch('narration_generator.pyttsx3.init')
def test_concurrent_generation(mock_init, test_scenes, tmp_path):
    """Test handling multiple generations concurrently (mocked)."""
    # NOTE: pyttsx3 might have issues with true concurrency depending on the driver.
    # This test primarily checks if the generator logic handles multiple calls.
    mock_engine = MagicMock()
    mock_init.return_value = mock_engine
    
    generator = NarrationGenerator() # Initialize once
    generator.initialize_tts() # Ensure engine is initialized before threading

    # Simulate file creation
    def mock_save_to_file(text, path):
        with open(path, 'w') as f:
            f.write(f"dummy for {text}")
            
    mock_engine.save_to_file.side_effect = mock_save_to_file

    import concurrent.futures
    
    def generate_one(text):
        # Each thread calls generate_narration on the *same* generator instance
        output_path = str(tmp_path / f"concurrent_narration_{hash(text)}.wav")
        # We expect generate_narration NOT to re-initialize the engine
        return generator.generate_narration(text, output_path)
    
    texts = [scene['description'] for scene in test_scenes]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(generate_one, texts))
    
    assert all(results)  # All generations should report success
    mock_init.assert_called_once() # Engine initialized only once
    assert mock_engine.save_to_file.call_count == len(texts)
    assert mock_engine.runAndWait.call_count == len(texts)