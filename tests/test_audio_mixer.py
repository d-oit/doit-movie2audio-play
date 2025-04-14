import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch # Import patch and MagicMock
from moviepy.editor import AudioFileClip, CompositeAudioClip # Import moviepy classes
from moviepy.audio.AudioClip import AudioArrayClip # Import AudioArrayClip specifically
from audio_mixer import AudioMixer

@pytest.fixture
def test_audio_files(tmp_path):
    """Create test audio files for mixing using moviepy."""
    sample_rate = 16000 # Use a standard rate
    
    # Function to create a sine wave array
    def make_sine_wave(freq, duration_sec):
        t = np.linspace(0., duration_sec, int(sample_rate * duration_sec))
        amplitude = np.iinfo(np.int16).max # Use int16 max amplitude
        wave = (amplitude * np.sin(2. * np.pi * freq * t)).astype(np.int16)
        # Need stereo for some moviepy operations? Let's try mono first.
        # wave_stereo = np.vstack((wave, wave)).T
        return wave.reshape(-1, 1) # Reshape for AudioArrayClip (num_samples, num_channels)

    # Create original audio (3 seconds, 440 Hz)
    original_duration = 3.0
    original_data = make_sine_wave(440, original_duration)
    original_clip = AudioArrayClip(original_data, fps=sample_rate)
    original_path = tmp_path / "original.wav"
    original_clip.write_audiofile(str(original_path), codec='pcm_s16le', logger=None)
    original_clip.close()

    # Create narration segments (1 second each, 880 Hz)
    narration_duration = 1.0
    narration_data = make_sine_wave(880, narration_duration)
    narration_clip = AudioArrayClip(narration_data, fps=sample_rate)
    
    narr1_path = tmp_path / "narration1.wav"
    narr2_path = tmp_path / "narration2.wav"
    
    narration_clip.write_audiofile(str(narr1_path), codec='pcm_s16le', logger=None)
    # Reuse the clip for the second file
    narration_clip.write_audiofile(str(narr2_path), codec='pcm_s16le', logger=None)
    narration_clip.close()
    
    return {
        'original': str(original_path),
        'narration1': str(narr1_path),
        'narration2': str(narr2_path)
    }

@pytest.fixture
def test_segments(test_audio_files):
    """Create test segment data."""
    return [
        {
            'start_time': 0.5,
            'end_time': 1.5,
            'narration_path': test_audio_files['narration1']
        },
        {
            'start_time': 2.0,
            'end_time': 3.0,
            'narration_path': test_audio_files['narration2']
        }
    ]

def test_audio_mixer_initialization():
    """Test AudioMixer initialization."""
    mixer = AudioMixer()
    assert mixer.export_format == "mp3"
    assert isinstance(mixer.export_params, dict)

def test_adjust_volume():
    """Test volume adjustment using moviepy mocks."""
    mixer = AudioMixer()
    # Create a mock AudioFileClip
    mock_audio_clip = MagicMock(spec=AudioFileClip)
    mock_audio_clip_adjusted_neg = MagicMock(spec=AudioFileClip) # Mock for the returned clip
    mock_audio_clip_adjusted_pos = MagicMock(spec=AudioFileClip) # Mock for the returned clip
    
    # Configure volumex to return different mocks based on input factor for verification
    def volumex_side_effect(factor):
        if factor < 1:
            return mock_audio_clip_adjusted_neg
        else:
            return mock_audio_clip_adjusted_pos
    mock_audio_clip.volumex.side_effect = volumex_side_effect

    # Test volume reduction (-10 dB)
    db_reduction = -10
    expected_factor_reduction = 10 ** (db_reduction / 20.0)
    result_reduced = mixer.adjust_volume(mock_audio_clip, db_reduction)
    mock_audio_clip.volumex.assert_called_with(expected_factor_reduction)
    assert result_reduced == mock_audio_clip_adjusted_neg # Check correct mock was returned

    # Test volume increase (+10 dB)
    db_increase = 10
    expected_factor_increase = 10 ** (db_increase / 20.0)
    result_increased = mixer.adjust_volume(mock_audio_clip, db_increase)
    # Check volumex was called again with the new factor
    mock_audio_clip.volumex.assert_called_with(expected_factor_increase)
    assert result_increased == mock_audio_clip_adjusted_pos # Check correct mock was returned
    
    # Test no change (0 dB)
    result_no_change = mixer.adjust_volume(mock_audio_clip, 0)
    assert result_no_change == mock_audio_clip # Should return original clip
    # Ensure volumex wasn't called for 0 dB change
    assert mock_audio_clip.volumex.call_count == 2 # Only called for non-zero dB

@patch('audio_mixer.concatenate_audioclips')
def test_crossfade_segments(mock_concatenate):
    """Test segment concatenation (crossfade logic might be simple concatenation)."""
    mixer = AudioMixer()
    # Create mock AudioFileClip objects
    mock_seg1 = MagicMock(spec=AudioFileClip)
    mock_seg2 = MagicMock(spec=AudioFileClip)
    mock_result_clip = MagicMock(spec=AudioFileClip) # Mock the returned clip
    mock_concatenate.return_value = mock_result_clip

    # Call the method under test
    result = mixer.crossfade_segments(mock_seg1, mock_seg2)

    # Assert that concatenate_audioclips was called correctly
    mock_concatenate.assert_called_once_with([mock_seg1, mock_seg2])
    # Assert the result is the mocked return value
    assert result == mock_result_clip

@patch('audio_mixer.AudioFileClip') # Patch AudioFileClip used within the mixer
@patch('audio_mixer.CompositeAudioClip') # Patch CompositeAudioClip
def test_overlay_narration(mock_composite_clip_class, mock_audio_clip_class, test_audio_files):
    """Test narration overlay using moviepy mocks."""
    mixer = AudioMixer()

    # --- Mock Setup ---
    # Mock the clips that would be loaded
    mock_original = MagicMock(spec=AudioFileClip)
    mock_narration = MagicMock(spec=AudioFileClip)
    mock_original.duration = 3.0 # Example duration
    mock_narration.duration = 1.0 # Example duration

    # Mock the subclips created internally
    mock_before = MagicMock(spec=AudioFileClip); mock_before.set_start.return_value = mock_before
    mock_during_orig = MagicMock(spec=AudioFileClip)
    mock_after = MagicMock(spec=AudioFileClip); mock_after.set_start.return_value = mock_after
    
    # Mock the adjusted volume clip and positioned clips
    mock_during_adjusted = MagicMock(spec=AudioFileClip); mock_during_adjusted.set_start.return_value = mock_during_adjusted
    mock_narration_positioned = MagicMock(spec=AudioFileClip); mock_narration_positioned.set_start.return_value = mock_narration_positioned

    # Configure the mocks returned by subclip and volumex
    mock_original.subclip.side_effect = lambda start, end=None: {
        (0, 0.5): mock_before,
        (0.5, 1.5): mock_during_orig,
        (1.5, None): mock_after # Assuming end=None means till the end
    }.get((start, end), MagicMock(spec=AudioFileClip)) # Default mock if args don't match
    
    # Mock the adjust_volume call (which uses volumex internally)
    # We patch the mixer's own method here for simplicity, assuming test_adjust_volume covers volumex call
    with patch.object(mixer, 'adjust_volume', return_value=mock_during_adjusted) as mock_adjust_volume:
        # Mock the final composite clip object
        mock_final_composite = MagicMock(spec=CompositeAudioClip)
        mock_composite_clip_class.return_value = mock_final_composite

        # --- Execute ---
        start_sec = 0.5
        bg_reduction = -15.0
        # Pass the *mock* clips to the function
        result_composite = mixer.overlay_narration(
            mock_original,
            mock_narration,
            start_sec=start_sec,
            background_reduction_db=bg_reduction
        )

        # --- Assertions ---
        # Check subclip calls
        mock_original.subclip.assert_any_call(0, start_sec)
        mock_original.subclip.assert_any_call(start_sec, start_sec + mock_narration.duration)
        mock_original.subclip.assert_any_call(start_sec + mock_narration.duration)

        # Check adjust_volume call
        mock_adjust_volume.assert_called_once_with(mock_during_orig, bg_reduction)

        # Check set_start calls
        mock_before.set_start.assert_called_once_with(0)
        mock_during_adjusted.set_start.assert_called_once_with(start_sec)
        mock_narration.set_start.assert_called_once_with(start_sec) # Check original narration mock had set_start called
        mock_after.set_start.assert_called_once_with(start_sec + mock_narration.duration)

        # Check CompositeAudioClip instantiation
        mock_composite_clip_class.assert_called_once_with([
            mock_before,
            mock_during_adjusted,
            mock_narration, # The mock passed to set_start
            mock_after
        ])
        
        # Check the result is the final composite mock
        assert result_composite == mock_final_composite
        # Check duration was set on the composite clip
        assert mock_final_composite.duration == mock_original.duration

@patch('audio_mixer.os.path.exists', return_value=True) # Assume narration files exist
@patch('audio_mixer.AudioFileClip')
@patch('audio_mixer.CompositeAudioClip')
@patch.object(AudioMixer, 'adjust_volume') # Mock the mixer's own method
def test_mix_audio(mock_adjust_volume, mock_composite_clip_class, mock_audio_clip_class, mock_os_exists, test_audio_files, test_segments, tmp_path):
    """Test complete audio mixing process with mocks."""
    mixer = AudioMixer()
    output_path = str(tmp_path / "mixed.mp3")
    
    # --- Mock Setup ---
    # Mocks for clips loaded by AudioFileClip
    mock_original_clip = MagicMock(spec=AudioFileClip)
    mock_original_clip.duration = 3.0
    mock_original_clip.fps = 16000 # Add fps attribute to the mock
    mock_narration1_clip = MagicMock(spec=AudioFileClip)
    mock_narration1_clip.duration = 1.0
    mock_narration2_clip = MagicMock(spec=AudioFileClip)
    mock_narration2_clip.duration = 1.0

    # Side effect for AudioFileClip to return correct mock based on path
    def audio_clip_side_effect(path):
        if path == test_audio_files['original']:
            return mock_original_clip
        elif path == test_audio_files['narration1']:
            return mock_narration1_clip
        elif path == test_audio_files['narration2']:
            return mock_narration2_clip
        else:
            # Use pytest.fail for clearer error messages in tests
            pytest.fail(f"Mock AudioFileClip received unexpected path: {path}")
    mock_audio_clip_class.side_effect = audio_clip_side_effect

    # Mocks for subclips
    mock_orig_part1 = MagicMock(spec=AudioFileClip); mock_orig_part1.set_start.return_value = mock_orig_part1
    mock_bg_seg1 = MagicMock(spec=AudioFileClip)
    mock_orig_part2 = MagicMock(spec=AudioFileClip); mock_orig_part2.set_start.return_value = mock_orig_part2
    mock_bg_seg2 = MagicMock(spec=AudioFileClip)
    mock_orig_part3 = MagicMock(spec=AudioFileClip); mock_orig_part3.set_start.return_value = mock_orig_part3
    
    # Configure subclip side effect
    def subclip_side_effect(start, end=None):
        # Simple mapping based on expected calls in test_segments
        if start == 0 and end == 0.5: return mock_orig_part1
        if start == 0.5 and end == 1.5: return mock_bg_seg1
        if start == 1.5 and end == 2.0: return mock_orig_part2
        if start == 2.0 and end == 3.0: return mock_bg_seg2
        if start == 3.0 and end is None: return mock_orig_part3 # Handle end=None case
        pytest.fail(f"Mock subclip received unexpected args: start={start}, end={end}")
        return MagicMock(spec=AudioFileClip) # Default mock (shouldn't be reached if mapping is correct)
    mock_original_clip.subclip.side_effect = subclip_side_effect

    # Mocks for adjusted/positioned clips
    mock_adjusted_bg1 = MagicMock(spec=AudioFileClip); mock_adjusted_bg1.set_start.return_value = mock_adjusted_bg1
    mock_adjusted_bg2 = MagicMock(spec=AudioFileClip); mock_adjusted_bg2.set_start.return_value = mock_adjusted_bg2
    # Use the actual narration mocks for positioning checks
    mock_narration1_clip.set_start.return_value = mock_narration1_clip
    mock_narration2_clip.set_start.return_value = mock_narration2_clip

    # Configure adjust_volume mock (returns the adjusted background mocks)
    # Need to return distinct mocks if adjust_volume creates new objects,
    # or the same mock if it modifies in place (depends on implementation detail we are mocking away)
    # Let's assume it returns new objects for safety in mocking.
    mock_adjust_volume.side_effect = [mock_adjusted_bg1, mock_adjusted_bg2] # Assumes narration vol adjust is 0

    # Mock the final composite clip and its write method
    mock_final_composite = MagicMock(spec=CompositeAudioClip)
    mock_final_composite.write_audiofile = MagicMock()
    mock_final_composite.close = MagicMock() # Mock close as well
    mock_composite_clip_class.return_value = mock_final_composite
    
    # Mock close methods for all created clips to avoid errors during cleanup
    # Use lists for clarity
    # Define the list of mocks that *should* be closed based on the test logic
    # mock_orig_part3 is excluded because the last segment ends exactly at the original duration
    clips_to_mock_close = [
        mock_original_clip, mock_narration1_clip, mock_narration2_clip,
        mock_orig_part1, mock_bg_seg1, mock_orig_part2, mock_bg_seg2, # mock_orig_part3 removed
        mock_adjusted_bg1, mock_adjusted_bg2
    ]
    for clip_mock in clips_to_mock_close:
        clip_mock.close = MagicMock()


    # --- Execute ---
    bg_reduction = -15.0
    narr_adjust = 0.0
    success = mixer.mix_audio(
        test_audio_files['original'],
        test_segments, # Uses paths from the fixture
        output_path,
        background_volume_reduction_db=bg_reduction,
        narration_volume_adjust_db=narr_adjust
    )

    # --- Assertions ---
    assert success
    # Check os.path.exists calls
    mock_os_exists.assert_any_call(test_segments[0]['narration_path'])
    mock_os_exists.assert_any_call(test_segments[1]['narration_path'])
    
    # Check AudioFileClip calls
    mock_audio_clip_class.assert_any_call(test_audio_files['original'])
    mock_audio_clip_class.assert_any_call(test_segments[0]['narration_path'])
    mock_audio_clip_class.assert_any_call(test_segments[1]['narration_path'])
    
    # Check subclip calls (based on test_segments times)
    mock_original_clip.subclip.assert_any_call(0, 0.5) # Before first narration
    mock_original_clip.subclip.assert_any_call(0.5, 1.5) # During first narration
    mock_original_clip.subclip.assert_any_call(1.5, 2.0) # Between narrations
    mock_original_clip.subclip.assert_any_call(2.0, 3.0) # During second narration
    # Check if the final part was needed (last_end_time < original_audio.duration)
    # In this case, last_end_time (3.0) == original_audio.duration (3.0), so subclip(3.0) shouldn't be called.
    # Let's verify it wasn't called with start=3.0
    for call_args in mock_original_clip.subclip.call_args_list:
        assert call_args[0][0] != 3.0


    # Check adjust_volume calls
    assert mock_adjust_volume.call_count == 2 # Once for each background segment
    mock_adjust_volume.assert_any_call(mock_bg_seg1, bg_reduction)
    mock_adjust_volume.assert_any_call(mock_bg_seg2, bg_reduction)
    # Add checks for narration volume adjustment if narr_adjust != 0 was tested

    # Check set_start calls
    mock_orig_part1.set_start.assert_called_with(0)
    mock_adjusted_bg1.set_start.assert_called_with(0.5)
    mock_narration1_clip.set_start.assert_called_with(0.5) # Check the mock loaded for narration 1
    mock_orig_part2.set_start.assert_called_with(1.5)
    mock_adjusted_bg2.set_start.assert_called_with(2.0)
    mock_narration2_clip.set_start.assert_called_with(2.0) # Check the mock loaded for narration 2
    # mock_orig_part3 should not have set_start called as it wasn't created/added

    # Check CompositeAudioClip call
    # Order matters here based on the implementation logic
    expected_composite_list = [
        mock_orig_part1, # 0 - 0.5
        mock_adjusted_bg1, # 0.5 - 1.5 (adjusted bg)
        mock_narration1_clip, # 0.5 - 1.5 (narration 1)
        mock_orig_part2, # 1.5 - 2.0
        mock_adjusted_bg2, # 2.0 - 3.0 (adjusted bg)
        mock_narration2_clip # 2.0 - 3.0 (narration 2)
    ]
    mock_composite_clip_class.assert_called_once_with(expected_composite_list)
    assert mock_final_composite.duration == mock_original_clip.duration # Check duration was set

    # Check final write call
    mock_final_composite.write_audiofile.assert_called_once_with(
        output_path,
        codec=mixer.export_params['codec'],
        bitrate=mixer.export_params['bitrate'],
        logger=None
    )
    # Check close was called on the final composite clip
    mock_final_composite.close.assert_called_once()
    
    # Check that close was called on intermediate clips
    for clip_mock in clips_to_mock_close:
         clip_mock.close.assert_called_once()

def test_error_handling_missing_files():
    """Test error handling for missing files."""
    mixer = AudioMixer()
    with pytest.raises(FileNotFoundError):
        mixer.mix_audio(
            "nonexistent.wav",
            [],
            "output.mp3"
        )

def test_error_handling_invalid_segments(tmp_path): # Add tmp_path fixture
    """Test error handling for invalid segments."""
    mixer = AudioMixer()
    
    # Create a dummy original audio file for the test
    dummy_original_path = tmp_path / "test.wav"
    # Create a simple dummy audio file (content doesn't matter much here)
    # Using numpy and AudioArrayClip to create a minimal valid WAV
    sample_rate = 16000
    duration = 3.0 # Shorter than the invalid segment start time
    dummy_data = np.zeros((int(sample_rate * duration), 1), dtype=np.int16)
    dummy_clip = AudioArrayClip(dummy_data, fps=sample_rate)
    dummy_clip.write_audiofile(str(dummy_original_path), codec='pcm_s16le', logger=None)
    dummy_clip.close()

    # Create a dummy narration file path (it might not need to exist if segment processing fails early)
    dummy_narration_path = tmp_path / "narration.wav"
    dummy_narration_path.touch() # Just create an empty file

    invalid_segments = [
        {
            'start_time': 5.0,  # Beyond dummy file duration (3.0s)
            'end_time': 6.0,
            'narration_path': str(dummy_narration_path) # Use the dummy path
        }
    ]
    
    output_path = tmp_path / "output.mp3"

    # Now call mix_audio with the valid dummy path
    # It should still fail, but due to segment processing, not FileNotFoundError
    success = mixer.mix_audio(
        str(dummy_original_path), # Use the created dummy file
        invalid_segments,
        str(output_path)
    )
    assert not success # Expect failure due to invalid segment times

def test_adjust_segment_timing():
    """Test segment timing adjustment."""
    mixer = AudioMixer()
    segments = [
        {'start_time': 0.0, 'end_time': 2.0},
        {'start_time': 1.5, 'end_time': 3.0}  # Overlapping
    ]
    
    adjusted = mixer.adjust_segment_timing(segments, min_gap=0.5)
    
    # Verify gaps between segments
    for i in range(len(adjusted) - 1):
        assert adjusted[i+1]['start_time'] - adjusted[i]['end_time'] >= 0.5

def test_concurrent_mixing(test_audio_files, tmp_path):
    """Test handling multiple mixing operations concurrently."""
    mixer = AudioMixer()
    import concurrent.futures
    
    def mix_one(index):
        output_path = str(tmp_path / f"mixed_{index}.mp3")
        segments = [{
            'start_time': 0.0,
            'end_time': 1.0,
            'narration_path': test_audio_files['narration1']
        }]
        return mixer.mix_audio(test_audio_files['original'], segments, output_path)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(mix_one, range(2)))
    
    assert all(results)  # All mixing operations should succeed