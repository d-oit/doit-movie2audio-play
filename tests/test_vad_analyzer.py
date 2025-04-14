import os
import pytest
import numpy as np
import torch
import torchaudio
from vad_analyzer import detect_non_dialogue_segments, get_audio_duration, calculate_inverse_segments

def test_get_audio_duration(tmp_path):
    """Test audio duration calculation."""
    # Create a test audio file (1 second of silence) using numpy and torchaudio
    sample_rate = 16000  # Standard sample rate for VAD
    duration_seconds = 1.0
    silence_np = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
    silence_tensor = torch.from_numpy(silence_np)
    test_file = tmp_path / "test_silence.wav"
    torchaudio.save(str(test_file), silence_tensor.unsqueeze(0), sample_rate)
    
    duration = get_audio_duration(str(test_file))
    assert duration == pytest.approx(1.0)

def test_calculate_inverse_segments():
    """Test calculation of non-speech segments."""
    speech_segments = [(1.0, 2.0), (4.0, 5.0)]
    total_duration = 6.0
    
    non_speech = calculate_inverse_segments(speech_segments, total_duration)
    
    expected = [(0.0, 1.0), (2.0, 4.0), (5.0, 6.0)]
    assert len(non_speech) == len(expected)
    
    for actual, expected_seg in zip(non_speech, expected):
        assert actual[0] == pytest.approx(expected_seg[0])
        assert actual[1] == pytest.approx(expected_seg[1])

def test_detect_non_dialogue_segments_file_not_found():
    """Test error handling for missing file."""
    with pytest.raises(FileNotFoundError):
        detect_non_dialogue_segments("nonexistent.wav")

def test_detect_non_dialogue_segments(tmp_path):
    """Test basic VAD functionality with a simple audio file."""
    # Create a test audio file (alternating silence and "speech") using numpy and torchaudio
    sample_rate = 16000
    duration_seconds = 3.0
    
    # 1s silence
    silence_np = np.zeros(sample_rate * 1, dtype=np.float32)
    
    # 1s 440Hz tone (simulating speech)
    t = np.linspace(0., 1., sample_rate, endpoint=False)
    tone_np = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    # Concatenate: silence + tone + silence
    audio_np = np.concatenate((silence_np, tone_np, silence_np))
    audio_tensor = torch.from_numpy(audio_np)
    
    test_file = tmp_path / "test_vad.wav"
    torchaudio.save(str(test_file), audio_tensor.unsqueeze(0), sample_rate)
    
    # Test VAD (this might need adjustment based on your VAD sensitivity)
    segments = detect_non_dialogue_segments(str(test_file))
    
    # We should get at least some segments
    assert len(segments) > 0
    # Each segment should be a tuple of (start_time, end_time)
    for segment in segments:
        assert len(segment) == 2
        assert segment[0] < segment[1]  # End time should be greater than start time
        assert segment[0] >= 0  # Start time should be non-negative
        assert segment[1] <= duration_seconds  # End time should not exceed file duration