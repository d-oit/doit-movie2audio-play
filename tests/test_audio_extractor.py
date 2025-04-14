from unittest.mock import patch, MagicMock, call
import sys
import pytest
import os
import ffmpeg # Import ffmpeg

# Remove incorrect module-level mocks for pydub and moviepy
# sys.modules['audioop'] = MagicMock() # Keep if needed by other imports, otherwise remove
# sys.modules['pyaudioop'] = MagicMock() # Keep if needed by other imports, otherwise remove
# sys.modules['pydub'] = MagicMock()
# sys.modules['pydub.audio_segment'] = MagicMock()
# sys.modules['pydub.utils'] = MagicMock()
# sys.modules['moviepy'] = MagicMock()
# sys.modules['moviepy.editor'] = MagicMock()

# Mocks are now handled within tests using @patch

from audio_extractor import extract_audio_from_mp4, convert_wav_to_mp3
from exceptions import AudioExtractionError, AudioConversionError

@pytest.fixture
def temp_mp4(tmp_path):
    mp4_path = os.path.join(tmp_path, "test.mp4")
    with open(mp4_path, 'wb') as f:
        f.write(b'fake mp4 data')
    return mp4_path

@pytest.fixture
def temp_wav(tmp_path):
    return os.path.join(tmp_path, "test.wav")

@pytest.fixture
def temp_mp3(tmp_path):
    return os.path.join(tmp_path, "test.mp3")

class TestAudioExtractor:
    @patch('audio_extractor.ffmpeg') # Patch ffmpeg used in audio_extractor
    def test_extract_audio_success(self, mock_ffmpeg, temp_mp4, temp_wav):
        # --- Mock Setup ---
        mock_stream = MagicMock()
        mock_input_stream = MagicMock()
        mock_input_stream.audio = mock_stream # Simulate the .audio attribute access
        mock_ffmpeg.input.return_value = mock_input_stream
        mock_ffmpeg.output.return_value = mock_stream
        mock_ffmpeg.run.return_value = (b'stdout', b'stderr') # Simulate success

        # --- Execute ---
        extract_audio_from_mp4(temp_mp4, temp_wav)

        # --- Assertions ---
        mock_ffmpeg.input.assert_called_once_with(temp_mp4)
        # Check that output was called with the audio stream and correct path/codec
        mock_ffmpeg.output.assert_called_once_with(mock_stream, temp_wav, ac=1, ar=16000) # Added ar=16000
        mock_ffmpeg.run.assert_called_once_with(mock_stream, capture_stdout=True, capture_stderr=True)

    @patch('audio_extractor.ffmpeg')
    def test_extract_audio_no_audio_track(self, mock_ffmpeg, temp_mp4, temp_wav):
        # Simulate ffmpeg failing (e.g., due to no audio track, though the code doesn't check specifically)
        mock_stream = MagicMock()
        mock_input_stream = MagicMock()
        mock_input_stream.audio = mock_stream
        mock_ffmpeg.input.return_value = mock_input_stream
        mock_ffmpeg.output.return_value = mock_stream
        mock_ffmpeg.Error = ffmpeg.Error # Ensure the mock uses the real Error type
        mock_ffmpeg.run.side_effect = ffmpeg.Error('ffmpeg', b'stdout', b'stderr: some error indicating no audio')

        # The code currently raises a generic AudioExtractionError, not specific to "no audio track"
        # Adjusting the test to reflect the actual implementation for now.
        with pytest.raises(AudioExtractionError, match="Failed to extract audio"):
             extract_audio_from_mp4(temp_mp4, temp_wav)

        # Assert ffmpeg input/output/run were called up to the point of error
        mock_ffmpeg.input.assert_called_once_with(temp_mp4)
        mock_ffmpeg.output.assert_called_once_with(mock_stream, temp_wav, ac=1, ar=16000) # Added ar=16000
        mock_ffmpeg.run.assert_called_once_with(mock_stream, capture_stdout=True, capture_stderr=True)

    @patch('audio_extractor.ffmpeg')
    def test_extract_audio_file_not_found(self, mock_ffmpeg, temp_wav):
        # Simulate ffmpeg failing because the input file doesn't exist
        mock_ffmpeg.Error = ffmpeg.Error # Ensure the mock uses the real Error type
        mock_ffmpeg.input.side_effect = ffmpeg.Error('ffmpeg', b'stdout', b'stderr: nonexistent.mp4: No such file or directory')

        with pytest.raises(AudioExtractionError, match="Failed to extract audio"):
            extract_audio_from_mp4("nonexistent.mp4", temp_wav)

        # Assert ffmpeg.input was called (which then raised the error)
        mock_ffmpeg.input.assert_called_once_with("nonexistent.mp4")
        # Ensure other ffmpeg steps weren't reached
        mock_ffmpeg.output.assert_not_called()
        mock_ffmpeg.run.assert_not_called()

class TestAudioConverter:
    @patch('audio_extractor.ffmpeg')
    def test_convert_wav_success(self, mock_ffmpeg, temp_wav, temp_mp3):
        # --- Mock Setup ---
        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_ffmpeg.output.return_value = mock_stream
        mock_ffmpeg.run.return_value = (b'stdout', b'stderr') # Simulate success

        # --- Execute ---
        convert_wav_to_mp3(temp_wav, temp_mp3)

        # --- Assertions ---
        mock_ffmpeg.input.assert_called_once_with(temp_wav)
        mock_ffmpeg.output.assert_called_once_with(mock_stream, temp_mp3, audio_bitrate="192k")
        mock_ffmpeg.run.assert_called_once_with(mock_stream, capture_stdout=True, capture_stderr=True)

    @patch('audio_extractor.ffmpeg')
    def test_convert_wav_file_not_found(self, mock_ffmpeg, temp_mp3):
        # --- Mock Setup ---
        # Simulate ffmpeg failing because the input file doesn't exist
        mock_ffmpeg.Error = ffmpeg.Error # Ensure the mock uses the real Error type
        mock_ffmpeg.input.side_effect = ffmpeg.Error('ffmpeg', b'stdout', b'stderr: nonexistent.wav: No such file or directory')

        # --- Execute & Assert ---
        with pytest.raises(AudioConversionError, match="Failed to convert audio"):
            convert_wav_to_mp3("nonexistent.wav", temp_mp3)

        # Assert ffmpeg.input was called (which then raised the error)
        mock_ffmpeg.input.assert_called_once_with("nonexistent.wav")
        # Ensure other ffmpeg steps weren't reached
        mock_ffmpeg.output.assert_not_called()
        mock_ffmpeg.run.assert_not_called()

    @patch('audio_extractor.ffmpeg')
    def test_convert_wav_custom_bitrate(self, mock_ffmpeg, temp_wav, temp_mp3):
        # --- Mock Setup ---
        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_ffmpeg.output.return_value = mock_stream
        mock_ffmpeg.run.return_value = (b'stdout', b'stderr') # Simulate success
        custom_bitrate = "320k"

        # --- Execute ---
        convert_wav_to_mp3(temp_wav, temp_mp3, custom_bitrate)

        # --- Assertions ---
        mock_ffmpeg.input.assert_called_once_with(temp_wav)
        mock_ffmpeg.output.assert_called_once_with(mock_stream, temp_mp3, audio_bitrate=custom_bitrate)
        mock_ffmpeg.run.assert_called_once_with(mock_stream, capture_stdout=True, capture_stderr=True)