import pytest
from unittest.mock import patch, MagicMock
from audio_extractor import AudioExtractor
from exceptions import InvalidVideoFileError

class TestAudioExtractor:
    @pytest.fixture
    def extractor(self):
        return AudioExtractor()

    def test_init_sets_default_output_dir(self, extractor):
        assert extractor.output_dir == "audio_output"

    def test_extract_audio_raises_for_invalid_file(self, extractor):
        with pytest.raises(InvalidVideoFileError):
            extractor.extract_audio("nonexistent.mp4")

    @patch('audio_extractor.ffmpeg')
    def test_extract_audio_calls_ffmpeg_correctly(self, mock_ffmpeg, extractor):
        test_file = "test.mp4"
        extractor.extract_audio(test_file)
        mock_ffmpeg.input.assert_called_once_with(test_file)
        mock_ffmpeg.output.assert_called_once()

    @patch('audio_extractor.ffmpeg')
    def test_extract_audio_uses_custom_output_dir(self, mock_ffmpeg, extractor):
        extractor.output_dir = "custom_dir"
        extractor.extract_audio("test.mp4")
        assert "custom_dir" in mock_ffmpeg.output.call_args[0][1]