import pytest
from unittest.mock import patch, MagicMock
from narration_generator import NarrationGenerator
from exceptions import TTSUnavailableError

class TestNarrationGenerator:
    @pytest.fixture
    def generator(self):
        return NarrationGenerator()

    def test_init_sets_default_voice(self, generator):
        assert generator.voice == "en-US-Wavenet-D"

    @patch('narration_generator.gTTS')
    def test_generate_uses_gtts_when_no_service(self, mock_gtts, generator):
        mock_audio = MagicMock()
        mock_gtts.return_value = mock_audio
        generator.generate("test text", tts_service=None)
        mock_gtts.assert_called_once_with(text="test text", lang="en")

    @patch('narration_generator.gTTS')
    def test_generate_returns_file_path(self, mock_gtts, generator):
        mock_audio = MagicMock()
        mock_gtts.return_value = mock_audio
        result = generator.generate("test text")
        assert result.endswith(".mp3")

    def test_generate_raises_without_tts(self, generator):
        with patch('narration_generator.gTTS', side_effect=Exception):
            with pytest.raises(TTSUnavailableError):
                generator.generate("test text")

    @patch('narration_generator.TTSService')
    def test_generate_uses_custom_service(self, mock_service, generator):
        mock_service.generate.return_value = "path.mp3"
        result = generator.generate("test text", tts_service=mock_service)
        mock_service.generate.assert_called_once_with("test text", voice="en-US-Wavenet-D")
        assert result == "path.mp3"

    def test_generate_summary_creates_concise_text(self, generator):
        long_text = "This is a very long text that should be summarized to be more concise."
        result = generator.generate_summary(long_text)
        assert len(result.split()) < len(long_text.split())
        assert "..." in result