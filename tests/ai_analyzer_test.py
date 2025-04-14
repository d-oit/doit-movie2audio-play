import pytest
from unittest.mock import patch, MagicMock
from ai_analyzer import AIAnalyzer
from exceptions import AnalysisError

class TestAIAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return AIAnalyzer()

    def test_init_sets_default_model(self, analyzer):
        assert analyzer.model_name == "base"

    @patch('ai_analyzer.whisper.load_model')
    def test_local_analysis_calls_whisper(self, mock_load, analyzer):
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        analyzer.analyze_local("audio.wav")
        mock_load.assert_called_once_with("base")
        mock_model.transcribe.assert_called_once()

    @patch('ai_analyzer.requests.post')
    def test_api_analysis_makes_request(self, mock_post, analyzer):
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "test"}
        mock_post.return_value = mock_response
        result = analyzer.analyze_api("audio.wav", "api_key")
        mock_post.assert_called_once()
        assert result == "test"

    @patch('ai_analyzer.requests.post')
    def test_api_analysis_handles_failure(self, mock_post, analyzer):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API error")
        mock_post.return_value = mock_response
        with pytest.raises(AnalysisError):
            analyzer.analyze_api("audio.wav", "api_key")

    def test_analyze_uses_local_by_default(self, analyzer):
        with patch.object(analyzer, 'analyze_local') as mock_local:
            analyzer.analyze("audio.wav")
            mock_local.assert_called_once()

    def test_analyze_uses_api_when_key_provided(self, analyzer):
        with patch.object(analyzer, 'analyze_api') as mock_api:
            analyzer.analyze("audio.wav", api_key="test")
            mock_api.assert_called_once_with("audio.wav", "test")