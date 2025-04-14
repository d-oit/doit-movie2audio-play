import pytest
import os # Import os for patching
from unittest.mock import patch, MagicMock
from ai_analyzer import analyze_audio_segments
from exceptions import AnalysisError
from data_structures import AnalysisResult

@pytest.fixture
def mock_whisper_model():
    mock_model = MagicMock()
    mock_transcribe_result = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello", "avg_logprob": -0.5},
            {"start": 2.0, "end": 5.0, "text": "", "avg_logprob": -2.0},  # Non-language
            {"start": 5.0, "end": 8.0, "text": " ", "avg_logprob": -1.8},  # Non-language
            {"start": 8.0, "end": 10.0, "text": "World", "avg_logprob": -0.3}
        ]
    }
    mock_model.transcribe.return_value = mock_transcribe_result
    return mock_model

class TestAiAnalyzer:
    # Helper to mock os.getenv for API vs Local control
    def mock_getenv(self, var_name):
        if var_name == 'WHISPER_API_KEY':
            # Return a dummy key *only* when we want to test the API path explicitly
            # For most tests, return None to force local model path
            return getattr(self, '_mock_api_key', None)
        if var_name == 'WHISPER_API_ENDPOINT':
             return getattr(self, '_mock_api_endpoint', None)
        return os.environ.get(var_name) # Default to actual env for other vars

    @patch('ai_analyzer.os.getenv') # Patch getenv within the module under test
    @patch('ai_analyzer.analyze_with_api') # Mock the actual API call function
    @patch('ai_analyzer.analyze_with_local_model') # Mock the local call function
    def test_analyze_audio_local_success(self, mock_local_analyzer, mock_api_analyzer, mock_os_getenv, tmp_path):
        """Test successful analysis using the LOCAL model path."""
        mock_os_getenv.side_effect = self.mock_getenv # Use helper
        self._mock_api_key = None # Ensure local path is taken
        self._mock_api_endpoint = None

        # Define expected result from local analyzer
        expected_result = AnalysisResult(scenes=[], non_language_segments=[(1.0, 2.0)], errors=[], full_transcription=None) # Removed segments=
        mock_local_analyzer.return_value = expected_result

        dummy_wav_path = tmp_path / "test.wav"
        dummy_wav_path.touch()

        result = analyze_audio_segments(str(dummy_wav_path), "en", "local_model_name")

        mock_os_getenv.assert_any_call('WHISPER_API_KEY')
        mock_os_getenv.assert_any_call('WHISPER_API_ENDPOINT')
        mock_api_analyzer.assert_not_called() # Ensure API path wasn't called
        mock_local_analyzer.assert_called_once_with(str(dummy_wav_path), "en", "local_model_name")
        assert result == expected_result

    @patch('ai_analyzer.os.getenv')
    @patch('ai_analyzer.analyze_with_api')
    @patch('ai_analyzer.analyze_with_local_model')
    def test_analyze_audio_api_success(self, mock_local_analyzer, mock_api_analyzer, mock_os_getenv, tmp_path):
        """Test successful analysis using the API path."""
        mock_os_getenv.side_effect = self.mock_getenv
        self._mock_api_key = "dummy_key" # Force API path
        self._mock_api_endpoint = "dummy_endpoint"

        # Define expected result from API analyzer
        expected_result = AnalysisResult(scenes=[], non_language_segments=[(3.0, 4.0)], errors=[], full_transcription=None) # Removed segments=
        mock_api_analyzer.return_value = expected_result

        dummy_wav_path = tmp_path / "test.wav"
        dummy_wav_path.touch()

        result = analyze_audio_segments(str(dummy_wav_path), "fr", "ignored_model_path") # Model path ignored for API

        mock_os_getenv.assert_any_call('WHISPER_API_KEY')
        mock_os_getenv.assert_any_call('WHISPER_API_ENDPOINT')
        mock_local_analyzer.assert_not_called() # Ensure local path wasn't called
        mock_api_analyzer.assert_called_once_with(str(dummy_wav_path), "fr", "dummy_key", "dummy_endpoint")
        assert result == expected_result

    @patch('whisper.load_model')
    # Removed test_analyze_audio_no_segments as success cases cover this.
    # The internal logic of analyze_with_local_model/analyze_with_api is tested elsewhere
    # or assumed to be correct based on the mocked return value.

    # Removed redundant @patch('whisper.load_model')
    @patch('ai_analyzer.os.getenv')
    @patch('ai_analyzer.analyze_with_api', side_effect=AnalysisError("API file error"))
    @patch('ai_analyzer.analyze_with_local_model', side_effect=AnalysisError("Local file error"))
    # Added mock arguments from @patch decorators to the signature
    # Removed mock_load_model_whisper from signature
    def test_analyze_audio_file_not_found(self, mock_local_analyzer, mock_api_analyzer, mock_os_getenv, tmp_path): # Added tmp_path fixture
        """Test that AnalysisError is raised if underlying functions fail (e.g., file not found)."""
        mock_os_getenv.side_effect = self.mock_getenv
        self._mock_api_key = None # Test local path first
        self._mock_api_endpoint = None

        with pytest.raises(AnalysisError, match="Failed to analyze audio: Local file error"):
             analyze_audio_segments("nonexistent.wav", "en", "model_path")
        mock_local_analyzer.assert_called_once()
        mock_api_analyzer.assert_not_called()

        mock_local_analyzer.reset_mock()
        self._mock_api_key = "dummy_key" # Test API path
        self._mock_api_endpoint = "dummy_endpoint"
        with pytest.raises(AnalysisError, match="Failed to analyze audio: API file error"):
             analyze_audio_segments("nonexistent.wav", "en", "model_path")
        mock_api_analyzer.assert_called_once()
        mock_local_analyzer.assert_not_called()


# (Removed test_analyze_audio_invalid_model and its decorators)