import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import requests
import whisper

from transcription_service import (
    load_transcription_config,
    transcribe_audio,
    _transcribe_with_api,
    _transcribe_with_local_model,
    ApiError,
    LocalModelError,
    ModelNotFoundError,
    NetworkError
)

class TestLoadTranscriptionConfig:
    # Use monkeypatch fixture instead of patch.dict for better control
    # Also patch load_dotenv to prevent it from interfering
    @patch('transcription_service.load_dotenv', return_value=True)
    def test_load_config_with_api_creds(self, mock_load_dotenv, monkeypatch):
        """Test loading config when API credentials are present"""
        # Set environment variables using monkeypatch
        monkeypatch.setenv('WHISPER_API_KEY', 'test_key')
        monkeypatch.setenv('WHISPER_API_ENDPOINT', 'test_endpoint')
        monkeypatch.setenv('WHISPER_LANGUAGE', 'de') # Test non-default language

        result = load_transcription_config()
        
        # Assert the expected dictionary, including language
        assert result == {
            "api_key": "test_key",
            "api_endpoint": "test_endpoint",
            "language": "de", # Check language is loaded
            "use_api": True
        }
        mock_load_dotenv.assert_called_once() # Ensure load_dotenv was called (though patched)

    @patch('transcription_service.load_dotenv', return_value=True)
    def test_load_config_without_api_creds(self, mock_load_dotenv, monkeypatch):
        """Test loading config when API credentials are missing"""
        # Ensure relevant env vars are unset
        monkeypatch.delenv('WHISPER_API_KEY', raising=False)
        monkeypatch.delenv('WHISPER_API_ENDPOINT', raising=False)
        monkeypatch.delenv('WHISPER_LANGUAGE', raising=False) # Test default language

        result = load_transcription_config()
        
        assert result == {
            "api_key": None,
            "api_endpoint": None,
            "language": "en", # Check default language
            "use_api": False
        }
        mock_load_dotenv.assert_called_once()

class TestTranscribeAudio:
    @patch('transcription_service.load_transcription_config')
    @patch('transcription_service._transcribe_with_api')
    @patch('os.path.exists')
    def test_transcribe_audio_with_api(self, mock_exists, mock_api, mock_config):
        """Test routing to API when config says to use API"""
        mock_config.return_value = {
            "api_key": "test_key",
            "api_endpoint": "test_endpoint",
            "use_api": True
        }
        mock_api.return_value = "test transcription"
        mock_exists.return_value = True
        
        result = transcribe_audio("test.wav")
        assert result == "test transcription"
        mock_api.assert_called_once_with("test.wav", "test_key", "test_endpoint")

    @patch('transcription_service.load_transcription_config')
    @patch('transcription_service._transcribe_with_local_model')
    @patch('os.path.exists')
    def test_transcribe_audio_with_local_model(self, mock_exists, mock_local, mock_config):
        """Test routing to local model when API not configured"""
        mock_config.return_value = {
            "api_key": None,
            "api_endpoint": None,
            "language": "en", # Explicitly include default language in mock config
            "use_api": False
        }
        mock_local.return_value = "test transcription"
        mock_exists.return_value = True
        
        result = transcribe_audio("test.wav")
        assert result == "test transcription"
        # Assert that the call includes the language from the config
        mock_local.assert_called_once_with("test.wav", language="en")

    @patch('os.path.exists')
    def test_transcribe_audio_file_not_found(self, mock_exists):
        """Test error when audio file doesn't exist"""
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            transcribe_audio("nonexistent.wav")

class TestTranscribeWithApi:
    # Patch the helper function within the module under test
    @patch('transcription_service._create_temp_audio_url')
    @patch('requests.post')
    # Remove mock_open as we patch the helper that uses it
    def test_transcribe_with_api_success(self, mock_post, mock_create_temp_url, tmp_path): # Add tmp_path and mock_create_temp_url
        """Test successful API transcription"""
        # Create dummy input file
        dummy_input_path = tmp_path / "test.wav"
        dummy_input_path.touch()

        # Configure the patched helper to return dummy values
        dummy_url = f"file://{dummy_input_path.resolve()}"
        dummy_temp_file = str(dummy_input_path) # Can reuse the same path for the mock
        mock_create_temp_url.return_value = (dummy_url, dummy_temp_file)

        # Configure requests.post mock
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "test transcription"}
        mock_response.raise_for_status = MagicMock() # Mock raise_for_status
        mock_post.return_value = mock_response
        
        # Call the function with the dummy path
        result = _transcribe_with_api(str(dummy_input_path), "test_key", "test_endpoint")
        
        assert result == "test transcription"
        mock_create_temp_url.assert_called_once_with(str(dummy_input_path))
        mock_post.assert_called_once()
        # Check the json payload passed to requests.post
        assert mock_post.call_args.kwargs['json'] == {'url': dummy_url}

    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open)
    def test_transcribe_with_api_missing_text(self, mock_file, mock_post):
        """Test API response missing text field"""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response
        
        with pytest.raises(ApiError):
            _transcribe_with_api("test.wav", "test_key", "test_endpoint")

    @patch('transcription_service._create_temp_audio_url')
    @patch('requests.post')
    def test_transcribe_with_api_network_error(self, mock_post, mock_create_temp_url, tmp_path): # Add tmp_path and mock_create_temp_url
        """Test network error during API call"""
        # Create dummy input file
        dummy_input_path = tmp_path / "test.wav"
        dummy_input_path.touch()

        # Configure the patched helper
        dummy_url = f"file://{dummy_input_path.resolve()}"
        dummy_temp_file = str(dummy_input_path)
        mock_create_temp_url.return_value = (dummy_url, dummy_temp_file)

        # Configure requests.post mock to raise an error
        mock_post.side_effect = requests.exceptions.RequestException("test network error")
        
        with pytest.raises(NetworkError, match="test network error"):
             _transcribe_with_api(str(dummy_input_path), "test_key", "test_endpoint")
        
        mock_create_temp_url.assert_called_once_with(str(dummy_input_path))
        mock_post.assert_called_once()

class TestTranscribeWithLocalModel:
    @patch('whisper.load_model')
    def test_transcribe_with_local_model_success(self, mock_load):
        """Test successful local model transcription"""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "test transcription"}
        mock_load.return_value = mock_model
        
        result = _transcribe_with_local_model("test.wav")
        assert result == "test transcription"

    @patch('whisper.load_model')
    def test_transcribe_with_local_model_invalid_result(self, mock_load):
        """Test invalid result format from local model"""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {}
        mock_load.return_value = mock_model
        
        with pytest.raises(LocalModelError):
            _transcribe_with_local_model("test.wav")

    @patch('whisper.load_model')
    def test_transcribe_with_local_model_not_found(self, mock_load):
        """Test model not found error"""
        mock_load.side_effect = Exception("NoSuchModelError")
        
        with pytest.raises(ModelNotFoundError):
            _transcribe_with_local_model("test.wav")