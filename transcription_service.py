import os
import logging
import tempfile
import shutil
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
import requests
import whisper

from exceptions import (
    TranscriptionError,
    ApiError,
    LocalModelError,
    ModelNotFoundError,
    NetworkError,
)

def _create_temp_audio_url(audio_file_path: str) -> Tuple[str, str]:
    """
    Creates a temporary copy of the audio file and returns a path that can be used as a URL.
    
    Args:
        audio_file_path: Path to the original audio file
        
    Returns:
        Tuple of (URL-formatted path, temporary file path)
    """
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a temporary file with the same extension
    _, ext = os.path.splitext(audio_file_path)
    temp_file = os.path.join(temp_dir, f"temp_audio_{os.path.basename(audio_file_path)}")
    
    # Copy the original file to temp location
    shutil.copy2(audio_file_path, temp_file)
    
    # Convert to URL format
    url_path = f"file://{os.path.abspath(temp_file)}"
    
    return url_path, temp_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_transcription_config() -> Dict[str, Optional[str]]:
    """
    Load transcription configuration from environment variables.
    
    Returns:
        Dict containing api_key, api_endpoint, language, and use_api flag
    """
    load_dotenv()  # Load environment variables from .env file
    
    api_key = os.getenv("WHISPER_API_KEY", "").strip()
    api_endpoint = os.getenv("WHISPER_API_ENDPOINT", "").strip()
    language = os.getenv("WHISPER_LANGUAGE", "en").strip()  # Default to English if not set
    
    # Convert empty strings to None
    api_key = api_key if api_key and api_key != '""' else None
    api_endpoint = api_endpoint if api_endpoint and api_endpoint != '""' else None
    
    config = {
        "api_key": api_key,
        "api_endpoint": api_endpoint,
        "language": language,
        "use_api": bool(api_key and api_endpoint)
    }
    
    return config

def _transcribe_with_api(audio_file_path: str, api_key: str, api_endpoint: str) -> str:
    """
    Transcribe audio using the Whisper API.
    
    Args:
        audio_file_path: Path to the audio file
        api_key: Whisper API key
        api_endpoint: Whisper API endpoint URL
    
    Returns:
        Transcribed text from the API
        
    Raises:
        ApiError: If API call fails
        NetworkError: If network issues occur
    """
    try:
        headers = {"Authorization": f"Api-Key {api_key}"}
        
        # Create a temporary URL for the audio file
        url_path, temp_file = _create_temp_audio_url(audio_file_path)
        
        try:
            # Make the API request with the temporary URL
            response = requests.post(
                api_endpoint,
                headers=headers,
                json={'url': url_path},
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            elif isinstance(result, list) and len(result) > 0 and "text" in result[0]:
                return result[0]["text"]
            else:
                raise ApiError("Unexpected API response format")
                
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error during API call: {str(e)}")
        finally:
            # Cleanup: Remove temporary file
            try:
                os.remove(temp_file)
            except OSError:
                pass  # Ignore cleanup errors
                
    except Exception as e:
        if not isinstance(e, (ApiError, NetworkError)):
            raise ApiError(f"Unexpected error during API transcription: {str(e)}")
        raise

def _transcribe_with_local_model(audio_file_path: str, language: str = "en", model_name: str = "base") -> str:
    """
    Transcribe audio using a local Whisper model.
    
    Args:
        audio_file_path: Path to the audio file
        model_name: Name of the Whisper model to use (default: "base")
    
    Returns:
        Transcribed text from the local model
        
    Raises:
        LocalModelError: If local model transcription fails
        ModelNotFoundError: If model cannot be loaded
    """
    try:
        # Load the local Whisper model
        model = whisper.load_model(model_name)
        
        # Perform transcription with specified language
        result = model.transcribe(audio_file_path, language=language, verbose=True)
        
        if not isinstance(result, dict) or "text" not in result:
            raise LocalModelError("Invalid result format from local model")
        
        return result["text"]
        
    except Exception as e:
        if isinstance(e, (LocalModelError, ModelNotFoundError)):
            raise
        elif "NoSuchModelError" in str(e):  # Check for whisper's model error in message
            raise ModelNotFoundError(f"Failed to load local model '{model_name}': {str(e)}")
        else:
            raise LocalModelError(f"Error during local model transcription: {str(e)}")

def transcribe_audio(audio_file_path: str, language: str = None) -> str:
    """
    Main entry point for audio transcription. Routes to API or local model based on config.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
    
    Returns:
        Transcribed text
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        TranscriptionError: If transcription fails
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_file_path}")
    
    config = load_transcription_config()
    
    try:
        # Use provided language or get from config
        lang = language or config.get("language", "en")
        logger.info(f"Using language: {lang} for transcription")
        
        if config["use_api"]:
            logger.info("Using Whisper API for transcription")
            transcription = _transcribe_with_api(
                audio_file_path,
                config["api_key"],
                config["api_endpoint"]
            )
        else:
            logger.info("Using local Whisper model for transcription")
            transcription = _transcribe_with_local_model(
                audio_file_path,
                language=lang
            )
            
        return transcription
        
    except Exception as e:
        logger.error(f"Transcription failed for file: {audio_file_path}. Reason: {str(e)}")
        if isinstance(e, (ApiError, LocalModelError, ModelNotFoundError, NetworkError)):
            raise
        raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")