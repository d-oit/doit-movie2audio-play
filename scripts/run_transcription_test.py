import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to Python path to import transcription_service
sys.path.append(str(Path(__file__).parent.parent))
from transcription_service import transcribe_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test transcription on an existing audio file"""
    
    # Load environment variables from .env
    if not os.path.exists(".env"):
        logger.error(".env file not found")
        sys.exit(1)
        
    load_dotenv(dotenv_path=".env.local", override=True)
    language = os.getenv("WHISPER_LANGUAGE", "en")
    
    # Force use of local model by explicitly setting empty API values
    os.environ["WHISPER_API_KEY"] = ""
    os.environ["WHISPER_API_ENDPOINT"] = ""
    
    # Get audio file path from command line or use default
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "temp/audio.wav"
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)
        
    try:
        logger.info(f"Starting transcription of {audio_path} with language: {language}")
        transcription = transcribe_audio(audio_path, language=language)
        logger.info("Transcription completed successfully")
        logger.info("\nTranscription result:")
        print(transcription)
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()