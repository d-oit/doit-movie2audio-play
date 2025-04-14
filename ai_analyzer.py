import logging
import whisper
import requests
import os
from typing import Dict, Any
from data_structures import AnalysisResult, Scene
from exceptions import AnalysisError

def analyze_audio_segments(wav_path: str, language: str, whisper_model_path: str, transcription: str = None) -> AnalysisResult:
    """
    Uses Whisper to transcribe and analyze audio.
    Returns AnalysisResult with non-language segments, transcriptions, and errors.
    
    Args:
        wav_path: Path to the wav file
        language: Language code for analysis
        whisper_model_path: Path to Whisper model
        transcription: Optional pre-generated transcription
        
    Returns:
        AnalysisResult object containing scenes, segments, and transcription data
        
    Raises:
        AnalysisError on failure.
    """
    try:
        # Check if API key exists in environment
        api_key = os.getenv('WHISPER_API_KEY')
        api_endpoint_raw = os.getenv('WHISPER_API_ENDPOINT')
        api_endpoint = None
        if api_endpoint_raw:
            # Clean the endpoint: remove comments first, then strip whitespace/quotes
            temp_endpoint = api_endpoint_raw.strip()
            if '#' in temp_endpoint:
                temp_endpoint = temp_endpoint.split('#', 1)[0].strip()
            # Strip potential quotes from the cleaned URL part
            api_endpoint = temp_endpoint.strip('"\'')

        if api_key and api_endpoint:
            # Use API-based analysis
            return analyze_with_api(wav_path, language, api_key, api_endpoint)
        else:
            # Use local model analysis
            return analyze_with_local_model(wav_path, language, whisper_model_path)

    except Exception as e:
        logging.error(f"Audio analysis failed: {e}", exc_info=True)
        raise AnalysisError(f"Failed to analyze audio: {e}")

def analyze_with_local_model(wav_path: str, language: str, model_path: str, transcription: str = None) -> AnalysisResult:
    """Uses local Whisper model for analysis"""
    try:
        # Load the model (model_path should be one of tiny, base, small, medium, large)
        model = whisper.load_model(model_path)
        
        # Only analyze for non-speech segments and timing
        result = model.transcribe(wav_path, language=language, task='detect')
        
        # Process segments
        segments = []
        non_language_segments = []
        
        for segment in result["segments"]:
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })
            
            # Check for non-language segments (no_speech_prob > threshold)
            if segment.get("no_speech_prob", 0) > 0.8:
                non_language_segments.append((segment["start"], segment["end"]))
        
        return AnalysisResult(
            scenes=[],  # Scene creation is handled by scene_segmenter
            segments=segments,
            non_language_segments=non_language_segments,
            errors=[],
            full_transcription=transcription
        )
        
    except Exception as e:
        logging.error(f"Local model analysis failed: {e}", exc_info=True)
        raise AnalysisError(str(e))

def analyze_with_api(wav_path: str, language: str, api_key: str, api_endpoint: str, transcription: str = None) -> AnalysisResult:
    """Uses Whisper API for analysis"""
    try:
        # Read audio file
        with open(wav_path, 'rb') as f:
            files = {'file': f}
            headers = {'Authorization': f'Bearer {api_key}'}
            data = {'language': language, 'task': 'detect'}
            
            # Make API request
            response = requests.post(api_endpoint, headers=headers, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            
            # Process API response
            segments = []
            non_language_segments = []
            
            for segment in result.get("segments", []):
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })
                
                # Check for non-language segments based on API response
                if segment.get("no_speech_prob", 0) > 0.8:
                    non_language_segments.append((segment["start"], segment["end"]))
            
            return AnalysisResult(
                scenes=[],  # Scene creation is handled by scene_segmenter
                segments=segments,
                non_language_segments=non_language_segments,
                errors=[],
                full_transcription=transcription
            )
            
    except Exception as e:
        logging.error(f"API analysis failed: {e}", exc_info=True)
        raise AnalysisError(str(e))