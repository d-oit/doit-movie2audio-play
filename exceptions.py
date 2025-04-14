class AudioExtractionError(Exception):
    """Raised when audio extraction from video fails"""
    pass

class AudioConversionError(Exception):
    """Raised when audio format conversion fails"""
    pass

class AnalysisError(Exception):
    """Raised when Whisper analysis fails"""
    pass

class SceneSegmentationError(Exception):
    """Raised when scene segmentation fails"""
    pass

class NarrationSynthesisError(Exception):
    """Raised when TTS narration synthesis fails"""
    pass

class AudioOverlayError(Exception):
    """Raised when overlaying narration onto scene audio fails"""
    pass

class AudioMergeError(Exception):
    """Raised when merging scene audio files fails"""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing"""
    pass

class TranscriptionError(Exception):
    """Raised when audio transcription fails"""
    pass

class ApiError(TranscriptionError):
    """Raised when Whisper API call fails"""
    pass

class LocalModelError(TranscriptionError):
    """Raised when local Whisper model transcription fails"""
    pass

class ModelNotFoundError(LocalModelError):
    """Raised when local Whisper model cannot be loaded"""
    pass

class NetworkError(ApiError):
    """Raised when network issues occur during API calls"""
    pass