import logging
import ffmpeg
from exceptions import AudioExtractionError, AudioConversionError
import os

def extract_audio_from_mp4(mp4_path: str, output_wav_path: str) -> None:
    """
    Extracts audio from MP4 and saves as WAV.
    Raises AudioExtractionError on failure.
    """
    try:
        logging.info(f"Extracting audio from {mp4_path} to {output_wav_path}")
        stream = ffmpeg.input(mp4_path)
        # Force mono output and 16kHz sample rate for VAD compatibility
        stream = ffmpeg.output(stream.audio, output_wav_path, ac=1, ar=16000)
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        logging.info("Audio extraction successful.")
    except Exception as e:
        logging.error(f"Audio extraction failed: {e}", exc_info=True)
        raise AudioExtractionError(f"Failed to extract audio: {e}")

def convert_wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k") -> None:
    """
    Converts WAV to MP3 with specified bitrate.
    Raises AudioConversionError on failure.
    """
    try:
        logging.info(f"Converting {wav_path} to {mp3_path} with bitrate {bitrate}")
        stream = ffmpeg.input(wav_path)
        stream = ffmpeg.output(stream, mp3_path, audio_bitrate=bitrate)
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        logging.info("Audio conversion to MP3 successful.")
    except Exception as e:
        logging.error(f"Audio conversion failed: {e}", exc_info=True)
        raise AudioConversionError(f"Failed to convert audio: {e}")