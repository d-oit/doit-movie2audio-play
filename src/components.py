import os
import logging
from ffmpeg import FFmpeg
import whisper
import torch
import numpy as np
from PIL import Image
from gtts import gTTS
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from moviepy.editor import AudioFileClip, concatenate_audioclips, VideoFileClip
from pydub import AudioSegment
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoModelForCausalLM, AutoTokenizer
)
from scenedetect import detect, ContentDetector
from .utils.memory_manager import MemoryManager
@dataclass
class Scene:
    """Data class representing a scene with timing and description."""
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    text_description: Optional[str] = None  # Description of the scene content
    raw_text: Optional[str] = None  # Raw text before filtering (for debugging)

class BaseComponent(ABC):
    """Abstract base class for processing components."""
    def __init__(self, logger: logging.Logger, config: Dict):
        self.logger = logger
        self.config = config
        self.memory_manager = MemoryManager(logger)
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """Process the input and return the result."""
        pass

class VideoAnalyzer(BaseComponent):
    """Component for extracting audio from video using FFmpeg."""
    def process(self, video_path: str) -> str:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Path to extracted audio file
        """
        self.logger.debug(f"Preparing to extract audio from {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory if needed
        output_dir = os.path.join(os.path.dirname(video_path), "extracted")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        audio_path = os.path.join(output_dir,
                                 f"{os.path.splitext(os.path.basename(video_path))[0]}.wav")
        
        try:
            # Configure FFmpeg options
            threads = self.config.get('threads', 4)
            sample_rate = self.config.get('sample_rate', 16000)
            
            # Build FFmpeg command
            self.logger.info(f"Extracting audio to {audio_path}")
            
            ffmpeg = (
                FFmpeg()
                .option("y")  # Overwrite output file
                .option("loglevel", "error")
                .input(video_path)
                .output(
                    audio_path,
                    acodec="pcm_s16le",  # Standard WAV format
                    ac=1,                 # Mono audio
                    ar=sample_rate,       # Sample rate
                    threads=threads
                )
            )
            
            # Execute FFmpeg command
            try:
                ffmpeg.execute()
            except Exception as e:
                self.logger.error(f"FFmpeg command failed: {str(e)}")
                raise RuntimeError("FFmpeg command failed") from e
            
            # Verify the output file exists
            if not os.path.exists(audio_path):
                raise RuntimeError("FFmpeg completed but output file not found")
            
            self.logger.info("Audio extraction completed successfully")
            return audio_path
            
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {str(e)}")
            raise RuntimeError("Audio extraction failed") from e

class Transcriber(BaseComponent):
    """Component for transcribing audio using Whisper."""
    def process(self, audio_path: str) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Get Whisper configuration
            model_name = self.config.get('model', 'base')
            language = self.config.get('language', 'en')
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Loading Whisper model: {model_name} on {device}")
            model = whisper.load_model(model_name, device=device)
            
            # Transcribe audio
            self.logger.info("Starting transcription...")
            result = model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                verbose=False
            )
            
            # Extract transcript text
            transcript = result["text"].strip()
            
            if not transcript:
                self.logger.warning("No speech detected in audio")
                return ""
            
            self.logger.info(f"Transcription complete ({len(transcript)} characters)")
            self.logger.debug(f"Raw transcript: {transcript[:100]}...")
            return transcript
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError("Failed to transcribe audio") from e

class SceneDetector(BaseComponent):
    """Component for detecting scenes and extracting visual information using BLIP."""
    def __init__(self, logger: logging.Logger, config: Dict):
        super().__init__(logger, config)
        self.processor = None
        self.model = None

    def _load_model(self):
        """Load BLIP2 model with memory management."""
        try:
            # Check available memory
            required_memory_mb = 6000  # BLIP2 requires ~6GB
            if torch.cuda.is_available() and not self.memory_manager.check_gpu_memory_available(required_memory_mb):
                self.logger.warning("Insufficient GPU memory, falling back to CPU")
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            model_name = self.config.get('model', 'Salesforce/blip2-opt-2.7b')
            self.logger.info(f"Loading BLIP2 model: {model_name} on {device}")
            
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
        except Exception as e:
            self.logger.error(f"Failed to load BLIP2 model: {e}")
            raise RuntimeError("Model loading failed") from e

    def _unload_model(self):
        """Safely unload BLIP2 model."""
        if self.model:
            self.memory_manager.unload_model(self.model)
            self.model = None
        if self.processor:
            self.processor = None
        
    def process(self, video_path: str) -> List[Scene]:
        """
        Detect scenes and generate descriptions.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of Scene objects
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        try:
            # Load model with memory management
            self._load_model()
            device = next(self.model.parameters()).device
            
            # Configure scene detection parameters
            min_scene_duration = self.config.get('min_scene_duration', 2.0)
            threshold = self.config.get('threshold', 27.0)
            
            # Detect scene boundaries
            self.logger.info("Detecting scene boundaries...")
            scene_list = detect(video_path,
                              ContentDetector(
                                  threshold=threshold,
                                  min_scene_len=int(min_scene_duration * 30)
                              ))
            
            # Process each scene
            scenes = []
            video = VideoFileClip(video_path)
            
            for i, scene in enumerate(scene_list):
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                
                # Extract middle frame for scene description
                mid_time = (start_time + end_time) / 2
                frame = video.get_frame(mid_time)
                image = Image.fromarray(np.uint8(frame)).convert('RGB')
                
                # Generate description using BLIP2
                inputs = self.processor(image, return_tensors="pt").to(device)
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)
                description = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                # Create Scene object
                scene_obj = Scene(
                    start_time=start_time,
                    end_time=end_time,
                    text_description=description.strip(),
                    raw_text=description  # Store raw for debugging
                )
                scenes.append(scene_obj)
                
                self.logger.debug(f"Scene {i+1}: {start_time:.1f}s - {end_time:.1f}s")
                self.logger.debug(f"Description: {description}")
            
            # Cleanup
            video.close()
            self._unload_model()
            self.memory_manager.clear_gpu_memory()
            
            self.logger.info(f"Detected and processed {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            self.logger.error(f"Scene detection failed: {str(e)}")
            raise RuntimeError("Failed to detect and describe scenes") from e

class DescriptionGenerator(BaseComponent):
    """Component for generating descriptions using Mistral."""
    def __init__(self, logger: logging.Logger, config: Dict):
        super().__init__(logger, config)
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """Load Mistral model with memory management."""
        try:
            # Check available memory
            required_memory_mb = 4000  # Mistral small requires ~4GB
            if torch.cuda.is_available() and not self.memory_manager.check_gpu_memory_available(required_memory_mb):
                self.logger.warning("Insufficient GPU memory, falling back to CPU")
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            model_name = self.config.get('model', 'mistralai/Mistral-7B-Instruct-v0.2')
            self.logger.info(f"Loading Mistral model: {model_name} on {device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto"
            )
        except Exception as e:
            self.logger.error(f"Failed to load Mistral model: {e}")
            raise RuntimeError("Model loading failed") from e

    def _unload_model(self):
        """Safely unload Mistral model."""
        if self.model:
            self.memory_manager.unload_model(self.model)
            self.model = None
        if self.tokenizer:
            self.tokenizer = None

    def process(self, scenes: List[Scene], transcript: str) -> List[str]:
        """
        Generate descriptions from scenes and transcript.
        
        Args:
            scenes: List of Scene objects
            transcript: Transcribed audio
            
        Returns:
            List of generated descriptions
        """
        if not scenes:
            self.logger.warning("No scenes provided for description generation")
            return []

        try:
            # Load model with memory management
            self._load_model()
            max_length = self.config.get('max_length', 100)
            temperature = self.config.get('temperature', 0.7)
            device = next(self.model.parameters()).device
            
            descriptions = []
            for i, scene in enumerate(scenes):
                # Create context-aware prompt
                scene_start = max(0, i-1)  # Include previous scene for context
                scene_end = min(len(scenes), i+2)  # Include next scene for context
                context_scenes = scenes[scene_start:scene_end]
                
                # Build prompt with scene context and relevant transcript
                prompt = (
                    "<s>[INST] Generate a concise audio description for a movie scene:\n\n"
                    f"Current Scene ({scene.start_time:.1f}s - {scene.end_time:.1f}s):\n"
                    f"{scene.text_description}\n\n"
                )
                
                # Add context from surrounding scenes if available
                if scene_start < i:
                    prompt += f"Previous Scene: {scenes[i-1].text_description}\n"
                if scene_end > i+1:
                    prompt += f"Next Scene: {scenes[i+1].text_description}\n"
                
                # Add transcript context
                prompt += f"\nRelevant Dialogue/Audio:\n{transcript}\n"
                prompt += "\nCreate a concise, descriptive narration suitable for audio description.[/INST]"
                
                # Generate description
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the response (remove the prompt and any system markers)
                description = description.split("[/INST]")[-1].strip()
                descriptions.append(description)
                
                self.logger.debug(f"Generated description for scene {i+1}: {description[:100]}...")
            
            self.logger.info(f"Generated {len(descriptions)} descriptions")
            # Cleanup
            self._unload_model()
            self.memory_manager.clear_gpu_memory()
            return descriptions
            
        except Exception as e:
            self.logger.error(f"Description generation failed: {str(e)}")
            raise RuntimeError("Failed to generate descriptions") from e

class SpeechSynthesizer(BaseComponent):
    """Component for synthesizing speech from descriptions."""
    def process(self, descriptions: List[str]) -> str:
        """
        Synthesize speech from descriptions.
        
        Args:
            descriptions: List of text descriptions
            
        Returns:
            Path to synthesized audio file
        """
        if not descriptions:
            self.logger.warning("No descriptions provided for speech synthesis")
            return None

        try:
            # Get TTS configuration
            language = self.config.get('language', 'de')  # Default to German
            tld = self.config.get('tld', 'de')  # TLD for voice selection
            slow = self.config.get('slow', False)  # Speech rate
            
            # Create output directory if needed
            output_dir = os.path.join("output", "narrations")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate temporary files for each description
            temp_files = []
            for i, description in enumerate(descriptions):
                if not description.strip():
                    self.logger.warning(f"Empty description at index {i}, skipping")
                    continue
                    
                temp_path = os.path.join(output_dir, f"narration_{i:03d}.mp3")
                
                # Generate speech using gTTS
                self.logger.debug(f"Synthesizing description {i+1}/{len(descriptions)}")
                try:
                    tts = gTTS(text=description, lang=language, tld=tld, slow=slow)
                    tts.save(temp_path)
                    temp_files.append(temp_path)
                except Exception as e:
                    self.logger.error(f"Failed to synthesize description {i+1}: {e}")
                    continue
            
            if not temp_files:
                self.logger.error("No descriptions were successfully synthesized")
                return None
            
            # Combine all audio files
            final_path = os.path.join(output_dir, "combined_narration.mp3")
            
            # Use pydub to concatenate audio files with a small gap between narrations
            combined = AudioSegment.from_mp3(temp_files[0])
            gap = AudioSegment.silent(duration=500)  # 500ms gap between narrations
            
            for audio_path in temp_files[1:]:
                next_segment = AudioSegment.from_mp3(audio_path)
                combined = combined + gap + next_segment
            
            # Export final audio
            self.logger.info("Exporting combined narrations...")
            combined.export(final_path, format="mp3", parameters=["-q:a", "2"])
            
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
            
            self.logger.info(f"Speech synthesis completed: {final_path}")
            return final_path
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {str(e)}")
            raise RuntimeError("Failed to synthesize speech") from e

class AudioAssembler(BaseComponent):
    """Component for assembling final audio track."""
    def process(self, original_audio: str, synthesized_audio: str, 
                scenes: List[Scene]) -> str:
        """
        Assemble final audio track.
        
        Args:
            original_audio: Path to original audio
            synthesized_audio: Path to synthesized descriptions
            scenes: List of scenes with timing information
            
        Returns:
            Path to final audio file
        """
        if not all([original_audio, synthesized_audio]):
            raise ValueError("Both original audio and synthesized descriptions are required")
            
        if not scenes:
            self.logger.warning("No scenes provided, returning original audio")
            return original_audio
            
        try:
            # Load audio files using pydub
            self.logger.info("Loading audio files...")
            original = AudioSegment.from_wav(original_audio)
            narrations = AudioSegment.from_mp3(synthesized_audio)
            
            # Split narrations into segments (assuming equal length for each scene)
            narration_duration = len(narrations)
            segment_duration = narration_duration // len(scenes)
            narration_segments = []
            
            for i in range(len(scenes)):
                start = i * segment_duration
                end = (i + 1) * segment_duration if i < len(scenes) - 1 else narration_duration
                segment = narrations[start:end]
                narration_segments.append(segment)
            
            # Create output directory if needed
            output_dir = os.path.join("output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "final_audio.mp3")
            
            # Create a copy of the original audio to modify
            self.logger.info("Mixing narrations with original audio...")
            final_audio = original
            
            # Calculate volume adjustments from config
            original_volume = self.config.get('original_volume', -10)  # dB
            narration_volume = self.config.get('narration_volume', 0)  # dB
            fade_duration = self.config.get('fade_duration', 500)  # ms
            
            # Add each narration at its scene's start time
            for scene, narration in zip(scenes, narration_segments):
                # Convert scene time to milliseconds
                start_ms = int(scene.start_time * 1000)
                
                # Adjust volume of original audio during narration
                segment_duration = len(narration)
                fade_in = final_audio[start_ms:start_ms + fade_duration].fade_in(fade_duration)
                fade_out = final_audio[start_ms + segment_duration - fade_duration:start_ms + segment_duration].fade_out(fade_duration)
                
                # Split and recombine the audio with fades
                before = final_audio[:start_ms]
                during = (final_audio[start_ms:start_ms + segment_duration] + original_volume)
                after = final_audio[start_ms + segment_duration:]
                
                # Apply volume adjustments and overlay narration
                narration = narration + narration_volume
                during = during.overlay(narration)
                
                # Reassemble the audio
                final_audio = before + during + after
            
            # Export final audio
            self.logger.info(f"Exporting final audio to {output_path}")
            final_audio.export(
                output_path,
                format="mp3",
                parameters=["-q:a", "0"]  # Highest quality MP3
            )
            
            self.logger.info("Audio assembly completed successfully")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Audio assembly failed: {str(e)}")
            raise RuntimeError("Failed to assemble final audio") from e