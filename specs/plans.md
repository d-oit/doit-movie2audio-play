# Video-to-Audio Description Processing Plan

This document outlines the pseudocode specification for processing a video file to generate an audio description track, focusing on separating progress reporting from detailed component logging and filtering out logo text.

## Core Requirements

1.  **Duration Estimation:** Calculate and display an estimated total processing time at the beginning.
2.  **Progress Reporting:** Show only high-level progress updates (e.g., current step, percentage complete) in the terminal output.
3.  **Component Logging:** Log detailed debug information from each processing component (FFmpeg, Whisper, BLIP, Mistral, TTS, etc.) into separate files within an `output/` directory.
4.  **Logo Filtering:** Identify and remove text associated with studio logos, watermarks, or other non-content elements during scene analysis and description generation.

## Pseudocode Specification

```pseudocode
// File: main_script.py (or equivalent entry point)

// --- TDD Anchor: test_duration_calculation ---
FUNCTION calculate_total_duration(input_video_path, processing_steps):
  // Logic to estimate duration based on video properties and steps
  // Example: duration = video_length * processing_factor_per_step * num_steps
  estimated_duration = ... // Placeholder for actual calculation
  RETURN estimated_duration
END FUNCTION
// --- End TDD Anchor ---

// --- TDD Anchor: test_logging_initialization ---
FUNCTION initialize_logging(output_dir="output"):
  // Ensure output directory exists
  CREATE DIRECTORY IF NOT EXISTS output_dir

  // Configure root logger (or dedicated progress logger) for terminal progress
  terminal_handler = CONFIGURE_TERMINAL_HANDLER(level=INFO)
  terminal_formatter = CREATE_FORMATTER("[%(levelname)s] %(message)s")
  SET_HANDLER_FORMATTER(terminal_handler, terminal_formatter)
  progress_logger = GET_LOGGER("progress_reporter")
  SET_LOGGER_LEVEL(progress_logger, INFO)
  ADD_HANDLER_TO_LOGGER(progress_logger, terminal_handler)
  SET_LOGGER_PROPAGATION(progress_logger, FALSE) // Prevent progress logs going elsewhere

  // Configure file loggers for components
  component_log_files = {}
  component_names = ["mistral", "blip", "whisper", "tts", "ffmpeg", "main"] // Add other relevant components
  FOR component_name IN component_names:
    log_file_path = JOIN_PATH(output_dir, f"{component_name}.log")
    // Configure file handler to log everything (DEBUG level) for this component
    file_handler = CONFIGURE_FILE_HANDLER(log_file_path, level=DEBUG)
    // Add detailed formatter for file logs
    file_formatter = CREATE_FORMATTER("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    SET_HANDLER_FORMATTER(file_handler, file_formatter)

    logger = GET_LOGGER(component_name)
    SET_LOGGER_LEVEL(logger, DEBUG) // Ensure logger captures DEBUG messages
    // Remove any default handlers if necessary (to avoid duplicate terminal output)
    REMOVE_DEFAULT_HANDLERS(logger)
    ADD_HANDLER_TO_LOGGER(logger, file_handler)
    // Prevent component logs from propagating to the progress/root logger
    SET_LOGGER_PROPAGATION(logger, FALSE)
    component_log_files[component_name] = log_file_path
  ENDFOR

  // Get the main logger for general script flow messages (will log to output/main.log)
  main_logger = GET_LOGGER("main")

  RETURN progress_logger, main_logger, component_log_files
END FUNCTION
// --- End TDD Anchor ---

// --- TDD Anchor: test_progress_reporting ---
FUNCTION report_progress(progress_logger, message, percentage=None):
  // Uses the dedicated progress logger configured in initialize_logging
  IF percentage IS NOT NULL:
    // Log with percentage, using the progress logger's INFO level
    LOG(progress_logger, INFO, f"[{percentage}%] {message}")
  ELSE:
    // Log simple message, using the progress logger's INFO level
    LOG(progress_logger, INFO, message)
  ENDIF
END FUNCTION
// --- End TDD Anchor ---

// --- TDD Anchor: test_logo_filtering ---
FUNCTION filter_logo_text(text_content):
  // Define patterns or keywords associated with logos
  logo_patterns = [
    "SONY PICTURES ENTERTAINMENT",
    "COLUMBIA PICTURES",
    "2DF HD", // Example watermark
    "PARAMOUNT",
    "WARNER BROS",
    // Add more known logo/watermark text patterns
    REGEX("a .* company"), // Example pattern
    REGEX("watermark", IGNORE_CASE)
  ]
  filtered_text = text_content
  FOR pattern IN logo_patterns:
    // Remove occurrences of the pattern (or sentences containing them)
    filtered_text = REPLACE_PATTERN(filtered_text, pattern, "")
  ENDFOR
  // Optional: Add more sophisticated NLP techniques if needed
  RETURN filtered_text.strip()
END FUNCTION
// --- End TDD Anchor ---


// --- Main Execution Flow ---
PROCEDURE main():
  // 1. Initialization
  input_video = GET_INPUT_VIDEO_PATH() // Function to get video path from args or config
  processing_steps = GET_PROCESSING_STEPS() // Function to determine steps
  output_directory = "output"

  // 2. Initialize Logging - Get specific loggers
  progress_logger, main_logger, log_files = initialize_logging(output_directory)
  LOG(main_logger, INFO, f"Logging initialized. Component logs in: {log_files}") // Goes to main.log

  // 3. Calculate and Display Duration
  TRY
    total_duration = calculate_total_duration(input_video, processing_steps)
    // Use report_progress for initial terminal output via progress_logger
    report_progress(progress_logger, f"Estimated total duration: {FORMAT_DURATION(total_duration)}")
  CATCH Exception as e:
    report_progress(progress_logger, f"WARN: Could not estimate duration: {e}")
    LOG(main_logger, WARNING, f"Duration estimation failed: {e}", exc_info=True) // Goes to main.log
  END TRY

  // 4. Start Processing
  report_progress(progress_logger, "Starting video processing...", percentage=0)
  LOG(main_logger, INFO, "Starting main processing workflow.") // Goes to main.log

  TRY
    // --- Step: Video Analysis ---
    report_progress(progress_logger, "Analyzing video...", percentage=10)
    ffmpeg_logger = GET_LOGGER("ffmpeg") // Get component-specific logger
    LOG(ffmpeg_logger, INFO, f"Extracting audio from {input_video}") // Goes to ffmpeg.log
    audio_path = EXTRACT_AUDIO(input_video, ffmpeg_logger) // Pass logger to function
    LOG(ffmpeg_logger, DEBUG, f"Audio extraction command details...") // Goes to ffmpeg.log
    LOG(ffmpeg_logger, INFO, f"Audio extracted to {audio_path}") // Goes to ffmpeg.log

    // --- Step: Transcription ---
    report_progress(progress_logger, "Transcribing audio...", percentage=30)
    whisper_logger = GET_LOGGER("whisper")
    LOG(whisper_logger, INFO, f"Starting transcription for {audio_path}") // Goes to whisper.log
    transcript = TRANSCRIBE_AUDIO(audio_path, whisper_logger) // Pass logger
    LOG(whisper_logger, INFO, f"Transcription complete.") // Goes to whisper.log

    // --- Step: Scene Detection (with Logo Filtering) ---
    report_progress(progress_logger, "Detecting scenes...", percentage=50)
    blip_logger = GET_LOGGER("blip")
    LOG(blip_logger, INFO, f"Running scene detection on {input_video}") // Goes to blip.log
    raw_scenes = DETECT_SCENES(input_video, blip_logger) // Function might return scene objects with text descriptions
    LOG(blip_logger, DEBUG, f"Raw scenes detected: {raw_scenes}") // Goes to blip.log
    // Apply filtering to scene descriptions generated by BLIP/equivalent
    filtered_scenes = []
    FOR scene IN raw_scenes:
       IF scene HAS text_description: // Check if the scene object contains text
           original_description = GET_SCENE_DESCRIPTION(scene)
           filtered_description = filter_logo_text(original_description)
           IF original_description != filtered_description:
               LOG(blip_logger, DEBUG, f"Filtered logo text from scene description: '{original_description}' -> '{filtered_description}'") // Goes to blip.log
           ENDIF
           UPDATE_SCENE_DESCRIPTION(scene, filtered_description) // Update the scene object
       ENDIF
       APPEND filtered_scenes, scene
    ENDFOR
    scenes = filtered_scenes
    LOG(blip_logger, INFO, f"Detected and filtered {len(scenes)} scenes.") // Goes to blip.log

    // --- Step: Content Generation (with Logo Filtering) ---
    report_progress(progress_logger, "Generating audio descriptions...", percentage=70)
    mistral_logger = GET_LOGGER("mistral")
    LOG(mistral_logger, INFO, "Generating descriptions based on filtered scenes and transcript.") // Goes to mistral.log
    // Pass filtered scenes to the generation function
    raw_descriptions = GENERATE_DESCRIPTIONS(scenes, transcript, mistral_logger) // Pass logger
    LOG(mistral_logger, DEBUG, f"Raw descriptions generated: {raw_descriptions}") // Goes to mistral.log
    // Apply filtering again to the output of Mistral/equivalent, just in case
    filtered_descriptions = []
    FOR desc IN raw_descriptions:
        filtered_desc = filter_logo_text(desc)
        IF desc != filtered_desc:
            LOG(mistral_logger, DEBUG, f"Filtered logo text from generated description: '{desc}' -> '{filtered_desc}'") // Goes to mistral.log
        ENDIF
        APPEND filtered_descriptions, filtered_desc
    ENDFOR
    descriptions = filtered_descriptions
    LOG(mistral_logger, INFO, f"Generated and filtered {len(descriptions)} descriptions.") // Goes to mistral.log

    // --- Step: Speech Synthesis ---
    report_progress(progress_logger, "Synthesizing speech...", percentage=90)
    tts_logger = GET_LOGGER("tts")
    LOG(tts_logger, INFO, "Synthesizing filtered descriptions to audio.") // Goes to tts.log
    synthesized_audio = SYNTHESIZE_SPEECH(descriptions, tts_logger) // Pass logger
    LOG(tts_logger, INFO, "Speech synthesis complete.") // Goes to tts.log

    // --- Step: Final Assembly ---
    report_progress(progress_logger, "Assembling final audio...", percentage=95)
    LOG(ffmpeg_logger, INFO, "Merging original audio with synthesized descriptions.") // Goes to ffmpeg.log
    final_output = ASSEMBLE_AUDIO(audio_path, synthesized_audio, scenes, ffmpeg_logger) // Pass logger
    LOG(ffmpeg_logger, INFO, f"Final audio created at {final_output}") // Goes to ffmpeg.log

    report_progress(progress_logger, "Processing complete.", percentage=100)
    LOG(main_logger, INFO, "Main processing workflow finished successfully.") // Goes to main.log

  CATCH Exception as e:
    // Log error to main log file AND report failure to terminal
    LOG(main_logger, ERROR, f"An error occurred during processing: {e}", exc_info=True) // Goes to main.log
    report_progress(progress_logger, f"ERROR: Processing failed. Check logs in '{output_directory}' for details.") // Shows in terminal
  END TRY

END PROCEDURE

// --- Helper/Placeholder Functions ---
// Assume these functions exist and accept a logger instance to perform logging

FUNCTION GET_LOGGER(name):
  // Wrapper for logging.getLogger(name)
  RETURN logging.getLogger(name)
END FUNCTION

FUNCTION LOG(logger, level, message, exc_info=False):
  // Wrapper for logger.log(level, message, exc_info=exc_info)
  logger.log(level, message, exc_info=exc_info)
END FUNCTION

FUNCTION EXTRACT_AUDIO(video_path, logger):
  LOG(logger, DEBUG, f"Preparing to extract audio from {video_path}")
  // ... ffmpeg command execution ...
  audio_path = "path/to/audio.wav" // Example
  LOG(logger, INFO, f"Successfully extracted audio to {audio_path}")
  RETURN audio_path
END FUNCTION

FUNCTION TRANSCRIBE_AUDIO(audio_path, logger):
  LOG(logger, DEBUG, f"Loading Whisper model...")
  // ... transcription logic ...
  transcript = "Detected speech..." // Example
  LOG(logger, DEBUG, f"Raw transcript data...")
  RETURN transcript
END FUNCTION

FUNCTION DETECT_SCENES(video_path, logger):
  LOG(logger, DEBUG, f"Starting scene detection for {video_path}")
  // ... BLIP or other scene detection logic ...
  // This function should return a list of scene objects or dictionaries,
  // potentially including raw text descriptions detected in frames.
  raw_scene_data = [ { "timestamp": 0.0, "text_description": "Opening scene with Columbia Pictures logo. Text reads 'a SONY PICTURES ENTERTAINMENT company'." }, ... ] // Example
  LOG(logger, INFO, f"Raw scene detection complete.")
  RETURN raw_scene_data
END FUNCTION

FUNCTION GENERATE_DESCRIPTIONS(scenes, transcript, logger):
  LOG(logger, DEBUG, f"Generating descriptions from {len(scenes)} scenes and transcript.")
  // ... Mistral or other LLM logic using filtered scene data and transcript ...
  raw_descriptions = [ "The movie opens.", "A character walks.", ... ] // Example
  LOG(logger, INFO, f"Raw description generation complete.")
  RETURN raw_descriptions
END FUNCTION

FUNCTION SYNTHESIZE_SPEECH(descriptions, logger):
    LOG(logger, DEBUG, f"Synthesizing {len(descriptions)} descriptions.")
    // ... TTS logic ...
    synthesized_audio_path = "path/to/synthesized.wav" // Example
    LOG(logger, INFO, f"Synthesized speech saved to {synthesized_audio_path}")
    RETURN synthesized_audio_path
END FUNCTION

FUNCTION ASSEMBLE_AUDIO(original_audio, synthesized_audio, scenes, logger):
    LOG(logger, DEBUG, f"Assembling final audio from {original_audio} and {synthesized_audio}")
    // ... Logic to merge/mix audio based on scene timings ...
    final_output_path = "path/to/final_audio.mp3" // Example
    LOG(logger, INFO, f"Final audio assembled at {final_output_path}")
    RETURN final_output_path
END FUNCTION

FUNCTION GET_SCENE_DESCRIPTION(scene):
    // Logic to extract text description from scene object/dict
    RETURN scene.get("text_description", "")
END FUNCTION

FUNCTION UPDATE_SCENE_DESCRIPTION(scene, new_description):
    // Logic to update text description in scene object/dict
    scene["text_description"] = new_description
END FUNCTION

FUNCTION GET_INPUT_VIDEO_PATH():
    // Logic to get video path (e.g., from command line args)
    RETURN "input/movie.mp4" // Example
END FUNCTION

FUNCTION GET_PROCESSING_STEPS():
    // Logic to determine which steps to run (e.g., from config)
    RETURN ["extract", "transcribe", "detect", "generate", "synthesize", "assemble"] // Example
END FUNCTION

FUNCTION FORMAT_DURATION(seconds):
    // Logic to format seconds into HH:MM:SS or similar
    RETURN str(seconds) + " seconds" // Example
END FUNCTION


// --- Standard Python Entry Point ---
IF __name__ == "__main__":
  // Setup basic logging config if needed before initialize_logging,
  // though initialize_logging should handle most setup.
  main()
ENDIF