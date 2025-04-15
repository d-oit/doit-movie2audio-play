# Video-to-Audio Description Generator

This project processes video files to generate audio descriptions, focusing on filtering out logo text and maintaining clean separation between progress reporting and component logging.

## Features

- Duration estimation for processing steps
- High-level progress updates in terminal
- Detailed component logs in separate files
- Logo and watermark text filtering
- Modular architecture with clear component separation

## Requirements

- Python 3.8+
- FFmpeg
- Additional dependencies (see `requirements.txt`)

## Project Structure

```
.
├── src/                   # Core application code
│   ├── __init__.py
│   ├── main.py            # Main entry point and orchestration
│   └── components.py      # Processing component implementations
├── tests/                 # Unit tests
│   ├── __init__.py
│   ├── test_main.py
│   └── test_components.py
├── scripts/               # Utility scripts
│   └── process_movie.sh   # Main processing script
├── output/                # Default output directory for logs and audio
├── input/                 # Default input directory for videos
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd video-to-audio-description
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure FFmpeg is installed and available in your PATH.

## Usage

1. Place your input video in a known location.

2. Run the processor:
```bash
python src/main.py
```

The script will:
- Show estimated processing time
- Display progress updates in the terminal
- Write detailed component logs to the `output/` directory
- Generate the final audio description track

## Component Logs

Each processing component writes detailed logs to separate files in the `output/` directory:

- `ffmpeg.log`: Video/audio processing operations
- `whisper.log`: Speech transcription details
- `blip.log`: Scene detection information
- `mistral.log`: Description generation process
- `tts.log`: Speech synthesis operations
- `main.log`: Overall processing flow

## Implementation Notes

- Progress updates are shown in the terminal at the INFO level
- Component-specific debug information goes to individual log files
- Logo text filtering is applied at both scene detection and description generation stages
- The system uses a modular architecture where each component can be independently modified or replaced

## Future Improvements

- Add CLI arguments for input/output paths and configuration
- Implement actual duration estimation based on video properties
- Add support for different TTS engines
- Enhance logo text detection patterns
- Add parallel processing support for appropriate steps
