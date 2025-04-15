import os
import sys
import logging
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import yaml

from src.main import (
    calculate_total_duration,
    initialize_logging,
    report_progress,
    filter_logo_text,
    load_config
)

@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return {
        'logging': {
            'progress_format': '%(message)s',
            'component_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'logo_patterns': {
            'studios': ['Columbia Pictures', 'Warner Bros'],
            'channels': ['HBO', 'Netflix'],
            'generic': ['logo', 'watermark']
        },
        'paths': {
            'input_dir': 'input',
            'output_dir': 'output'
        },
        'components': {
            'ffmpeg': {'threads': 4},
            'whisper': {'model': 'base'},
            'blip': {'model': 'blip2-opt-2.7b'},
            'mistral': {'model': 'mistral-7b'},
            'tts': {'voice': 'en-US-Neural2-F'}
        }
    }

@pytest.fixture
def mock_config_file(tmp_path, test_config):
    """Create a temporary config file for testing."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    return config_path

def test_load_config(mock_config_file):
    """Test loading configuration from YAML file."""
    config = load_config(str(mock_config_file))
    assert config['logging']['progress_format'] == '%(message)s'
    assert 'ffmpeg' in config['components']

def test_calculate_total_duration():
    """Test duration calculation placeholder."""
    input_video = "test.mp4"
    steps = ["extract", "transcribe", "describe"]
    result = calculate_total_duration(input_video, steps)
    assert isinstance(result, timedelta)
    assert result == timedelta(minutes=15)  # 3 steps * 5 minutes

def test_initialize_logging(tmp_path):
    """Test logging initialization and file creation."""
    output_dir = tmp_path / "logs"
    progress_logger, main_logger, log_files = initialize_logging(str(output_dir))
    
    # Verify loggers were created
    assert isinstance(progress_logger, logging.Logger)
    assert isinstance(main_logger, logging.Logger)
    
    # Verify log files were created
    assert len(log_files) == 6  # mistral, blip, whisper, tts, ffmpeg, main
    for log_file in log_files.values():
        assert Path(log_file).exists()
    
    # Verify logging works
    test_message = "Test logging message"
    progress_logger.info(test_message)
    main_logger.info(test_message)

def test_report_progress(capsys):
    """Test progress reporting format."""
    logger = logging.getLogger("test_progress")
    logger.handlers = [logging.StreamHandler(sys.stdout)]  # Explicitly use stdout
    logger.setLevel(logging.INFO)
    
    # Test without percentage
    report_progress(logger, "Test message")
    captured = capsys.readouterr()
    assert "Test message" in captured.out
    
    # Test with percentage
    report_progress(logger, "Test progress", 42.5)
    captured = capsys.readouterr()
    assert "[42.5%] Test progress" in captured.out

def test_filter_logo_text(test_config):
    """Test logo text filtering."""
    # Patch the CONFIG to use our test config
    with patch('src.main.CONFIG', test_config):
        # Test studio patterns
        text = "Columbia Pictures presents a Warner Bros production"
        filtered = filter_logo_text(text)
        assert "Columbia Pictures" not in filtered
        assert "Warner Bros" not in filtered
        
        # Test channel patterns
        text = "Now streaming on HBO and Netflix"
        filtered = filter_logo_text(text)
        assert "HBO" not in filtered
        assert "Netflix" not in filtered
        
        # Test generic patterns
        text = "This contains a logo and watermark"
        filtered = filter_logo_text(text)
        assert "logo" not in filtered
        assert "watermark" not in filtered
        
        # Test empty string
        assert filter_logo_text("") == ""