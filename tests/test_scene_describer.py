import os
import pytest
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock # Import patch and MagicMock
from scene_describer import SceneDescriber
from moviepy.editor import VideoFileClip
import tempfile
@pytest.fixture
def test_image(tmp_path):
    """Create a simple test image."""
    # Create a 100x100 white image with a black rectangle
    img = Image.new('RGB', (100, 100), color='white')
    # Draw a black rectangle
    for x in range(20, 80):
        for y in range(40, 60):
            img.putpixel((x, y), (0, 0, 0))
    
    path = tmp_path / "test_image.jpg"
    img.save(str(path))
    return str(path)

@pytest.fixture
def test_video(tmp_path):
    """Create a simple test video."""
    # Create a series of frames
    frames = []
    for i in range(40):  # 40 frames (4 seconds at 10fps)
        img = Image.new('RGB', (100, 100), color='white')
        # Draw a moving rectangle
        for x in range(20 + i*2, 80 + i*2):
            for y in range(40, 60):
                img.putpixel((x % 100, y), (0, 0, 0))
        frames.append(np.array(img))
    
    # Create video file
    video_path = str(tmp_path / "test_video.mp4")
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=10)
    clip.write_videofile(video_path)
    return video_path

def test_scene_describer_initialization():
    """Test SceneDescriber initialization."""
    describer = SceneDescriber()
    assert describer.model_name == "Salesforce/blip-image-captioning-large"
    # With lazy loading, these should be None initially
    assert describer.processor is None
    assert describer.model is None

@patch('scene_describer.BlipProcessor.from_pretrained')
@patch('scene_describer.BlipForConditionalGeneration.from_pretrained')
def test_load_model(mock_model_gen_load, mock_processor_load):
    """Test model loading logic (mocked)."""
    # Configure mocks
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_load.return_value = mock_processor
    mock_model_gen_load.return_value = mock_model

    describer = SceneDescriber()
    # Explicitly call load_model for the test
    describer.load_model()

    # Assert mocks were called by load_model
    mock_processor_load.assert_called_once_with(describer.model_name)
    mock_model_gen_load.assert_called_once_with(describer.model_name)
    assert describer.processor is mock_processor
    assert describer.model is mock_model
    mock_model.to.assert_called_once_with(describer.device)

def test_extract_representative_frames(test_video):
    """Test frame extraction from video."""
    describer = SceneDescriber()
    frames = describer.extract_representative_frames(
        test_video,
        start_time=0.0,
        end_time=1.0,
        num_frames=3
    )
    
    assert len(frames) == 3
    for frame in frames:
        assert isinstance(frame, Image.Image)
        assert frame.size == (100, 100)

@patch('scene_describer.BlipProcessor.from_pretrained')
@patch('scene_describer.BlipForConditionalGeneration.from_pretrained')
def test_generate_description_for_frame(mock_model_gen_load, mock_processor_load, test_image):
    """Test description generation for a single frame (mocked model)."""
    # --- Mock Setup ---
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_processor_load.return_value = mock_processor
    mock_model_gen_load.return_value = mock_model
    
    # Mock the processor call and model generation
    mock_inputs = MagicMock()
    mock_processor.return_value = mock_inputs
    mock_outputs = MagicMock()
    mock_model.generate.return_value = mock_outputs
    mock_processor.decode.return_value = " a test description " # Add spaces for strip() test

    # --- Execution ---
    describer = SceneDescriber() # Calls load_model -> mocks
    
    description = describer.generate_description_for_frame(Image.open(test_image))
    
    # --- Assertions ---
    assert description == "a test description" # Check stripped result
    # Check processor call (image object might differ slightly, check type)
    # Access call_args directly on the mock object
    assert mock_processor.call_args[1]['images'].size == Image.open(test_image).size
    assert mock_processor.call_args[1]['return_tensors'] == "pt"
    mock_inputs.to.assert_called_once_with(describer.device)
    # Check model generation call
    mock_model.generate.assert_called_once_with(**mock_inputs, max_length=50, num_beams=5, temperature=1.0, length_penalty=1.0)
    # Check decoding call
    mock_processor.decode.assert_called_once_with(mock_outputs[0], skip_special_tokens=True)

# Mock model loading here too, as generate_descriptions calls generate_description_for_frame
@patch('scene_describer.BlipProcessor.from_pretrained')
@patch('scene_describer.BlipForConditionalGeneration.from_pretrained')
@patch('scene_describer.SceneDescriber.extract_representative_frames')
# We also need to mock generate_description_for_frame as it uses the real model otherwise
@patch('scene_describer.SceneDescriber.generate_description_for_frame')
def test_generate_descriptions(mock_gen_desc_frame, mock_extract_frames, mock_model_gen_load, mock_processor_load, test_video):
    """Test description generation for video segments (mocked)."""
    # --- Mock Setup ---
    # Mock model loading
    mock_processor_load.return_value = MagicMock()
    mock_model_gen_load.return_value = MagicMock()
    
    # Mock frame extraction to return dummy PIL images
    dummy_frame = Image.new('RGB', (60, 30), color = 'blue')
    mock_extract_frames.return_value = [dummy_frame] * 3 # Return 3 dummy frames per segment

    # Mock description generation for a frame
    mock_gen_desc_frame.side_effect = ["desc1", "desc2", "desc3", "desc4", "desc5", "desc6"] # Unique desc per frame call

    # --- Execution ---
    describer = SceneDescriber() # Instantiation triggers mocked loading
    segments = [(0.0, 1.0), (2.0, 3.0)]
    descriptions = describer.generate_descriptions(test_video, segments)

    # --- Assertions ---
    assert len(descriptions) == len(segments)
    # Check extract_representative_frames calls
    assert mock_extract_frames.call_count == len(segments)
    mock_extract_frames.assert_any_call(test_video, 0.0, 1.0)
    mock_extract_frames.assert_any_call(test_video, 2.0, 3.0)
    # Check generate_description_for_frame calls (3 frames per segment)
    assert mock_gen_desc_frame.call_count == len(segments) * 3
    # Check final descriptions (should be one of the descriptions returned for that segment's frames)
    assert descriptions[0]['description'] in ["desc1", "desc2", "desc3"]
    assert descriptions[1]['description'] in ["desc4", "desc5", "desc6"]
    assert descriptions[0]['start_time'] == 0.0
    assert descriptions[1]['end_time'] == 3.0

# Mock model loading for this test too
@patch('scene_describer.BlipProcessor.from_pretrained')
@patch('scene_describer.BlipForConditionalGeneration.from_pretrained')
def test_error_handling_invalid_video(mock_model_gen_load, mock_processor_load):
    """Test error handling for invalid video file."""
    # Mock model loading
    mock_processor_load.return_value = MagicMock()
    mock_model_gen_load.return_value = MagicMock()

    describer = SceneDescriber() # Instantiation triggers mocked loading
    with pytest.raises(FileNotFoundError):
        # generate_descriptions checks file existence first
        describer.generate_descriptions(
            "nonexistent.mp4",
            [(0.0, 1.0)]
        )

def test_error_handling_invalid_segment(test_video):
    """Test error handling for invalid segment times."""
    describer = SceneDescriber()
    with pytest.raises(ValueError):
        describer.extract_representative_frames(
            test_video,
            start_time=2.0,
            end_time=1.0  # End before start
        )

def test_error_handling_model_load():
    """Test error handling for model loading issues."""
    # Instantiation should succeed with lazy loading
    describer = SceneDescriber(model_name="nonexistent/model")
    # Expect RuntimeError when load_model is explicitly called
    with pytest.raises(RuntimeError, match="Failed to load BLIP model"):
        describer.load_model()