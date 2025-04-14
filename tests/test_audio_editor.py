from unittest.mock import patch, MagicMock, call
import sys
import pytest
import os
import ffmpeg # Import ffmpeg
# Setup module mocks
mock_audio = MagicMock()
mock_audio_segment = MagicMock()
mock_audio_segment.from_file = MagicMock(return_value=mock_audio)
mock_audio_segment.from_wav = MagicMock(return_value=mock_audio)
mock_audio_segment.empty = MagicMock(return_value=mock_audio)

mock_pydub = MagicMock()
mock_pydub.AudioSegment = mock_audio_segment

# Mock modules at import time
sys.modules['audioop'] = MagicMock()
sys.modules['pyaudioop'] = MagicMock()
sys.modules['pydub'] = mock_pydub
sys.modules['pydub.audio_segment'] = mock_audio_segment
sys.modules['pydub.utils'] = MagicMock()

from audio_editor import overlay_narration_on_scene, merge_scenes_to_final_mp3
from exceptions import AudioOverlayError, AudioMergeError
from data_structures import Scene
import os

@pytest.fixture
def mock_scene():
    return Scene(
        scene_id=1,
        start_time=0.0,
        end_time=5.0,
        description="Mock scene description", # Added missing description
        narration_audio_path="narration.mp3"
    )

class TestAudioOverlay:
    @patch('audio_editor.ffmpeg') # Patch ffmpeg used in audio_editor
    def test_overlay_success(self, mock_ffmpeg):
        # Mock the ffmpeg chain
        mock_input_orig = MagicMock()
        mock_input_narr = MagicMock()
        mock_mixed = MagicMock()
        mock_stream = MagicMock()

        # Configure return values for the chained calls
        mock_ffmpeg.input.side_effect = [mock_input_orig, mock_input_narr]
        mock_input_orig.audio = MagicMock() # Need audio attribute for filter
        mock_input_narr.audio = MagicMock() # Need audio attribute for filter
        mock_ffmpeg.filter.return_value = mock_mixed
        mock_ffmpeg.output.return_value = mock_stream
        mock_ffmpeg.run.return_value = (b'stdout', b'stderr') # Simulate successful run

        mock_scene = Scene(description="Test scene",
            scene_id=1,
            start_time=0.0,
            end_time=5.0,
            narration_audio_path="narration.mp3"
        )

        overlay_narration_on_scene(mock_scene, "original.mp3", "output.mp3")

        # Assert ffmpeg functions were called correctly
        assert mock_ffmpeg.input.call_count == 2
        mock_ffmpeg.input.assert_any_call("original.mp3")
        mock_ffmpeg.input.assert_any_call("narration.mp3")

        mock_ffmpeg.filter.assert_called_once_with(
            [mock_input_orig.audio, mock_input_narr.audio],
            'amix',
            inputs=2,
            duration='longest'
        )
        mock_ffmpeg.output.assert_called_once_with(mock_mixed, "output.mp3")
        mock_ffmpeg.run.assert_called_once_with(mock_stream, capture_stdout=True, capture_stderr=True)

    @patch('audio_editor.ffmpeg') # Patch ffmpeg used in audio_editor
    def test_overlay_file_not_found(self, mock_ffmpeg, mock_scene):
        # Simulate ffmpeg failing to find the input file by raising its specific error
        # We need to ensure the type is correct for the except block in the source code
        mock_ffmpeg.Error = ffmpeg.Error # Ensure the mock uses the real Error type for isinstance checks if any
        mock_ffmpeg.input.side_effect = ffmpeg.Error('ffmpeg', b'stdout', b'stderr: nonexistent.mp3: No such file or directory')

        with pytest.raises(AudioOverlayError, match="Failed to overlay narration"):
            overlay_narration_on_scene(mock_scene, "nonexistent.mp3", "output.mp3")

        # Assert ffmpeg.input was called (which then raised the error)
        mock_ffmpeg.input.assert_called_once_with("nonexistent.mp3")
        # Ensure other ffmpeg steps weren't reached
        mock_ffmpeg.filter.assert_not_called()
        mock_ffmpeg.output.assert_not_called()
        mock_ffmpeg.run.assert_not_called()
class TestAudioMerge:
    @patch('audio_editor.os.unlink')
    @patch('audio_editor.tempfile.NamedTemporaryFile')
    @patch('audio_editor.os.path.abspath')
    @patch('audio_editor.os.path.exists')
    @patch('audio_editor.ffmpeg')
    def test_merge_success(self, mock_ffmpeg, mock_exists, mock_abspath, mock_tempfile, mock_unlink):
        # --- Mock Setup ---
        mock_exists.return_value = True # Assume files exist
        # Make abspath return the input path for simplicity in assertions
        mock_abspath.side_effect = lambda x: x

        # Mock the temp file context manager
        mock_file_handle = MagicMock()
        mock_file_handle.write = MagicMock()
        mock_file_handle.name = "mock_temp_list.txt"
        mock_tempfile.return_value.__enter__.return_value = mock_file_handle

        # Mock ffmpeg calls
        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_ffmpeg.output.return_value = mock_stream
        mock_ffmpeg.run.return_value = (b'stdout', b'stderr') # Simulate success

        # --- Test Data ---
        scenes = [
            Scene(scene_id=1, start_time=0.0, end_time=5.0, description="Scene 1", narration_audio_path="scene1.mp3"),
            Scene(scene_id=2, start_time=5.0, end_time=10.0, description="Scene 2", narration_audio_path="scene2.mp3")
        ]
        output_path = "final.mp3"

        # --- Execute ---
        merge_scenes_to_final_mp3(scenes, output_path)

        # --- Assertions ---
        # Check temp file creation and writing
        mock_tempfile.assert_called_once_with('w', suffix='.txt', delete=False)
        assert mock_file_handle.write.call_count == 2
        mock_file_handle.write.assert_any_call("file 'scene1.mp3'\n")
        mock_file_handle.write.assert_any_call("file 'scene2.mp3'\n")

        # Check ffmpeg calls
        mock_ffmpeg.input.assert_called_once_with("mock_temp_list.txt", format='concat', safe=0)
        mock_ffmpeg.output.assert_called_once_with(mock_stream, output_path, acodec='libmp3lame')
        mock_ffmpeg.run.assert_called_once_with(mock_stream, capture_stdout=True, capture_stderr=True)

        # Check temp file cleanup
        mock_unlink.assert_called_once_with("mock_temp_list.txt")

    @patch('audio_editor.os.unlink')
    @patch('audio_editor.tempfile.NamedTemporaryFile')
    @patch('audio_editor.os.path.abspath')
    @patch('audio_editor.os.path.exists')
    @patch('audio_editor.ffmpeg')
    def test_merge_missing_files(self, mock_ffmpeg, mock_exists, mock_abspath, mock_tempfile, mock_unlink):
        # --- Mock Setup ---
        mock_exists.side_effect = [False, True] # First file missing, second exists
        mock_abspath.side_effect = lambda x: x

        mock_file_handle = MagicMock()
        mock_file_handle.write = MagicMock()
        mock_file_handle.name = "mock_temp_list_missing.txt"
        mock_tempfile.return_value.__enter__.return_value = mock_file_handle

        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_ffmpeg.output.return_value = mock_stream
        mock_ffmpeg.run.return_value = (b'stdout', b'stderr')

        # --- Test Data ---
        scenes = [
            Scene(scene_id=1, start_time=0.0, end_time=5.0, description="Missing scene", narration_audio_path="missing.mp3"),
            Scene(scene_id=2, start_time=5.0, end_time=10.0, description="Existing scene", narration_audio_path="exists.mp3")
        ]
        output_path = "final_missing.mp3"

        # --- Execute ---
        merge_scenes_to_final_mp3(scenes, output_path)

        # --- Assertions ---
        # Check temp file writing (only existing file should be written)
        mock_tempfile.assert_called_once_with('w', suffix='.txt', delete=False)
        mock_file_handle.write.assert_called_once_with("file 'exists.mp3'\n")

        # Check ffmpeg calls
        mock_ffmpeg.input.assert_called_once_with("mock_temp_list_missing.txt", format='concat', safe=0)
        mock_ffmpeg.output.assert_called_once_with(mock_stream, output_path, acodec='libmp3lame')
        mock_ffmpeg.run.assert_called_once_with(mock_stream, capture_stdout=True, capture_stderr=True)

        # Check temp file cleanup
        mock_unlink.assert_called_once_with("mock_temp_list_missing.txt")
    @patch('audio_editor.os.unlink')
    @patch('audio_editor.tempfile.NamedTemporaryFile')
    @patch('audio_editor.os.path.abspath')
    @patch('audio_editor.os.path.exists')
    @patch('audio_editor.ffmpeg')
    def test_merge_all_files_missing(self, mock_ffmpeg, mock_exists, mock_abspath, mock_tempfile, mock_unlink):
        # --- Mock Setup ---
        mock_exists.return_value = False # All files missing
        mock_abspath.side_effect = lambda x: x

        mock_file_handle = MagicMock()
        mock_file_handle.write = MagicMock()
        mock_file_handle.name = "mock_temp_list_all_missing.txt"
        mock_tempfile.return_value.__enter__.return_value = mock_file_handle

        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_ffmpeg.output.return_value = mock_stream
        # Simulate ffmpeg error when the input list is empty or invalid
        mock_ffmpeg.Error = ffmpeg.Error # Ensure the mock uses the real Error type
        mock_ffmpeg.run.side_effect = ffmpeg.Error('ffmpeg', b'stdout', b'stderr: Invalid data found when processing input')

        # --- Test Data ---
        scenes = [
            Scene(scene_id=1, start_time=0.0, end_time=5.0, description="Missing 1", narration_audio_path="missing1.mp3"),
            Scene(scene_id=2, start_time=5.0, end_time=10.0, description="Missing 2", narration_audio_path="missing2.mp3")
        ]
        output_path = "final_all_missing.mp3"

        # --- Execute & Assert ---
        # Expect AudioMergeError because ffmpeg fails with an empty/invalid list
        with pytest.raises(AudioMergeError, match="Failed to merge scenes"):
             merge_scenes_to_final_mp3(scenes, output_path)

        # --- Assertions ---
        # Check temp file writing (should not write anything)
        mock_tempfile.assert_called_once_with('w', suffix='.txt', delete=False)
        mock_file_handle.write.assert_not_called()

        # Check ffmpeg calls (input and run should be called, output might be called before run)
        mock_ffmpeg.input.assert_called_once_with("mock_temp_list_all_missing.txt", format='concat', safe=0)
        mock_ffmpeg.output.assert_called_once_with(mock_stream, output_path, acodec='libmp3lame')
        mock_ffmpeg.run.assert_called_once_with(mock_stream, capture_stdout=True, capture_stderr=True)

        # Check temp file cleanup
        mock_unlink.assert_called_once_with("mock_temp_list_all_missing.txt")