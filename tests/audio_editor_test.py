import pytest
from unittest.mock import patch, MagicMock
from audio_editor import AudioEditor
from data_structures import Scene
from exceptions import AudioEditError

class TestAudioEditor:
    @pytest.fixture
    def editor(self):
        return AudioEditor()

    def test_init_sets_default_output_dir(self, editor):
        assert editor.output_dir == "edited_audio"

    @patch('audio_editor.ffmpeg')
    def test_trim_audio_calls_ffmpeg_correctly(self, mock_ffmpeg, editor):
        test_file = "input.mp3"
        scene = Scene(start=10, end=20, content="test")
        editor.trim_audio(test_file, scene)
        mock_ffmpeg.input.assert_called_once_with(test_file)
        mock_ffmpeg.filter.assert_called_once_with('atrim', start=10, end=20)
        mock_ffmpeg.output.assert_called_once()

    @patch('audio_editor.ffmpeg')
    def test_concatenate_audio_joins_files(self, mock_ffmpeg, editor):
        files = ["1.mp3", "2.mp3"]
        output = "output.mp3"
        editor.concatenate_audio(files, output)
        assert mock_ffmpeg.input.call_count == 2
        mock_ffmpeg.concat.assert_called_once()
        mock_ffmpeg.output.assert_called_once_with(output)

    @patch('audio_editor.ffmpeg')
    def test_mix_audio_combines_tracks(self, mock_ffmpeg, editor):
        tracks = ["voice.mp3", "bg.mp3"]
        output = "mixed.mp3"
        editor.mix_audio(tracks, output)
        assert mock_ffmpeg.input.call_count == 2
        mock_ffmpeg.filter.assert_called_once_with('amix', inputs=2, duration='longest')
        mock_ffmpeg.output.assert_called_once_with(output)

    @patch('audio_editor.ffmpeg')
    def test_apply_effects_adds_filters(self, mock_ffmpeg, editor):
        test_file = "input.mp3"
        output = "output.mp3"
        editor.apply_effects(test_file, output, fade_in=1, fade_out=1, volume=0.8)
        assert mock_ffmpeg.filter.call_count == 3
        mock_ffmpeg.output.assert_called_once_with(output)

    @patch('audio_editor.ffmpeg')
    def test_operations_raise_on_ffmpeg_error(self, mock_ffmpeg, editor):
        mock_ffmpeg.side_effect = Exception("FFmpeg error")
        with pytest.raises(AudioEditError):
            editor.trim_audio("input.mp3", Scene(start=0, end=10))