import pytest
from unittest.mock import patch, MagicMock
from main_orchestrator import MovieAudioProcessor
from exceptions import InvalidVideoFileError, AnalysisError, TTSUnavailableError

class TestMovieAudioProcessor:
    @pytest.fixture
    def processor(self):
        return MovieAudioProcessor()

    def test_init_sets_default_values(self, processor):
        assert processor.whisper_model == "base"
        assert processor.tts_voice == "en-US-Wavenet-D"

    @patch('main_orchestrator.AudioExtractor')
    @patch('main_orchestrator.AIAnalyzer')
    @patch('main_orchestrator.SceneSegmenter')
    @patch('main_orchestrator.NarrationGenerator')
    @patch('main_orchestrator.AudioEditor')
    def test_process_movie_calls_all_components(
        self, mock_editor, mock_narrator, mock_segmenter, 
        mock_analyzer, mock_extractor, processor
    ):
        # Setup mocks
        mock_extractor.return_value.extract_audio.return_value = "audio.wav"
        mock_analyzer.return_value.analyze.return_value = [{"text": "test", "start": 0, "end": 10}]
        mock_segmenter.return_value.segment_scenes.return_value = [MagicMock()]
        mock_narrator.return_value.generate.return_value = "narration.mp3"
        mock_editor.return_value.mix_audio.return_value = "final.mp3"

        # Test
        processor.process_movie("input.mp4")

        # Verify calls
        mock_extractor.return_value.extract_audio.assert_called_once_with("input.mp4")
        mock_analyzer.return_value.analyze.assert_called_once_with("audio.wav")
        mock_segmenter.return_value.segment_scenes.assert_called_once()
        mock_narrator.return_value.generate.assert_called_once()
        mock_editor.return_value.mix_audio.assert_called_once()

    @patch('main_orchestrator.AudioExtractor')
    def test_process_movie_handles_invalid_video(self, mock_extractor, processor):
        mock_extractor.return_value.extract_audio.side_effect = InvalidVideoFileError
        with pytest.raises(InvalidVideoFileError):
            processor.process_movie("invalid.mp4")

    @patch('main_orchestrator.AIAnalyzer')
    @patch('main_orchestrator.AudioExtractor')
    def test_process_movie_handles_analysis_error(self, mock_extractor, mock_analyzer, processor):
        mock_extractor.return_value.extract_audio.return_value = "audio.wav"
        mock_analyzer.return_value.analyze.side_effect = AnalysisError
        with pytest.raises(AnalysisError):
            processor.process_movie("input.mp4")

    @patch('main_orchestrator.NarrationGenerator')
    @patch('main_orchestrator.SceneSegmenter')
    @patch('main_orchestrator.AIAnalyzer')
    @patch('main_orchestrator.AudioExtractor')
    def test_process_movie_handles_tts_error(
        self, mock_extractor, mock_analyzer, 
        mock_segmenter, mock_narrator, processor
    ):
        mock_extractor.return_value.extract_audio.return_value = "audio.wav"
        mock_analyzer.return_value.analyze.return_value = [{"text": "test", "start": 0, "end": 10}]
        mock_segmenter.return_value.segment_scenes.return_value = [MagicMock()]
        mock_narrator.return_value.generate.side_effect = TTSUnavailableError
        with pytest.raises(TTSUnavailableError):
            processor.process_movie("input.mp4")

    def test_run_handles_cli_args(self, processor):
        with patch.object(processor, 'process_movie') as mock_process:
            processor.run(["script.py", "movie.mp4"])
            mock_process.assert_called_once_with("movie.mp4")