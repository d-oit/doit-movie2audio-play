#!/bin/bash

# Test script for Whisper audio analysis
set -euo pipefail

# Check requirements
command -v python >/dev/null 2>&1 || { echo >&2 "Python is required but not installed. Aborting."; exit 1; }

# Setup test environment
TEST_DIR="test_output/whisper_analysis"
mkdir -p "$TEST_DIR"
TEST_AUDIO="test_data/sample.wav"

echo "Running Whisper analysis tests..."

# Run unit tests
if ! python -m pytest tests/ai_analyzer_test.py -v; then
    echo "Unit tests failed"
    exit 1
fi

# Local model test
echo "Testing local Whisper analysis..."
LOCAL_OUTPUT="$TEST_DIR/local_output.json"
if ! python -c "from ai_analyzer import AIAnalyzer; \
    result = AIAnalyzer().analyze_local('$TEST_AUDIO'); \
    import json; \
    with open('$LOCAL_OUTPUT', 'w') as f: json.dump(result, f)"; then
    echo "Local analysis failed"
    exit 1
fi

# API test (if key provided)
if [ -z "${WHISPER_API_KEY:-}" ]; then
    echo "Skipping API test (no WHISPER_API_KEY set)"
else
    echo "Testing Whisper API analysis..."
    API_OUTPUT="$TEST_DIR/api_output.json"
    if ! python -c "from ai_analyzer import AIAnalyzer; \
        result = AIAnalyzer().analyze_api('$TEST_AUDIO', '$WHISPER_API_KEY'); \
        import json; \
        with open('$API_OUTPUT', 'w') as f: json.dump(result, f)"; then
        echo "API analysis failed"
        exit 1
    fi
fi

echo "Whisper analysis tests completed successfully"
exit 0