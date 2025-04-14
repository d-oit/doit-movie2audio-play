#!/bin/bash

# Test script for scene segmentation
set -euo pipefail

# Check requirements
command -v python >/dev/null 2>&1 || { echo >&2 "Python is required but not installed. Aborting."; exit 1; }

# Setup test environment
TEST_DIR="test_output/scene_segmentation"
mkdir -p "$TEST_DIR"
TEST_ANALYSIS="test_data/sample_analysis.json"
TEST_OUTPUT="$TEST_DIR/segmented_scenes.json"

echo "Running scene segmentation tests..."

# Run unit tests
if ! python -m pytest tests/scene_segmenter_test.py -v; then
    echo "Unit tests failed"
    exit 1
fi

# Integration test
echo "Running segmentation integration test..."
if ! python -c "import json; \
    from scene_segmenter import SceneSegmenter; \
    with open('$TEST_ANALYSIS') as f: \
        analysis = json.load(f); \
    segments = SceneSegmenter().segment_scenes(analysis); \
    with open('$TEST_OUTPUT', 'w') as f: \
        json.dump([s.__dict__ for s in segments], f)"; then
    echo "Segmentation failed"
    exit 1
fi

# Validate output
if [ ! -f "$TEST_OUTPUT" ]; then
    echo "Output file not created"
    exit 1
fi

SCENE_COUNT=$(python -c "import json; \
    with open('$TEST_OUTPUT') as f: \
        print(len(json.load(f)))")
if [ "$SCENE_COUNT" -eq 0 ]; then
    echo "No scenes detected in output"
    exit 1
fi

echo "Scene segmentation tests passed successfully"
exit 0