#!/bin/bash

# Cleanup script to remove obsolete files from the old transcription-based pipeline
set -euo pipefail

# Files to remove
OBSOLETE_FILES=(
    "transcription_service.py"
    "ai_analyzer.py"
    "scripts/test_transcription.sh"
    "scripts/run_transcription_test.py"
    "scripts/process_movie.sh"
)

# Counter for removed files
removed_count=0

# Function to safely remove a file
remove_file() {
    local file="$1"
    if [ -f "$file" ]; then
        echo "Removing: $file"
        rm "$file"
        ((removed_count++))
    else
        echo "Not found: $file"
    fi
}

echo "=== Cleaning up obsolete files ==="
echo "This script will remove files from the old transcription-based pipeline."
echo "The following files will be removed if they exist:"
printf '%s\n' "${OBSOLETE_FILES[@]}"

# Prompt for confirmation
read -p "Do you want to proceed? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 1
fi

# Remove obsolete files
for file in "${OBSOLETE_FILES[@]}"; do
    remove_file "$file"
done

# Update exceptions.py if it exists
if [ -f "exceptions.py" ]; then
    echo "Updating exceptions.py..."
    # Create temporary file
    temp_file=$(mktemp)
    
    # Filter out obsolete exceptions
    grep -v -E "TranscriptionError|ApiError|LocalModelError|ModelNotFoundError|NetworkError" exceptions.py > "$temp_file" || true
    
    # Add header if file is empty
    if [ ! -s "$temp_file" ]; then
        echo "# Custom exceptions for the VAD-based audio description pipeline" > "$temp_file"
    fi
    
    # Replace original file
    mv "$temp_file" exceptions.py
    echo "Updated exceptions.py"
fi

echo
echo "=== Cleanup Complete ==="
echo "Removed $removed_count files"
echo
echo "Next steps:"
echo "1. Review exceptions.py for any remaining unused exceptions"
echo "2. Update import statements in remaining files"
echo "3. Run tests to ensure nothing was broken"
echo "4. Commit the changes"