#!/bin/bash

# Script to install test dependencies without TTS
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Installing core dependencies for testing (excluding TTS)...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment (handle both Windows and Unix)
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo -e "${RED}Could not find virtual environment activation script${NC}"
    exit 1
fi

# Function to safely install packages
install_package() {
    local package=$1
    echo -e "${YELLOW}Installing $package...${NC}"
    pip install $package || echo -e "${RED}Failed to install $package${NC}"
}

# Install packages one by one
echo "Installing basic dependencies..."
install_package "torch"
install_package "torchvision"
install_package "torchaudio"
install_package "numpy"
install_package "Pillow"
install_package "opencv-python"
install_package "pydub"
install_package "moviepy"

echo "Installing testing dependencies..."
install_package "pytest"
install_package "pytest-cov"

echo "Installing optional dependencies..."
install_package "transformers"
install_package "accelerate"

echo -e "${GREEN}Basic dependencies installed${NC}"
echo -e "${YELLOW}Note: TTS dependencies were skipped${NC}"