#!/bin/bash

# Script to check and validate the development environment
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check Python version
check_python_version() {
    local version
    if command -v python3.11 &> /dev/null; then
        version=$(python3.11 -V 2>&1)
        echo -e "${GREEN}✓ Found $version${NC}"
        return 0
    elif command -v python3.10 &> /dev/null; then
        version=$(python3.10 -V 2>&1)
        echo -e "${YELLOW}⚠ Found $version (Python 3.11 recommended)${NC}"
        return 0
    elif command -v python3.9 &> /dev/null; then
        version=$(python3.9 -V 2>&1)
        echo -e "${YELLOW}⚠ Found $version (Python 3.11 recommended)${NC}"
        return 0
    else
        echo -e "${RED}✗ No compatible Python version found (3.9-3.11 required)${NC}"
        return 1
    fi
}

# Function to check FFmpeg
check_ffmpeg() {
    if command -v ffmpeg &> /dev/null; then
        echo -e "${GREEN}✓ FFmpeg installed${NC}"
        return 0
    else
        echo -e "${RED}✗ FFmpeg not found${NC}"
        return 1
    fi
}

# Function to check CUDA
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
        echo "  $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
        return 0
    else
        echo -e "${YELLOW}⚠ No NVIDIA GPU detected (CPU only mode will be slower)${NC}"
        return 0
    fi
}

# Function to check virtual environment
check_venv() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo -e "${GREEN}✓ Virtual environment active: $VIRTUAL_ENV${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ No virtual environment active${NC}"
        return 1
    fi
}

# Function to check Hugging Face token
check_hf_token() {
    if [[ -f .env ]] && grep -q "HUGGING_FACE_TOKEN" .env; then
        echo -e "${GREEN}✓ Hugging Face token found in .env${NC}"
        return 0
    else
        echo -e "${RED}✗ No Hugging Face token found in .env${NC}"
        return 1
    fi
}

# Main function to run all checks
main() {
    echo "=== Environment Check ==="
    echo "Checking system requirements..."
    echo
    
    local errors=0
    
    echo "1. Python Version:"
    check_python_version || ((errors++))
    echo
    
    echo "2. FFmpeg:"
    check_ffmpeg || ((errors++))
    echo
    
    echo "3. CUDA Support:"
    check_cuda
    echo
    
    echo "4. Virtual Environment:"
    check_venv || ((errors++))
    echo
    
    echo "5. Hugging Face Token:"
    check_hf_token || ((errors++))
    echo
    
    if [[ $errors -eq 0 ]]; then
        echo -e "${GREEN}All checks passed!${NC}"
        echo "You can proceed with:"
        echo "  pip install -r requirements.txt"
    else
        echo -e "${RED}Found $errors issue(s) that need to be addressed.${NC}"
        echo "Please check PYTHON_VERSION.md for setup instructions."
        exit 1
    fi
}

# Run main function
main