#!/bin/bash

# Setup script for the VAD-based audio description pipeline
set -euo pipefail

# Configuration
PYTHON_VERSION="3.8"  # Minimum required version
VENV_DIR=".venv"
LOG_FILE="setup.log"

# Error handling
handle_error() {
    echo "Error: $1" >&2
    echo "See $LOG_FILE for details"
    exit 1
}

# Logging function
log() {
    echo "$1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check Python version
check_python_version() {
    if ! command -v python3 &> /dev/null; then
        handle_error "Python 3 is not installed"
    fi
    
    local version
    version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if ! printf '%s\n%s\n' "$PYTHON_VERSION" "$version" | sort -C -V; then
        handle_error "Python $PYTHON_VERSION or higher is required (found $version)"
    fi
}

# Check for required system packages
check_system_dependencies() {
    local missing_deps=()
    
    # Check for ffmpeg
    if ! command -v ffmpeg &> /dev/null; then
        missing_deps+=("ffmpeg")
    fi
    
    # Check for pip
    if ! command -v pip3 &> /dev/null; then
        missing_deps+=("python3-pip")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "Missing system dependencies: ${missing_deps[*]}"
        echo "Please install them using your system's package manager."
        echo "For example, on Ubuntu:"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi
}

# Create and activate virtual environment
setup_virtualenv() {
    log "Setting up virtual environment in $VENV_DIR"
    
    # Create venv if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR" || handle_error "Failed to create virtual environment"
    fi
    
    # Activate venv
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate" || handle_error "Failed to activate virtual environment"
    
    # Upgrade pip
    python -m pip install --upgrade pip >> "$LOG_FILE" 2>&1 || handle_error "Failed to upgrade pip"
}

# Install Python dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    # Install torch first (GPU if available)
    if python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
        log "GPU detected, installing PyTorch with CUDA support"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 >> "$LOG_FILE" 2>&1 || \
            handle_error "Failed to install PyTorch"
    else
        log "No GPU detected, installing CPU-only PyTorch"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu >> "$LOG_FILE" 2>&1 || \
            handle_error "Failed to install PyTorch"
    fi
    
    # Install other dependencies
    pip install -r requirements.txt >> "$LOG_FILE" 2>&1 || handle_error "Failed to install dependencies"
}

# Create required directories
create_directories() {
    local dirs=("temp" "output" "temp/narrations" "test_data")
    
    log "Creating required directories..."
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
}

# Setup environment files
setup_env_files() {
    log "Setting up environment files..."
    
    # Copy .env.template to .env if it doesn't exist
    if [ ! -f .env ] && [ -f .env.template ]; then
        cp .env.template .env
        log "Created .env file from template. Please edit it with your settings."
    fi
}

# Main setup process
main() {
    echo "=== Setting up VAD-based Audio Description Pipeline ==="
    echo "Logging to $LOG_FILE"
    
    # Clear or create log file
    > "$LOG_FILE"
    
    # Run setup steps
    check_python_version
    check_system_dependencies
    setup_virtualenv
    install_dependencies
    create_directories
    setup_env_files
    
    echo "
Setup completed successfully!

Next steps:
1. Edit .env file with your settings
2. Activate the virtual environment:
   source $VENV_DIR/bin/activate
3. Run the test pipeline:
   ./scripts/test_pipeline.sh test_data/test_clip.mp4

For more information, see the README.md file.
"
}

# Run main setup
main