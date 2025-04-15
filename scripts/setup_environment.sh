#!/bin/bash

# Setup script for the VAD-based audio description pipeline
set -euo pipefail

# Configuration
# PYTHON_VERSION="3.8" # Version check removed, rely on pip install
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

# Check Python command exists
check_python_command() {
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        handle_error "Python 3 is not installed or not in PATH"
    fi
    # Use python3 if available, otherwise python
    PYTHON_CMD=$(command -v python3 || command -v python)
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
        $PYTHON_CMD -m venv "$VENV_DIR" || handle_error "Failed to create virtual environment"
    fi
    
    # Activate venv
    # Activate venv (handle Unix and Windows)
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck source=/dev/null
        source "$VENV_DIR/bin/activate" || handle_error "Failed to activate Unix virtual environment"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        # shellcheck source=/dev/null
        source "$VENV_DIR/Scripts/activate" || handle_error "Failed to activate Windows virtual environment"
    else
         handle_error "Could not find activation script in $VENV_DIR"
    fi
    
    # Upgrade pip
    python -m pip install --upgrade pip >> "$LOG_FILE" 2>&1 || handle_error "Failed to upgrade pip"
}

# Install Python dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    # Install torch first (GPU if available)
    if $PYTHON_CMD -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
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
    check_python_command # Check python exists
    check_system_dependencies
    setup_virtualenv
    install_dependencies
    create_directories
    setup_env_files
    
    echo "
Setup completed successfully!

Next steps:
1. Edit `.env` file if needed (e.g., for API keys if using API-based models).
2. Activate the virtual environment:
   - Unix/macOS: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
3. Run the tests:
   `pytest tests/ -v`
4. Run the main script (example):
   `python -m src.main input/your_video.mp4`

For more information, see the README.md file.
"
}

# Run main setup
main