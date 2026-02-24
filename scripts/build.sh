#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="drl-arm"
PYTHON_VERSION="3.9"

echo "======================================"
echo "DRL MuJoCo - Environment Setup"
echo "======================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: 'conda' command not found."
    echo "Please install Miniforge/Miniconda first."
    echo "On macOS: brew install --cask miniforge"
    exit 1
fi

# Check if conda is initialized
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "ERROR: Conda not initialized."
    echo "Please run 'conda init zsh' (or your shell) and restart your terminal."
    exit 1
fi

echo "Checking Python architecture..."
if [[ "$(uname -s)" == "Darwin" ]]; then
    # On macOS, check if Python is arm64
    PYTHON_ARCH="$(python3 -c "import platform; print(platform.machine())" 2>/dev/null || echo "unknown")"
    if [[ "$PYTHON_ARCH" == "x86_64" ]]; then
        echo "WARNING: Current Python is x86_64 on Apple Silicon."
        echo "MuJoCo requires arm64 Python. Please ensure you have an arm64 conda installation."
    fi
fi

echo "Creating/activating conda environment: $ENV_NAME"
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating new conda environment with Python $PYTHON_VERSION..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

# Activate environment and install dependencies
echo "Activating environment and installing dependencies..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing Python packages..."
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "To use this environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run training:"
echo "  bash start.sh"
echo ""
