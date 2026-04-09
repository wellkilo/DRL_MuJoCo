#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing Python packages..."
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo ""
echo "======================================"
echo "Additional Components (Optional)"
echo "======================================"
echo ""

# Ask about Next.js frontend
read -p "Build Next.js frontend? [y/N]: " build_next
if [[ "$build_next" =~ ^[Yy]$ ]]; then
    if command -v npm &> /dev/null; then
        echo "Building Next.js frontend..."
        cd "$REPO_ROOT/web"
        npm install
        npm run build
        cd "$REPO_ROOT"
        echo "Next.js frontend built successfully!"
    else
        echo "WARNING: npm not found. Skipping Next.js frontend build."
        echo "Install Node.js from https://nodejs.org/ to build frontend."
    fi
fi

# Ask about Rust Buffer
read -p "Build Rust Buffer? [y/N]: " build_rust
if [[ "$build_rust" =~ ^[Yy]$ ]]; then
    if command -v cargo &> /dev/null; then
        echo "Building Rust Buffer..."
        cd "$REPO_ROOT/rust_buffer"
        pip install maturin
        maturin develop --release
        cd "$REPO_ROOT"
        echo "Rust Buffer built successfully!"
    else
        echo "WARNING: cargo not found. Skipping Rust Buffer build."
        echo "Install Rust from https://www.rust-lang.org/tools/install to build buffer."
    fi
fi

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "To use this environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run training/UI:"
echo "  bash scripts/start.sh"
echo ""
echo "To build Next.js frontend manually:"
echo "  cd web && npm install && npm run build"
echo ""
echo "To build Rust Buffer manually:"
echo "  cd rust_buffer && pip install maturin && maturin develop --release"
echo ""
