#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

ENV_NAME="drl-arm"

echo "======================================"
echo "DRL MuJoCo - Training Launcher"
echo "======================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: 'conda' command not found."
    echo "Please run 'bash build.sh' first to set up the environment."
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if environment exists
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "ERROR: Conda environment '$ENV_NAME' not found."
    echo "Please run 'bash build.sh' first to set up the environment."
    exit 1
fi

conda activate "$ENV_NAME"

# Create output directory if it doesn't exist
mkdir -p "$REPO_ROOT/output"

echo ""
echo "Select mode:"
echo "1) Run distributed training (8 actors)"
echo "2) Run single-agent training (1 actor)"
echo "3) Plot training curves"
echo "4) Plot comparison curves"
echo "5) Launch Web UI (FastAPI backend)"
echo "6) Launch Next.js Dev Server + Web UI (推荐)"
read -p "Enter choice [1]: " choice

choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "Starting distributed training..."
        "$CONDA_PREFIX/bin/python" "$REPO_ROOT/main.py"
        ;;
    2)
        echo ""
        echo "Starting single-agent training..."
        "$CONDA_PREFIX/bin/python" "$REPO_ROOT/main.py" "$REPO_ROOT/config/config_single.yaml"
        ;;
    3)
        echo ""
        echo "Plotting training curves..."
        "$CONDA_PREFIX/bin/python" "$REPO_ROOT/scripts/plot_training.py"
        echo "Curves saved to: output/training_curves.png"
        ;;
    4)
        echo ""
        echo "Plotting comparison curves..."
        "$CONDA_PREFIX/bin/python" "$REPO_ROOT/scripts/plot_comparison.py"
        echo "Curves saved to: output/comparison_curves.png"
        ;;
    5)
        echo ""
        echo "Launching Web UI..."
        echo "Open your browser and go to: http://127.0.0.1:8000"
        "$CONDA_PREFIX/bin/python" "$REPO_ROOT/web/server.py"
        ;;
    6)
        echo ""
        if ! command -v npm &> /dev/null; then
            echo "ERROR: npm not found."
            echo "Install Node.js from https://nodejs.org/"
            exit 1
        fi
        cd "$REPO_ROOT/web"
        if [ ! -d "node_modules" ]; then
            echo "Installing npm dependencies..."
            npm install
        fi
        echo "Launching Next.js Dev Server + Web UI..."
        echo "Open your browser and go to: http://localhost:3000"
        echo "Next.js will proxy /api and /ws to FastAPI backend at http://127.0.0.1:8000"
        echo "Starting FastAPI backend in background..."
        cd "$REPO_ROOT"
        "$CONDA_PREFIX/bin/python" "$REPO_ROOT/web/server.py" > /tmp/fastapi.log 2>&1 &
        FASTAPI_PID=$!
        echo "Waiting for FastAPI backend to start..."
        for i in {1..20}; do
            if curl -s "http://127.0.0.1:8000/" > /dev/null 2>&1; then
                echo "FastAPI backend is ready!"
                break
            fi
            echo "  Waiting... ($i/20)"
            sleep 1
        done
        cd "$REPO_ROOT/web"
        echo "Starting Next.js dev server..."
        npm run dev
        echo "Stopping FastAPI backend..."
        kill $FASTAPI_PID 2>/dev/null || true
        cd "$REPO_ROOT"
        ;;
    *)
        echo "ERROR: Invalid choice."
        exit 1
        ;;
esac

echo ""
echo "Done!"
echo ""
