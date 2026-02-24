#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
read -p "Enter choice [1]: " choice

choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "Starting distributed training..."
        python "$REPO_ROOT/main.py"
        ;;
    2)
        echo ""
        echo "Starting single-agent training..."
        python "$REPO_ROOT/main.py" "$REPO_ROOT/config/config_single.yaml"
        ;;
    3)
        echo ""
        echo "Plotting training curves..."
        python "$REPO_ROOT/scripts/plot_training.py"
        echo "Curves saved to: output/training_curves.png"
        ;;
    4)
        echo ""
        echo "Plotting comparison curves..."
        python "$REPO_ROOT/scripts/plot_comparison.py"
        echo "Curves saved to: output/comparison_curves.png"
        ;;
    *)
        echo "ERROR: Invalid choice."
        exit 1
        ;;
esac

echo ""
echo "Done!"
echo ""
