#!/bin/bash
# run_training.sh — launches training with GPU CUDA libs from the venv's
# nvidia-*-cu12 pip packages visible to TensorFlow.
#
# Usage: bash run_training.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# Collect all nvidia lib dirs from the venv
NVIDIA_LIBS=$(find "$VENV/lib" -path "*/nvidia/*/lib" -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}"

echo "✓ LD_LIBRARY_PATH set for GPU training"
exec "$VENV/bin/python" "$SCRIPT_DIR/optimized_emotion_model.py" "$@"
