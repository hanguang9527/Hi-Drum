#!/bin/bash
set -euo pipefail

# Hi-Drum environment bootstrap for Ubuntu
# Recreates the Conda env used for training/evaluation.

ENV_NAME="drum"
PYTHON_VERSION="3.10.19"
REQ_FILE="requirements_ubuntu.txt"

echo "Configuring Conda env: $ENV_NAME ..."

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found. Install Anaconda/Miniconda first."
    exit 1
fi

if [ ! -f "$REQ_FILE" ]; then
    echo "Error: $REQ_FILE not found. Run this script from the repo root."
    exit 1
fi

echo "Creating Conda env ($ENV_NAME) with Python $PYTHON_VERSION ..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

# Activate inside bash scripts via conda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "Activating $ENV_NAME ..."
conda activate "$ENV_NAME"

echo "Upgrading pip ..."
python -m pip install --upgrade pip

echo "Installing dependencies from $REQ_FILE ..."
# Optional mirror for mainland China networks; remove -i for overseas hosts.
python -m pip install -r "$REQ_FILE" -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Environment ready."
echo "Activate with: conda activate $ENV_NAME"
