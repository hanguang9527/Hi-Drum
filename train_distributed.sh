#!/usr/bin/env bash
set -euo pipefail

# Launch distributed training with torchrun.
# Usage:
#   ./train_distributed.sh [NUM_GPUS] [extra sacred overrides...]
# Examples:
#   ./train_distributed.sh 2
#   ./train_distributed.sh 8 iterations=50000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS="${1:-2}"
shift || true

PYTHON_BIN="${PYTHON_BIN:-/data1/weiyangding/miniforge3/envs/drum/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  train.py with "$@"
