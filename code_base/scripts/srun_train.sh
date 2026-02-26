#!/usr/bin/env bash
set -euo pipefail

# EDIT for MSI:
PARTITION="<gpu_partition>"
GRES="gpu:1"
MEM="16G"
TIME="02:00:00"

export TORCH_HOME=$PWD/.torch_cache

srun --partition="$PARTITION" --gres="$GRES" --mem="$MEM" --time="$TIME" \
  bash -lc "source .venv/bin/activate && export TORCH_HOME=$PWD/.torch_cache && \
  python -m src.train \
    --data-root data/raw \
    --split-dir data/splits \
    --ckpt-dir checkpoints \
    --epochs 6 \
    --batch-size 128 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --img-size 224 \
    --num-workers 4 \
    --seed 0 \
    --pretrained"
