#!/bin/bash
# Smoke test entrypoint for the temporal training loop.

set -euo pipefail

export PYTHONUNBUFFERED=1
python examples/wanvideo/model_training/train_with_temporal.py \
  --device cpu \
  --steps 50 \
  --batch-size 1 \
  --num-frames 4 \
  --height 64 \
  --width 64 \
  --latent-channels 32 \
  --style-dim 0 \
  --log-interval 10 \
  --disable-clip
