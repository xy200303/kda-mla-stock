#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_qlib_real.py
python scripts/train.py \
  --model-config configs/model-fast.json \
  --train-config data/train-real.json \
  --output-dir outputs/kda-mla-fast \
  --batch-size 128 \
  --num-workers 4 \
  --train-stride 5 \
  --epochs 30 \
  --patience 6
