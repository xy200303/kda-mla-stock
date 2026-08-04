#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_qlib_real.py
python scripts/train.py --train-config data/train-real.json
