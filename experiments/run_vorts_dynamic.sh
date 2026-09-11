#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/esvr-mpl}"
.venv/bin/python -u train_dynamic.py \
  --manifest experiments/vorts_90_all_train.json \
  --output output/dynamic/vorts90_full_geometry \
  --init-ply output/experiments/vorts0001_baseline/vorts0001_chunk_0/point_cloud/iteration_22400/point_cloud.ply \
  --steps 40000 --warmup 0 --block 64 --tile-sampling \
  --depth 8 --width 256 --xyz-bands 10 --time-bands 10 \
  --lr 0.00005 --deform-lr 0.00016 --lambda-ssim 0.2 --motion-reg 0.0001 \
  --eval-every 2000 --snapshot-every 10000 --seed 42
