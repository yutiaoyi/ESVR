#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/esvr-mpl}"

python train_chunk.py \
  --chunk_idx 0 \
  --dataset_name vorts0001 \
  --output_dir experiments/vorts0001_baseline/vorts0001 \
  --subvol_size 64 \
  --iterations 2800 \
  --densify_from_iter 16 \
  --densify_until_iter 400 \
  --densification_interval 4 \
  --opacity_reset_interval 64 \
  --darkpruning_iter 16 200 400 1200 1800 2100 2700 \
  --save_iterations 2800 \
  --globalpruning_iter 1200 \
  --position_lr_max_steps 1200 \
  --opacity_lr_max_steps 16 \
  --scaling_lr_max_steps 1200 \
  --rotation_lr_max_steps 1200 \
  --position_lr_init 0.000008 \
  --position_lr_final 0.000001 \
  --opacity_lr_init 0.0075 \
  --opacity_lr_final 0.0005 \
  --scaling_lr_init 0.00005 \
  --scaling_lr_final 0.00005 \
  --rotation_lr_init 0.00001 \
  --rotation_lr_final 0.00001 \
  --position_lr_delay_mult 0.01 \
  --lambda_dssim 0.2 \
  --densify_grad_threshold 0.00000025 \
  --percent_dense 0.008 \
  --init_num 20000 \
  --pruning_remainweight 0.66
