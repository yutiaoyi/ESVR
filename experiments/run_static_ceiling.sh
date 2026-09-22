#!/usr/bin/env bash
# Static single-frame ESVR capacity sweep on vorts0001 (2026-09-21): what
# PSNR can this primitive + kernel reach on this data with more primitives
# and longer training? This bounds any dynamic model built on the same
# primitive. Recipe = experiments/run_vorts0001.sh with the CUDA-fixed
# position_lr (Adam is scale-invariant) and densify_grad_threshold x4096
# (position gradients are 4096x larger); the 09-13 "cuda_fixed_lr_only" runs
# had actually used the unfixed kernel (see RESULTS.md here).
# Schedules are in scans (x8 = iterations for 128^3 / 64^3).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python   # never "source .venv/bin/activate" here
$PY -c "import gaussian_rasterization_voxelization as g, sys; sys.exit(0 if '/ESVR-4dgs/' in g.__file__ else 1)" || { echo "wrong voxelizer"; exit 1; }
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/esvr-mpl}"
BASE="--chunk_idx 0 --dataset_name vorts0001 --subvol_size 64 --opacity_lr_init 0.0075 --opacity_lr_final 0.0005 --scaling_lr_init 0.00005 --scaling_lr_final 0.00005 --rotation_lr_init 0.00001 --rotation_lr_final 0.00001 --position_lr_delay_mult 0.01 --lambda_dssim 0.2 --densify_grad_threshold 0.001024 --percent_dense 0.008 --position_lr_init 0.000008 --position_lr_final 0.000001 --opacity_lr_max_steps 16 --densify_from_iter 16 --densification_interval 4 --opacity_reset_interval 64"
run() {  # name scans mult init_num prune_fraction
  name=$1; scans=$2; m=$3; init=$4; keep=$5
  [ -e "output/experiments/static_ceiling/${name}_chunk_0/psnr_log.txt" ] && { echo "skip $name"; return; }
  $PY train_chunk.py $BASE --output_dir experiments/static_ceiling/$name --iterations $scans \
     --densify_until_iter $((400*m)) --darkpruning_iter $((16*m)) $((200*m)) $((400*m)) $((1200*m)) $((1800*m)) $((2100*m)) $((2700*m)) \
     --globalpruning_iter $((1200*m)) --position_lr_max_steps $((1200*m)) --scaling_lr_max_steps $((1200*m)) --rotation_lr_max_steps $((1200*m)) \
     --save_iterations $scans --init_num $init --pruning_remainweight $keep > "output/experiments/static_ceiling_$name.log" 2>&1 || echo "FAILED $name"
  echo "done $name: $(tail -1 output/experiments/static_ceiling/${name}_chunk_0/psnr_log.txt 2>/dev/null)"
}
mkdir -p output/experiments/static_ceiling
run S1b_init20k_x1 2800 1 20000 0.66
run S2b_init60k_x1 2800 1 60000 0.66
run S3b_init20k_x3 8400 3 20000 0.66
run S4b_init60k_x3_noprune 8400 3 60000 0.0
run S5b_init150k_x3_noprune 8400 3 150000 0.0
