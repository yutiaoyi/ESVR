#!/usr/bin/env bash
# Parameterisation diagnostic on vorts frame 1 (2026-09-26): optimise a scaffold4d
# time slice as FREE ESVR primitives with ESVR's own optimiser (S1b learning rates,
# 2,800 scans = 22,400 iterations).
#   U1 slice, no densification, no pruning      -> gain of freeing the parameters at fixed count/layout
#   U2 same count, random init, no densify/prune -> free primitives of the same count from scratch
#   U3 slice + ESVR densification (S1b schedule, global prune off) -> what extra capacity adds on top
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -c "import gaussian_rasterization_voxelization as g, sys; sys.exit(0 if '/ESVR-4dgs/' in g.__file__ else 1)" || { echo "wrong voxelizer"; exit 1; }
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/esvr-mpl}"
LR="--opacity_lr_init 0.0075 --opacity_lr_final 0.0005 --scaling_lr_init 0.00005 --scaling_lr_final 0.00005 --rotation_lr_init 0.00001 --rotation_lr_final 0.00001 --position_lr_delay_mult 0.01 --lambda_dssim 0.2 --position_lr_init 0.000008 --position_lr_final 0.000001 --opacity_lr_max_steps 16 --position_lr_max_steps 1200 --scaling_lr_max_steps 1200 --rotation_lr_max_steps 1200 --percent_dense 0.008 --densify_grad_threshold 0.001024 --densify_from_iter 16 --densification_interval 4 --opacity_reset_interval 64"
NODENS="--densify_until_iter 0 --darkpruning_iter 99999 --globalpruning_iter 99999 --pruning_remainweight 0.0"
DENS="--densify_until_iter 400 --darkpruning_iter 16 200 400 1200 1800 2100 2700 --globalpruning_iter 1200 --pruning_remainweight 0.0"
run() { name=$1; ck=$2; shift 2
  [ -e "output/experiments/unshackle/${name}_chunk_0/psnr_log.txt" ] && grep -q "Total training time" "output/experiments/unshackle/${name}_chunk_0/psnr_log.txt" && { echo "skip $name"; return; }
  $PY train_chunk.py --chunk_idx 0 --dataset_name vorts_frame_0001 --subvol_size 64 $LR --iterations 2800 --save_iterations 2800 --init_num 1000 \
     --output_dir experiments/unshackle/$name --start_checkpoint "$ck" "$@" > "output/experiments/unshackle/$name.log" 2>&1 || echo "FAILED $name"
  echo "$name: $(grep '|' output/experiments/unshackle/${name}_chunk_0/psnr_log.txt | tail -1 | awk -F'|' '{print $4, $5}')"
}
case "$1" in
  U1) run U1_P3slice_free_nodens output/experiments/unshackle/P3_frame1_slice.pth $NODENS ;;
  U2) run U2_random60813_nodens output/experiments/unshackle/random60813.pth $NODENS ;;
  U3) run U3_P3slice_free_densify output/experiments/unshackle/P3_frame1_slice.pth $DENS ;;
esac
