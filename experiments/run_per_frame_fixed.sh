#!/usr/bin/env bash
# 90-frame per-frame static ESVR on the FIXED voxelizer (2026-09-26): the honest
# "ESVR baseline" for the time-varying models. All fixed-kernel per-frame numbers
# so far (S1b 41.72, S3b 44.73, S5b 49.67) came from frame 1 only, which is the
# easiest frame of the sequence (unfixed-kernel 90-frame run: frame 1 = 41.77,
# mean 41.22). Recipes are those of experiments/run_static_ceiling.sh:
#   S1b = ESVR's own vorts recipe (20k init, 2,800 scans = 22,400 iters, 66 % global prune)
#   S5b = capacity ceiling (150k init, 8,400 scans = 67,200 iters, no global prune)
# Usage: run_per_frame_fixed.sh RECIPE LANE NLANES   (frames with (i-1) % NLANES == LANE)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python   # never "source .venv/bin/activate" here
$PY -c "import gaussian_rasterization_voxelization as g, sys; sys.exit(0 if '/ESVR-4dgs/' in g.__file__ else 1)" || { echo "wrong voxelizer"; exit 1; }
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/esvr-mpl}"
recipe=$1; lane=$2; nlanes=$3; extra=""
case "$recipe" in
  S1b) scans=2800; m=1; init=20000;  keep=0.66 ;;
  S5b) scans=8400; m=3; init=150000; keep=0.0 ;;
  # 2026-09-27 fairness control: the same recipes with data-sampled init (train_chunk --init_from_data)
  S1d) scans=2800; m=1; init=20000;  keep=0.66; extra=--init_from_data ;;
  S5d) scans=8400; m=3; init=150000; keep=0.0;  extra=--init_from_data ;;
  *) echo "unknown recipe $recipe"; exit 1 ;;
esac
OUT=output/experiments/per_frame_fixed/$recipe
mkdir -p "$OUT"
for i in $(seq 1 90); do
  [ $(( (i - 1) % nlanes )) -eq "$lane" ] || continue
  name=$(printf "vorts_frame_%04d" "$i")
  [ -e "$OUT/${name}_chunk_0/psnr_log.txt" ] && grep -q "Total training time" "$OUT/${name}_chunk_0/psnr_log.txt" && { echo "skip $name"; continue; }
  $PY train_chunk.py --chunk_idx 0 --dataset_name "$name" --subvol_size 64 \
     --opacity_lr_init 0.0075 --opacity_lr_final 0.0005 --scaling_lr_init 0.00005 --scaling_lr_final 0.00005 \
     --rotation_lr_init 0.00001 --rotation_lr_final 0.00001 --position_lr_delay_mult 0.01 --lambda_dssim 0.2 \
     --densify_grad_threshold 0.001024 --percent_dense 0.008 --position_lr_init 0.000008 --position_lr_final 0.000001 \
     --opacity_lr_max_steps 16 --densify_from_iter 16 --densification_interval 4 --opacity_reset_interval 64 \
     --output_dir experiments/per_frame_fixed/$recipe/$name --iterations $scans \
     --densify_until_iter $((400*m)) --darkpruning_iter $((16*m)) $((200*m)) $((400*m)) $((1200*m)) $((1800*m)) $((2100*m)) $((2700*m)) \
     --globalpruning_iter $((1200*m)) --position_lr_max_steps $((1200*m)) --scaling_lr_max_steps $((1200*m)) --rotation_lr_max_steps $((1200*m)) \
     --save_iterations $scans --init_num $init --pruning_remainweight $keep $extra > "$OUT/$name.log" 2>&1 || echo "FAILED $name"
done
