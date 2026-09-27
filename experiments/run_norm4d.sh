#!/usr/bin/env bash
# Normalized explicit 4D primitives on vorts (2026-09-26). Single variable: initial temporal half-width.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
O=output/scaffold4d/norm4d
WREC="--steps 240000 --eval-every 20000 --lambda-ssim 0.2 --lambda-vol 0.01 --lr-schedule expon --final-lr-scale 0.01 --floor-vox 0.5 --sigma-max 100 --seed 42 --init-halfwidth 2"
run() { name=$1; shift; [ -e "$O/$name/summary.json" ] && { echo "skip $name"; return; }; rm -rf "$O/$name"
  $PY experiments/train_norm4d.py --manifest experiments/vorts_90_all_train.json --output "$O/$name" "$@" > "$O/$name.log" 2>&1; }
case "$1" in
  A) run N4A_400k_hw2_90k --n 400000 --steps 90000 --init-halfwidth 2 ;;
  B) run N4B_400k_hw4_90k --n 400000 --steps 90000 --init-halfwidth 4 ;;
  # 2026-09-27 fair comparison with W2 (Scaffold4D + normalized + data init): the explicit 4D
  # primitives trained with W2's recipe -- 240k steps, 0.8 L1 + 0.2 SSIM + 0.01 volume term,
  # log-linear LR decay to 1 %, scale floor 0.5 voxel, sigma_max 100, seed 42; whole-volume
  # random-frame steps as W2 (block 128 = the whole vorts volume). Only the representation
  # differs: 17 free floats per primitive instead of anchors + shared MLPs, fixed count
  # (W2 grows anchors). Counts: size-matched to W2 (585k x 68 B = 39.8 MB), count-matched to
  # W2's children (1,198,420 = 81.5 MB), and N4A's 400k (27.2 MB; isolates the recipe vs N4A).
  W585) run N4W_585k_Wrecipe_240k $WREC --n 585000 ;;
  W1198) run N4W_1198k_Wrecipe_240k $WREC --n 1198420 ;;
  W400) run N4W_400k_Wrecipe_240k $WREC --n 400000 ;;
esac
