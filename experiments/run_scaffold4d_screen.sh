#!/usr/bin/env bash
# 30k-step screening of newborn-anchor fixes (2026-09-20). Diagnosis: grown
# anchors' children are born stacked at one point (offset 0) at the scale
# floor (cell 1 voxel x sigmoid 0.5), die under additive accumulation, and
# stay pinned; fixed-population runs do not show this. Each run changes one
# thing vs S0; all: 30k steps, growing window 1500-15000, cap 40k, seed 42.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=output/scaffold4d/screen
mkdir -p "$OUT"
COMMON="--manifest experiments/vorts_90_all_train.json --eval-every 5000 --tile-sampling --steps 30000"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  python train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
case "$1" in
  1) run S0_control; run S3_spread --offset-spread 0.5; run S6_sigma1000 --sigma-max 1000 ;;
  2) run S4_childscale4 --child-scale-init-voxels 4; run S1_novol --lambda-vol 0; run S7_lowlr --scale-lr 0.0007 --shape-mlp-lr 0.0004 ;;
  3) run S5_spread_childscale4 --offset-spread 0.5 --child-scale-init-voxels 4; run S2_floor1 --scale-floor-voxels 1.0; run S8_spread_cs4_novol --offset-spread 0.5 --child-scale-init-voxels 4 --lambda-vol 0 ;;
esac
