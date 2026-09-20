#!/usr/bin/env bash
# 40k-step screening of capacity-recycling fixes on the FIXED voxelizer
# (2026-09-20; the earlier S0-S8 screen was trained through the unfixed one).
# Diagnosis on output/scaffold4d/fixed_kernel/F_default_s42: only 23 % of
# children useful, 39 % of anchors without a useful child, 60 % of rendered
# children cover no voxel. Each run changes one thing vs T0; all: 40k steps,
# growing window 1500-20000, cap 40k, seed 42, current defaults otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python   # never "source .venv/bin/activate" here
$PY -c "import gaussian_rasterization_voxelization as g, sys; sys.exit(0 if '/ESVR-4dgs/' in g.__file__ else 1)" || { echo "wrong voxelizer"; exit 1; }
OUT=output/scaffold4d/screen_fixed
mkdir -p "$OUT"
COMMON="--manifest experiments/vorts_90_all_train.json --eval-every 5000 --tile-sampling --steps 40000 --densify-until 20000"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  $PY train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
case "$1" in
  1) run T0_control; run T1_volreg_visible --vol-reg-visible; run T2_prune_dead --prune-dead; run T3_spread --offset-spread 0.5 ;;
  2) run T4_childscale4 --child-scale-init-voxels 4; run T5_sigma1000 --sigma-max 1000; run T6_novol --lambda-vol 0
     run T7_combo --vol-reg-visible --prune-dead --offset-spread 0.5 --child-scale-init-voxels 4 ;;
esac
