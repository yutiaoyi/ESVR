#!/usr/bin/env bash
# Phase 2 toward 40 dB (2026-09-20): budget and optimisation steps, fixed
# voxelizer. The capacity-recycling screen (output/scaffold4d/screen_fixed,
# T0-T7) found no fix that helps at 40k anchors / 40k steps, so this scales
# the plain default recipe: anchor cap x3.75-6 and steps x4. Growing window
# and LR decay follow --steps (densify_until = steps/2). K=5 pairs are
# roughly size-matched to the K=10 runs (63 vs 83 floats per anchor).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python   # never "source .venv/bin/activate" here
$PY -c "import gaussian_rasterization_voxelization as g, sys; sys.exit(0 if '/ESVR-4dgs/' in g.__file__ else 1)" || { echo "wrong voxelizer"; exit 1; }
OUT=output/scaffold4d/scale
mkdir -p "$OUT"
COMMON="--manifest experiments/vorts_90_all_train.json --eval-every 10000 --tile-sampling --snapshot-every 120000"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  $PY train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
case "$1" in
  1) run G1_K10_150k_120k --max-anchors 150000
     run G3_K10_150k_480k --max-anchors 150000 --steps 480000 ;;
  2) run G2_K5_240k_120k --children 5 --max-anchors 240000
     run G4_K5_240k_480k --children 5 --max-anchors 240000 --steps 480000 ;;
  # G1's monitored PSNR oscillates +-0.7 dB between late checkpoints: the
  # natively constant LR groups never decay. Single-variable test vs G1.
  3) run G5_K10_150k_120k_decay --max-anchors 150000 --final-lr-scale 0.01 ;;
esac
