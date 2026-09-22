#!/usr/bin/env bash
# Toward 50 dB (2026-09-22). Baseline H4: 150k anchors, 480k steps, decay
# 1/100, fixed voxelizer, 39.43 dB. Each run changes one thing.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python   # never "source .venv/bin/activate" here
OUT=output/scaffold4d/fifty
mkdir -p "$OUT"
COMMON="--eval-every 20000 --tile-sampling --max-anchors 150000 --final-lr-scale 0.01"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  $PY train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
case "$1" in
  # Optimisation- or representation-limited? Same steps on 10 frames = 9x more optimisation per frame.
  1) run K1_10frames_480k --manifest experiments/vorts_10_frames_41_50.json --steps 480000 --grow-t-size 0.011235955 ;;  # t cell = one frame of the 90-frame range, not 1/(10-1)
  # Whole-volume supervision per step; 60k steps of 128^3 ~ the compute of 480k steps of 64^3.
  2) run K2_block128_60k --manifest experiments/vorts_90_all_train.json --steps 60000 --block 128 ;;
  # Growing for 75 % of training instead of 50 %.
  3) run K3_grow75_480k --manifest experiments/vorts_90_all_train.json --steps 480000 --densify-until 360000 ;;
  # Small trained model: 60k cap with ESVR-style dead-anchor recycling.
  4) run K4_cap60k_prune_480k --manifest experiments/vorts_90_all_train.json --steps 480000 --max-anchors 60000 --prune-dead ;;
esac
