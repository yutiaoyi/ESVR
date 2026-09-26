#!/usr/bin/env bash
# jet_Y_OH (480x720x120 float32, 122 frames, values [0, 0.002351]) with the best vorts recipe
# (next/P1b: block 128, growth threshold 3e-4, decay 1/100, tile sampling). The volume is 20x
# vorts, so the anchor cap is raised and steps doubled; a 128^3 tile is 1/24 of a frame here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python   # never "source .venv/bin/activate" here
OUT=output/scaffold4d/jet
mkdir -p "$OUT"
COMMON="--manifest experiments/jet_Y_OH_122_all_train.json --eval-every 40000 --tile-sampling --final-lr-scale 0.01 --block 128 --densify-grad-threshold 3e-4"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  $PY train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
case "$1" in
  smoke) run JET_smoke_2k --steps 2000 --anchors 20000 --max-anchors 40000 --eval-every 0 ;;
  J1)    run J1_block128_thr3e-4_480k_cap400k --steps 480000 --max-anchors 400000 --anchors 40000 ;;
esac
