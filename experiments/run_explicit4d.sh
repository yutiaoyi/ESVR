#!/usr/bin/env bash
# Explicit 4D ESVR primitives distilled from next/P3 and trained on all 90 frames (block = whole volume).
#   X1 ESVR learning rates (as in the frame-1 unshackle test U1)
#   X2 all learning rates / 3 (the X1 smoke run dipped 1 dB in the first 300 steps)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SRC=output/scaffold4d/next/P3_block128_thr1e-4_240k
M=experiments/vorts_90_all_train.json
OUT=output/scaffold4d/explicit
run() { name=$1; shift; [ -e "$OUT/$name/summary.json" ] && { echo "skip $name"; return; }; rm -rf "$OUT/$name"
  $PY experiments/finetune_explicit4d.py $SRC --manifest $M --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1; }
case "$1" in
  X1) run X1_fromP3_60k --steps 60000 ;;
  X2) run X2_fromP3_60k_lr3 --steps 60000 --lr-xyz 2.9e-5 3.7e-6 --lr-logit 1.8e-3 1.8e-4 --lr-scale 1.8e-4 --lr-rotation 3.7e-5 --lr-t 3.3e-5 3.3e-6 --lr-flow 3.3e-5 3.3e-6 --lr-sigma 3.3e-4 3.3e-5 ;;
esac
