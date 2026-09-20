#!/usr/bin/env bash
# Around G5 (2026-09-20): decaying the natively constant LR groups to 1/100
# over training took the 150k-anchor / 120k-step model from 32.99 to
# 38.04 dB. Single-variable controls on the decay factor, budget, steps and
# seed. Fixed voxelizer, current defaults otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python   # never "source .venv/bin/activate" here
$PY -c "import gaussian_rasterization_voxelization as g, sys; sys.exit(0 if '/ESVR-4dgs/' in g.__file__ else 1)" || { echo "wrong voxelizer"; exit 1; }
OUT=output/scaffold4d/decay
mkdir -p "$OUT"
COMMON="--manifest experiments/vorts_90_all_train.json --eval-every 10000 --tile-sampling"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  $PY train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
case "$1" in
  1) run H1_40k_decay0.01 --final-lr-scale 0.01
     run H2_150k_decay0.1 --max-anchors 150000 --final-lr-scale 0.1
     run H6_150k_decay0.01_s1 --max-anchors 150000 --final-lr-scale 0.01 --seed 1 ;;
  2) run H3_150k_decay0.001 --max-anchors 150000 --final-lr-scale 0.001
     run H5_300k_decay0.01 --max-anchors 300000 --final-lr-scale 0.01 ;;
  3) run H4_150k_decay0.01_480k --max-anchors 150000 --final-lr-scale 0.01 --steps 480000 --snapshot-every 120000 ;;
esac
