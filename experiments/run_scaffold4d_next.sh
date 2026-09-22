#!/usr/bin/env bash
# Next round after the 50 dB probes (2026-09-22). Baseline H4: 150k anchors,
# 480k steps, decay 1/100, fixed voxelizer, 39.43 dB. Single-variable each.
#   L1  sigma-max 100 -> 1000 (37-45 % of children sit on the 100 clamp in H4/I3)
#   E   "grouping at equal total compute": 9 groups x 10 frames, 53,333 steps and
#       16,667-anchor cap each (sum = H4's 480k steps / 150k anchors)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python   # never "source .venv/bin/activate" here
OUT=output/scaffold4d/next
mkdir -p "$OUT"
COMMON="--eval-every 20000 --tile-sampling --final-lr-scale 0.01"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  $PY train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
group() {  # group index 1..9
  g=$1; lo=$(( (g-1)*10+1 )); hi=$(( lo+9 ))
  run E_group${g}_frames_$(printf %02d $lo)_$(printf %02d $hi) --manifest experiments/vorts_group${g}_frames_$(printf %02d $lo)_$(printf %02d $hi).json \
      --steps 53333 --max-anchors 16667 --grow-t-size 0.011235955   # t cell = one frame of the 90-frame range
}
case "$1" in
  L1) run L1_sigma1000_480k --manifest experiments/vorts_90_all_train.json --steps 480000 --max-anchors 150000 --sigma-max 1000 ;;
  EA) for g in 1 3 5 7 9; do group $g; done ;;
  EB) for g in 2 4 6 8; do group $g; done ;;
esac
