#!/usr/bin/env bash
# Toward 40 dB (2026-09-21): H4 (150k anchors, 480k steps, decay 1/100)
# reached 39.43 dB, level with per-frame static ESVR. Each run changes one
# thing vs H4. Fixed voxelizer, native defaults otherwise; the ESVR
# adaptation recipe (--final-lr-scale 0.01) is passed explicitly.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python   # never "source .venv/bin/activate" here
$PY -c "import gaussian_rasterization_voxelization as g, sys; sys.exit(0 if '/ESVR-4dgs/' in g.__file__ else 1)" || { echo "wrong voxelizer"; exit 1; }
OUT=output/scaffold4d/long
mkdir -p "$OUT"
COMMON="--manifest experiments/vorts_90_all_train.json --eval-every 20000 --tile-sampling --snapshot-every 240000"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  $PY train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
case "$1" in
  1) run I1_150k_decay0.1_480k --max-anchors 150000 --steps 480000 --final-lr-scale 0.1 ;;
  2) run I2_150k_decay0.01_960k --max-anchors 150000 --steps 960000 --final-lr-scale 0.01 ;;
  3) run I3_300k_decay0.01_480k --max-anchors 300000 --steps 480000 --final-lr-scale 0.01 ;;
  # Steps-only control at the 40k cap: same as decay/H1_40k_decay0.01 but
  # 480k steps (growing window and decay stretch with --steps).
  4) run J1_40k_decay0.01_480k --max-anchors 40000 --steps 480000 --final-lr-scale 0.01 ;;
esac
