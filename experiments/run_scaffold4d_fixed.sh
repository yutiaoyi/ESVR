#!/usr/bin/env bash
# Re-run of the headline ESVR-Scaffold4D configurations with the FIXED CUDA
# voxelizer (2026-09-20). output/scaffold4d/ablation was trained through the
# unfixed kernel by mistake (see its RESULTS.md banner). Same recipe as
# there: native 4D-Scaffold-GS hyperparameters, 120k steps, growing window
# 1500-60000, current train_scaffold4d.py defaults; the growing threshold is
# recalibrated to the fixed kernel's gradient scale (p90 ~1.2e-3).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python   # never "source .venv/bin/activate" here
$PY -c "import gaussian_rasterization_voxelization as g, sys; sys.exit(0 if '/ESVR-4dgs/' in g.__file__ else 1)" || { echo "wrong voxelizer"; exit 1; }
OUT=output/scaffold4d/fixed_kernel
mkdir -p "$OUT"
COMMON="--manifest experiments/vorts_90_all_train.json --eval-every 2000 --tile-sampling --densify-grad-threshold 1.2e-3"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  $PY train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
case "$1" in
  1)
    run F_default_s42 --seed 42
    run F_cap100k --seed 42 --max-anchors 100000
    run F_default_s1 --seed 1
    ;;
  2)
    run F_fixed6000 --seed 42 --densify-until 0
    run F_floor_old --seed 42 --scale-floor-voxels 0.0064
    run F_default_s2 --seed 2
    ;;
esac
