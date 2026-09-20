#!/usr/bin/env bash
# Leave-one-out ablation of the ESVR-Scaffold4D adaptation fixes (2026-09-20).
# Every run: native 4D-Scaffold-GS recipe (120k steps, 7 LR groups, growing
# window 1500-60000) + the current train_scaffold4d.py defaults, except the
# single flag each run reverts. Usage: run_scaffold4d_ablation.sh <lane>
set -euo pipefail
cd "$(dirname "$0")/.."
# Never "source .venv/bin/activate" here: until 2026-09-20 the copied
# activate script pointed at ESVR/.venv, whose voxelizer has the unfixed
# position gradient. Every run in output/scaffold4d/ablation used it.
PY=.venv/bin/python
OUT=output/scaffold4d/ablation
COMMON="--manifest experiments/vorts_90_all_train.json --eval-every 2000 --tile-sampling"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  $PY train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
case "$1" in
  1)
    run B_full
    run D_flow1 --flow-scale 1
    run F_opacity_init0.5 --opacity-init 0.5
    run H_old_optimizer --adam-eps 1e-8 --clip-grad 1.0
    ;;
  2)
    run C_cap100k --max-anchors 100000
    run E_floor_old --scale-floor-voxels 0.0064
    run G_init_random --init random
    run I_no_sigma_weight --lambda-temporal-sigma 0
    ;;
esac
