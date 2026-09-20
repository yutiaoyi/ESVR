#!/usr/bin/env bash
# Seed-noise estimate for the ESVR-Scaffold4D configurations (2026-09-20).
# J_*: the post-ablation defaults (flow 1, opacity init 0.5, random init).
# B_full_s*: extra seeds of ablation run B (flow 8, opacity 0.1, data init).
# K: fixed 6000 anchors with flow 1, to test the flow x growing interaction.
set -euo pipefail
cd "$(dirname "$0")/.."
# See run_scaffold4d_ablation.sh: never "source .venv/bin/activate" here.
PY=.venv/bin/python
OUT=output/scaffold4d/ablation
COMMON="--manifest experiments/vorts_90_all_train.json --eval-every 2000 --tile-sampling"
J="--flow-scale 1 --opacity-init 0.5 --init random"
B="--flow-scale 8 --opacity-init 0.1 --init data"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  $PY train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
case "$1" in
  1)
    run J_default_s42 $J --seed 42
    run J_default_s1 $J --seed 1
    run J_default_s2 $J --seed 2
    ;;
  2)
    run B_full_s1 $B --seed 1
    run B_full_s2 $B --seed 2
    run K_fixed6000_flow1 $J --seed 42 --densify-until 0
    ;;
esac
