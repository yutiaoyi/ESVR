#!/usr/bin/env bash
# Normalized explicit 4D primitives on vorts (2026-09-26). Single variable: initial temporal half-width.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
O=output/scaffold4d/norm4d
run() { name=$1; shift; [ -e "$O/$name/summary.json" ] && { echo "skip $name"; return; }; rm -rf "$O/$name"
  $PY experiments/train_norm4d.py --manifest experiments/vorts_90_all_train.json --output "$O/$name" "$@" > "$O/$name.log" 2>&1; }
case "$1" in
  A) run N4A_400k_hw2_90k --n 400000 --steps 90000 --init-halfwidth 2 ;;
  B) run N4B_400k_hw4_90k --n 400000 --steps 90000 --init-halfwidth 4 ;;
esac
