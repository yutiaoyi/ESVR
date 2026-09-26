#!/usr/bin/env bash
# Single-frame (vorts frame 1) additive vs normalized kernel regression (2026-09-26).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
O=output/experiments/normalized
run() { name=$1; shift; [ -e "$O/$name/result.json" ] && { echo "skip $name"; return; }
  $PY experiments/static_normalized.py --output "$O/$name" "$@" > "$O/$name.log" 2>&1 || echo "FAILED $name"
  echo "$name: $(grep DONE $O/$name.log | cut -c1-160)"; }
run N_norm32k_3000     --mode normalized --n 32000 --steps 3000
run N_add32k_3000      --mode additive   --n 32000 --steps 3000
run N_norm32k_3000_noreloc --mode normalized --n 32000 --steps 3000 --relocate ""
run N_norm10k_3000     --mode normalized --n 10000 --steps 3000
run N_norm21k_3000     --mode normalized --n 21000 --steps 3000
run N_norm64k_3000     --mode normalized --n 64000 --steps 3000
run N_add139k_3000     --mode additive   --n 139000 --steps 3000
run N_norm32k_3000_f45 --mode normalized --n 32000 --steps 3000 --frame 44
