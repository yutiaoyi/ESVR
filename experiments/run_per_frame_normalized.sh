#!/usr/bin/env bash
# Fairness baseline (2026-09-27): per-frame static ESVR with NORMALIZED
# composition, i.e. the recipe of experiments/static_normalized.py (data-sampled
# init, L1, cosine LR 8e-3, 3000 whole-volume steps, relocation on) run
# independently on each of the 90 vorts frames. Separates "normalization" from
# "4D sharing": compare against S1b/S5b (additive per-frame) and N4A/W1/W2 (4D).
# Lanes differ only in primitives per frame (12 fp32 floats each):
#   PN21 21000/frame = 90.7 MB  (size of S1b, 93.8 MB)
#   PN10 10000/frame = 43.2 MB  (size of W1/W2)
#   PN6   6300/frame = 27.2 MB  (size of N4A)
# Waits for W1/W2 to finish, and before each frame for >= 6 GB free GPU memory.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
O=output/experiments/per_frame_normalized
mkdir -p $O
W=output/scaffold4d/next
done90() { $PY -c "import json,sys;m=json.load(open('$1/evaluation/metrics.json'));sys.exit(0 if len(m)==90 else 1)" 2>/dev/null; }
until done90 $W/W1_P3_normalized_240k && done90 $W/W2_P3_normalized_initdata_240k; do sleep 120; done
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
lane() { name=$1 n=$2; mkdir -p $O/$name
  for f in $(seq 0 89); do
    d=$O/$name/frame_$(printf %02d $f)
    [ -e $d/result.json ] && continue
    until [ "$(free_gb)" -ge 6 ]; do sleep 60; done
    $PY experiments/static_normalized.py --mode normalized --n $n --steps 3000 --frame $f --output $d > $d.log 2>&1 || echo "FAILED $name frame $f" >> $O/lane.out
  done
  $PY - $O/$name <<'PY'
import glob, json, statistics, sys
rs = [json.load(open(p)) for p in sorted(glob.glob(sys.argv[1] + "/frame_*/result.json"))]
ps = [r["psnr_db"] for r in rs]
s = dict(frames=len(rs), n_per_frame=rs[0]["n"], mean_psnr=statistics.mean(ps), min_psnr=min(ps), max_psnr=max(ps),
         mb_fp32=sum(r["mb_fp32"] for r in rs), mb_fp16=sum(r["mb_fp32"] for r in rs) / 2)
json.dump(s, open(sys.argv[1] + "/summary.json", "w"), indent=2); print(json.dumps(s))
PY
  echo "$(date) $name: $(cat $O/$name/summary.json | tr -d '\n ')" >> $O/lane.out; }
lane PN21 21000
lane PN10 10000
lane PN6 6300
