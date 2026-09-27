#!/usr/bin/env bash
# 2026-09-27: 90-frame per-frame ESVR with normalized composition on ESVR's own recipes.
# S1nr (S1b recipe) on 2 lanes now; S5nr (S5b recipe) on 3 lanes once jet J2b has finished
# and >= 12 GB are free. Both without the periodic intensity reset (see RESULTS in
# output/experiments/per_frame_fixed). Log: output/scaffold4d/supervisor.log
cd "$(dirname "$0")/.."
LOG=output/scaffold4d/supervisor.log
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
echo "$(date) start S1nr x2 lanes ($(free_mb) MB free)" >> $LOG
for l in 0 1; do ( bash experiments/run_per_frame_fixed.sh S1nr $l 2; echo "$(date) S1nr lane $l exited $?" >> $LOG ) & done
until [ -e output/scaffold4d/jet/J2b_normalized_initdata_480k_cap400k/evaluation/metrics.json ] && [ "$(free_mb)" -ge 12000 ]; do sleep 120; done
echo "$(date) start S5nr x3 lanes ($(free_mb) MB free)" >> $LOG
for l in 0 1 2; do ( bash experiments/run_per_frame_fixed.sh S5nr $l 3; echo "$(date) S5nr lane $l exited $?" >> $LOG ) & sleep 60; done
wait
