#!/usr/bin/env bash
# Memory-aware start of W3 and W4 (2026-09-27) next to the running jet_Y_OH J2b
# (which keeps growing anchors until step 240k): each lane starts only with
# >= 18 GB free, and the second waits 30 min after the first so the first has
# reached its working size. Log: output/scaffold4d/supervisor.log
cd "$(dirname "$0")/.."
LOG=output/scaffold4d/supervisor.log
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for lane in W3 W4; do
  until [ "$(free_mb)" -ge 18000 ]; do sleep 120; done
  echo "$(date) start $lane ($(free_mb) MB free)" >> $LOG
  ( bash experiments/run_scaffold4d_next.sh $lane; echo "$(date) $lane lane exited $?" >> $LOG ) &
  sleep 1800
done
wait
