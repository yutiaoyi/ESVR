#!/usr/bin/env bash
# Memory-aware queue: S1 when >= 12 GB free; then the jet smoke and J1 only when no
# train_scaffold4d process is left and >= 28 GB is free. Log: output/scaffold4d/supervisor.log
cd "$(dirname "$0")/.."
LOG=output/scaffold4d/supervisor.log
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
busy() { pgrep -f "train_scaffold4d[.]py" >/dev/null; }
until [ "$(free_mb)" -ge 12000 ]; do sleep 120; done
echo "$(date) start S1 ($(free_mb) MB free)" >> $LOG; bash experiments/run_scaffold4d_next.sh S1; echo "$(date) S1 lane exited $?" >> $LOG
while busy || [ "$(free_mb)" -lt 28000 ]; do sleep 180; done
echo "$(date) start jet smoke ($(free_mb) MB free)" >> $LOG; bash experiments/run_jet_scaffold4d.sh smoke; echo "$(date) jet smoke exited $?" >> $LOG
sleep 30
echo "$(date) start J1 ($(free_mb) MB free)" >> $LOG; bash experiments/run_jet_scaffold4d.sh J1; echo "$(date) J1 lane exited $?" >> $LOG
