#!/usr/bin/env bash
# Wait until next/W1 and next/W2 (Scaffold4D + normalized on vorts) have full
# 90-frame evaluations, then run the jet_Y_OH lane with the better of the two
# initialisations: J2b (data-sampled anchors) if W2 >= W1, else J2a.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=output/scaffold4d/supervisor.log
done90() { $PY -c "import json,sys;m=json.load(open('$1/evaluation/metrics.json'));sys.exit(0 if len(m)==90 else 1)" 2>/dev/null; }
mean() { $PY -c "import json,statistics as s;print(s.mean(x['psnr_db'] for x in json.load(open('$1/evaluation/metrics.json'))))"; }
W1=output/scaffold4d/next/W1_P3_normalized_240k; W2=output/scaffold4d/next/W2_P3_normalized_initdata_240k
until done90 $W1 && done90 $W2; do sleep 120; done
while pgrep -f "train_scaffold4d[.]py.*W[12]_P3_normalized" >/dev/null; do sleep 30; done
m1=$(mean $W1); m2=$(mean $W2)
lane=$($PY -c "print('J2b' if $m2 >= $m1 else 'J2a')")
echo "$(date) W1 $m1 W2 $m2 -> start jet $lane" >> $LOG
bash experiments/run_jet_scaffold4d.sh $lane
echo "$(date) jet $lane exited $?" >> $LOG
