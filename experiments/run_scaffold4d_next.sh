#!/usr/bin/env bash
# Next round after the 50 dB probes (2026-09-22). Baseline H4: 150k anchors,
# 480k steps, decay 1/100, fixed voxelizer, 39.43 dB. Single-variable each.
#   L1  sigma-max 100 -> 1000 (37-45 % of children sit on the 100 clamp in H4/I3)
#   M1  error-proportional block sampling (--error-sampling 1.0)
#   N1-N6 base-LR sweep x3 / x0.3 for feature, offset, MLPs (120k steps, vs scale/G5)
#   O1  N2+N3+N5 combined at 480k (vs H4); O2 offset LR x10 at 120k (vs G5)
#   P1  block 128^3 at equal wall-clock to H4 (240k steps)
#   Q1  H4 recipe with the straight-through scale floor (gradient passes below the floor)
#   E   "grouping at equal total compute": 9 groups x 10 frames, 53,333 steps and
#       16,667-anchor cap each (sum = H4's 480k steps / 150k anchors)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python   # never "source .venv/bin/activate" here
OUT=output/scaffold4d/next
mkdir -p "$OUT"
COMMON="--eval-every 20000 --tile-sampling --final-lr-scale 0.01"
run() {
  name=$1; shift
  [ -e "$OUT/$name/evaluation/metrics.json" ] && { echo "skip $name"; return; }
  rm -rf "$OUT/$name"
  $PY train_scaffold4d.py $COMMON --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
}
group() {  # group index 1..9
  g=$1; lo=$(( (g-1)*10+1 )); hi=$(( lo+9 ))
  run E_group${g}_frames_$(printf %02d $lo)_$(printf %02d $hi) --manifest experiments/vorts_group${g}_frames_$(printf %02d $lo)_$(printf %02d $hi).json \
      --steps 53333 --max-anchors 16667 --grow-t-size 0.011235955   # t cell = one frame of the 90-frame range
}
case "$1" in
  L1) run L1_sigma1000_480k --manifest experiments/vorts_90_all_train.json --steps 480000 --max-anchors 150000 --sigma-max 1000 ;;
  # Error-proportional (frame, tile) sampling, exponent 1 (block MSE), 25 % uniform share.
  M1) run M1_errsample_480k --manifest experiments/vorts_90_all_train.json --steps 480000 --max-anchors 150000 --error-sampling 1.0 ;;
  EA) for g in 1 3 5 7 9; do group $g; done ;;
  # Base-LR sweep, single variable each, on the 120k-step G5 recipe (150k cap, decay 1/100, 38.04 dB).
  # Upstream's seven LRs were tuned for multi-camera colour scenes; only the decay has been adapted so far.
  # Combine the three positive LR directions of N1-N6 on the 480k-step H4 recipe; offset x10 probe at 120k.
  O1) run O1_lr_combo_480k --manifest experiments/vorts_90_all_train.json --steps 480000 --max-anchors 150000 \
        --feat-lr 0.00225 --offset-lr-init 0.03 --offset-lr-final 0.0003 --shape-mlp-lr 0.012 --flow-mlp-lr 0.003 --opacity-mlp-lr-init 0.006 --opacity-mlp-lr-final 0.00006 ;;
  O2) run O2_offset_x10_120k --manifest experiments/vorts_90_all_train.json --steps 120000 --max-anchors 150000 --eval-every 10000 --offset-lr-init 0.1 --offset-lr-final 0.001 ;;
  # Equal wall-clock control for block size: a 128^3 step costs 51 ms vs 30 ms for 64^3 (profiled on an idle GPU),
  # so 240k steps of 128^3 ~= H4's 480k steps of 64^3 in time, with 4x the supervised voxels overall.
  P1) run P1_block128_240k --manifest experiments/vorts_90_all_train.json --steps 240000 --max-anchors 150000 --block 128 ;;
  # Straight-through scale floor (third review F1): same command as H4, code after commit that fixed the clamp_min gradient trap.
  Q1) run Q1_floor_st_480k --manifest experiments/vorts_90_all_train.json --steps 480000 --max-anchors 150000 ;;
  # P1 stalled at 14k anchors: the growth threshold calibrated on 64^3-block gradients does not transfer to whole-volume
  # steps (smoother gradient estimates, thinner tail above the threshold). P1b lowers it 4x; exploratory.
  P1b) run P1b_block128_thr3e-4_240k --manifest experiments/vorts_90_all_train.json --steps 240000 --max-anchors 150000 --block 128 --densify-grad-threshold 3e-4 ;;
  # Follow-ups to P1b (block 128, thr 3e-4, 240k = 40.17 dB): more steps; lower threshold (more anchors).
  P2) run P2_block128_thr3e-4_480k --manifest experiments/vorts_90_all_train.json --steps 480000 --max-anchors 150000 --block 128 --densify-grad-threshold 3e-4 ;;
  P3) run P3_block128_thr1e-4_240k --manifest experiments/vorts_90_all_train.json --steps 240000 --max-anchors 150000 --block 128 --densify-grad-threshold 1e-4 ;;
  # On the P1b recipe (block 128, thr 3e-4, 240k): residual-driven growing; ESVR score pruning with regrowth.
  R1) run R1_block128_resid200_240k --manifest experiments/vorts_90_all_train.json --steps 240000 --max-anchors 150000 --block 128 --densify-grad-threshold 3e-4 --grow-residual 200 ;;
  S1) run S1_block128_scoreprune_240k --manifest experiments/vorts_90_all_train.json --steps 240000 --max-anchors 150000 --block 128 --densify-grad-threshold 3e-4 --prune-score-ratio 0.3 --prune-score-at 40000,80000 ;;
  # Normalized composition on the P3 recipe (block 128, thr 1e-4, 240k steps, cap 150k); single variable vs P3; W2 also --init data.
  W1) run W1_P3_normalized_240k --manifest experiments/vorts_90_all_train.json --steps 240000 --max-anchors 150000 --block 128 --densify-grad-threshold 1e-4 --composition normalized ;;
  W2) run W2_P3_normalized_initdata_240k --manifest experiments/vorts_90_all_train.json --steps 240000 --max-anchors 150000 --block 128 --densify-grad-threshold 1e-4 --composition normalized --init data ;;
  # Single-variable follow-ups of W2 (the best normalized Scaffold4D, 54.94 dB, 39.8 MB): W3 halves the steps (training time),
  # W4 caps anchors at 60k (W2 stopped growing at 119,842 below its 150k cap, so only a lower cap changes the size).
  W3) run W3_W2_steps120k --manifest experiments/vorts_90_all_train.json --steps 120000 --max-anchors 150000 --block 128 --densify-grad-threshold 1e-4 --composition normalized --init data ;;
  W4) run W4_W2_cap60k --manifest experiments/vorts_90_all_train.json --steps 240000 --max-anchors 60000 --block 128 --densify-grad-threshold 1e-4 --composition normalized --init data ;;
  NA) G5="--manifest experiments/vorts_90_all_train.json --steps 120000 --max-anchors 150000 --eval-every 10000"
      run N1_feat_x3_120k   $G5 --feat-lr 0.0225
      run N3_offset_x3_120k $G5 --offset-lr-init 0.03 --offset-lr-final 0.0003
      run N5_mlp_x3_120k    $G5 --shape-mlp-lr 0.012 --flow-mlp-lr 0.003 --opacity-mlp-lr-init 0.006 --opacity-mlp-lr-final 0.00006 ;;
  NB) G5="--manifest experiments/vorts_90_all_train.json --steps 120000 --max-anchors 150000 --eval-every 10000"
      run N2_feat_x0.3_120k   $G5 --feat-lr 0.00225
      run N4_offset_x0.3_120k $G5 --offset-lr-init 0.003 --offset-lr-final 0.00003
      run N6_mlp_x0.3_120k    $G5 --shape-mlp-lr 0.0012 --flow-mlp-lr 0.0003 --opacity-mlp-lr-init 0.0006 --opacity-mlp-lr-final 0.000006 ;;
  EB) for g in 2 4 6 8; do group $g; done ;;
esac
