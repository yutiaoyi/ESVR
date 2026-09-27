# ESVR-Scaffold4D — second round of probes (2026-09-22)

Fixed voxelizer (every run wrote `voxelizer.json`), `--final-lr-scale 0.01`,
K = 10, seed 42, `--tile-sampling`. Script `experiments/run_scaffold4d_next.sh`
(lanes L1, EA/EB, M1, NA/NB). Baselines: `decay/H4_150k_decay0.01_480k`
39.431 dB (150k anchors, 480k steps) and `scale/G5_K10_150k_120k_decay`
38.036 dB (same at 120k steps). One change per run.

## Motivating diagnostic (`experiments/diagnose_scaffold4d.py`, H4 / I3 / K1)

Median temporal half-width of *useful* children (intensity > 1/255, above
the scale floor): H4 3.5 frames, I3 4.0, K1 (10 frames, 480k steps) 0.9. The
model localises children to a few frames on its own; sharing across the 90
frames happens at the anchor-feature / MLP level, not at the child level.
37 % (H4) / 45 % (I3) of *all* children sat on the σ = 100 clamp — L1 below
shows those were the dead children, not the useful ones.

## Runs

| Run | Change | Baseline | Mean PSNR | Δ | Trained / compact |
|---|---|---|---:|---:|---|
| L1_sigma1000_480k | `--sigma-max 1000` | H4 39.43 | 38.962 | −0.47 | 49.8 / 6.0 MB |
| E1–E9 (9 × 10 frames) | 53,333 steps + 16,667-anchor cap each (sum = H4) | H4 39.43 | **33.307** | −6.12 | 50.1 / 16.8 MB (10.3 with `--drop-never-selected`) |
| M1_errsample_480k | `--error-sampling 1.0` (25 % uniform) | H4 39.43 | 39.151 | −0.28 | 49.8 / 6.5 MB |
| N1_feat_x3_120k | feature LR 0.0075 → 0.0225 | G5 38.04 | 38.079 | +0.04 | 49.8 / 7.3 MB |
| N2_feat_x0.3_120k | feature LR → 0.00225 | G5 38.04 | **38.507** | **+0.47** | 49.8 / 9.3 MB |
| N3_offset_x3_120k | offset LR 0.01/1e-4 → 0.03/3e-4 | G5 38.04 | 38.341 | +0.30 | 49.8 / 7.4 MB |
| N4_offset_x0.3_120k | offset LR → 0.003/3e-5 | G5 38.04 | 36.927 | **−1.11** | 49.8 / 7.2 MB |
| N5_mlp_x3_120k | shape/flow/opacity MLP LR ×3 | G5 38.04 | 38.324 | +0.28 | 49.8 / 8.2 MB |
| N6_mlp_x0.3_120k | shape/flow/opacity MLP LR ×0.3 | G5 38.04 | 37.504 | **-0.53** | 49.8 / 6.9 MB |
| O2_offset_x10_120k | offset LR 0.01/1e-4 → 0.1/1e-3 | G5 38.04 | 35.778 | **−2.26** | 49.8 / 4.4 MB |
| O1_lr_combo_480k | feature ×0.3 + offset ×3 + MLP ×3 together, 480k steps | H4 39.43 | 38.415 | **−1.02** | 49.8 / 4.8 MB |

Per group (E, 10-frame means; H4 on the same frames in brackets): E1 32.09
(39.82), E2 34.91 (39.52), E3 33.81 (39.09), E4 34.76 (39.56), E5 34.83
(39.43), E6 33.12 (39.34), E7 35.85 (39.47), E8 32.30 (39.23), E9 28.09
(39.42). Total 8,023 s vs one H4 run. K1 (same frames as E5, 480k steps)
was 41.20: its +1.8 dB over H4 is the 9× per-frame compute, not the grouping.

## What each probe says

- **L1 (σ clamp)**: only 0.1 % of children reach the new clamp and the
  useful children's half-width distribution is unchanged (p5/50/95 =
  1.0/3.56/10.9 frames vs 1.15/3.48/10.6 for H4). The clamp was binding for
  dark / floor-pinned children only. Not a bottleneck; −0.47 is at the edge
  of seed noise (0.42).
- **E (grouped fits at equal total compute)**: −6.1 dB. One 90-frame model
  beats nine 10-frame models trained for 1/9 each by a wide margin: the
  shared anchor features and MLPs let an update on one frame help the
  others. At 53k steps the temporal gates are still wide (median half-width
  12–30 frames; E9 117), so gates narrow late in training — σ is learned
  slowly. 358k of 893k useful children are parked at times outside their
  group's window (random-t init) and can be removed exactly with
  `--drop-never-selected` (16.8 → 10.3 MB).
- **M1 (error-proportional sampling)**: −0.28 mean, 87 of 90 frames below
  H4, per-frame std 0.42 → 0.34, worst frame 37.05 → 37.26. The final error
  table spans only a 2× MSE ratio between the easiest and hardest of the
  720 (frame, tile) cells — there are no hard blocks to exploit, and the
  lost uniform coverage costs a little. Frame sampling counts 4,697–7,151
  (uniform 5,333). First attempt died of CUDA OOM at 5k steps with four
  concurrent runs; rerun alone.
- **N1–N6 (base LRs, all upstream values until now)**: offset LR is clearly
  too low (×0.3 −1.11, ×3 +0.30); feature LR slightly too high (×0.3
  +0.47, ×3 0.0); MLP LR ×3 +0.28, ×0.3 −0.53 (so MLP LR also wants to go up); offset ×10 (O2) −2.26, so the offset optimum is near ×3. Single effects are 0.3–0.5 dB, at the
  noise level, but consistent in direction. N2 keeps more children alive
  (compact 9.3 MB vs 7–8).

## Scale-floor gradient trap (Q1) and block size (P1 / P1b)

| Run | Change | Baseline | Mean PSNR | Δ | Trained / compact |
|---|---|---|---:|---:|---|
| Q1_floor_st_480k | H4 recipe on the code with the straight-through scale floor (commit aae731c) | H4 39.43 | 39.094 | −0.34 | 49.8 / 6.0 MB |
| P1_block128_240k | block 128³, 240k steps (equal wall-clock to H4 by the step profile: 51 vs 30 ms) | H4 39.43 | stopped at 35k steps | — | growth stalled at 14k anchors |
| **P1b_block128_thr3e-4_240k** | P1 with growth threshold 1.2e-3 → 3e-4 | H4 39.43 | **40.171** | **+0.74** | **29.0 / 6.8 MB**, 0.75 h, 87,325 anchors |
| **P3_block128_thr1e-4_240k** | P1b with growth threshold 1e-4 (cap 150k filled) | P1b 40.17 | **40.585** | **+0.41** | 49.8 / 10.5 MB, 3.3 h (3 concurrent runs), 150,000 anchors, useful/query 26.8k |
| P2_block128_thr3e-4_480k | P1b at 480k steps | P1b 40.17 | 40.498 | +0.33 | 34.4 / 7.7 MB; 103k anchors; 5.0 h with 4 concurrent runs |
| R1_block128_resid200_240k | P1b + `--grow-residual 200` | P1b 40.17 | 40.257 | +0.09 | 49.8 / 8.7 MB; 150k anchors (cap filled by 60k steps, 76k of 155k newborns from residual candidates); below P3 at equal anchors |
| S1_block128_scoreprune_240k | P1b + `--prune-score-ratio 0.3 --prune-score-at 40000,80000` | P1b 40.17 | 39.869 | −0.30 | **17.9** / 5.65 MB; 53.7k anchors (24.6k + 20.4k pruned, only 12 + 5 regrown) |

Q1: the third external review found that `clamp_min` gave zero gradient to
the 90.9 % of H4's children whose pre-clamp size sits below the 0.5-voxel
floor (52 % of anchors entirely; median pre-clamp size 1e-11 of the floor).
With the gradient passed through, the collapse gets *deeper* (91.8 % / 61 %
/ 1e-26) and PSNR is unchanged: the optimizer actively shrinks these children
(the fastest way to remove additive over-prediction); the floor merely keeps
them as 0.5-voxel points (75 % still above 1/255 in intensity; dropping them
at compaction costs 0.01 dB). The model is effectively ~1.3 M sub-voxel
points plus 8 % real ellipsoids. The fix stays (correctness) but is not a
lever; the lever is removing those children during training and re-using
their capacity (importance pruning + residual-driven growth/relocation).

P1: with whole-volume steps the growth threshold calibrated on 64³-block
gradients no longer fires (smoother gradient estimates, thinner tail): 14k
anchors at 35k steps vs 150k for H4. The same confound applies to `fifty/K2`
(12k anchors). Step profile on an idle GPU (H4 checkpoint): generate+at
13.6 ms, voxelize forward 2.9 ms, loss+backward 11.5 ms, optimizer 2.2 ms at
64³ (30 ms/step); 51 ms at 128³. Raising the query cutoff to the kernel's
1/255 threshold leaves outputs bit-identical, sends 27 % fewer children to
the kernel and saves only 3 %: the step cost is the fixed overhead of the
1.5 M-child population, not voxelization.

P1b: 40.171 dB (min 37.73 / max 41.39) in 2,713 s alone on the GPU, 87k
anchors (threshold-limited, cap not reached), 29.0 MB trained / 6.8 MB
compact (40.152), useful children per query 21.6k (H4 16.7k), useful total
150k (H4 129k). Better than H4 and I3 on PSNR, pre-compaction size and
training time at once. Whole-volume supervision costs almost nothing per
step (the step is dominated by the fixed 1.5 M-child overhead), so it gives
8x the supervised voxels per step for free; the only trap is the growth
threshold, which must be re-calibrated for the smoother whole-volume
gradients. `fifty/K2`'s "whole-volume is worse" conclusion is withdrawn
(equal-voxel-count control and an uncalibrated threshold).

Summary of the block-128 variants: more steps +0.33 (P2), lower threshold
+0.41 (P3, cap filled), residual growing +0.09 (R1, cap filled, below P3 at
equal anchors), score pruning −0.30 but 17.9 MB trained (S1: the pruned
regions did not regrow because their gradients stay below the threshold,
so ESVR's prune-then-redensify gain did not materialise). Nothing buys PSNR
at equal size; the size/PSNR trade is monotone. Best PSNR: P3 40.59; best
pre-compaction size: S1 39.87 at 17.9 MB.

## Verification (O1 / O2)

O2 (offset ×10) −2.26: overshoot, the offset optimum is near ×3. O1 (the
three positive directions combined, 480k steps) −1.02, and its monitor
curve sits ~1 dB below H4 from step 20k on (25.7 vs 28.4 at 20k, 36.9 vs
38.0 at 240k, 37.8 vs 38.9 at 480k): the 120k-step single-variable gains
neither add up nor transfer to the 480k recipe. The upstream learning
rates stay; the LR line is closed.

## Next

P2 = P1b at 480k steps, P3 = P1b with threshold 1e-4 (more anchors); then implement in-training importance pruning (ESVR score: voxel coverage × intensity × fractional anisotropy) with relocation of freed slots to high-residual voxels; that is the common root of size, step time and dead capacity.

## Not done

Single seed throughout; the 120k-step LR results have not been confirmed at
480k; no held-out frames (by decision: this is data fitting of all 90 frames).

## Normalized composition on the P3 recipe (W1 / W2, 2026-09-27)

Same recipe as P3 (block 128, growth threshold 1e-4, 240k steps, cap 150k,
0.8 L1 + 0.2 SSIM + 0.01 volume regulariser). W1 changes one thing vs P3:
`--composition normalized` (V = Σ w g K s / max(Σ w g K, 0.05)). W2 changes
one thing vs W1: `--init data` (anchors sampled from the data instead of
uniformly at random).

| Run | Change | Baseline | Mean PSNR (min / max) | Δ | fp32 | fp16 | fp16 + drop floor | Anchors | Train time |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| W1_P3_normalized_240k | normalized composition | P3 40.585 | **54.663** (50.71 / 56.46) | **+14.08** | 49.8 MB | 25.1 MB (−0.147 dB) | 21.5 MB (−0.164 dB) | 150,000 (cap) | 15.6 h, 2–10 concurrent jobs |
| W2_P3_normalized_initdata_240k | + data-sampled anchors | W1 54.663 | **54.939** (50.92 / 56.99) | +0.28 | 39.8 MB | 20.1 MB (−0.161 dB) | 19.0 MB (−0.163 dB) | 119,842 (growth stopped below cap) | 15.0 h, 2–10 concurrent jobs |

Verification:
- The saved predictions (`evaluation/frame_*.npy`) against the raw data reproduce the stored per-frame PSNR to 1.4e-14 dB; no value outside [0, 1].
- The reloaded compact file, evaluated in 64³ blocks instead of the training 128³ block, gives the same mean PSNR (Δ 0.000 dB for both runs).
- Compaction bug fixed first (b7ea10a): under normalized composition the never-drawn test is w < (1/255) / norm_C, not w < 1/255.

What it says:
- Normalization, not the 4D structure, is the jump: +14.1 dB at identical size, recipe and random init (W1 vs P3).
- Almost every child is used: 99.4 % (W1) and 99.99 % (W2) survive the exact dark test, vs 12 % for P3. So compaction no longer shrinks the model much (P3 10.5 MB compact; W2 19.0 MB).
- fp16 costs 0.15–0.16 dB here, vs 0.03 dB for P3 and N4A: the ratio amplifies small errors in weights and positions. Mixed precision (positions in fp32) is untested.
- Data-sampled init: +0.28 dB with 20 % fewer anchors. Growth ends at step 120k (densify-until = steps/2); both curves are nearly flat after 140k (W2: 54.46 at 140k, 54.99 at 240k on the 3 monitor frames).

Fairness (see also the table in docs): ESVR (S1b/S5b) uses random init, W1 does too (except that the value head starts at the global data mean); W2 and N4A use the data. S1b/S5b training times were measured alone, W1/W2 under concurrency. The per-frame normalized ESVR baseline is in output/experiments/per_frame_normalized (PN21 51.05 dB at 90.7 MB, PN10 43.70 at 43.2 MB; W2 is better on 90/90 frames against both).

### W3: W2 with half the steps (2026-09-27)

| Run | Change | Baseline | Mean PSNR (min / max) | Δ | fp32 | fp16 | fp16 + drop floor | Anchors | Train time |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| W3_W2_steps120k | `--steps 120000` (LR decay, growth window and prune window all scale with it) | W2 54.939 | **52.983** (48.18 / 54.90) | −1.96, worse on 90/90 frames | 38.3 MB | 19.3 MB (−0.101 dB) | 18.3 MB (−0.105 dB) | 115,138 | 6.1 h, concurrent (W2 15.0 h) |

Verified as W1/W2 (saved predictions reproduce the stored PSNR to 1.4e-14; exact compaction reload Δ 0.000 dB). W3 leads W2 early (50.92 vs 48.18 at 60k steps on the monitor frames, because its LR decays twice as fast) but ends 2 dB lower; W2's gain after 120k (53.13 → 54.99) is the low-LR tail. Halving training time costs ~2 dB; W3 still beats per-frame ESVR's ceiling S5b (49.52, 555.6 MB) and per-frame normalized PN21 (51.05, 90.7 MB). W4 (60k anchor cap) is running.
