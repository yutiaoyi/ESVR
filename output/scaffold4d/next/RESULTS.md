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
  +0.47, ×3 0.0); MLP LR ×3 +0.28, ×0.3 −0.53 (so MLP LR also wants to go up). Single effects are 0.3–0.5 dB, at the
  noise level, but consistent in direction. N2 keeps more children alive
  (compact 9.3 MB vs 7–8).

## Next

O1 = H4 + feature ×0.3 + offset ×3 + MLP ×3 at 480k steps (one run, then
the three-way split only if it moves ≥ 1 dB); offset ×10 probe. Nothing
else in this round is worth a second run.

## Not done

Single seed throughout; the 120k-step LR results have not been confirmed at
480k; no held-out frames (by decision: this is data fitting of all 90 frames).
