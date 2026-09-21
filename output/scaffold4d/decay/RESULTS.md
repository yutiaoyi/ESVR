# ESVR-Scaffold4D — learning-rate decay: the largest single lever (2026-09-20/21)

> **Headline**: decaying the four natively constant learning-rate groups
> (feature, offset scale, shape-MLP, flow-MLP) log-linearly to 1/100 over
> training takes the 150k-anchor / 120k-step model from **32.99 → 38.04 dB**
> with no other change (G5 vs G1), and with 480k steps to **39.43 dB** (H4)
> — level with per-frame static ESVR (39.38 dB, 90 MB) at 49.8 MB, or
> **7.2 MB after post-training compaction (39.42 dB)**. Now the default
> (`--final-lr-scale 0.01`).

All runs: fixed voxelizer through `.venv/bin/python`, native 4D-Scaffold-GS
recipe with the ESVR adaptations documented in `docs/ESVR_SCAFFOLD4D.md`,
K = 10, growing window = first half of training, seed 42 unless noted, all
90 frames trained (no held-out split). Every PSNR below is recomputed from
the run's complete 90-frame `evaluation/metrics.json`.

## Why

`scale/G1_K10_150k_120k` (no decay) plateaued from step 60k and its
monitored PSNR swung ±0.7 dB between consecutive checkpoints; 33 → 50 MB of
anchor budget had added only 0.04 dB. Upstream 4D-Scaffold-GS keeps
`feature_lr`, `scaling_lr`, `mlp_cov_lr` and `mlp_flow_lr` constant (only
`offset` and `mlp_opacity` decay); that works under alpha compositing with
whole-frame multi-camera supervision, but ESVR's additive accumulation plus
a single random 64³ block per step is a noisier gradient, and a constant
late learning rate keeps the model bouncing around the optimum. ESVR's own
position and intensity learning rates decay. `--final-lr-scale s` applies
`expon_lr(step, lr, lr·s, steps)` to those four groups.

## Results (each run changes one thing versus G5)

| Run | Change vs G5 | Anchors | Size | Mean PSNR | Min / max frame | Compact (fp16 + prune) |
|---|---|---:|---:|---:|---|---|
| `scale/G1_K10_150k_120k` | no decay (native) | 150,000 | 49.8 MB | 32.991 | 29.91 / 34.64 | — |
| **`scale/G5_K10_150k_120k_decay`** | reference: decay 1/100, 120k steps | 142,303 | 47.3 MB | **38.036** | 34.99 / 38.93 | 7.2 MB, 38.03 |
| `H1_40k_decay0.01` | anchor cap 40k | 40,000 | 13.3 MB | 35.464 | 32.62 / 36.17 | 3.1 MB, 35.46 |
| `H2_150k_decay0.1` | decay 1/10 | 150,000 | 49.8 MB | 38.403 | 35.86 / 39.06 | 7.7 MB, 38.40 |
| `H3_150k_decay0.001` | decay 1/1000 | 150,000 | 49.8 MB | 37.849 | 34.88 / 38.80 | 9.6 MB, 37.83 |
| **`H4_150k_decay0.01_480k`** | **480k steps** | 150,000 | 49.8 MB | **39.431** | 37.05 / **40.30** | **7.2 MB, 39.42** |
| `H5_300k_decay0.01` | anchor cap 300k | 300,000 | 64.8 MB | 38.148 | 35.09 / 39.17 | 8.7 MB, 38.13 |
| `H6_150k_decay0.01_s1` | seed 1 | 150,000 | 49.8 MB | 38.453 | 35.70 / 39.62 | 10.2 MB, 38.44 |

Seed spread (G5 vs H6): 0.42 dB. Differences below ~0.5 dB (H2, H3) are
not resolved; any decay factor between 1/10 and 1/1000 works.

## Reading

- **Decay**: +5.05 dB at 150k anchors (G5 vs G1), +3.6 dB at 40k anchors
  (H1 vs `fixed_kernel/F_default` 31.84 ± 0.29). Monitored curves become
  monotonic (G5: 20k 29.7 → 60k 35.8 → 100k 37.2 → 120k 37.2).
- **Steps**: ×4 gives +1.4 dB (H4 vs G5); the training budget, not the
  anchor budget, is what limits the model now.
- **Budget**: 150k → 300k anchors adds nothing (H5 vs G5), consistent with
  G1 vs `fixed_kernel/F_cap100k`.
- **Useful capacity**: G5 has ~22k useful children per query (intensity
  > 1/255 and off the scale floor), per-frame static ESVR ~24k primitives
  per frame; H3 reaches ~46k useful per query without gaining PSNR.
- H4 verified: `--eval-only --resume` reproduces all 90 frames with max
  abs diff 0.0.

## Post-training compaction (`experiments/compact_scaffold4d.py`)

Reads `checkpoint.pt`, never writes to it; outputs go to `RUN/compact/`.
Drops children with intensity < 1/255 (the kernel never draws them: exact,
0.000 dB on G5), optionally children pinned on the 0.5-voxel scale floor
(−0.01 dB), and anchors left without children; stores kept offsets only
plus a 1-bit-per-child mask; optionally fp16 (−0.003 dB). G5: 47.3 → 33.9
(exact) → 17.0 (fp16) → 7.2 MB (also floor-pinned). Roughly 90 % of the
trained model's bytes belong to children the optimiser has switched off —
storage waste, not a PSNR limit (see `screen_fixed/RESULTS.md`).

## Next (`experiments/run_scaffold4d_40db.sh`, running)

Single-variable moves from H4 toward 40 dB: decay 1/10 at 480k steps (I1),
960k steps (I2), 300k anchors at 480k steps (I3).

## Not done

Single seed per configuration except G5/H6; no held-out frames; no
quantisation/entropy coding beyond fp16.
