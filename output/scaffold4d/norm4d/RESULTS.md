# Normalized explicit 4D primitives on vorts (2026-09-26)

`experiments/train_norm4d.py` (`run_norm4d.sh`). No anchors, no MLP: each
primitive has centre (3), home time (1), log-scale (3), quaternion (4), value
logit (1), weight logit (1), log-sigma (1) and velocity (3) = 17 floats. The
4D-Scaffold-GS time model is kept (gate exp(-|dt sigma|^4) on the weight,
linear motion). Composition V = sum w g K s / max(sum w g K, 0.05), two calls of
the unchanged voxelizer. Initialisation sampled from the data (frame, then
voxel by importance/uniform/peak; value = GT; initial temporal half-width as
listed). 400,000 primitives, 90,000 whole-volume steps (one random frame per
step), Adam lr 8e-3 with cosine decay, L1. All 90 frames trained and evaluated.

| Run | Init half-width | Mean PSNR | Min / max | Active per frame | Size fp32 / fp16 | Time |
|---|---:|---:|---|---:|---:|---|
| **N4A_400k_hw2_90k** | 2 frames | **53.252** | 48.42 / 54.58 | 65,949 | 27.2 / 13.6 MB | 4.7 h (8-10 concurrent GPU jobs) |
| N4B_400k_hw4_90k | 4 frames | 53.067 | 47.51 / 54.43 | 71,795 | 27.2 / 13.6 MB | 4.9 h |

N4A verification: an independent reload reproduces every per-frame PSNR
exactly; an fp16 round-trip of all parameters costs 0.026 dB (53.226 dB at
13.6 MB). Worst frames: 90 (48.42), 21 (48.61), 22, 89, 1 — sequence ends and
the fastest-changing frames.

## Comparison (vorts, 90 frames, 3D PSNR on [0,1])

| Method | PSNR | Size |
|---|---:|---:|
| **Normalized explicit 4D (N4A)** | **53.25** | 27.2 MB fp32 / **13.6 MB fp16** |
| Per-frame ESVR S5b (ceiling recipe) | 49.52 | 555.6 MB fp32 |
| Per-frame ESVR S1b (ESVR's recipe) | 41.21 | 93.8 MB fp32 |
| Explicit additive 4D distilled from P3 (X1) | 40.96 | 11.6 / 5.8 MB |
| Scaffold4D additive P3 (previous best) | 40.59 | 49.8 MB trained / 10.5 MB compact |

The 50 dB goal is met: +3.7 dB over the per-frame ceiling at 1/20 of its
fp32 size (1/40 against its fp32 vs our fp16), and +12.7 dB over the best
additive Scaffold4D. Scaffold4D + normalized (same P3 recipe, only the
composition changed) is running as `next/W1`/`W2`.

Not done yet: training time alone on the GPU; SZ3 / TTHRESH on the same data;
quantisation below fp16; jet_Y_OH.
