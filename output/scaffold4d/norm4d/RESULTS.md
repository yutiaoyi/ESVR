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

## N4W: the N4A representation trained with W2's recipe (2026-09-27, running)

Question: at equal training recipe, is Scaffold4D (anchors + shared MLPs) more compact than
explicit 4D primitives? N4A cannot answer it: it used 90k steps, L1 only, cosine LR, scale floor
0.3 voxel, sigma_max 1000, while W2 used 240k steps, 0.8 L1 + 0.2 SSIM + 0.01 volume term,
log-linear decay to 1 %, floor 0.5 voxel, sigma_max 100. The N4W lanes
(`experiments/run_norm4d.sh W585|W1198|W400`, commit 049b6e5) use W2's recipe for everything
except the representation itself.

| Lane | Primitives | fp32 | Matched to |
|---|---:|---:|---|
| N4W_585k_Wrecipe_240k | 585,000 | 39.8 MB | W2's size |
| N4W_1198k_Wrecipe_240k | 1,198,420 | 81.5 MB | W2's number of children |
| N4W_400k_Wrecipe_240k | 400,000 | 27.2 MB | N4A (recipe effect alone) |

Remaining differences, inherent to the two representations: W2 grows anchors from 6,000 to
119,842 during the first half of training, while the explicit primitives are a fixed population
from the start; W2's children start at the data mean value (decoded by the MLP), the explicit
primitives at the ground-truth value of the voxel they were sampled from (favours explicit).
The Scaffold4D side of the size curve is W4 (60k anchor cap, 19.9 MB), W2 (39.8 MB) and W1 (49.8 MB).
Smoke test: 1.2M primitives peak 0.8 GB, ~0.13 s/step with other jobs on the GPU.
