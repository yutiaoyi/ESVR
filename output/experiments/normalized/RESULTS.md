# Single frame: additive ESVR sum vs normalized kernel regression (2026-09-26)

`experiments/static_normalized.py` (`run_static_normalized.sh`). vorts frame 1
unless noted, whole-volume L1 per step, 3,000 steps, Adam with cosine decay,
identical data-sampled initialisation (50 % importance ~ (value+|grad|)^0.6,
35 % uniform, 15 % 2^3-bin peaks). Only the composition differs:

- additive: V = sum a_i K_i (ESVR)
- normalized: V = sum w_i K_i s_i / max(sum w_i K_i, 0.05) (Martin et al. 2026,
  arXiv 2609.16024; Image-GS), computed with two calls of the unchanged
  voxelizer (it is linear in the amplitude).

| Primitives | Additive | Normalized | Size (normalized, fp32) |
|---:|---:|---:|---:|
| 10,000 | — | 45.30 | 0.48 MB |
| 21,000 | (S1b, ESVR's own recipe: 41.81) | **52.62** | 1.01 MB |
| 32,000 | 42.84 | **55.75** | 1.54 MB |
| 32,000, no relocation | — | 55.73 | 1.54 MB |
| 32,000, frame 45 | — | 53.56 | 1.54 MB |
| 64,000 | — | 58.76 | 3.07 MB |
| 139,000 | 45.90 (S5b, ESVR's own recipe: 49.63) | — | — |

Normalization is worth +12.9 dB at equal count and settings. With 21k
primitives (ESVR's default budget) it exceeds 50 dB; the tuned additive
ceiling needs 139k primitives for 49.6. Relocation of low-support primitives
adds nothing here. Reason: vorts is dense (90 % of voxels carry a non-zero
baseline); an additive sum must tile that with overlapping positive bumps,
which ripples, while a weighted average represents a plateau with a few
primitives that report the same value.

## Composition x initialisation, frame 1 (fairness control, 2026-09-27)

ESVR initialises uniformly at random (`create_random_onthefly`: centres
uniform, value 0.1, scale = nearest-neighbour distance). The runs above used
data-sampled centres. `--init random` reproduces ESVR's init; everything else
is unchanged (32,000 primitives, 3,000 whole-volume steps, L1, cosine LR 8e-3,
relocation on).

| Init | Additive | Normalized | Normalization gain |
|---|---:|---:|---:|
| ESVR random | 39.93 | **54.13** | +14.19 |
| Data-sampled | 42.84 | **55.75** | +12.91 |
| Init gain | +2.91 | +1.62 | |

The normalization gain holds under ESVR's own init. With this fixed-count
optimiser the data-sampled init is worth +1.6 dB (normalized) to +2.9 dB
(additive). With ESVR's own recipe (densify, prune, opacity reset) the same
data-sampled init is worth +0.17 dB on frame 1 for S1b (S1d 41.98 vs 41.81)
and -0.96 dB for S5b (S5d 48.67 vs 49.63; `output/experiments/per_frame_fixed`),
so the per-frame ESVR baseline is not held back by its random init.
