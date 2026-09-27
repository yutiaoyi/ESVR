# Per-frame normalized ESVR (fairness baseline, 2026-09-27)

`experiments/run_per_frame_normalized.sh`: the single-frame normalized recipe
of `experiments/static_normalized.py` (data-sampled init, L1, cosine LR 8e-3,
3,000 whole-volume steps, relocation on) run independently on each of the 90
vorts frames. It separates "normalized composition" from "4D sharing".
12 fp32 floats per primitive (centre 3, log-scale 3, quaternion 4, value 1,
weight 1). The script does not save the models; PSNR is computed on the final
forward pass of each frame against the ground truth (float64 MSE).

| Run | Primitives / frame | Mean PSNR (min / max) | fp32 | fp16 | Time (concurrent) |
|---|---:|---:|---:|---:|---|
| PN21 | 21,000 | **51.045** (49.72 / 52.65) | 90.7 MB | 45.4 MB | 67 s / frame, 1.7 h |
| PN10 | 10,000 | **43.703** (42.14 / 45.18) | 43.2 MB | 21.6 MB | 55 s / frame, 1.4 h |
| PN6 | 6,300 | queued | 27.2 MB | | |

Comparisons over the same 90 frames:

| Pair | What differs | Result |
|---|---|---|
| PN21 vs S1b (additive per-frame ESVR, 93.8 MB, 41.21) | composition, plus init / loss / optimiser | +9.8 dB at equal size. Init alone is not the reason: on frame 1, ESVR's own recipe with data init moves by +0.17 (S1b) / -0.96 (S5b), and normalization gains +14.2 dB under ESVR's random init (`output/experiments/normalized`). |
| PN21 vs S5b (additive per-frame ceiling, 555.6 MB, 49.52) | as above, 6x smaller | +1.5 dB |
| W2 (normalized Scaffold4D, 39.8 MB, 54.94) vs PN21 | 4D sharing across frames (and W2 has SSIM in its loss, 2,670 vs 3,000 passes per frame) | **+3.89 dB on average, better on 90 / 90 frames, at 44 % of the size** |
| W2 (39.8 MB) vs PN10 (43.2 MB) | 4D sharing at equal size | **+11.24 dB, better on 90 / 90 frames** |
| N4A (normalized explicit 4D, 27.2 MB, 53.25) vs PN21 | 4D sharing only (same init family, loss, optimiser) | +2.2 dB at 30 % of the size |

So both effects are real and they add: normalization is the large one
(~10-14 dB), and sharing primitives across time is worth a further ~2-4 dB at
a fraction of the size once the composition is normalized. (With additive
composition the earlier unshackle test found the opposite sign on one frame:
time sharing cost ~4 dB.)
