# Explicit additive 4D primitives distilled from P3 (2026-09-26)

`experiments/finetune_explicit4d.py` converts every useful child of
`next/P3_block128_thr1e-4_240k` (intensity >= 1/255, above the scale floor:
181,934) into an explicit 4D primitive with its own parameters (no anchors,
no MLP; same gate, motion and additive composition) and trains 60,000
whole-volume steps on all 90 frames with ESVR's learning rates.

| Run | Mean PSNR | Min / max | Size fp32 / fp16 |
|---|---:|---|---:|
| X1_fromP3_60k | 40.959 | 39.48 / 42.10 | 11.6 / 5.8 MB |
| X2 (all LRs / 3) | stopped at 40k steps (monitor 40.49), redundant with X1 | | |

+0.37 dB over P3 at about half P3's compacted size. Removing the MLP while
keeping each primitive shared across ~13 frames gains little; the ~4 dB that
freeing a single frame gives (output/experiments/unshackle) comes mainly from
the time sharing, and the far larger lever is the composition
(output/scaffold4d/norm4d).
