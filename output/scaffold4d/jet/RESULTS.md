# ESVR-Scaffold4D on jet_Y_OH (2026-09-26)

Dataset: S3D combustion jet, OH mass fraction. 122 frames of 480x720x120
float32 (`.dat`, little-endian, storage order z-y-x with x fastest), values in
[0, 0.002351] (global range from `datasets/jet_Y_OH.txt`, per-frame maxima
match the files one-to-one; no duplicate frames). Manifest
`experiments/jet_Y_OH_122_all_train.json` maps the range to [0, 1]; all 122
frames are training data (20 GB in RAM); PSNR is the 3D voxel PSNR on the
same frames, peak 1. Loader needs no format change (any suffix is read as
RAW when the byte count matches).

Recipe: the best vorts recipe (`next/P1b`: block 128^3, growth threshold
3e-4, decay 1/100, tile sampling) scaled for a 20x larger volume: anchor cap
400k, 40k initial anchors, 480k steps. A 128^3 tile is 1/24 of a frame here
(24 tiles per frame), so supervision per voxel is far below vorts.
Script: `experiments/run_jet_scaffold4d.sh {smoke,J1}`.

| Run | Setting | Mean PSNR | Min / max | Anchors | Trained / compact | Time |
|---|---|---:|---|---:|---|---|
| JET_smoke_2k | 2,000 steps, 20k anchors (pipeline check) | 16.883 | 10.26 / 21.27 | 20,000 | — | 4.5 min incl. evaluation |
| **J1_block128_thr3e-4_480k_cap400k** | full recipe | **42.808** | 39.49 / 47.94 | 400,000 (cap filled by step 10k) | 132.8 / **21.9 MB** (42.787) | 2.14 h alone on the GPU |

## J1 notes

- Compression: raw 20.24 GB -> 132.8 MB trained (152x) -> 21.9 MB compact
  (924x, fp16 + dark/floor children dropped, -0.02 dB).
- PSNR falls along the sequence as the jet develops: frames 1-10 46.5 dB,
  56-66 43.0, 112-122 40.9. Per-decile means: 46.2, 43.6, 43.8, 43.6, 43.2,
  42.7, 41.9, 41.4, 41.2, 41.0, 40.2.
- The 400k cap fills within 10k steps; afterwards 938k anchors are grown and
  578k pruned (churn). Useful children per query 34.5k, useful total 339k of
  4.0M (8.5 %), median temporal half-width 4.1 frames.
- The monitor curve is still rising at the end (400k 43.04 -> 440k 43.07 ->
  480k 43.43): under-trained and cap-limited. Obvious follow-ups: larger cap
  (800k), more steps, and a per-frame static ESVR baseline on jet for the
  same comparison as on vorts.
