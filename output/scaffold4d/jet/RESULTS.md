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
| J1_block128_thr3e-4_480k_cap400k | full recipe | pending | | | | |
