# ESVR-Scaffold4D — four probes toward 50 dB (2026-09-22)

Fixed voxelizer (every run wrote `voxelizer.json`), `--final-lr-scale 0.01`
passed explicitly, K = 10, seed 42, `--tile-sampling`. Baseline
`decay/H4_150k_decay0.01_480k`: 150k anchors, 480k steps, 39.431 dB on 90
frames (39.425 on frames 41–50). Each run changes one thing relative to H4.
Script: `experiments/run_scaffold4d_50db.sh {1,2,3,4}`.

| Run | Change vs H4 | Anchors | Trained size | Mean PSNR | Min / max | Compact (fp16 + prune) |
|---|---|---:|---:|---:|---|---|
| K1_10frames_480k | frames 41–50 only, `--grow-t-size 1/89` | 49,289 | 16.4 MB | **41.200** (10 frames) | 39.93 / 41.65 | 3.5 MB, 41.17 |
| K2_block128_60k | 128³ blocks, 60k steps (same compute, 1/8 updates) | 12,152 | 4.1 MB | 33.322 | 30.63 / 34.15 | 1.8 MB, 33.32 |
| K3_grow75_480k | `--densify-until 360000` (grow for 75 %) | 150,000 | 49.8 MB | 38.937 | 36.27 / 39.74 | 6.1 MB, 38.92 |
| K4_cap60k_prune_480k | `--max-anchors 60000 --prune-dead` | 60,000 | 20.0 MB | 38.198 | 35.51 / 39.14 | 4.0 MB, 38.19 |

Useful children per query (intensity > 1/255 and above the scale floor,
averaged over the trained time points): K1 38,943; K2 41,589; K3 13,292;
K4 15,010; H4 16,671; J1 12,153.

## What each probe says

- **K1 (optimisation- or representation-limited?)** 9× more optimisation
  per frame and 2.3× more useful children per query buy +1.8 dB over H4 on
  the same 10 frames. The static single-frame model with the same primitive
  and kernel reaches 44.7 dB with 16k primitives and a comparable per-frame
  budget (S3b) and 49.7 with 139k (S5b). The dynamic model is therefore not
  only under-trained: sharing one anchor feature + MLP + linear motion +
  one temporal gate across frames costs ~3.5 dB even at 10 frames. Growing
  stopped at 49k anchors of the 150k cap (2,384 growth checks; the last
  ones add 0–1 anchors). Side effect of the random-t anchor init on a
  10-frame subset: 301k of 493k children are never selected at any of the
  10 training times, 63k of them "useful" — capacity parked at times the
  run never sees. K1 numbers are not comparable with 90-frame numbers.
- **K2 (whole-volume supervision)** −6.1 dB at equal GPU compute: the
  number of optimizer updates is the bottleneck, not the supervised extent
  per step. Growing also stalled (12k anchors).
- **K3 (grow longer)** −0.49 dB, slightly beyond seed noise (0.42). The cap
  still fills, but anchors added after the learning rate has decayed never
  train: useful total 87.7k vs 129k for H4. The upstream ratio (grow for the
  first half) is right.
- **K4 (small cap + dead-anchor recycling)** 38.20 dB at 20 MB trained /
  4.0 MB compact. 26,190 anchors recycled over training, refilled by growth.
  Lies on the budget curve 40k → 60k → 150k = 36.9 (J1) → 38.2 → 39.4 (H4):
  recycling controls trained size but adds no PSNR (consistent with
  screen_fixed/T2). Best "small before compaction" point so far.

## Conclusion

None of the four moves the 90-frame model past 40 dB. The 50 dB gap has
three parts: the time-shared representation (K1), too few useful Gaussians
per time point (~17–20k vs ~140k needed by the static ceiling), and step
saturation near 480k (long/I2). Untested next candidates: more surviving
children per time point (relax the empty-cell-only growth rule, lower the
threshold) with a K4-style cap + recycling to hold size; more temporal
resolution per anchor (piecewise motion, narrower gates, time-local
anchors); and held-out-frame validation before adding capacity.

## Not done

Single seed; no held-out frames; K1's 10-frame result has no 90-frame
counterpart at the same per-frame budget (that would be 4.3 M steps).
