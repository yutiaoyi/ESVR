# Deformable-MLP (dynamic_esvr) — matched-configuration comparison run

> **What this run is, precisely**: an *architecture-isolation* experiment.
> Initialization and LR are deliberately matched to ESVR4D's own settings
> (random init, 30,000 primitives) rather than deformable-MLP's own PLY-
> warm-start recipe. An earlier version of this note pointed to a PLY-warm-
> started rerun as a complementary "own best practice" comparison; that run
> was removed (2026-09-12) because warm-starting from a point cloud that was
> itself already optimized on frame 1 gives deformable-MLP an initialization
> advantage ESVR4D has no equivalent way to use, confounding "which motion
> representation is better" with "which method got a head start." The
> cleaner single-variable comparison — this run's exact configuration with
> *only* the LR changed to deformable-MLP's own native scale (5e-5/1.6e-4),
> initialization and primitive count still random/30,000 as here — is at
> [output/dynamic/vorts90_matched_native_lr/RESULTS.md](../vorts90_matched_native_lr/RESULTS.md).
>
> **2026-09-13**: that "native LR" comparison used one shared 5e-5 value for
> all four canonical parameters, itself a simplification — natively
> position/scale/rotation/opacity are four separate LRs up to ~1000x apart
> (canonical position/scale/rotation/opacity conflated into one group was a
> real bug, not just a coarser approximation). The real fix, keeping the
> deformable-MLP recipe's own initialization (random, matching this run) but
> with all four groups properly separated, reached **28.90 dB** — see
> [output/dynamic/vorts90_native_canonical_lr/RESULTS.md](../vorts90_native_canonical_lr/RESULTS.md),
> currently the best real-90-frame result of any run in this repository.

Companion to [output/esvr4d/vorts90_fixed_points/RESULTS.md](../../esvr4d/vorts90_fixed_points/RESULTS.md).
Same 90-frame `vorts` manifest, same CUDA-fixed copy of the repository, and
configuration deliberately matched to that ESVR4D run so the two motion
representations can be compared with fewer confounding variables than the
original `output/dynamic/vorts90_full_geometry/RESULTS.md` (which warm-started
from a pretrained static PLY and used 21,401 primitives).

**Deliberately NOT matched to the original historical run**: no `--init-ply`
(random init, like ESVR4D — which has no PLY-init option at all), 30,000
canonical primitives (was 21,401), same LR scale (5e-3/2e-3 vs. the original's
5e-5/1.6e-4 fine-tuning rates, appropriate for training from scratch instead
of fine-tuning).

Completed 2026-09-12. Branch: `codex/4dgs-esvr`.

## Configuration

| Item | Value |
|---|---|
| Initialization | Random (no `--init-ply`) |
| Canonical primitives | 30,000, fixed (this method has no densification either) |
| Deformation network | 8 layers, width 256, 10 Fourier bands each for XYZ/time (defaults) |
| Static warmup | 1,000 steps before deformation activates |
| Training steps | 40,000 |
| Spatial sampling | 64³ non-overlapping tiles, `--tile-sampling` |
| LR | 5e-3 canonical, 2e-3 deformation network; exponential decay to 10% |
| Loss | 0.8·L1 + 0.2·(1 − SSIM) + 1e-4·squared deformation penalty |
| Seed | 42 |
| Device | RTX 5090; same CUDA-fixed `gaussian_rasterization_voxelization` build as the ESVR4D run |

Wall time: 1023.2 s (~17.1 min) for 40,000 steps — noticeably slower than
ESVR4D's 178.9 s at the same primitive count and step count, because this
method evaluates an 8-layer MLP forward+backward every step in addition to
the voxelizer, where ESVR4D's motion is a closed-form function of the
covariance with no network.

## Final full-volume metrics (all 90 training frames)

| Metric | This run (deformable-MLP) | ESVR4D (same config) |
|---|---:|---:|
| Per-frame PSNR, arithmetic mean | 25.958 dB | 28.772 dB |
| Per-frame PSNR, median | 26.049 dB | 28.995 dB |
| Lowest / highest per-frame PSNR | 23.384 / 26.828 dB | 25.398 / 30.718 dB |
| Frame 1 / 46 / 90 PSNR | 25.646 / 26.074 / 23.384 dB | 28.582 / 29.300 / 25.398 dB |
| Mean normalized MAE | 0.03825 | 0.02743 |
| Training wall time | 1023.2 s | 178.9 s |

Under this matched configuration, ESVR4D reaches ~2.8 dB higher mean PSNR
and trains ~5.7x faster (fewer FLOPs per step: no MLP evaluation). This is
**still a training-set reconstruction comparison, not a generalization
comparison** — all 90 frames were used for training in both runs (see
"Not done" below) — and both runs share the same limitations already noted
for ESVR4D (from-scratch init only, no PLY warm start available for either
run's config, single seed, no densification for either method).

## Verification

- Training exited cleanly (code 0) at step 40,000.
- `--eval-only --resume checkpoint.pt` into a fresh directory reproduced all
  90 exported `.npy` frames with **max abs diff 0.0**.
- `experiments/summarize_dynamic_run.py` ran clean: all per-frame artifacts
  present (npy/raw/ply/3-axis png), `quality_curves.png`,
  `slice_comparison.png`, `training_frames_dvr.gif`, `summary.json`,
  `data_sha256.json`, `provenance.json` generated.

## Not done

- **No held-out time split** — this run and the ESVR4D run it's compared
  against both use `experiments/vorts_90_all_train.json`, which marks all 90
  frames `"split": "train"`. Reported PSNR measures how well each fixed-
  capacity representation fits 90 *known* volumes, not whether it can
  interpolate an unseen time. The train/test split mechanism in
  `dynamic_esvr/data.py` (`VolumeSequence.train_indices`, per-frame `split`
  field propagated into `metrics.json`) already works — validated on the
  synthetic 5-frame smoke test (`experiments/make_dynamic_synthetic.py`,
  t=0.25/0.75 held out) — but has not been exercised on the real 90-frame
  sequence for either method. A genuine held-out comparison on real data is
  a natural next step, not yet run.
- Single seed (42) for both methods; no variance estimate.
- No 2D/DVR or LPIPS-style supplemental metrics for either matched run yet
  (only the 3D-volume PSNR/MAE/RMSE both training scripts compute directly).
