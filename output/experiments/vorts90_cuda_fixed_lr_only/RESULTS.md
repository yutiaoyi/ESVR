# 90-frame independent native ESVR — CUDA fix, position LR rescaled (clean comparison)

> **The single-variable, 90-frame version of
> [output/experiments/vorts0001_cuda_fixed_lr_only/RESULTS.md](../vorts0001_cuda_fixed_lr_only/RESULTS.md).**
> Same recipe as `output/esvr_original_per_frame/` (native `train_chunk.py`,
> adaptive densification, per-parameter LR schedule, 90 frames independently
> trained), only two things differ: the CUDA voxelizer is fixed, and
> `position_lr_init`/`position_lr_final` are rescaled ÷4096 to compensate.
> Result: **39.379 dB mean**, vs. **41.218 dB** on the same recipe with the
> unfixed CUDA — a real **−1.839 dB** from the fix, consistent with the
> single-frame diagnostic's −1.83 dB.

Completed 2026-09-13. Branch: `codex/deformable-esvr`. Requires
`dataset_configs.py`'s `vorts_frame_0001`..`vorts_frame_0090` entries, added
to this copy for this run (mirrors the original `ESVR/` working copy's own
entries, byte-identical source data, confirmed via sha256 on frame 1).

## Why this run exists

`output/esvr_original_per_frame/RESULTS.md` (unfixed CUDA) and
`output/esvr_per_frame_baseline/RESULTS.md` (fixed CUDA) are **not** a valid
before/after comparison of the CUDA fix — they use different training
scripts, densification strategies, iteration counts, and LR schedules
simultaneously. `output/experiments/vorts0001_cuda_fixed_lr_only/RESULTS.md`
resolved this for a single frame (39.94 dB vs. the unfixed 41.77 dB, −1.83 dB,
isolating only the CUDA fix with a derived compensating position LR). This
run extends that exact single-variable methodology to all 90 frames, so the
"real, per-frame ceiling" comparison in the summary report can finally be made
on a fair basis.

## Configuration (identical to `output/esvr_original_per_frame/` except position LR)

| Item | `esvr_original_per_frame` (unfixed CUDA) | This run (fixed CUDA) |
|---|---|---|
| CUDA voxelizer | unfixed | **fixed** (`ddelx_dx = 1.0f/dVoxel_x`) |
| `position_lr_init` / `_final` | `8e-6` / `1e-6` | **`1.953125e-9`** / **`2.44140625e-10`** (÷4096) |
| `opacity_lr_init` / `_final` | `7.5e-3` / `5e-4` | unchanged |
| `scaling_lr_init` / `_final` | `5e-5` / `5e-5` | unchanged |
| `rotation_lr_init` / `_final` | `1e-5` / `1e-5` | unchanged |
| `densify_grad_threshold` | `2.5e-7` | unchanged (see
  [vorts0001_cuda_fixed_matched_lr/RESULTS.md](../vorts0001_cuda_fixed_matched_lr/RESULTS.md)
  for why this is *not* rescaled) |
| `init_num`, pruning schedule, iterations (2,800 scans = 22,400) | — | unchanged |
| Frames | 90, fully independent (no shared parameters, no warm start between frames) |

## Results

| Metric | `esvr_original_per_frame` (unfixed CUDA) | This run (fixed CUDA, LR-compensated) |
|---|---:|---:|
| Mean 3D PSNR | **41.218 dB** | **39.379 dB** |
| Median | 41.260 dB | 39.395 dB |
| Min / Max | 40.340 / 41.770 dB | 38.650 / 39.940 dB |
| Mean final primitives | ~21,400 | 24,084 (range 22,391–26,262) |

All 90 frames landed in a narrow 1.29 dB band (38.65–39.94 dB) with no
outliers, mirroring the tight spread already seen on the unfixed-CUDA
baseline (40.34–41.77 dB, a 1.43 dB band) — the fix does not introduce
instability or high-variance frames, it shifts the whole distribution down
by a consistent amount. Frame 1 alone reached 39.940 dB, identical to the
single-frame diagnostic run to three decimal places, confirming the two runs
used byte-identical configuration.

**Mean drop: 41.218 → 39.379 dB = −1.839 dB**, matching the single-frame
diagnostic's −1.83 dB almost exactly. This is now the most direct, fully
controlled evidence in this repository for the actual cost of the CUDA fix
on native per-frame static reconstruction: a real, small, consistent
quality drop from correcting the position gradient, once the position LR is
properly rescaled to avoid the divergence/OOM that a naive port of the old
recipe causes.

## Timing — recorded two ways, per request

**1. Internal (each frame's own `psnr_log.txt` "Total training time")**:
sum over 90 frames = **3351.7 s**, mean **37.24 s/frame**. This is the same
methodology already used for `esvr_original_per_frame` (3627.0 s) and
`esvr_per_frame_baseline` (466.1 s).

**2. External wall-clock (measured by the driving shell script with
`date +%s.%N` around each `python train_chunk.py` subprocess, from process
launch to exit)**: sum over 90 frames = **3466.202 s** (57 min 46 s), mean
**38.51 s/frame**. Per-frame timings recorded in
`experiments/vorts90_cuda_fixed_lr_only_logs/wall_clock_timing.tsv`.

The **114.5 s gap** between the two totals (1.27 s/frame average) is process
startup and data loading overhead that happens *outside* the training loop's
own timer — not visible if only the internal number is quoted. Both numbers
are reported here because "how long did this actually take to run" (external)
and "how long did the training loop itself run" (internal) are different
questions; prior 90-frame entries in this repository only reported the
internal figure.

## Verification

- All 90 frames exited with code 0; `grep -li "error\|traceback\|out of
  memory\|nan"` across all 90 `experiments/vorts90_cuda_fixed_lr_only_logs/
  vorts_frame_*.log` files found nothing.
- Frame 1's result (39.940 dB, matching `vorts0001_cuda_fixed_lr_only` to
  3 decimal places) confirms this run's per-frame configuration is identical
  to the already-verified single-frame diagnostic, not a fresh unvalidated
  recipe.

## Not done

- No `--eval-only`/reload determinism check performed for this batch (the
  single-frame diagnostic and the historical `esvr_original_per_frame` /
  `esvr_per_frame_baseline` runs already established this training script's
  reload behavior; not re-verified per-frame here for time).
- No visualization (`quality_curves.png` / `slice_comparison.png` /
  `training_frames_dvr.gif`) generated yet for this run — can be added the
  same way as `esvr_original_per_frame`'s, on request.
- Single seed per frame (42, inherited default), no variance study.
