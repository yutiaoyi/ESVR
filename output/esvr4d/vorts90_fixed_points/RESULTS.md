# ESVR4D — vorts 90-frame fixed-population run

> **2026-09-13 update**: this run's optimizer conflated `xyz`/`time`/
> `log_scale`/`logit_scalar` into one shared LR group — checked against
> upstream, natively these are five separate groups up to ~300x apart. Fixed
> in [output/esvr4d/vorts90_native_lr/RESULTS.md](../vorts90_native_lr/RESULTS.md),
> which also found the native LR *values* (as opposed to just the group
> structure) give lower measured PSNR here (25.09 dB vs. this run's
> 28.77 dB) — read that report's Interpretation section before treating
> either number as simply "better."

First full real-data run of the native-spacetime-primitive representation
(`esvr4d/`), on the same 90-frame `vorts` sequence as
`output/dynamic/vorts90_full_geometry/RESULTS.md` (deformable-MLP branch).
See [docs/ESVR4D.md](../../../docs/ESVR4D.md) for the representation, the
CUDA position-gradient bug found and fixed to make this branch's copy of the
repository usable, and the densification investigation this run's
configuration follows from.

Completed 2026-09-12. Branch: `codex/4dgs-esvr`. Run in
`ESVR-4dgs/` (the CUDA-fixed copy) — **not** applied to the original working
copy this review was run from; see chat history for that decision.

## Why densification is off

A short real-data validation (not saved as a run, see chat log) found that
`ESVR4D.densify()`, even after fixing an unrelated gradient-accumulation
scaling bug in `train_esvr4d.py` and empirically recalibrating its
thresholds against real observed gradients, still made reconstruction
*worse* than a fixed population at matched step counts (16.5 dB vs 18.4 dB
average of 3 monitored frames at step 1500). A partial diagnosis: ESVR's
voxelizer **adds** scalar contributions rather than alpha-compositing them,
so a split/clone's near-duplicate primitives roughly double the local
reconstructed value where they land — unlike the donor paper's alpha-blended
renderer, where the same recipe is safe. An opacity-halving compensation for
that specific mechanism was implemented and unit-tested but did not fully
fix the regression, so a second, unidentified factor is also at play
(candidates: zero-initialized Adam moments for newborns colliding with a
still-high learning rate, or a splitting cadence too fast relative to how
quickly Adam can stabilize new primitives). This run uses a **fixed**
30,000-primitive population instead, matching the precedent already set by
`codex/deformable-esvr` (`docs/DEFORMABLE_ESVR.md`: "Fixed primitive count
intentionally avoids transplanting RGB screen-gradient densification
criteria into a 3D scalar problem"). Debugging densification further is
out of scope for this run.

## Data

Identical manifest and files as the deformable-MLP branch's 90-frame run:
`experiments/vorts_90_all_train.json`, `data/xueliao_timevaryingdata/vorts_timevarying/vorts0001.raw`
through `vorts0090.raw`, 128×128×128 little-endian float32, ZYX storage /
X fastest, common range [-1,1], all 90 frames used for training (**no
held-out split** — these are training-reconstruction metrics, not a
generalization test, same caveat as the deformable-MLP run).

## Configuration

| Item | Value |
|---|---|
| Initialization | Random (`torch.rand`/identity quaternions); no PLY warm start — `ESVR4D` has no `initialize_ply` yet |
| Primitives | 30,000, **fixed** (densification disabled, `--densify-from` set past `--steps`) |
| Representation | Native 4D covariance (`xyz`, `time`, 4D log-scale, dual-quaternion `SO(4)` rotation, scalar logit); motion via Gaussian conditioning on `t`, not an MLP |
| Temporal init | `temporal_scale=0.5`, `temporal_cutoff=0.05` (normalized time units) |
| Training steps | 40,000 |
| Spatial sampling | 64³ non-overlapping tiles, `--tile-sampling` (boundary-inclusive) |
| Time sampling | Uniform random over 90 training frames |
| LR | 5e-3 (xyz/time/log_scale/logit_scalar), 2e-3 (rotation quaternions); exponential decay to 10% over the schedule |
| Loss | 0.8·L1 + 0.2·(1 − existing ESVR SSIM) |
| Seed | 42 |
| Device | RTX 5090; Python 3.10; Torch 2.7.1+cu128 (rebuilt `gaussian_rasterization_voxelization` with the position-gradient fix) |

Wall time: 178.9 s (~3.0 min) for 40,000 steps, monitoring included, per
`train.jsonl`'s final `seconds` field — no per-frame PLY-warm-start
pretraining is part of this number (there is none, since there is no PLY
init in this branch yet).

## Final full-volume metrics (all 90 training frames)

All metrics use unclipped predictions, normalized data range 1, same
definitions as `output/dynamic/vorts90_full_geometry/RESULTS.md` (3D PSNR:
per-frame MSE over the full 128³ volume, then arithmetic mean over frames).

| Metric | Value |
|---|---:|
| Per-frame PSNR, arithmetic mean | **28.772500 dB** |
| Per-frame PSNR, median | 28.995255 dB |
| Lowest / highest per-frame PSNR | 25.397757 / 30.717905 dB |
| Frame 1 PSNR | 28.582181 dB |
| Frame 46 PSNR | 29.300029 dB |
| Frame 90 PSNR | 25.397757 dB |
| Mean normalized MAE | 0.027426 |
| Mean RMSE, original RAW units | 0.073188 |

Frame 90 (the last, most-motion-accumulated frame, same as the deformable-MLP
run's weakest frame) is again the weakest here, though far less severely
(25.4 dB vs 19.8 dB there).

**For reference only, not a controlled comparison**: the deformable-MLP
branch's equivalent 90-frame, all-training, same-metric-definition run
(`output/dynamic/vorts90_full_geometry/RESULTS.md`) reported 22.352944 dB
mean PSNR. This run's 28.77 dB is ~6.4 dB higher, but the two runs differ in
more than the motion representation: this one trains from scratch (no PLY
warm start, unlike the other's iteration-22400 PLY initialization) with
30,000 primitives instead of 21,401, 40,000 steps either way but different
LR schedules and loss weighting, and — critically — **this copy of the
repository has the CUDA position-gradient fix documented in
docs/ESVR4D.md; the deformable-MLP run did not.** That fix plausibly would
also improve the deformable-MLP result if rerun here, so this 6.4 dB gap
should not be read as isolating "native 4D covariance vs. deformation MLP"
as a representation choice — re-running the deformable-MLP branch under the
same CUDA fix, with matched primitive count and initialization, would be
needed to make that comparison meaningful. Not done as part of this run.

## Verification

- Training exited cleanly (code 0) at step 40,000; `train.jsonl`, `monitor.jsonl`,
  `frame_sampling_counts.json`, `checkpoint.pt`, `checkpoint_010000/020000/030000.pt`
  snapshots, and `evaluation/` (90 `.npy`, 90 `.raw`, 270 `*_gt_pred.png`,
  `metrics.json`, `export_format.json` — 452 files total) are all present.
- `--eval-only --resume checkpoint.pt` into a fresh directory reproduced all
  90 exported `.npy` frames with **max abs diff 0.0** against the original
  run's `evaluation/` output.
- Visually inspected `frame_0000_axis0_gt_pred.png` (GT left, prediction
  right) and `frame_0089_axis0_gt_pred.png` (the weakest frame): frame 0
  is close to indistinguishable from GT; frame 89 shows visible softening
  of fine turbulent structure but the same overall flow pattern, consistent
  with 28.6 dB and 25.4 dB respectively — not a degenerate or collapsed
  reconstruction.
- `tests/test_esvr4d.py` (5/5) and `tests/test_dynamic_esvr.py` (3/3) pass
  against the CUDA-fixed extension used for this run.

## Not done

- No held-out time split (matches the deformable-MLP run's own caveat).
- No PLY export (documented limitation in `docs/ESVR4D.md`); no
  `evaluate_paper_metrics.py`/`evaluate_official_tf.py`-style 2D PSNR/SSIM/LPIPS
  or offline-DVR comparison for this checkpoint yet.
- Densification remains unresolved (see above) — this is a fixed-capacity
  result, not evidence that adaptive density control works for this
  representation on this data.
- No controlled comparison isolating the motion representation from the
  CUDA fix, initialization, or primitive count (see caveat above).
