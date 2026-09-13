# vorts0001 single-frame static baseline — CUDA fix, position LR rescaled to match

> **The one genuinely single-variable comparison of the CUDA position-gradient
> fix against the original, unmodified `train_chunk.py` recipe.** Everything
> in `experiments/run_vorts0001.sh` is unchanged except `position_lr_init`/
> `position_lr_final`, rescaled to compensate for the fix's effect on
> gradient magnitude. Result: **39.94 dB**, vs. the pre-fix baseline's
> **41.77 dB** — a real **−1.83 dB** from the fix, under a fair comparison.

Completed 2026-09-13. Branch: `codex/deformable-esvr`. Not yet committed as
part of a dedicated commit for this table entry — training outputs
(checkpoints/PLYs/logs) are gitignored as usual; only this file is tracked.

## Why this run exists

`output/esvr_original_per_frame/RESULTS.md` (in the separate `ESVR/` working
copy) documents that reusing `train_chunk.py`'s tuned recipe directly against
the CUDA-fixed voxelizer **diverges and OOMs** — its `position_lr` was tuned
under the pre-fix bug, which suppressed position gradients by `dVoxel² =
(2/128)² = 1/4096` for this 128³ grid. The two existing 90-frame baselines
in this repository's `output/esvr_original_per_frame/RESULTS.md` (41.22 dB,
unfixed CUDA, native `train_chunk.py`) and `output/esvr_per_frame_baseline/
RESULTS.md` (38.86 dB, fixed CUDA, a *different, simplified* `train_dynamic.py
--allow-single-frame` pipeline) are **not a valid before/after comparison of
the fix** — they differ in training script, point-budget/densification
strategy, iteration count, and LR schedule simultaneously, not just CUDA.
This run isolates the fix as intended: same script, same densification
thresholds, same opacity/scaling/rotation LR, same iteration count and point
budget — only `position_lr` changed, and changed by a *derived*, not
arbitrary, amount.

## Derivation of the compensated position LR

The bug scaled `dL_dmean3D` by an erroneous `dVoxel` instead of the correct
`1/dVoxel` factor, so `gradient_buggy = gradient_correct × dVoxel²`. Fixing
it makes gradients `1/dVoxel² = 4096×` larger for this grid. To hold the
actual per-step position update (`lr × gradient`) constant relative to the
original tuned recipe:

```
new_lr = old_lr × dVoxel² = old_lr / 4096
position_lr_init:  0.000008   → 0.000000001953125   (8e-6 / 4096)
position_lr_final: 0.000001   → 0.000000000244140625 (1e-6 / 4096)
```

`densify_grad_threshold` also compares against this same gradient tensor
(`add_densification_stats` reads `viewspace_point_tensor.grad`, and
`gaussian_renderer.query`'s `"3d_mean"` is literally `pc.get_xyz` —
confirmed by reading both, not assumed). **A first attempt also rescaled
this threshold by ×4096 (to `0.001024`) on the same reasoning as the LR —
that was wrong in practice** (see
[../vorts0001_cuda_fixed_matched_lr/RESULTS.md](../vorts0001_cuda_fixed_matched_lr/RESULTS.md)
for the failed run and why). This run leaves `densify_grad_threshold` at its
original native value, `0.00000025`, unchanged.

## Configuration (identical to `experiments/run_vorts0001.sh` except position LR)

| Item | Original (unfixed CUDA) | This run (fixed CUDA) |
|---|---|---|
| CUDA voxelizer | unfixed | **fixed** (`ddelx_dx = 1.0f/dVoxel_x`) |
| `position_lr_init` / `_final` | `8e-6` / `1e-6` | **`1.953125e-9`** / **`2.44140625e-10`** |
| `opacity_lr_init` / `_final` | `7.5e-3` / `5e-4` | unchanged |
| `scaling_lr_init` / `_final` | `5e-5` / `5e-5` | unchanged |
| `rotation_lr_init` / `_final` | `1e-5` / `1e-5` | unchanged |
| `densify_grad_threshold` | `2.5e-7` | unchanged |
| `percent_dense`, pruning schedule, `init_num=20000` | — | unchanged |
| Iterations | 22,400 (2,800 scans) | unchanged |

## Results

| Metric | Original (unfixed CUDA) | This run (fixed CUDA, LR-compensated) |
|---|---:|---:|
| Final 3D PSNR | **41.77 dB** | **39.94 dB** |
| Final primitive count | 21,401 | 22,424 |
| Peak primitive count (before iter-9601 global prune) | 63,777 | 83,546 |
| Primitive count at iter 9601 (post-prune) | 21,537 | 22,976 |
| Training wall time | 36.8 s | 36.0 s |

Both runs show the same qualitative shape (rapid early growth from the
20,000-point initialization up to a peak by iter ~3,000–9,600, then a sharp
drop at the scheduled global-pruning step at iter 9,601, followed by gradual
refinement) — this run's peak (83,546) is about 31% higher than the
original's (63,777), and its final count (22,424) is about 5% higher
(21,401) — same ballpark, not a qualitatively different training regime.

**Applying only the CUDA fix, with the position LR rescaled to hold the
per-step update magnitude constant, reduces mean 3D PSNR from 41.77 dB to
39.94 dB — a genuine −1.83 dB, obtained without divergence, NaN, or OOM.**
This is a smaller, more defensible number than the uncontrolled 41.22→38.86 dB
(−2.36 dB) implied by comparing the two 90-frame baselines directly (see
"Why this run exists" above) — some of that gap was never about the CUDA fix
at all.

## What's not explained

There is still a ~1.83 dB gap unaccounted for beyond "the fix is real and
changes the result." Candidate explanations, **none tested here**:

- The compensation holds the *nominal* update magnitude constant on average,
  but position gradients are now correctly *directional* rather than merely
  correctly *scaled* — the fix changes more than a global multiplier in
  general (the buggy version scaled all three axes by the same erroneous
  `dVoxel` constant per axis, so direction was preserved to the extent
  `dVoxel_x == dVoxel_y == dVoxel_z`, which holds here since the grid is
  cubic — this is a weak candidate explanation for this particular grid).
- The heavier overgrowth-then-prune cycle (83,546 → 22,976 vs. 63,777 →
  21,537) discards a different, not-necessarily-equivalent set of primitives
  than the original run's cycle, since which primitives get flagged
  `large`/`small` and survive dark/global pruning depends on the whole
  training trajectory up to that point, not just point count.
- `densify_grad_threshold` was deliberately left unscaled (see Derivation);
  it is plausible that *some* partial, non-4096x rescaling of it would track
  the original recipe's densification behavior more closely than either the
  unscaled or the fully-4096x-scaled value tested so far — not swept.

## Verification

- Training exited cleanly (`Training finished` / `Training complete.`),
  22,400/22,400 iterations, no `Error`, `Traceback`, `CUDA out of memory`, or
  `nan` in the full training log.
- Per-iteration PSNR/primitive-count trajectory (`psnr_log.txt`) is smooth
  and monotonically improving after the iter-9601 pruning event, consistent
  with normal convergence, not an artifact of a crashed/truncated run.
- Cross-checked that `gaussian_renderer.query`'s `"3d_mean"` output (the
  tensor whose `.grad` feeds `add_densification_stats`) is literally
  `pc.get_xyz`/`means3D`, by reading `gaussian_renderer/__init__.py` directly
  — confirming the densify-trigger statistic goes through the same
  CUDA-fixed gradient path as the position optimizer update, not an
  independent, unaffected quantity.

## Not done

- Single seed; no repeat run to check run-to-run variance at this LR.
- No sweep of `densify_grad_threshold` to see whether a partial rescaling
  closes the residual ~1.83 dB gap (see "What's not explained").
- Not yet extended to the full 90-frame sequence — this is the single-frame
  diagnostic requested before committing to a 90-frame rerun.
- No `--eval-only --resume` reload-reproducibility check (unlike the dynamic-
  method RESULTS.md files) — `train_chunk.py`'s checkpoint format and reload
  path were not exercised here; only the training run's own logged metrics
  are reported.
