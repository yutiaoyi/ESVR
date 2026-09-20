# ESVR-Scaffold4D — native 4D-Scaffold-GS hyperparameters, real 90-frame data

> **Superseded (2026-09-20)** by [output/scaffold4d/ablation/RESULTS.md](../ablation/RESULTS.md).
> This run's 24.35 dB came from a model that degraded during training
> (active children 330k → 21k; half of all children pinned on the 1e-4
> scale floor and permanently gradient-free; 14k/40k anchors transparent
> but never pruned; growing was a clone/split routine, not the paper's
> anchor growing). Same recipe with those fixed: 31.9 dB at 40k anchors.

> **What this run is**: uses 4D-Scaffold-GS's own hyperparameters
> (`train_scaffold4d.py`'s current defaults, all sourced from
> `arguments/__init__.py` and `scripts/train_n3dv.sh` in the pinned upstream
> commit — see `docs/ESVR_SCAFFOLD4D.md`), not values chosen to match the
> other two branches. No config alignment was applied here, per explicit
> request. Same 90-frame `vorts` manifest, all frames used for training (no
> held-out split), same protocol as the other real-data runs in this
> repository (`esvr4d/vorts90_fixed_points`,
> `dynamic/vorts90_matched_comparison`,
> `dynamic/vorts90_native_recipe_cuda_fixed`).

Completed 2026-09-12. Branch: `codex/scaffold4d-esvr`.

## Getting here required two real bug fixes, not just config changes

Switching from a hand-picked matched configuration to the paper's own
optimizer structure and step count surfaced two genuine architectural bugs
that the earlier matched-config runs and synthetic smoke test never
exercised — see `docs/ESVR_SCAFFOLD4D.md` for the full account:

1. Anchor position and offset had been wrongly merged into one optimizer
   group; upstream freezes anchor position (`lr=0`) and gives offset its own
   independently-scheduled LR (`0.01→0.0001`). Fixed by giving all seven of
   upstream's native parameter groups their own optimizer group and
   `expon_lr` schedule.
2. The final per-child scale (`offset_scale × sigmoid(shape_gate)`) had no
   real floor, since `sigmoid` can approach 0 arbitrarily closely. Real
   90-frame training reproducibly hit a nonfinite loss — at a *different*
   step on every identically-seeded rerun (a float32/CUDA-atomics precision
   cascade, not a fixed logic bug), traced via diagnostics to the scale
   collapsing to `~1e-9`–`~1e-10` right before the crash, degenerate enough
   to make that ellipsoid's covariance singular in the CUDA voxelizer. Fixed
   with a direct floor (`clamp_min(1e-4)`) on the product, plus matching
   guardrails on `sigma` and the raw `offset`. Verified across 5 reseeded
   reruns without densification and 3 with densification, all clean where
   every one of those seeds reliably crashed before the fix.

## Configuration (4D-Scaffold-GS's own values, no alignment to the other two branches)

| Item | Value | Source |
|---|---|---|
| Anchors (initial) | 6,000 | ESVR-chosen starting point (paper initializes from COLMAP/SfM points, no direct equivalent here) |
| Children per anchor | 10 | Native `n_offsets` |
| Anchor position LR | 0.0 (frozen) | Native `position_lr_init`/`_final` |
| Offset LR | 0.01 → 0.0001 | Native `offset_lr_init`/`_final` |
| Scale LR (`log_offset_scale`) | 0.007 (constant) | Native `scaling_lr` |
| Feature LR | 0.0075 (constant) | Native `feature_lr` |
| Opacity-MLP LR | 0.002 → 0.00002 | Native `mlp_opacity_lr_init`/`_final` |
| Shape-MLP LR | 0.004 (constant) | Native `mlp_cov_lr_init` == `_final` |
| Flow-MLP LR | 0.001 (constant) | Native `mlp_flow_lr_init` == `_final` |
| Steps | 120,000 | `scripts/train_n3dv.sh`'s `--iterations` |
| Densify window | step 1,500–60,000 | Native `update_from` / `train_n3dv.sh`'s `--update_until 60000` |
| Densify interval | every 100 steps | Native `update_interval` |
| Densify grad threshold | 2e-4 | Native `densify_grad_threshold` |
| λ_ssim / λ_vol | 0.2 / 0.01 | Native `lambda_dssim` / volume regularizer weight |
| Temporal β | 4.0 | Native `hparam_beta` |
| Max anchors | 40,000 | **ESVR-only safety cap, not native** — the paper has no hard ceiling; hit during this run (see below) |
| Block / tile-sampling | 64³ / on | **ESVR-only** — no native equivalent (paper samples camera rays, not 3D blocks) |

## Final full-volume metrics (all 90 training frames)

| Metric | Value |
|---|---:|
| Per-frame PSNR, arithmetic mean | 24.348 dB |
| Per-frame PSNR, median | 24.529 dB |
| Lowest / highest per-frame PSNR | 21.940 / 25.431 dB |
| Frame 1 / 46 / 90 PSNR | 24.040 / 25.064 / 21.940 dB |
| Mean normalized MAE | 0.04631 |
| Training wall time | 725.3 s (~12.1 min) for 120,000 steps |
| Final anchor count | 40,000 (hit the safety cap; densification stopped growing once the budget ran out) |

For reference, not a controlled comparison (different steps, optimizer
structure, and anchor budget in each case — see each run's own RESULTS.md
for what varies):

| Run | Config | Mean PSNR |
|---|---|---:|
| This run | Native hyperparameters, 120,000 steps, densify on | 24.35 dB |
| `scaffold4d/vorts90_matched_comparison` | Matched to ESVR4D/deformable-MLP, 40,000 steps, densify off | 25.03 dB (used `feat_lr`/`children` values that turned out not to match the paper either — see chat history) |
| `esvr4d/vorts90_fixed_points` | ESVR4D, matched config, densify off | 28.77 dB |
| `dynamic/vorts90_native_recipe_cuda_fixed` | Deformable-MLP's own native recipe | 24.25 dB |

## Verification

- Training exited cleanly (code 0) at step 120,000; no nonfinite-loss
  crashes with the scale/sigma/offset guardrails in place.
- `--eval-only --resume checkpoint.pt` into a fresh directory reproduced all
  90 exported `.npy` frames with **max abs diff 0.0**.
- Densification ran throughout its native window (1,500–60,000), growing
  from 6,000 to the 40,000-anchor cap; `active` children per query settled
  to roughly 18,000–23,000 out of a possible 400,000 (40,000 × 10) by the
  end of training — i.e. the temporal gate is genuinely selective (only
  ~5% of children "on" at any one query time), unlike the pre-fix runs
  where 95-99% of children were active simultaneously regardless of query
  time, one of the symptoms that led to the scale-floor diagnosis.

## Not done

- No held-out time split (same caveat as every other 90-frame run in this
  repository).
- Single seed (42).
- `max_anchors=40,000` is an ESVR-practical safety cap, not a paper value;
  the run saturated it, so a higher cap might change the result — not
  tested.
- No investigation of whether densification's reconstruction-quality
  effect is net positive here the way `esvr4d`'s own densification
  investigation found it net-harmful on that different representation;
  this run only confirms densification no longer *crashes*.
