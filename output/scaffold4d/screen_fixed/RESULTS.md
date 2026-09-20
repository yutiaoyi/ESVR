# ESVR-Scaffold4D — capacity-recycling and children-per-anchor screen (fixed kernel, 2026-09-20)

40,000 steps, growing window 1,500–20,000, 40,000-anchor cap unless noted,
seed 42, current `train_scaffold4d.py` defaults, fixed voxelizer through
`.venv/bin/python` (`experiments/run_scaffold4d_screen.sh`). Each run changes
one thing versus `T0_control`. Seed noise at 120k steps is ±0.3–0.8 dB, so
only differences above ~1 dB are read as effects.

## Why this screen exists

Diagnosis of the fixed-kernel 40k-anchor model (`fixed_kernel/F_default_s42`,
`experiments/diagnose_scaffold4d.py` + `count_query`): 23 % of children
useful (intensity > 1/255 and not on the 0.5-voxel floor), 39 % of anchors
without a useful child, 60 % of the children selected for a query cover no
voxel. The hypothesis was that recycling or protecting this dead capacity
would raise PSNR.

## Results

| Run | Single change vs T0 | Size | Mean PSNR | Δ | Min / max frame |
|---|---|---:|---:|---:|---|
| T0_control | — | 13.3 MB | 30.823 | — | 27.91 / 32.49 |
| T1_volreg_visible | volume regularizer only on visible children | 13.3 | 30.779 | −0.04 | 27.47 / 32.27 |
| T2_prune_dead | ESVR-style dead-anchor recycling (`--prune-dead`) | 13.3 | 30.972 | +0.15 | 27.26 / 32.82 |
| T3_spread | newborn children spread over the cell (`--offset-spread 0.5`) | 13.3 | 30.075 | −0.75 | 27.40 / 31.49 |
| T4_childscale4 | newborn child scale 4 voxels | 13.3 | 29.626 | **−1.20** | 26.20 / 31.00 |
| T5_sigma1000 | σ clamp 100 → 1000 | 13.3 | 30.493 | −0.33 | 27.49 / 31.71 |
| T6_novol | λ_vol = 0 | 13.3 | 30.253 | −0.57 | 26.41 / 31.77 |
| T7_combo | T1 + T2 + T3 + T4 | 13.3 | 30.305 | −0.52 | 27.35 / 31.81 |
| T8_K5 | K = 5 | 10.1 | 30.475 | −0.35 | 26.90 / 31.91 |
| T9_K5_cap80k | K = 5, cap 80,000 | 20.2 | 31.581 | +0.76 | 27.79 / 33.11 |
| T10_K3 | K = 3 | 8.8 | 30.332 | −0.49 | 27.01 / 31.58 |

## Reading

- **No capacity-recycling fix helps; three hurt.** `T2` did what it was
  built to do — 12,385 anchors pruned as dead, 12.7k regrown in their place
  — yet the useful-child fraction stayed at 20 % (T0: 20 %, T2: 20 %) and
  PSNR did not move: newborn anchors die at the same rate as the ones they
  replace. Making newborns larger (T4) or spreading them (T3) is worse,
  removing the volume regularizer is worse (it does restrain growth of
  large children), gating it by visibility does nothing (uncovered children
  are not being shrunk to death by it).
- Interpretation: at 40k anchors the model already uses the capacity it can;
  surplus children being driven dead is the optimiser switching them off
  (upstream masks them with tanh at no cost; here they must be shrunk or
  dimmed). Dead capacity is a storage inefficiency, not the PSNR limit.
- **Children per anchor**: K = 3 / 5 / 10 at 40k anchors trade ~0.35–0.5 dB
  for 25–35 % less size; K = 5 at 80k anchors gains 0.76 dB for 1.5× size.
  Per byte the three K values are close, K = 5 marginally best. Phase 2
  (`output/scaffold4d/scale/`) scales both K = 10 and K = 5.
- `--prune-dead`, `--vol-reg-visible`, `--offset-spread`,
  `--child-scale-init-voxels`, `--sigma-max` stay available but off.

## Not done

Single seed; 40k steps (effects on the growing phase only); no held-out
frames.
