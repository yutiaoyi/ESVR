# Deformable-MLP (dynamic_esvr) — matched config, native LR only

> **What this run is, precisely**: a *single-variable* comparison. Identical
> to [output/dynamic/vorts90_matched_comparison/RESULTS.md](../vorts90_matched_comparison/RESULTS.md)
> (random init, 30,000 primitives, 1,000-step warmup, everything else) —
> the **only** change is the learning rate, switched from that run's 5e-3/2e-3
> to deformable-MLP's own original 5e-5/1.6e-4 (from
> `experiments/run_vorts_dynamic.sh`). This isolates whether the LR itself
> (as opposed to initialization strategy) explains the gap to ESVR4D,
> without giving deformable-MLP a PLY-warm-start head start that ESVR4D has
> no equivalent way to use (an earlier version of this comparison did warm-
> start from a PLY and was removed for exactly that reason — see git
> history and the note in `vorts90_matched_comparison/RESULTS.md`).

Completed 2026-09-12. Branch: `codex/4dgs-esvr`.

## Configuration

| Item | Value | vs. `vorts90_matched_comparison` |
|---|---|---|
| Initialization | Random (no `--init-ply`) | same |
| Canonical primitives | 30,000, fixed | same |
| Static warmup | 1,000 steps | same |
| Training steps | 40,000 | same |
| Spatial sampling | 64³ tiles, `--tile-sampling` | same |
| **LR** | **5e-5 canonical, 1.6e-4 deformation network** | **changed (was 5e-3 / 2e-3)** |
| Loss | 0.8·L1 + 0.2·(1 − SSIM) + 1e-4·squared deformation penalty | same |
| Seed | 42 | same |

## Final full-volume metrics (all 90 training frames)

| Metric | This run (native LR) | `vorts90_matched_comparison` (larger LR) | ESVR4D (same init/points) |
|---|---:|---:|---:|
| Per-frame PSNR, arithmetic mean | 24.207 dB | 25.958 dB | 28.772 dB |
| Per-frame PSNR, median | 24.279 dB | 26.049 dB | 28.995 dB |
| Lowest / highest per-frame PSNR | 21.321 / 25.053 dB | 23.384 / 26.828 dB | 25.398 / 30.718 dB |
| Frame 1 / 46 / 90 PSNR | 23.380 / 24.271 / 21.321 dB | 25.646 / 26.074 / 23.384 dB | 28.582 / 29.300 / 25.398 dB |
| Training wall time | 488.3 s | 1023.2 s | 178.9 s |

**Isolating the LR variable**: with random init and 30,000 primitives held
fixed, deformable-MLP's own native LR reaches 24.21 dB — **1.75 dB lower**
than the larger, from-scratch-appropriate LR used in
`vorts90_matched_comparison`. This confirms the native LR (5e-5/1.6e-4,
tuned historically for *fine-tuning* a PLY that's already close to correct)
is measurably too conservative for training from random noise, independent
of whether a PLY warm-start is used at all — a PLY-warm-started run at this
same native LR (run earlier, then deleted per the reasoning above) reached
24.25 dB, essentially identical to this run's 24.21 dB. In other words: once
LR is controlled for, **the apparent benefit of PLY warm-start on this
metric nearly vanishes** — most of what looked like a warm-start advantage
was actually a learning-rate effect.

Under either LR choice, deformable-MLP (24.2–26.0 dB) remains well below
ESVR4D's 28.77 dB at matched initialization/primitive count/step budget.

## Verification

- Training exited cleanly (code 0) at step 40,000.
- `--eval-only --resume checkpoint.pt` into a fresh directory reproduced all
  90 exported `.npy` frames with **max abs diff 0.0**.
- `experiments/summarize_dynamic_run.py` ran clean: all artifacts present,
  `quality_curves.png`, `slice_comparison.png`, `training_frames_dvr.gif`,
  `summary.json`, `data_sha256.json`, `provenance.json` generated.

## Not done

- No held-out time split (same caveat as every other 90-frame run in this
  set).
- Single seed (42).
- The PLY-warm-started/native-LR data point (24.25 dB) is reported here only
  as a number recalled from conversation, not as a reproducible file in this
  repository — it was deleted, so it should not be treated as an independently
  verifiable result going forward, only as context for why it was removed.
