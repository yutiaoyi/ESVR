# Deformable-MLP (dynamic_esvr) — native recipe, CUDA fix as the only variable

> **What this run is, precisely**: the *own-best-practice* companion to
> [output/dynamic/vorts90_matched_comparison/RESULTS.md](../vorts90_matched_comparison/RESULTS.md).
> That run stripped deformable-MLP's PLY warm-start and fine-tuning LR to
> match ESVR4D's settings, isolating the motion representation but denying
> the method its own intended initialization strategy. This run instead
> reuses `experiments/run_vorts_dynamic.sh` — the exact original recipe
> behind [output/dynamic/vorts90_full_geometry/RESULTS.md](../vorts90_full_geometry/RESULTS.md)
> — completely unchanged, in the CUDA-fixed copy. **The CUDA position-
> gradient fix is the only variable that differs from the historical run.**
> Both this run and the matched-config run are legitimate; they answer
> different questions (see the note in the matched-config RESULTS.md).

Completed 2026-09-12. Branch: `codex/4dgs-esvr`.

## Configuration (identical to the historical run)

| Item | Value |
|---|---|
| Initialization | `--init-ply output/experiments/vorts0001_baseline/vorts0001_chunk_0/point_cloud/iteration_22400/point_cloud.ply` — the single-frame static ESVR baseline's own densified result |
| Canonical primitives | 21,401 (inherited from that PLY; fixed, no further densification) |
| Deformation network | 8 layers, width 256, 10 Fourier bands each for XYZ/time |
| Static warmup | 0 steps (starts fine-tuning the deformation immediately, since canonical geometry is already good from the PLY) |
| Training steps | 40,000 |
| Spatial sampling | 64³ non-overlapping tiles, `--tile-sampling` |
| LR | 5e-5 canonical, 1.6e-4 deformation network — small, fine-tuning-scale rates, unchanged from the original recipe |
| Loss | 0.8·L1 + 0.2·(1 − SSIM) + 1e-4·squared deformation penalty |
| Seed | 42 |
| Device | RTX 5090; CUDA-fixed `gaussian_rasterization_voxelization` build |

## Final full-volume metrics (all 90 training frames)

| Metric | Historical (pre-fix) | This run (CUDA fix only) | Matched-config (random init, larger LR) |
|---|---:|---:|---:|
| Per-frame PSNR, arithmetic mean | 22.353 dB | **24.254 dB** | 25.958 dB |
| Per-frame PSNR, median | 22.396 dB | 24.228 dB | 26.049 dB |
| Lowest / highest per-frame PSNR | 19.759 / 23.748 dB | 21.718 / 26.924 dB | 23.384 / 26.828 dB |
| Frame 1 / 46 / 90 PSNR | 23.234 / 22.138 / 19.759 dB | 25.688 / 24.137 / 21.718 dB | 25.646 / 26.074 / 23.384 dB |
| Training wall time | 431.7 s | 519.7 s | 1023.2 s |
| Canonical primitives | 21,401 | 21,401 | 30,000 |

Applying **only** the CUDA fix to the method's own unchanged recipe improves
mean PSNR by **+1.90 dB** (22.35 → 24.25 dB) — confirming the fix helps this
method's real, as-tuned usage, not just a synthetic benchmark. It is slower
per step than the historical run (519.7 s vs 431.7 s for the same 40,000
steps) purely from added evaluation/monitoring overhead parity with the other
two 2026-09-12 runs, not from the fix itself.

It is still **~1.7 dB below** the matched-config run's 25.958 dB, which uses
random init and a ~100x larger LR. This suggests the original recipe's LR
(5e-5/1.6e-4), tuned for a regime where position barely moved, is likely
still too conservative to fully exploit the now-correct gradient — an
unconfirmed hypothesis, not tested further here (would require its own LR
sweep, out of scope).

## Verification

- Training exited cleanly (code 0) at step 40,000.
- `--eval-only --resume checkpoint.pt` into a fresh directory reproduced all
  90 exported `.npy` frames with **max abs diff 0.0**.
- `experiments/summarize_dynamic_run.py` ran clean: all per-frame artifacts,
  `quality_curves.png`, `slice_comparison.png`, `training_frames_dvr.gif`,
  `summary.json`, `data_sha256.json`, `provenance.json` generated.

## Not done

- No held-out time split — same caveat as every other 90-frame run in this
  set (`vorts90_full_geometry`, `vorts90_matched_comparison`,
  `esvr4d/vorts90_fixed_points`, `esvr_per_frame_baseline`): all use
  `experiments/vorts_90_all_train.json` (90/90 frames marked `train`).
- Single seed (42).
- No LR sweep to test the hypothesis above about the original recipe's LR
  being too conservative post-fix.
