# Deformable-MLP — native canonical LR groups, real 90-frame data

> **Best real-data 3D PSNR of any run in this repository so far (28.90 dB),
> from a structural bug fix, not a hyperparameter search.**

Completed 2026-09-13. Branch: `codex/4dgs-esvr`.

## What changed

`canonical = [model.xyz, model.log_scale, model.rotation, model.logit_scalar]`
shared one `--lr`. Checked against this repository's own
`experiments/run_vorts0001.sh` (the script behind the validated
single-frame static baseline, not just `arguments/__init__.py`'s bare
defaults): natively these are four separate groups — position
`8e-6→1e-6` (decaying), scaling `5e-5` (constant), rotation `1e-5`
(constant), opacity `7.5e-3→5e-4` (decaying) — spanning ~1000x. Fixed with
five real optimizer groups (four canonical + the deformation network,
already correctly a single group matching upstream
`scene/deform_model.py`). See `docs/DEFORMABLE_ESVR.md`.

## Configuration

| Item | This run | Prior best (`vorts90_matched_comparison`) |
|---|---|---|
| Canonical position LR | 8e-6 → 1e-6 (native, decaying) | 5e-3 (shared, constant-ratio decay) |
| Canonical scale LR | 5e-5 (native, constant) | 5e-3 (shared) |
| Canonical rotation LR | 1e-5 (native, constant) | 5e-3 (shared) |
| Canonical opacity LR | 7.5e-3 → 5e-4 (native, decaying) | 5e-3 (shared) |
| Deformation network LR | 8e-4 (unchanged; already correctly its own group) | 2e-3 |
| Steps / points / warmup | 40,000 / 30,000 / 1,000 (all unchanged) | same |

## Final full-volume metrics (all 90 training frames)

| Metric | This run | `vorts90_matched_comparison` | `vorts90_matched_native_lr` | ESVR4D `vorts90_fixed_points` |
|---|---:|---:|---:|---:|
| Per-frame PSNR, arithmetic mean | **28.895 dB** | 25.958 dB | 24.207 dB | 28.772 dB |
| Per-frame PSNR, median | 28.907 dB | 26.049 dB | 24.279 dB | 28.995 dB |
| Lowest / highest per-frame PSNR | 25.565 / 30.014 dB | 23.384 / 26.828 dB | 21.321 / 25.053 dB | 25.398 / 30.718 dB |
| Frame 1 / 46 / 90 PSNR | 28.518 / 29.046 / 25.565 dB | 25.646 / 26.074 / 23.384 dB | 23.380 / 24.271 / 21.321 dB | 28.582 / 29.300 / 25.398 dB |
| Training wall time | 500.3 s | 1023.2 s | 488.3 s | 178.9 s |

## Interpretation — this fix helped here, unlike the same fix on ESVR4D

`output/esvr4d/vorts90_native_lr/RESULTS.md` found the analogous fix
*lowered* ESVR4D's measured PSNR (28.77 → 25.09 dB), because ESVR4D's
motion representation depends directly on `xyz` moving (the cross-covariance
term), and native position LR is far more conservative than what that
run had been using. Deformable-MLP is structurally different: motion is
carried by the **deformation network** (`args.deform_lr`, untouched by this
fix, already correctly its own optimizer group matching upstream), not by
canonical position. Canonical `xyz`/`scale`/`rotation` only need to settle
into a good *static* geometry — exactly what their native (conservative,
long-horizon) LRs are tuned for — while canonical **opacity** getting its
own properly-sized native LR (`7.5e-3`, ~1.5x the previous shared `5e-3`,
with a matching decay) plausibly explains most of the gain, since opacity
directly controls which primitives contribute to the reconstruction at all.
This is a mechanistic hypothesis consistent with the numbers, not
independently ablated (no run isolating the opacity-LR change alone).

## Verification

- Training exited cleanly (code 0) at step 40,000.
- `--eval-only --resume checkpoint.pt` into a fresh directory reproduced all
  90 exported `.npy` frames with **max abs diff 0.0**.
- `experiments/summarize_dynamic_run.py` ran clean: all artifacts present,
  `quality_curves.png`, `slice_comparison.png`, `training_frames_dvr.gif`,
  `summary.json`, `data_sha256.json`, `provenance.json` generated.

## Not done

- No held-out time split (same caveat as every other 90-frame run in this
  repository).
- No ablation isolating which of the four regrouped LRs (position/scale/
  rotation/opacity) drives the improvement; the Interpretation above is a
  plausible mechanistic read, not a controlled isolation.
- Single seed (42).
- Densification still not attempted for this branch's real data (matches
  the sibling branches' own open densification questions).
