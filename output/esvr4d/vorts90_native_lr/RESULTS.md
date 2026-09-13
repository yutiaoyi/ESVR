# ESVR4D — native optimizer-group structure and LR values, real 90-frame data

> **Fixing the structural bug lowered the measured number — read the
> "Interpretation" section before treating 28.77 dB as the better result.**
>
> **2026-09-13 correction**: this run's `--steps 30000` (this method's own
> native `--iterations`) was *also* a change from `vorts90_fixed_points`'
> 40,000, alongside the LR-group fix — a second, unintended variable in the
> "28.77 vs 25.09 dB" comparison below. A step-matched rerun at 40,000 steps,
> everything else identical, is at
> [output/esvr4d/vorts90_native_lr_40k/RESULTS.md](../vorts90_native_lr_40k/RESULTS.md):
> it lands at 25.68 dB, confirming the LR-structure fix genuinely lowers
> ESVR4D's PSNR here (not just an artifact of fewer steps), though the true
> effect (−3.09 dB) is a little smaller than this page's confounded −3.68 dB
> headline number suggested.

Completed 2026-09-13. Branch: `codex/4dgs-esvr`.

## What changed from `esvr4d/vorts90_fixed_points`

That earlier run's optimizer conflated `xyz`, `time`, `log_scale`, and
`logit_scalar` into one shared `--lr` group. Checked against
`fudan-zvg/4d-gaussian-splatting`'s own `scene/gaussian_model.py`: natively
these are five separate groups — `xyz` (`0.00016→0.0000016`, the only one
that actually decays; `update_learning_rate`'s scheduling call for `"t"` is
commented out in the source), `t` (constant at `xyz`'s *initial* value),
`scaling` (constant `0.005`), `opacity` (constant `0.05`), `rotation`
(constant `0.001`) — up to ~300x apart. `train_esvr4d.py` now builds these
five groups for real; see `docs/ESVR4D.md`.

## Configuration

| Item | This run | `vorts90_fixed_points` (superseded) |
|---|---|---|
| xyz LR | 0.00016 → 0.0000016 (native, decaying) | 0.005 (shared, constant-ratio decay) |
| time LR | 0.00016 (native, constant) | 0.005 (shared with xyz/scale/opacity) |
| scale LR | 0.005 (native, constant) | 0.005 (shared) |
| opacity LR | 0.05 (native, constant) | 0.005 (shared) |
| rotation LR | 0.001 (native, constant) | 0.002 |
| Steps | 30,000 (native `--iterations`) | 40,000 |
| Points | 30,000 (unchanged) | 30,000 |
| Densification | Off | Off |

## Final full-volume metrics (all 90 training frames)

| Metric | This run (native) | `vorts90_fixed_points` (superseded) |
|---|---:|---:|
| Per-frame PSNR, arithmetic mean | 25.091 dB | 28.772 dB |
| Per-frame PSNR, median | 25.184 dB | 28.995 dB |
| Lowest / highest per-frame PSNR | 21.911 / 26.986 dB | 25.398 / 30.718 dB |
| Frame 1 / 46 / 90 PSNR | 25.207 / 24.935 / 21.911 dB | 28.582 / 29.300 / 25.398 dB |
| Training wall time | 158.3 s | 178.9 s |

## Interpretation — the structural fix was still correct

Fixing the conflated LR groups is not what caused the drop by itself in any
simple sense; the *values* changed too, and the native values are more
conservative for the parameter that matters most here. `xyz`'s native LR
(`0.00016`, decaying to near zero) is ~30x smaller than the shared value the
superseded run used (`0.005`) for the entire 40,000 steps. Since ESVR4D's
motion representation depends on the cross-covariance term and the mean
*shift* with time (`docs/ESVR4D.md`'s "Mathematical representation") — i.e.
literally on `xyz` moving — a much smaller, decaying position LR gives the
model less room to reposition primitives to track motion within a fixed step
budget. This mirrors what the matched-vs-native-LR comparison already found
for the deformable-MLP branch (`dynamic/vorts90_matched_native_lr/RESULTS.md`):
donor papers' own LR schedules are tuned for their own, typically much
longer, training budgets and real-scene complexity, and can be genuinely
more conservative than a shorter custom schedule tolerates well.

**What this run does confirm, independent of the LR magnitude debate**: the
five-group structure itself is correct (verified against upstream source,
not assumed) and stable — 3 seeded reruns without densification and 3 with
it, all clean, before this real run was launched. Whether to use these exact
native LR *values* going forward, or a custom (larger, xyz-favoring)
schedule on top of the now-correct group structure, is a separate, open
question — not resolved by this run.

## Verification

- Training exited cleanly (code 0) at step 30,000.
- `--eval-only --resume checkpoint.pt` into a fresh directory reproduced all
  90 exported `.npy` frames with **max abs diff 0.0**.

## Not done

- No held-out time split (same caveat as every other 90-frame run in this
  repository).
- No LR sweep to find a value that keeps the correct native group structure
  but recovers or exceeds the superseded run's quality.
- Single seed (42).
