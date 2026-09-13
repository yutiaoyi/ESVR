# vorts0001 single-frame — CUDA fix, position LR *and* densify threshold rescaled (superseded)

> **Superseded by
> [../vorts0001_cuda_fixed_lr_only/RESULTS.md](../vorts0001_cuda_fixed_lr_only/RESULTS.md).**
> This run also rescaled `densify_grad_threshold` by the same ×4096 factor as
> `position_lr`, on the hypothesis that both compare against the same
> CUDA-fixed gradient tensor and so both needed compensating. That hypothesis
> was directionally wrong in practice — kept here as a documented negative
> result, not deleted.

Completed 2026-09-13. Branch: `codex/deformable-esvr`.

## What was tried

Same base config as `experiments/run_vorts0001.sh`, fixed CUDA voxelizer,
with **both**:
- `position_lr_init` / `_final`: `8e-6`/`1e-6` → `1.953125e-9`/`2.44140625e-10`
  (÷4096, to compensate the fix making position gradients 4096× larger on
  this 128³ grid)
- `densify_grad_threshold`: `2.5e-7` → `0.001024` (×4096, on the reasoning
  that this threshold is compared against the *same* gradient tensor, so it
  should scale the opposite direction to preserve the original
  fraction-of-primitives-triggering-densification)

## What happened

Training completed cleanly (no NaN/OOM) but **densification barely
triggered at all**: primitive count only ever *decreased* from the 20,000
initialization (20,000 → 19,940 → … → 18,221) through the entire
`densify_until_iter` window, then the scheduled global prune at iter 9,601
cut it further to 5,473, ending at **5,409 primitives** — roughly a quarter
of the original run's 21,401. Final PSNR: **38.81 dB**.

This is inconsistent with genuine densification behavior (the original run
grows from 20,000 up to a 63,777-primitive peak before the same prune step —
see the corrected run's RESULTS.md for the comparable, healthy trajectory).
The ×4096 threshold scaling suppressed splitting almost entirely instead of
preserving it.

## Why this was reverted

Not fully root-caused (see the corrected run's "Not done" / "What's not
explained" for what remains open), but empirically: leaving
`densify_grad_threshold` at its **original, unscaled** value while keeping
only the `position_lr` compensation reproduces a primitive-count trajectory
in the same range as the original run (peak 83,546 vs. 63,777, final 22,424
vs. 21,401) and a notably higher, more defensible PSNR (39.94 dB vs. this
run's 38.81 dB). Whatever the precise mechanism, treating the densification
threshold as needing the same compensation as the position LR was the wrong
call for this pipeline.

## Not done

- Root-causing exactly why the threshold shouldn't be rescaled (candidate
  explanations belong with the corrected run, not repeated here).
