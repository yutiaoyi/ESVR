# ESVR-Scaffold4D — matched-configuration comparison run (superseded)

> **Superseded by [output/scaffold4d/vorts90_native_recipe/RESULTS.md](../vorts90_native_recipe/RESULTS.md)**,
> which the user asked for instead: run the paper's own hyperparameters
> rather than aligning to the other two branches. Kept for the record, with
> the caveats below made explicit rather than silently dropped.

Completed 2026-09-12. Branch: `codex/scaffold4d-esvr`.

## Configuration — intended to match ESVR4D/deformable-MLP, imperfectly

| Item | Value used | Intent | What it actually was |
|---|---|---|---|
| Anchors × children | 6,000 × 5 = 30,000 | Match ESVR4D/deformable-MLP's 30,000-primitive budget | Achieved, but `children=5` is not the paper's own default (`n_offsets=10`) — a compromise between total-budget alignment and native per-anchor ratio, not purely either |
| `lr` (anchor+offset+scale, one group) | 0.005 | "Match ESVR4D's position LR" | Not a clean analogy: natively anchor position (frozen, lr=0) and offset (lr 0.01→0.0001) are different parameters upstream keeps separate; this run's single shared group conflates them |
| `mlp-lr` (all three heads, one group) | 0.002 | Match deformable-MLP's deform-net LR | Natively each head (opacity/shape/flow) has its own distinct LR (0.002→0.00002, 0.004 constant, 0.001 constant) |
| `feat-lr` | 0.005 | (no stated intent — an oversight) | Native value is 0.0075; 0.005 was not verified against the paper before this run |
| Steps | 40,000 | Match the other two branches | Native is 120,000 |
| Densification | Off | Match the other two branches' disabled-densification comparisons | Native has densification on for most of training |

## Final full-volume metrics (all 90 training frames)

| Metric | Value |
|---|---:|
| Per-frame PSNR, arithmetic mean | 25.033 dB |
| Per-frame PSNR, median | 25.172 dB |
| Lowest / highest per-frame PSNR | 22.119 / 27.105 dB |
| Frame 1 / 46 / 90 PSNR | 24.615 / 25.359 / 22.119 dB |
| Training wall time | 197.1 s for 40,000 steps |

## Verification

- Training exited cleanly (code 0); `active` children stayed in the
  26,000–28,000 range out of 30,000 possible, i.e. genuinely temporally
  selective, not collapsed.
- Not independently reload-verified with the same rigor as the native run
  (superseded before that pass was done).

## Why this comparison isn't reused for the final cross-method table

Per the discussion that led to the native-recipe run: config alignment
across three structurally different methods conflates "which motion
representation is better" with "which method's own tuned recipe got
overridden the least." `vorts90_native_recipe`'s own comparison table
carries the equivalent numbers for the other two branches' *own* native or
near-native recipes instead.
