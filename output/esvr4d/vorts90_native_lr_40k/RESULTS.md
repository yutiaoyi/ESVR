# ESVR4D — native LR groups, step-matched to the conflated-LR baseline

> **Resolves the open question from `vorts90_native_lr/RESULTS.md`:** does the
> native-LR-structure fix genuinely hurt ESVR4D, or was the earlier 28.77→25.09 dB
> drop partly an artifact of also shrinking `--steps` from 40,000 to 30,000
> alongside the LR change? **Answer: the drop is real, not a step-count
> artifact** — but step count was hiding part of the earlier headline number.
> With steps held fixed at 40,000 (identical to `vorts90_fixed_points`), the
> native LR structure still lands at **25.68 dB**, a genuine **−3.09 dB** from
> the conflated-LR run, not the **−3.68 dB** the confounded comparison implied.

Completed 2026-09-13. Branch: `codex/4dgs-esvr`.

## Why this run exists

`output/esvr4d/vorts90_native_lr/RESULTS.md` applied the same native-LR-group
fix used successfully on deformable-MLP
(`output/dynamic/vorts90_native_canonical_lr/RESULTS.md`, +2.9 dB there), but
its own `--steps` argparse default had *also* been changed to ESVR4D's native
30,000 alongside the LR change — an unintended second variable. That run
measured 25.09 dB against the `vorts90_fixed_points` baseline's 28.77 dB at
40,000 steps: two different training budgets, not a clean ablation. This run
repeats the exact native-LR-group configuration with `--steps 40000` restored,
isolating the LR-structure change as the only variable.

## Configuration

| Item | This run | `vorts90_native_lr` (confounded) | `vorts90_fixed_points` (baseline) |
|---|---|---|---|
| xyz LR | 0.00016 → 0.0000016 (native, decaying) | same | 0.005 (shared, constant) |
| time LR | 0.00016 (native, constant) | same | 0.005 (shared with xyz) |
| scale LR | 0.005 (native, constant) | same | 0.005 (shared) |
| opacity LR | 0.05 (native, constant) | same | 0.005 (shared) |
| rotation LR | 0.001 (native, constant) | same | 0.005 (shared) |
| Steps | **40,000 (matched to baseline)** | 30,000 | 40,000 |
| Points / densify | 30,000 fixed, no densification (all three runs) | same | same |
| Seed | 42 | 42 | 42 |

## Final full-volume metrics (all 90 training frames)

| Metric | This run (native LR, 40k steps) | `vorts90_native_lr` (native LR, 30k steps) | `vorts90_fixed_points` (conflated LR, 40k steps) |
|---|---:|---:|---:|
| Per-frame PSNR, arithmetic mean | **25.684 dB** | 25.091 dB | 28.772 dB |
| Per-frame PSNR, median | 25.935 dB | — | 28.995 dB |
| Lowest / highest per-frame PSNR | 21.720 / 26.872 dB | 21.911 / 26.986 dB | 25.398 / 30.718 dB |
| Frame 1 / 46 / 90 PSNR | 25.724 / 26.280 / 21.720 dB | — | 28.582 / 29.300 / 25.398 dB |

`monitor.jsonl` shows PSNR at step 40,000 essentially flat versus step 30,000
(sampled frames 0/46/90 moved by well under 1 dB between the two checkpoints),
so the extra 10,000 steps were not the reason `vorts90_native_lr` scored lower
than this run — the two runs' results are close (25.09 vs. 25.68 dB) and the
small residual gap is consistent with simply finishing training rather than
a meaningfully different fit.

## Interpretation

The step-matched result confirms the effect first suggested by
`vorts90_native_lr`, with the confound removed: giving ESVR4D's `xyz` group
its own conservative native learning rate (0.00016→0.0000016, vs. the
previous shared 0.005 — roughly 30x smaller at the start of training and
1000x smaller by the end) measurably limits how far primitives can reposition
during training, and ESVR4D's motion representation depends on `xyz` moving
directly (it conditions a 4D covariance on `xyz`'s deviation from a per-
primitive mean, unlike deformable-MLP where motion is carried by a separate
network). This is consistent with, and now verified independent of, the
step-count confound in the earlier run — but it remains a single ablation
(one seed, no isolation of *which* of the five regrouped LRs — xyz vs. time
vs. scale vs. opacity vs. rotation — drives the drop), the same caveat already
on record for the deformable-MLP fix's own interpretation.

Net conclusion for the project: enforcing each method's own paper-native LR
grouping is a **real structural correctness fix**, not a tuning nicety — it
was required so each branch's hyperparameters actually mean what the
originating paper intended. It happens to *improve* results for deformable-
MLP and *worsen* them for ESVR4D under this codebase's specific rendering
model (ESVR's additive, non-alpha-composited voxelizer, not either paper's
own renderer) and this specific real dataset. Reporting the native-fixed
numbers for all three methods is the fair, apples-to-apples way to compare
them; it does not mean the native LR is "wrong" for ESVR4D outside this
context — a follow-up native-LR-but-more-densification-friendly variant
would be a reasonable next step, not yet attempted.

## Verification

- Training exited cleanly (code 0) at step 40,000.
- `--eval-only --resume checkpoint.pt` into a fresh directory reproduced all
  90 exported `.npy` frames with **max abs diff 0.0**.
- Mean PSNR (25.684 dB) recomputed directly from `evaluation/metrics.json`
  independent of any training-time logging.

## Not done

- No isolation of which individual LR group (xyz/time/scale/opacity/rotation)
  drives the drop versus the others — a five-way ablation, not yet run.
- No held-out time split (same caveat as every other 90-frame run in this
  repository).
- Single seed (42).
- Densification not attempted for this configuration (same open question
  noted for every other real-data run of this method).
