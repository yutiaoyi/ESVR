# 90 independent per-frame ESVR baselines — vs. shared dynamic representations

Answers the question "what does the shared dynamic representation (deformable-MLP
or ESVR4D) buy you over just training 90 independent single-frame ESVR models?"
Companion to [output/esvr4d/vorts90_fixed_points/RESULTS.md](../esvr4d/vorts90_fixed_points/RESULTS.md)
and [output/dynamic/vorts90_matched_comparison/RESULTS.md](../dynamic/vorts90_matched_comparison/RESULTS.md).

Completed 2026-09-12. Branch: `codex/4dgs-esvr`, in the CUDA-fixed copy.

## Method

Reuses `train_dynamic.py`'s own single-frame path (`--allow-single-frame`,
which forces `deformation=False` for the whole run since there is only one
training time) rather than the separate legacy `train_chunk.py` pipeline.
Rationale: `train_dynamic.py`'s canonical representation, loss, and LR
schedule are the exact same code the two shared/dynamic runs use — this
isolates the one variable in question (shared-across-time vs. independent-
per-frame parameters) instead of also changing the training code,
hyperparameters, and densification schedule at the same time. (A quick
attempt to reuse `train_chunk.py`'s existing tuned config for this instead
diverged and hit an out-of-memory error under the CUDA-fixed kernel — its
`position_lr` was evidently tuned against the pre-fix, artificially-suppressed
position gradient; not investigated further, out of scope here.)

Per frame: 30,000 canonical primitives (fixed, no densification — same
representation, same point budget as the two shared runs, just not shared
across time), 3,000 steps, 64³ tile-sampling, `--lr 0.005 --lambda-ssim 0.2`
— identical hyperparameters to the shared runs' canonical-parameter group.
90 independent manifests (one frame each, `split: train`), one process per
frame, sequential.

## Results

| Method | Mean PSNR | Median | Min / Max | Total wall time | Model storage (no optimizer) |
|---|---:|---:|---:|---:|---:|
| **90 independent ESVR models** | **38.862 dB** | 38.882 dB | 38.449 / 39.124 dB | 466.1 s (90 seq. runs, 5.18 s/frame avg) | 301.1 MB (2,700,000 primitives total) |
| Deformable-MLP (shared, matched config) | 25.958 dB | 26.049 dB | 23.384 / 26.828 dB | 1023.2 s (one joint run) | 3.35 MB |
| ESVR4D (shared, matched config) | 28.772 dB | 28.995 dB | 25.398 / 30.718 dB | 178.9 s (one joint run) | 2.40 MB |

Storage compression vs. independent baseline: **~90x** (deformable-MLP),
**~125x** (ESVR4D). Quality cost of that compression: **~13 dB** (deformable-MLP),
**~10 dB** (ESVR4D) mean PSNR below the independent-per-frame ceiling.

All 90 independent runs' PSNR fell in a 0.7 dB band (38.45–39.12 dB) — no
failures, no outliers — consistent with each frame getting its own dedicated
30,000-primitive budget on visually similar turbulence data, so this is a
fairly tight, reliable ceiling estimate, not a lucky/unlucky single sample.

## What this does and doesn't show

**Shows**: under this protocol (all 90 frames used for training, evaluated on
those same 90 frames), the value of the shared/dynamic representation is a
**compression argument** — two orders of magnitude less storage for a
meaningfully lower but still workable reconstruction fidelity — not evidence
that either dynamic method "learned" a physically meaningful motion field
better than the other, or at all.

**Does not show**: whether the deformation MLP or the 4D covariance actually
captures a smooth, generalizable motion representation, vs. exploiting its
temporal encoding capacity to fit 90 known constraints independently with no
meaningful behavior between them (a real possibility with dense-in-time
supervision and no held-out test — see conversation). The historical
synthetic 5-frame smoke test (`experiments/make_dynamic_synthetic.py`, held
out t=0.25/0.75) is the only place in this repository where that distinction
has actually been tested, for either method, and not on real data. A held-out
split of the real 90-frame sequence is the natural next step and has not
been run.

## Caveats

- Independent-baseline budget (30,000 points / 3,000 steps) was chosen to
  match the shared runs' point budget and to plateau within a practical time
  budget (loss visibly flattening by step ~2,750–3,000 in a preliminary
  single-frame check), not from a convergence study; the true single-frame
  ceiling with more steps/points could be somewhat higher (the historical
  `output/experiments/vorts0001_baseline/RESULTS.md` reached 41.77 dB with
  21,404 densified primitives over 22,400 iterations — but under the
  *pre-fix* CUDA kernel, and reusing its exact config now diverges; not
  reconciled here).
- Single seed (42) throughout.
- "Model storage" excludes optimizer state (Adam moments) for all three
  methods, consistent with how the other two RESULTS.md files report it;
  optimizer-inclusive checkpoint size for the 90 independent models is
  540.9 MB total.
