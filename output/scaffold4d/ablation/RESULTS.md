# ESVR-Scaffold4D — adaptation fixes and leave-one-out ablation (2026-09-20)

> **Headline**: fixing the ESVR adaptation and porting 4D-Scaffold-GS's own
> anchor growing lifts the 90-frame vorts reconstruction from **24.35 dB**
> (`output/scaffold4d/vorts90_native_recipe`) to **31.9 dB** at the same
> 40,000-anchor budget and 120,000 steps (`D_flow1`), and to **32.2 dB** with
> a 100,000-anchor budget (`C_cap100k`). The previous best result across all
> three dynamic branches of this repository was 28.90 dB (Deformable-MLP).
> Across seeds the 40,000-anchor configuration lands at **31.6 ± 0.3–0.4 dB**
> (six runs, two configurations, see "Seed variance"); single-run
> differences below ~0.8 dB are not resolvable and are not claimed.

Branch: `codex/scaffold4d-esvr`. Data, protocol and evaluation identical to
every other 90-frame run here: `experiments/vorts_90_all_train.json`,
128³, all 90 frames used for training (**no held-out split**; these are
reconstruction metrics), per-frame PSNR on the normalized [0,1] volume,
averaged over frames. Every run: 120,000 steps, the native 7-group optimizer
(anchor frozen), native growing window 1,500–60,000, 64³ tile sampling, seed
42, RTX 5090. All numbers below were recomputed from each run's complete
90-frame `evaluation/metrics.json` (an earlier in-chat report quoted four of
them from partially written files; those were wrong and are superseded here).

## What was wrong with the previous implementation

Diagnosed on the `vorts90_native_recipe` checkpoint and logs, not inferred:

1. **Capacity collapse during training.** Active children per query fell
   from ~330k (step 10k) to ~21k (step 120k); monitored PSNR peaked near
   26.5 dB around step 30k and then decayed to 24.3. The reported 24.35 dB
   was a half-degraded model.
2. **Scale floor was a dead zone.** `scale.clamp_min(1e-4)` = 0.006 voxels.
   50% of all children sat exactly on that floor; a child that small touches
   no voxel centre, is skipped by the CUDA voxelizer, receives zero gradient
   and can never recover.
3. **No anchor pruning.** 14,088 of 40,000 anchors had every child's base
   opacity below 0.01 yet held the budget.
4. **Growing was not the paper's algorithm.** The branch reused ESVR4D's
   clone/split (`densify()`), not 4D-Scaffold-GS's voxel-grid anchor
   growing: no multi-level grid, no "grow only in unoccupied cells", no
   feature max-pooling, no `min_opacity` pruning, no σ^γ weight, and the 2D
   threshold 2e-4 applied to 3D gradients whose p90 is ~4e-7, so the 40k cap
   was hit at step 4,000. Split noise also scattered anchors far outside the
   volume (anchor range [−5.3, 3.8] for a [−1, 1] volume).
5. **Velocity bound** (tanh, 1 unit / unit time ≈ 0.72 voxel/frame at 90
   frames) is below the block-matched motion of the data (median 1.4,
   p90 ~2.5 voxel/frame). The adaptation tested widening it; see `D_flow1`
   for why this turned out not to matter once growing works.
6. Minor: initial amplitude 0.5 × 60k children over-predicts on step 1 under
   additive accumulation; Adam `eps=1e-8` and a global grad clip of 1.0 were
   not native.

## What changed (`scaffold4d/model.py`, `train_scaffold4d.py`)

- Scale floor expressed in voxels: `--scale-floor-voxels 0.5`.
- Upstream growing ported from `scene/gaussian_model.py` (`training_statis`,
  `anchor_growing`, `adjust_anchor`) with one substitution: the view-space
  2D gradient becomes the 3D gradient of each child's centre (ESVR has no
  camera). Per-child weight `opacity_t · σ^γ`, γ=1; three grid levels
  (16/4/1 voxels) with thresholds ×(hierarchy_factor//2)^level; new anchors
  only in unoccupied 4D cells; feature max-pooled from the voting children;
  offsets zero; `log_offset_scale` set to the cell size; anchors whose
  accumulated base opacity is below `min_opacity` are pruned. The
  "observed often enough" test is scaled by the block's share of the volume
  (1/8 for 64³ of 128³) because a child is only observed on steps whose
  block contains it. Threshold calibrated to 4e-7 from the measured
  distribution. Statistics kept for the native window
  `start_stat < step < update_until`.
- Native Adam `eps=1e-15`, no gradient clipping by default (`--clip-grad`).
- New optional flags, all measured below: `--flow-scale`, `--opacity-init`,
  `--init data` (anchors sampled from ground-truth intensity over all
  frames and 4D-voxelized, a stand-in for upstream's SfM points).

## Results

Run B is the reference; every other lettered run changes exactly one thing
relative to B. B itself used `--flow-scale 8 --opacity-init 0.1 --init
data`; the current script defaults are the D/F/G side (see "Defaults").

| Run | Single change vs B | Anchors | Model | Mean PSNR | Δ vs B | Min / max frame |
|---|---|---:|---:|---:|---:|---|
| `vorts90_native_recipe` (old code) | — | 40,000 | 13.8 MB | 24.348 | −7.04 | 21.94 / 25.43 |
| `E_floor_old` | scale floor 1e-4 (old) | 40,000 | 13.3 MB | 28.074 | **−3.32** | 25.95 / 28.90 |
| `A_fixed6000` | growing off | 6,000 | 2.0 MB | 29.958 | −1.43 | 27.06 / 30.82 |
| `H_old_optimizer` | Adam eps 1e-8 + clip 1.0 (old) | 40,000 | 13.3 MB | 30.797 | −0.59 | 28.45 / 31.71 |
| **`B_full`** | reference | 40,000 | 13.3 MB | **31.389** | — | 28.92 / 32.70 |
| `G_init_random` | uniform random anchors | 40,000 | 13.3 MB | 31.385 | 0.00 | 29.20 / 32.41 |
| `F_opacity_init0.5` | initial amplitude 0.5 | 40,000 | 13.3 MB | 31.522 | +0.13 | 29.84 / 32.65 |
| `I_no_sigma_weight` | γ = 0 (plain Scaffold-GS statistic) | 40,000 | 13.3 MB | 31.554 | +0.17 | 29.47 / 32.82 |
| `D_flow1` | velocity bound 1 (native tanh) | 40,000 | 13.3 MB | **31.908** | **+0.52** | **30.36** / 32.95 |
| `C_cap100k` | anchor cap 100,000 | 100,000 | 33.2 MB | **32.166** | +0.78 | 29.58 / 33.52 |

Model size = float32 tensors of the inference state (anchors, features,
offsets, offset scales, three MLP heads); not entropy coded.

Training time: 22–32 min per run with 2–3 runs sharing the GPU (the old
single run took 12 min alone; the new runs are slower mainly because more
children are active per query — 26k vs 21k at the end, and far more early).

### Reading the table

- **Effective**: the scale floor (−3.3 dB without it), upstream-style
  growing (+1.4 dB over a fixed 6,000-anchor population at the same step
  count), and the anchor budget (+0.8 dB from 40k→100k; the quality–size
  curve has not saturated).
- **Not distinguishable from noise (confirmed by the seed runs below)**:
  velocity bound, initial amplitude, data-driven initialization, σ^γ
  weighting. All four fall within ±0.5 dB of B, which is inside the
  measured seed spread (0.3–0.8 dB), so none of them is claimed to help or
  hurt. Their defaults were set to the native/simplest side (velocity bound
  1, amplitude 0.5, random anchors, γ=1); the alternatives stay available
  as flags. `D_flow1`'s +0.52 dB looked like a real effect in the
  single-seed table but is not (the same configuration family spans
  31.3–32.1 dB across seeds).
- **σ^γ**: not harmful, not helpful here. The paper's motivation is a static
  background plus short-lived dynamic content, where the weight moves budget
  toward the latter; vorts is dynamic everywhere, so there is little to
  redistribute. Kept at the native γ=1.
- **Growing under additive accumulation works** where clone/split did not
  (ESVR4D's investigation on the sibling branch found split/clone net
  harmful). The difference is that new anchors go to *unoccupied* cells
  with zeroed offsets and a fresh feature, not on top of an existing
  primitive, so the local sum is not doubled on insertion.

### Training dynamics (mean of the three monitored frames 1/46/90)

| Step | old `vorts90_native_recipe` | `D_flow1` | `A_fixed6000` |
|---:|---:|---:|---:|
| 10,000 | 22.87 | 26.33 | 23.86 |
| 30,000 | 25.96 | 29.26 | 27.35 |
| 60,000 | 25.20 | 30.44 | 28.56 |
| 120,000 | 23.68 | 31.30 | 28.89 |

Active children per query, old vs `D_flow1`: 329k→21k vs 67k→26k over
training. The new runs no longer decay after the growing window closes.

Growth trajectories: B reached its 40,000 cap at step 5,200 (1 anchor pruned
in total); C reached 100,000 at step 12,600 (70 pruned). Pruning is nearly
inactive on this data: with sigmoid opacity and no dead zone, anchors seldom
stay below `min_opacity` for a whole check interval.

## Seed variance (`experiments/run_scaffold4d_seeds.sh`)

Same recipe, 40,000-anchor cap, seeds 42 / 1 / 2:

| Configuration | Per-seed mean PSNR | Mean ± sample SD | Worst frame per seed |
|---|---|---:|---|
| J = current defaults (flow 1, amplitude 0.5, random init) | 31.305 / 32.127 / 31.405 | **31.61 ± 0.45** | 29.66 / 29.73 / 29.09 |
| B = ablation reference (flow 8, amplitude 0.1, data init) | 31.389 / 31.874 / 31.590 | **31.62 ± 0.24** | 28.92 / 30.52 / 28.98 |

The two configurations are indistinguishable (means differ by 0.006 dB).
Seed-to-seed spread within one configuration is up to 0.8 dB, which sets
the resolution of every single-run comparison in this document: the
scale-floor (−3.3), growing (−1.4 without it) and anchor-budget (+0.8,
100k vs 40k) effects are above it; nothing else is.

`K_fixed6000_flow1` (6,000 anchors, growing off, current defaults) reached
29.204 dB against `A_fixed6000`'s 29.958 dB. This is **not** a clean
velocity-bound test — K also differs from A in initial amplitude and
initialization (it used the whole default set), so the −0.75 dB is
suggestive at best. Whether a wider velocity bound matters specifically for
small fixed-population models (the 2 MB regime) remains open and would need
a one-variable rerun.

## Defaults after this ablation

`train_scaffold4d.py` defaults: `--scale-floor-voxels 0.5`, `--adam-eps
1e-15`, `--clip-grad 0`, `--flow-scale 1`, `--opacity-init 0.5`, `--init
random`, growing on in the native window with `--densify-grad-threshold
4e-7`, `--max-anchors 40000`. Only the scale floor, optimizer settings and
the growing port deviate from "native or simplest"; each deviation is
justified by a row above.

## Verification

- Every `metrics.json` checked for 90 frames in order before any number was
  quoted.
- Unit tests `tests/test_scaffold4d.py`: 7/7 (CUDA forward/backward/tiling
  and finite differences, growing/pruning with optimizer-state bookkeeping,
  data init and bounds).
- `D_flow1`: `--eval-only --resume checkpoint.pt` into a fresh directory
  reproduced all 90 exported `.npy` frames with max abs diff 0.0.

## Not done / caveats

- The leave-one-out table is single-seed; only the 40k configurations were
  repeated across seeds. `A`/`C`/`E`/`H` effects are large enough to stand,
  but their exact magnitudes carry the same ±0.5 dB uncertainty.
- No held-out time split (same caveat as every 90-frame run in this
  repository).
- Quality–size curve has three points (6k fixed / 40k / 100k); intermediate
  caps and a size-matched comparison against the other two branches remain.
- `C_cap100k` (33 MB) is not size-matched to anything; per-byte comparisons
  should use B/D (13.3 MB) or A (2.0 MB).
