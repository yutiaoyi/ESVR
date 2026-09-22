# ESVR-Scaffold4D — first implementation

## Source and scope

Official source: https://github.com/raikuma/4D-Scaffold-GS

Locally cloned to `.tools/4d-scaffold-gs` (ignored, no nested repository
committed), commit `e96bb5dd919213f43e4ef13a112d7023b566796f` (default
branch). To recover the source:

```bash
git clone https://github.com/raikuma/4D-Scaffold-GS.git .tools/4d-scaffold-gs
git -C .tools/4d-scaffold-gs checkout e96bb5dd919213f43e4ef13a112d7023b566796f
```

Paper: Cho et al., *4D Scaffold Gaussian Splatting with Dynamic-Aware Anchor
Growing for Efficient and High-Fidelity Dynamic Scene Reconstruction*,
[arXiv:2411.17044](https://arxiv.org/pdf/2411.17044).

This is a third, independent scalar-volume adaptation of a dynamic-scene
Gaussian-splatting method, alongside the canonical + deformation-MLP
approach (`codex/deformable-esvr`) and the native 4D-covariance approach
(`codex/4dgs-esvr`, `esvr4d/`). All three are kept separate, not merged, so
they can be compared. This branch (`codex/scaffold4d-esvr`) starts from the
same base commit as the other two and cherry-picks the CUDA voxelizer
position-gradient fix from `codex/4dgs-esvr` (see that branch's
`docs/ESVR4D.md` for the bug and its derivation) — this method's motion
model depends on position gradients just as directly as ESVR4D's does, so
training would be expected to hit the same suppressed-gradient failure mode
without the fix.

## What makes this method structurally different from the other two

Both prior branches move/deform *individual* primitives directly (a
displacement MLP per primitive, or a native 4D covariance per primitive).
4D-Scaffold-GS instead uses **Scaffold-GS's anchor/offset structure**: a
sparse set of *anchors* (points on an adaptive spatiotemporal voxel grid),
each holding one learned feature vector, from which small shared MLPs
*generate* several child ellipsoids per anchor on every forward pass. Only
the anchors' own parameters and the MLP weights are optimized; child
ellipsoid parameters are never directly learned tensors, always MLP outputs.
Motion here is not a general nonlinear deformation (branch 1) or a full 4D
covariance (branch 2) — it is deliberately simple: **each child ellipsoid
has a "home" time and a constant velocity**, i.e. `position(t) = home
position + velocity · (home time − query time)`, gated on/off by a temporal
opacity window centered at its home time. Scale and rotation do **not**
vary with time in the official method at all.

## Source-reading map and decisions

| Upstream code | Behavior verified | Integration decision |
|---|---|---|
| `scene/gaussian_model.py: create_from_pcd` | Anchor position is 4D: `anchor[:, :3]` = space, `anchor[:, 3]` = the anchor's own reference time (from the point cloud's per-point timestamps, or `U(0, max_init_t)` if none given) | Same: `anchor` is `(N, 4)`, xyz + t |
| `scene/gaussian_model.py.__init__` (MLP definitions) | `mlp_opacity`: feat→K·1, Tanh, masked `>0`; `mlp_cov`: feat→K·8 (3 scale-gate + 1 temporal σ-pre-ELU + 4 quaternion); `mlp_color`: feat+viewdir+dist+appearance→K·3, Sigmoid; `mlp_flow`: feat→K·3, Tanh (velocity) | `mlp_opacity` and `mlp_cov` kept (renamed `heads.opacity`, `heads.shape`); `mlp_flow` kept (`heads.flow`); **`mlp_color` and the view-adaptive feature-bank MLP dropped** — ESVR has no camera, no view direction, no RGB/SH, matching the "no RGB/view-dependent" decision already made on the other two branches; **opacity head's `Tanh` replaced with sigmoid** — this specific deviation was not a stylistic choice but a fix for a real training collapse, see below |
| `gaussian_renderer/__init__.py: generate_neural_gaussians` | Exact formulas (this is the part that matters most, verified by direct reading, not the paper prose — see "Mathematical representation" below) | Reimplemented following the upstream formulas in `scaffold4d/model.py` (see the audit table below for every deviation); one legitimate simplification: the anchor's own 8-column log-scale had a column (index 7) never read anywhere in the generation function — dropped to a clean 7 columns (3 spatial-offset-scale + 1 temporal-offset-scale + 3 final-ellipsoid-scale) |
| `scene/gaussian_model.py: self._rotation`, `self._opacity` (per-anchor, `requires_grad_(False)`) | Created but **never read** in `generate_neural_gaussians` — appear to be vestigial fields inherited from the static-3DGS codebase this was built on | Omitted entirely; not used by upstream's own forward pass either |
| `scripts/train_n3dv.sh` (the paper's actual reported configuration) | Enables `--temporal_opacity ours --use_flow --sigma_denom_weight`; `arguments/__init__.py`'s bare defaults (`temporal_opacity='4dgs'`, `use_flow=False`, `sigma_denom_weight=False`) are a weaker ablation, not what the paper reports | Implemented **only** the flags actually used for the paper's reported numbers: velocity-based motion, the generalized/super-Gaussian temporal opacity gate, sigma-weighted dynamic-aware growing statistics |
| `scene/gaussian_model.py: training_statis`, `anchor_growing`, `adjust_anchor` | Hierarchical multi-depth (`update_depth`, default 3) voxel-grid growing with feature max-pooling into new cells, gradient statistic reweighted by `opacity_t · (σ·λ)^γ + weight_base` when "dynamic-aware" is active; anchors pruned when their accumulated base opacity stays below `min_opacity` | **Ported** (`ScaffoldESVR.record_statistics` / `adjust_anchors`, 2026-09-20; deviations listed in the audit section below) with one substitution: the view-space 2D gradient becomes the 3D gradient of each child's centre. An earlier version reused ESVR4D's clone/split machinery instead; that was replaced after it was found to drive the training-time capacity collapse documented in `output/scaffold4d/ablation/RESULTS.md` — see "Densification" below |
| `train.py`: loss composition | `(1−λ_ssim)·L1 + λ_ssim·(1−SSIM) + 0.01·mean(scale.prod(dim=-1))` (a volume regularizer, encourages small ellipsoids), `λ_ssim=0.2` | Same three terms, reusing ESVR's existing L1/SSIM routine on 3D blocks exactly as the other two branches do |
| `gaussian_renderer/__init__.py: render` (2D rasterizer, alpha-compositing, SH evaluation) | Not used | ESVR's existing scalar voxelizer (`gaussian_renderer.query`) is used instead, exactly as in the other two branches — **still no camera, no 2D rendering, no novel-view-synthesis evaluation anywhere in this branch**; supervision remains direct 3D-voxel L1/SSIM against the ground-truth volumes, the same property confirmed and discussed at length for the other two branches |

## Mathematical representation

For anchor position `(x_a, t_a) ∈ R⁴`, anchor feature `f ∈ R^{feat}`, and
`k = 1..K` children per anchor with raw learnable offset `(δx_k, δt_k) ∈ R⁴`:

```text
(shape_k) = heads.shape(f)          # per anchor, K·8 outputs: [scale_gate(3), sigma_pre(1), quat(4)]
(op_k)    = heads.opacity(f)        # per anchor, K outputs, tanh
(flow_k)  = heads.flow(f)           # per anchor, K·3 outputs, tanh (velocity)

home_xyz_k = x_a + δx_k · scale_offset_xyz      # scale_offset_xyz: per-anchor learned magnitude (3,)
home_t_k   = t_a + δt_k · scale_offset_t        # scale_offset_t:   per-anchor learned magnitude (scalar)

scale_k    = scale_final_xyz · sigmoid(shape_k.scale_gate)   # time-independent
rotation_k = normalize(shape_k.quat)                          # time-independent
sigma_k    = ELU(shape_k.sigma_pre) + 1                        # > 0; larger sigma = narrower temporal window

xyz_k(t)      = home_xyz_k + flow_k · (home_t_k − t)           # constant-velocity motion
opacity_t_k(t) = exp(−|(home_t_k − t) · sigma_k|^β)            # β = hparam_beta, default 4 ("ours"/dynamic-aware gate)
opacity_k(t)   = max(op_k, 0) · opacity_t_k(t)                 # children with op_k <= 0 are masked out entirely

volume(t) = ESVR_voxelizer({xyz_k(t)}, {scale_k}, {rotation_k}, {opacity_k(t)} : opacity_k(t) above cutoff)
```

Anchors act like Scaffold-GS's sparse control points; children are never
stored, only ever regenerated from `f` on every query. A child's temporal
existence is a soft window in time around its own `home_t_k`, and its
position drifts linearly at constant velocity around that window — this is
deliberately simpler than either sibling branch's motion model, and is the
paper's own design choice (they call this "we found this baseline behavior
sufficient"), not a simplification introduced here.

## Densification

**Ported**: the *statistic* — this method's contribution over plain
Scaffold-GS is entirely in how the growing gradient is weighted:
`weight_k(t) = opacity_t_k(t) · (σ_k · λ_mult)^γ` (`γ = lambda_temporal_sigma`,
default 1; `λ_mult = lambda_sigma_multiply`, default 1), so a child that is
barely "on" at the currently-rendered time contributes almost nothing to the
growing statistic, and children with a narrow temporal window (large σ) are
upweighted relative to children that persist for a long time and would
otherwise dominate the accumulated-gradient statistic purely by being
visible more often. This directly targets short-lived, fast dynamic content
for extra capacity — the paper's stated motivation.

**Also ported (2026-09-20)**: the growing algorithm itself, following
`train.py → training_statis → adjust_anchor → anchor_growing`:

- Statistics (`record_statistics`): for every child visible in the query
  (`visibility_filter`, i.e. radius > 0), accumulate `‖∂L/∂xyz_child‖ ·
  opacity_t · σ^γ` and the weight itself (`offset_gradient_accum` /
  `offset_denom`); per anchor, accumulate the sum of its children's base
  opacities (`opacity_accum`) and a visit count. Kept only inside the native
  window `start_stat < step < update_until`.
- Growing (`adjust_anchors`, every `update_interval`): children whose
  averaged statistic exceeds `threshold · (hierarchy_factor//2)^level` and
  were observed often enough are voxelized on a 4D grid at three levels
  (`voxel_size · init_factor / hierarchy_factor^level`, temporal cell = one
  frame interval, the paper's setting); a new anchor is placed at the centre
  of every cell not already holding an anchor, with feature = max-pool of
  the voting children's anchor features, zero offsets, `log_offset_scale`
  = cell size; upstream's random 50%/25%/12.5% subsampling per level is
  kept (`--grow-all-candidates` disables it). Anchors whose accumulated base
  opacity is below `min_opacity` × visits are pruned. Adam moments are
  sliced/zero-padded exactly as before (`replace_population`).
- Two ESVR-specific calibrations, both recorded in `config.json`: the
  gradient threshold (native 2e-4 is a 2D screen-space magnitude; the 3D
  centre gradient of a per-voxel-mean loss has p90 ≈ 4e-7 here, so the
  default is 4e-7), and the "observed often enough" interval, scaled by the
  block's share of the volume because a child is only observed on steps
  whose 64³ block contains it (upstream renders full frames).

Why this works under additive accumulation where clone/split did not: new
anchors are created only in **unoccupied** cells with zeroed offsets and a
pooled feature, never on top of an existing primitive, so insertion does
not double any local sum. Measured effect: +1.4 dB over a fixed population
at equal step count, and no post-window decay (`output/scaffold4d/ablation/RESULTS.md`).

## Provenance map: what comes from ESVR, what from 4D-Scaffold-GS, what from neither (audit 2026-09-20)

Rule applied: the *primitive* and everything the CUDA voxelizer sees is
ESVR's (centre, covariance, one scalar intensity — no colour, no view
dependence, no alpha compositing); everything about how primitives are
*generated over time* is 4D-Scaffold-GS's; where 4D-Scaffold-GS relies on
something a scalar volume does not have (camera, screen space, alpha
compositing) the ESVR counterpart is used; anything that is in neither is
listed as such with its justification.

| Component | Ours | ESVR (`train_chunk.py`, `scene/gaussian_model.py`, `cuda_voxelizer`) | 4D-Scaffold-GS (`train.py`, `scene/gaussian_model.py`, `gaussian_renderer`) | Source used |
|---|---|---|---|---|
| Primitive | centre, 3 scales, quaternion, one scalar in (0,1) | same (`_xyz`, `_scaling`, `_rotation`, `_opacity`=intensity) | 3D Gaussian + RGB colour + opacity | **ESVR** |
| Kernel / accumulation | `alpha = I·σ(5(1−dᵀΣ⁻¹d))`, cutoff `dᵀΣ⁻¹d>2`, per-voxel **sum** | same (`forward.cu`, k=5, cutoff 1.42²) | alpha compositing of 2D-projected Gaussians | **ESVR** (unchanged kernel) |
| Supervision | random 64³ block of the GT volume at a random training frame, L1 + SSIM | random 64³ subvolume, L1 + SSIM (`lambda_dssim` 0.2) | full rendered frame vs image, L1 + SSIM (0.2) | both agree |
| Volume regulariser `0.01·mean(∏scale)` | yes (`--lambda-vol`) | none | yes | 4D-Scaffold-GS |
| Anchor = 4D point (xyz, t), frozen (lr 0) | yes | — | yes (`position_lr` 0) | 4D-Scaffold-GS |
| Anchor feature 32-d, K=10 children, 4D offsets × per-anchor offset scale | yes | — | yes | 4D-Scaffold-GS |
| Opacity / shape / velocity MLPs (2 layers, hidden 32) | yes | — | yes; colour MLP and feature bank dropped (no view) | 4D-Scaffold-GS minus view-dependent parts |
| Opacity head activation | **sigmoid** | intensity = sigmoid(logit) | tanh, children with output ≤0 masked out | ESVR's activation; deviation from upstream documented above (tanh+mask collapsed to zero children at step ~50 with zero-initialised features) |
| Temporal opacity `exp(−|(t−t_k)σ_k|^β)`, β=4 | yes | — | yes (`temporal_opacity ours`, `hparam_beta` 4) | 4D-Scaffold-GS |
| Linear motion `x_k + u_k (t_k − t)` | yes | — | yes (`use_flow`) | 4D-Scaffold-GS |
| Time-invariant scale / rotation per child | yes | — | yes | 4D-Scaffold-GS |
| Optimizer: 7 groups, Adam eps 1e-15, log-linear decay on offset / opacity-MLP | yes | 4 groups, eps 1e-15 | yes | 4D-Scaffold-GS (eps agrees with ESVR) |
| Densification statistic | mean over observations of ‖∂L/∂(child centre)‖ weighted by `opacity_t·σ^γ` | mean over observations of ‖∂L/∂(3D centre)‖ (`add_densification_stats`, 3D not 2D) | mean of ‖∂L/∂(2D screen position)‖ weighted by `opacity_t·σ^γ` | **ESVR's quantity** (3D centre gradient — there is no screen space) inside **4D-Scaffold-GS's weighting** |
| Where new capacity goes | new **anchors** at centres of unoccupied 4D grid cells (16/4/1 voxel levels), feature max-pooled from voting children | clone (small) / split (large) of the primitive itself | same as ours | 4D-Scaffold-GS |
| Grid sizes | finest cell = 1 data voxel, temporal cell = 1 frame interval | — | `voxel_size` 0.001 scene units, `t_grid_size` = frame interval | 4D-Scaffold-GS (spatial value re-expressed in voxels) |
| Growing threshold | 4e-7 | 2.5e-7 – 1e-8 (dataset-tuned, also a 3D-gradient threshold) | 2e-4 (2D pixels) | **calibrated** to the measured 3D-gradient distribution; same order as ESVR's own thresholds, which is a useful sanity check |
| "Observed enough" gate before a child may vote | `denom > interval·0.8·0.5`, interval scaled by the block's share of the volume | — | `offset_denom > update_interval·success_threshold·0.5` | 4D-Scaffold-GS; block scaling is ours (a child is only observed when its block is sampled) |
| Pruning | anchors whose children's summed base intensity stays `< min_opacity` (0.005) over a check window | intensity `< 1/255` (dark pruning), out of bbox, too large, global importance `|Ω|·I·FA` keep 66% | same as ours | 4D-Scaffold-GS (nearly inactive here, see audit findings) |
| Opacity reset | none | every `opacity_reset_interval` (ESVR relies on it heavily: every 64 iterations in the vorts recipe) | none | neither — not expressible with a shared opacity MLP |
| Scale floor 0.5 voxel on the child's final scale | yes | none needed (init at kNN distance ≈ 3–5 voxels, scale lr 5e-5 keeps it there) | none needed (2D rasteriser dilates every Gaussian by 0.3 px, so nothing is ever gradient-free) | **neither**; the ESVR-space analogue of 3DGS's pixel dilation — a child below ~0.45 voxel touches no voxel centre, is skipped by the kernel and can never recover. Measured: −3.3 dB without it |
| Temporal cutoff `opacity > 1e-3` for inclusion in a query | yes | kernel skips `alpha < 1/255` per voxel anyway | no hard gate | ours, efficiency only (below the kernel's own cutoff) |
| σ clamp [1e-2, `--sigma-max`], offset clamp ±10, log-offset-scale clamp | yes | — | none | ours, float32 guardrails (nonfinite-loss crashes documented above); σ clamp at 100 is **binding** for 13–27 % of children, see findings |
| Initial anchors | uniform random in the volume (default) or GT-intensity-sampled 4D points (`--init data`) | uniform random primitives, 20k–100k | SfM point cloud of frame 0, voxelised | neither exactly; `--init data` is the closest analogue of SfM points; measured difference 0.0 dB |
| Initial child amplitude | 0.5 (zero bias) | 0.1 | tanh(0) = 0 → masked | measured difference 0.13 dB, kept simplest |
| Velocity bound | tanh × `--flow-scale` (default 1) | — | tanh (1) | 4D-Scaffold-GS; measured ×8 = no gain under growing |
| Newborn anchor children | offsets 0 (stacked), scale = cell size, feature = max-pool of voters | new primitives from clone/split carry the parent's size and intensity, and `reset_opacity` limits over-brightness | offsets 0, scale = cell size, feature max-pool | 4D-Scaffold-GS verbatim — **this is where the audit found the current defect**, see below |

### What the two papers emphasise (for tuning reference)

4D-Scaffold-GS (paper + `arguments/__init__.py` + `scripts/train_n3dv.sh`):
K=10 (Table S2: 3/5/8/10 → 27.4/28.6/29.1/29.2 dB; larger K needs fewer
anchors), β=2β′ with the |·| removed (code `hparam_beta`=4; >8 unstable),
γ=1 (Fig. S3: larger γ moves anchors from static to dynamic regions), anchor
grid `voxel_size` is the quality–storage knob (Fig. 5), temporal grid = one
frame interval, feature 32, MLP 2×32, 120k iterations, λ_SSIM 0.2,
λ_vol 0.01, growing 1,500–60,000 every 100 with threshold 2e-4,
`update_init_factor` 16 / `update_hierachy_factor` 4 / `update_depth` 3,
`success_threshold` 0.8, `min_opacity` 0.005, LRs inherited from
Scaffold-GS, Adam eps 1e-15, no gradient clipping, initial points < 100k.

ESVR (paper + `arguments/__init__.py` + `experiments/run_vorts0001.sh`):
k=5 and cutoff √2 fixed (learnable k marginal), L1+SSIM with `lambda_dssim`
0.2, 100k primitives per chunk (20k for the 128³ vorts recipe), densify by
3D-centre gradient with dataset-tuned thresholds 1e-10–6.5e-9 (2.5e-7 for
vorts) and `percent_dense` 0.01 (0.008), opacity reset (every 64 scans for
vorts), dark pruning at 1/255, one global importance pruning keeping 66 %
by `|Ω|·I·FA`, tiny learning rates (position 8e-6→1e-6, scale 5e-5,
rotation 1e-5, intensity 7.5e-3→5e-4), Adam eps 1e-15, subvolume 64³.
Per frame, the vorts recipe converges to ~24k primitives of 2–6 voxels
(median geometric mean 3.4, min axis never below 0.5 voxel, anisotropy ~2),
intensities 0.02–0.14, 41.2 dB, 1.0 MB/frame.

### Audit findings (2026-09-20)

1. **CUDA voxelizer verified correct through the scaffold path.** Direct
   finite differences on the installed extension (`.venv` editable install
   of this tree, built 2026-09-12 from the fixed source): numeric/analytic
   position-gradient ratio 1.00 (16³) and 1.0–1.2 (128³ grid / 64³ block,
   the training configuration) for children ≥ 1 voxel; the original
   `ESVR/` extension gives 64 = 1/dVoxel² on the same inputs. Children
   below ~0.45 voxel touch no voxel centre and therefore have neither an
   analytic nor a meaningful numeric gradient — an earlier check that used
   such children produced a spurious 64× and was discarded.
   `tests/test_scaffold4d.py::CUDA` now asserts the ratio with a relative
   tolerance instead of an absolute one.
2. **Growing works, but its newborns mostly die.** Fixed-population runs
   (`A`, `K`) end with 80–89 % of children alive and useful (7–15 voxels,
   intensity ~0.04). Runs with growing end with only 36 % (`J`, 40k
   anchors) and 13 % (`C`, 100k anchors) useful children; the rest are
   below 1/255 intensity and/or pinned at the 0.5-voxel floor. Upstream's
   newborn recipe puts all K children on one point at the cell size: under
   additive accumulation that is a K-fold over-prediction at birth, and at
   the finest level the cell (1 voxel) times the gate (≈0.5) *is* the
   floor. Useful children per query at t=0.5: ~15k in `J` and ~14k in `C`
   — the extra 60k anchors of `C` added almost nothing useful, which is why
   the quality–size curve bends. Options under test: spread newborn
   offsets over the cell, start children at an ESVR-like 4-voxel scale,
   drop λ_vol (which shrinks invisible children onto the floor), floor 1.0.
3. **σ clamp binds**: 13–27 % of children sit at σ=100 (gate half-width
   0.8 frame). With the temporal grid at one frame this may limit how
   sharply a child can switch off; `--sigma-max` exposed for testing.
4. Pruning is nearly inactive (1–70 anchors per run) because sigmoid
   children rarely fall below `min_opacity` for a whole window; anchors with
   zero useful children are 32 % (`J`) / 48 % (`C`). An ESVR-style dark
   pruning on *useful children per anchor* would free that budget.
5. Anchors outside the volume: 1.4–2.5 % (grid cells straddling the
   boundary); harmless.

## What this branch is, stated precisely (audit 2026-09-22)

An ESVR scalar-volume **adaptation** of 4D-Scaffold-GS: the primitive, the
voxelizer and the supervision are ESVR's; the anchor/feature/child
structure, the linear motion, the generalized-Gaussian temporal gate, the
optimizer groups and the dynamic-aware anchor growing follow the upstream
code. It is not a line-for-line port. Every deviation, with its reason and
what was measured:

| Deviation from upstream | Reason | Measured |
|---|---|---|
| opacity head sigmoid instead of tanh + `>0` mask | tanh + zero-initialised features drove every child below zero within ~50 steps; sigmoid has no dead zone | required to train at all |
| upstream `min_opacity` anchor pruning nearly inactive | the sum of K sigmoid intensities rarely drops below 0.005 per visit; upstream's tanh yields exact zeros | 1–70 anchors pruned per run; `--prune-dead` (ESVR prune_zeros / dark rules) recycles them but does not change PSNR (`screen_fixed/RESULTS.md`) |
| 3D centre gradient replaces the 2D view-space gradient in the growing statistic | no camera; this is ESVR's own densification quantity | threshold recalibrated to the 3D gradient's p90 |
| "observed enough" interval scaled by the block's share of the volume | a child is only observed when its 64³ block is sampled | — |
| anchor cap with vote-ranked admission | upstream has no cap | 40k → 100k → 150k → 300k measured |
| finer grid levels always run (`--grow-upstream-level-skip` restores upstream's skip) | upstream runs finer levels only when a coarser level grew this round; not noticed until the 09-22 audit | not measured |
| anchor time initialised uniformly in [0,1] (upstream: all at t₀) | volumes have no SfM frame | `--init data` (GT-sampled 4D anchors) vs random: 0.0 dB |
| 0.5-voxel floor on child scale; σ ∈ [1e-2, 1e2]; offset ∈ [−10, 10] | float32/CUDA failure modes (NaN, zero-gradient sub-voxel children) | floor: −3.3 dB without it |
| temporal inclusion cutoff 1e-3 | efficiency; below the kernel's 1/255 | — |
| global volume regularizer on all time-active children (upstream: on rendered children) | mini-batch data term + global regularizer | visibility-gated variant 0.0 dB; removing it −0.6 dB |
| `--final-lr-scale 0.01` (ESVR recipe, **not the default**) | four upstream-constant LR groups oscillate ±0.7 dB late in training under block supervision | +5 dB at 150k anchors; default stays 1.0 so argparse defaults remain the paper's |
| per-frame optimisation budget ≈ 1/12 of static ESVR's | 90 frames share one run | static ceiling study, `output/experiments/static_ceiling/RESULTS.md` |

## Data contract

Reuses `dynamic_esvr/data.py`'s manifest format unchanged, identical to both
sibling branches — see `docs/DEFORMABLE_ESVR.md#data-contract`.

## License

Network design adapted from Woong Oh Cho et al.'s 4D-Scaffold-GS code
(itself built on Scaffold-GS and 3D Gaussian Splatting); upstream directs
users to the 3D-GS license for redistribution terms. The upstream repository
carries no separate LICENSE file of its own at the pinned commit; see
`third_party/` for what could be retained. Existing ESVR/Inria components
retain their original restrictions.

## A collapse mode found and fixed before any training worked at all

Upstream's opacity head is `Tanh`, with children masked into the query only
where the raw output is `> 0` — anything below zero is excluded and (since
excluded elements never enter the loss for that forward pass) receives
exactly zero gradient. Every anchor's feature starts at an identical zero
vector (matching upstream's own `create_from_pcd`, `anchors_feat =
torch.zeros(...)`), so every anchor produces an identical opacity on step 1,
and every anchor's gradient points the same direction — empirically, this
drove **every single child in a synthetic 256-anchor smoke test negative
within the first ~50 steps** (1024/1024 active at step 1, 0/1024 from step
50 onward), after which training could never recover: nothing was ever
included in a query again, so the loss froze at whatever an all-zero
prediction happens to score and stayed there for the rest of the run. This
is a real architectural hazard of combining a zero-gradient-below-zero gate
with a fully-shared, initially-identical starting point — it is plausible
upstream avoids it in practice via `position_lr_delay_mult`-style LR warmup
and/or the greater gradient diversity multiple camera views provide from the
first step, neither of which this ESVR adaptation had. Fixed by using
**sigmoid** instead of `Tanh`+hard-mask for the opacity head — never zero
gradient, matching the activation `dynamic_esvr`/`esvr4d` already use for
their own scalar opacity in this repository — and treating the temporal-
cutoff mask purely as a per-query efficiency cutoff (never a permanent gate,
since an anchor's shared feature can still move from its siblings' gradient
even while one specific child sits below cutoff).

## Status

Representation, motion model, and rendering implemented and tested
(`scaffold4d/model.py`, `tests/test_scaffold4d.py`, 6/6 passing including a
CUDA forward/backward/tiling and finite-difference check). A synthetic
5-frame smoke test (`experiments/make_dynamic_synthetic.py`, held out
t=0.25/0.75, 500 steps, 256 anchors) reached 42–45 dB PSNR on training times
and **24.8–25.9 dB on the held-out interpolated times** — this is the first
of the three dynamic branches in this repository to be smoke-tested with a
genuine train/test split rather than an all-frames-trained real-data run,
and it shows real (if imperfect) interpolation, plausibly because the
paper's constant-velocity motion model matches this specific synthetic
scene's actual motion (rigid translation) closely. Checkpoint reload
verified bit-exact.

## Two structural bugs found only once real data and native hyperparameters were used together

An earlier version of this branch matched its optimizer to the other two
branches' settings (single shared LR across anchor+offset+everything else,
40,000 steps, no densification) rather than 4D-Scaffold-GS's own. Once asked
to use the paper's own hyperparameters instead, two more real problems
surfaced that the matched-config runs and the synthetic smoke test had not
exercised:

1. **Anchor position and offset were wrongly lumped into one optimizer
   group.** Upstream's own `training_setup` gives anchor position its own
   group at `position_lr_init = position_lr_final = 0.0` — anchor position
   is frozen for the entire training, and only `offset` (its own group,
   `0.01→0.0001`) actually moves. `train_scaffold4d.py` originally put both
   under a single shared LR, letting the "coarse fixed grid" the anchors are
   supposed to be move freely — not a value that was off by some factor, a
   structure the paper deliberately keeps separate that got merged. Fixed
   by giving every one of the paper's parameter groups (anchor, offset,
   scale, feature, three MLP heads) its own optimizer group and its own
   `expon_lr`-scheduled learning rate, matching `scene/gaussian_model.py`'s
   `training_setup`/`update_learning_rate` (verified upstream's own
   `lr_delay_steps` is never actually set above 0 anywhere, so despite
   `*_delay_mult` being configured on every group, the warmup branch of
   their scheduler is dead code and never fires — the correct native
   behavior is plain log-linear decay, no warmup, confirmed by reading
   `update_learning_rate`'s call sites, not assumed from the docstring).

2. **The final per-child scale had no real floor.** `scale = offset_scale *
   sigmoid(shape_gate)`: `offset_scale` was floored by `stabilize()`, but
   `sigmoid` has no floor of its own — it can approach 0 as closely as the
   MLP output goes negative, so the *product* was still unbounded below.
   Real-90-frame training with the corrected native optimizer hit a
   nonfinite loss reproducibly, but at a **different step on every
   identically-seeded rerun** (1215, 1379, 1951, 1021, 1543, 278 across
   different attempts) — the signature of a float32 precision cascade
   through CUDA's nondeterministic atomics, not a fixed logic bug. Added
   direct diagnostics (`pred`/scale min-max at the crash) across 4 seeded
   reruns and found the same thing every time: `pred` itself was already
   NaN, and the per-child scale had collapsed to `~1e-9`–`~1e-10` in at
   least one axis just before it — degenerate enough to make that
   ellipsoid's covariance numerically singular in the CUDA voxelizer.
   Fixed with a direct floor on the product itself
   (`scale.clamp_min(1e-4)`), plus the same kind of numerical guardrail
   already used elsewhere in this repository for `sigma` (unclamped, raised
   to the 4th power downstream) and for the raw `offset` (unclamped,
   compounds with `flow*dt` with no natural bound). None of these change
   the loss or what is being optimized; they only prevent float32 failure,
   the same justification `esvr4d`'s `stabilize()` already uses. Verified
   fixed across 5 reseeded reruns without densification and 3 with
   densification enabled (5000 and 4000 steps respectively), all completing
   cleanly where every one of those same seeds reliably crashed before the
   fix.

Neither bug was visible in the CPU/GPU unit tests or the synthetic smoke
test — both are small, controlled problems that never drive the MLPs'
outputs anywhere near these extremes. This is a caution about this
representation specifically: shared-decoder architectures can develop
failure modes that only appear at the scale and duration of real training.

## Status (2026-09-20)

First native-recipe run: `output/scaffold4d/vorts90_native_recipe/RESULTS.md`,
24.35 dB — superseded. Inspecting its checkpoint showed the model had
degraded during training (active children 330k → 21k, PSNR peaking ~26.5 dB
at step 30k and falling to 24.3), with half of all children pinned on the
1e-4 scale floor (too small to touch any voxel centre, hence permanently
gradient-free) and 14k of 40k anchors fully transparent but never pruned.

Fixes and a nine-run leave-one-out ablation are documented in
`output/scaffold4d/ablation/RESULTS.md`. Current numbers at 120,000 steps:
**31.9 dB** with 40,000 anchors (13.3 MB), **32.2 dB** with 100,000 anchors
(33 MB), **30.0 dB** with 6,000 fixed anchors and no growing (2.0 MB). The
scale floor (now 0.5 voxels) and the growing port account for the gain;
velocity bound, initial amplitude, data-driven initialization and the σ^γ
weight were each measured and found not to matter beyond ~0.5 dB on this
data, so their defaults follow the native/simplest setting. Across three
seeds the 40,000-anchor configuration is **31.6 ± 0.3–0.4 dB** (seed spread
up to 0.8 dB, which bounds what any single-run comparison can claim).

Still open: held-out time split (never done on any branch); a size-matched
comparison against the other two branches; byte-exact reload check of the
new runs; RetimeGS/MoDec-GS-style multi-segment motion as a possible next
representation step.
