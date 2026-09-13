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
| `gaussian_renderer/__init__.py: generate_neural_gaussians` | Exact formulas (this is the part that matters most, verified by direct reading, not the paper prose — see "Mathematical representation" below) | Reimplemented verbatim in `scaffold4d/model.py`; one legitimate simplification: the anchor's own 8-column log-scale had a column (index 7) never read anywhere in the generation function — dropped to a clean 7 columns (3 spatial-offset-scale + 1 temporal-offset-scale + 3 final-ellipsoid-scale) |
| `scene/gaussian_model.py: self._rotation`, `self._opacity` (per-anchor, `requires_grad_(False)`) | Created but **never read** in `generate_neural_gaussians` — appear to be vestigial fields inherited from the static-3DGS codebase this was built on | Omitted entirely; not used by upstream's own forward pass either |
| `scripts/train_n3dv.sh` (the paper's actual reported configuration) | Enables `--temporal_opacity ours --use_flow --sigma_denom_weight`; `arguments/__init__.py`'s bare defaults (`temporal_opacity='4dgs'`, `use_flow=False`, `sigma_denom_weight=False`) are a weaker ablation, not what the paper reports | Implemented **only** the flags actually used for the paper's reported numbers: velocity-based motion, the generalized/super-Gaussian temporal opacity gate, sigma-weighted dynamic-aware growing statistics |
| `scene/gaussian_model.py: training_statis`, `anchor_growing`, `adjust_anchor` | Hierarchical multi-depth (`update_depth`, default 3) voxel-grid growing with feature max-pooling into new cells, gradient statistic reweighted by `opacity_t · (σ·λ)^γ + weight_base` when "dynamic-aware" is active | Reused ESVR4D's already-built, already-tested split/clone/prune population-replacement machinery (`esvr4d/model.py`'s `densify()`/`replace_population()`, Adam-moment-preserving) rather than porting the multi-depth voxel-hash algorithm verbatim; the *statistic* being thresholded is this paper's `opacity_t`-weighted gradient, not ESVR4D's own xyz/time-gradient statistic — see "Densification" below for what was and wasn't ported, and why |
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

**Not ported as literally implemented**: the multi-depth (`update_depth`
levels), voxel-hash-deduplicated, feature-max-pooling anchor-growing
algorithm itself. Reimplementing that exactly, on top of a representation
with no camera/2D-gradient signal to begin with, is a large amount of
additional machinery for uncertain benefit given `codex/4dgs-esvr`'s own
densification investigation (`docs/ESVR4D.md`) already found that
**any** split/clone-style growing interacts badly with ESVR's *additive*
(not alpha-composited) accumulation unless compensated, and even a partial
fix did not fully resolve it. Instead, this branch reuses ESVR4D's already
built, already tested, opacity-halving-compensated split/clone/prune
population-replacement code (`esvr4d/model.py`'s `densify()`), substituting
this method's `opacity_t`-weighted gradient statistic for ESVR4D's own
xyz/time-gradient statistic as the growing trigger. Consistent with the
other two branches, **densification is validated separately and disabled by
default for the first real-data training run** until it is confirmed not to
hurt — see the training results doc once that run exists.

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
verified bit-exact. Densification did not trigger at the paper's default
gradient threshold on this small easy problem (never recalibrated against
real observed gradients, the same kind of miscalibration `esvr4d`'s own
densification investigation ran into) — left disabled/uninvestigated
pending a real-data run. Real 90-frame `vorts` training not yet attempted.
