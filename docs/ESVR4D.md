# ESVR4D — native spacetime primitives, first implementation

## Source and scope

Official source: https://github.com/fudan-zvg/4d-gaussian-splatting

Locally cloned to `.tools/fudan-4dgs-shallow` (ignored, no nested repository
committed), branch `main`, commit `63725f21d4adc29669e565ae10e6b3ad6e0d1250`.
To recover the source:

```bash
git clone https://github.com/fudan-zvg/4d-gaussian-splatting.git .tools/fudan-4dgs-shallow
git -C .tools/fudan-4dgs-shallow checkout 63725f21d4adc29669e565ae10e6b3ad6e0d1250
```

Paper: Yin et al., *4D Gaussian Splatting for Real-Time Dynamic Scene
Rendering*, [arXiv:2310.10642](https://arxiv.org/pdf/2310.10642).

This is a scalar-volume adaptation of the paper's **native 4D representation**
(time-space covariance, motion via conditioning on `t`), not a reproduction of
the donor's RGB/camera benchmark. It is a separate representation from the
canonical + deformation-MLP approach on `codex/deformable-esvr`
([docs/DEFORMABLE_ESVR.md](DEFORMABLE_ESVR.md)); both are kept, not merged, so
they can be compared. Static ESVR's training, voxelizer and vorts data
adaptations are unchanged and reused via `dynamic_esvr/data.py`.

## Source-reading map and decisions

| Upstream code | Behavior verified | Integration decision |
|---|---|---|
| `scene/gaussian_model.py` (`gaussian_dim=4`) | 4D mean `(xyz, t)`; 4D scale; rotation as a pair of unit quaternions (`rotation_l`, `rotation_r`) combining into an `SO(4)` matrix; per-Gaussian learned `t` | Same parameterization: `xyz`, `time`, 4D `log_scale`, `left`/`right` quaternions, plus a scalar (not RGB/SH) opacity logit |
| `utils/general_utils.py: build_rotation_4d` | Conjugated-right, axis-reversed `SO(4)` construction (`lmat @ rmat`, then `.flip((-2,-1))`) | Reimplemented as `rotation4d`; `tests/test_esvr4d.py::test_rotation_and_upstream` execs the upstream function directly (AST-parsed, no CUDA deps) from the pinned commit and asserts bit-for-bit match against our own implementation |
| `scene/gaussian_model.py: get_covariance` (4D case) | `Cov4D = R S S^T R^T`; marginalize/condition on time via the Gaussian conditioning identity to get the 3D mean/covariance at a query time, and the temporal marginal as an opacity gate | Implemented in `condition()`: Schur complement `spatial = Cov_xyz - cross·cross^T/var_t`, `mean(t) = xyz + (cross/var_t)·(t - time)`, `marginal(t) = exp(-0.5·dt²/var_t)`; `tests/test_esvr4d.py::test_conditional_factorization_and_gradcheck` checks this factorization reproduces the full 4D joint density exactly (`joint == conditional · marginal`) |
| `scene/gaussian_model.py: add_densification_stats` / `densify_and_prune` | View-space 2D positional gradient accumulated per camera ray, plus a separate time-gradient accumulator (`densify_grad_t_threshold`); split large Gaussians, clone small ones, prune by opacity/screen-size | No camera/view-space gradient exists in this volumetric setting: accumulate the 3D `get_xyz` gradient norm and the raw `time` gradient instead, normalized by observation count; same split/clone/prune structure (`densify()`), size threshold on 4D scale (3 spatial + 1 temporal) instead of screen-space radius. No opacity-reset cycle (paper's anti-floater trick for RGB alpha blending; not adapted, see Limitations) |
| `gaussian_renderer/__init__.py` (RGB rasterizer) | Depth-sorted alpha compositing of view-projected 2D Gaussians, SH-evaluated color | Not used. The conditioned-on-`t` 3D ellipsoids (`ESVR4D.at(t)`) are queried through ESVR's existing scalar voxelizer (`gaussian_renderer.query`, `compute_cov3D_python=True`) exactly as `dynamic_esvr`/static ESVR do; no RGB, SH or camera path is introduced |

Donor camera calibration, GUI, RGB rasterizer/SH, and image-only data loaders
are not dependencies. Review focused on the 4D covariance/conditioning math
and densification control, not every unrelated viewer/training utility.

## A previously-undiscovered CUDA backward bug, found and fixed on this branch

Bringing up ESVR4D required backpropagating through primitive *position*
far more directly than the deformation-MLP branch does (motion here comes
from the position/time cross-covariance itself, not an MLP offset added on
top of a frozen canonical position gradient path). A test comparing the CUDA
voxelizer's analytic gradients against an independent dense PyTorch
reimplementation of the same forward math
(`tests/test_esvr4d.py::CUDA::test_covariance_forward_backward_and_tiles`)
failed only on the backward pass — forward matched to `~1e-6`. Isolating this
outside of ESVR4D (raw `voxelize_gaussians` autograd call, one hand-placed
Gaussian, real finite differences) showed the analytic `means3D` gradient is
off by a factor of `dVoxel²` (`dVoxel = sVoxel/nVoxel`, the physical size of
one voxel), confirmed at three different `dVoxel` values:

| dVoxel | analytic / numeric | dVoxel² |
|---:|---:|---:|
| 1.0 | 1.001 | 1.0 |
| 0.25 | 0.063 | 0.0625 |
| 4.0 | 16.02 | 16.0 |

Root cause, in `submodules/gaussian-rasterization-voxelization/cuda_voxelizer/backward.cu`:
the render kernel computes the position gradient in voxel-index units, then
scales it by `dVoxel` (`ddelx_dx = dVoxel_x`, a constant carried over from the
donor rasterizer's NDC-to-pixel compensation) before `preprocessCUDA` copies
that value verbatim into the world-space `means3D` gradient. The correct
chain-rule factor is `1/dVoxel` (`mean_world = center - sVoxel/2 +
dVoxel·point_vol`, so `d(point_vol)/d(mean_world) = 1/dVoxel`), not `dVoxel`.
Fixed by using `1.0f/dVoxel_{x,y,z}` instead
(see the comment at that line for the full derivation).

**This bug is not specific to ESVR4D.** It has been present, unmodified,
since the CUDA voxelizer was first added (`9d9f535`), identically on `main`
and `codex/deformable-esvr`. Because every query in this repository uses
`dVoxel = step = 2/max(shape)` (e.g. `≈0.0156` for a 128³ grid, i.e.
`dVoxel² ≈ 2.4e-4`), **every prior position gradient computed by this
voxelizer — static ESVR, `codex/deformable-esvr`'s canonical `xyz` and its
deformation MLP's `xyz` head — was suppressed by roughly four orders of
magnitude relative to the scale/rotation/opacity gradients**, which are
computed through an independent, unaffected path
(`computeCov3DCUDA`'s `dL_dcov`). This plausibly explains why the
deformable-esvr branch's 90-frame run (`output/dynamic/vorts90_full_geometry/RESULTS.md`)
showed clear underfitting specifically in the motion channel while
scale/opacity-driven static fidelity was fine — but this is a hypothesis
based on the mechanism, not something re-verified by retraining that branch
here; only the fix and its unit-test/finite-difference validation are
confirmed.

**This copy of the repository (`ESVR-4dgs/`) carries the fix; the original
working copy the review was run from does not.** Fixing the shared kernel
there — and whether to retrain/re-evaluate the static and deformable-ESVR
results under it — is a separate decision for the user, out of scope for
this branch. `submodules/gaussian-rasterization-voxelization` was rebuilt
in-place here with `uv pip install -e . --no-build-isolation` after the
one-line-per-axis fix; all 5 `tests/test_esvr4d.py` cases and all 3
pre-existing `tests/test_dynamic_esvr.py` cases pass against the rebuilt
extension.

## Mathematical representation

For canonical position `x∈R³`, canonical time `τ`, 4D log-scale
`(s_x,s_y,s_z,s_t)`, rotation quaternion pair `(l,r)`, scalar logit `o`:

```text
R = rotation4d(l, r)                       # SO(4), block form (l conjugated on the right, axes reversed)
Cov4D = R · diag(exp(s))² · R^T            # 4x4, full spacetime covariance
[Σ_xyz  c]   = Cov4D                        # c = cross-covariance (3,), v = temporal variance (scalar)
[c^T    v]

dt = t - τ
mean(t)     = x + (c / v) · dt
spatial(t)  = Σ_xyz - c·c^T / v            # Schur complement; positive definite whenever Cov4D is
gate(t)     = exp(-0.5 · dt² / v)          # temporal marginal, doubles as an opacity multiplier
scalar(t)   = sigmoid(o) · gate(t)
volume(t)   = ESVR_voxelizer(mean(t), spatial(t), scalar(t))   # primitives with gate(t) <= temporal_cutoff are skipped
```

Primitives are only ever "spawned" for a query time by the `temporal_cutoff`
mask in `at()` — there is no MLP, no explicit displacement field, and no
canonical/deformed split: the position and shape *at time t* are the full
Gaussian-conditional mean and covariance, so a translating or rotating object
is represented purely by off-diagonal spacetime covariance (`c`,
`c/v` acting as a constant velocity term locally) rather than a learned
function of `t`. This is the paper's key structural difference from the
deformable-MLP family and is why position gradients matter so much more here
(see the CUDA bug section above).

## Densification

`densify()` follows the paper's split/clone/prune structure with volumetric
substitutions:

- **Trigger**: `max(accumulated |xyz-gradient| / space_threshold, accumulated
  |time-gradient| / time_threshold) >= 1`, both normalized by observation
  count. No view-space 2D gradient exists here, so the raw 3D position
  gradient magnitude (`sliced.get_xyz.grad`, recorded only for primitives
  actually visible at the sampled time/block) stands in for it; the time
  gradient is `model.time.grad` directly.
- **Split vs. clone**: primitives whose spatial or temporal scale exceeds
  `densify_spatial_size`/`densify_temporal_size` are split into two children
  (sampled by the paper's approach: draw from `N(0, scale)` in the
  primitive's own 4D frame, rotate by `rotation4d(left,right)`, shrink scale
  by `1/1.6`); smaller ones are cloned once.
- **Prune**: primitives whose maximum-over-all-time scalar contribution
  (`sigmoid(logit_scalar) · exp(-0.5·dt²/var_t)` with `dt` measured against
  the *closest point in `[0,1]`*, not one frame) falls below `min_scalar` are
  removed; at least one primitive is always kept.
- **Optimizer moments**: `replace_population()` preserves Adam `exp_avg`/
  `exp_avg_sq` for surviving/cloned-from primitives and zero-initializes them
  for newborns, matching the paper's approach (verified by
  `tests/test_esvr4d.py::test_time_densification_and_optimizer`).

## Data contract

Reuses `dynamic_esvr/data.py`'s manifest format unchanged — see
[docs/DEFORMABLE_ESVR.md](DEFORMABLE_ESVR.md#data-contract) for the full
schema (common grid/axes/intensity range, explicit train/test split, `time`
values normalized against a shared `time_range`). `train_esvr4d.py` and
`inference_esvr4d.py` both take the same manifest and time-unit conventions
(training/evaluation use manifest-normalized `[0,1]` time; `inference_esvr4d.py
--time` takes original manifest units and converts internally).

## Commands

```bash
# Synthetic smoke test (5-frame moving anisotropic blob, held-out t=0.25/0.75)
.venv/bin/python experiments/make_dynamic_synthetic.py --output /tmp/esvr4d_smoke/data
.venv/bin/python train_esvr4d.py --manifest /tmp/esvr4d_smoke/data/sequence.json \
  --output /tmp/esvr4d_smoke/run --steps 300 --points 256 --block 32 --lambda-ssim 0 \
  --lr 0.01 --rotation-lr 0.005 --densify-from 50 --densify-every 50 --densify-until 200

# Unit tests, including CUDA gradient/tiling/conditional-distribution checks
.venv/bin/python -m unittest tests.test_esvr4d -v

# Arbitrary-time reconstruction in ORIGINAL time units; no source volumes needed
.venv/bin/python inference_esvr4d.py --checkpoint /tmp/esvr4d_smoke/run/checkpoint.pt \
  --time 0.33 --output /tmp/esvr4d_smoke/infer

# Checkpoint reload and full source-frame evaluation
.venv/bin/python train_esvr4d.py --manifest /tmp/esvr4d_smoke/data/sequence.json \
  --resume /tmp/esvr4d_smoke/run/checkpoint.pt --eval-only --output /tmp/esvr4d_smoke/reload \
  --steps 300 --points 256 --block 32 --lambda-ssim 0 --lr 0.01 --rotation-lr 0.005 \
  --densify-from 50 --densify-every 50 --densify-until 200
```

Verified on this smoke run: eval-only reload after densification changed the
population from 256 to 1074 primitives reproduces all 5 exported frames
bit-for-bit (`max abs diff == 0.0`); resuming an interrupted run from a
mid-training snapshot to the same step target completes without error and
lands within ~0.01 dB of the uninterrupted run (small CUDA-atomic run-to-run
variance, same as documented for `dynamic_esvr`). Held-out synthetic frames
(t=0.25/0.75) reached ~24.7–24.9 dB PSNR after only 300 steps on 256→1074
primitives — a smoke test, not a converged or tuned result.

Checkpoints are written every 500 steps and at completion, same convention as
`train_dynamic.py`. `--resume` requires identical training arguments and
manifest (same "resume = continue this exact schedule" semantics as
`train_dynamic.py`; it does not extend `--steps`).

## Limitations / not yet done

- **Not yet run on the real 90-frame vorts sequence.** Only the synthetic
  5-frame smoke test above has been executed end to end; real-data
  quality (and whether the CUDA fix changes it) is unmeasured.
- **No PLY export.** A spacetime primitive's slice at time `t` is a derived
  ellipsoid (Schur-complement mean/covariance), not a stored 3D Gaussian;
  exporting it as a static PLY needs an extra covariance→scale/quaternion
  eigendecomposition that isn't implemented yet. NPY/RAW/PNG export and full
  metrics work today (`evaluate()` in `train_esvr4d.py`).
  Adapting `experiments/evaluate_paper_metrics.py`/`evaluate_official_tf.py`
  (3D/2D PSNR/SSIM, LPIPS) to this branch's checkpoint format is also
  unstarted.
- **No opacity-reset cycle.** The paper periodically resets opacity to fight
  floaters accumulated from bad early splits; not adapted here, since there
  is no camera-visibility notion of "floater" in this scalar-volume setting
  and the effect on a scalar reconstruction is unverified.
- **Densification thresholds are engineering defaults**, not paper values —
  the paper's `densify_grad_threshold`/`densify_grad_t_threshold` are
  calibrated to view-space pixel gradients from a camera rasterizer, which
  has no equivalent unit here; `space_threshold`/`time_threshold` in
  `ESVR4D.densify()` were chosen to produce a reasonable split rate on the
  smoke test, not tuned against real data or the donor paper's ablations.
- **No learning-rate schedule differentiation between `xyz`/`time`/`scale`
  the way the paper's optimizer groups do** beyond the two groups already
  present (position-like parameters vs. rotation quaternions); this may
  under- or over-train specific channels relative to the paper.

## License

Network/covariance design adapted from the Fudan Zhang Vision Group's
MIT-licensed 4D Gaussian Splatting code; the notice is retained in
`third_party/4d-gaussian-splatting.LICENSE`. Existing ESVR/Inria components
retain their original restrictions. The upstream MIT notice does not
relicense the surrounding repository or donor submodules.
