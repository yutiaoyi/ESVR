# Deformable ESVR integration checks

Historical initial-integration checks below. The user has subsequently supplied
90 real frames; the completed full run is documented in
[VORTS90_DYNAMIC_RESULTS.md](VORTS90_DYNAMIC_RESULTS.md).

Implementation branch: `codex/deformable-esvr`, based on `ca1db2d`.
Donor `paper` branch: `ab49a1149a06b734e7a3026bb5e424cc08e94240`.
Environment: local `.venv`, Python 3.10, Torch 2.7.1+cu128, RTX 5090.
These are initial engineering checks, not full research benchmark results.

## Synthetic motion: dynamic vs shared static model

Generated 32³ scalar sigmoid ellipsoid translating along X. Train times 0, 0.5,
1; held-out test times 0.25, 0.75. No user volume was modified to manufacture
additional observed time steps. Both runs: 256 canonical primitives, 4-layer
64-wide MLP allocated, XYZ/time encoding with 4 frequency bands, seed 42,
1000 optimizer steps, full 32³ blocks, L1 supervision, canonical LR 0.005,
MLP LR 0.002, exponential end factor 0.1. Dynamic warmup 100 steps, motion
penalty 1e-4, no time jitter or scalar deformation. Static ablation disables
the MLP. This matches primitive/iteration budget, not effective trainable
parameter count or wall time. CUDA atomic reductions introduce small run-to-run
differences even with the same seed.

| Time | Split | Static PSNR (dB) | Dynamic PSNR (dB) |
|---|---|---:|---:|
| 0 | train | 25.6871 | 45.2991 |
| 0.25 | test | 27.0708 | 29.6305 |
| 0.5 | train | 28.8679 | 46.6446 |
| 0.75 | test | 26.5413 | 35.8710 |
| 1 | train | 24.8055 | 46.2943 |
| Train arithmetic mean | | 26.4535 | 46.0793 |
| Test arithmetic mean | | 26.8061 | 32.7507 |

Metrics use all voxels, data range 1, **unclipped** predictions. Sparse background
contributes to PSNR; these numbers do not imply equivalent fidelity on dense
real data. The lower held-out scores demonstrate remaining interpolation error.
The experiment supports basic functionality, not generalization to unseen
scientific sequences or a compression-rate claim.

```bash
.venv/bin/python experiments/make_dynamic_synthetic.py --output output/dynamic/synthetic_data
.venv/bin/python train_dynamic.py \
  --manifest output/dynamic/synthetic_data/sequence.json \
  --output output/dynamic/synthetic_deform --steps 1000 --warmup 100 \
  --points 256 --depth 4 --width 64 --xyz-bands 4 --time-bands 4 \
  --block 32 --lambda-ssim 0 --lr 0.005 --deform-lr 0.002
# Identical command with --static and --output output/dynamic/synthetic_static
```

Run outputs live under `output/dynamic/synthetic_deform` and `synthetic_static`:
config, training log, joint checkpoint, five reconstructed NPY/RAW/PLY frames,
15 GT/prediction PNGs, and full `evaluation/metrics.json`.

## Real user volume: static compatibility only

`experiments/vorts_single_frame.json` points to the only currently available
128³ little-endian float32 frame `data/xueliao_data/vorts0001.raw`, X fastest,
common source intensity range [-1,1]. Initialized from the existing ESVR
`output/experiments/vorts0001_baseline/vorts0001_chunk_0/point_cloud/iteration_22400/point_cloud.ply`.
Ran 20 steps with L1/SSIM (weight 0.2), 32³ blocks, LR 1e-6. Deformation is
explicitly disabled for a single training time.

- Full-volume PSNR: **41.768864 dB**.
- Normalized MAE: **0.00617388**.
- Original-unit RMSE: **0.01631499**.
- Output: `output/dynamic/vorts_single_smoke/evaluation` (NPY, original-unit
  float32 RAW, PLY, and three orthographic GT/prediction PNGs).
- This mostly measures compatibility with the pre-trained static PLY, **not**
  successful dynamic fitting or a 20-step from-scratch reconstruction.

```bash
.venv/bin/python train_dynamic.py --manifest experiments/vorts_single_frame.json \
  --output output/dynamic/vorts_single_smoke --steps 20 --warmup 20 \
  --allow-single-frame \
  --init-ply output/experiments/vorts0001_baseline/vorts0001_chunk_0/point_cloud/iteration_22400/point_cloud.ply \
  --lr 0.000001 --block 32 --depth 4 --width 64
```

## Verification

- CPU tests: identity initialization, positive scales, normalized/zero fallback
  quaternion, all output head gradients, axis conversion, shared normalization,
  time normalization, split handling, RAW size validation.
- GPU test: finite nonzero gradients through the actual ESVR voxelizer for XYZ,
  anisotropic rotation, scale and scalar heads; noncubic full/block consistency;
  state-dict round trip. All three unit tests passed with CUDA enabled.
- Synthetic joint checkpoint reloaded through `train_dynamic.py --eval-only`:
  all five reconstructed arrays have maximum absolute difference **0.0** from
  original evaluation. Results at `output/dynamic/synthetic_reload`.
- Independent `inference_dynamic.py --time 0.33` succeeded without loading source
  volumes. NPY, RAW, PLY and three renders at `output/dynamic/synthetic_time033`.
  No ground-truth metric claimed for that arbitrary time.
- Python compilation and `git diff --check` passed.
- Final default-architecture smoke: 8 layers/256 width, both encodings 10 bands,
  scalar deformation enabled, L1/SSIM enabled, 3 steps (warmup 1), 32 primitives,
  16³ blocks. Training, atomic checkpoint save and all exports succeeded at
  `output/dynamic/default_arch_smoke`. This is only an execution test, not a
  converged model or a quality comparison.

## Remaining data dependency

Need at least two genuine training time steps from the user's sequence and their
time/grid metadata; preferably additional held-out times. Currently only
`vorts0001.raw` is available. Dynamic densification/pruning, compression and a
full-resolution multi-frame experiment are not completed by these checks.
