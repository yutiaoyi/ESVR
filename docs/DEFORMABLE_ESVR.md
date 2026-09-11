# Deformable ESVR — first implementation

## Source and scope

Official source: https://github.com/ingra14m/Deformable-3D-Gaussians

Locally cloned to `.tools/Deformable-3D-Gaussians` (ignored, no nested repository
committed), branch `paper`, commit `ab49a1149a06b734e7a3026bb5e424cc08e94240`.
The upstream default branch also incorporates Lightweight-Deformable-GS; it is
not the version used here. To recover the source:

```bash
git clone --branch paper --single-branch https://github.com/ingra14m/Deformable-3D-Gaussians.git .tools/Deformable-3D-Gaussians
git -C .tools/Deformable-3D-Gaussians checkout ab49a1149a06b734e7a3026bb5e424cc08e94240
```

This is a scalar-volume adaptation, **not** a reproduction of the donor paper's
RGB benchmark or a complete dynamic-compression method. Static ESVR's existing
training, voxelizer, and vorts data adaptations committed on `main` at
`62ccfaa` are preserved. The deformable branch inherits that single-frame baseline.

## Source-reading map and decisions

| Upstream code | Behavior verified | Integration decision |
|---|---|---|
| `utils/time_utils.py` | 8-layer/256-wide ReLU MLP; Fourier XYZ/time; skip after layer D//2; XYZ/rotation/scale heads | Same basic non-Blender, non-6DoF architecture, configurable size; identity-initialized heads |
| `scene/deform_model.py` | Separate Adam and exponential LR decay; standalone MLP weights | Optimizer groups plus joint canonical/network checkpoint with optimizer and RNG state |
| `train.py` | Canonical XYZ detached at MLP input; static warmup; randomized frame selection; decaying timestamp noise on non-Blender data | Detached input, warmup, frame sampling retained; exact timestamps by default, optional decaying jitter |
| `gaussian_renderer/__init__.py` | `xyz + delta`; **activated scale + delta**; normalized canonical quaternion + delta; unchanged opacity/SH | ESVR voxelization, **log-scale + delta then exp**, renormalized quaternion; fixed scalar by default, optional scalar-logit head |
| `scene/gaussian_model.py` | Canonical PLY, activated parameters; clone/split/prune canonical parameters and Adam moments | Import/export ESVR scalar PLY; first version fixed point budget, no donor RGB/SH parameters |
| `scene/__init__.py`, `scene/dataset_readers.py` | Camera-based dispatch, dataset-specific frame IDs/time normalization and train/test lists | Explicit volume manifest: common grid, units, timestamps and train/test split |
| `utils/rigid_utils.py` | Optional screw-axis SE(3) exponential | Not used: basic displacement head only |

Donor camera calibration, GUI, RGB rasterizer, SH evaluation and image-only data
loaders are not dependencies. Review focused on the algorithm and its complete
training/data/render path, not every unrelated viewer utility.

ESVR's current CUDA kernel evaluates a compact sigmoid ellipsoid and **adds**
scalar contributions, not donor Gaussian alpha-composited RGB. Both values of
`use_step_function` currently execute the same sigmoid expression in its forward
kernel. The CUDA quaternion normalization is commented out, so normalization is
performed explicitly before calling it. No CUDA code is changed.

## Mathematical representation

For canonical XYZ `x`, log scale `s`, quaternion `q`, scalar logit `o`:

```text
(dx, dq, ds [, do]) = MLP(encode(stop_gradient(x)), encode(t))
x(t) = x + dx
scale(t) = exp(clamp(s + ds, -12, 2))
rotation(t) = normalize(normalize(q) + dq)
scalar(t) = sigmoid(o [+ do])
volume(t) = ESVR_voxelizer(x(t), scale(t), rotation(t), scalar(t))
```

Near-zero quaternions fall back to identity. Positive scales and normalized
quaternions ensure valid covariance. Identity initialization differs from donor
random output heads and supports safe warm-starting from an ESVR PLY. There is no
hard canonical-time anchor; canonical parameters are learned jointly. The scalar
head is an optional extension, not part of the original deformation network.

Training uses ESVR's existing L1/SSIM routine on random voxel blocks plus an optional
small squared deformation penalty. The SSIM helper is inherited unchanged; do not
interpret it as a newly implemented isotropic 3D SSIM metric. Learning rates and
initialization are baseline engineering choices, not paper hyperparameter parity.
Fixed primitive count intentionally avoids transplanting RGB screen-gradient
densification criteria into a 3D scalar problem. Dynamic pruning, temporal
regularization, quantization and entropy coding remain future work.

## Data contract

JSON paths are relative to the manifest, never the process working directory.
Every frame must use the same spatial grid, axis convention, and intensity range.
`storage_axes: "zyx"` means file dimensions Z,Y,X in C order, X fastest. RAW byte
counts, shapes, finite values, common value range, and unique times are checked.
NPY and TIFF are also accepted with the declared storage axes. Values map to [0,1]
using **one common range**, never independent per-frame min/max. Frames currently
reside in CPU RAM; only a sampled block is moved to GPU. Anisotropic physical
spacing and variable grids are not supported; resample externally if needed.

```json
{
  "shape_xyz": [128, 128, 128],
  "storage_axes": "zyx",
  "dtype": "<f4",
  "intensity_range": [-1, 1],
  "time_range": [0, 2],
  "frames": [
    {"path": "vorts0001.raw", "time": 0, "split": "train"},
    {"path": "vorts0002.raw", "time": 1, "split": "test"},
    {"path": "vorts0003.raw", "time": 2, "split": "train"}
  ]
}
```

The example above illustrates a held-out split. The user has since supplied all
90 frames at `data/xueliao_timevaryingdata/vorts_timevarying/vorts0001.raw` through
`vorts0090.raw`; `experiments/vorts_90_all_train.json` uses all of them for training.
Times 1–90 are ordinal, assumed equally spaced, not physical seconds. Fewer than two training
times are rejected unless `--allow-single-frame` is explicitly set, in which case
deformation is disabled. Such a run cannot demonstrate dynamic reconstruction.

## Commands

```bash
# Full 90-frame reconstruction schedule (all frames trained, not held-out testing)
bash experiments/run_vorts_dynamic.sh
.venv/bin/python experiments/summarize_dynamic_run.py \
  --run output/dynamic/vorts90_full_geometry --manifest experiments/vorts_90_all_train.json

# Unit tests, including CUDA gradient/anisotropic rotation/block consistency tests
.venv/bin/python -m unittest discover -s tests -p test_dynamic_esvr.py -v

# Real time-series training, once an actual multi-frame manifest is available
.venv/bin/python train_dynamic.py --manifest /absolute/path/sequence.json \
  --output output/dynamic/my_sequence --steps 10000 --warmup 1000 --points 20000
# Optional: --init-ply <ESVR scalar PLY> and/or --scalar-deformation

# Arbitrary-time reconstruction in ORIGINAL time units; no source volumes needed
.venv/bin/python inference_dynamic.py --checkpoint output/dynamic/my_sequence/checkpoint.pt \
  --time 0.5 --output output/dynamic/my_sequence_time05

# Checkpoint reload and full source-frame evaluation
.venv/bin/python train_dynamic.py --manifest /absolute/path/sequence.json \
  --resume output/dynamic/my_sequence/checkpoint.pt --eval-only \
  --output output/dynamic/my_sequence_reload
```

Outputs: `config.json`, `manifest.json`, `train.jsonl`, `checkpoint.pt`, per-frame
metrics, unclipped normalized XYZ NPY, original-unit little-endian float32 RAW,
deformed PLY snapshots, three orthographic GT/prediction PNGs per frame. Display
clipping does not affect metrics. PLY snapshots alone do not contain the motion
network; preserve `checkpoint.pt`. Only load trusted local PyTorch checkpoints.

Checkpoints are written every 500 steps and at completion. For interrupted-run
resume use the identical training arguments and manifest, `--resume checkpoint.pt`
and a new empty `--output` directory. `--steps` denotes optimizer steps (not ESVR
static scans). Model dimensions/config are loaded from the checkpoint. Evaluation
and inference do not require respecifying network dimensions.

For full-sequence training, `--tile-sampling` samples nonoverlapping tiles to
include volume boundaries. `--eval-every` monitors the first/middle/last training
frames, and `--snapshot-every` retains numbered checkpoints in addition to the
latest one. The full-run summary verifies export completeness, records data
hashes, plots metrics/slices, creates a 90-frame DVR comparison GIF, and writes
an optimizer-free `model_inference.pt` usable by `inference_dynamic.py`.

## License

Network design adapted from Ziyi Yang's MIT-licensed code; the notice is retained
in `third_party/Deformable-3D-Gaussians.LICENSE`. Existing ESVR/Inria components
retain their original restrictions. The upstream MIT notice does not relicense
the surrounding repository or donor submodules.
