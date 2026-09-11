# vorts0001 ESVR experiment

Input: `data/xueliao_data/vorts0001.raw`, 128³, little-endian float32,
single scalar channel, X fastest, spacing (1,1,1). Loaded as ZYX and
transposed to logical XYZ. SHA256:
`33071ee7228efc8ad31741923cb0419d3d4a39f2d693e49da6ba3356750c01b4`.

Original scalar range [-1,1]; training uses (v+1)/2 exactly once. No thresholding,
resampling or integer quantization. Source file preserved.
Spatial coordinate normalization remains the repository implementation.

## Configuration and results

- RTX 5090; full volume, 64³ subvolumes, 2800 scans = 22400 iterations.
- Initial primitives: 20000; densification threshold: 2.5e-7;
  percent_dense: 0.008; global pruning fraction: 0.66.
- Other schedule/LR settings inherited from the Aneurysm experiment;
  this is a custom-data baseline, not an optimized parameter search.
- Training time reported by training script: 36.8 seconds.
- Final primitives: 21404; PLY: 941939 bytes (~8.91x smaller than source RAW,
  excluding checkpoints and metadata).
- Independent PLY reconstruction PSNR: 41.768953 dB, fixed normalized range 1.
- Original scalar units RMSE: 0.01631482.
- Training evaluation SSIM: 0.9793; 3D SSIM: 0.9811.

Initial attempt with 100000 primitives and 2.5e-9 threshold exhausted GPU memory
at iteration 480. Its stdout is retained in `failed_aggressive_training.log`.
The conservative rerun completed successfully. Earlier partial checkpoint files
may have been replaced by the successful run in the same output directory.

## Outputs and reproduction

Output root: `output/experiments/vorts0001_baseline/`.
Final model: `vorts0001_chunk_0/point_cloud/iteration_22400/point_cloud.ply`.
The `renders/` directory contains three-view GT/prediction/error images,
center-slice comparisons, metrics and transfer-function metadata, and a reconstructed
RAW in original scalar units (little-endian float32, X fastest, 128³).
Rendering is offline DVR after voxelization, not the paper's real-time renderer.
The current opacity mapping obscures the interior of this dense field; use slices
to assess internal reconstruction. Slices show preserved structures with residual
small-scale texture error.

```bash
bash experiments/run_vorts0001.sh
.venv/bin/python experiments/render_preview.py --vorts
```

Code changes: explicit dtype/intensity-range support in training/inference loaders,
dataset config, baseline script and preview outputs. Core ellipsoid, loss,
densification/pruning algorithms and CUDA kernels are unchanged.
