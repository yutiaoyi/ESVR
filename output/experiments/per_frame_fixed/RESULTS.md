# 90-frame per-frame static ESVR on the fixed voxelizer (2026-09-26)

Every frame of vorts (128^3 float32, values mapped to [0,1]) fitted
independently with `train_chunk.py` on this repository's fixed voxelizer
(`experiments/run_per_frame_fixed.sh`). Until now every fixed-kernel
per-frame number (S1b 41.72, S3b 44.73, S5b 49.67) came from frame 1 only,
the easiest frame (unfixed-kernel 90-frame run: frame 1 = 41.77, mean 41.22).

| Recipe | Settings | Mean PSNR | Min / max | Primitives (90 frames) | Size fp32 / fp16 |
|---|---|---:|---|---:|---:|
| **S1b** | ESVR's own vorts recipe: 20k init, 2,800 scans = 22,400 iterations, 66 % global prune | **41.207** | 40.36 / 41.81 | 2,132,018 | 93.8 / 46.9 MB |
| **S5b** | ceiling: 150k init, 8,400 scans = 67,200 iterations, global prune off | **49.521** | 49.24 / 49.78 | 12,628,052 | 555.6 / 277.8 MB |

- S1b matches the unfixed-kernel 90-frame run (41.22): the kernel fix costs
  nothing once the densify threshold is rescaled.
- S5b is frame-independent (49.2–49.8); S1b dips in frames 5–9 (to 40.4).
- Time with four concurrent lanes: S1b 307 s/frame, S5b 870 s/frame
  (alone on the GPU: 37 s and 130 s). S5b's PSNR plateaus by ~22k
  iterations (49.85 at 22k on frame 1), so the x3 schedule is not needed.
- Size counts 11 floats per primitive (centre 3, log-scale 3, quaternion 4,
  intensity 1), as ESVR stores them.
- Per-frame values: `summary.json`.

Known issue: `train_chunk.py` writes `voxelizer.json` to `args.model_path`,
which is empty in this call path, so the provenance file landed at the
repository root (overwritten by each frame) instead of each frame's folder.
The runner's self-check still guarantees the repository kernel was loaded.
