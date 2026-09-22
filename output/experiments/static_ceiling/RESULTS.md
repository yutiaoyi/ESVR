# Static single-frame ESVR on the fixed voxelizer: recipe correction and capacity ceiling (2026-09-21/22)

## 1. The 09-13 "CUDA-fixed" static baselines ran on the unfixed kernel

`output/experiments/vorts0001_cuda_fixed_lr_only` (39.94 dB) and
`output/experiments/vorts90_cuda_fixed_lr_only` (39.38 dB) were launched
through `experiments/run_vorts0001.sh`, which does `source
.venv/bin/activate` — and until 2026-09-20 that activate script pointed at
`ESVR/.venv`, whose voxelizer has the unfixed position gradient (see
`output/scaffold4d/ablation/RESULTS.md`). Evidence:

- Re-running the identical configuration (`S1_init20k_x1`, only the output
  path differs — `cfg_args` diff) on the fixed kernel diverges at the first
  densification: 20,000 → 4.85 M primitives, loss 0.14 → 1.3, OOM. The
  09-13 run's own `psnr_log.txt` shows a normal trajectory (peak 83,537
  primitives). The same command cannot behave both ways on the same kernel.
- Mechanism: the fix makes position gradients 4096× larger, so
  `densify_grad_threshold 2.5e-7` (tuned on the unfixed kernel) selects
  every primitive. The 09-13 diagnosis attributed the OOM to the position
  learning rate and divided it by 4096 — but Adam normalises gradient
  scale, so that change only froze positions. The −1.84 dB "cost of the
  fix" reported in `docs/PROJECT_HANDOFF.md` §3.2 is the cost of a 4096×
  smaller position step, measured on the unfixed kernel.

Correct adaptation: keep the native learning rates, multiply the densify
threshold by 4096 (2.5e-7 → 1.024e-3). `S1b_init20k_x1` = **41.72 dB** with
21,247 primitives, matching the original unfixed-kernel baseline (41.77).
Everything else below uses this recipe (`experiments/run_static_ceiling.sh`).

## 2. Capacity / training-length sweep (single frame vorts0001, fixed kernel)

Schedules in scans (×8 = iterations); ×3 multiplies every schedule value.
"no prune" = `--pruning_remainweight 0.0` (the global importance prune,
which otherwise removes the lowest 66 %, is effectively disabled).

| Run | Init | Scans (iters) | Global prune | Final primitives | Peak | Size (11 float32) | PSNR |
|---|---:|---:|---|---:|---:|---:|---:|
| S1b_init20k_x1 | 20k | 2,800 (22,400) | 66 % | 21,247 | 63,332 | 0.93 MB | 41.72 |
| S2b_init60k_x1 | 60k | 2,800 | 66 % | 29,817 | 88,809 | 1.31 MB | 42.29 |
| S3b_init20k_x3 | 20k | 8,400 (67,200) | 66 % | 16,243 | 58,938 | 0.71 MB | 44.73 |
| S4b_init60k_x3_noprune | 60k | 8,400 | none | 63,246 | 87,428 | 2.78 MB | 47.05 |
| **S5b_init150k_x3_noprune** | 150k | 8,400 | none | 139,250 | 164,175 | 6.13 MB | **49.67** |

Reading:

- **The primitive and kernel can reach ~50 dB on this data**: 139k
  primitives and 67k iterations per frame. Per-frame that is 6.1 MB, i.e.
  ~550 MB for 90 frames (raw data 755 MB).
- Training length matters as much as capacity: S3b beats S2b by 2.4 dB with
  half the primitives. ESVR's own recipe (2,800 scans) is far from
  converged on this data.
- Per-frame static ESVR with ~21k primitives and 67k iterations reaches
  44.7 dB (S3b). The best dynamic model so far (`scaffold4d/long/I3`,
  39.73 dB) has ~18k useful Gaussians per query and sees each frame for
  480k/90/8 ≈ 670 full-volume-equivalent passes versus S3b's 8,400 — the
  dynamic model is under-optimised per frame by an order of magnitude, and
  ~5 dB below a static model of the same per-time capacity.

## 3. Consequences for earlier conclusions

- `PROJECT_HANDOFF.md` §3.2 "fix cost −1.84 dB": wrong; the fix has no
  intrinsic cost once the densify threshold is rescaled.
- The two "cuda_fixed_lr_only" baselines are unfixed-kernel runs with a
  crippled position LR; they should not be used as references.
  `esvr_original_per_frame` (41.22, unfixed kernel, original recipe) remains
  a valid number for that kernel.
- The dynamic branches' "per-frame ESVR = 39.4 dB" comparisons overstate
  how close they are: the honest static reference at matched capacity and
  generous training is 42–45 dB, and the ceiling is ~50.

## Not done

Single seed; only vorts0001. A 90-frame per-frame run with the S3b/S5b
recipes (to give a per-frame reference at 45 / 50 dB) is ~1.5 / 3 h.
