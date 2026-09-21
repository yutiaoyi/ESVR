# ESVR-Scaffold4D — steps × budget toward 40 dB (2026-09-21)

Fixed voxelizer, `--final-lr-scale 0.01` (default since `5fa9812`), K = 10,
seed 42, all 90 frames trained, every PSNR from the complete 90-frame
`evaluation/metrics.json`. Baselines: `decay/H4_150k_decay0.01_480k`
(39.431 dB) and `decay/H1_40k_decay0.01` (35.464 dB).

| Run | Anchors cap | Steps | Trained size | Mean PSNR | Min / max | Compact (fp16 + prune) |
|---|---:|---:|---:|---:|---|---|
| **I3_300k_decay0.01_480k** | 300k | 480k | 87.6 MB | **39.734** | 37.24 / **40.69** | **9.3 MB, 39.71** |
| I2_150k_decay0.01_960k | 150k | 960k | 49.8 MB | 39.326 | 36.80 / 40.37 | 6.1 MB, 39.30 |
| J1_40k_decay0.01_480k | 40k | 480k | 13.3 MB | 36.940 | 34.13 / 37.64 | 2.6 MB, 36.93 |
| I1_150k_decay0.1_480k | 150k | 480k | — | stopped | — | — |

## Steps × budget (all with decay)

| Anchor cap | 120k steps | 480k steps | Δ |
|---:|---:|---:|---:|
| 40k | H1 35.46 | J1 36.94 | +1.48 |
| 150k | G5 38.04 | H4 39.43 | +1.40 |
| 300k | H5 38.15 | I3 39.73 | +1.59 |

- Steps ×4 is worth ~1.5 dB at every budget; ×8 adds nothing (I2 vs H4:
  −0.10, within the 0.42 dB seed spread).
- Budget: 40k → 150k ≈ +2.5 dB; 150k → 300k ≈ +0.3 at 480k steps (and 0 at
  120k steps, H5 vs G5).
- I3 is 0.27 dB below the 40 dB target, inside seed noise; its best frame is
  40.69 dB. I3 reload check: max abs diff 0.0 over 90 frames.
- I1 was stopped early to keep four concurrent runs within GPU memory;
  `decay/H2` already showed decay 1/10 vs 1/100 is within noise.

## Compaction

`experiments/compact_scaffold4d.py --fp16 --drop-floor` (reads
`checkpoint.pt`, writes `compact/`): I3 87.6 → 9.3 MB (−0.02 dB), J1 13.3 →
2.6 MB (−0.01 dB). Per-frame static ESVR: 90 MB at 39.38 dB.

## Not done

Single seed; no held-out frames; the remaining 0.3 dB to the target is
below the resolution of a single run.
