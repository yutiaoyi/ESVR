# Parameterisation diagnostic: free a Scaffold4D time slice (2026-09-26)

`experiments/export_slice_to_esvr.py` exports the children of
`next/P3_block128_thr1e-4_240k` selected at frame 1 (60,813 ellipsoids;
rendering the export reproduces P3's 40.272 dB exactly) as a static ESVR
checkpoint; `run_unshackle.sh` trains it with ESVR's own optimiser (S1b
learning rates, 22,400 iterations) on frame 1.

| Run | Start | Primitives | Densify | Frame-1 PSNR |
|---|---|---:|---|---:|
| P3 itself | — | 60,813 | — | 40.27 |
| U1 slice, freed | P3 layout | 58,338 | no | **44.38** |
| U2 random init | random | 60,813 | no | 33.21 |
| U3 slice + ESVR densification | P3 layout | ~52k | yes (opacity resets) | 42.20 |
| S1b / S4b references | random | 21k / 63k | yes | 41.81 / 47.05 |

Freeing the parameters of the same primitives gains +4.1 dB: sharing across
time and the shared MLP decoding cost ~4 dB on this frame. The learned
layout is good (random init of the same count: 33.2). `scaffold4d/explicit/X1`
(same children as free explicit 4D primitives, still shared across time)
gains only +0.37 dB over 90 frames, so most of the 4 dB is the time sharing,
not the MLP.
