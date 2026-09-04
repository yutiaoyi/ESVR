# ESVR Reproduction Results

Paper: [arXiv:2608.05564](https://arxiv.org/abs/2608.05564)

## Environment

- GPU: NVIDIA GeForce RTX 5090 (32 GB, compute capability 12.0)
- Python: 3.10.21
- PyTorch: 2.7.1+cu128
- CUDA: 12.8

## Dataset availability

| Dataset | Paper resolution | Paper size | Local status | Notes |
|---|---:|---:|---|---|
| Aneurysm | 256 x 256 x 256 | 65.5 MB as float32 | Available | Repository TIFF is uint8 (16.8 MB) and is normalized to float32 by the loader. |
| Breast | 448 x 448 x 160 | 125.4 MB as float32 | Missing | Public-source URL still needs to be resolved/downloaded. |
| Neuron | 2537 x 1295 x 133 | 1.7 GB | Unavailable | The paper states that this dataset is private. |
| BigBrain | 1749 x 1643 x 1428 | 15.1 GB | Missing | Public but large; paper uses 8 chunks. |
| Woodbranch | 2048 x 2048 x 2048 | 32 GB | Missing | Public but large; paper uses 8 chunks. |
| Hemibrain (vDeltaA) | 4072 x 4372 x 4171 | 259 GB | Missing | Public but extremely large; paper uses 64 chunks. |
| Hemibrain (lLN2P) | 5115 x 4774 x 6128 | 568 GB | Missing | Public but extremely large; paper uses 125 chunks. |

## Completed experiments

### Aneurysm repository-default run

- Date: 2026-09-04
- Input: `data/aneurism_256x256x256_uint8.tif`
- Initial primitives: 100,000
- Subvolume: 128 cubed
- Epochs/scans: 2,800 (22,400 subvolume iterations)
- Densification epochs: 16 through 400
- Structure-aware pruning epoch: 1,200
- Gradient threshold: 2.5e-9
- `percent_dense`: 0.01 (repository script; paper Appendix C reports 0.008)
- Final primitives: 45,200
- Final PLY size: 1,989,095 bytes
- Training time: 68.7 seconds
- Full-volume 3D PSNR reported during training: 51.90 dB
- Independent chunked inference 3D PSNR: 51.904929 dB
- 3D SSIM: 0.9976
- Output: `output/aneurysm/s_2800_0.0000000025_0.01_0.00005_128/aneurysm_chunk_0`

Paper comparison (Aneurysm): 59K primitives, 2.6 MB, 51.54 dB 3D PSNR, and 388 seconds on RTX A6000.

## Pending experiments

### Aneurysm paper-configuration run

The reproducible command is saved in `experiments/run_aneurysm_paper.sh`. It changes
`percent_dense` from the repository default 0.01 to the paper Appendix C value 0.008.
Execution is pending GPU permission approval.
