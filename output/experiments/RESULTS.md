# ESVR Reproduction Results

This historical report covers multiple static runs under `output/experiments/`
and the earlier `output/aneurysm/` baseline. Paths in the text are repo-relative.

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
| Breast | 448 x 448 x 160 | 125.4 MB as float32 | Downloaded | Volume Library `Tumor-DCIS.pvm`, 8-bit, decoded checksum `EDE5D510`. The paper does not identify this source or publish its preprocessing. |
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

### Aneurysm paper-configuration run

- Date: 2026-09-04
- Paper Appendix C parameters, including `percent_dense=0.008`
- Final primitives: 57,142
- Final PLY size: 2,514,543 bytes
- Training time: 109.5 seconds
- Full-volume 3D PSNR: 52.31 dB
- SSIM / 3D SSIM: 0.9978 / 0.9972
- Output: `output/experiments/paper_repro_p0008/aneurysm_chunk_0`

Compared with the paper (59K, 2.6 MB, 51.54 dB, 388 seconds), model size and
quality closely agree; the RTX 5090 run is 3.54x faster than the reported RTX A6000 run.

### Breast source and preprocessing investigation

Downloaded source:

- URL: `http://volume.open-terrain.org/Tumor-DCIS.pvm`
- PVM SHA-256: `8e84db0b62b27959f2e02e56e58b9c1d7250bd9d57c58d3b1cfc74557a527853`
- Decoded RAW SHA-256: `117d3227bdaa10d65250c4fe97f7cfb9f480b8dec23fb206f16a177b75f807d7`
- Decoded metadata: 448 x 448 x 160, uint8, checksum `EDE5D510`
- Raw nonzero ratio: 99.2581%; paper nonzero ratio: 3.48%

This exact dimension/type match is not sufficient to identify the paper's processed
Breast volume. The paper gives neither a Breast download citation nor preprocessing
steps. Thresholding values below 60/255 produces a 3.5120% nonzero volume, close to
the paper's 3.48%; this is explicitly an inferred diagnostic preprocessing step.

| Run | Input | Pruning fraction | Final primitives | PLY bytes | PSNR | SSIM / 3D SSIM | Time |
|---|---|---:|---:|---:|---:|---:|---:|
| Raw, repository Aneurysm pruning | Original PVM values | 66% | 408,167 | 17,958,148 | 42.48 | 0.9591 / 0.9611 | 940.7 s |
| Raw, size-targeted pruning | Original PVM values | 95% (inferred) | 59,768 | 2,630,087 | 36.81 | 0.9192 / 0.9202 | 975.4 s |
| Sparse diagnostic | values below 60 set to zero | 66% | 131,574 | 5,789,200 | 43.16 | 0.9946 / 0.9934 | 233.0 s |

Paper Breast result: 67K primitives, 2.9 MB, 46.00 dB, 756 seconds. None of the
three runs is claimed as an exact reproduction. The experiments show that model size
can be matched by inferred 95% pruning, while quality cannot be matched without the
authors' exact processed volume and undisclosed pruning configuration.

## Dataset scope and blockers

- Neuron cannot be downloaded because the paper explicitly calls it proprietary.
- BigBrain and both Hemibrain volumes are not direct downloadable artifacts matching
  the repository. The paper describes custom MIP filtering/cropping and neuron-type
  extraction via neuPrint and CloudVolume, but provides no bounding boxes, body IDs,
  preprocessing code, or hashes needed to reproduce the configured RAW files.
- Woodbranch has a matching public 2048-cubed uint16 RAW (16 GiB), but a full paper
  experiment is eight sequential chunks and the paper reports 45,408 seconds. It was
  not downloaded because it does not resolve the missing Breast preprocessing and is
  not necessary to validate the local GPU pipeline.

## Reproduction commands

- `bash experiments/download_breast.sh`
- `bash experiments/run_aneurysm_paper.sh`
- `bash experiments/run_breast_paper.sh`
