# Preserved original evaluation: CUSTOM TF, not the paper TF

The existing summary.json, images.json, volumes.json and decode_ms.json are
preserved unchanged. Image scores use the custom offline preview transfer function:

- Input: scalar normalized with common raw range [-1,1], then clipped to s in [0,1].
- RGB(s) = (0.95, 0.22 + 0.7*s, 0.08 + 0.7*s).
- Alpha(s) = 1-exp(-0.18*s), per voxel.
- Front-to-back emission/alpha compositing, black background, no lighting.
- 90 training frames x XYZ orthographic views, 128x128 RGB (270 pairs).
- Float32 rendering; metrics computed before PNG quantization.
- Mean 2D PSNR 31.164083 dB, SSIM 0.924334, LPIPS-Alex 0.145573.

These values are NOT results obtained with the official distributed RGBA LUT.
New LUT-based results are stored separately in ../official_tf_metrics/.
