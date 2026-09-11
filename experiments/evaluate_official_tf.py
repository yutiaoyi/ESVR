"""Evaluate the distributed ESVR RGBA LUT using an explicitly scoped offline DVR.

Not a reproduction of the unavailable customized native CUDA renderer.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCH_HOME', str(ROOT / '.tools/torch-cache'))
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from dynamic_esvr.data import VolumeSequence


def sample_lut(values, lut):
    position = np.clip(values, 0, 1) * 255
    low = np.floor(position).astype(np.int32)
    high = np.minimum(low + 1, 255)
    fraction = (position - low)[..., None]
    return (lut[low] * (1-fraction) + lut[high] * fraction).astype(np.float32)


def render(volume, axis, lut):
    slices = np.moveaxis(volume, axis, 0)
    rgb = np.zeros((*slices.shape[1:], 3), np.float32)
    transmittance = np.ones(slices.shape[1:], np.float32)
    for values in slices:
        rgba = sample_lut(values, lut)
        alpha = rgba[..., 3].clip(0, 1)
        rgb += (transmittance * alpha)[..., None] * rgba[..., :3]
        transmittance *= 1-alpha
    return np.flipud(np.swapaxes(rgb, 0, 1)).copy()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--tf', default='SIBR_viewer/install/bin/transferFunc.txt')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        p.error('Output must be new or empty; previous results are never overwritten')
    lut = np.loadtxt(args.tf, dtype=np.float32)
    if lut.shape != (256,4) or not np.isfinite(lut).all() or lut.min()<0 or lut[:,:3].max()>1:
        raise ValueError('Expected 256 finite nonnegative RGBA rows with RGB in [0,1]')
    # Analytical tests: endpoint/interpolation and front-to-back compositing.
    ramp = np.repeat(np.linspace(0,1,256,dtype=np.float32)[:,None],4,axis=1)
    np.testing.assert_allclose(sample_lut(np.array([0,.5,1]),ramp)[:,0], [0,.5,1],atol=1e-7)
    constant = np.tile(np.array([.2,.4,.8,.25],np.float32),(256,1))
    expected = np.array([.2,.4,.8])*(1-.75**2)
    for axis in range(3):
        np.testing.assert_allclose(render(np.ones((2,2,2),np.float32),axis,constant),
                                  np.broadcast_to(expected,(2,2,3)),atol=1e-7)
    seq = VolumeSequence(args.manifest)
    import lpips
    net = lpips.LPIPS(net='alex',version='0.1').cuda().eval()
    run = Path(args.run)
    old_path = run/'supplemental_metrics/summary.json'
    original_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    (output/'transferFunc.txt').write_bytes(Path(args.tf).read_bytes())
    images = []
    for i, gt in enumerate(seq.volumes):
        pred = np.load(run/'evaluation'/f'frame_{i:04d}.npy')
        for axis in range(3):
            a,b = [render(v,axis,lut).clip(0,1) for v in (gt.numpy(),pred)]
            mse = float(np.mean((a.astype(np.float64)-b)**2))
            tensors=[torch.from_numpy(v).permute(2,0,1)[None].cuda()*2-1 for v in (a,b)]
            images.append(dict(frame=i+1,axis=axis,psnr2d=-10*math.log10(max(mse,1e-30)),
                ssim2d=float(structural_similarity(a,b,data_range=1,channel_axis=-1,gaussian_weights=True,
                    sigma=1.5,use_sample_covariance=False,win_size=11)),
                lpips_alex=float(net(*tensors).item())))
            Image.fromarray((np.concatenate([a,b],axis=1)*255).astype(np.uint8)).save(output/f'frame_{i+1:04d}_axis{axis}_gt_pred.png')
        if (i+1)%10==0:
            print(f'Evaluated {i+1}/{len(seq.frames)}',flush=True)
    old = json.loads(old_path.read_text())
    summary=dict(protocol='Distributed official ESVR TF LUT, offline DVR adaptation; NOT full paper/native renderer parity',
        tf_source=str(Path(args.tf).resolve()),tf_sha256=hashlib.sha256(Path(args.tf).read_bytes()).hexdigest(),
        tf_sampling='RGBA rows at i/255; linear interpolation, intensity clipped to [0,1]',
        opacity='Interpolated LUT alpha clipped to [0,1] per voxel; source row255 alpha=1.005144; no extra 0.18 exponential mapping; no step correction',
        rendering='One sample per voxel; front-to-back emission/alpha compositing; black background; no lighting',
        views='90 training frames x 3 orthographic axes, 128x128 RGB; same views as custom-TF evaluation',
        image_metrics='Float32 images before PNG quantization; PSNR range=1; Gaussian SSIM sigma1.5 win11 population; LPIPS0.1 AlexNet',
        mean_2d_psnr=float(np.mean([r['psnr2d'] for r in images])),
        mean_2d_ssim=float(np.mean([r['ssim2d'] for r in images])),
        mean_lpips_alex=float(np.mean([r['lpips_alex'] for r in images])),
        image_count=len(images),old_custom_tf={k:old[k] for k in ['mean_2d_psnr','mean_2d_ssim','mean_lpips_alex']},
        old_summary_sha256=original_hash,
        limitation='Paper dataset-specific TF correspondence and native LUT sampling/opacity correction unconfirmed; not 800x800/720 views')
    assert hashlib.sha256(old_path.read_bytes()).hexdigest()==original_hash
    (output/'images.json').write_text(json.dumps(images,indent=2))
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
