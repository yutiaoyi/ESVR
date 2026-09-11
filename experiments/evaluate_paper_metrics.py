"""Supplement full-volume metrics with explicitly scoped offline image metrics."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".tools/torch-cache"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/esvr-mpl")
import numpy as np
import torch
from skimage.metrics import structural_similarity
import lpips
from dynamic_esvr.data import VolumeSequence
from dynamic_esvr.model import DynamicESVR, query_block
from experiments.render_preview import dvr
from query import compute_ssim_3d


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    run = Path(args.run)
    seq = VolumeSequence(args.manifest)
    metric = lpips.LPIPS(net="alex", version="0.1").cuda().eval()
    volumes, images = [], []
    for i, gt in enumerate(seq.volumes):
        pred = np.load(run / "evaluation" / f"frame_{i:04d}.npy")
        ground = gt.numpy()
        mse = float(np.mean((pred.astype(np.float64)-ground)**2))
        volumes.append(dict(frame=i+1, psnr3d=-10*math.log10(mse),
            ssim3d=compute_ssim_3d(gt.cuda(), torch.from_numpy(pred).cuda(), data_range=1, window_size=7)))
        for axis in range(3):
            a, b = [dvr(v, axis).clip(0, 1) for v in (ground, pred)]
            mse2d = float(np.mean((a.astype(np.float64)-b)**2))
            tensors = [torch.from_numpy(v.copy()).permute(2,0,1).unsqueeze(0).cuda()*2-1 for v in (a,b)]
            images.append(dict(frame=i+1, axis=axis, psnr2d=-10*math.log10(mse2d),
                ssim2d=float(structural_similarity(a,b,data_range=1,channel_axis=-1,
                    gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11)),
                lpips_alex=float(metric(*tensors).item())))
        if (i+1)%10==0:
            print(f"Metrics {i+1}/{len(seq.frames)}", flush=True)
    del tensors, metric
    torch.cuda.empty_cache()
    saved = torch.load(run / "model_inference.pt", map_location="cuda", weights_only=False)
    model = DynamicESVR(**saved["model_config"]).cuda().eval()
    model.load_state_dict(saved["model"])

    def decode(t):
        volume = torch.empty(seq.shape, device="cuda")
        for x in range(0,seq.shape[0],64):
            for y in range(0,seq.shape[1],64):
                for z in range(0,seq.shape[2],64):
                    start=(x,y,z)
                    size=tuple(min(64,s-a) for s,a in zip(seq.shape,start))
                    volume[x:x+size[0],y:y+size[1],z:z+size[2]]=query_block(model,t,seq.shape,start,size,True)
        return volume
    for t in seq.times[:3]:
        decode(t)
    torch.cuda.synchronize()
    milliseconds=[]
    for t in seq.times:
        torch.cuda.synchronize()
        before=time.perf_counter()
        decode(t)
        torch.cuda.synchronize()
        milliseconds.append(1000*(time.perf_counter()-before))
    summary = dict(
        protocol="All 90 training volumes; offline float32 DVR, 3 orthographic views each, 128x128 RGB; NOT paper's 800x800/720-view renderer",
        ssim3d_protocol="Existing query.compute_ssim_3d; 7^3 Gaussian sigma=1.5, zero padding, population moments, data_range=1; unclipped volumes",
        ssim2d_protocol="skimage Gaussian sigma=1.5, 11x11, population moments, RGB channel average, data_range=1",
        lpips_protocol="lpips 0.1.4, LPIPS v0.1 AlexNet pretrained, RGB [-1,1], no resizing",
        renderer_protocol="experiments.render_preview.dvr; alpha=1-exp(-0.18*clip(v,0,1)); fixed RGB transfer function; no PNG quantization",
        mean_3d_psnr=float(np.mean([r['psnr3d'] for r in volumes])),
        mean_3d_ssim=float(np.mean([r['ssim3d'] for r in volumes])),
        mean_2d_psnr=float(np.mean([r['psnr2d'] for r in images])),
        mean_2d_ssim=float(np.mean([r['ssim2d'] for r in images])),
        mean_lpips_alex=float(np.mean([r['lpips_alex'] for r in images])),
        image_count=len(images),volume_count=len(volumes),
        full_volume_decode_mean_ms=float(np.mean(milliseconds)),
        full_volume_decode_median_ms=float(np.median(milliseconds)),
        decode_protocol="RTX5090; model resident; 3 warmups then 90 timestamps; synchronized wall time, 8 x 64^3 queries and GPU output assembly; excludes load, disk IO, CPU transfer, DVR",
        paper_comparable_render_fps=None,
        inference_file_bytes=(run/'model_inference.pt').stat().st_size,
        storage_ratio_vs_float32_raw=90*128**3*4/(run/'model_inference.pt').stat().st_size)
    dest=run/'supplemental_metrics'
    dest.mkdir(exist_ok=True)
    for name, obj in [('summary',summary),('volumes',volumes),('images',images),('decode_ms',milliseconds)]:
        (dest/f'{name}.json').write_text(json.dumps(obj,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
