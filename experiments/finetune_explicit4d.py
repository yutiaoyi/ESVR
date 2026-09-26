"""Distil a trained ScaffoldESVR into explicit 4D ESVR primitives and keep
training them on all frames (no anchors, no MLP). Diagnostic + candidate method.

Question it answers: U1 (output/experiments/unshackle) showed that freeing one
time slice's primitives gains +4.1 dB on that frame. That frees two kinds of
sharing at once -- the shared MLP decoding and the sharing of each primitive
across ~13 frames. Here only the MLP sharing is removed: every child keeps its
home time, temporal gate and linear motion, but all of its parameters are free.

Usage: finetune_explicit4d.py SRC_RUN --manifest M --output OUT [--steps 60000]
Learning rates default to ESVR's own values times its spatial_lr_scale (as in
U1); the time parameters (home_t, flow, log_sigma) get small decaying rates.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_esvr.data import VolumeSequence  # noqa: E402
from scaffold4d.explicit import ExplicitESVR4D  # noqa: E402
from scaffold4d.model import ScaffoldESVR, query_block  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


def expon(step, a, b, n):
    s = min(max(step / n, 0.0), 1.0)
    return math.exp(math.log(a) * (1 - s) + math.log(b) * s)


@torch.no_grad()
def psnr_frames(model, seq, indices):
    out = []
    for i in indices:
        pred = query_block(model, seq.times[i], seq.shape, (0, 0, 0), seq.shape)
        mse = (pred.double() - seq.volumes[i].cuda().double()).square().mean().item()
        out.append(dict(frame=i, time=seq.frames[i]["time"], mse=mse, psnr_db=-10 * math.log10(max(mse, 1e-30))))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--keep-floor", action="store_true", help="Keep bright children pinned on the scale floor (dropped by default, as in compaction)")
    p.add_argument("--lambda-ssim", type=float, default=0.2)
    p.add_argument("--lr-xyz", nargs=2, type=float, default=[8.8e-5, 1.1e-5])
    p.add_argument("--lr-logit", nargs=2, type=float, default=[5.5e-3, 5.5e-4])
    p.add_argument("--lr-scale", type=float, default=5.5e-4)
    p.add_argument("--lr-rotation", type=float, default=1.1e-4)
    p.add_argument("--lr-t", nargs=2, type=float, default=[1e-4, 1e-5])
    p.add_argument("--lr-flow", nargs=2, type=float, default=[1e-4, 1e-5])
    p.add_argument("--lr-sigma", nargs=2, type=float, default=[1e-3, 1e-4])
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        p.error("Choose a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    from utils.voxelizer_provenance import record as record_voxelizer
    record_voxelizer(out)
    seq = VolumeSequence(args.manifest)
    ck = torch.load(Path(args.src) / "checkpoint.pt", map_location="cuda", weights_only=False)
    src = ScaffoldESVR(**ck["model_config"]).cuda()
    src.load_state_dict(ck["model"], strict=False)
    with torch.no_grad():
        g = src.generate()
        keep = g.intensity[..., 0] >= 1 / 255
        if not args.keep_floor:
            keep &= g.scale.amax(-1) > ck["model_config"]["scale_floor"] * 1.2
    model = ExplicitESVR4D.from_scaffold(src, keep)
    del src, g
    torch.cuda.empty_cache()
    (out / "config.json").write_text(json.dumps({**vars(args), "primitives": model.config["n"], "model_config": model.config}, indent=2))
    monitor_idx = [0, len(seq.frames) // 2, len(seq.frames) - 1]
    start = psnr_frames(model, seq, monitor_idx)
    print("STEP0 " + json.dumps(start), flush=True)
    groups = {
        "xyz": ([model.home_xyz], args.lr_xyz), "logit": ([model.logit], args.lr_logit),
        "scale": ([model.log_scale], [args.lr_scale, args.lr_scale]), "rotation": ([model.rotation], [args.lr_rotation, args.lr_rotation]),
        "t": ([model.home_t], args.lr_t), "flow": ([model.flow], args.lr_flow), "sigma": ([model.log_sigma], args.lr_sigma)}
    opt = torch.optim.Adam([{"params": v[0], "lr": v[1][0], "name": k} for k, v in groups.items()], eps=1e-15)
    begin = time.perf_counter()
    with (out / "train.jsonl").open("w") as log, (out / "monitor.jsonl").open("w") as mon:
        mon.write(json.dumps(dict(step=0, frames=start)) + "\n")
        for step in range(1, args.steps + 1):
            i = random.choice(seq.train_indices)
            target = seq.volumes[i].cuda()
            pred = query_block(model, seq.times[i], seq.shape, (0, 0, 0), seq.shape)
            l1 = (pred - target).abs().mean()
            loss = (1 - args.lambda_ssim) * l1 + (args.lambda_ssim * (1 - ssim(pred, target)) if args.lambda_ssim else 0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for gr in opt.param_groups:
                a, b = groups[gr["name"]][1]
                gr["lr"] = expon(step, a, b, args.steps)
            opt.step()
            model.stabilize()
            if step % 100 == 0 or step == args.steps:
                log.write(json.dumps(dict(step=step, frame=i, loss=loss.item(), l1=l1.item(), seconds=time.perf_counter() - begin)) + "\n")
            if args.eval_every and step % args.eval_every == 0:
                vals = psnr_frames(model, seq, monitor_idx)
                mon.write(json.dumps(dict(step=step, frames=vals)) + "\n"); mon.flush()
                print("MONITOR " + json.dumps(dict(step=step, mean=float(np.mean([v["psnr_db"] for v in vals])))), flush=True)
    torch.save(dict(model_config=model.config, model=model.state_dict(), source=args.src, args=vars(args)), out / "checkpoint.pt")
    rows = psnr_frames(model, seq, range(len(seq.frames)))
    (out / "evaluation").mkdir(exist_ok=True)
    (out / "evaluation" / "metrics.json").write_text(json.dumps(rows, indent=2))
    ps = [r["psnr_db"] for r in rows]
    summary = dict(primitives=model.config["n"], mb_fp32=model.stored_bytes() / 1e6, mb_fp16=model.stored_bytes(True) / 1e6,
                   mean_psnr=float(np.mean(ps)), min_psnr=float(min(ps)), max_psnr=float(max(ps)), seconds=time.perf_counter() - begin)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
