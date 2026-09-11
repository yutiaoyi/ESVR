"""Train canonical scalar ellipsoids and a shared deformation field on volumes."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from dynamic_esvr.data import VolumeSequence
from dynamic_esvr.model import DynamicESVR, query_block


@torch.no_grad()
def monitor(model, sequence, indices, block, deformation):
    rows = []
    for index in indices:
        squared_error = 0.0
        for x in range(0, sequence.shape[0], block):
            for y in range(0, sequence.shape[1], block):
                for z in range(0, sequence.shape[2], block):
                    start = (x, y, z)
                    size = tuple(min(block, s-a) for s, a in zip(sequence.shape, start))
                    gt = sequence.volumes[index][tuple(slice(a,a+n) for a,n in zip(start,size))].cuda()
                    pred = query_block(model, sequence.times[index], sequence.shape, start, size, deformation)
                    squared_error += (pred.double()-gt.double()).square().sum().item()
        mse = squared_error / math.prod(sequence.shape)
        rows.append(dict(frame=index, time=sequence.frames[index]["time"], mse=mse,
                         psnr_db=-10 * math.log10(max(mse, 1e-30))))
    return rows


@torch.no_grad()
def evaluate(model, sequence, output, block=64, deformation=True):
    from PIL import Image
    from experiments.render_preview import dvr
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (t, gt) in enumerate(zip(sequence.times, sequence.volumes)):
        pred = torch.empty(sequence.shape)
        for x in range(0, sequence.shape[0], block):
            for y in range(0, sequence.shape[1], block):
                for z in range(0, sequence.shape[2], block):
                    start = (x, y, z)
                    size = tuple(min(block, s - a) for s, a in zip(sequence.shape, start))
                    slices = tuple(slice(a, a + n) for a, n in zip(start, size))
                    pred[slices] = query_block(model, t, sequence.shape, start, size, deformation).cpu()
        if not torch.isfinite(pred).all():
            raise RuntimeError("Nonfinite reconstruction")
        mse = (pred.double() - gt.double()).square().mean().item()
        rows.append(dict(frame=index, path=sequence.frames[index]["path"], time=sequence.frames[index]["time"],
                         normalized_time=t, split=sequence.frames[index].get("split", "train"),
                         mse=mse, psnr_db=-10 * math.log10(max(mse, 1e-30)),
                         mae=(pred - gt).abs().mean().item(),
                         rmse_original_units=math.sqrt(mse) * (sequence.hi - sequence.lo)))
        # Unclipped normalized XYZ volume; RAW restores original units and storage axes.
        np.save(output / f"frame_{index:04d}.npy", pred.numpy())
        raw = (pred.numpy() * (sequence.hi - sequence.lo) + sequence.lo)
        raw.transpose(tuple("xyz".index(a) for a in sequence.axes)).astype("<f4").tofile(output / f"frame_{index:04d}.raw")
        for axis in range(3):
            comparison = np.concatenate([dvr(v.numpy(), axis) for v in (gt, pred)], axis=1)
            Image.fromarray((comparison.clip(0, 1) * 255).astype(np.uint8)).save(output / f"frame_{index:04d}_axis{axis}_gt_pred.png")
        model.export_ply(output / f"frame_{index:04d}.ply", t, deformation)
        print("EVAL " + json.dumps(rows[-1]), flush=True)
        (output / "metrics.json").write_text(json.dumps(rows, indent=2))
    (output / "metrics.json").write_text(json.dumps(rows, indent=2))
    (output / "export_format.json").write_text(json.dumps(dict(
        npy="float32, normalized, XYZ, unclipped", raw_dtype="<f4", raw_axes=sequence.axes,
        raw_shape=[sequence.shape["xyz".index(a)] for a in sequence.axes],
        raw_units="original intensity units, unclipped", previews="GT left; prediction right; offline DVR, clipped only for display"), indent=2))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--points", type=int, default=20000)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--xyz-bands", type=int, default=10)
    p.add_argument("--time-bands", type=int, default=10)
    p.add_argument("--scalar-deformation", action="store_true")
    p.add_argument("--static", action="store_true", help="Same shared canonical model, no deformation (ablation)")
    p.add_argument("--allow-single-frame", action="store_true")
    p.add_argument("--init-ply")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--deform-lr", type=float, default=0.0008)
    p.add_argument("--lambda-ssim", type=float, default=0.2)
    p.add_argument("--motion-reg", type=float, default=0.0001)
    p.add_argument("--time-jitter", type=float, default=0.0, help="Normalized-time stddev, decays linearly; disabled for exact volume timestamps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", help="Trusted local checkpoint; same manifest and training options required")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--tile-sampling", action="store_true", help="Sample nonoverlapping spatial tiles, including boundaries")
    p.add_argument("--eval-every", type=int, default=0, help="Monitor first/middle/last training frames every N steps")
    p.add_argument("--snapshot-every", type=int, default=0, help="Retain additional numbered checkpoints every N steps")
    args = p.parse_args()
    if args.steps < 1 or args.block < 1 or min(args.warmup, args.eval_every, args.snapshot_every) < 0 or not 0 <= args.lambda_ssim <= 1:
        p.error("Invalid steps, block, warmup or SSIM weight")
    if min(args.lr, args.deform_lr) <= 0 or min(args.motion_reg, args.time_jitter) < 0:
        p.error("Invalid learning rate or regularization")
    if args.eval_only and not args.resume:
        p.error("--eval-only requires --resume")
    sequence = VolumeSequence(args.manifest)
    if len(sequence.train_indices) < 2 and not args.allow_single_frame:
        p.error("Dynamic training requires >=2 training times; --allow-single-frame is a static smoke test only")
    if not torch.cuda.is_available():
        raise RuntimeError("ESVR's voxelizer requires a CUDA GPU")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.eval_only and any(output.iterdir()):
        p.error("Choose a new empty output directory (also when resuming)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    saved = torch.load(args.resume, map_location="cuda", weights_only=False) if args.resume else None
    config = saved["model_config"] if saved else dict(points=args.points, shape=sequence.shape,
        depth=args.depth, width=args.width, xyz_bands=args.xyz_bands, time_bands=args.time_bands,
        scalar=args.scalar_deformation)
    model = DynamicESVR(**config).cuda()
    if saved:
        if saved["manifest"] != sequence.config:
            p.error("Checkpoint manifest differs from supplied sequence")
        model.load_state_dict(saved["model"])
        if args.eval_only:
            print(json.dumps(evaluate(model, sequence, output, args.block, saved["deformation_active"]), indent=2))
            return
        ignored = {"resume", "output", "eval_only", "init_ply"}
        for name, value in saved["args"].items():
            if name not in ignored and getattr(args, name) != value:
                p.error(f"Resume option differs: {name}")
    elif args.init_ply:
        model.initialize_ply(args.init_ply)
    canonical = [model.xyz, model.log_scale, model.rotation, model.logit_scalar]
    optimizer = torch.optim.Adam([{"params": canonical, "lr": args.lr},
                                  {"params": model.deform.parameters(), "lr": args.deform_lr}])
    start_step = 0
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        start_step = saved["step"]
        if start_step >= args.steps:
            p.error("Checkpoint already completed this schedule; use --eval-only")
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        random.setstate(saved["python_rng"])
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    (output / "manifest.json").write_text(json.dumps(sequence.config, indent=2))
    from utils.loss_utils import ssim
    begin = time.perf_counter()
    active = False
    monitor_indices = sorted(set(sequence.train_indices[i] for i in (0, len(sequence.train_indices)//2, -1)))
    frame_counts = saved.get("frame_counts", [0]*len(sequence.frames)) if saved else [0]*len(sequence.frames)
    with (output / "train.jsonl").open("w") as log:
        for step in range(start_step + 1, args.steps + 1):
            index = random.choice(sequence.train_indices)
            frame_counts[index] += 1
            size = tuple(min(args.block, s) for s in sequence.shape)
            start = tuple(random.randrange(s - n + 1) for s, n in zip(sequence.shape, size))
            if args.tile_sampling:
                start = tuple(random.randrange((s + args.block - 1)//args.block)*args.block for s in sequence.shape)
                size = tuple(min(args.block, s-a) for s,a in zip(sequence.shape,start))
            target = sequence.volumes[index][tuple(slice(a, a+n) for a, n in zip(start, size))].cuda()
            t = sequence.times[index]
            active = not args.static and step > args.warmup and len(sequence.train_indices) > 1
            if active and args.time_jitter:
                t += random.gauss(0, args.time_jitter * (1 - step / args.steps))
                t = min(1.0, max(0.0, t))
            optimizer.zero_grad(set_to_none=True)
            pred = query_block(model, t, sequence.shape, start, size, active)
            l1 = (pred - target).abs().mean()
            loss = (1 - args.lambda_ssim) * l1
            if args.lambda_ssim:
                loss = loss + args.lambda_ssim * (1 - ssim(pred, target))
            if active and args.motion_reg:
                delta = model.deform(model.xyz.detach(), t)
                loss = loss + args.motion_reg * sum(v.square().mean() for v in delta.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            factor = 0.1 ** (step / args.steps)
            for group, lr in zip(optimizer.param_groups, [args.lr, args.deform_lr]):
                group["lr"] = lr * factor
            optimizer.step()
            if args.eval_every and (step == 1 or step % args.eval_every == 0 or step == args.steps):
                values = monitor(model, sequence, monitor_indices, args.block, active)
                with (output / "monitor.jsonl").open("a") as metrics_log:
                    metrics_log.write(json.dumps(dict(step=step, frames=values)) + "\n")
                print("MONITOR " + json.dumps(dict(step=step, frames=values)), flush=True)
            if step == 1 or step % 50 == 0 or step == args.steps:
                row = dict(step=step, frame=index, loss=loss.item(), l1=l1.item(),
                           deformation=active, seconds=time.perf_counter() - begin)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
            snapshot = args.snapshot_every and step % args.snapshot_every == 0
            if step % 500 == 0 or step == args.steps or snapshot:
                torch.save(dict(model_config=model.config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, args=vars(args), manifest=sequence.config, deformation_active=active,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    python_rng=random.getstate(), frame_counts=frame_counts), output / "checkpoint.tmp")
                (output / "checkpoint.tmp").replace(output / "checkpoint.pt")
                if snapshot:
                    import shutil
                    shutil.copyfile(output / "checkpoint.pt", output / f"checkpoint_{step:06d}.pt")
    (output / "frame_sampling_counts.json").write_text(json.dumps(frame_counts))
    rows = evaluate(model, sequence, output / "evaluation", args.block, active)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
