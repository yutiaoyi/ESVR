"""Train anchor/offset scaffold + shared MLP decoders on a volume time series.

See docs/ESVR_SCAFFOLD4D.md for the math, the source-reading map, and
deliberate differences from 4D-Scaffold-GS and the other two dynamic
branches in this repository.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from dynamic_esvr.data import VolumeSequence
from scaffold4d.model import ScaffoldESVR, query_block


@torch.no_grad()
def monitor(model, sequence, indices, block):
    rows = []
    for index in indices:
        squared_error = 0.0
        for x in range(0, sequence.shape[0], block):
            for y in range(0, sequence.shape[1], block):
                for z in range(0, sequence.shape[2], block):
                    start = (x, y, z)
                    size = tuple(min(block, s - a) for s, a in zip(sequence.shape, start))
                    gt = sequence.volumes[index][tuple(slice(a, a + n) for a, n in zip(start, size))].cuda()
                    pred = query_block(model, sequence.times[index], sequence.shape, start, size)
                    squared_error += (pred.double() - gt.double()).square().sum().item()
        mse = squared_error / math.prod(sequence.shape)
        rows.append(dict(frame=index, time=sequence.frames[index]["time"], mse=mse,
                         psnr_db=-10 * math.log10(max(mse, 1e-30))))
    return rows


@torch.no_grad()
def evaluate(model, sequence, output, block=64):
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
                    pred[slices] = query_block(model, t, sequence.shape, start, size).cpu()
        if not torch.isfinite(pred).all():
            raise RuntimeError("Nonfinite reconstruction")
        mse = (pred.double() - gt.double()).square().mean().item()
        rows.append(dict(frame=index, path=sequence.frames[index]["path"], time=sequence.frames[index]["time"],
                         normalized_time=t, split=sequence.frames[index].get("split", "train"),
                         mse=mse, psnr_db=-10 * math.log10(max(mse, 1e-30)),
                         mae=(pred - gt).abs().mean().item(),
                         rmse_original_units=math.sqrt(mse) * (sequence.hi - sequence.lo)))
        np.save(output / f"frame_{index:04d}.npy", pred.numpy())
        raw = (pred.numpy() * (sequence.hi - sequence.lo) + sequence.lo)
        raw.transpose(tuple("xyz".index(a) for a in sequence.axes)).astype("<f4").tofile(output / f"frame_{index:04d}.raw")
        for axis in range(3):
            comparison = np.concatenate([dvr(v.numpy(), axis) for v in (gt, pred)], axis=1)
            Image.fromarray((comparison.clip(0, 1) * 255).astype(np.uint8)).save(output / f"frame_{index:04d}_axis{axis}_gt_pred.png")
        print("EVAL " + json.dumps(rows[-1]), flush=True)
        (output / "metrics.json").write_text(json.dumps(rows, indent=2))
    (output / "export_format.json").write_text(json.dumps(dict(
        npy="float32, normalized, XYZ, unclipped", raw_dtype="<f4", raw_axes=sequence.axes,
        raw_shape=[sequence.shape["xyz".index(a)] for a in sequence.axes],
        raw_units="original intensity units, unclipped",
        previews="GT left; prediction right; offline DVR, clipped only for display",
        note="No PLY export: anchors generate children via shared MLPs, not "
             "stored per-primitive parameters; a PLY snapshot at one time "
             "would not carry the generative model needed for any other time."), indent=2))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--anchors", type=int, default=6000)
    p.add_argument("--children", type=int, default=5)
    p.add_argument("--feat-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--temporal-beta", type=float, default=4.0)
    p.add_argument("--temporal-cutoff", type=float, default=1e-3)
    p.add_argument("--max-anchors", type=int, default=40000)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.005, help="LR for anchor/offset/log_offset_scale")
    p.add_argument("--feat-lr", type=float, default=0.005, help="LR for anchor_feat")
    p.add_argument("--mlp-lr", type=float, default=0.002, help="LR for the shared opacity/shape/flow heads")
    p.add_argument("--lambda-ssim", type=float, default=0.2)
    p.add_argument("--lambda-vol", type=float, default=0.01, help="Volume regularizer weight, matches the paper's own default")
    p.add_argument("--densify-from", type=int, default=500)
    p.add_argument("--densify-until", type=int, default=None, help="Default: 2/3 of --steps")
    p.add_argument("--densify-every", type=int, default=100)
    p.add_argument("--densify-grad-threshold", type=float, default=2e-4)
    p.add_argument("--densify-spatial-size", type=float, default=0.02)
    p.add_argument("--densify-temporal-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", help="Trusted local checkpoint; same manifest and training options required")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--tile-sampling", action="store_true", help="Sample nonoverlapping spatial tiles, including boundaries")
    p.add_argument("--eval-every", type=int, default=0, help="Monitor first/middle/last training frames every N steps")
    p.add_argument("--snapshot-every", type=int, default=0, help="Retain additional numbered checkpoints every N steps")
    args = p.parse_args()
    if args.densify_until is None:
        args.densify_until = 2 * args.steps // 3
    if args.steps < 1 or args.block < 1 or not 0 <= args.lambda_ssim <= 1:
        p.error("Invalid steps, block or SSIM weight")
    if min(args.lr, args.feat_lr, args.mlp_lr) <= 0:
        p.error("Invalid learning rate")
    if args.eval_only and not args.resume:
        p.error("--eval-only requires --resume")
    sequence = VolumeSequence(args.manifest)
    if len(sequence.train_indices) < 2:
        p.error("Dynamic training requires >=2 training times")
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
    config = saved["model_config"] if saved else dict(anchors=args.anchors, children=args.children,
        shape=sequence.shape, feat_dim=args.feat_dim, hidden=args.hidden,
        temporal_beta=args.temporal_beta, temporal_cutoff=args.temporal_cutoff)
    model = ScaffoldESVR(**config).cuda()
    if saved:
        if saved["manifest"] != sequence.config:
            p.error("Checkpoint manifest differs from supplied sequence")
        model.load_state_dict(saved["model"])
        if args.eval_only:
            print(json.dumps(evaluate(model, sequence, output, args.block), indent=2))
            return
        ignored = {"resume", "output", "eval_only"}
        for name, value in saved["args"].items():
            if name not in ignored and getattr(args, name) != value:
                p.error(f"Resume option differs: {name}")

    def build_optimizer():
        return torch.optim.Adam([
            {"params": [model.anchor, model.offset, model.log_offset_scale], "lr": args.lr},
            {"params": [model.anchor_feat], "lr": args.feat_lr},
            {"params": model.heads.parameters(), "lr": args.mlp_lr}])

    optimizer = build_optimizer()
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
    monitor_indices = sorted(set(sequence.train_indices[i] for i in (0, len(sequence.train_indices) // 2, -1)))
    frame_counts = saved.get("frame_counts", [0] * len(sequence.frames)) if saved else [0] * len(sequence.frames)
    with (output / "train.jsonl").open("w") as log:
        for step in range(start_step + 1, args.steps + 1):
            index = random.choice(sequence.train_indices)
            frame_counts[index] += 1
            size = tuple(min(args.block, s) for s in sequence.shape)
            start = tuple(random.randrange(s - n + 1) for s, n in zip(sequence.shape, size))
            if args.tile_sampling:
                start = tuple(random.randrange((s + args.block - 1) // args.block) * args.block for s in sequence.shape)
                size = tuple(min(args.block, s - a) for s, a in zip(sequence.shape, start))
            target = sequence.volumes[index][tuple(slice(a, a + n) for a, n in zip(start, size))].cuda()
            t = sequence.times[index]
            optimizer.zero_grad(set_to_none=True)
            package, sliced = query_block(model, t, sequence.shape, start, size, return_package=True)
            pred = package["vol"]
            l1 = (pred - target).abs().mean()
            loss = (1 - args.lambda_ssim) * l1
            if args.lambda_ssim:
                loss = loss + args.lambda_ssim * (1 - ssim(pred, target))
            if args.lambda_vol and len(sliced.get_xyz):
                loss = loss + args.lambda_vol * sliced.get_scaling.prod(dim=-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            if len(sliced.get_xyz):
                model.record_gradients(sliced, package["visibility_filter"])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            factor = 0.1 ** (step / args.steps)
            for group, lr in zip(optimizer.param_groups, [args.lr, args.feat_lr, args.mlp_lr]):
                group["lr"] = lr * factor
            optimizer.step()
            model.stabilize()
            if args.densify_from <= step <= args.densify_until and step % args.densify_every == 0:
                event = model.densify(optimizer, grad_threshold=args.densify_grad_threshold,
                    max_anchors=args.max_anchors, spatial_size=args.densify_spatial_size,
                    temporal_size=args.densify_temporal_size)
                print("DENSIFY " + json.dumps(dict(step=step, **event)), flush=True)
            if args.eval_every and (step == 1 or step % args.eval_every == 0 or step == args.steps):
                values = monitor(model, sequence, monitor_indices, args.block)
                with (output / "monitor.jsonl").open("a") as metrics_log:
                    metrics_log.write(json.dumps(dict(step=step, anchors=len(model.anchor), frames=values)) + "\n")
                print("MONITOR " + json.dumps(dict(step=step, anchors=len(model.anchor), frames=values)), flush=True)
            if step == 1 or step % 50 == 0 or step == args.steps:
                row = dict(step=step, frame=index, loss=loss.item(), l1=l1.item(), anchors=len(model.anchor),
                           active=len(sliced.get_xyz), seconds=time.perf_counter() - begin)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
            snapshot = args.snapshot_every and step % args.snapshot_every == 0
            if step % 500 == 0 or step == args.steps or snapshot:
                torch.save(dict(model_config=model.config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, args=vars(args), manifest=sequence.config,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    python_rng=random.getstate(), frame_counts=frame_counts), output / "checkpoint.tmp")
                (output / "checkpoint.tmp").replace(output / "checkpoint.pt")
                if snapshot:
                    import shutil
                    shutil.copyfile(output / "checkpoint.pt", output / f"checkpoint_{step:06d}.pt")
    (output / "frame_sampling_counts.json").write_text(json.dumps(frame_counts))
    rows = evaluate(model, sequence, output / "evaluation", args.block)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
