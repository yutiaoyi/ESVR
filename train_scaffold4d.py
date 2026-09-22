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
from scaffold4d.model import ScaffoldESVR, anchors_from_sequence, query_block


def expon_lr(step, lr_init, lr_final, max_steps):
    """Log-linear (exponential) decay, matching 4D-Scaffold-GS's/3DGS's
    get_expon_lr_func (utils/general_utils.py) with lr_delay_steps=0 (i.e.
    without the warmup branch: upstream sets lr_delay_mult on every group
    but never sets lr_delay_steps > 0 anywhere, so that branch of their own
    scheduler is dead code and warmup never actually applies -- verified by
    reading update_learning_rate's call sites, not assumed)."""
    if lr_init == 0.0 and lr_final == 0.0:
        return 0.0
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


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
    # Defaults below are 4D-Scaffold-GS's own values from arguments/__init__.py
    # and scripts/train_n3dv.sh (the script behind the paper's reported
    # numbers), not values chosen or tuned for this ESVR adaptation. See
    # docs/ESVR_SCAFFOLD4D.md for the source-reading map.
    p.add_argument("--steps", type=int, default=120000, help="train_n3dv.sh's --iterations")
    p.add_argument("--anchors", type=int, default=6000)
    p.add_argument("--children", type=int, default=10, help="Native n_offsets")
    p.add_argument("--feat-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--temporal-beta", type=float, default=4.0, help="Native hparam_beta")
    p.add_argument("--temporal-cutoff", type=float, default=1e-3,
                   help="ESVR-only efficiency cutoff; no native equivalent (upstream alpha-composites, no hard inclusion gate)")
    p.add_argument("--max-anchors", type=int, default=40000)
    # ESVR-specific adaptation choices (no native equivalent). Only the scale
    # floor measurably matters (output/scaffold4d/ablation/RESULTS.md); the
    # others default to the native/simplest setting and stay available.
    p.add_argument("--flow-scale", type=float, default=1.0,
                   help="Velocity bound in normalized units per unit time (native tanh bound: 1)")
    p.add_argument("--scale-floor-voxels", type=float, default=0.5, help="Floor on a child's final scale, in voxels")
    p.add_argument("--opacity-init", type=float, default=0.5, help="Initial sigmoid amplitude of every child (0.5 = zero bias)")
    p.add_argument("--init", choices=("random", "data"), default="random",
                   help="random: uniform; data: sample anchors from ground-truth intensity over all training frames (stand-in for upstream's SfM points)")
    p.add_argument("--sigma-max", type=float, default=1e2, help="Upper clamp on the temporal inverse-scale (numerical guardrail; native: none)")
    p.add_argument("--offset-spread", type=float, default=0.0,
                   help="Initial child offsets ~U(-s, s) in offset units (native: 0, all children stacked at the anchor)")
    p.add_argument("--child-scale-init-voxels", type=float, default=None,
                   help="Initial final-scale of children in voxels for initial and grown anchors (native: anchor's own cell size)")
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--tile-sampling", action="store_true", help="ESVR-only spatial sampling; no native equivalent (upstream samples camera rays)")
    # Anchor position is frozen (lr 0) natively -- not a tunable default here,
    # it is the paper's own design (scene/gaussian_model.py: position_lr_init
    # = position_lr_final = 0.0). Exposed only so config.json records it.
    p.add_argument("--anchor-lr", type=float, default=0.0)
    p.add_argument("--offset-lr-init", type=float, default=0.01)
    p.add_argument("--offset-lr-final", type=float, default=0.0001)
    p.add_argument("--feat-lr", type=float, default=0.0075, help="Native feature_lr; constant, no decay")
    p.add_argument("--scale-lr", type=float, default=0.007, help="Native scaling_lr (maps to log_offset_scale); constant, no decay")
    p.add_argument("--opacity-mlp-lr-init", type=float, default=0.002)
    p.add_argument("--opacity-mlp-lr-final", type=float, default=0.00002)
    p.add_argument("--shape-mlp-lr", type=float, default=0.004, help="Native mlp_cov_lr; constant, no decay")
    p.add_argument("--flow-mlp-lr", type=float, default=0.001, help="Native mlp_flow_lr; constant, no decay")
    p.add_argument("--lambda-ssim", type=float, default=0.2, help="Native lambda_dssim")
    p.add_argument("--lambda-vol", type=float, default=0.01, help="Native volume regularizer weight")
    p.add_argument("--vol-reg-visible", action="store_true",
                   help="Apply the volume regularizer only to children whose footprint intersects the block (ESVR has no pixel dilation, so an uncovered child otherwise only ever shrinks)")
    p.add_argument("--prune-dead", action="store_true",
                   help="ESVR prune_zeros/dark rules per anchor: drop anchors whose children all covered no voxel or are below 1/255")
    p.add_argument("--start-stat", type=int, default=500, help="Native start_stat")
    p.add_argument("--densify-from", type=int, default=1500, help="Native update_from")
    p.add_argument("--densify-until", type=int, default=None, help="Native update_until; default steps//2 (train_n3dv.sh: 60000 of 120000)")
    p.add_argument("--densify-every", type=int, default=100, help="Native update_interval")
    p.add_argument("--densify-grad-threshold", type=float, default=1.2e-3,
                   help="Native densify_grad_threshold is 2e-4 for 2D view-space gradients; here it thresholds 3D centre gradients of a per-voxel-mean loss (measured p90 ~1.2e-3 with the fixed voxelizer; the earlier 4e-7 was measured through the unfixed one, which scales position gradients by 1/4096)")
    p.add_argument("--success-threshold", type=float, default=0.8, help="Native success_threshold")
    p.add_argument("--min-opacity", type=float, default=0.005, help="Native min_opacity (anchor pruning)")
    p.add_argument("--lambda-temporal-sigma", type=float, default=1.0, help="Native gamma: gradient weight opacity_t * sigma^gamma")
    p.add_argument("--grow-voxel-size", type=float, default=1.0, help="Finest anchor-grid cell, in data voxels (native voxel_size, scene units)")
    p.add_argument("--grow-t-size", type=float, default=None, help="Temporal grid cell in normalized time; default one frame interval (paper)")
    p.add_argument("--grow-init-factor", type=int, default=16, help="Native update_init_factor")
    p.add_argument("--grow-hierarchy-factor", type=int, default=4, help="Native update_hierachy_factor")
    p.add_argument("--grow-depth", type=int, default=3, help="Native update_depth")
    p.add_argument("--grow-all-candidates", action="store_true", help="Native disable_partial_densify")
    p.add_argument("--grow-upstream-level-skip", action="store_true",
                   help="Upstream anchor_growing quirk: run finer grid levels only if a coarser level grew anchors this round")
    p.add_argument("--final-lr-scale", type=float, default=1.0,
                   help="Log-linear decay of the natively constant groups (scale, feature, shape/flow MLP) to lr*scale at --steps. 1 = native (no decay). The ESVR adaptation recipe uses 0.01 (+5 dB at 150k anchors, output/scaffold4d/decay/RESULTS.md); kept off by default so argparse defaults stay the paper's")
    p.add_argument("--adam-eps", type=float, default=1e-15, help="Native Adam eps")
    p.add_argument("--clip-grad", type=float, default=0.0, help="Global grad-norm clip; 0 disables (native: none)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", help="Trusted local checkpoint; same manifest and training options required")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-every", type=int, default=0, help="Monitor first/middle/last training frames every N steps")
    p.add_argument("--snapshot-every", type=int, default=0, help="Retain additional numbered checkpoints every N steps")
    args = p.parse_args()
    if args.densify_until is None:
        args.densify_until = args.steps // 2
    if args.steps < 1 or args.block < 1 or not 0 <= args.lambda_ssim <= 1:
        p.error("Invalid steps, block or SSIM weight")
    if min(args.offset_lr_init, args.offset_lr_final, args.feat_lr, args.scale_lr, args.opacity_mlp_lr_init,
           args.opacity_mlp_lr_final, args.shape_mlp_lr, args.flow_mlp_lr) <= 0 or args.anchor_lr < 0:
        p.error("Invalid learning rate")
    if (min(args.flow_scale, args.scale_floor_voxels, args.grow_voxel_size, args.adam_eps, args.sigma_max, args.final_lr_scale) <= 0
            or args.final_lr_scale > 1
            or args.offset_spread < 0 or (args.child_scale_init_voxels is not None and args.child_scale_init_voxels <= 0)
            or not 0 < args.opacity_init < 1 or args.clip_grad < 0 or args.min_opacity < 0
            or not 0 < args.success_threshold <= 1 or args.grow_depth < 1 or args.grow_init_factor < 1
            or args.grow_hierarchy_factor < 2 or args.max_anchors < 1):
        p.error("Invalid adaptation/growing option")
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
    voxel = 2 / max(sequence.shape)
    grow_voxel_size = args.grow_voxel_size * voxel
    grow_t_size = args.grow_t_size or 1.0 / max(len(sequence.train_indices) - 1, 1)
    saved = torch.load(args.resume, map_location="cuda", weights_only=False) if args.resume else None
    if saved:
        model = ScaffoldESVR(**saved["model_config"]).cuda()
        if saved["manifest"] != sequence.config:
            p.error("Checkpoint manifest differs from supplied sequence")
        # Growing statistics are non-persistent buffers; older checkpoints
        # carried a different set of (also transient) statistics.
        model.load_state_dict(saved["model"], strict=False)
    else:
        anchor_init = None
        if args.init == "data":
            anchor_init = anchors_from_sequence(sequence, args.anchors, grow_voxel_size, grow_t_size,
                                                generator=torch.Generator().manual_seed(args.seed))
        model = ScaffoldESVR(anchors=args.anchors, children=args.children, shape=sequence.shape,
                             feat_dim=args.feat_dim, hidden=args.hidden, temporal_beta=args.temporal_beta,
                             temporal_cutoff=args.temporal_cutoff, flow_scale=args.flow_scale,
                             scale_floor=args.scale_floor_voxels * voxel, opacity_init=args.opacity_init,
                             anchor_init=anchor_init, sigma_max=args.sigma_max, offset_spread=args.offset_spread,
                             child_scale_init=None if args.child_scale_init_voxels is None else args.child_scale_init_voxels * voxel).cuda()
    if saved:
        if args.eval_only:
            print(json.dumps(evaluate(model, sequence, output, args.block), indent=2))
            return
        ignored = {"resume", "output", "eval_only"}
        for name, value in saved["args"].items():
            if name not in ignored and getattr(args, name) != value:
                p.error(f"Resume option differs: {name}")

    def build_optimizer():
        # Seven native optimizer groups (scene/gaussian_model.py:
        # training_setup), not the collapsed 3-group scheme an earlier
        # version of this script used. Most critically: anchor position gets
        # its own group at lr 0 (frozen) instead of being lumped with offset
        # -- conflating the two was a real bug, not a stylistic choice, see
        # docs/ESVR_SCAFFOLD4D.md.
        return torch.optim.Adam([
            {"params": [model.anchor], "lr": args.anchor_lr, "name": "anchor"},
            {"params": [model.offset], "lr": args.offset_lr_init, "name": "offset"},
            {"params": [model.log_offset_scale], "lr": args.scale_lr, "name": "scale"},
            {"params": [model.anchor_feat], "lr": args.feat_lr, "name": "feat"},
            {"params": model.heads["opacity"].parameters(), "lr": args.opacity_mlp_lr_init, "name": "opacity_mlp"},
            {"params": model.heads["shape"].parameters(), "lr": args.shape_mlp_lr, "name": "shape_mlp"},
            {"params": model.heads["flow"].parameters(), "lr": args.flow_mlp_lr, "name": "flow_mlp"}],
            eps=args.adam_eps)

    fin = args.final_lr_scale
    lr_schedule = dict(
        anchor=lambda step: expon_lr(step, args.anchor_lr, args.anchor_lr, args.steps),
        offset=lambda step: expon_lr(step, args.offset_lr_init, args.offset_lr_final, args.steps),
        scale=lambda step: expon_lr(step, args.scale_lr, args.scale_lr * fin, args.steps),
        feat=lambda step: expon_lr(step, args.feat_lr, args.feat_lr * fin, args.steps),
        opacity_mlp=lambda step: expon_lr(step, args.opacity_mlp_lr_init, args.opacity_mlp_lr_final, args.steps),
        shape_mlp=lambda step: expon_lr(step, args.shape_mlp_lr, args.shape_mlp_lr * fin, args.steps),
        flow_mlp=lambda step: expon_lr(step, args.flow_mlp_lr, args.flow_mlp_lr * fin, args.steps))

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
    from utils.voxelizer_provenance import record as record_voxelizer
    record_voxelizer(output)
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
                scaling = sliced.get_scaling[package["visibility_filter"]] if args.vol_reg_visible else sliced.get_scaling
                if len(scaling):
                    loss = loss + args.lambda_vol * scaling.prod(dim=-1).mean()
            if not torch.isfinite(loss):
                diag = dict(step=step, pred_min=pred.min().item(), pred_max=pred.max().item(),
                    active=len(sliced.get_xyz),
                    scale_min=sliced.get_scaling.min().item() if len(sliced.get_xyz) else None,
                    scale_max=sliced.get_scaling.max().item() if len(sliced.get_xyz) else None)
                print("NONFINITE " + json.dumps(diag), flush=True)
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            # Native schedule (train.py): statistics between start_stat and
            # update_until, anchor adjustment every update_interval after update_from.
            if args.start_stat < step < args.densify_until and len(sliced.get_xyz):
                model.record_statistics(sliced, package["visibility_filter"], args.lambda_temporal_sigma)
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
            for group in optimizer.param_groups:
                group["lr"] = lr_schedule[group["name"]](step)
            optimizer.step()
            model.stabilize()
            if args.densify_from < step < args.densify_until and step % args.densify_every == 0:
                # Upstream renders whole frames, so a Gaussian is observed on
                # every step; here a child is only observed on steps whose
                # block contains it, so the "observed enough" interval is
                # scaled by the block's share of the volume.
                block_fraction = math.prod(min(args.block, s) / s for s in sequence.shape)
                event = model.adjust_anchors(optimizer, check_interval=args.densify_every * block_fraction,
                    success_threshold=args.success_threshold, grad_threshold=args.densify_grad_threshold,
                    min_opacity=args.min_opacity, voxel_size=grow_voxel_size, t_size=grow_t_size,
                    init_factor=args.grow_init_factor, hierarchy_factor=args.grow_hierarchy_factor,
                    depth=args.grow_depth, max_anchors=args.max_anchors, partial=not args.grow_all_candidates,
                    prune_dead=args.prune_dead, upstream_level_skip=args.grow_upstream_level_skip)
                print("DENSIFY " + json.dumps(dict(step=step, **event)), flush=True)
                with (output / "densify.jsonl").open("a") as densify_log:
                    densify_log.write(json.dumps(dict(step=step, **event)) + "\n")
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
