"""Post-training compaction of a scaffold4d checkpoint: drop children the
voxelizer can never draw (intensity below its 1/255 cutoff), drop anchors
left with no child, optionally store the rest in fp16, then re-evaluate so
the size/PSNR trade is measured rather than assumed.

Usage: compact_scaffold4d.py RUN_DIR --manifest M [--fp16] [--drop-floor]
Writes RUN_DIR/compact/{model_compact.pt, metrics.json, report.json}.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_esvr.data import VolumeSequence
from scaffold4d.model import ScaffoldESVR, query_block


def size_bytes(state):
    return sum(t.numel() * t.element_size() for t in state.values())


@torch.no_grad()
def psnr_all_frames(model, sequence, block=64):
    rows = []
    for index, (t, gt) in enumerate(zip(sequence.times, sequence.volumes)):
        pred = torch.empty(sequence.shape)
        for x in range(0, sequence.shape[0], block):
            for y in range(0, sequence.shape[1], block):
                for z in range(0, sequence.shape[2], block):
                    start = (x, y, z)
                    size = tuple(min(block, s - a) for s, a in zip(sequence.shape, start))
                    pred[tuple(slice(a, a + n) for a, n in zip(start, size))] = query_block(model, t, sequence.shape, start, size).cpu()
        mse = (pred.double() - gt.double()).square().mean().item()
        rows.append(dict(frame=index, mse=mse, psnr_db=-10 * math.log10(max(mse, 1e-30))))
    return rows


class CompactScaffold(ScaffoldESVR):
    """ScaffoldESVR whose children carry a keep-mask: masked children are
    excluded from every query exactly as if the voxelizer had skipped them."""

    def at(self, timestamp, generated=None):
        g = generated if generated is not None else self.generate()
        sliced = super().at(timestamp, g)
        keep = self.child_keep.reshape(-1)[sliced.child_index]
        for name in ("get_xyz", "get_scaling", "get_rotation", "get_opacity", "opacity_t", "sigma", "child_index"):
            setattr(sliced, name, getattr(sliced, name)[keep])
        return sliced


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--fp16", action="store_true", help="Store anchor tensors in fp16 (MLPs stay fp32)")
    p.add_argument("--drop-floor", action="store_true",
                   help="Also drop children pinned on the scale floor (they can still touch a voxel centre, so this is not exact)")
    p.add_argument("--dark", type=float, default=1 / 255, help="Intensity below which the kernel never draws a child")
    args = p.parse_args()
    run = Path(args.run)
    out = run / "compact"
    out.mkdir(exist_ok=True)
    sequence = VolumeSequence(args.manifest)
    ck = torch.load(run / "checkpoint.pt", map_location="cuda", weights_only=False)
    cfg = ck["model_config"]
    model = ScaffoldESVR(**cfg).cuda()
    model.load_state_dict(ck["model"], strict=False)
    before = size_bytes({k: v for k, v in model.state_dict().items()})

    with torch.no_grad():
        g = model.generate()
        intensity = g.opacity_raw.sigmoid()[..., 0]
        keep = intensity >= args.dark
        if args.drop_floor:
            keep &= g.scale.amax(-1) > cfg["scale_floor"] * 1.2
        anchor_keep = keep.any(1)
    n, k = cfg["anchors"], cfg["children"]
    report = dict(run=str(run), anchors_before=n, children_before=n * k,
                  children_kept=int(keep.sum()), anchors_kept=int(anchor_keep.sum()),
                  fp16=args.fp16, drop_floor=args.drop_floor)

    compact = CompactScaffold(**{**cfg, "anchors": int(anchor_keep.sum())}).cuda()
    with torch.no_grad():
        for name in ("anchor", "anchor_feat", "offset", "log_offset_scale"):
            getattr(compact, name).copy_(getattr(model, name)[anchor_keep])
        compact.heads.load_state_dict(model.heads.state_dict())
        compact.register_buffer("child_keep", keep[anchor_keep].clone())
        if args.fp16:
            # Round-trip through fp16 so the evaluation sees exactly what will be stored.
            for name in ("anchor", "anchor_feat", "offset", "log_offset_scale"):
                t = getattr(compact, name)
                t.copy_(t.half().float())

    # Storage: anchor tensors (fp16 or fp32) for kept anchors, kept-children
    # offsets only (dead children's offsets are not stored), a bit-mask per
    # child, and the fp32 MLP heads.
    dtype_bytes = 2 if args.fp16 else 4
    kept_children = int(keep.sum())
    kept_anchors = int(anchor_keep.sum())
    stored = (kept_anchors * (4 + cfg["feat_dim"] + 7) * dtype_bytes + kept_children * 4 * dtype_bytes
              + math.ceil(kept_anchors * k / 8) + size_bytes(compact.heads.state_dict()))
    report.update(bytes_before=before, bytes_after=stored, mb_before=round(before / 1e6, 2), mb_after=round(stored / 1e6, 2),
                  ratio=round(before / stored, 2))

    torch.save(dict(model_config={**cfg, "anchors": kept_anchors},
                    anchor=compact.anchor.detach().half().cpu() if args.fp16 else compact.anchor.detach().cpu(),
                    anchor_feat=compact.anchor_feat.detach().half().cpu() if args.fp16 else compact.anchor_feat.detach().cpu(),
                    log_offset_scale=compact.log_offset_scale.detach().half().cpu() if args.fp16 else compact.log_offset_scale.detach().cpu(),
                    offset_kept=(compact.offset.detach()[compact.child_keep].half() if args.fp16 else compact.offset.detach()[compact.child_keep]).cpu(),
                    child_keep=np.packbits(compact.child_keep.cpu().numpy().reshape(-1)),
                    heads=compact.heads.state_dict()), out / "model_compact.pt")

    rows = psnr_all_frames(compact, sequence)
    original = json.load(open(run / "evaluation" / "metrics.json"))
    report.update(psnr_after=round(float(np.mean([r["psnr_db"] for r in rows])), 3),
                  psnr_before=round(float(np.mean([r["psnr_db"] for r in original])), 3))
    report["psnr_delta"] = round(report["psnr_after"] - report["psnr_before"], 3)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
