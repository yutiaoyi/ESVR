"""Post-training compaction of a scaffold4d checkpoint: drop children the
voxelizer can never draw (intensity below its 1/255 cutoff), drop anchors
left with no child, optionally store the rest in fp16, then re-evaluate so
the size/PSNR trade is measured rather than assumed.

Reads RUN_DIR/checkpoint.pt (never writes to it). Writes
RUN_DIR/compact/{model_compact.pt, metrics.json, report.json}; the compact
file is self-contained (manifest + training args) and is accepted by
inference_scaffold4d.py --checkpoint.

Usage: compact_scaffold4d.py RUN_DIR --manifest M [--fp16] [--drop-floor]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_esvr.data import VolumeSequence
from scaffold4d.compact import build_compact, load_compact, save_compact, stored_bytes
from scaffold4d.model import ScaffoldESVR, query_block


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
    if ck["manifest"] != sequence.config:
        p.error("Checkpoint manifest differs from supplied sequence")
    cfg = ck["model_config"]
    model = ScaffoldESVR(**cfg).cuda()
    model.load_state_dict(ck["model"], strict=False)
    before = sum(t.numel() * t.element_size() for t in model.state_dict().values())

    with torch.no_grad():
        g = model.generate()
        keep = g.opacity_raw.sigmoid()[..., 0] >= args.dark
        if args.drop_floor:
            keep &= g.scale.amax(-1) > cfg["scale_floor"] * 1.2
    compact = build_compact(model, keep, fp16=args.fp16)
    save_compact(compact, out / "model_compact.pt", ck["manifest"], ck["args"], fp16=args.fp16)
    # Evaluate what was written, not the in-memory object.
    reloaded = load_compact(torch.load(out / "model_compact.pt", map_location="cuda", weights_only=False))
    rows = psnr_all_frames(reloaded, sequence)
    original = json.load(open(run / "evaluation" / "metrics.json"))
    stored = stored_bytes(compact, args.fp16)
    report = dict(run=str(run), anchors_before=cfg["anchors"], children_before=cfg["anchors"] * cfg["children"],
                  anchors_kept=compact.config["anchors"], children_kept=int(keep.sum()), fp16=args.fp16, drop_floor=args.drop_floor,
                  bytes_before=before, bytes_after=stored, mb_before=round(before / 1e6, 2), mb_after=round(stored / 1e6, 2),
                  ratio=round(before / stored, 2), file_mb=round((out / "model_compact.pt").stat().st_size / 1e6, 2),
                  psnr_before=round(float(np.mean([r["psnr_db"] for r in original])), 3),
                  psnr_after=round(float(np.mean([r["psnr_db"] for r in rows])), 3))
    report["psnr_delta"] = round(report["psnr_after"] - report["psnr_before"], 3)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
