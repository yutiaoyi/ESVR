"""Reconstruct an arbitrary time from a trusted local ESVR4D checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from esvr4d.model import ESVR4D, query_block


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--time", type=float, required=True, help="Time in original manifest units")
    parser.add_argument("--output", required=True)
    parser.add_argument("--block", type=int, default=64)
    args = parser.parse_args()
    if args.block < 1 or not np.isfinite(args.time):
        parser.error("Invalid block or time")
    saved = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    manifest = saved["manifest"]
    times = [f["time"] for f in manifest["frames"]]
    lo, hi = manifest.get("time_range", [min(times), max(times)])
    if not lo <= args.time <= hi:
        parser.error("Time outside sequence range; extrapolation is not validated")
    t = (args.time - lo) / (hi - lo) if hi > lo else 0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Choose an empty output directory")
    model = ESVR4D(**saved["model_config"]).cuda()
    model.load_state_dict(saved["model"])
    shape = model.config["shape"]
    volume = np.empty(shape, dtype=np.float32)
    for x in range(0, shape[0], args.block):
        for y in range(0, shape[1], args.block):
            for z in range(0, shape[2], args.block):
                start = (x, y, z)
                size = tuple(min(args.block, n - a) for n, a in zip(shape, start))
                slices = tuple(slice(a, a + n) for a, n in zip(start, size))
                volume[slices] = query_block(model, t, shape, start, size).cpu().numpy()
    if not np.isfinite(volume).all():
        raise RuntimeError("Nonfinite reconstruction")
    np.save(output / "volume_xyz_normalized.npy", volume)
    a, b = manifest["intensity_range"]
    axes = manifest.get("storage_axes", "zyx")
    (volume * (b - a) + a).transpose(tuple("xyz".index(c) for c in axes)).astype("<f4").tofile(output / "volume.raw")
    from experiments.render_preview import dvr
    for axis in range(3):
        Image.fromarray((dvr(volume, axis).clip(0, 1) * 255).astype(np.uint8)).save(output / f"axis{axis}.png")
    (output / "metadata.json").write_text(json.dumps(dict(time=args.time, normalized_time=t,
        shape_xyz=shape, raw_dtype="<f4", raw_axes=axes, raw_units="original intensity units",
        checkpoint=str(Path(args.checkpoint).resolve()), points=len(model.xyz)), indent=2))


if __name__ == "__main__":
    main()
