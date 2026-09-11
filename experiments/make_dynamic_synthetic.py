"""Generate a labeled synthetic moving anisotropic blob (not user data)."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()
    path = Path(args.output).resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        parser.error("Output must be empty")
    a = (np.arange(args.size) + 0.5) * 2 / args.size - 1
    x, y, z = np.meshgrid(a, a, a, indexing="ij")
    frames = []
    for i, t in enumerate([0, 0.25, 0.5, 0.75, 1]):
        power = ((x - (t - 0.5) * 0.65) / 0.28) ** 2 + (y / 0.19) ** 2 + (z / 0.35) ** 2
        volume = (0.7 / (1 + np.exp(np.minimum(5 * (power - 1), 80)))).astype(np.float32)
        volume[power > 2] = 0
        np.save(path / f"frame_{i:02d}.npy", volume)
        frames.append(dict(path=f"frame_{i:02d}.npy", time=t, split="test" if i in (1, 3) else "train"))
    (path / "sequence.json").write_text(json.dumps(dict(shape_xyz=[args.size]*3, storage_axes="xyz",
        dtype="<f4", intensity_range=[0, 1], frames=frames), indent=2))


if __name__ == "__main__":
    main()
