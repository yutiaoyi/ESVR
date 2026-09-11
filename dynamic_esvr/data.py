"""Explicit time-series manifest: common spatial grid and common intensity range."""
import json
from pathlib import Path

import numpy as np
import torch


class VolumeSequence:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.config = json.loads(self.path.read_text())
        self.shape = tuple(self.config["shape_xyz"])
        self.axes = self.config.get("storage_axes", "zyx")
        self.dtype = np.dtype(self.config.get("dtype", "<f4"))
        self.lo, self.hi = self.config["intensity_range"]
        if (len(self.shape) != 3 or any(type(s) is not int or s < 1 for s in self.shape)
                or sorted(self.axes) != list("xyz") or not np.isfinite([self.lo, self.hi]).all()
                or self.hi <= self.lo):
            raise ValueError("Invalid grid, storage_axes or intensity_range")
        self.frames = self.config["frames"]
        times = [f["time"] for f in self.frames]
        if not times or not np.isfinite(times).all() or len(set(times)) != len(times):
            raise ValueError("Frames require unique finite times")
        self.time_range = self.config.get("time_range", [min(times), max(times)])
        a, b = self.time_range
        if not np.isfinite([a, b]).all() or b < a or min(times) < a or max(times) > b:
            raise ValueError("Invalid common time_range")
        self.times = [(t - a) / (b - a) if b > a else 0.0 for t in times]
        self.train_indices = [i for i, f in enumerate(self.frames) if f.get("split", "train") == "train"]
        if not self.train_indices or any(f.get("split", "train") not in ("train", "test") for f in self.frames):
            raise ValueError("Need train frames and valid train/test splits")
        self.volumes = [self._load(f) for f in self.frames]

    def _load(self, frame):
        path = (self.path.parent / frame["path"]).resolve()
        storage_shape = tuple(self.shape["xyz".index(a)] for a in self.axes)
        if path.suffix == ".npy":
            array = np.load(path, allow_pickle=False, mmap_mode="r")
        elif path.suffix.lower() in (".tif", ".tiff"):
            import tifffile
            array = tifffile.imread(path)
        else:
            if path.stat().st_size != int(np.prod(storage_shape)) * self.dtype.itemsize:
                raise ValueError(f"RAW byte count mismatch: {path}")
            array = np.memmap(path, dtype=self.dtype, mode="r", shape=storage_shape)
        if array.shape != storage_shape:
            raise ValueError(f"Storage shape mismatch: {path}: {array.shape} != {storage_shape}")
        array = np.asarray(array.transpose(tuple(self.axes.index(a) for a in "xyz")), dtype=np.float32)
        if not np.isfinite(array).all() or array.min() < self.lo - 1e-5 or array.max() > self.hi + 1e-5:
            raise ValueError(f"Nonfinite or out-of-range values: {path}; use one range for all frames")
        return torch.from_numpy(np.ascontiguousarray((array - self.lo) / (self.hi - self.lo)))
