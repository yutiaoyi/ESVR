"""Error-proportional training-block sampling (ESVR-side addition; neither
ESVR nor 4D-Scaffold-GS has it -- ESVR draws sub-volumes uniformly, the
upstream renders whole camera frames).

Keeps one squared-error value per (frame, tile) from the last time that
block was trained on, and samples the next (frame, tile) with probability
(1 - uniform) * err**alpha / sum + uniform / N. Unvisited cells count as the
current maximum error so every cell is trained at least once early on; the
uniform share keeps well-fitted regions from being abandoned.
"""
import math

import numpy as np


class ErrorTable:
    def __init__(self, n_frames, shape, block, alpha=1.0, uniform=0.25, state=None):
        if alpha < 0 or not 0 < uniform <= 1:
            raise ValueError("alpha must be >= 0 and 0 < uniform <= 1")
        self.shape = tuple(shape)
        self.block = block
        self.tiles_per_axis = tuple((s + block - 1) // block for s in self.shape)
        self.n_tiles = math.prod(self.tiles_per_axis)
        self.alpha = alpha
        self.uniform = uniform
        if state is None:
            self.table = np.full((n_frames, self.n_tiles), np.nan)
        else:
            self.table = np.asarray(state, dtype=float)
            if self.table.shape != (n_frames, self.n_tiles):
                raise ValueError("Saved error table does not match this sequence/block")

    def tile_start(self, tile):
        idx = np.unravel_index(tile, self.tiles_per_axis)
        return tuple(int(i) * self.block for i in idx)

    def tile_of(self, start):
        idx = tuple(a // self.block for a in start)
        return int(np.ravel_multi_index(idx, self.tiles_per_axis))

    def probabilities(self, frame_indices):
        sub = self.table[list(frame_indices)]
        visited = ~np.isnan(sub)
        fill = sub[visited].max() if visited.any() else 1.0
        err = np.where(visited, sub, fill)
        w = np.power(np.maximum(err, 0.0), self.alpha)
        total = w.sum()
        n = w.size
        prop = w / total if total > 0 else np.full(w.shape, 1.0 / n)
        return ((1 - self.uniform) * prop + self.uniform / n).reshape(-1)

    def sample(self, frame_indices, rng):
        """rng: an object with .choices (the `random` module or random.Random)."""
        p = self.probabilities(frame_indices)
        k = rng.choices(range(len(p)), weights=p)[0]
        f, tile = divmod(k, self.n_tiles)
        return frame_indices[f], self.tile_start(tile)

    def update(self, frame, start, squared_error):
        self.table[frame, self.tile_of(start)] = float(squared_error)

    def state(self):
        return self.table.tolist()
