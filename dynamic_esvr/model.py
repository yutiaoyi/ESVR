"""Canonical ellipsoids + position/time MLP, inspired by Deformable-3D-Gaussians.

Network architecture adapted from Ziyi Yang (2023), MIT license;
see third_party/Deformable-3D-Gaussians.LICENSE.
See docs/DEFORMABLE_ESVR.md for the pinned upstream and deliberate differences.
No donor rasterizer or RGB/SH representation is used.
"""
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class Encoding(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(bands))

    def forward(self, x):
        parts = [x]
        for frequency in self.frequencies:
            parts.extend([(x * frequency).sin(), (x * frequency).cos()])
        return torch.cat(parts, -1)


class Deformation(nn.Module):
    def __init__(self, depth=8, width=256, xyz_bands=10, time_bands=10,
                 scalar=False):
        super().__init__()
        if depth < 2 or width < 1 or min(xyz_bands, time_bands) < 0:
            raise ValueError("Invalid MLP dimensions")
        self.xyz_encoding, self.time_encoding = Encoding(xyz_bands), Encoding(time_bands)
        inputs = 3 * (1 + 2 * xyz_bands) + 1 + 2 * time_bands
        self.skip = depth // 2
        self.layers = nn.ModuleList([
            nn.Linear(inputs if i == 0 else width + (inputs if i == self.skip + 1 else 0), width)
            for i in range(depth)
        ])
        self.heads = nn.ModuleDict({name: nn.Linear(width, size) for name, size in
                                   [("xyz", 3), ("rotation", 4), ("scale", 3)] +
                                   ([("scalar", 1)] if scalar else [])})
        # Identity initialization makes a static PLY a safe starting point.
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, xyz, time):
        time = torch.as_tensor(time, device=xyz.device, dtype=xyz.dtype).reshape(1, 1)
        encoded = torch.cat([self.xyz_encoding(xyz), self.time_encoding(time).expand(len(xyz), -1)], -1)
        h = encoded
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(h))
            if i == self.skip and i + 1 < len(self.layers):
                h = torch.cat([encoded, h], -1)
        return {name: head(h) for name, head in self.heads.items()}


def normalize_quaternion(q):
    norm = q.norm(dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[:, 0] = 1
    return torch.where(norm > 1e-8, q / norm.clamp_min(1e-8), identity)


class DynamicESVR(nn.Module):
    def __init__(self, points, shape, depth=8, width=256, xyz_bands=10,
                 time_bands=10, scalar=False):
        super().__init__()
        if points < 1 or len(shape) != 3 or min(shape) < 1:
            raise ValueError("Invalid point count or XYZ shape")
        self.config = dict(points=points, shape=list(shape), depth=depth, width=width,
                           xyz_bands=xyz_bands, time_bands=time_bands, scalar=scalar)
        extent = torch.tensor(shape) / max(shape)
        self.xyz = nn.Parameter((torch.rand(points, 3) * 2 - 1) * extent)
        scale = (float((2 * extent).prod()) / points) ** (1 / 3) * 0.8
        self.log_scale = nn.Parameter(torch.full((points, 3), math.log(scale)))
        rotation = torch.zeros(points, 4)
        rotation[:, 0] = 1
        self.rotation = nn.Parameter(rotation)
        self.logit_scalar = nn.Parameter(torch.full((points, 1), math.log(0.1 / 0.9)))
        self.deform = Deformation(depth, width, xyz_bands, time_bands, scalar)

    def at(self, time, deform=True):
        delta = self.deform(self.xyz.detach(), time) if deform else {}
        return SimpleNamespace(
            get_xyz=self.xyz + delta.get("xyz", 0),
            get_scaling=(self.log_scale + delta.get("scale", 0)).clamp(-12, 2).exp(),
            get_rotation=normalize_quaternion(normalize_quaternion(self.rotation) + delta.get("rotation", 0)),
            get_opacity=(self.logit_scalar + delta.get("scalar", 0)).sigmoid())

    def initialize_ply(self, path):
        from scene.gaussian_model import GaussianModel
        source = GaussianModel(*self.config["shape"])
        source.load_ply(str(path))
        self.config["points"] = len(source.get_xyz)
        for name, value in [("xyz", source._xyz), ("log_scale", source._scaling),
                            ("rotation", source._rotation), ("logit_scalar", source._opacity)]:
            setattr(self, name, nn.Parameter(value.detach().clone().to(self.xyz.device)))

    def export_ply(self, path, time=0, deform=True):
        from scene.gaussian_model import GaussianModel
        source = GaussianModel(*self.config["shape"])
        frame = self.at(time, deform)
        source._xyz = frame.get_xyz
        source._scaling = frame.get_scaling.log()
        source._rotation = frame.get_rotation
        source._opacity = torch.logit(frame.get_opacity.clamp(1e-6, 1 - 1e-6))
        source.save_ply(str(path))


def query_block(model, time, shape, start=None, size=None, deform=True):
    from gaussian_renderer import query
    start = (0, 0, 0) if start is None else start
    size = shape if size is None else size
    step = 2.0 / max(shape)
    center = [(start[i] + size[i] / 2 - shape[i] / 2) * step for i in range(3)]
    return query(model.at(time, deform), center, size, [s * step for s in size],
                 SimpleNamespace(debug=False, compute_cov3D_python=False))["vol"]
