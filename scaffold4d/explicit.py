"""Explicit 4D ESVR primitives: every child carries its own parameters (no
anchor feature, no shared MLP), with the same time model as ScaffoldESVR:

    x(t) = home_xyz + flow * (home_t - t)
    I(t) = sigmoid(logit) * exp(-|(home_t - t) * sigma|^beta)
    scale = max(exp(log_scale), floor)   (straight-through floor)

Built from a trained ScaffoldESVR (distillation) to separate the cost of the
shared MLP decoding from the cost of sharing primitives across time. The
`at()` output is duck-type compatible with gaussian_renderer.query and
scaffold4d.model.query_block.
"""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplicitESVR4D(nn.Module):
    param_names = ("home_xyz", "home_t", "log_scale", "rotation", "logit", "log_sigma", "flow")

    def __init__(self, n, shape, temporal_beta=4.0, temporal_cutoff=1e-3, scale_floor=1e-4, sigma_max=1e2):
        super().__init__()
        self.config = dict(n=n, shape=list(shape), temporal_beta=temporal_beta, temporal_cutoff=temporal_cutoff,
                           scale_floor=scale_floor, sigma_max=sigma_max)
        self.home_xyz = nn.Parameter(torch.zeros(n, 3))
        self.home_t = nn.Parameter(torch.zeros(n, 1))
        self.log_scale = nn.Parameter(torch.zeros(n, 3))
        self.rotation = nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1))
        self.logit = nn.Parameter(torch.zeros(n, 1))
        self.log_sigma = nn.Parameter(torch.zeros(n, 1))
        self.flow = nn.Parameter(torch.zeros(n, 3))
        # query_block builds a zero volume from model.anchor when nothing is selected
        self.register_buffer("anchor", torch.zeros(1, 4), persistent=False)

    @classmethod
    @torch.no_grad()
    def from_scaffold(cls, model, keep=None):
        g = model.generate()
        c = model.config
        k = torch.ones(g.intensity.shape[:2], dtype=torch.bool, device=g.intensity.device) if keep is None else keep
        flat = lambda t: t[k]
        out = cls(int(k.sum()), c["shape"], c["temporal_beta"], c["temporal_cutoff"], c["scale_floor"], c["sigma_max"]).to(model.anchor.device)
        intensity = flat(g.intensity).clamp(1e-6, 1 - 1e-6)
        out.home_xyz.copy_(flat(g.home_xyz))
        out.home_t.copy_(flat(g.home_t))
        out.log_scale.copy_(flat(g.scale).clamp_min(1e-8).log())
        out.rotation.copy_(flat(g.rotation))
        out.logit.copy_(torch.log(intensity / (1 - intensity)))
        out.log_sigma.copy_(flat(g.sigma).clamp_min(1e-4).log())
        out.flow.copy_(flat(g.flow))
        return out

    def at(self, timestamp, generated=None):
        t = torch.as_tensor(timestamp, device=self.home_xyz.device, dtype=self.home_xyz.dtype)
        sigma = self.log_sigma.exp().clamp(1e-2, self.config["sigma_max"])
        dt = self.home_t - t
        gate = torch.exp(-(dt * sigma).abs().pow(self.config["temporal_beta"]))
        opacity = self.logit.sigmoid() * gate
        mask = opacity[:, 0] > self.config["temporal_cutoff"]
        pre = self.log_scale.exp()
        scale = pre + (pre.clamp_min(self.config["scale_floor"]) - pre).detach()
        xyz = (self.home_xyz + self.flow * dt)[mask]
        if xyz.requires_grad:
            xyz.retain_grad()
        return SimpleNamespace(get_xyz=xyz, get_scaling=scale[mask], get_rotation=F.normalize(self.rotation[mask], dim=-1),
                               get_opacity=opacity[mask], child_index=mask.nonzero().flatten(), opacity_t=gate[mask], sigma=sigma[mask])

    def stabilize(self):
        with torch.no_grad():
            self.log_scale.clamp_(math.log(1e-4), math.log(1.0))
            self.log_sigma.clamp_(math.log(1e-2), math.log(self.config["sigma_max"]))

    def stored_bytes(self, fp16=False):
        return sum(p.numel() for p in self.parameters()) * (2 if fp16 else 4)
