"""Anchor/offset scaffold + shared MLP decoders, ESVR-scalar adaptation of
4D-Scaffold-GS (Cho et al., arXiv:2411.17044).

No RGB, view-dependent color, feature bank, or camera supervision. Motion is
the paper's own design: constant per-child velocity around a per-child home
time, gated by a temporal opacity window. See docs/ESVR_SCAFFOLD4D.md for the
source-reading map and deliberate differences from upstream.
"""
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


def normalize_quaternion(q):
    norm = q.norm(dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    return torch.where(norm > 1e-8, q / norm.clamp_min(1e-8), identity)


class ScaffoldESVR(nn.Module):
    def __init__(self, anchors=5000, children=5, shape=(128, 128, 128), feat_dim=32,
                 hidden=32, temporal_beta=4.0, temporal_cutoff=1e-3):
        super().__init__()
        if anchors < 1 or children < 1 or len(shape) != 3 or min(shape) < 1:
            raise ValueError("Invalid anchor/children count or XYZ shape")
        if temporal_beta <= 0 or not 0 <= temporal_cutoff < 1:
            raise ValueError("Invalid temporal_beta or temporal_cutoff")
        self.config = dict(anchors=anchors, children=children, shape=list(shape),
                           feat_dim=feat_dim, hidden=hidden, temporal_beta=temporal_beta,
                           temporal_cutoff=temporal_cutoff)
        extent = torch.tensor(shape) / max(shape)
        self.anchor = nn.Parameter(torch.cat([
            (torch.rand(anchors, 3) * 2 - 1) * extent,
            torch.rand(anchors, 1),
        ], -1))
        self.anchor_feat = nn.Parameter(torch.zeros(anchors, feat_dim))
        spatial_offset_scale = (float((2 * extent).prod()) / anchors) ** (1 / 3) * 0.5
        self.log_offset_scale = nn.Parameter(torch.tensor(
            [math.log(spatial_offset_scale)] * 3 + [math.log(0.1)] + [math.log(spatial_offset_scale)] * 3
        ).expand(anchors, 7).contiguous())
        self.offset = nn.Parameter(torch.zeros(anchors, children, 4))
        self.heads = nn.ModuleDict({
            # Sigmoid, not upstream's Tanh-then-hard-mask-at-zero: with every
            # anchor starting from an identical zero feature, a Tanh-masked
            # opacity has EXACTLY zero gradient once a child's raw output
            # goes negative, and empirically every child in a freshly
            # initialized model crosses zero together within ~50 steps and
            # never recovers (verified: all 1024 children active at step 1,
            # 0 active from step 50 onward, loss frozen for the rest of
            # training). Sigmoid has no such dead zone, matching the
            # opacity/scalar activation already used by dynamic_esvr and
            # esvr4d in this repository.
            "opacity": nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(True),
                                     nn.Linear(hidden, children)),
            "shape": nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(True),
                                   nn.Linear(hidden, children * 8)),
            "flow": nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(True),
                                  nn.Linear(hidden, children * 3), nn.Tanh()),
        })
        self.register_buffer("grad_accum", torch.zeros(anchors))
        self.register_buffer("weight_accum", torch.zeros(anchors))
        self.register_buffer("observations", torch.zeros(anchors))

    def generate(self):
        """Per-anchor MLP decode; independent of query time. Shapes: (N, K, ...)."""
        n, k = len(self.anchor), self.config["children"]
        opacity_raw = self.heads["opacity"](self.anchor_feat).view(n, k, 1)
        shape_out = self.heads["shape"](self.anchor_feat).view(n, k, 8)
        flow = self.heads["flow"](self.anchor_feat).view(n, k, 3)
        offset_scale = self.log_offset_scale.exp()
        home_xyz = self.anchor[:, None, :3] + self.offset[..., :3] * offset_scale[:, None, :3]
        home_t = self.anchor[:, None, 3:4] + self.offset[..., 3:4] * offset_scale[:, None, 3:4]
        scale = offset_scale[:, None, 4:7] * shape_out[..., :3].sigmoid()
        rotation = normalize_quaternion(shape_out[..., 4:8])
        sigma = F.elu(shape_out[..., 3:4]) + 1
        return SimpleNamespace(opacity_raw=opacity_raw, home_xyz=home_xyz, home_t=home_t,
                               scale=scale, rotation=rotation, sigma=sigma, flow=flow)

    def at(self, timestamp, generated=None):
        g = generated if generated is not None else self.generate()
        timestamp = torch.as_tensor(timestamp, device=self.anchor.device, dtype=self.anchor.dtype)
        dt = g.home_t - timestamp
        opacity_t = torch.exp(-(dt * g.sigma).abs().pow(self.config["temporal_beta"]))
        opacity = g.opacity_raw.sigmoid() * opacity_t
        xyz = g.home_xyz + g.flow * dt
        # Inclusion in this particular CUDA query is an efficiency cutoff,
        # not a permanent gate: sigmoid(opacity_raw) always has a nonzero
        # gradient, so a child below cutoff now can still recover later
        # (through its anchor's shared feature, updated by its siblings).
        mask = opacity[..., 0] > self.config["temporal_cutoff"]
        flat = lambda t: t.reshape(-1, t.shape[-1])[mask.reshape(-1)]
        return SimpleNamespace(get_xyz=flat(xyz), get_scaling=flat(g.scale),
            get_rotation=flat(g.rotation), get_opacity=flat(opacity), mask=mask,
            opacity_t=flat(opacity_t),
            anchor_index=torch.arange(len(self.anchor), device=self.anchor.device)
                .unsqueeze(1).expand(-1, self.config["children"]).reshape(-1)[mask.reshape(-1)])

    @torch.no_grad()
    def stabilize(self):
        self.log_offset_scale.clamp_(math.log(1e-4), math.log(10))

    @torch.no_grad()
    def record_gradients(self, sliced, visibility):
        if sliced.get_xyz.grad is None:
            return
        grad = sliced.get_xyz.grad[visibility].norm(dim=-1)
        weight = sliced.opacity_t[visibility].reshape(-1).clamp_min(0)
        indices = sliced.anchor_index[visibility]
        self.grad_accum.scatter_add_(0, indices, grad * weight)
        self.weight_accum.scatter_add_(0, indices, weight)
        touched = torch.unique(indices)
        self.observations.scatter_add_(0, touched, torch.ones_like(touched, dtype=torch.float))

    population_params = ("anchor", "anchor_feat", "offset", "log_offset_scale")

    @torch.no_grad()
    def replace_population(self, optimizer, keep, children_dict):
        # Only anchor/child tensors are indexed by population size; the
        # shared MLP heads are fixed-size networks, not per-anchor data.
        for name in self.population_params:
            old = getattr(self, name)
            extra = children_dict[name]
            new = nn.Parameter(torch.cat((old[keep], extra), 0))
            for group in optimizer.param_groups:
                for i, value in enumerate(group["params"]):
                    if value is old:
                        group["params"][i] = new
            state = optimizer.state.pop(old, {})
            for key, value in state.items():
                if torch.is_tensor(value) and value.shape == old.shape:
                    state[key] = torch.cat((value[keep], torch.zeros_like(extra)), 0)
            if state:
                optimizer.state[new] = state
            setattr(self, name, new)
        n = len(self.anchor)
        self.config["anchors"] = n
        for name in ("grad_accum", "weight_accum", "observations"):
            setattr(self, name, self.anchor.new_zeros(n))

    @torch.no_grad()
    def densify(self, optimizer, grad_threshold=2e-4, max_anchors=50000, spatial_size=0.02,
                temporal_size=0.2, min_children_active=1):
        denom = self.weight_accum.clamp_min(1e-8)
        grads = (self.grad_accum / denom).nan_to_num(0.0)
        active = self.observations > 0
        scales = self.log_offset_scale.exp()
        large = (scales[:, 4:7].amax(-1) > spatial_size) | (scales[:, 3] > temporal_size)
        candidates = ((grads >= grad_threshold) & active).nonzero().flatten()
        budget = max(0, max_anchors - len(self.anchor))
        candidates = candidates[grads[candidates].argsort(descending=True)[:budget]]
        splits = candidates[large[candidates]]
        clones = candidates[~large[candidates]]
        parents = torch.cat((splits.repeat_interleave(2), clones))
        keep = torch.ones(len(self.anchor), dtype=torch.bool, device=self.anchor.device)
        children_dict = {name: getattr(self, name)[parents].clone() for name in self.population_params}
        count = len(splits) * 2
        if count:
            sample = torch.randn_like(children_dict["anchor"][:count]) * scales[parents[:count], 4:7].mean(-1, keepdim=True).expand(-1, 4)
            children_dict["anchor"][:count] += sample
            children_dict["log_offset_scale"][:count] -= math.log(1.6)
            keep[splits] = False
        if len(parents):
            # ESVR's voxelizer sums scalar contributions rather than alpha-
            # compositing (see docs/ESVR4D.md); halve each new anchor's
            # opacity head bias contribution indirectly is not possible
            # (opacity comes from a shared MLP, not a per-anchor scalar), so
            # instead halve the *feature* magnitude feeding the opacity head
            # as an approximate compensation, consistent in spirit with
            # ESVR4D's split/clone opacity halving.
            children_dict["anchor_feat"] *= 0.5
        before = len(self.anchor)
        self.replace_population(optimizer, keep, children_dict)
        return dict(before=before, after=len(self.anchor), split=len(splits), cloned=len(clones),
                    pruned=before - int(keep.sum()) - len(splits))


def query_block(model, timestamp, shape, start, size, return_package=False):
    from gaussian_renderer import query
    step = 2 / max(shape)
    center = [(a + n / 2 - s / 2) * step for a, n, s in zip(start, size, shape)]
    sliced = model.at(timestamp)
    if not len(sliced.get_xyz):
        vol = model.anchor.sum() * 0 + model.anchor.new_zeros(size)
        package = dict(vol=vol, visibility_filter=torch.zeros(0, dtype=torch.bool, device=vol.device))
    else:
        package = query(sliced, center, size, [s * step for s in size],
                        SimpleNamespace(compute_cov3D_python=False, debug=False))
    return (package, sliced) if return_package else package["vol"]
