"""Anchor/offset scaffold + shared MLP decoders, ESVR-scalar adaptation of
4D-Scaffold-GS (Cho et al., arXiv:2411.17044).

No RGB, view-dependent color, feature bank, or camera supervision. Motion is
the paper's own design: constant per-child velocity around a per-child home
time, gated by a temporal opacity window. Anchor growing/pruning follows
upstream's `training_statis`/`anchor_growing`/`adjust_anchor` with the
view-space 2D gradient replaced by the 3D gradient of each child's centre
(ESVR has no camera). See docs/ESVR_SCAFFOLD4D.md.
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


def inverse_sigmoid(x):
    return math.log(x / (1 - x))


def voxel_centers(shape, index):
    """ESVR coordinate convention (gaussian_renderer.query / query_block):
    voxel i of an axis of length s sits at (i + 0.5 - s/2) * 2/max(shape)."""
    step = 2 / max(shape)
    return (index.double() + 0.5 - torch.tensor(shape, dtype=torch.double, device=index.device) / 2) * step


@torch.no_grad()
def anchors_from_sequence(sequence, count, voxel_size, t_size, generator=None):
    """Upstream initialises anchors from an SfM point cloud voxelised on a 4D
    grid (create_from_pcd/voxelize_sample). Volumes have no point cloud, but
    the ground truth itself says where structure is: sample voxel centres with
    probability proportional to intensity, spread over all training frames
    (each sample carries its frame's time), then dedupe on the same 4D grid."""
    frames = sequence.train_indices
    per_frame = [count // len(frames) + (1 if i < count % len(frames) else 0) for i in range(len(frames))]
    points = []
    for index, n in zip(frames, per_frame):
        if n == 0:
            continue
        volume = sequence.volumes[index]
        weights = volume.reshape(-1).double().clamp_min(0)
        if weights.sum() <= 0:
            weights = torch.ones_like(weights)
        flat = torch.multinomial(weights, n, replacement=False, generator=generator)
        index3 = torch.stack(torch.unravel_index(flat, sequence.shape), -1)
        xyz = voxel_centers(sequence.shape, index3)
        t = torch.full((n, 1), float(sequence.times[index]), dtype=torch.double)
        points.append(torch.cat([xyz, t], -1))
    points = torch.cat(points, 0)
    grid = torch.cat([torch.round(points[:, :3] / voxel_size), torch.round(points[:, 3:] / t_size)], -1)
    grid = torch.unique(grid, dim=0)
    return torch.cat([grid[:, :3] * voxel_size, grid[:, 3:] * t_size], -1).float()


class ScaffoldESVR(nn.Module):
    def __init__(self, anchors=5000, children=5, shape=(128, 128, 128), feat_dim=32,
                 hidden=32, temporal_beta=4.0, temporal_cutoff=1e-3, flow_scale=1.0,
                 scale_floor=1e-4, opacity_init=0.5, anchor_init=None, sigma_max=1e2,
                 offset_spread=0.0, child_scale_init=None):
        super().__init__()
        if anchors < 1 or children < 1 or len(shape) != 3 or min(shape) < 1:
            raise ValueError("Invalid anchor/children count or XYZ shape")
        if temporal_beta <= 0 or not 0 <= temporal_cutoff < 1:
            raise ValueError("Invalid temporal_beta or temporal_cutoff")
        if flow_scale <= 0 or scale_floor <= 0 or not 0 < opacity_init < 1:
            raise ValueError("Invalid flow_scale, scale_floor or opacity_init")
        if sigma_max <= 1e-2 or offset_spread < 0 or (child_scale_init is not None and child_scale_init <= 0):
            raise ValueError("Invalid sigma_max, offset_spread or child_scale_init")
        if anchor_init is not None:
            anchors = len(anchor_init)
        self.config = dict(anchors=anchors, children=children, shape=list(shape),
                           feat_dim=feat_dim, hidden=hidden, temporal_beta=temporal_beta,
                           temporal_cutoff=temporal_cutoff, flow_scale=flow_scale,
                           scale_floor=scale_floor, opacity_init=opacity_init, sigma_max=sigma_max,
                           offset_spread=offset_spread, child_scale_init=child_scale_init)
        extent = torch.tensor(shape) / max(shape)
        if anchor_init is None:
            anchor_init = torch.cat([(torch.rand(anchors, 3) * 2 - 1) * extent, torch.rand(anchors, 1)], -1)
            spatial = torch.full((anchors,), (float((2 * extent).prod()) / anchors) ** (1 / 3) * 0.5)
        else:
            # Upstream: offset/final scale start at each anchor's nearest-neighbour distance.
            d = torch.cdist(anchor_init[:, :3], anchor_init[:, :3])
            d.fill_diagonal_(float("inf"))
            spatial = d.min(1).values.clamp(1e-3, 1.0)
        self.anchor = nn.Parameter(anchor_init.clone().float())
        self.anchor_feat = nn.Parameter(torch.zeros(anchors, feat_dim))
        log_spatial = spatial.log()[:, None]
        # child_scale_init: ESVR initialises every primitive at its
        # nearest-neighbour distance and barely changes it (scale lr 5e-5);
        # upstream instead starts children at the anchor's own offset scale.
        log_child = log_spatial if child_scale_init is None else torch.full((anchors, 1), math.log(child_scale_init))
        self.log_offset_scale = nn.Parameter(torch.cat([
            log_spatial.expand(-1, 3), torch.full((anchors, 1), math.log(0.1)), log_child.expand(-1, 3)], 1).contiguous())
        self.offset = nn.Parameter(self._initial_offsets(anchors))
        self.heads = nn.ModuleDict({
            # Sigmoid, not upstream's Tanh-then-hard-mask-at-zero: with every
            # anchor starting from an identical zero feature, a Tanh-masked
            # opacity has EXACTLY zero gradient once a child's raw output
            # goes negative, and empirically every child in a freshly
            # initialized model crosses zero together within ~50 steps and
            # never recovers. Sigmoid has no such dead zone.
            "opacity": nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(True),
                                     nn.Linear(hidden, children)),
            "shape": nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(True),
                                   nn.Linear(hidden, children * 8)),
            "flow": nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(True),
                                  nn.Linear(hidden, children * 3), nn.Tanh()),
        })
        with torch.no_grad():
            # ESVR sums contributions instead of alpha-compositing, so the
            # initial per-child amplitude sets the initial over-prediction
            # directly; 0.1 is both ESVR's and upstream's own opacity init.
            self.heads["opacity"][-1].bias.fill_(inverse_sigmoid(opacity_init))
        n = anchors * children
        self.register_buffer("child_grad_accum", torch.zeros(n), persistent=False)
        self.register_buffer("child_denom", torch.zeros(n), persistent=False)
        self.register_buffer("anchor_opacity_accum", torch.zeros(anchors), persistent=False)
        self.register_buffer("anchor_denom", torch.zeros(anchors), persistent=False)

    def _initial_offsets(self, n, device=None):
        # Upstream starts all K children of an anchor at zero offset, i.e.
        # stacked on one point. Under alpha compositing K stacked children
        # saturate; under ESVR's additive accumulation they sum to K times
        # the intensity, so an optional uniform spread over the cell exists.
        offsets = torch.zeros(n, self.config["children"], 4, device=device)
        if self.config["offset_spread"]:
            offsets[..., :3].uniform_(-self.config["offset_spread"], self.config["offset_spread"])
        return offsets

    def generate(self):
        """Per-anchor MLP decode; independent of query time. Shapes: (N, K, ...)."""
        n, k = len(self.anchor), self.config["children"]
        opacity_raw = self.heads["opacity"](self.anchor_feat).view(n, k, 1)
        shape_out = self.heads["shape"](self.anchor_feat).view(n, k, 8)
        # tanh bounds the velocity to 1 normalised unit per unit time; with
        # time normalised to [0,1] over the whole sequence that is far below
        # the measured motion of real data, so the bound is rescaled.
        flow = self.heads["flow"](self.anchor_feat).view(n, k, 3) * self.config["flow_scale"]
        offset_scale = self.log_offset_scale.exp()
        home_xyz = self.anchor[:, None, :3] + self.offset[..., :3] * offset_scale[:, None, :3]
        home_t = self.anchor[:, None, 3:4] + self.offset[..., 3:4] * offset_scale[:, None, 3:4]
        # Floor on the final product (offset_scale * sigmoid gate): a scale
        # small enough to touch no voxel centre is skipped by the voxelizer,
        # gets no gradient, and can never recover on its own; a near-singular
        # covariance also produced NaN in the CUDA kernel.
        scale = (offset_scale[:, None, 4:7] * shape_out[..., :3].sigmoid()).clamp_min(self.config["scale_floor"])
        rotation = normalize_quaternion(shape_out[..., 4:8])
        # Numerical guardrail: sigma is raised to the 4th power downstream.
        sigma = (F.elu(shape_out[..., 3:4]) + 1).clamp(1e-2, self.config["sigma_max"])
        return SimpleNamespace(opacity_raw=opacity_raw, home_xyz=home_xyz, home_t=home_t,
                               scale=scale, rotation=rotation, sigma=sigma, flow=flow)

    def at(self, timestamp, generated=None):
        g = generated if generated is not None else self.generate()
        timestamp = torch.as_tensor(timestamp, device=self.anchor.device, dtype=self.anchor.dtype)
        dt = g.home_t - timestamp
        opacity_t = torch.exp(-(dt * g.sigma).abs().pow(self.config["temporal_beta"]))
        opacity_base = g.opacity_raw.sigmoid()
        opacity = opacity_base * opacity_t
        xyz = g.home_xyz + g.flow * dt
        # Inclusion in this particular CUDA query is an efficiency cutoff,
        # not a permanent gate: sigmoid(opacity_raw) always has a nonzero
        # gradient, so a child below cutoff now can still recover later.
        mask = opacity[..., 0] > self.config["temporal_cutoff"]
        flat_mask = mask.reshape(-1)
        flat = lambda t: t.reshape(-1, t.shape[-1])[flat_mask]
        selected_xyz = flat(xyz)
        if selected_xyz.requires_grad:
            selected_xyz.retain_grad()
        return SimpleNamespace(
            get_xyz=selected_xyz, get_scaling=flat(g.scale), get_rotation=flat(g.rotation),
            get_opacity=flat(opacity), mask=mask, opacity_t=flat(opacity_t), sigma=flat(g.sigma),
            child_index=flat_mask.nonzero().flatten(),
            anchor_opacity_sum=opacity_base.detach().sum((1, 2)))

    @torch.no_grad()
    def stabilize(self):
        self.log_offset_scale.clamp_(math.log(1e-4), math.log(10))
        # dt has no natural bound: an unbounded offset let a child's home
        # time drift arbitrarily far and feed the nonfinite-loss cascade.
        self.offset.clamp_(-10, 10)

    @torch.no_grad()
    def record_statistics(self, sliced, visibility, lambda_temporal_sigma=1.0):
        """Upstream training_statis: per-child gradient weighted by
        opacity_t * sigma^gamma (dynamic-aware, sigma_denom_weight on), and
        per-anchor base-opacity sums for pruning."""
        self.anchor_opacity_accum += sliced.anchor_opacity_sum
        self.anchor_denom += 1
        if sliced.get_xyz.grad is None:
            return
        grad = sliced.get_xyz.grad[visibility].norm(dim=-1)
        weight = sliced.opacity_t[visibility].reshape(-1) * sliced.sigma[visibility].reshape(-1).pow(lambda_temporal_sigma)
        index = sliced.child_index[visibility]
        self.child_grad_accum.index_add_(0, index, grad * weight)
        self.child_denom.index_add_(0, index, weight)

    population_params = ("anchor", "anchor_feat", "offset", "log_offset_scale")

    @torch.no_grad()
    def replace_population(self, optimizer, keep, extra):
        # Only anchor/child tensors are indexed by population size; the
        # shared MLP heads are fixed-size networks, not per-anchor data.
        for name in self.population_params:
            old = getattr(self, name)
            new = nn.Parameter(torch.cat((old[keep], extra[name].to(old)), 0))
            for group in optimizer.param_groups:
                for i, value in enumerate(group["params"]):
                    if value is old:
                        group["params"][i] = new
            state = optimizer.state.pop(old, {})
            for key, value in state.items():
                if torch.is_tensor(value) and value.shape == old.shape:
                    state[key] = torch.cat((value[keep], torch.zeros_like(extra[name], device=value.device)), 0)
            if state:
                optimizer.state[new] = state
            setattr(self, name, new)
        self.config["anchors"] = len(self.anchor)

    @staticmethod
    def _grid(points4d, size, t_size):
        return torch.cat([torch.round(points4d[:, :3] / size), torch.round(points4d[:, 3:4] / t_size)], 1).long()

    @torch.no_grad()
    def adjust_anchors(self, optimizer, check_interval=100, success_threshold=0.8, grad_threshold=2e-4,
                       min_opacity=0.005, voxel_size=2 / 128, t_size=1 / 89, init_factor=16,
                       hierarchy_factor=4, depth=3, max_anchors=40000, partial=True):
        """Upstream adjust_anchor: multi-level voxel-grid anchor growing from
        sufficiently observed, high-gradient children, then pruning of anchors
        whose children stayed near-transparent."""
        n, k = len(self.anchor), self.config["children"]
        grads = (self.child_grad_accum / self.child_denom).nan_to_num(0.0)
        observed = self.child_denom > check_interval * success_threshold * 0.5
        checked = self.anchor_denom > check_interval * success_threshold
        prune = (self.anchor_opacity_accum < min_opacity * self.anchor_denom) & checked
        g = self.generate()
        child4d = torch.cat([g.home_xyz, g.home_t], -1).reshape(-1, 4)
        child_feat = self.anchor_feat.repeat_interleave(k, 0)
        new = dict(anchor=[], anchor_feat=[], offset=[], log_offset_scale=[])
        added = 0
        budget = max(0, max_anchors - (n - int(prune.sum())))
        for level in range(depth):
            threshold = grad_threshold * (hierarchy_factor // 2) ** level
            candidate = (grads >= threshold) & observed
            if partial:
                candidate &= torch.rand_like(grads) > 0.5 ** (level + 1)
            if budget - added <= 0 or not candidate.any():
                continue
            size = voxel_size * (init_factor // hierarchy_factor ** level)
            existing = torch.cat([self.anchor] + new["anchor"]) if new["anchor"] else self.anchor
            existing_cells = self._grid(existing, size, t_size)
            selected = candidate.nonzero().flatten()
            selected_cells = self._grid(child4d[selected], size, t_size)
            cells, inverse = torch.unique(torch.cat([existing_cells, selected_cells]), dim=0, return_inverse=True)
            occupied = torch.zeros(len(cells), dtype=torch.bool, device=cells.device)
            occupied[inverse[:len(existing_cells)]] = True
            cell_of_selected = inverse[len(existing_cells):]
            fresh = ~occupied
            if not fresh.any():
                continue
            # Feature of a new anchor: max-pool over the children that voted
            # for its cell (upstream scatter_max); rank cells by their
            # strongest vote so a finite anchor budget goes to the worst areas.
            feat = torch.zeros(len(cells), child_feat.shape[1], device=cells.device).scatter_reduce_(
                0, cell_of_selected[:, None].expand(-1, child_feat.shape[1]), child_feat[selected], "amax", include_self=False)
            vote = torch.zeros(len(cells), device=cells.device).scatter_reduce_(
                0, cell_of_selected, grads[selected], "amax", include_self=False)
            order = fresh.nonzero().flatten()
            order = order[vote[order].argsort(descending=True)[:budget - added]]
            cells = cells[order].float()
            m = len(order)
            new["anchor"].append(torch.cat([cells[:, :3] * size, cells[:, 3:4] * t_size], 1))
            new["anchor_feat"].append(feat[order])
            new["offset"].append(self._initial_offsets(m, cells.device))
            child = size if self.config["child_scale_init"] is None else self.config["child_scale_init"]
            new["log_offset_scale"].append(torch.tensor(
                [math.log(size)] * 3 + [math.log(t_size)] + [math.log(child)] * 3, device=cells.device).expand(m, 7).contiguous())
            added += m
        extra = {name: (torch.cat(v) if v else getattr(self, name)[:0]) for name, v in new.items()}
        keep = ~prune
        # Upstream resets the statistics only of children/anchors that were
        # actually checked this round; the rest keep accumulating.
        self.child_grad_accum[observed] = 0
        self.child_denom[observed] = 0
        self.anchor_opacity_accum[checked] = 0
        self.anchor_denom[checked] = 0
        child_keep = keep.repeat_interleave(k)
        self.child_grad_accum = torch.cat([self.child_grad_accum[child_keep], self.child_grad_accum.new_zeros(added * k)])
        self.child_denom = torch.cat([self.child_denom[child_keep], self.child_denom.new_zeros(added * k)])
        self.anchor_opacity_accum = torch.cat([self.anchor_opacity_accum[keep], self.anchor_opacity_accum.new_zeros(added)])
        self.anchor_denom = torch.cat([self.anchor_denom[keep], self.anchor_denom.new_zeros(added)])
        self.replace_population(optimizer, keep, extra)
        return dict(before=n, after=len(self.anchor), grown=added, pruned=int(prune.sum()),
                    observed_children=int(observed.sum()), grad_p50=float(grads[observed].median()) if observed.any() else 0.0,
                    grad_p90=float(grads[observed].quantile(0.9)) if observed.any() else 0.0)


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
