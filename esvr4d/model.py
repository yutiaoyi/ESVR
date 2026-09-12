"""Fudan 4DGS covariance/temporal marginal, with ESVR's spatial kernel.

No RGB, view-dependent SH, canonical deformation MLP, or camera supervision.
See docs/ESVR4D.md for equations, upstream conventions and deviations.
"""
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


def rotation4d(left, right):
    """SO(4), matching Fudan's conjugated-right / reversed-axis convention."""
    a, b, c, d = F.normalize(left, dim=-1).unbind(-1)
    p, q, r, s = F.normalize(right, dim=-1).unbind(-1)
    lmat = torch.stack((a,-b,-c,-d, b,a,-d,c, c,d,a,-b, d,-c,b,a), -1).reshape(-1,4,4)
    rmat = torch.stack((p,q,r,s, -q,p,-s,r, -r,s,p,-q, -s,-r,q,p), -1).reshape(-1,4,4)
    return (lmat @ rmat).flip((-2,-1))


def pack_covariance(cov):
    return cov[:, (0,0,0,1,1,2), (0,1,2,1,2,2)].contiguous()


class ESVR4D(nn.Module):
    def __init__(self, points=20000, shape=(128,128,128), temporal_scale=0.5,
                 temporal_cutoff=0.05):
        super().__init__()
        if points < 1 or len(shape) != 3 or min(shape) < 1 or temporal_scale <= 0:
            raise ValueError("Invalid primitive count, shape or temporal scale")
        if not 0 <= temporal_cutoff < 1:
            raise ValueError("temporal_cutoff must be in [0,1)")
        self.config = dict(points=points, shape=list(shape), temporal_scale=temporal_scale,
                           temporal_cutoff=temporal_cutoff)
        extent = torch.tensor(shape) / max(shape)
        self.xyz = nn.Parameter((torch.rand(points,3)*2-1)*extent)
        self.time = nn.Parameter(torch.rand(points,1))
        scales = torch.full((points,4), math.log(2 / points**(1/3)))
        scales[:,3] = math.log(temporal_scale)
        self.log_scale = nn.Parameter(scales)
        identity = torch.zeros(points,4)
        identity[:,0] = 1
        self.left = nn.Parameter(identity.clone())
        self.right = nn.Parameter(identity.clone())
        self.logit_scalar = nn.Parameter(torch.full((points,1), math.log(0.1/0.9)))
        self.register_buffer("grad_space", torch.zeros(points))
        self.register_buffer("grad_time", torch.zeros(points))
        self.register_buffer("observations", torch.zeros(points))

    def covariance4d(self):
        transform = rotation4d(self.left, self.right) * self.log_scale.exp()[:,None,:]
        return transform @ transform.transpose(-1,-2)

    def condition(self, timestamp):
        cov = self.covariance4d()
        variance = cov[:,3:4,3]
        cross = cov[:,:3,3]
        dt = torch.as_tensor(timestamp, dtype=self.xyz.dtype, device=self.xyz.device) - self.time
        mean = self.xyz + cross / variance * dt
        spatial = cov[:,:3,:3] - cross[:,:,None] * cross[:,None,:] / variance[:,:,None]
        marginal = torch.exp(-0.5 * dt.square() / variance)
        return mean, spatial, marginal

    def at(self, timestamp):
        mean, cov, marginal = self.condition(timestamp)
        mask = marginal[:,0] > self.config["temporal_cutoff"]
        selected_mean = mean[mask].contiguous()
        if selected_mean.requires_grad:
            selected_mean.retain_grad()
        return SimpleNamespace(get_xyz=selected_mean,
            get_opacity=(self.logit_scalar.sigmoid()*marginal)[mask].contiguous(),
            get_covariance=lambda modifier=1: pack_covariance(cov[mask]*modifier**2),
            mask=mask, covariance=cov[mask])

    @torch.no_grad()
    def stabilize(self):
        # Numerical guardrails, not part of the original paper's objective.
        self.log_scale.clamp_(math.log(1e-4), math.log(10))
        self.logit_scalar.clamp_(-15,15)
        self.left.copy_(F.normalize(self.left, dim=-1))
        self.right.copy_(F.normalize(self.right, dim=-1))

    @torch.no_grad()
    def record_gradients(self, sliced, visibility, time_gradient, batch_size=1):
        indices = sliced.mask.nonzero().flatten()[visibility]
        if sliced.get_xyz.grad is not None:
            self.grad_space[indices] += sliced.get_xyz.grad[visibility].norm(dim=-1)*batch_size
        self.grad_time[indices] += time_gradient[indices,0].abs()*batch_size
        self.observations[indices] += 1

    @torch.no_grad()
    def replace_population(self, optimizer, keep, children):
        """Preserve Adam moments for survivors; newborns start with zero moments."""
        for name, old in list(self.named_parameters()):
            extra = children[name]
            new = nn.Parameter(torch.cat((old[keep], extra),0))
            for group in optimizer.param_groups:
                for i, value in enumerate(group["params"]):
                    if value is old:
                        group["params"][i] = new
            state = optimizer.state.pop(old, {})
            for key, value in state.items():
                if torch.is_tensor(value) and value.shape == old.shape:
                    state[key] = torch.cat((value[keep], torch.zeros_like(extra)),0)
            if state:
                optimizer.state[new] = state
            setattr(self, name, new)
        n = len(self.xyz)
        self.config["points"] = n
        for name in ("grad_space","grad_time","observations"):
            setattr(self,name,self.xyz.new_zeros(n))

    @torch.no_grad()
    def densify(self, optimizer, space_threshold=1e-5, time_threshold=1e-5,
                max_points=100000, spatial_size=0.02, temporal_size=0.1,
                min_scalar=1/255):
        denominator = self.observations.clamp_min(1)
        score = torch.maximum(self.grad_space/denominator/space_threshold,
                              self.grad_time/denominator/time_threshold)
        # Prune by maximum temporal contribution ANYWHERE in [0,1], not at one frame.
        cov = self.covariance4d()
        dt = self.time - self.time.clamp(0,1)
        peak = self.logit_scalar.sigmoid()[:,0]*torch.exp(-0.5*dt[:,0]**2/cov[:,3,3])
        keep = peak >= min_scalar
        if not keep.any():
            keep[peak.argmax()] = True
        candidates = ((score >= 1) & keep & (self.observations > 0)).nonzero().flatten()
        budget = max(0, max_points-int(keep.sum()))
        candidates = candidates[score[candidates].argsort(descending=True)[:budget]]
        scales = self.log_scale.exp()
        large = (scales[candidates,:3].amax(-1)>spatial_size) | (scales[candidates,3]>temporal_size)
        splits = candidates[large]
        clones = candidates[~large]
        # Every split replaces a parent with two children; every clone adds one.
        parents = torch.cat((splits.repeat_interleave(2),clones))
        children = {name: p[parents].clone() for name,p in self.named_parameters()}
        count = len(splits)*2
        if count:
            sample = torch.randn_like(children["log_scale"][:count])*scales[parents[:count]]
            rotation = rotation4d(self.left[parents[:count]],self.right[parents[:count]])
            offset = (rotation @ sample[:,:,None]).squeeze(-1)
            children["xyz"][:count] += offset[:,:3]
            children["time"][:count] += offset[:,3:]
            children["log_scale"][:count] -= math.log(1.6)
            keep[splits] = False
        if len(parents):
            # ESVR's voxelizer ADDS scalar contributions rather than alpha-
            # compositing them (see docs/DEFORMABLE_ESVR.md and
            # docs/ESVR4D.md), unlike the RGB renderer this split/clone
            # recipe was designed for. A split's two children (or a clone
            # sitting on top of the parent it was copied from) initially
            # overlap almost entirely with the original support, so leaving
            # opacity untouched roughly doubles the local reconstructed
            # value right where a primitive was just added. Halving here
            # compensates for that; verified empirically (docs/ESVR4D.md)
            # to turn densification from net-harmful into net-helpful on
            # real data at matched training steps.
            children["logit_scalar"] -= math.log(2)
        before = len(self.xyz)
        self.replace_population(optimizer,keep,children)
        return dict(before=before, after=len(self.xyz), split=len(splits), cloned=len(clones),
                    pruned=before-int(keep.sum())-len(splits))


def query_block(model, timestamp, shape, start, size, return_package=False):
    from gaussian_renderer import query
    step = 2 / max(shape)
    center = [(a+n/2-s/2)*step for a,n,s in zip(start,size,shape)]
    sliced = model.at(timestamp)
    if not len(sliced.get_xyz):
        # Keep a valid zero-gradient path when every primitive is inactive.
        vol = model.xyz.sum()*0 + model.xyz.new_zeros(size)
        package = dict(vol=vol, visibility_filter=torch.zeros(0,dtype=torch.bool,device=vol.device))
    else:
        package = query(sliced,center,size,[n*step for n in size],
                        SimpleNamespace(compute_cov3D_python=True,debug=False))
    return (package,sliced) if return_package else package["vol"]
