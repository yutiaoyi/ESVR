"""Explicit 4D primitives with NORMALIZED kernel regression, trained on all frames.

    V(x, t) = sum_i w_i g_i(t) K_i(x, t) s_i / max(sum_i w_i g_i(t) K_i(x, t), eps)
    g_i(t)  = exp(-|(t_i - t) sigma_i|^beta)            (temporal gate, 4D-Scaffold-GS form)
    K_i     = ESVR kernel at centre mu_i + v_i (t_i - t)  (linear motion)

Per primitive: centre (3), home time (1), log-scale (3), quaternion (4), value
logit (1), weight logit (1), log-sigma (1), velocity (3) = 17 floats.
The CUDA voxelizer is linear in the amplitude, so numerator and denominator are
two calls (amplitudes C*w*g*s and C*w*g). Single-frame evidence (vorts frame 1,
32k primitives, same init/optimiser): additive 42.8 dB vs normalized 55.7 dB.

Initialisation from the data in 4D: each primitive picks a training frame
uniformly, then a voxel of that frame (50 % importance ~ (value + |grad|)^0.6,
35 % uniform, 15 % 2^3-bin peaks), value = GT there, home time = that frame,
temporal half-width --init-halfwidth frames, isotropic scale from the per-frame
density, zero velocity.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_esvr.data import VolumeSequence  # noqa: E402
from gaussian_renderer import query  # noqa: E402

PIPE = SimpleNamespace(compute_cov3D_python=False, debug=False)


def render(xyz, scale, rot, amp, shape):
    pc = SimpleNamespace(get_xyz=xyz, get_scaling=scale, get_rotation=rot, get_opacity=amp)
    return query(pc, [0.0, 0.0, 0.0], shape, [2.0 * s / max(shape) for s in shape], PIPE)["vol"]


class Norm4D(torch.nn.Module):
    names = ("xyz", "t", "log_scale", "rot", "s", "w", "log_sigma", "flow")

    def __init__(self, n, shape, beta=4.0, cutoff=1e-3, eps=0.05, C=100.0, floor_vox=0.3, sigma_max=1e3):
        super().__init__()
        self.config = dict(n=n, shape=list(shape), beta=beta, cutoff=cutoff, eps=eps, C=C, floor_vox=floor_vox, sigma_max=sigma_max)
        z = lambda *s: torch.nn.Parameter(torch.zeros(*s))
        self.xyz, self.t, self.log_scale, self.s, self.w, self.log_sigma, self.flow = z(n, 3), z(n, 1), z(n, 3), z(n, 1), z(n, 1), z(n, 1), z(n, 3)
        self.rot = torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1))

    def forward(self, t, shape):
        c = self.config
        step = 2 / max(shape)
        sigma = self.log_sigma.exp().clamp(1e-2, c["sigma_max"])
        dt = self.t - t
        gate = torch.exp(-(dt * sigma).abs().pow(c["beta"]))
        wg = self.w.sigmoid() * gate
        m = wg[:, 0] > c["cutoff"]
        sc = self.log_scale.exp()
        sc = sc + (sc.clamp_min(c["floor_vox"] * step) - sc).detach()
        xyz = (self.xyz + self.flow * dt)[m]
        r = F.normalize(self.rot[m], dim=-1)
        a = wg[m] * c["C"]
        num = render(xyz, sc[m], r, a * self.s[m].sigmoid(), shape)
        den = render(xyz, sc[m], r, a, shape)
        return (num / den.clamp_min(c["eps"] * c["C"])).clamp(0, 1), int(m.sum())


@torch.no_grad()
def init_from_data(model, seq, halfwidth, gen):
    n = model.config["n"]
    shape = seq.shape
    step = 2 / max(shape)
    frames = torch.tensor(seq.train_indices)[torch.randint(0, len(seq.train_indices), (n,), generator=gen)]
    xyz = torch.empty(n, 3); val = torch.empty(n); tt = torch.empty(n)
    per_frame = max(1, n // len(seq.train_indices))
    for f in frames.unique().tolist():
        sel = (frames == f).nonzero().flatten()
        k = len(sel)
        g = seq.volumes[f].cuda()
        grad = torch.stack(torch.gradient(g), 0).norm(dim=0)
        wts = (g + grad).clamp_min(0).pow(0.6).flatten()
        ki, ku = int(0.5 * k), int(0.35 * k)
        kp = k - ki - ku
        gg = torch.Generator(device="cuda").manual_seed(int(torch.randint(0, 2**31, (1,), generator=gen)))
        idx = [torch.multinomial(wts / wts.sum(), ki, replacement=True, generator=gg),
               torch.randint(0, g.numel(), (ku,), device="cuda", generator=gg)]
        pooled, arg = F.max_pool3d(g[None, None], 2, return_indices=True)
        pk = arg.flatten()[pooled.flatten().argsort(descending=True)[torch.randperm(pooled.numel() // 4, generator=gg, device="cuda")[:kp]]]
        idx.append(pk)
        idx = torch.cat(idx)
        ijk = torch.stack(torch.unravel_index(idx, shape), 1).float()
        ijk += torch.rand(ijk.shape, device="cuda", generator=gg) - 0.5
        xyz[sel] = ((ijk + 0.5 - torch.tensor(shape, device="cuda") / 2) * step).cpu()
        val[sel] = g.flatten()[idx].cpu()
        tt[sel] = seq.times[f]
    s_vox = min(max((math.prod(shape) / per_frame) ** (1 / 3) * 0.8, 0.5), 6.0)
    dt_frame = min(b - a for a, b in zip(sorted(seq.times), sorted(seq.times)[1:])) if len(seq.times) > 1 else 1.0
    sigma = (math.log(2) ** 0.25) / (halfwidth * dt_frame)
    v = val.clamp(1e-4, 1 - 1e-4)
    model.xyz.copy_(xyz); model.t.copy_(tt[:, None]); model.s.copy_(torch.log(v / (1 - v))[:, None])
    model.log_scale.fill_(math.log(s_vox * step)); model.log_sigma.fill_(math.log(sigma))
    return s_vox


@torch.no_grad()
def evaluate(model, seq, idx):
    rows = []
    for i in idx:
        pred, active = model(seq.times[i], seq.shape)
        mse = (pred.double() - seq.volumes[i].cuda().double()).square().mean().item()
        rows.append(dict(frame=i, time=seq.frames[i]["time"], mse=mse, psnr_db=-10 * math.log10(max(mse, 1e-30)), active=active))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=400000)
    p.add_argument("--steps", type=int, default=90000)
    p.add_argument("--init-halfwidth", type=float, default=2.0, help="Initial temporal half-width in frames")
    p.add_argument("--lr", type=float, default=8e-3)
    p.add_argument("--lr-t", type=float, default=1e-3, help="Home-time LR (normalised time; one frame = 1/(n-1))")
    p.add_argument("--lr-flow", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        p.error("Choose a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    from utils.voxelizer_provenance import record as record_voxelizer
    record_voxelizer(out)
    seq = VolumeSequence(args.manifest)
    shape = seq.shape
    step_len = 2 / max(shape)
    model = Norm4D(args.n, shape, beta=args.beta).cuda()
    s_vox = init_from_data(model, seq, args.init_halfwidth, torch.Generator().manual_seed(args.seed))
    (out / "config.json").write_text(json.dumps({**vars(args), "init_scale_vox": s_vox, "model_config": model.config}, indent=2))
    groups = [("xyz", [model.xyz], args.lr / 8 * step_len), ("t", [model.t], args.lr_t), ("log_scale", [model.log_scale], args.lr),
              ("rot", [model.rot], args.lr / 8), ("s", [model.s], args.lr), ("w", [model.w], args.lr),
              ("log_sigma", [model.log_sigma], args.lr), ("flow", [model.flow], args.lr_flow)]
    opt = torch.optim.Adam([{"params": ps, "lr": lr, "name": nm} for nm, ps, lr in groups], eps=1e-15)
    base = [g["lr"] for g in opt.param_groups]
    mon_idx = [0, len(seq.frames) // 2, len(seq.frames) - 1]
    begin = time.perf_counter()
    with (out / "train.jsonl").open("w") as log, (out / "monitor.jsonl").open("w") as mon:
        for it in range(1, args.steps + 1):
            cos = 0.5 * (1 + math.cos(math.pi * it / args.steps))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * (0.01 + 0.99 * cos)
            i = random.choice(seq.train_indices)
            pred, active = model(seq.times[i], shape)
            loss = (pred - seq.volumes[i].cuda()).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.log_scale.clamp_(math.log(0.3 * step_len), math.log(8 * step_len))
                model.log_sigma.clamp_(math.log(1e-2), math.log(model.config["sigma_max"]))
            if it % 100 == 0:
                log.write(json.dumps(dict(step=it, frame=i, l1=loss.item(), active=active, seconds=time.perf_counter() - begin)) + "\n")
            if it % args.eval_every == 0 or it == args.steps:
                rows = evaluate(model, seq, mon_idx)
                mon.write(json.dumps(dict(step=it, frames=rows)) + "\n"); mon.flush()
                print("MONITOR " + json.dumps(dict(step=it, mean=float(np.mean([r["psnr_db"] for r in rows])), active=[r["active"] for r in rows])), flush=True)
    torch.save(dict(model_config=model.config, model=model.state_dict(), args=vars(args)), out / "checkpoint.pt")
    rows = evaluate(model, seq, range(len(seq.frames)))
    (out / "evaluation").mkdir(exist_ok=True)
    (out / "evaluation" / "metrics.json").write_text(json.dumps(rows, indent=2))
    ps = [r["psnr_db"] for r in rows]
    nparam = sum(p.numel() for p in model.parameters())
    summary = dict(n=args.n, mb_fp32=nparam * 4 / 1e6, mb_fp16=nparam * 2 / 1e6, mean_psnr=float(np.mean(ps)), min_psnr=float(min(ps)),
                   max_psnr=float(max(ps)), active_mean=float(np.mean([r["active"] for r in rows])), seconds=time.perf_counter() - begin)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
