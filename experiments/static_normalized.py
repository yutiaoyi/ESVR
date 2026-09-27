"""Single-frame test: additive ESVR sum vs normalized kernel regression with the
same primitives, initialisation, optimiser and step count (single variable).

  additive:    V(x) = sum_i a_i K_i(x)                       (ESVR)
  normalized:  V(x) = sum_i w_i K_i(x) s_i / max(sum_i w_i K_i(x), eps)
               (Martin et al. 2026, arXiv 2609.16024; Image-GS)

K_i is the ESVR kernel sigmoid(5(1 - d^T S^-1 d)) truncated at d^T S^-1 d > 2,
evaluated by the unchanged CUDA voxelizer. The voxelizer is linear in the
per-primitive amplitude, so the normalized form is two calls (amplitudes
C*w*s and C*w) and a division; C scales both so that the kernel's per-voxel
alpha < 1/255 skip is negligible and cancels in the ratio.

Init (both modes): N centres sampled 50 % proportional to (value + |grad|)^0.6,
35 % uniform, 15 % at local maxima of 2^3 bins; value = GT at the centre;
isotropic scale = (voxels / N)^(1/3) voxels * 0.8, clipped to [0.5, 6] voxels.
Additive amplitudes start at value * overlap-compensation k (R2-Gaussian style).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_esvr.data import VolumeSequence  # noqa: E402
from gaussian_renderer import query  # noqa: E402

PIPE = SimpleNamespace(compute_cov3D_python=False, debug=False)


def render(xyz, scale, rot, amp, shape):
    pc = SimpleNamespace(get_xyz=xyz, get_scaling=scale, get_rotation=rot, get_opacity=amp)
    return query(pc, [0.0, 0.0, 0.0], shape, [2.0 * s / max(shape) for s in shape], PIPE)["vol"]


def init_random(gt, n, gen):
    """ESVR's create_random_onthefly: centres uniform in the volume, isotropic
    scale = sqrt(mean squared distance to the 3 nearest neighbours) (distCUDA2),
    every primitive at value 0.1; the data is not looked at."""
    from simple_knn._C import distCUDA2
    xyz = torch.rand(n, 3, device="cuda", generator=gen) * 2 - 1
    scale = torch.sqrt(torch.clamp_min(distCUDA2(xyz), 1e-7))
    return xyz, torch.full((n,), 0.1, device="cuda"), scale


def init_points(gt, n, gen):
    shape = gt.shape
    g = gt.cuda()
    grad = torch.stack(torch.gradient(g), 0).norm(dim=0)
    w = (g + grad).clamp_min(0).pow(0.6).flatten()
    n_imp, n_uni = int(0.5 * n), int(0.35 * n)
    n_peak = n - n_imp - n_uni
    idx_imp = torch.multinomial(w / w.sum(), n_imp, replacement=True, generator=gen)
    idx_uni = torch.randint(0, g.numel(), (n_uni,), device="cuda", generator=gen)
    # local maxima of 2^3 bins
    pooled, arg = F.max_pool3d(g[None, None], 2, return_indices=True)
    order = pooled.flatten().argsort(descending=True)[:n_peak]
    idx_peak = arg.flatten()[order]
    idx = torch.cat([idx_imp, idx_uni, idx_peak])
    ijk = torch.stack(torch.unravel_index(idx, shape), 1).float()
    jitter = torch.rand(ijk.shape, device="cuda", generator=gen) - 0.5
    step = 2 / max(shape)
    xyz = (ijk + 0.5 + jitter - torch.tensor(shape, device="cuda") / 2) * step
    values = g.flatten()[idx]
    s_vox = min(max((g.numel() / n) ** (1 / 3) * 0.8, 0.5), 6.0)
    return xyz, values, s_vox * step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="experiments/vorts_90_all_train.json")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--n", type=int, default=32000)
    p.add_argument("--mode", choices=("additive", "normalized"), required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=8e-3, help="Base LR (values/weights/log-scale); positions use lr/8, rotations lr/8")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--C", type=float, default=100.0)
    p.add_argument("--k", type=float, default=0.15, help="Additive init amplitude = value * k")
    p.add_argument("--relocate", default="0.35,0.55,0.75", help="Fractions of training at which the 8 %% lowest-support primitives are re-seeded at max-error voxels ('' = off)")
    p.add_argument("--init", choices=("data", "random"), default="data",
                   help="data: sampled from the frame (default); random: ESVR's uniform init, value 0.1, NN scale (additive amplitude 0.1, --k ignored)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    seq = VolumeSequence(args.manifest)
    gt = seq.volumes[args.frame].cuda()
    shape = tuple(gt.shape)
    step_len = 2 / max(shape)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    n = args.n
    if args.init == "random":
        xyz0, values, s_each = init_random(gt, n, gen)
        log_scale = torch.nn.Parameter(torch.log(s_each)[:, None].repeat(1, 3).clamp(math.log(0.3 * step_len), math.log(8 * step_len)).contiguous())
    else:
        xyz0, values, s0 = init_points(gt, n, gen)
        log_scale = torch.nn.Parameter(torch.full((n, 3), math.log(s0), device="cuda"))
    xyz = torch.nn.Parameter(xyz0.clone())
    rot = torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(n, 1))
    logit = lambda v: torch.log(v.clamp(1e-4, 1 - 1e-4) / (1 - v.clamp(1e-4, 1 - 1e-4)))
    if args.mode == "normalized":
        s = torch.nn.Parameter(logit(values)[:, None])
        w = torch.nn.Parameter(torch.zeros(n, 1, device="cuda"))   # sigmoid(0) = 0.5
        extra = [{"params": [s], "lr": args.lr, "name": "s"}, {"params": [w], "lr": args.lr, "name": "w"}]
    else:
        a = torch.nn.Parameter(logit(values * (1.0 if args.init == "random" else args.k))[:, None])
        extra = [{"params": [a], "lr": args.lr, "name": "a"}]
    opt = torch.optim.Adam([{"params": [xyz], "lr": args.lr / 8 * step_len, "name": "xyz"},
                            {"params": [log_scale], "lr": args.lr, "name": "scale"},
                            {"params": [rot], "lr": args.lr / 8, "name": "rot"}] + extra, eps=1e-15)
    base = [g["lr"] for g in opt.param_groups]
    reloc_at = {int(float(f) * args.steps) for f in args.relocate.split(",") if f.strip()}
    floor = 0.3 * step_len

    def forward():
        sc = log_scale.exp()
        sc = sc + (sc.clamp_min(floor) - sc).detach()
        r = F.normalize(rot, dim=-1)
        if args.mode == "normalized":
            wk = w.sigmoid() * args.C
            num = render(xyz, sc, r, wk * s.sigmoid(), shape)
            den = render(xyz, sc, r, wk, shape)
            return (num / den.clamp_min(args.eps * args.C)).clamp(0, 1), den
        return render(xyz, sc, r, a.sigmoid(), shape), None

    log = []
    begin = time.perf_counter()
    for it in range(1, args.steps + 1):
        cos = 0.5 * (1 + math.cos(math.pi * it / args.steps))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * (0.01 + 0.99 * cos)
        pred, den = forward()
        err = pred - gt
        loss = err.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_scale.clamp_(math.log(floor), math.log(8 * step_len))
        if it in reloc_at:
            with torch.no_grad():
                sc = log_scale.exp()
                support = (w.sigmoid() if args.mode == "normalized" else a.sigmoid())[:, 0] * sc.prod(-1)
                m = max(1, int(0.08 * n))
                dead = support.argsort()[:m]
                e = err.detach().abs().flatten()
                top = e.topk(m).indices
                ijk = torch.stack(torch.unravel_index(top, shape), 1).float()
                xyz[dead] = (ijk + 0.5 - torch.tensor(shape, device="cuda") / 2) * step_len
                log_scale[dead] = math.log(0.8 * step_len)
                rot[dead] = torch.tensor([1.0, 0, 0, 0], device="cuda")
                v = gt.flatten()[top]
                if args.mode == "normalized":
                    s[dead] = logit(v)[:, None]; w[dead] = 0.0
                else:
                    a[dead] = logit(err.detach().flatten()[top].neg().clamp_min(1e-3))[:, None]
                for pgroup in opt.param_groups:          # reset Adam moments of re-seeded rows
                    st = opt.state.get(pgroup["params"][0], {})
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in st:
                            st[key][dead] = 0
        if it % 500 == 0 or it == args.steps:
            mse = err.detach().double().square().mean().item()
            row = dict(step=it, psnr_db=-10 * math.log10(mse), l1=loss.item())
            log.append(row); print(json.dumps(row), flush=True)
    with torch.no_grad():
        pred, _ = forward()
        mse = (pred.double() - gt.double()).square().mean().item()
    floats = n * (3 + 3 + 4 + (2 if args.mode == "normalized" else 1))
    res = dict(mode=args.mode, n=n, frame=args.frame, steps=args.steps, psnr_db=-10 * math.log10(mse), mb_fp32=floats * 4 / 1e6,
               seconds=time.perf_counter() - begin, args=vars(args))
    (out / "result.json").write_text(json.dumps(res, indent=2)); (out / "log.json").write_text(json.dumps(log))
    print("DONE " + json.dumps({k: res[k] for k in ("mode", "n", "psnr_db", "mb_fp32")}), flush=True)


if __name__ == "__main__":
    main()
