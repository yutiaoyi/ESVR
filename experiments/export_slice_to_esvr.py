"""Export one time slice of a scaffold4d model as a static ESVR checkpoint.

Diagnostic for "is the bottleneck the shared MLP parameterisation or the
primitive count/layout?": the children selected at time t (position, scale,
rotation, intensity x temporal gate) become free ESVR primitives, which
train_chunk.py can then optimise with ESVR's own optimiser (--start_checkpoint).
Before writing, the slice is rendered through the same voxelizer and its PSNR
against the ground-truth frame is printed, so the conversion is verified.

Usage: export_slice_to_esvr.py RUN_DIR --manifest M --frame INDEX --out CKPT
       [--random N]   (instead: N random primitives, same init as ESVR's create_random)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_esvr.data import VolumeSequence  # noqa: E402
from scaffold4d.model import ScaffoldESVR  # noqa: E402
from gaussian_renderer import query  # noqa: E402
from types import SimpleNamespace  # noqa: E402

SPATIAL_LR_SCALE = 11.02160816192627   # scene/__init__.py, what train_chunk uses


def render_psnr(xyz, scale, rot, opa, gt):
    shape = tuple(gt.shape)
    with torch.no_grad():
        pc = SimpleNamespace(get_xyz=xyz, get_scaling=scale, get_rotation=rot, get_opacity=opa)
        vol = query(pc, [0.0, 0.0, 0.0], shape, [2.0 * s / max(shape) for s in shape],
                    SimpleNamespace(compute_cov3D_python=False, debug=False))["vol"]
    mse = (vol.double() - gt.cuda().double()).square().mean().item()
    return -10 * math.log10(mse)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--frame", type=int, required=True, help="0-based frame index in the manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--random", type=int, default=0)
    args = p.parse_args()
    seq = VolumeSequence(args.manifest)
    gt = seq.volumes[args.frame]
    t = seq.times[args.frame]
    if args.random:
        n = args.random
        xyz = (torch.rand(n, 3, device="cuda") * 2 - 1)
        scale = torch.full((n, 3), 2 / 128 * 1.5, device="cuda")
        rot = torch.zeros(n, 4, device="cuda"); rot[:, 0] = 1
        opa = torch.full((n, 1), 0.1, device="cuda")
    else:
        ck = torch.load(Path(args.run) / "checkpoint.pt", map_location="cuda", weights_only=False)
        model = ScaffoldESVR(**ck["model_config"]).cuda()
        model.load_state_dict(ck["model"], strict=False)
        with torch.no_grad():
            s = model.at(t)
            xyz, scale, rot, opa = s.get_xyz.detach(), s.get_scaling.detach(), s.get_rotation.detach(), s.get_opacity.detach()
        rot = torch.nn.functional.normalize(rot, dim=-1)
    opa = opa.clamp(1e-6, 1 - 1e-6)
    psnr = render_psnr(xyz, scale, rot, opa, gt)
    print(json.dumps(dict(source=("random" if args.random else args.run), frame=args.frame, time=t, primitives=len(xyz), psnr_db=round(psnr, 3))))

    params = dict(xyz=nn.Parameter(xyz.clone()), opacity=nn.Parameter(torch.log(opa / (1 - opa))),
                  scaling=nn.Parameter(scale.clamp_min(1e-8).log()), rotation=nn.Parameter(rot.clone()))
    # Same group order and names as GaussianModel.training_setup so restore() can load the state dict.
    optimizer = torch.optim.Adam([{"params": [params[k]], "lr": 0.0, "name": k} for k in ("xyz", "opacity", "scaling", "rotation")], eps=1e-15)
    n = len(xyz)
    capture = (params["xyz"], params["scaling"], params["rotation"], params["opacity"],
               torch.zeros(n, device="cuda"), torch.zeros(n, 1, device="cuda"), torch.zeros(n, 1, device="cuda"),
               optimizer.state_dict(), SPATIAL_LR_SCALE)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save((capture, 0), args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
