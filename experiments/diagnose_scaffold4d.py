"""Population diagnostics for a scaffold4d checkpoint: how many children are
alive (intensity above the kernel's 1/255 cutoff), how many sit on the scale
floor, sizes/intensities of the useful ones, sigma-clamp saturation and
per-anchor usefulness. Usage: diagnose_scaffold4d.py RUN_DIR [RUN_DIR ...]"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaffold4d.model import ScaffoldESVR


def pct(x, ps=(5, 25, 50, 75, 95)):
    return np.round(np.percentile(x.detach().flatten().cpu().numpy(), ps), 2).tolist()


@torch.no_grad()
def diagnose(run):
    ck = torch.load(Path(run) / "checkpoint.pt", map_location="cuda", weights_only=False)
    model = ScaffoldESVR(**ck["model_config"]).cuda()
    model.load_state_dict(ck["model"], strict=False)
    voxels = max(model.config["shape"]) / 2
    g = model.generate()
    intensity = g.intensity[..., 0]
    scale = g.scale * voxels
    floor = model.config["scale_floor"] * voxels
    alive = intensity > 1 / 255
    above = scale.amax(-1) > floor * 1.2
    useful = alive & above
    out = dict(run=str(run), anchors=len(model.anchor), children=int(intensity.numel()),
               alive=float(alive.float().mean()), useful=float(useful.float().mean()),
               alive_at_floor=float((alive & ~above).float().mean()), dead_at_floor=float((~alive & ~above).float().mean()),
               useful_per_anchor=float(useful.sum(1).float().mean()), anchors_without_useful=float((useful.sum(1) == 0).float().mean()),
               sigma_at_max=float((g.sigma >= model.config["sigma_max"] * 0.999).float().mean()),
               anchors_outside=int((model.anchor[:, :3].abs() > 1.05).any(-1).sum()))
    if useful.any():
        srt = scale[useful].sort(-1).values
        out.update(useful_max_axis_vox_pct=pct(srt[:, 2]), useful_min_axis_vox_pct=pct(srt[:, 0]),
                   useful_intensity_pct=pct(intensity[useful]),
                   useful_halfwidth_frames_pct=pct((math.log(2) ** 0.25) / g.sigma[..., 0][useful] * (len(ck["manifest"]["frames"]) - 1)))
    per_time = []
    for t in (0.0, 0.5, 1.0):
        s = model.at(t, g)
        u = useful.reshape(-1)[s.child_index]
        per_time.append(dict(t=t, selected=len(s.child_index), useful=int(u.sum()), intensity_sum=float(s.get_opacity[u].sum())))
    out["per_time"] = per_time
    return out


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(json.dumps(diagnose(run)))
