"""Same three figures as experiments/summarize_scaffold4d_run.py for models that
do not write per-frame predictions: normalized explicit 4D (train_norm4d.py),
explicit additive 4D (finetune_explicit4d.py) and per-frame static ESVR
(train_chunk.py, one PLY per frame).

Writes into --out (default: the run folder): quality_curves.png,
slice_comparison.png (central Z slices of 6 frames: GT | prediction | |error|),
training_frames_dvr.gif (GT DVR | prediction DVR | voxel |error| mean
projection | rendered-image |error|, error panels inferno 0-0.1) and
visualize_metrics.json (per-frame PSNR recomputed here; compared with the
run's own metrics when present).

Usage:
  visualize_run.py --kind norm4d     --run output/scaffold4d/norm4d/N4A_400k_hw2_90k
  visualize_run.py --kind explicit4d --run output/scaffold4d/explicit/X1_fromP3_60k
  visualize_run.py --kind perframe   --run output/experiments/per_frame_fixed/S1b
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/esvr-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from dynamic_esvr.data import VolumeSequence


def predictor(kind, run, seq):
    run = Path(run)
    if kind == "norm4d":
        from train_norm4d import Norm4D
        ck = torch.load(run / "checkpoint.pt", map_location="cuda", weights_only=False)
        c = ck["model_config"]
        m = Norm4D(c["n"], c["shape"], beta=c["beta"], cutoff=c["cutoff"], eps=c["eps"], C=c["C"],
                   floor_vox=c["floor_vox"], sigma_max=c["sigma_max"]).cuda()
        m.load_state_dict(ck["model"])
        return lambda i: m(seq.times[i], seq.shape)[0]
    if kind == "explicit4d":
        from scaffold4d.explicit import ExplicitESVR4D
        from scaffold4d.model import query_block
        ck = torch.load(run / "checkpoint.pt", map_location="cuda", weights_only=False)
        c = ck["model_config"]
        m = ExplicitESVR4D(c["n"], c["shape"], c["temporal_beta"], c["temporal_cutoff"], c["scale_floor"], c["sigma_max"]).cuda()
        m.load_state_dict(ck["model"], strict=False)
        return lambda i: query_block(m, seq.times[i], seq.shape, (0, 0, 0), seq.shape)
    if kind == "perframe":
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import query

        def pred(i):
            d = run / f"vorts_frame_{int(seq.frames[i]['time']):04d}_chunk_0"
            ply = max(glob.glob(str(d / "point_cloud" / "iteration_*" / "point_cloud.ply")),
                      key=lambda p: int(re.search(r"iteration_(\d+)", p).group(1)))
            gm = GaussianModel(*seq.shape)
            gm.load_ply(ply)
            return query(gm, [0.0, 0.0, 0.0], seq.shape, [2.0 * s / max(seq.shape) for s in seq.shape],
                         SimpleNamespace(compute_cov3D_python=False, debug=False))["vol"]
        return pred
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("norm4d", "explicit4d", "perframe"), required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", default="experiments/vorts_90_all_train.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    run = Path(args.run)
    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)
    seq = VolumeSequence(args.manifest)
    pred_fn = predictor(args.kind, run, seq)
    from render_preview import dvr
    from matplotlib import cm
    inferno = cm.get_cmap("inferno")
    to_img = lambda a: Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((256, 256))
    chosen = set(np.linspace(0, len(seq.frames) - 1, 6).round().astype(int).tolist())
    rows, slices, animation = [], {}, []
    for i in range(len(seq.frames)):
        with torch.no_grad():
            pred_t = pred_fn(i)
        pred = pred_t.detach().float().cpu().numpy()
        gt = seq.volumes[i].numpy()
        mse = float(((pred.astype(np.float64) - gt) ** 2).mean())
        rows.append(dict(frame=i, time=seq.frames[i]["time"], mse=mse, psnr_db=-10 * math.log10(max(mse, 1e-30))))
        if i in chosen:
            slices[i] = (gt[:, :, seq.shape[2] // 2].T, pred[:, :, seq.shape[2] // 2].T)
        gt_img, pred_img = dvr(gt, 2), dvr(pred, 2)
        voxel_err = np.flipud(np.swapaxes(np.abs(gt - pred).mean(axis=2), 0, 1))
        image_err = np.abs(gt_img - pred_img).mean(-1)
        panels = [to_img(gt_img), to_img(pred_img),
                  to_img(inferno(np.clip(voxel_err / 0.1, 0, 1))[..., :3]),
                  to_img(inferno(np.clip(image_err / 0.1, 0, 1))[..., :3])]
        canvas = Image.new("RGB", (1024, 288), "black")
        for k, panel in enumerate(panels):
            canvas.paste(panel, (k * 256, 32))
        ImageDraw.Draw(canvas).text((8, 8), f"Frame {seq.frames[i]['time']:g}   GT | Prediction | Voxel |error| mean-projection (0-0.1) | Rendered-image |error| (0-0.1)", fill="white")
        animation.append(canvas)
    psnr = np.array([r["psnr_db"] for r in rows])
    stored = run / "evaluation" / "metrics.json"
    check = None
    if stored.exists():
        s = json.loads(stored.read_text())
        if len(s) == len(rows):
            check = float(max(abs(a["psnr_db"] - b["psnr_db"]) for a, b in zip(s, rows)))
    (out / "visualize_metrics.json").write_text(json.dumps(dict(kind=args.kind, mean_psnr=float(psnr.mean()), min_psnr=float(psnr.min()),
                                                              max_psnr=float(psnr.max()), max_abs_diff_vs_stored=check, frames=rows), indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([r["time"] for r in rows], psnr)
    axes[0].set(xlabel="Frame number (equally spaced)", ylabel="PSNR (dB)", title=f"All training frames; mean {psnr.mean():.2f} dB")
    mon = run / "monitor.jsonl"
    if mon.exists():
        monitored = [json.loads(line) for line in mon.read_text().splitlines() if line.strip()]
        if monitored and "frames" in monitored[0]:
            for k in range(len(monitored[0]["frames"])):
                axes[1].plot([r["step"] for r in monitored], [r["frames"][k]["psnr_db"] for r in monitored],
                             label=f"Frame {monitored[0]['frames'][k]['time']}")
            axes[1].set(xlabel="Optimizer step", ylabel="PSNR (dB)", title="Full-volume monitoring")
            axes[1].legend()
    else:
        axes[1].hist(psnr, bins=20)
        axes[1].set(xlabel="PSNR (dB)", ylabel="frames", title="Per-frame PSNR distribution")
    fig.tight_layout()
    fig.savefig(out / "quality_curves.png", dpi=150)
    plt.close(fig)
    order = sorted(slices)
    fig, axes = plt.subplots(len(order), 3, figsize=(9, 3 * len(order)))
    for r, i in enumerate(order):
        gt, pr = slices[i]
        for col, (label, image) in enumerate([("GT", gt), ("Prediction", pr), ("Absolute error", np.abs(gt - pr))]):
            axes[r, col].imshow(image, origin="lower", cmap="inferno" if col == 2 else "viridis", vmin=0, vmax=0.2 if col == 2 else 1)
            axes[r, col].set_title(f"Frame {seq.frames[i]['time']} / {label}")
            axes[r, col].axis("off")
    fig.suptitle("Central Z slices; common [0,1] range; error display [0,0.2]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "slice_comparison.png", dpi=120)
    plt.close(fig)
    animation[0].save(out / "training_frames_dvr.gif", save_all=True, append_images=animation[1:], duration=100, loop=0)
    print(json.dumps(dict(out=str(out), mean_psnr=float(psnr.mean()), max_abs_diff_vs_stored=check)))


if __name__ == "__main__":
    main()
