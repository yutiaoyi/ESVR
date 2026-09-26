"""Collect report inputs for finished Scaffold4D runs.

Adds/refreshes the run's row in s4d_runs.json, its population statistics in
s4d_stats.json and its compaction report in s4d_compact.json (all next to this
file), and — for runs named in --assets — writes the page thumbnails, full-size
images and the MP4 animation into output/report_artifact/img/.

Usage: collect_scaffold4d_run.py GROUP/RUN [GROUP/RUN ...] [--assets GROUP/RUN,...]
e.g.   collect_scaffold4d_run.py next/Q1_floor_st_480k --assets next/Q1_floor_st_480k
Run with .venv/bin/python from the repository root (needs the CUDA voxelizer).
"""
import json
import os
import statistics
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from scaffold4d.model import ScaffoldESVR  # noqa: E402
from dynamic_esvr.data import VolumeSequence  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "output", "report_artifact", "img")


def main():
    args = sys.argv[1:]
    assets = set()
    if "--assets" in args:
        i = args.index("--assets")
        assets = set(args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
    rows = json.load(open(os.path.join(D, "s4d_runs.json")))
    stats = json.load(open(os.path.join(D, "s4d_stats.json")))
    compact = json.load(open(os.path.join(D, "s4d_compact.json")))
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover
        ffmpeg = None
    for key in args:
        run = os.path.join(ROOT, "output", "scaffold4d", key)
        cfg = json.load(open(f"{run}/config.json"))
        metrics = json.load(open(f"{run}/evaluation/metrics.json"))
        seq = VolumeSequence(os.path.join(ROOT, cfg["manifest"]) if not os.path.isabs(cfg["manifest"]) else cfg["manifest"])
        times = seq.times
        assert len(metrics) == len(times), (key, len(metrics))
        ck = torch.load(f"{run}/checkpoint.pt", map_location="cuda", weights_only=False)
        c = ck["model_config"]
        mb = sum(t.numel() * t.element_size() for t in ck["model"].values()) / 1e6
        train = [json.loads(l) for l in open(f"{run}/train.jsonl")]
        ps = [x["psnr_db"] for x in metrics]
        rows = [r for r in rows if r["run"] != key]
        rows.append(dict(run=key, psnr=round(statistics.mean(ps), 3), pmin=round(min(ps), 2), pmax=round(max(ps), 2),
                         anchors=c["anchors"], children=c["children"], mb=round(mb, 2), seconds=int(train[-1]["seconds"]),
                         steps=cfg["steps"], frames=len(times)))
        if os.path.exists(f"{run}/compact/report.json"):
            compact[key] = json.load(open(f"{run}/compact/report.json"))
            if os.path.exists(f"{run}/compact_dropnever/report.json"):
                compact[key]["dropnever"] = json.load(open(f"{run}/compact_dropnever/report.json"))
        n, k = c["anchors"], c["children"]
        last = train[-1]["step"]
        active = int(np.mean([r["active"] for r in train if r["step"] > last - 2000]))
        model = ScaffoldESVR(**c).cuda()
        model.load_state_dict(ck["model"], strict=False)
        with torch.no_grad():
            g = model.generate()
            intensity = g.intensity[..., 0]
            useful = ((intensity > 1 / 255) & (g.scale.amax(-1) > c.get("scale_floor", 1e-4) * 1.2)).reshape(-1)
            mid = model.at(0.5, g)
            useful_query = int(useful[mid.child_index].sum())
            ever = torch.zeros(useful.numel(), dtype=torch.bool, device="cuda")
            per, act = [], []
            for t in times:
                sl = model.at(t, g)
                ever[sl.child_index] = True
                per.append(int(useful[sl.child_index].sum()))
                act.append(len(sl.child_index))
            never = ~ever
            sigma_at_max = float((g.sigma >= c["sigma_max"] * 0.999).float().mean())
            hw = ((np.log(2) ** 0.25) / g.sigma[..., 0][useful.reshape(intensity.shape)] * (len(times) - 1 if len(times) > 1 else 1)).flatten().cpu().numpy()
        stats[key] = dict(anchors=n, children=k, total=n * k, active=active, useful_query=useful_query, useful_total=int(useful.sum()),
                          useful_query_mean=int(np.mean(per)), useful_query_min=int(min(per)), useful_query_max=int(max(per)),
                          active_query_mean=int(np.mean(act)), never_selected=int(never.sum()),
                          useful_never_selected=int((useful & never).sum()), frames=len(times),
                          sigma_at_max=round(sigma_at_max, 3), halfwidth_frames_p50=round(float(np.median(hw)), 2) if len(hw) else None)
        del model, ck
        torch.cuda.empty_cache()
        if key in assets:
            ik = "scaffold4d__" + key.replace("/", "__")
            os.makedirs(IMG, exist_ok=True)
            for src, width, tag in (("quality_curves.png", 700, "curve"), ("slice_comparison.png", 360, "slice"),
                                    ("quality_curves.png", 1100, "curve_full"), ("slice_comparison.png", 1080, "slice_full")):
                im = Image.open(f"{run}/{src}").convert("RGB")
                im.thumbnail((width, 10000))
                im.save(f"{IMG}/{ik}__{tag}.jpg", quality=85 if not tag.endswith("_full") else 72, optimize=True)
            if ffmpeg:
                subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", f"{run}/training_frames_dvr.gif",
                                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=10", "-c:v", "libx264", "-preset", "slow",
                                "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart", f"{IMG}/{ik}__dvr.mp4"], check=True)
        print(key, json.dumps(stats[key]))
    json.dump(rows, open(os.path.join(D, "s4d_runs.json"), "w"), indent=1)
    json.dump(stats, open(os.path.join(D, "s4d_stats.json"), "w"), indent=1)
    json.dump(compact, open(os.path.join(D, "s4d_compact.json"), "w"), indent=1)
    print(len(rows), "runs")


if __name__ == "__main__":
    main()
