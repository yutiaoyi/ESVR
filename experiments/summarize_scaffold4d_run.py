"""Audit and summarize a completed scaffold4d full-sequence reconstruction run.

Mirrors experiments/summarize_dynamic_run.py / summarize_esvr4d_run.py's
outputs (quality_curves.png, slice_comparison.png, training_frames_dvr.gif,
summary.json, data_sha256.json, provenance.json) adapted for scaffold4d's
checkpoint format (anchors/children instead of points; no PLY export, see
docs/ESVR_SCAFFOLD4D.md limitations).
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/esvr-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from dynamic_esvr.data import VolumeSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    seq = VolumeSequence(args.manifest)
    evaluation = run / "evaluation"
    rows = json.loads((evaluation / "metrics.json").read_text())
    if len(rows) != len(seq.frames):
        raise ValueError("Evaluation incomplete")
    for i, row in enumerate(rows):
        if row["frame"] != i or row["time"] != seq.frames[i]["time"]:
            raise ValueError("Evaluation ordering mismatch")
        for suffix in (".npy", ".raw", "_axis0_gt_pred.png", "_axis1_gt_pred.png", "_axis2_gt_pred.png"):
            if not (evaluation / f"frame_{i:04d}{suffix}").is_file():
                raise ValueError(f"Missing frame artifact {i}: {suffix}")
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint["step"] != checkpoint["args"]["steps"]:
        raise ValueError("Training schedule incomplete")
    audit = []
    for frame in seq.frames:
        path = (seq.path.parent / frame["path"]).resolve()
        audit.append(dict(path=str(path), bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    model_bytes = sum(t.numel()*t.element_size() for t in checkpoint["model"].values())
    torch.save(dict(model_config=checkpoint["model_config"], model=checkpoint["model"],
                    manifest=checkpoint["manifest"]), run / "model_inference.pt")
    psnr = np.array([r["psnr_db"] for r in rows])
    counts = checkpoint["frame_counts"]
    logs = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
    result = dict(frames=len(rows), steps=checkpoint["step"], shape_xyz=list(seq.shape),
        method="scaffold4d: anchor/offset scaffold, shared MLP decoders, constant-velocity motion, no PLY export yet",
        densification=("enabled, native window" if checkpoint["args"]["densify_from"] < checkpoint["args"]["densify_until"]
                       else "disabled (fixed population)"),
        training_frames=len(seq.train_indices),
        evaluation=("All frames used for training; reconstruction metrics, NOT held-out generalization"
                    if len(seq.train_indices)==len(seq.frames) else "Mixed train/test; see per-frame splits in metrics.json"),
        final_anchors=checkpoint["model_config"]["anchors"],
        children_per_anchor=checkpoint["model_config"]["children"],
        initial_anchors=checkpoint["args"]["anchors"],
        mean_frame_psnr_db=float(psnr.mean()), median_frame_psnr_db=float(np.median(psnr)),
        min_frame_psnr_db=float(psnr.min()), max_frame_psnr_db=float(psnr.max()),
        pooled_psnr_db=-10*math.log10(np.mean([r["mse"] for r in rows])),
        mean_mae_normalized=float(np.mean([r["mae"] for r in rows])),
        pooled_rmse_original_units=math.sqrt(np.mean([r["mse"] for r in rows]))*(seq.hi-seq.lo),
        min_frame_samples=min(counts), max_frame_samples=max(counts),
        training_seconds=logs[-1]["seconds"], model_tensor_bytes=model_bytes,
        inference_file_bytes=(run/"model_inference.pt").stat().st_size,
        optimizer_checkpoint_bytes=(run/"checkpoint.pt").stat().st_size,
        source_bytes=sum(r["bytes"] for r in audit),
        size_note="Float32 model serialization, not entropy-coded compression; raw source size is only a reference")
    (run / "summary.json").write_text(json.dumps(result, indent=2))
    (run / "data_sha256.json").write_text(json.dumps(audit, indent=2))
    repo = Path(__file__).resolve().parents[1]
    sources = ["train_scaffold4d.py", "inference_scaffold4d.py", "scaffold4d/model.py", "dynamic_esvr/data.py",
               "utils/loss_utils.py", "gaussian_renderer/__init__.py", "experiments/render_preview.py",
               "experiments/summarize_scaffold4d_run.py",
               "submodules/gaussian-rasterization-voxelization/cuda_voxelizer/backward.cu"]
    provenance = dict(python=sys.version, torch=torch.__version__, torch_cuda=torch.version.cuda,
                      source_sha256={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in sources})
    (run / "provenance.json").write_text(json.dumps(provenance, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([r["time"] for r in rows], psnr)
    axes[0].set(xlabel="Frame number (equally spaced)", ylabel="PSNR (dB)", title="All training frames; unclipped reconstruction")
    monitored = [json.loads(line) for line in (run / "monitor.jsonl").read_text().splitlines()]
    for i in range(len(monitored[0]["frames"])):
        axes[1].plot([r["step"] for r in monitored], [r["frames"][i]["psnr_db"] for r in monitored],
                     label=f"Frame {monitored[0]['frames'][i]['time']}")
    axes[1].set(xlabel="Optimizer step", ylabel="PSNR (dB)", title="Full-volume monitoring")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(run / "quality_curves.png", dpi=150)
    plt.close(fig)
    chosen = np.linspace(0, len(rows)-1, 6).round().astype(int)
    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 3*len(chosen)))
    for row_index, i in enumerate(chosen):
        gt = seq.volumes[i].numpy()[:, :, seq.shape[2]//2].T
        pred = np.load(evaluation / f"frame_{i:04d}.npy", mmap_mode="r")[:, :, seq.shape[2]//2].T
        for col, (label, image) in enumerate([("GT", gt), ("Prediction", pred), ("Absolute error", np.abs(gt-pred))]):
            axes[row_index,col].imshow(image, origin="lower", cmap="inferno" if col==2 else "viridis",
                                      vmin=0, vmax=0.2 if col==2 else 1)
            axes[row_index,col].set_title(f"Frame {seq.frames[i]['time']} / {label}")
            axes[row_index,col].axis("off")
    fig.suptitle("Central Z slices; common [0,1] range; error display [0,0.2]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(run / "slice_comparison.png", dpi=120)
    plt.close(fig)
    animation = []
    for i in range(len(rows)):
        with Image.open(evaluation / f"frame_{i:04d}_axis2_gt_pred.png") as source:
            image = source.convert("RGB").resize((512,256))
        canvas = Image.new("RGB", (512,280), "black")
        canvas.paste(image, (0,24))
        ImageDraw.Draw(canvas).text((8,6), f"Time {seq.frames[i]['time']:g}   GT (left) | Prediction (right)", fill="white")
        animation.append(canvas)
    animation[0].save(run / "training_frames_dvr.gif", save_all=True, append_images=animation[1:], duration=100, loop=0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
