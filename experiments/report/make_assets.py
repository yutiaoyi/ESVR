"""Page assets for one run folder: thumbnails, full-size images and an MP4 of
training_frames_dvr.gif, written to output/report_artifact/img/KEY__{curve,slice,curve_full,slice_full}.jpg
and KEY__dvr.mp4. Usage: make_assets.py RUN_DIR KEY [RUN_DIR KEY ...]"""
import os, subprocess, sys
from PIL import Image
import imageio_ffmpeg
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG = os.path.join(ROOT, "output", "report_artifact", "img")
FF = imageio_ffmpeg.get_ffmpeg_exe()
args = sys.argv[1:]
for run, key in zip(args[::2], args[1::2]):
    for src, width, tag, q in (("quality_curves.png", 700, "curve", 85), ("slice_comparison.png", 360, "slice", 85),
                               ("quality_curves.png", 1100, "curve_full", 72), ("slice_comparison.png", 1080, "slice_full", 72)):
        im = Image.open(os.path.join(run, src)).convert("RGB"); im.thumbnail((width, 10000))
        im.save(os.path.join(IMG, f"{key}__{tag}.jpg"), quality=q, optimize=True)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", os.path.join(run, "training_frames_dvr.gif"), "-vf",
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=10", "-c:v", "libx264", "-preset", "slow", "-crf", "30",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", os.path.join(IMG, f"{key}__dvr.mp4")], check=True)
    print("assets", key)
