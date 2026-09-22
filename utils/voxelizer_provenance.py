"""Record which CUDA voxelizer a run actually loaded.

Written at the start of every training run as <output>/voxelizer.json so a
finished run can be audited afterwards. Motivation: ESVR-4dgs/.venv was
copied from ESVR/ and its activate script pointed at ESVR/.venv until
2026-09-20, so runs launched via `source .venv/bin/activate` silently used
the original folder's unfixed position-gradient kernel; nothing in their
outputs recorded it (see output/scaffold4d/ablation/RESULTS.md and
output/experiments/static_ceiling/RESULTS.md).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

EXPECTED_ROOT = Path(__file__).resolve().parents[1]


def describe():
    import torch
    import gaussian_rasterization_voxelization as voxelizer
    package_dir = Path(voxelizer.__file__).resolve().parent
    so = sorted(package_dir.glob("_C*.so"))
    info = dict(
        voxelizer_package=str(package_dir),
        voxelizer_so=str(so[0]) if so else None,
        voxelizer_so_sha256=hashlib.sha256(so[0].read_bytes()).hexdigest() if so else None,
        voxelizer_so_mtime=os.path.getmtime(so[0]) if so else None,
        inside_this_repo=str(package_dir).startswith(str(EXPECTED_ROOT)),
        python_executable=sys.executable,
        sys_prefix=sys.prefix,
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
    )
    return info


def record(output_dir, require_repo_kernel=True):
    info = describe()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "voxelizer.json").write_text(json.dumps(info, indent=2))
    if require_repo_kernel and not info["inside_this_repo"]:
        raise RuntimeError(
            f"Loaded voxelizer {info['voxelizer_package']} is outside this repository "
            f"({EXPECTED_ROOT}); launch with .venv/bin/python, not a foreign activate script")
    return info
