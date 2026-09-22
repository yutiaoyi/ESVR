"""Compact storage for a trained ScaffoldESVR: only children the voxelizer
can draw are kept (a per-child bit mask makes the generated children of a
kept anchor match the original model exactly), anchors without children are
dropped, anchor tensors may be fp16. The file carries the data manifest and
the training arguments so it can be evaluated or queried on its own.
"""
import json

import numpy as np
import torch
from torch import nn

from scaffold4d.model import ScaffoldESVR

FORMAT = "scaffold4d-compact-1"
ANCHOR_TENSORS = ("anchor", "anchor_feat", "log_offset_scale")


class CompactScaffold(ScaffoldESVR):
    """ScaffoldESVR with a keep-mask over children: masked children are
    excluded from every query, exactly as if the voxelizer had skipped them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("child_keep", torch.ones(self.config["anchors"], self.config["children"], dtype=torch.bool))

    def at(self, timestamp, generated=None):
        g = generated if generated is not None else self.generate()
        sliced = super().at(timestamp, g)
        keep = self.child_keep.reshape(-1)[sliced.child_index]
        for name in ("get_xyz", "get_scaling", "get_rotation", "get_opacity", "opacity_t", "sigma", "child_index"):
            setattr(sliced, name, getattr(sliced, name)[keep])
        return sliced


@torch.no_grad()
def build_compact(model, keep, fp16=False):
    """keep: bool (anchors, children). Anchors with no kept child are dropped."""
    anchor_keep = keep.any(1)
    compact = CompactScaffold(**{**model.config, "anchors": int(anchor_keep.sum())}).to(model.anchor.device)
    for name in ANCHOR_TENSORS + ("offset",):
        getattr(compact, name).copy_(getattr(model, name)[anchor_keep])
    compact.heads.load_state_dict(model.heads.state_dict())
    compact.child_keep.copy_(keep[anchor_keep])
    if fp16:
        # Round-trip through fp16 so evaluation sees exactly what is stored.
        for name in ANCHOR_TENSORS + ("offset",):
            t = getattr(compact, name)
            t.copy_(t.half().float())
    return compact


def stored_bytes(compact, fp16):
    n, k = compact.config["anchors"], compact.config["children"]
    dtype = 2 if fp16 else 4
    kept_children = int(compact.child_keep.sum())
    heads = sum(t.numel() * t.element_size() for t in compact.heads.state_dict().values())
    return n * (4 + compact.config["feat_dim"] + 7) * dtype + kept_children * 4 * dtype + (n * k + 7) // 8 + heads


@torch.no_grad()
def save_compact(compact, path, manifest, source_args, fp16=False):
    cast = (lambda t: t.detach().half().cpu()) if fp16 else (lambda t: t.detach().cpu())
    payload = dict(format=FORMAT, fp16=fp16, model_config=dict(compact.config), manifest=manifest, source_args=source_args,
                   heads=compact.heads.state_dict(),
                   child_keep=np.packbits(compact.child_keep.cpu().numpy().reshape(-1)),
                   offset_kept=cast(compact.offset[compact.child_keep]))
    for name in ANCHOR_TENSORS:
        payload[name] = cast(getattr(compact, name))
    torch.save(payload, path)
    return payload


def is_compact(payload):
    return isinstance(payload, dict) and payload.get("format") == FORMAT


@torch.no_grad()
def load_compact(payload, device="cuda"):
    cfg = payload["model_config"]
    n, k = cfg["anchors"], cfg["children"]
    compact = CompactScaffold(**cfg).to(device)
    keep = torch.from_numpy(np.unpackbits(payload["child_keep"])[: n * k].astype(bool)).reshape(n, k).to(device)
    compact.child_keep.copy_(keep)
    for name in ANCHOR_TENSORS:
        getattr(compact, name).copy_(payload[name].to(device).float())
    offset = torch.zeros(n, k, 4, device=device)
    offset[keep] = payload["offset_kept"].to(device).float()
    compact.offset.copy_(offset)
    compact.heads.load_state_dict(payload["heads"])
    compact.eval()
    return compact
