"""Shared V2.32 F/B/recolor/recompose input contract."""

from __future__ import annotations

import torch

from models.v831_runtime_inputs import build_v831_runtime_inputs


def build_v832_runtime_inputs(**kwargs) -> dict[str, torch.Tensor]:
    runtime = build_v831_runtime_inputs(**kwargs)
    required = ("pp_original_rgb", "base_rgb", "v226_rgb", "target_hair_mask", "source_face_mask", "source_subject_mask")
    missing = [key for key in required if key not in runtime]
    if missing:
        raise ValueError(f"V2.32 runtime missing: {missing}")
    return runtime


__all__ = ["build_v832_runtime_inputs"]
