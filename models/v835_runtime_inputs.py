"""V2.35 hair-only chroma runtime contract."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.dim() == 2:
        value = value[None, None]
    elif value.dim() == 3:
        value = value[:, None]
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value.float(), size=reference.shape[-2:], mode="nearest")
    return value.to(reference).clamp(0, 1)


def build_v835_runtime_inputs(
    *, strong_anchor_rgb: torch.Tensor, color_reference_rgb: torch.Tensor,
    target_hair_mask: torch.Tensor, face_mask: torch.Tensor,
    anchor_hair_mask: torch.Tensor | None = None,
    reference_hair_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if strong_anchor_rgb.dim() != 4 or strong_anchor_rgb.size(1) != 3:
        raise ValueError("V2.35 Strong Anchor must be BCHW RGB")
    if color_reference_rgb.dim() != 4 or color_reference_rgb.size(1) != 3:
        raise ValueError("V2.35 color reference must be BCHW RGB")
    if color_reference_rgb.size(0) != strong_anchor_rgb.size(0):
        raise ValueError("V2.35 color reference batch must match Strong Anchor batch")
    reference = strong_anchor_rgb[:, :1]
    runtime = {
        "strong_anchor_rgb": strong_anchor_rgb.float().clamp(0, 1),
        "color_reference_rgb": color_reference_rgb.float().clamp(0, 1),
        "target_hair_mask": _mask(target_hair_mask, reference),
        "face_mask": _mask(face_mask, reference),
    }
    runtime["hair_mask"] = runtime["target_hair_mask"]
    if anchor_hair_mask is not None:
        runtime["anchor_hair_mask"] = _mask(anchor_hair_mask, reference)
    if reference_hair_mask is not None:
        runtime["reference_hair_mask"] = _mask(reference_hair_mask, reference)
    return runtime


__all__ = ["build_v835_runtime_inputs"]
