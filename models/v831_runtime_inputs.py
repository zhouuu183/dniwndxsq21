"""Shared V2.31 diagnostic and real-inference tensor contract."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.v830_runtime_inputs import build_parser_regions_v830


def _mask(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.dim() == 2:
        value = value[None, None]
    elif value.dim() == 3:
        value = value[:, None]
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value.float(), size=reference.shape[-2:], mode="nearest")
    return value.to(device=reference.device, dtype=reference.dtype).clamp(0, 1)


def build_v831_runtime_inputs(
    *,
    pp_original_rgb: torch.Tensor,
    base_rgb: torch.Tensor,
    v226_rgb: torch.Tensor,
    target_hair_mask: torch.Tensor,
    parser_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if pp_original_rgb.dim() != 4 or pp_original_rgb.size(1) != 3:
        raise ValueError("V2.31 Original PP must be BCHW RGB")
    for name, image in {"base_rgb": base_rgb, "v226_rgb": v226_rgb}.items():
        if image.dim() != 4 or image.size(1) != 3 or image.size(0) != pp_original_rgb.size(0):
            raise ValueError(f"V2.31 {name} must be batch RGB")
    hair = _mask(target_hair_mask, base_rgb[:, :1])
    regions = build_parser_regions_v830(parser_labels, hair)
    return {
        "pp_original_rgb": pp_original_rgb.clamp(0, 1),
        "base_rgb": base_rgb.to(pp_original_rgb).clamp(0, 1),
        "v226_rgb": v226_rgb.to(pp_original_rgb).clamp(0, 1),
        "target_hair_mask": hair.to(pp_original_rgb),
        **{key: value.to(pp_original_rgb) for key, value in regions.items()},
    }


__all__ = ["build_v831_runtime_inputs"]
