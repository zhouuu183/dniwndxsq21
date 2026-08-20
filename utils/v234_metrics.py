"""Diagnostics for V2.34 carrier preservation and chroma ownership."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


def _match(value: torch.Tensor, size: tuple[int, int], mask: bool = False) -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    return F.interpolate(value.float(), size=size, mode="nearest" if mask else "bilinear",
                         align_corners=None if mask else False)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(value)
    return (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1e-6)


def v234_metric_tensors(*, strong_anchor_rgb: torch.Tensor, color_reference_rgb: torch.Tensor,
                        final_rgb: torch.Tensor, hair_mask: torch.Tensor, face_mask: torch.Tensor,
                        base_rgb: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    size = strong_anchor_rgb.shape[-2:]
    reference = _match(color_reference_rgb, size)
    final = _match(final_rgb, size)
    hair = _match(hair_mask, size, True).clamp(0, 1)
    face = _match(face_mask, size, True).clamp(0, 1)
    anchor_lab, ref_lab, final_lab = rgb_to_lab(strong_anchor_rgb), rgb_to_lab(reference), rgb_to_lab(final)
    hair_ab = torch.linalg.vector_norm(final_lab[:, 1:] - ref_lab[:, 1:], dim=1, keepdim=True)
    anchor_hf = strong_anchor_rgb - gaussian_blur_v829(strong_anchor_rgb, 2)
    final_hf = final - gaussian_blur_v829(final, 2)
    result = {
        "hair_reference_ab_error": masked_mean(hair_ab, hair),
        "anchor_high_frequency_error": masked_mean((final_hf - anchor_hf).abs(), hair),
        "face_rgb_change": masked_mean((final - strong_anchor_rgb).abs(), face),
        "face_injection_max": ((final - strong_anchor_rgb).abs() * face).amax(dim=(1, 2, 3)),
        "outside_hair_change": ((final - strong_anchor_rgb).abs() * (1.0 - hair)).amax(dim=(1, 2, 3)),
        "boundary_chroma_jump": masked_mean(
            (final_lab[:, 1:] - gaussian_blur_v829(final_lab[:, 1:], 2)).abs(),
            hair * (1.0 - F.avg_pool2d(hair, 7, stride=1, padding=3, count_include_pad=False)),
        ),
        "reference_progress_fraction": masked_mean(
            (torch.linalg.vector_norm(final_lab[:, 1:] - ref_lab[:, 1:], dim=1, keepdim=True)
             < torch.linalg.vector_norm(anchor_lab[:, 1:] - ref_lab[:, 1:], dim=1, keepdim=True)).float(), hair
        ),
        "base_leakage_fraction": torch.zeros(
            strong_anchor_rgb.shape[0], device=strong_anchor_rgb.device,
            dtype=strong_anchor_rgb.dtype,
        ),
    }
    if base_rgb is not None:
        base = _match(base_rgb, size)
        base_lab = rgb_to_lab(base)
        base_distance = torch.linalg.vector_norm(final_lab[:, 1:] - base_lab[:, 1:], dim=1, keepdim=True)
        ref_distance = torch.linalg.vector_norm(final_lab[:, 1:] - ref_lab[:, 1:], dim=1, keepdim=True)
        result["base_leakage_ratio"] = masked_mean(base_distance, hair) / masked_mean(ref_distance, hair).clamp_min(1e-6)
        result["base_leakage_fraction"] = masked_mean((base_distance < ref_distance).float(), hair)
    return result


__all__ = ["v234_metric_tensors"]
