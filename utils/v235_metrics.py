"""V2.35 hair chroma purity, leakage, face, and texture diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829


def _match(value: torch.Tensor, size: tuple[int, int], *, mask: bool = False) -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    return F.interpolate(value.float(), size=size, mode="nearest" if mask else "bilinear", align_corners=None if mask else False)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(value)
    return (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1e-6)


def v235_metric_tensors(
    *, strong_anchor_rgb: torch.Tensor, color_reference_rgb: torch.Tensor,
    final_rgb: torch.Tensor, target_hair_mask: torch.Tensor, face_mask: torch.Tensor,
    base_rgb: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    size = strong_anchor_rgb.shape[-2:]
    reference = _match(color_reference_rgb, size)
    final = _match(final_rgb, size)
    hair = _match(target_hair_mask, size, mask=True).clamp(0, 1)
    face = _match(face_mask, size, mask=True).clamp(0, 1)
    anchor_lab, reference_lab, final_lab = rgb_to_lab(strong_anchor_rgb), rgb_to_lab(reference), rgb_to_lab(final)
    anchor_hf = strong_anchor_rgb - gaussian_blur_v829(strong_anchor_rgb, 2)
    final_hf = final - gaussian_blur_v829(final, 2)
    ref_ab_error = torch.linalg.vector_norm(final_lab[:, 1:] - reference_lab[:, 1:], dim=1, keepdim=True)
    result = {
        "hair_chroma_ab_error": masked_mean(ref_ab_error, hair),
        "hair_reference_progress": masked_mean(
            (ref_ab_error < torch.linalg.vector_norm(anchor_lab[:, 1:] - reference_lab[:, 1:], dim=1, keepdim=True)).float(), hair
        ),
        "texture_preservation_error": masked_mean((final_hf - anchor_hf).abs(), hair),
        "face_rgb_change": masked_mean((final - strong_anchor_rgb).abs(), face),
        "face_rgb_change_max": ((final - strong_anchor_rgb).abs() * face).amax(dim=(1, 2, 3)),
        "non_hair_leakage_max": ((final - strong_anchor_rgb).abs() * (1.0 - hair)).amax(dim=(1, 2, 3)),
        "non_hair_leakage_mean": masked_mean((final - strong_anchor_rgb).abs(), 1.0 - hair),
    }
    boundary = hair * (1.0 - F.avg_pool2d(hair, 7, stride=1, padding=3, count_include_pad=False))
    result["boundary_chroma_error"] = masked_mean(
        (final_lab[:, 1:] - gaussian_blur_v829(final_lab[:, 1:], 2)).abs(), boundary
    )
    if base_rgb is not None:
        base = _match(base_rgb, size)
        base_lab = rgb_to_lab(base)
        result["base_leakage_fraction"] = masked_mean(
            (torch.linalg.vector_norm(final_lab[:, 1:] - base_lab[:, 1:], dim=1, keepdim=True)
             < ref_ab_error).float(), hair
        )
    return result


def hair_chroma_consistency_loss(final_rgb: torch.Tensor, reference_rgb: torch.Tensor, hair_mask: torch.Tensor) -> torch.Tensor:
    final_lab, reference_lab = rgb_to_lab(final_rgb), rgb_to_lab(reference_rgb)
    return masked_mean((final_lab[:, 1:] - reference_lab[:, 1:]).abs(), hair_mask).mean()


def face_preservation_loss(final_rgb: torch.Tensor, target_rgb: torch.Tensor, face_mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((final_rgb - target_rgb).abs(), face_mask).mean()


def leakage_penalty(final_rgb: torch.Tensor, target_rgb: torch.Tensor, hair_mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((final_rgb - target_rgb).abs(), 1.0 - hair_mask).mean()


def texture_preservation_loss(final_rgb: torch.Tensor, target_rgb: torch.Tensor, hair_mask: torch.Tensor) -> torch.Tensor:
    final_hf = final_rgb - gaussian_blur_v829(final_rgb, 2)
    target_hf = target_rgb - gaussian_blur_v829(target_rgb, 2)
    return masked_mean((final_hf - target_hf).abs(), hair_mask).mean()


def v235_training_objective(
    *, final_rgb: torch.Tensor, reference_rgb: torch.Tensor,
    target_rgb: torch.Tensor, hair_mask: torch.Tensor, face_mask: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the four V2.35 objectives for future trainable chroma encoders."""
    weights = {
        "chroma": 1.0, "face": 1.0, "leakage": 1.0, "texture": 1.0,
        **(weights or {}),
    }
    terms = {
        "chroma": hair_chroma_consistency_loss(final_rgb, reference_rgb, hair_mask),
        "face": face_preservation_loss(final_rgb, target_rgb, face_mask),
        "leakage": leakage_penalty(final_rgb, target_rgb, hair_mask),
        "texture": texture_preservation_loss(final_rgb, target_rgb, hair_mask),
    }
    terms["total"] = sum((float(weights[key]) * terms[key] for key in terms), terms["chroma"].new_zeros(()))
    return terms


__all__ = [
    "v235_metric_tensors", "hair_chroma_consistency_loss", "face_preservation_loss",
    "leakage_penalty", "texture_preservation_loss", "v235_training_objective",
]
