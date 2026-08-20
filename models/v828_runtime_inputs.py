"""Shared, deterministic runtime input normalization for Blending V8.28."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask_like(value: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(reference)
    value = value.float()
    if value.dim() == 3:
        value = value.unsqueeze(1)
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value, size=reference.shape[-2:], mode="nearest")
    if value.size(0) == 1 and reference.size(0) != 1:
        value = value.expand(reference.size(0), -1, -1, -1)
    if value.shape != reference.shape:
        raise ValueError(
            f"V2.28 mask shape mismatch: expected {tuple(reference.shape)}, got {tuple(value.shape)}"
        )
    return value.to(device=reference.device, dtype=reference.dtype).clamp(0, 1)


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * width + 1, stride=1, padding=width)


def _erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    return 1.0 - _dilate(1.0 - mask, width)


def build_v828_runtime_inputs(
    *,
    base_rgb: torch.Tensor,
    anchor_rgb: torch.Tensor,
    target_hair_mask: torch.Tensor,
    target_hair_eroded: torch.Tensor | None = None,
    target_hair_dilated: torch.Tensor | None = None,
    face_keep_mask: torch.Tensor | None = None,
    skin_protect_mask: torch.Tensor | None = None,
    hard_protect_mask: torch.Tensor | None = None,
    soft_hair_probability: torch.Tensor | None = None,
    matte_width: int = 4,
) -> dict[str, torch.Tensor]:
    """Build the exact tensor contract shared by diagnostic and real inference.

    V2.27 parity drift came from reconstructing masks independently in its two
    paths. This function is the single normalization point used by both paths.
    """
    if base_rgb.shape != anchor_rgb.shape or base_rgb.dim() != 4 or base_rgb.size(1) != 3:
        raise ValueError("V2.28 base_rgb and anchor_rgb must be matching NCHW RGB tensors")
    base_rgb = base_rgb.float().clamp(0, 1)
    anchor_rgb = anchor_rgb.to(device=base_rgb.device, dtype=base_rgb.dtype).clamp(0, 1)

    hair = target_hair_mask.float()
    if hair.dim() == 3:
        hair = hair.unsqueeze(1)
    if hair.shape[-2:] != base_rgb.shape[-2:]:
        hair = F.interpolate(hair, size=base_rgb.shape[-2:], mode="nearest")
    hair = hair.to(device=base_rgb.device, dtype=base_rgb.dtype).clamp(0, 1)
    # Geometry is intentionally binary. Fractional coverage exists only inside
    # the narrow trimap produced by HairCoverageMatteV828.
    hair = (hair >= 0.5).to(base_rgb.dtype)

    eroded = _mask_like(target_hair_eroded, hair) if target_hair_eroded is not None else _erode(hair, matte_width)
    dilated = _mask_like(target_hair_dilated, hair) if target_hair_dilated is not None else _dilate(hair, matte_width)
    eroded = ((eroded >= 0.5) & (hair > 0.5)).to(base_rgb.dtype)
    dilated = ((dilated >= 0.5) | (hair > 0.5)).to(base_rgb.dtype)

    soft = None
    if soft_hair_probability is not None:
        soft = _mask_like(soft_hair_probability, hair) * dilated

    face = _mask_like(face_keep_mask, hair)
    skin = _mask_like(skin_protect_mask, hair)
    hard = _mask_like(hard_protect_mask, hair)
    return {
        "base_rgb": base_rgb,
        "anchor_rgb": anchor_rgb,
        "target_hair_mask": hair,
        "target_hair_eroded": eroded,
        "target_hair_dilated": dilated,
        "face_keep_mask": face,
        "skin_protect_mask": skin,
        "hard_protect_mask": hard,
        "soft_hair_probability": soft,
    }


__all__ = ["build_v828_runtime_inputs"]
