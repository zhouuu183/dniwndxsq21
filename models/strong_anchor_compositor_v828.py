"""Strong-Anchor appearance compositor and strict PostProcess hair lock."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from models.hair_coverage_matte_v828 import HairCoverageMatteV828


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    return F.max_pool2d(mask, kernel_size=2 * width + 1, stride=1, padding=width)


def _gaussian_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return image
    sigma = max(float(radius) / 3.0, 0.5)
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(x.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    channels = image.size(1)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    image = F.pad(image, (radius, radius, 0, 0), mode="replicate")
    image = F.conv2d(image, horizontal, groups=channels)
    image = F.pad(image, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(image, vertical, groups=channels)


def apply_pp_hair_lock_v828(
    prepp_rgb: torch.Tensor,
    pp_rgb_original: torch.Tensor,
    hair_alpha: torch.Tensor,
    pp_unlock_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve V2.28 hair while leaving non-hair PostProcess untouched."""
    lock = hair_alpha.to(prepp_rgb.dtype).clamp(0, 1)
    if pp_unlock_mask is not None:
        unlock = pp_unlock_mask.to(device=prepp_rgb.device, dtype=prepp_rgb.dtype)
        if unlock.dim() == 3:
            unlock = unlock.unsqueeze(1)
        if unlock.shape[-2:] != lock.shape[-2:]:
            unlock = F.interpolate(unlock, size=lock.shape[-2:], mode="nearest")
        unlock = unlock.clamp(0, 1)
        lock = lock * (1.0 - unlock)
    final = pp_rgb_original + lock * (prepp_rgb - pp_rgb_original)
    return final.clamp(0, 1), lock


class StrongAnchorAppearanceCompositorV828(nn.Module):
    """Composite Strong Anchor hair over Base exactly once."""

    def __init__(
        self,
        matte_max_distance: int = 8,
        bg_residual_radius: int = 7,
        bg_residual_strength: float = 1.0,
        outer_ring_width: int = 6,
        background_valid_threshold: float = 1e-3,
    ):
        super().__init__()
        self.matte = HairCoverageMatteV828(max_distance=matte_max_distance)
        self.bg_residual_radius = int(bg_residual_radius)
        self.bg_residual_strength = float(bg_residual_strength)
        self.outer_ring_width = int(outer_ring_width)
        self.background_valid_threshold = float(background_valid_threshold)

    def config_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "matte_max_distance": self.matte.max_distance,
            "bg_residual_radius": self.bg_residual_radius,
            "bg_residual_strength": self.bg_residual_strength,
            "outer_ring_width": self.outer_ring_width,
            "coverage_alpha_multiply_count": 1,
            "rgb_carrier": "STRONG_ANCHOR",
            "pp_hair_lock": "STRICT",
        }

    def forward(
        self,
        *,
        base_rgb: torch.Tensor,
        anchor_rgb: torch.Tensor,
        target_hair_mask: torch.Tensor,
        target_hair_eroded: torch.Tensor,
        target_hair_dilated: torch.Tensor,
        face_keep_mask: torch.Tensor,
        skin_protect_mask: torch.Tensor,
        hard_protect_mask: torch.Tensor,
        soft_hair_probability: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        alpha, core, unknown, outside, matte_aux = self.matte(
            target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded,
            target_hair_dilated=target_hair_dilated,
            base_rgb=base_rgb,
            anchor_rgb=anchor_rgb,
            soft_hair_probability=soft_hair_probability,
            skin_protect_mask=skin_protect_mask,
        )
        hard = (hard_protect_mask >= 0.5).to(base_rgb.dtype)
        protected = torch.maximum(hard, torch.maximum(face_keep_mask, skin_protect_mask)).clamp(0, 1)

        residual = anchor_rgb - base_rgb
        wide = _dilate(target_hair_mask, self.outer_ring_width)
        outer_ring = (wide - target_hair_mask).clamp(0, 1) * (1.0 - protected)
        numerator = _gaussian_blur(residual * outer_ring, self.bg_residual_radius)
        denominator = _gaussian_blur(outer_ring, self.bg_residual_radius)
        valid = (denominator >= self.background_valid_threshold).to(base_rgb.dtype) * unknown
        background_residual = (numerator / denominator.clamp_min(1e-6)) * valid
        clean_residual = residual - self.bg_residual_strength * background_residual
        edge_candidate = (base_rgb + clean_residual).clamp(0, 1)

        # This is the only fractional coverage multiplication in pre-PP.
        edge_composite = base_rgb + alpha * (edge_candidate - base_rgb)
        prepp = core * anchor_rgb + unknown * edge_composite + outside * base_rgb
        effective_hard = (hard > 0.5).expand_as(prepp)
        prepp = torch.where(effective_hard, base_rgb, prepp).clamp(0, 1)

        effective_core = core * (1.0 - hard)
        effective_outside = torch.maximum(outside, hard).clamp(0, 1)
        outside_delta = ((prepp - base_rgb).abs() * effective_outside).amax(dim=(1, 2, 3))
        hard_delta = ((prepp - base_rgb).abs() * hard).amax(dim=(1, 2, 3))
        core_delta = ((prepp - anchor_rgb).abs() * effective_core).amax(dim=(1, 2, 3))
        if not torch.isfinite(prepp).all():
            raise ValueError("V2.28 compositor produced NaN or Inf")
        if not return_aux:
            return prepp
        aux = {
            "hair_alpha": alpha,
            "sure_fg": effective_core,
            "sure_bg": outside,
            "unknown_band": unknown,
            "outside": effective_outside,
            "anchor_base_residual": residual,
            "background_residual_estimate": background_residual,
            "clean_boundary_residual": clean_residual,
            "background_est_valid": valid,
            "background_sample_ring": outer_ring,
            "edge_candidate": edge_candidate,
            "prepp_rgb": prepp,
            "hard_protect": hard,
            "outside_max_delta": outside_delta,
            "hard_protect_max_delta": hard_delta,
            "core_anchor_max_delta": core_delta,
            **matte_aux,
        }
        return prepp, aux


__all__ = ["StrongAnchorAppearanceCompositorV828", "apply_pp_hair_lock_v828"]
