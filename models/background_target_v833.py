"""Alpha-consistent, continuous transition background target for V2.33."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.hybrid_hair_carrier_v829 import normalized_blur_v829


class BackgroundTargetV833(nn.Module):
    def __init__(self, radius: int = 9, support_full: float = 0.20):
        super().__init__()
        self.radius = int(radius)
        self.support_full = float(support_full)

    def config_dict(self) -> dict[str, object]:
        return {
            "ownership_alpha": "ALPHA_EFF", "continuous_transition": True,
            "continuous_support": True, "same_pixel_base_first": True,
            "radius": self.radius, "support_full": self.support_full,
        }

    def forward(self, *, background_pp_rgb: torch.Tensor, base_rgb: torch.Tensor,
                alpha_eff: torch.Tensor, background_confidence: torch.Tensor,
                sure_fg: torch.Tensor, sure_bg: torch.Tensor, unknown: torch.Tensor,
                source_face_mask: torch.Tensor, source_subject_mask: torch.Tensor,
                return_aux: bool = False):
        size = alpha_eff.shape[-2:]
        base = F.interpolate(base_rgb, size=size, mode="bicubic", align_corners=False).clamp(0, 1)
        face = F.interpolate(source_face_mask, size=size, mode="nearest").clamp(0, 1)
        subject = F.interpolate(source_subject_mask, size=size, mode="nearest").clamp(0, 1)
        fractional = ((alpha_eff > 0.01) & (alpha_eff < 0.99)).to(alpha_eff.dtype)
        transition_strength = 4.0 * alpha_eff * (1.0 - alpha_eff) * unknown.clamp(0, 1)

        same_pixel_region = ((face > 0.5) | (subject < 0.5)).to(alpha_eff.dtype) * (1.0 - sure_fg)
        same_pixel_valid = same_pixel_region * background_confidence.clamp(0, 1)
        reliable_nonhair = background_confidence * (1.0 - sure_fg) * torch.maximum(sure_bg, same_pixel_valid)
        propagated, local_support = normalized_blur_v829(base, reliable_nonhair, self.radius)
        local_support_conf = (local_support / max(self.support_full, 1e-6)).clamp(0, 1)
        base_candidate = same_pixel_valid * base + (1.0 - same_pixel_valid) * propagated
        support_conf = same_pixel_valid + (1.0 - same_pixel_valid) * local_support_conf
        context_conf = (transition_strength * support_conf).clamp(0, 1)
        target = background_pp_rgb + context_conf * (base_candidate - background_pp_rgb)
        target = torch.where((1.0 - fractional).expand_as(target) > 0.5, background_pp_rgb, target).clamp(0, 1)
        if not return_aux:
            return target
        return target, {
            "fractional_alpha_eff": fractional,
            "transition_strength": transition_strength,
            "same_pixel_context_valid": same_pixel_valid,
            "propagated_base_context_rgb": propagated,
            "local_context_support": local_support,
            "support_confidence": support_conf,
            "context_confidence": context_conf,
            "background_target_rgb": target,
            "recomposition_support": torch.maximum(fractional, unknown * (alpha_eff > 0.01).to(alpha_eff.dtype)),
            "far_outside_max_delta": ((target - background_pp_rgb).abs() * (1.0 - fractional)).flatten(1).amax(1),
        }


__all__ = ["BackgroundTargetV833"]
