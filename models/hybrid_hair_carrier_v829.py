"""Hybrid low-frequency color and high-frequency appearance carrier for V2.29."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def gaussian_blur_v829(image: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return image
    sigma = max(float(radius) / 3.0, 0.5)
    coordinate = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-coordinate.square() / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    channels = image.size(1)
    kernel_x = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    value = F.pad(image, (radius, radius, 0, 0), mode="replicate")
    value = F.conv2d(value, kernel_x, groups=channels)
    value = F.pad(value, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(value, kernel_y, groups=channels)


def normalized_blur_v829(
    value: torch.Tensor,
    support: torch.Tensor,
    radius: int,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    support = support.to(device=value.device, dtype=value.dtype).clamp(0, 1)
    numerator = gaussian_blur_v829(value * support, radius)
    denominator = gaussian_blur_v829(support, radius)
    return numerator / denominator.clamp_min(eps), denominator


class HybridHairCarrierV829(nn.Module):
    """Use V2.26 for hair tone and Strong Anchor for local appearance."""

    def __init__(self, low_radius: int = 5, anchor_hf_gain: float = 1.0):
        super().__init__()
        self.low_radius = int(low_radius)
        self.anchor_hf_gain = float(anchor_hf_gain)
        if self.low_radius < 1:
            raise ValueError("V2.29 carrier low_radius must be positive")

    def config_dict(self) -> dict[str, float | int | str]:
        return {
            "low_radius": self.low_radius,
            "anchor_hf_gain": self.anchor_hf_gain,
            "low_frequency_owner": "V2.26_CORE",
            "high_frequency_owner": "STRONG_ANCHOR",
        }

    def forward(
        self,
        *,
        anchor_rgb: torch.Tensor,
        v226_rgb: torch.Tensor,
        base_rgb: torch.Tensor,
        hair_core: torch.Tensor,
        return_aux: bool = False,
    ):
        del base_rgb
        low_v226, low_support = normalized_blur_v829(
            v226_rgb, hair_core, self.low_radius
        )
        anchor_low = gaussian_blur_v829(anchor_rgb, self.low_radius)
        anchor_hf = anchor_rgb - anchor_low
        raw_hybrid = low_v226 + self.anchor_hf_gain * anchor_hf
        hybrid = raw_hybrid.clamp(0, 1)
        tone_delta_core = low_v226 - anchor_low
        clip_low = (-raw_hybrid).clamp_min(0)
        clip_high = (raw_hybrid - 1.0).clamp_min(0)
        if not torch.isfinite(hybrid).all():
            raise ValueError("V2.29 hybrid carrier produced NaN or Inf")
        if not return_aux:
            return hybrid
        aux = {
            "low_v226_core": low_v226,
            "low_v226_support": low_support,
            "anchor_low": anchor_low,
            "anchor_hf": anchor_hf,
            "tone_delta_core": tone_delta_core,
            "hybrid_raw": raw_hybrid,
            "clip_low_fraction": (clip_low > 0).float().mean(dim=(1, 2, 3)),
            "clip_high_fraction": (clip_high > 0).float().mean(dim=(1, 2, 3)),
            "clip_magnitude": (clip_low + clip_high).mean(dim=(1, 2, 3)),
        }
        return hybrid, aux


__all__ = [
    "HybridHairCarrierV829",
    "gaussian_blur_v829",
    "normalized_blur_v829",
]
