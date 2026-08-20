"""V2.30 hair core with V2.26 color and achromatic Anchor detail."""

from __future__ import annotations

import torch
from torch import nn

from models.hybrid_hair_carrier_v829 import gaussian_blur_v829, normalized_blur_v829


class AchromaticHairCarrierV830(nn.Module):
    """Apply one shared luminance-detail gain to all target-color channels."""

    def __init__(
        self,
        low_radius: int = 5,
        detail_radius: int = 5,
        detail_log_cap: float = 0.25,
        detail_gain: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.low_radius = int(low_radius)
        self.detail_radius = int(detail_radius)
        self.detail_log_cap = float(detail_log_cap)
        self.detail_gain = float(detail_gain)
        self.eps = float(eps)

    def config_dict(self) -> dict[str, object]:
        return {
            "low_radius": self.low_radius,
            "detail_radius": self.detail_radius,
            "detail_log_cap": self.detail_log_cap,
            "detail_gain": self.detail_gain,
            "low_frequency_owner": "V2.26_CORE",
            "high_frequency_owner": "STRONG_ANCHOR_LUMINANCE_ONLY",
            "rgb_high_frequency_injection": False,
        }

    def forward(
        self,
        *,
        anchor_rgb: torch.Tensor,
        v226_rgb: torch.Tensor,
        repaired_hair_core: torch.Tensor,
        return_aux: bool = False,
    ):
        low_v226, support = normalized_blur_v829(
            v226_rgb, repaired_hair_core, self.low_radius
        )
        luminance = (
            0.2126 * anchor_rgb[:, 0:1]
            + 0.7152 * anchor_rgb[:, 1:2]
            + 0.0722 * anchor_rgb[:, 2:3]
        )
        luminance_low = gaussian_blur_v829(luminance, self.detail_radius)
        log_detail = (
            torch.log(luminance.clamp_min(0) + self.eps)
            - torch.log(luminance_low.clamp_min(0) + self.eps)
        ).clamp(-self.detail_log_cap, self.detail_log_cap)
        shared_gain = torch.exp(self.detail_gain * log_detail)
        raw_core = low_v226 * shared_gain
        core_rgb = raw_core.clamp(0, 1)
        if not torch.isfinite(core_rgb).all():
            raise ValueError("V2.30 achromatic carrier produced NaN or Inf")
        if not return_aux:
            return core_rgb
        clip = (-raw_core).clamp_min(0) + (raw_core - 1.0).clamp_min(0)
        return core_rgb, {
            "low_v226_core": low_v226,
            "low_v226_support": support,
            "anchor_luminance": luminance,
            "anchor_luminance_low": luminance_low,
            "anchor_log_luminance_detail": log_detail,
            "achromatic_detail_gain": shared_gain,
            "core_raw": raw_core,
            "clip_fraction": (clip > 0).float().mean(dim=(1, 2, 3)),
            "clip_magnitude": clip.mean(dim=(1, 2, 3)),
        }


__all__ = ["AchromaticHairCarrierV830"]
