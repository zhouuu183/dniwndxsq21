"""Original-PP unified low-frequency Lab recoloring for V2.31."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.hybrid_hair_carrier_v829 import normalized_blur_v829


class PPUnifiedRecolorV831(nn.Module):
    def __init__(
        self,
        tone_radius: int = 9,
        tone_propagation_radius: int = 15,
        max_l_delta: float = 35.0,
        max_ab_delta: float = 30.0,
    ):
        super().__init__()
        self.tone_radius = int(tone_radius)
        self.tone_propagation_radius = int(tone_propagation_radius)
        self.max_l_delta = float(max_l_delta)
        self.max_ab_delta = float(max_ab_delta)

    def config_dict(self) -> dict[str, object]:
        return {
            "carrier": "ORIGINAL_PP",
            "low_frequency_target": "V2.26",
            "tone_radius": self.tone_radius,
            "tone_propagation_radius": self.tone_propagation_radius,
            "max_l_delta": self.max_l_delta,
            "max_ab_delta": self.max_ab_delta,
            "binary_final_alpha": False,
            "strong_anchor_final_injection": False,
        }

    def forward(
        self,
        *,
        pp_rgb: torch.Tensor,
        v226_rgb: torch.Tensor,
        alpha_hr: torch.Tensor,
        sure_fg: torch.Tensor,
        return_aux: bool = False,
    ):
        output_size = pp_rgb.shape[-2:]
        v226 = F.interpolate(
            v226_rgb, size=output_size, mode="bicubic", align_corners=False
        ).clamp(0, 1)
        alpha = alpha_hr.to(device=pp_rgb.device, dtype=pp_rgb.dtype).clamp(0, 1)
        sure_fg = sure_fg.to(alpha)
        dense_core = ((alpha >= 0.98) & (sure_fg > 0.5)).to(alpha.dtype)
        pp_lab = rgb_to_lab(pp_rgb.clamp(0, 1))
        v226_lab = rgb_to_lab(v226)
        low_pp, pp_support = normalized_blur_v829(pp_lab, dense_core, self.tone_radius)
        low_v226, v226_support = normalized_blur_v829(v226_lab, dense_core, self.tone_radius)
        delta_core_raw = low_v226 - low_pp
        cap = torch.tensor(
            [self.max_l_delta, self.max_ab_delta, self.max_ab_delta],
            device=pp_rgb.device,
            dtype=pp_rgb.dtype,
        ).view(1, 3, 1, 1)
        delta_core = torch.maximum(torch.minimum(delta_core_raw, cap), -cap)
        delta_lab, propagation_support = normalized_blur_v829(
            delta_core, dense_core, self.tone_propagation_radius
        )
        valid = (propagation_support >= 1e-4).to(alpha.dtype)
        applied_delta = alpha * valid * delta_lab
        phase_b_lab = pp_lab + applied_delta
        phase_b = lab_to_rgb(phase_b_lab).clamp(0, 1)
        if not torch.isfinite(phase_b).all():
            raise ValueError("V2.31 unified PP recolor produced NaN or Inf")
        if not return_aux:
            return phase_b
        cap_hit = (delta_core_raw.abs() > cap).to(pp_rgb.dtype)
        return phase_b, {
            "dense_core": dense_core,
            "pp_lab": pp_lab,
            "v226_lab": v226_lab,
            "low_pp_lab": low_pp,
            "low_v226_lab": low_v226,
            "low_pp_support": pp_support,
            "low_v226_support": v226_support,
            "delta_lab_core_raw": delta_core_raw,
            "delta_lab": delta_lab,
            "applied_tone_delta_lab": applied_delta,
            "tone_support": valid,
            "cap_hit_fraction": cap_hit.mean(dim=(1, 2, 3)),
            "phase_b_rgb": phase_b,
        }


__all__ = ["PPUnifiedRecolorV831"]
