"""Foreground-space recolor: target adoption is independent of alpha."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.hybrid_hair_carrier_v829 import normalized_blur_v829


class ForegroundRecolorV832(nn.Module):
    def __init__(self, tone_radius: int = 9, residual_radius: int = 21,
                 max_l_delta: float = 35.0, max_ab_delta: float = 30.0):
        super().__init__()
        self.tone_radius = int(tone_radius)
        self.residual_radius = int(residual_radius)
        self.max_l_delta = float(max_l_delta)
        self.max_ab_delta = float(max_ab_delta)

    def config_dict(self) -> dict[str, object]:
        return {
            "global_target_fallback": True,
            "tone_radius": self.tone_radius,
            "residual_radius": self.residual_radius,
            "alpha_is_color_strength": False,
            "foreground_target": "F_pp + global_delta + supported_local_residual",
        }

    @staticmethod
    def _masked_median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        result = []
        for row, support in zip(value, mask):
            selected = row.flatten(1)[:, support.flatten() > 0.5]
            result.append(torch.median(selected, dim=1).values if selected.numel() else row.flatten(1).median(dim=1).values)
        return torch.stack(result).unsqueeze(-1).unsqueeze(-1)

    def forward(self, *, foreground_pp_rgb: torch.Tensor, v226_rgb: torch.Tensor,
                alpha_hr: torch.Tensor, sure_fg: torch.Tensor, return_aux: bool = False):
        output_size = foreground_pp_rgb.shape[-2:]
        v226 = F.interpolate(v226_rgb, size=output_size, mode="bicubic", align_corners=False).clamp(0, 1)
        pp_lab = rgb_to_lab(foreground_pp_rgb.clamp(0, 1))
        v226_lab = rgb_to_lab(v226)
        dense = ((alpha_hr >= 0.95) & (sure_fg > 0.5)).to(pp_lab.dtype)
        residual = v226_lab - pp_lab
        global_delta = self._masked_median(residual, dense)
        cap = torch.tensor([self.max_l_delta, self.max_ab_delta, self.max_ab_delta], device=pp_lab.device, dtype=pp_lab.dtype).view(1, 3, 1, 1)
        global_delta = global_delta.clamp(-cap, cap)
        low_pp, _ = normalized_blur_v829(pp_lab, dense, self.tone_radius)
        low_v226, _ = normalized_blur_v829(v226_lab, dense, self.tone_radius)
        local_raw = (low_v226 - low_pp) - global_delta
        local_delta, support = normalized_blur_v829(local_raw, dense, self.residual_radius)
        local_delta = torch.where(support >= 1e-4, local_delta, torch.zeros_like(local_delta))
        delta = (global_delta + local_delta).clamp(-cap, cap)
        target_lab = pp_lab + delta
        target_rgb = lab_to_rgb(target_lab).clamp(0, 1)
        if not torch.isfinite(target_rgb).all():
            raise ValueError("V2.32 foreground recolor produced NaN or Inf")
        if not return_aux:
            return target_rgb
        return target_rgb, {
            "dense_foreground": dense,
            "global_delta_lab": global_delta,
            "local_delta_lab": local_delta,
            "delta_lab": delta,
            "local_support": support,
            "foreground_target_rgb": target_rgb,
            "cap_hit_fraction": (delta.abs() >= cap * 0.999).to(delta.dtype).mean(dim=(1, 2, 3)),
        }


__all__ = ["ForegroundRecolorV832"]
