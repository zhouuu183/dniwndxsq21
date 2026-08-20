"""Build a continuous hair foreground target without trusting low-alpha F."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829, normalized_blur_v829


class ReliableHairForegroundTargetV833(nn.Module):
    def __init__(self, tone_radius: int = 9, local_radius: int = 21,
                 propagation_radius: int = 15, detail_radius: int = 3,
                 detail_gain: float = 0.5, reliable_threshold: float = 0.70,
                 max_l_delta: float = 35.0, max_ab_delta: float = 30.0):
        super().__init__()
        self.tone_radius = int(tone_radius)
        self.local_radius = int(local_radius)
        self.propagation_radius = int(propagation_radius)
        self.detail_radius = int(detail_radius)
        self.detail_gain = min(float(detail_gain), 0.5)
        self.reliable_threshold = float(reliable_threshold)
        self.max_l_delta = float(max_l_delta)
        self.max_ab_delta = float(max_ab_delta)

    @staticmethod
    def _masked_median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        rows = []
        for sample, support in zip(value, mask):
            selected = sample.flatten(1)[:, support.flatten() > 0.5]
            rows.append(selected.median(1).values if selected.numel() else sample.flatten(1).median(1).values)
        return torch.stack(rows)[..., None, None]

    def config_dict(self) -> dict[str, object]:
        return {
            "reliable_threshold": self.reliable_threshold,
            "propagation_radius": self.propagation_radius,
            "detail_gain": self.detail_gain,
            "low_alpha_direct_f_use": False,
            "chromatic_high_frequency": False,
        }

    def forward(self, *, foreground_pp_rgb: torch.Tensor, original_pp_rgb: torch.Tensor,
                v226_rgb: torch.Tensor, foreground_confidence: torch.Tensor,
                return_aux: bool = False):
        size = foreground_pp_rgb.shape[-2:]
        v226 = F.interpolate(v226_rgb, size=size, mode="bicubic", align_corners=False).clamp(0, 1)
        pp = original_pp_rgb.clamp(0, 1)
        foreground_lab = rgb_to_lab(foreground_pp_rgb.clamp(0, 1))
        target_lab = rgb_to_lab(v226)
        reliable = (foreground_confidence >= self.reliable_threshold).to(foreground_lab.dtype)
        residual = target_lab - foreground_lab
        global_delta = self._masked_median(residual, reliable)
        global_target_lab = self._masked_median(target_lab, reliable)
        cap = foreground_lab.new_tensor([self.max_l_delta, self.max_ab_delta, self.max_ab_delta])[None, :, None, None]
        global_delta = global_delta.clamp(-cap, cap)
        low_f, _ = normalized_blur_v829(foreground_lab, reliable, self.tone_radius)
        low_target, _ = normalized_blur_v829(target_lab, reliable, self.tone_radius)
        local_raw = (low_target - low_f) - global_delta
        local_delta, local_support = normalized_blur_v829(local_raw, reliable, self.local_radius)
        local_weight = (local_support / 0.20).clamp(0, 1)
        reliable_delta = (global_delta + local_weight * local_delta).clamp(-cap, cap)
        recolored_reliable = foreground_lab + reliable_delta

        propagated_low, propagation_support = normalized_blur_v829(
            recolored_reliable, reliable * foreground_confidence, self.propagation_radius
        )
        propagated_low = torch.where(
            (propagation_support >= 1e-6).expand_as(propagated_low),
            propagated_low,
            global_target_lab.expand_as(propagated_low),
        )
        pp_lab = rgb_to_lab(pp)
        pp_l_hf = pp_lab[:, :1] - gaussian_blur_v829(pp_lab[:, :1], self.detail_radius)
        propagated = propagated_low.clone()
        propagated[:, :1] = propagated[:, :1] + self.detail_gain * pp_l_hf.clamp(-8.0, 8.0)
        confidence = foreground_confidence.clamp(0, 1)
        final_lab = confidence * recolored_reliable + (1.0 - confidence) * propagated
        final_rgb = lab_to_rgb(final_lab).clamp(0, 1)
        if not torch.isfinite(final_rgb).all():
            raise ValueError("V2.33 foreground target produced NaN or Inf")
        if not return_aux:
            return final_rgb
        return final_rgb, {
            "reliable_foreground_mask": reliable,
            "global_delta_lab": global_delta,
            "global_target_lab": global_target_lab,
            "supported_local_delta_lab": local_weight * local_delta,
            "recolored_reliable_foreground_rgb": lab_to_rgb(recolored_reliable).clamp(0, 1),
            "propagated_target_foreground_rgb": lab_to_rgb(propagated).clamp(0, 1),
            "foreground_target_rgb": final_rgb,
            "propagation_support": propagation_support,
            "foreground_confidence_blend": confidence,
            "cap_hit_fraction": (reliable_delta.abs() >= cap * 0.999).float().mean((1, 2, 3)),
        }


__all__ = ["ReliableHairForegroundTargetV833"]
