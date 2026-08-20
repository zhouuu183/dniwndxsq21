"""Exact alpha-effective foreground/background recomposition for V2.32."""

from __future__ import annotations

import torch
from torch import nn


class MattingRecomposerV832(nn.Module):
    def __init__(self):
        super().__init__()

    def config_dict(self) -> dict[str, object]:
        return {"equation": "alpha_eff*F_target+(1-alpha_eff)*B_target", "binary_final_alpha": False}

    def forward(self, *, alpha_eff: torch.Tensor, foreground_target_rgb: torch.Tensor,
                background_target_rgb: torch.Tensor, pp_original_rgb: torch.Tensor,
                transition_support: torch.Tensor, sure_fg: torch.Tensor,
                sure_bg: torch.Tensor, return_aux: bool = False):
        final = alpha_eff * foreground_target_rgb + (1.0 - alpha_eff) * background_target_rgb
        equation = final.clone()
        far = (1.0 - transition_support) * (1.0 - sure_fg) * (1.0 - sure_bg)
        final = torch.where(far.expand_as(final) > 0.5, pp_original_rgb, final).clamp(0, 1)
        if not torch.isfinite(final).all():
            raise ValueError("V2.32 recomposer produced NaN or Inf")
        if not return_aux:
            return final
        return final, {
            "final_rgb": final,
            "recomposition_equation_error": (equation - (alpha_eff * foreground_target_rgb + (1 - alpha_eff) * background_target_rgb)).abs().flatten(1).amax(dim=1),
            "recomposition_support": (1.0 - far),
            "far_outside_max_delta": ((final - pp_original_rgb).abs() * far).flatten(1).amax(dim=1),
            "sure_fg_final_error": ((final - foreground_target_rgb).abs() * sure_fg).flatten(1).amax(dim=1),
        }


__all__ = ["MattingRecomposerV832"]
