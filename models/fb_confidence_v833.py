"""Confidence maps for the frozen V2.32 foreground/background decomposition."""

from __future__ import annotations

import torch
from torch import nn


def smoothstep01(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


class FBConfidenceV833(nn.Module):
    def __init__(self, fg_low: float = 0.15, fg_high: float = 0.70,
                 bg_low: float = 0.15, bg_high: float = 0.70,
                 reconstruction_tau: float = 0.02):
        super().__init__()
        self.fg_low = float(fg_low)
        self.fg_high = float(fg_high)
        self.bg_low = float(bg_low)
        self.bg_high = float(bg_high)
        self.reconstruction_tau = float(reconstruction_tau)

    def config_dict(self) -> dict[str, float | bool]:
        return {
            "fg_alpha_low": self.fg_low, "fg_alpha_high": self.fg_high,
            "bg_alpha_low": self.bg_low, "bg_alpha_high": self.bg_high,
            "reconstruction_tau": self.reconstruction_tau,
            "sure_masks_exact": True,
        }

    def forward(self, *, alpha: torch.Tensor, sure_fg: torch.Tensor,
                sure_bg: torch.Tensor, unknown: torch.Tensor,
                reconstruction_error: torch.Tensor, return_aux: bool = False):
        error = reconstruction_error.abs().mean(dim=1, keepdim=True)
        c_recon = torch.exp(-error / max(self.reconstruction_tau, 1e-6))
        c_f_alpha = smoothstep01((alpha - self.fg_low) / max(self.fg_high - self.fg_low, 1e-6))
        background_ownership = 1.0 - alpha
        c_b_alpha = smoothstep01((background_ownership - self.bg_low) / max(self.bg_high - self.bg_low, 1e-6))
        foreground = c_f_alpha * c_recon
        background = c_b_alpha * c_recon
        foreground = torch.where(sure_fg > 0.5, torch.ones_like(foreground), foreground)
        foreground = torch.where(sure_bg > 0.5, torch.zeros_like(foreground), foreground)
        background = torch.where(sure_bg > 0.5, torch.ones_like(background), background)
        background = torch.where(sure_fg > 0.5, torch.zeros_like(background), background)
        transition = unknown.clamp(0, 1) * c_recon
        if not return_aux:
            return foreground, background, transition
        return foreground, background, transition, {
            "foreground_confidence": foreground,
            "background_confidence": background,
            "transition_confidence": transition,
            "foreground_alpha_confidence": c_f_alpha,
            "background_alpha_confidence": c_b_alpha,
            "reconstruction_confidence": c_recon,
        }


__all__ = ["FBConfidenceV833", "smoothstep01"]
