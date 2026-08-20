"""Small, tensor-only metrics used by the deterministic V2.32 validator."""

from __future__ import annotations

import torch


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    weight = mask.expand_as(value).to(value.dtype)
    return (value * weight).flatten(1).sum(1) / weight.flatten(1).sum(1).clamp_min(eps)


def masked_max(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rows = []
    for row, support in zip(value, mask):
        selected = row.flatten()[support.expand_as(row).flatten() > 0.5]
        rows.append(selected.max() if selected.numel() else row.new_tensor(0.0))
    return torch.stack(rows)


def foreground_reconstruction_metrics(image: torch.Tensor, foreground: torch.Tensor,
                                      background: torch.Tensor, alpha: torch.Tensor,
                                      transition: torch.Tensor) -> dict[str, torch.Tensor]:
    error = (alpha * foreground + (1.0 - alpha) * background - image).abs()
    return {
        "rgb_mae": error.flatten(1).mean(1),
        "rgb_max": error.flatten(1).amax(1),
        "transition_rgb_mae": masked_mean(error, transition),
        "foreground_min": foreground.flatten(1).amin(1),
        "foreground_max": foreground.flatten(1).amax(1),
        "background_min": background.flatten(1).amin(1),
        "background_max": background.flatten(1).amax(1),
        "nan_count": (~torch.isfinite(torch.cat((foreground, background), dim=1))).flatten(1).sum(1).float(),
    }


def alpha_color_correlation(alpha: torch.Tensor, delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    a = alpha.flatten(1)
    d = delta.flatten(1)
    m = mask.flatten(1).expand_as(a).to(a.dtype)
    a_mean = (a * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)
    d_mean = (d * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)
    ac = (a - a_mean) * m
    dc = (d - d_mean) * m
    return (ac * dc).sum(1).abs() / (ac.square().sum(1).sqrt() * dc.square().sum(1).sqrt()).clamp_min(1e-8)


__all__ = ["masked_mean", "masked_max", "foreground_reconstruction_metrics", "alpha_color_correlation"]
