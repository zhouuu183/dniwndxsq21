"""Deterministic narrow-trimap coverage matte for Blending V8.28."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _gradient_energy(image: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=1, keepdim=True)
    dx = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    return dx + dy


def _finite_distance(seed: torch.Tensor, max_steps: int) -> torch.Tensor:
    seed = seed > 0.5
    distance = torch.full_like(seed, float(max_steps + 1), dtype=torch.float32)
    distance = torch.where(seed, torch.zeros_like(distance), distance)
    reached = seed
    frontier = seed.float()
    for step in range(1, max_steps + 1):
        expanded = F.max_pool2d(frontier, 3, stride=1, padding=1) > 0.5
        newly = expanded & ~reached
        distance = torch.where(newly, torch.full_like(distance, float(step)), distance)
        reached = reached | newly
        frontier = reached.float()
    return distance


class HairCoverageMatteV828(nn.Module):
    """Create fractional alpha only within a deterministic boundary trimap."""

    def __init__(self, max_distance: int = 8, evidence_adjustment: float = 0.15):
        super().__init__()
        self.max_distance = int(max_distance)
        self.evidence_adjustment = float(evidence_adjustment)
        if self.max_distance < 1:
            raise ValueError("max_distance must be positive")
        if not 0.0 <= self.evidence_adjustment <= 0.15:
            raise ValueError("evidence_adjustment must be in [0, 0.15]")

    def forward(
        self,
        *,
        target_hair_mask: torch.Tensor,
        target_hair_eroded: torch.Tensor,
        target_hair_dilated: torch.Tensor,
        base_rgb: torch.Tensor,
        anchor_rgb: torch.Tensor,
        soft_hair_probability: torch.Tensor | None = None,
        skin_protect_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        hair = (target_hair_mask >= 0.5).to(base_rgb.dtype)
        core = ((target_hair_eroded >= 0.5) & (hair > 0.5)).to(base_rgb.dtype)
        support = ((target_hair_dilated >= 0.5) | (hair > 0.5)).to(base_rgb.dtype)
        outside = 1.0 - support
        unknown = (1.0 - core - outside).clamp(0, 1)

        d_fg = _finite_distance(core, self.max_distance).to(base_rgb.dtype)
        d_bg = _finite_distance(outside, self.max_distance).to(base_rgb.dtype)
        alpha_geom = d_bg / (d_fg + d_bg).clamp_min(1e-6)
        alpha_geom = alpha_geom.square() * (3.0 - 2.0 * alpha_geom)

        if soft_hair_probability is not None:
            alpha_seed = soft_hair_probability.to(base_rgb.dtype).clamp(0, 1) * support
            alpha_boundary = alpha_seed
            soft_used = torch.ones((), device=base_rgb.device, dtype=base_rgb.dtype)
        else:
            evidence = _gradient_energy(anchor_rgb) + _gradient_energy(base_rgb)
            local_min = -F.max_pool2d(-evidence, 5, stride=1, padding=2)
            local_max = F.max_pool2d(evidence, 5, stride=1, padding=2)
            normalized = (evidence - local_min) / (local_max - local_min).clamp_min(1e-6)
            adjustment = (normalized - 0.5) * (2.0 * self.evidence_adjustment)
            alpha_boundary = (alpha_geom + adjustment).clamp(0, 1)
            soft_used = torch.zeros((), device=base_rgb.device, dtype=base_rgb.dtype)

        if skin_protect_mask is not None:
            skin = skin_protect_mask.to(base_rgb.dtype).clamp(0, 1)
            alpha_boundary = torch.where(skin > 1e-6, torch.minimum(alpha_boundary, alpha_geom), alpha_boundary)

        alpha = core + unknown * alpha_boundary
        alpha = (alpha * support).clamp(0, 1)
        # Enforce the trimap invariants after all evidence adjustments.
        alpha = torch.where(core > 0.5, torch.ones_like(alpha), alpha)
        alpha = torch.where(outside > 0.5, torch.zeros_like(alpha), alpha)
        aux = {
            "alpha_geom": alpha_geom,
            "edge_evidence": evidence if soft_hair_probability is None else torch.zeros_like(alpha),
            "soft_probability_used": soft_used,
            "partition_error": (core + unknown + outside - 1.0).abs().amax(),
            "coverage_alpha_multiply_count": torch.ones((), device=base_rgb.device, dtype=base_rgb.dtype),
        }
        if not torch.isfinite(alpha).all():
            raise ValueError("V2.28 coverage matte produced NaN or Inf")
        return alpha, core, unknown, outside, aux


__all__ = ["HairCoverageMatteV828"]
