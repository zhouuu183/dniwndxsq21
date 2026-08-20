"""Coverage-independent spatial ownership for V2.29 hair compositing."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.hybrid_hair_carrier_v829 import normalized_blur_v829


class HairOwnershipV829(nn.Module):
    """Assign HM_X pixels to hair while rejecting dilation into source skin."""

    def __init__(
        self,
        outer_strand_recovery: bool = False,
        residual_alignment_threshold: float = 0.80,
        prototype_radius: int = 4,
    ):
        super().__init__()
        self.outer_strand_recovery = bool(outer_strand_recovery)
        self.residual_alignment_threshold = float(residual_alignment_threshold)
        self.prototype_radius = int(prototype_radius)

    def config_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "phase": "B" if self.outer_strand_recovery else "A",
            "outer_strand_recovery": self.outer_strand_recovery,
            "residual_alignment_threshold": self.residual_alignment_threshold,
            "prototype_radius": self.prototype_radius,
            "target_hair_priority_over_source_skin": True,
        }

    def forward(
        self,
        *,
        base_rgb: torch.Tensor,
        anchor_rgb: torch.Tensor,
        target_hair_mask: torch.Tensor,
        target_hair_eroded: torch.Tensor,
        target_hair_dilated: torch.Tensor,
        source_subject_mask: torch.Tensor,
        source_skin_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        hair = (target_hair_mask >= 0.5).to(base_rgb.dtype)
        core = ((target_hair_eroded >= 0.5) & (hair > 0.5)).to(base_rgb.dtype)
        inner = (hair - core).clamp(0, 1)
        outer = ((target_hair_dilated >= 0.5).to(base_rgb.dtype) - hair).clamp(0, 1)
        face_side_outer = outer * torch.maximum(source_subject_mask, source_skin_mask)
        background_side_outer = outer * (1.0 - torch.maximum(source_subject_mask, source_skin_mask))

        residual = anchor_rgb - base_rgb
        prototype, prototype_support = normalized_blur_v829(
            residual, (inner + core).clamp(0, 1), self.prototype_radius
        )
        dot = (residual * prototype).sum(dim=1, keepdim=True)
        norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True) * torch.linalg.vector_norm(
            prototype, dim=1, keepdim=True
        )
        alignment = dot / norm.clamp_min(1e-6)
        magnitude_ratio = torch.linalg.vector_norm(residual, dim=1, keepdim=True) / torch.linalg.vector_norm(
            prototype, dim=1, keepdim=True
        ).clamp_min(1e-6)
        evidence = (
            (alignment >= self.residual_alignment_threshold)
            & (magnitude_ratio >= 0.25)
            & (magnitude_ratio <= 4.0)
            & (prototype_support >= 1e-3)
        ).to(base_rgb.dtype) * background_side_outer
        # Remove isolated specks while retaining aligned strand runs.
        neighbor_count = F.conv2d(evidence, torch.ones(1, 1, 3, 3, device=evidence.device, dtype=evidence.dtype), padding=1)
        outer_owner = evidence * (neighbor_count >= 3).to(evidence.dtype)
        if not self.outer_strand_recovery:
            outer_owner = torch.zeros_like(outer_owner)

        # HM_X always owns core and inner edge, even where Source says skin.
        ownership = (core + inner + outer_owner).clamp(0, 1)
        nonhair = 1.0 - ownership
        partition_error = (core + inner + outer_owner + nonhair - 1.0).abs().amax()
        if not return_aux:
            return ownership
        aux = {
            "core_owner": core,
            "inner_edge_owner": inner,
            "outer_strand_owner": outer_owner,
            "nonhair_owner": nonhair,
            "hair_ownership": ownership,
            "outer_candidate": outer,
            "face_side_outer": face_side_outer,
            "background_side_outer": background_side_outer,
            "outer_hair_evidence": evidence,
            "residual_alignment": alignment,
            "residual_magnitude_ratio": magnitude_ratio,
            "neighbor_support_count": neighbor_count,
            "partition_error": partition_error,
        }
        return ownership, aux


def apply_pp_hair_ownership_lock_v829(
    prepp_rgb: torch.Tensor,
    pp_rgb_original: torch.Tensor,
    hair_ownership: torch.Tensor,
    pp_unlock_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign completed owner pixels to pre-PP without coverage attenuation."""
    owner = (hair_ownership.to(device=prepp_rgb.device, dtype=prepp_rgb.dtype) >= 0.5).to(prepp_rgb.dtype)
    if pp_unlock_mask is not None:
        unlock = pp_unlock_mask.to(device=prepp_rgb.device, dtype=prepp_rgb.dtype)
        if unlock.dim() == 3:
            unlock = unlock.unsqueeze(1)
        if unlock.shape[-2:] != owner.shape[-2:]:
            unlock = F.interpolate(unlock, size=owner.shape[-2:], mode="nearest")
        owner = owner * (1.0 - unlock.clamp(0, 1))
        owner = (owner >= 0.5).to(prepp_rgb.dtype)
    final = torch.where(owner.expand_as(prepp_rgb) > 0.5, prepp_rgb, pp_rgb_original)
    return final.clamp(0, 1), owner


__all__ = ["HairOwnershipV829", "apply_pp_hair_ownership_lock_v829"]
