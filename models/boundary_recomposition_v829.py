"""Ownership-aware background replacement for already-composited hair edges."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.hair_coverage_matte_v828 import HairCoverageMatteV828
from models.hair_ownership_v829 import HairOwnershipV829
from models.hybrid_hair_carrier_v829 import (
    gaussian_blur_v829,
    normalized_blur_v829,
)


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    return F.max_pool2d(mask, 2 * width + 1, stride=1, padding=width)


def recompose_observed_boundary_v829(
    anchor_rgb: torch.Tensor,
    coverage_alpha: torch.Tensor,
    anchor_background: torch.Tensor,
    base_background: torch.Tensor,
    tone_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace only the background contribution of an observed composite."""
    tone = torch.zeros_like(anchor_rgb) if tone_delta is None else tone_delta
    return anchor_rgb + coverage_alpha * tone + (1.0 - coverage_alpha) * (
        base_background - anchor_background
    )


class BoundaryRecompositionV829(nn.Module):
    """Build V2.29 pre-PP without multiplying Anchor hair coverage twice."""

    def __init__(
        self,
        coverage_max_distance: int = 8,
        carrier_low_radius: int = 5,
        tone_propagation_radius: int = 7,
        background_radius: int = 7,
        background_ring_inner: int = 2,
        background_ring_outer: int = 12,
        background_valid_threshold: float = 1e-3,
        outer_strand_recovery: bool = False,
    ):
        super().__init__()
        self.coverage = HairCoverageMatteV828(max_distance=coverage_max_distance)
        self.ownership = HairOwnershipV829(outer_strand_recovery=outer_strand_recovery)
        self.carrier_low_radius = int(carrier_low_radius)
        self.tone_propagation_radius = int(tone_propagation_radius)
        self.background_radius = int(background_radius)
        self.background_ring_inner = int(background_ring_inner)
        self.background_ring_outer = int(background_ring_outer)
        self.background_valid_threshold = float(background_valid_threshold)

    def config_dict(self) -> dict[str, object]:
        return {
            "equation": "A + alpha*tone_delta + (1-alpha)*(B_base-B_anchor)",
            "double_coverage_deleted": True,
            "coverage_max_distance": self.coverage.max_distance,
            "carrier_low_radius": self.carrier_low_radius,
            "tone_propagation_radius": self.tone_propagation_radius,
            "background_radius": self.background_radius,
            "background_ring": [self.background_ring_inner, self.background_ring_outer],
            **self.ownership.config_dict(),
        }

    def _context_estimates(
        self,
        anchor_rgb: torch.Tensor,
        base_rgb: torch.Tensor,
        hair: torch.Tensor,
        source_subject: torch.Tensor,
        owner_region: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        near_exclusion = _dilate(hair, self.background_ring_inner)
        far_support = _dilate(hair, self.background_ring_outer)
        sample_ring = (far_support - near_exclusion).clamp(0, 1)
        face_samples = sample_ring * source_subject
        background_samples = sample_ring * (1.0 - source_subject)

        anchor_face, face_den = normalized_blur_v829(anchor_rgb, face_samples, self.background_radius)
        base_face, _ = normalized_blur_v829(base_rgb, face_samples, self.background_radius)
        anchor_bg, bg_den = normalized_blur_v829(anchor_rgb, background_samples, self.background_radius)
        base_bg, _ = normalized_blur_v829(base_rgb, background_samples, self.background_radius)

        face_valid = (face_den >= self.background_valid_threshold).to(base_rgb.dtype)
        bg_valid = (bg_den >= self.background_valid_threshold).to(base_rgb.dtype)
        # A locally available face estimate takes precedence at the hairline so
        # distant background color cannot leak into a hair-to-face boundary.
        use_face = face_valid
        use_bg = bg_valid * (1.0 - use_face)
        valid = ((use_face + use_bg) > 0.5).to(base_rgb.dtype) * owner_region
        anchor_estimate = use_face * anchor_face + use_bg * anchor_bg
        base_estimate = use_face * base_face + use_bg * base_bg
        return {
            "anchor_background_estimate": anchor_estimate,
            "base_background_estimate": base_estimate,
            "background_recompose_valid": valid,
            "face_context_map": use_face * owner_region,
            "background_context_map": use_bg * owner_region,
            "face_samples": face_samples,
            "background_samples": background_samples,
            "face_estimate_denominator": face_den,
            "background_estimate_denominator": bg_den,
        }

    def forward(
        self,
        *,
        base_rgb: torch.Tensor,
        anchor_rgb: torch.Tensor,
        v226_rgb: torch.Tensor,
        hybrid_core_rgb: torch.Tensor,
        target_hair_mask: torch.Tensor,
        target_hair_eroded: torch.Tensor,
        target_hair_dilated: torch.Tensor,
        source_subject_mask: torch.Tensor,
        source_skin_mask: torch.Tensor,
        hard_protect_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        coverage, _, _, _, coverage_aux = self.coverage(
            target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded,
            target_hair_dilated=target_hair_dilated,
            base_rgb=base_rgb,
            anchor_rgb=anchor_rgb,
            # Source skin must not attenuate HM_X-owned inner hair.
            skin_protect_mask=None,
        )
        ownership, owner_aux = self.ownership(
            base_rgb=base_rgb,
            anchor_rgb=anchor_rgb,
            target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded,
            target_hair_dilated=target_hair_dilated,
            source_subject_mask=source_subject_mask,
            source_skin_mask=source_skin_mask,
            return_aux=True,
        )
        core = owner_aux["core_owner"]
        boundary_owner = (owner_aux["inner_edge_owner"] + owner_aux["outer_strand_owner"]).clamp(0, 1)
        nonhair = owner_aux["nonhair_owner"]

        low_v226, _ = normalized_blur_v829(v226_rgb, core, self.carrier_low_radius)
        anchor_low = gaussian_blur_v829(anchor_rgb, self.carrier_low_radius)
        tone_delta_core = low_v226 - anchor_low
        tone_delta_edge, tone_support = normalized_blur_v829(
            tone_delta_core, core, self.tone_propagation_radius
        )
        contexts = self._context_estimates(
            anchor_rgb, base_rgb, target_hair_mask, source_subject_mask, boundary_owner
        )
        raw_boundary = recompose_observed_boundary_v829(
            anchor_rgb,
            coverage,
            contexts["anchor_background_estimate"],
            contexts["base_background_estimate"],
            tone_delta_edge,
        )
        # No estimate fallback preserves Anchor composite plus foreground tone.
        fallback_boundary = anchor_rgb + coverage * tone_delta_edge
        valid = contexts["background_recompose_valid"]
        boundary_raw_selected = valid * raw_boundary + (1.0 - valid) * fallback_boundary
        boundary_rgb = boundary_raw_selected.clamp(0, 1)

        prepp = core * hybrid_core_rgb + boundary_owner * boundary_rgb + nonhair * base_rgb
        hard = (hard_protect_mask >= 0.5).to(base_rgb.dtype)
        prepp = torch.where(hard.expand_as(prepp) > 0.5, base_rgb, prepp).clamp(0, 1)
        effective_ownership = ownership * (1.0 - hard)
        effective_nonhair = 1.0 - effective_ownership

        clip_low = (-boundary_raw_selected).clamp_min(0)
        clip_high = (boundary_raw_selected - 1.0).clamp_min(0)
        if not torch.isfinite(prepp).all():
            raise ValueError("V2.29 boundary recomposition produced NaN or Inf")
        if not return_aux:
            return prepp
        aux = {
            "coverage_alpha": coverage,
            "hair_ownership": effective_ownership,
            "nonhair_owner": effective_nonhair,
            "boundary_owner": boundary_owner * (1.0 - hard),
            "hybrid_core_rgb": hybrid_core_rgb,
            "tone_delta_core": tone_delta_core,
            "tone_delta_edge": tone_delta_edge,
            "tone_delta_support": tone_support,
            "boundary_rgb": boundary_rgb,
            "boundary_raw": boundary_raw_selected,
            "background_replacement_delta": (1.0 - coverage) * (
                contexts["base_background_estimate"] - contexts["anchor_background_estimate"]
            ) * valid,
            "prepp_rgb": prepp,
            "hard_protect": hard,
            "outside_max_delta": ((prepp - base_rgb).abs() * effective_nonhair).amax(dim=(1, 2, 3)),
            "hard_protect_max_delta": ((prepp - base_rgb).abs() * hard).amax(dim=(1, 2, 3)),
            "owner_partition_error": (
                owner_aux["core_owner"] + owner_aux["inner_edge_owner"]
                + owner_aux["outer_strand_owner"] + owner_aux["nonhair_owner"] - 1.0
            ).abs().amax(),
            "clip_low_fraction": ((clip_low > 0).float() * boundary_owner).mean(dim=(1, 2, 3)),
            "clip_high_fraction": ((clip_high > 0).float() * boundary_owner).mean(dim=(1, 2, 3)),
            "clip_magnitude": ((clip_low + clip_high) * boundary_owner).mean(dim=(1, 2, 3)),
            **coverage_aux,
            **owner_aux,
            **contexts,
        }
        # Ensure effective masks overwrite their pre-hard-protect counterparts.
        aux["hair_ownership"] = effective_ownership
        aux["nonhair_owner"] = effective_nonhair
        return prepp, aux


__all__ = ["BoundaryRecompositionV829", "recompose_observed_boundary_v829"]
