"""V2.35 hair-only chroma disentanglement and leakage-free transfer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.hybrid_hair_carrier_v829 import gaussian_blur_v829, normalized_blur_v829


def _match(value: torch.Tensor, size: tuple[int, int], *, mask: bool = False) -> torch.Tensor:
    if value.dim() == 2:
        value = value[None, None]
    elif value.dim() == 3:
        value = value[:, None]
    if value.shape[-2:] != size:
        value = F.interpolate(
            value.float(), size=size, mode="nearest" if mask else "bilinear",
            align_corners=None if mask else False,
        )
    return value


def build_hair_ownership_v835(
    *, anchor_hair_mask: torch.Tensor, target_hair_mask: torch.Tensor,
    face_mask: torch.Tensor, size: tuple[int, int],
) -> torch.Tensor:
    """Intersection ownership: Anchor hair and target hair, excluding face."""
    anchor = _match(anchor_hair_mask, size, mask=True).float().clamp(0, 1)
    target = _match(target_hair_mask, size, mask=True).float().clamp(0, 1)
    face = _match(face_mask, size, mask=True).float().clamp(0, 1)
    return (anchor * target * (1.0 - face)).clamp(0, 1)


class HairOnlyChromaDisentanglementV835(nn.Module):
    """Transfer only masked reference Lab AB while retaining Anchor structure/L."""

    def __init__(
        self, *, chroma_radius: int = 9, boundary_radius: int = 3,
        boundary_min_confidence: float = 0.20, chroma_scale: float = 18.0,
    ):
        super().__init__()
        if chroma_radius < 1 or boundary_radius < 1:
            raise ValueError("V2.35 radii must be positive")
        if not 0.0 <= boundary_min_confidence <= 1.0:
            raise ValueError("V2.35 boundary_min_confidence must be in [0, 1]")
        self.chroma_radius = int(chroma_radius)
        self.boundary_radius = int(boundary_radius)
        self.boundary_min_confidence = float(boundary_min_confidence)
        self.chroma_scale = float(chroma_scale)

    def config_dict(self) -> dict[str, object]:
        return {
            "chroma_radius": self.chroma_radius,
            "boundary_radius": self.boundary_radius,
            "boundary_min_confidence": self.boundary_min_confidence,
            "chroma_scale": self.chroma_scale,
            "reference_input": "HAIR_ONLY_RGB_CROP",
            "encoded_channels": "LAB_AB_ONLY",
            "target_structure_owner": "STRONG_ANCHOR",
            "fusion": "TARGET_AB_PLUS_GATED_CHROMA_RESIDUAL",
            "luminance_owner": "STRONG_ANCHOR",
        }

    @staticmethod
    def _masked_chroma_field(
        reference_rgb: torch.Tensor, reference_hair_mask: torch.Tensor,
        radius: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference_lab = rgb_to_lab(reference_rgb)
        reference_ab = reference_lab[:, 1:]
        support = reference_hair_mask.clamp(0, 1)
        local_ab, local_support = normalized_blur_v829(reference_ab, support, radius)
        global_num = (reference_ab * support).flatten(2).sum(2, keepdim=True)
        global_den = support.flatten(2).sum(2, keepdim=True).clamp_min(1e-6)
        global_ab = global_num / global_den
        global_ab = global_ab.unsqueeze(-1)
        field = torch.where(local_support > 0.05, local_ab, global_ab)
        # This is the explicit hair-only RGB crop required by V2.35. It is a
        # debug/contract tensor; the production field above contains AB only.
        neutral = torch.full_like(reference_rgb, 0.5)
        hair_only_rgb = reference_rgb * support + neutral * (1.0 - support)
        return field, local_support.clamp(0, 1), hair_only_rgb

    def forward(
        self, *, strong_anchor_rgb: torch.Tensor, color_reference_rgb: torch.Tensor,
        hair_mask: torch.Tensor, face_mask: torch.Tensor,
        anchor_hair_mask: torch.Tensor | None = None,
        target_hair_mask: torch.Tensor | None = None,
        reference_hair_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        anchor = _match(strong_anchor_rgb, strong_anchor_rgb.shape[-2:]).float().clamp(0, 1)
        size = anchor.shape[-2:]
        reference = _match(color_reference_rgb, size).to(anchor).clamp(0, 1)
        target_hair = _match(hair_mask if target_hair_mask is None else target_hair_mask, size, mask=True).to(anchor).clamp(0, 1)
        anchor_hair = target_hair if anchor_hair_mask is None else _match(anchor_hair_mask, size, mask=True).to(anchor).clamp(0, 1)
        face = _match(face_mask, size, mask=True).to(anchor).clamp(0, 1)
        reference_hair = target_hair if reference_hair_mask is None else _match(reference_hair_mask, size, mask=True).to(anchor).clamp(0, 1)
        ownership = build_hair_ownership_v835(
            anchor_hair_mask=anchor_hair, target_hair_mask=target_hair,
            face_mask=face, size=size,
        ).to(anchor)

        # Distance-to-boundary confidence: interior=1, boundary approaches the
        # specified 0.20 floor. No alpha is used as a color strength.
        interior_support = F.avg_pool2d(
            ownership, 2 * self.boundary_radius + 1, stride=1,
            padding=self.boundary_radius, count_include_pad=False,
        )
        distance_confidence = self.boundary_min_confidence + (
            1.0 - self.boundary_min_confidence
        ) * ((interior_support - 0.80) / 0.20).clamp(0, 1)
        distance_confidence = distance_confidence * ownership

        reference_ab, reference_support, hair_only_rgb = self._masked_chroma_field(
            reference, reference_hair, self.chroma_radius,
        )
        anchor_lab = rgb_to_lab(anchor)
        target_ab = anchor_lab[:, 1:]
        delta_ab = reference_ab - target_ab
        local_reference_ab = reference_ab
        global_ab = (local_reference_ab * reference_support).flatten(2).sum(2, keepdim=True) / reference_support.flatten(2).sum(2, keepdim=True).clamp_min(1e-6)
        chroma_distance = torch.linalg.vector_norm(local_reference_ab - global_ab.unsqueeze(-1), dim=1, keepdim=True)
        chroma_similarity = torch.exp(-chroma_distance / max(self.chroma_scale, 1e-4)).clamp(0.20, 1.0)
        support_confidence = reference_support.clamp(0, 1)
        confidence = (ownership * distance_confidence * support_confidence * chroma_similarity).clamp(0, 1)

        chroma_only_lab = anchor_lab.clone()
        chroma_only_lab[:, 1:] = target_ab + delta_ab * ownership
        chroma_only_rgb = lab_to_rgb(chroma_only_lab).clamp(0, 1)
        final_lab = anchor_lab.clone()
        final_lab[:, 1:] = target_ab + delta_ab * confidence
        edge_aware_rgb = lab_to_rgb(final_lab).clamp(0, 1)
        # Explicit face/background composition guarantees exact carrier pixels
        # outside hair ownership, independent of Lab round-trip precision.
        final_rgb = anchor + confidence * (edge_aware_rgb - anchor)
        if not torch.isfinite(final_rgb).all():
            raise ValueError("V2.35 hair-only chroma disentanglement produced NaN or Inf")
        if not return_aux:
            return final_rgb
        leakage_map = (final_rgb - anchor).abs().mean(1, keepdim=True) * (1.0 - ownership)
        anchor_hf = anchor - gaussian_blur_v829(anchor, 2)
        final_hf = final_rgb - gaussian_blur_v829(final_rgb, 2)
        aux = {
            "hair_ownership": ownership,
            "final_hair_mask": ownership,
            "color_hair_mask": reference_hair,
            "color_hair_only_rgb": hair_only_rgb,
            "chroma_feature_visual": torch.cat((reference_ab, torch.zeros_like(reference_ab[:, :1])), dim=1),
            "target_chroma_ab": target_ab,
            "color_chroma_ab": reference_ab,
            "delta_chroma_ab": delta_ab,
            "confidence_map": confidence,
            "distance_confidence": distance_confidence,
            "support_confidence": support_confidence,
            "chroma_similarity": chroma_similarity,
            "leakage_map": leakage_map,
            "chroma_only_rgb": chroma_only_rgb,
            "edge_aware_rgb": edge_aware_rgb,
            "final_rgb": final_rgb,
            "anchor_high_frequency": anchor_hf,
            "final_high_frequency": final_hf,
            "face_rgb_change_max": ((final_rgb - anchor) * face).abs().amax(dim=(1, 2, 3)),
            "non_hair_change_max": ((final_rgb - anchor) * (1.0 - ownership)).abs().amax(dim=(1, 2, 3)),
        }
        return final_rgb, aux


HairOnlyChromaTransferV835 = HairOnlyChromaDisentanglementV835

__all__ = [
    "HairOnlyChromaDisentanglementV835",
    "HairOnlyChromaTransferV835",
    "build_hair_ownership_v835",
]
