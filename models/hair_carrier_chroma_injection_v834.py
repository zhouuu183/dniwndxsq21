"""V2.34 Strong Anchor hair-carrier, low-frequency chroma-only injection."""

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
        value = F.interpolate(value.float(), size=size, mode="nearest" if mask else "bilinear",
                              align_corners=None if mask else False)
    return value


def _gradient_confidence(anchor_rgb: torch.Tensor, hair: torch.Tensor) -> torch.Tensor:
    luminance = 0.2126 * anchor_rgb[:, :1] + 0.7152 * anchor_rgb[:, 1:2] + 0.0722 * anchor_rgb[:, 2:3]
    dx = F.pad(luminance[..., :, 1:] - luminance[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(luminance[..., 1:, :] - luminance[..., :-1, :], (0, 0, 0, 1))
    magnitude = torch.sqrt(dx.square() + dy.square() + 1e-12)
    local = gaussian_blur_v829(magnitude, 5)
    normalized = (magnitude / (local + 1e-4)).clamp(0, 2.0) / 2.0
    # Strong gradients are handled conservatively at the carrier boundary.
    return (1.0 - 0.50 * normalized).clamp(0.35, 1.0) * hair


def build_hair_ownership_v834(anchor_hair_mask: torch.Tensor, target_hair_mask: torch.Tensor,
                              face_mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Fuse Anchor/target ownership and hard-exclude face/background pixels."""
    anchor = _match(anchor_hair_mask, size, mask=True).float().clamp(0, 1)
    target = _match(target_hair_mask, size, mask=True).float().clamp(0, 1)
    face = _match(face_mask, size, mask=True).float().clamp(0, 1)
    return (anchor * target * (1.0 - face)).clamp(0, 1)


class HairCarrierChromaInjectionV834(nn.Module):
    """Keep Strong Anchor appearance and inject only a smooth reference AB field."""

    def __init__(self, *, reference_radius: int = 9, edge_radius: int = 3,
                 edge_min_confidence: float = 0.35, edge_max_confidence: float = 0.50):
        super().__init__()
        if reference_radius < 1 or edge_radius < 1:
            raise ValueError("V2.34 blur radii must be positive")
        self.reference_radius = int(reference_radius)
        self.edge_radius = int(edge_radius)
        self.edge_min_confidence = float(edge_min_confidence)
        self.edge_max_confidence = float(edge_max_confidence)
        if not 0.0 <= self.edge_min_confidence <= self.edge_max_confidence <= 1.0:
            raise ValueError("V2.34 edge confidence bounds must satisfy 0 <= min <= max <= 1")

    def config_dict(self) -> dict[str, object]:
        return {
            "reference_radius": self.reference_radius,
            "edge_radius": self.edge_radius,
            "edge_min_confidence": self.edge_min_confidence,
            "edge_max_confidence": self.edge_max_confidence,
            "carrier": "STRONG_ANCHOR",
            "injection": "REFERENCE_LOW_FREQUENCY_AB_ONLY",
            "reference_field": "REFERENCE_HAIR_MASKED_WHEN_AVAILABLE",
            "luminance_preserve": True,
        }

    def forward(self, *, strong_anchor_rgb: torch.Tensor, color_reference_rgb: torch.Tensor,
                hair_mask: torch.Tensor, face_mask: torch.Tensor,
                anchor_hair_mask: torch.Tensor | None = None,
                reference_hair_mask: torch.Tensor | None = None,
                return_aux: bool = False):
        anchor = _match(strong_anchor_rgb, strong_anchor_rgb.shape[-2:]).float().clamp(0, 1)
        size = anchor.shape[-2:]
        reference = _match(color_reference_rgb, size).to(anchor).clamp(0, 1)
        hair = _match(hair_mask, size, mask=True).to(anchor).clamp(0, 1)
        anchor_hair = hair if anchor_hair_mask is None else _match(anchor_hair_mask, size, mask=True).to(anchor).clamp(0, 1)
        face = _match(face_mask, size, mask=True).to(anchor).clamp(0, 1)
        ownership = build_hair_ownership_v834(anchor_hair, hair, face, size).to(anchor)

        # A soft mask gives full ownership in the interior and a bounded 0.35
        # to 1.0 confidence at the carrier boundary.
        soft = F.avg_pool2d(ownership, 2 * self.edge_radius + 1, stride=1,
                            padding=self.edge_radius, count_include_pad=False)
        edge_weight = self.edge_min_confidence + (self.edge_max_confidence - self.edge_min_confidence) * soft
        interior = ((soft - 0.75) / 0.25).clamp(0, 1)
        boundary_confidence = edge_weight + (1.0 - edge_weight) * interior
        gradient_raw = _gradient_confidence(anchor, ownership)
        gradient_confidence = torch.where(interior >= 1.0, ownership, gradient_raw)
        confidence = (boundary_confidence * gradient_confidence * ownership).clamp(0, 1)

        anchor_lab = rgb_to_lab(anchor)
        if reference_hair_mask is None:
            reference_low = gaussian_blur_v829(reference, self.reference_radius)
        else:
            ref_support = _match(reference_hair_mask, size, mask=True).to(anchor).clamp(0, 1)
            propagated, support = normalized_blur_v829(reference, ref_support, self.reference_radius)
            reference_low = torch.where(support > 0.05, propagated, reference)
        anchor_low = gaussian_blur_v829(anchor, self.reference_radius)
        reference_low_lab = rgb_to_lab(reference_low)
        anchor_low_lab = rgb_to_lab(anchor_low)
        chroma_delta = reference_low_lab[:, 1:] - anchor_low_lab[:, 1:]

        chroma_only_lab = anchor_lab.clone()
        chroma_only_lab[:, 1:] = chroma_only_lab[:, 1:] + chroma_delta * ownership
        chroma_only = lab_to_rgb(chroma_only_lab).clamp(0, 1)
        final_lab = anchor_lab.clone()
        final_lab[:, 1:] = final_lab[:, 1:] + chroma_delta * confidence
        final = lab_to_rgb(final_lab).clamp(0, 1)
        if not torch.isfinite(final).all():
            raise ValueError("V2.34 hair carrier chroma injection produced NaN or Inf")
        if not return_aux:
            return final
        aux = {
            "hair_ownership": ownership,
            "anchor_hair_mask": anchor_hair,
            "boundary_confidence": boundary_confidence * ownership,
            "gradient_confidence": gradient_confidence,
            "confidence_map": confidence,
            "chroma_delta": chroma_delta,
            "anchor_lab": anchor_lab,
            "reference_low_rgb": reference_low,
            "anchor_low_rgb": anchor_low,
            "chroma_only_rgb": chroma_only,
            "edge_aware_rgb": final,
            "face_injection_max": (confidence * face).amax(dim=(1, 2, 3)),
            "outside_injection_max": (confidence * (1.0 - hair)).amax(dim=(1, 2, 3)),
        }
        return final, aux


HairCarrierChromaFieldInjectionV834 = HairCarrierChromaInjectionV834
HairCarrierChromaInjectorV834 = HairCarrierChromaInjectionV834

__all__ = [
    "HairCarrierChromaInjectionV834",
    "HairCarrierChromaFieldInjectionV834",
    "HairCarrierChromaInjectorV834",
    "build_hair_ownership_v834",
]
