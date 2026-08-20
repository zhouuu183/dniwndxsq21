"""Low-frequency recoloring of completed PostProcess hair-edge pixels."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.SG_IDCT_v16 import rgb_to_lab
from models.hybrid_hair_carrier_v829 import normalized_blur_v829


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    return F.max_pool2d(mask, 2 * width + 1, stride=1, padding=width)


def _erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    return 1.0 - _dilate(1.0 - mask, width)


def _gradient(image: torch.Tensor) -> torch.Tensor:
    luminance = (
        0.2126 * image[:, 0:1]
        + 0.7152 * image[:, 1:2]
        + 0.0722 * image[:, 2:3]
    )
    dx = F.pad(luminance[..., :, 1:] - luminance[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(luminance[..., 1:, :] - luminance[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


class PPGuidedEdgeRecolorV830(nn.Module):
    """Preserve PP edge geometry while shifting only its local low-frequency tone."""

    def __init__(
        self,
        pp_tone_radius: int = 5,
        tone_propagation_radius: int = 7,
        face_contact_width: int = 4,
        prototype_radius: int = 5,
        lab_weight: float = 1.0,
        gradient_weight: float = 0.25,
        confidence_temperature: float = 8.0,
    ):
        super().__init__()
        self.pp_tone_radius = int(pp_tone_radius)
        self.tone_propagation_radius = int(tone_propagation_radius)
        self.face_contact_width = int(face_contact_width)
        self.prototype_radius = int(prototype_radius)
        self.lab_weight = float(lab_weight)
        self.gradient_weight = float(gradient_weight)
        self.confidence_temperature = float(confidence_temperature)

    def config_dict(self) -> dict[str, object]:
        return {
            "pp_tone_radius": self.pp_tone_radius,
            "tone_propagation_radius": self.tone_propagation_radius,
            "face_contact_width": self.face_contact_width,
            "prototype_radius": self.prototype_radius,
            "lab_weight": self.lab_weight,
            "gradient_weight": self.gradient_weight,
            "confidence_temperature": self.confidence_temperature,
            "pixel_carrier": "ORIGINAL_PP_COMPLETED_IMAGE",
            "tone_confidence_is_opacity": False,
        }

    def forward(
        self,
        *,
        pp_rgb: torch.Tensor,
        base_rgb: torch.Tensor,
        target_low_rgb: torch.Tensor,
        repaired_hair_mask: torch.Tensor,
        source_face_mask: torch.Tensor,
        final_unlock_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        output_size = pp_rgb.shape[-2:]
        base = F.interpolate(base_rgb, size=output_size, mode="bicubic", align_corners=False).clamp(0, 1)
        target_low = F.interpolate(
            target_low_rgb, size=output_size, mode="bicubic", align_corners=False
        ).clamp(0, 1)
        hair_soft = F.interpolate(
            repaired_hair_mask, size=output_size, mode="bilinear", align_corners=False
        ).clamp(0, 1)
        hair_hard = (hair_soft >= 0.5).to(pp_rgb.dtype)
        face = F.interpolate(source_face_mask, size=output_size, mode="nearest").clamp(0, 1)
        inner = (hair_hard - _erode(hair_hard, self.face_contact_width)).clamp(0, 1)
        outer = (_dilate(hair_hard, self.face_contact_width) - hair_hard).clamp(0, 1)
        contact = (inner + outer).clamp(0, 1) * face
        near_core = _erode(hair_hard, self.face_contact_width).clamp(0, 1)
        face_samples = face * (1.0 - _dilate(hair_hard, 1))

        low_pp_core, low_pp_support = normalized_blur_v829(
            pp_rgb, near_core, self.pp_tone_radius
        )
        delta_core = target_low - low_pp_core
        tone_delta, tone_support = normalized_blur_v829(
            delta_core, near_core, self.tone_propagation_radius
        )

        pp_lab = rgb_to_lab(pp_rgb)
        hair_proto_lab, hair_proto_support = normalized_blur_v829(
            pp_lab, near_core, self.prototype_radius
        )
        face_proto_lab, face_proto_support = normalized_blur_v829(
            rgb_to_lab(base), face_samples, self.prototype_radius
        )
        pp_gradient = _gradient(pp_rgb)
        hair_gradient, _ = normalized_blur_v829(
            pp_gradient, near_core, self.prototype_radius
        )
        face_gradient, _ = normalized_blur_v829(
            _gradient(base), face_samples, self.prototype_radius
        )
        d_hair = self.lab_weight * torch.linalg.vector_norm(
            pp_lab - hair_proto_lab, dim=1, keepdim=True
        ) + self.gradient_weight * (pp_gradient - hair_gradient).abs() * 100.0
        d_face = self.lab_weight * torch.linalg.vector_norm(
            pp_lab - face_proto_lab, dim=1, keepdim=True
        ) + self.gradient_weight * (pp_gradient - face_gradient).abs() * 100.0
        valid_hair = hair_proto_support >= 1e-3
        valid_face = face_proto_support >= 1e-3
        face_contact_confidence = torch.sigmoid(
            (d_face - d_hair) / max(self.confidence_temperature, 1e-6)
        )
        face_contact_confidence = torch.where(
            valid_hair & valid_face,
            face_contact_confidence,
            torch.where(valid_hair, torch.ones_like(face_contact_confidence), torch.zeros_like(face_contact_confidence)),
        )

        geometry_confidence = hair_soft
        tone_confidence = geometry_confidence * (1.0 - contact + contact * face_contact_confidence)
        tone_confidence = tone_confidence * (tone_support >= 1e-3).to(pp_rgb.dtype)
        if final_unlock_mask is not None:
            unlock = F.interpolate(
                final_unlock_mask, size=output_size, mode="nearest"
            ).clamp(0, 1)
            tone_confidence = tone_confidence * (1.0 - unlock)
        corrected_raw = pp_rgb + tone_confidence * tone_delta
        corrected = corrected_raw.clamp(0, 1)
        far_outside = 1.0 - _dilate(hair_hard, self.face_contact_width)
        corrected = torch.where(
            (far_outside >= 0.5).expand_as(corrected), pp_rgb, corrected
        )
        if not torch.isfinite(corrected).all():
            raise ValueError("V2.30 PP-guided recolor produced NaN or Inf")
        if not return_aux:
            return corrected
        return corrected, {
            "hair_soft_geometry": hair_soft,
            "hair_hard_geometry": hair_hard,
            "near_core": near_core,
            "contact_inside": inner * face,
            "contact_outside": outer * face,
            "face_contact_band": contact,
            "face_samples": face_samples,
            "low_pp_core": low_pp_core,
            "low_pp_support": low_pp_support,
            "target_low_core": target_low,
            "tone_delta_core": delta_core,
            "tone_delta_edge": tone_delta,
            "tone_delta_support": tone_support,
            "hair_prototype_lab": hair_proto_lab,
            "face_prototype_lab": face_proto_lab,
            "hair_prototype_distance": d_hair,
            "face_prototype_distance": d_face,
            "face_contact_confidence": face_contact_confidence,
            "hair_tone_confidence": tone_confidence,
            "far_outside": far_outside,
            "corrected_pp_raw": corrected_raw,
            "corrected_pp_rgb": corrected,
            "outside_exact_pp_max_delta": (
                (corrected - pp_rgb).abs() * far_outside
            ).flatten(1).amax(dim=1),
        }


__all__ = ["PPGuidedEdgeRecolorV830"]
