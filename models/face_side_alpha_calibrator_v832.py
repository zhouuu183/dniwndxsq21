"""Face-side alpha calibration using foreground/background physical hypotheses."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.hybrid_hair_carrier_v829 import normalized_blur_v829


class FaceSideAlphaCalibratorV832(nn.Module):
    def __init__(self, prototype_radius: int = 7, temperature: float = 0.04):
        super().__init__()
        self.prototype_radius = int(prototype_radius)
        self.temperature = float(temperature)

    def config_dict(self) -> dict[str, object]:
        return {
            "prototype_radius": self.prototype_radius,
            "temperature_rgb": self.temperature,
            "observed_image": "OriginalPP",
            "sure_masks_immutable": True,
        }

    def forward(self, *, alpha_hr: torch.Tensor, foreground_pp_rgb: torch.Tensor,
                background_pp_rgb: torch.Tensor, base_rgb: torch.Tensor,
                source_face_mask: torch.Tensor, sure_fg: torch.Tensor,
                sure_bg: torch.Tensor, unknown: torch.Tensor,
                observed_pp_rgb: torch.Tensor | None = None,
                return_aux: bool = False):
        size = alpha_hr.shape[-2:]
        face = F.interpolate(source_face_mask, size=size, mode="nearest").clamp(0, 1)
        base = F.interpolate(base_rgb, size=size, mode="bicubic", align_corners=False).clamp(0, 1)
        nonface = 1.0 - face
        hair_samples = (alpha_hr >= 0.90).to(alpha_hr.dtype) * nonface
        face_samples = (alpha_hr <= 0.05).to(alpha_hr.dtype) * face
        hair_proto, hair_support = normalized_blur_v829(foreground_pp_rgb, hair_samples, self.prototype_radius)
        face_proto, face_support = normalized_blur_v829(base, face_samples, self.prototype_radius)
        near_hair = F.max_pool2d((alpha_hr > 0.01).to(alpha_hr.dtype), 9, stride=1, padding=4)
        contact = unknown * face * near_hair
        observed = observed_pp_rgb if observed_pp_rgb is not None else (
            alpha_hr * foreground_pp_rgb + (1.0 - alpha_hr) * background_pp_rgb
        )
        i_pp = observed
        e_face = torch.linalg.vector_norm(i_pp - face_proto, dim=1, keepdim=True)
        i_hair = alpha_hr * hair_proto + (1.0 - alpha_hr) * face_proto
        e_hair = torch.linalg.vector_norm(i_pp - i_hair, dim=1, keepdim=True)
        posterior = torch.sigmoid((e_face - e_hair) / max(self.temperature, 1e-5))
        valid = (hair_support >= 1e-4) & (face_support >= 1e-4)
        posterior = torch.where(valid, posterior, torch.ones_like(posterior))
        alpha_eff = alpha_hr.clone()
        alpha_eff = torch.where(contact > 0.5, alpha_hr * posterior, alpha_eff)
        alpha_eff = torch.where(sure_fg > 0.5, torch.ones_like(alpha_eff), alpha_eff)
        alpha_eff = torch.where(sure_bg > 0.5, torch.zeros_like(alpha_eff), alpha_eff).clamp(0, 1)
        if not return_aux:
            return alpha_eff
        return alpha_eff, {
            "hair_prototype": hair_proto, "face_prototype": face_proto,
            "hair_posterior": posterior, "face_contact_unknown": contact,
            "alpha_eff": alpha_eff, "alpha_calibration_delta": alpha_eff - alpha_hr,
            "face_alpha_eff": alpha_eff * contact,
        }


__all__ = ["FaceSideAlphaCalibratorV832"]
