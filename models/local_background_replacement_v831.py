"""Alpha-aware local face/background decontamination for V2.31 Phase C."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.hybrid_hair_carrier_v829 import normalized_blur_v829


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    return F.max_pool2d(mask, 2 * width + 1, stride=1, padding=width)


class LocalBackgroundReplacementV831(nn.Module):
    def __init__(self, context_radius: int = 9, transition_expand: int = 2):
        super().__init__()
        self.context_radius = int(context_radius)
        self.transition_expand = int(transition_expand)

    def config_dict(self) -> dict[str, object]:
        return {
            "equation": "PP_TONED + (1-alpha)*(B_BASE-B_PP)",
            "context_radius": self.context_radius,
            "transition_expand": self.transition_expand,
            "far_outside_exact_pp": True,
        }

    def forward(
        self,
        *,
        phase_b_rgb: torch.Tensor,
        pp_rgb: torch.Tensor,
        base_rgb: torch.Tensor,
        alpha_hr: torch.Tensor,
        unknown: torch.Tensor,
        source_face_mask: torch.Tensor,
        source_subject_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        output_size = pp_rgb.shape[-2:]
        base = F.interpolate(
            base_rgb, size=output_size, mode="bicubic", align_corners=False
        ).clamp(0, 1)
        face = F.interpolate(source_face_mask, size=output_size, mode="nearest").clamp(0, 1)
        subject = F.interpolate(
            source_subject_mask, size=output_size, mode="nearest"
        ).clamp(0, 1)
        alpha = alpha_hr.to(pp_rgb).clamp(0, 1)
        confident_nonhair = (alpha <= 0.02).to(alpha.dtype)
        face_samples = confident_nonhair * face
        scene_samples = confident_nonhair * (1.0 - subject)
        b_pp_face, face_support = normalized_blur_v829(pp_rgb, face_samples, self.context_radius)
        b_base_face, _ = normalized_blur_v829(base, face_samples, self.context_radius)
        b_pp_scene, scene_support = normalized_blur_v829(pp_rgb, scene_samples, self.context_radius)
        b_base_scene, _ = normalized_blur_v829(base, scene_samples, self.context_radius)

        transition = ((alpha > 0.01) & (alpha < 0.99)).to(alpha.dtype)
        transition = torch.maximum(transition, unknown.to(alpha))
        transition = _dilate(transition, self.transition_expand).clamp(0, 1)
        face_context = transition * face
        scene_context = transition * (1.0 - subject)
        face_valid = (face_support >= 1e-4).to(alpha.dtype)
        scene_valid = (scene_support >= 1e-4).to(alpha.dtype)
        face_delta = (b_base_face - b_pp_face) * face_context * face_valid
        scene_delta = (b_base_scene - b_pp_scene) * scene_context * scene_valid
        replacement = (1.0 - alpha) * (face_delta + scene_delta)
        phase_c = (phase_b_rgb + replacement).clamp(0, 1)
        far_outside = 1.0 - transition
        phase_c = torch.where(far_outside.expand_as(phase_c) > 0.5, phase_b_rgb, phase_c)
        if not torch.isfinite(phase_c).all():
            raise ValueError("V2.31 local background replacement produced NaN or Inf")
        if not return_aux:
            return phase_c
        return phase_c, {
            "confident_nonhair": confident_nonhair,
            "face_samples": face_samples,
            "scene_samples": scene_samples,
            "transition_support": transition,
            "face_context": face_context,
            "scene_context": scene_context,
            "b_pp_face": b_pp_face,
            "b_base_face": b_base_face,
            "b_pp_scene": b_pp_scene,
            "b_base_scene": b_base_scene,
            "background_replacement": replacement,
            "far_outside": far_outside,
            "far_outside_max_delta": (
                (phase_c - phase_b_rgb).abs() * far_outside
            ).flatten(1).amax(dim=1),
            "phase_c_rgb": phase_c,
        }


__all__ = ["LocalBackgroundReplacementV831"]
