"""Transition-only local Base background target for V2.32."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.hybrid_hair_carrier_v829 import normalized_blur_v829


class BackgroundTargetV832(nn.Module):
    def __init__(self, radius: int = 9, transition_expand: int = 2):
        super().__init__()
        self.radius = int(radius)
        self.transition_expand = int(transition_expand)

    def config_dict(self) -> dict[str, object]:
        return {"radius": self.radius, "transition_expand": self.transition_expand, "far_outside": "EXACT_PP"}

    def forward(self, *, background_pp_rgb: torch.Tensor, base_rgb: torch.Tensor,
                alpha_hr: torch.Tensor, unknown: torch.Tensor,
                source_face_mask: torch.Tensor, source_subject_mask: torch.Tensor,
                return_aux: bool = False):
        size = alpha_hr.shape[-2:]
        base = F.interpolate(base_rgb, size=size, mode="bicubic", align_corners=False).clamp(0, 1)
        face = F.interpolate(source_face_mask, size=size, mode="nearest").clamp(0, 1)
        subject = F.interpolate(source_subject_mask, size=size, mode="nearest").clamp(0, 1)
        transition = ((alpha_hr > 0.01) & (alpha_hr < 0.99)).to(alpha_hr.dtype)
        transition = torch.maximum(transition, unknown).clamp(0, 1)
        kernel = 2 * self.transition_expand + 1
        transition = F.max_pool2d(transition, kernel, stride=1, padding=self.transition_expand)
        confident = (alpha_hr <= 0.02).to(alpha_hr.dtype)
        face_samples = confident * face
        scene_samples = confident * (1.0 - subject)
        base_face, face_support = normalized_blur_v829(base, face_samples, self.radius)
        base_scene, scene_support = normalized_blur_v829(base, scene_samples, self.radius)
        base_local = face * base_face + (1.0 - face) * base_scene
        support = face * (face_support >= 1e-4).to(alpha_hr.dtype) + (1.0 - face) * (scene_support >= 1e-4).to(alpha_hr.dtype)
        context_conf = transition * support
        target = background_pp_rgb + context_conf * (base_local - background_pp_rgb)
        target = torch.where((1.0 - transition).expand_as(target) > 0.5, background_pp_rgb, target).clamp(0, 1)
        if not return_aux:
            return target
        return target, {
            "transition_support": transition, "context_confidence": context_conf,
            "base_local": base_local, "background_target_rgb": target,
            "far_outside_max_delta": ((target - background_pp_rgb).abs() * (1.0 - transition)).flatten(1).amax(dim=1),
        }


__all__ = ["BackgroundTargetV832"]
