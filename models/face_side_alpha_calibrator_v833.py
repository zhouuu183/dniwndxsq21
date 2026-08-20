"""Face-side ownership calibration using reliable hair and geometry priors."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.hybrid_hair_carrier_v829 import normalized_blur_v829


class FaceSideAlphaCalibratorV833(nn.Module):
    def __init__(self, posterior_radius: int = 5, temperature: float = 0.04):
        super().__init__()
        self.posterior_radius = int(posterior_radius)
        self.temperature = float(temperature)

    def config_dict(self) -> dict[str, object]:
        return {
            "hair_hypothesis": "RELIABLE_PROPAGATED_TARGET_FOREGROUND",
            "face_hypothesis": "SAME_PIXEL_BASE",
            "connectivity_prior": True,
            "invalid_fallback": "PROPAGATED_OR_CONNECTIVITY",
            "invalid_fallback_one": False,
            "sure_masks_exact": True,
        }

    @staticmethod
    def _connectivity(sure_fg: torch.Tensor) -> torch.Tensor:
        # Nested dilations form a deterministic distance-like prior.
        near3 = F.max_pool2d(sure_fg, 7, stride=1, padding=3)
        near7 = F.max_pool2d(sure_fg, 15, stride=1, padding=7)
        near15 = F.max_pool2d(sure_fg, 31, stride=1, padding=15)
        return torch.maximum(near3, torch.maximum(0.60 * near7, 0.25 * near15)).clamp(0, 1)

    def forward(self, *, alpha: torch.Tensor, observed_pp_rgb: torch.Tensor,
                hair_target_prior_rgb: torch.Tensor, base_rgb: torch.Tensor,
                foreground_confidence: torch.Tensor, source_face_mask: torch.Tensor,
                coarse_target_hair: torch.Tensor, sure_fg: torch.Tensor,
                sure_bg: torch.Tensor, unknown: torch.Tensor,
                return_aux: bool = False):
        size = alpha.shape[-2:]
        face = F.interpolate(source_face_mask, size=size, mode="nearest").clamp(0, 1)
        coarse = F.interpolate(coarse_target_hair, size=size, mode="nearest").clamp(0, 1)
        base = F.interpolate(base_rgb, size=size, mode="bicubic", align_corners=False).clamp(0, 1)
        connectivity = self._connectivity(sure_fg.float())
        near_hair = F.max_pool2d((coarse > 0.5).to(alpha.dtype), 9, stride=1, padding=4)
        contact = unknown * face * near_hair
        face_unknown_all = unknown * face
        false_positive_proxy = contact * (1.0 - F.avg_pool2d(coarse, 9, stride=1, padding=4).clamp(0, 1))

        face_hypothesis = base
        hair_hypothesis = alpha * hair_target_prior_rgb + (1.0 - alpha) * face_hypothesis
        e_face = torch.linalg.vector_norm(observed_pp_rgb - face_hypothesis, dim=1, keepdim=True)
        e_hair = torch.linalg.vector_norm(observed_pp_rgb - hair_hypothesis, dim=1, keepdim=True)
        evidence = torch.sigmoid((e_face - e_hair) / max(self.temperature, 1e-6))
        valid = (foreground_confidence >= 0.15).to(alpha.dtype) * contact
        evidence_prop, prop_support = normalized_blur_v829(evidence, valid, self.posterior_radius)
        invalid_connectivity = connectivity.clamp(max=0.95)
        evidence_final = torch.where(
            valid > 0.5,
            evidence,
            torch.where(prop_support >= 0.02, evidence_prop, invalid_connectivity),
        ).clamp(0, 1)
        posterior = torch.sqrt((evidence_final * connectivity).clamp_min(0.0))
        alpha_eff = torch.where(contact > 0.5, alpha * posterior, alpha)
        alpha_eff = torch.where(sure_fg > 0.5, torch.ones_like(alpha_eff), alpha_eff)
        alpha_eff = torch.where(sure_bg > 0.5, torch.zeros_like(alpha_eff), alpha_eff).clamp(0, 1)
        if not return_aux:
            return alpha_eff
        return alpha_eff, {
            "face_unknown_all": face_unknown_all,
            "face_contact_actual": contact,
            "false_positive_proxy": false_positive_proxy,
            "hair_hypothesis_rgb": hair_hypothesis,
            "face_hypothesis_rgb": face_hypothesis,
            "hair_posterior_evidence": evidence,
            "connectivity_prior": connectivity,
            "posterior_valid": valid,
            "posterior_propagation_support": prop_support,
            "hair_posterior": posterior,
            "alpha_eff": alpha_eff,
            "alpha_calibration_delta": alpha_eff - alpha,
        }


__all__ = ["FaceSideAlphaCalibratorV833"]
