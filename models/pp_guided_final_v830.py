"""V2.30 completed-image soft interior seam."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.pp_guided_edge_recolor_v830 import PPGuidedEdgeRecolorV830


def _erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    return 1.0 - F.max_pool2d(
        1.0 - mask, 2 * width + 1, stride=1, padding=width
    )


def build_soft_core_weight_v830(hair_mask: torch.Tensor, width: int = 5) -> torch.Tensor:
    """Select between completed hair renderings strictly inside the silhouette."""
    hard = (hair_mask >= 0.5).to(hair_mask.dtype)
    layers = [_erode(hard, offset) for offset in range(1, width + 1)]
    return torch.stack(layers, dim=0).mean(dim=0).clamp(0, 1)


class PPGuidedFinalV830(nn.Module):
    def __init__(
        self,
        core_seam_width: int = 5,
        pp_tone_radius: int = 5,
        tone_propagation_radius: int = 7,
        face_contact_width: int = 4,
        confidence_temperature: float = 8.0,
    ):
        super().__init__()
        self.core_seam_width = int(core_seam_width)
        self.recolor = PPGuidedEdgeRecolorV830(
            pp_tone_radius=pp_tone_radius,
            tone_propagation_radius=tone_propagation_radius,
            face_contact_width=face_contact_width,
            confidence_temperature=confidence_temperature,
        )

    def config_dict(self) -> dict[str, object]:
        return {
            "core_seam_width_output": self.core_seam_width,
            "visible_contour_carrier": "ORIGINAL_PP",
            "binary_pp_lock": False,
            "soft_core_weight_is_opacity": False,
            **self.recolor.config_dict(),
        }

    def forward(
        self,
        *,
        core_carrier_rgb: torch.Tensor,
        pp_original_rgb: torch.Tensor,
        base_rgb: torch.Tensor,
        target_low_rgb: torch.Tensor,
        repaired_hair_mask: torch.Tensor,
        source_face_mask: torch.Tensor,
        final_unlock_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        output_size = pp_original_rgb.shape[-2:]
        core = F.interpolate(
            core_carrier_rgb, size=output_size, mode="bicubic", align_corners=False
        ).clamp(0, 1)
        repaired_output = F.interpolate(
            repaired_hair_mask, size=output_size, mode="bilinear", align_corners=False
        )
        repaired_hard = (repaired_output >= 0.5).to(core.dtype)
        corrected_pp, recolor_aux = self.recolor(
            pp_rgb=pp_original_rgb,
            base_rgb=base_rgb,
            target_low_rgb=target_low_rgb,
            repaired_hair_mask=repaired_hair_mask,
            source_face_mask=source_face_mask,
            final_unlock_mask=final_unlock_mask,
            return_aux=True,
        )
        # This weight selects between two completed hair renderings inside the
        # hair interior. It is not an alpha matte or foreground opacity.
        soft_core_weight = build_soft_core_weight_v830(
            repaired_hard, self.core_seam_width
        )
        unlock = None
        if final_unlock_mask is not None:
            unlock = F.interpolate(
                final_unlock_mask, size=output_size, mode="nearest"
            ).clamp(0, 1)
            soft_core_weight = soft_core_weight * (1.0 - unlock)
        final = soft_core_weight * core + (1.0 - soft_core_weight) * corrected_pp
        if unlock is not None:
            final = torch.where(unlock.expand_as(final) >= 0.5, pp_original_rgb, final)
        final = final.clamp(0, 1)
        if not torch.isfinite(final).all():
            raise ValueError("V2.30 final seam produced NaN or Inf")
        if not return_aux:
            return final
        contour = repaired_hard - _erode(repaired_hard, 1)
        return final, {
            **recolor_aux,
            "core_carrier_output": core,
            "soft_core_weight": soft_core_weight,
            "completed_image_weight_sum_error": (
                soft_core_weight + (1.0 - soft_core_weight) - 1.0
            ).abs().amax(),
            "visible_contour_band": contour,
            "visible_contour_core_weight_max": (
                soft_core_weight * contour
            ).flatten(1).amax(dim=1),
            "seam_residual": final - pp_original_rgb,
            "final_rgb": final,
        }


__all__ = ["PPGuidedFinalV830", "build_soft_core_weight_v830"]
