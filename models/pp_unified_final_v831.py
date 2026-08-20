"""V2.31 Original-PP unified recolor and alpha-aware decontamination."""

from __future__ import annotations

import torch
from torch import nn

from models.local_background_replacement_v831 import LocalBackgroundReplacementV831
from models.pp_unified_recolor_v831 import PPUnifiedRecolorV831


class PPUnifiedFinalV831(nn.Module):
    def __init__(
        self,
        tone_radius: int = 9,
        tone_propagation_radius: int = 15,
        context_radius: int = 9,
        transition_expand: int = 2,
        enable_phase_c: bool = True,
    ):
        super().__init__()
        self.enable_phase_c = bool(enable_phase_c)
        self.recolor = PPUnifiedRecolorV831(
            tone_radius=tone_radius,
            tone_propagation_radius=tone_propagation_radius,
        )
        self.background = LocalBackgroundReplacementV831(
            context_radius=context_radius,
            transition_expand=transition_expand,
        )

    def config_dict(self) -> dict[str, object]:
        return {
            "architecture": "hires_vitmatte_pp_unified_recolor_v831",
            "phase_c_enabled": self.enable_phase_c,
            "final_binary_hair_mask": False,
            "manual_face_classifier": False,
            "topology_repair_final": False,
            **self.recolor.config_dict(),
            **self.background.config_dict(),
        }

    def forward(
        self,
        *,
        pp_original_rgb: torch.Tensor,
        base_rgb: torch.Tensor,
        v226_rgb: torch.Tensor,
        alpha_hr: torch.Tensor,
        sure_fg: torch.Tensor,
        unknown: torch.Tensor,
        source_face_mask: torch.Tensor,
        source_subject_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        phase_b, recolor_aux = self.recolor(
            pp_rgb=pp_original_rgb,
            v226_rgb=v226_rgb,
            alpha_hr=alpha_hr,
            sure_fg=sure_fg,
            return_aux=True,
        )
        phase_c, background_aux = self.background(
            phase_b_rgb=phase_b,
            pp_rgb=pp_original_rgb,
            base_rgb=base_rgb,
            alpha_hr=alpha_hr,
            unknown=unknown,
            source_face_mask=source_face_mask,
            source_subject_mask=source_subject_mask,
            return_aux=True,
        )
        final = phase_c if self.enable_phase_c else phase_b
        if not torch.isfinite(final).all():
            raise ValueError("V2.31 final produced NaN or Inf")
        if not return_aux:
            return final
        return final, {
            **recolor_aux,
            **background_aux,
            "phase_b_rgb": phase_b,
            "phase_c_rgb": phase_c,
            "final_rgb": final,
            "binary_final_alpha_use": torch.tensor(False, device=final.device),
        }


__all__ = ["PPUnifiedFinalV831"]
