"""V2.26 metric-aligned projector with bounded AB-only compensation."""

from __future__ import annotations

import torch

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.boundary_target_v826 import build_desired_edge_ab_target
from models.selective_color_projector_v825 import (
    FULL_COLOR_ARCH_V8_8,
    ReferenceConditionedBoundaryProjectorV825,
    fixed_direct_anchor_tail,
)

FULL_COLOR_ARCH_V8_9 = "boundary_target_metric_aligned_v826"
fixed_direct_anchor_tail_v826 = fixed_direct_anchor_tail
V226_COMPENSATION_GAMMAS = (0.0, 0.25, 0.50, 0.75)
V226_DEFICIT_CAP_RATIO = 0.40
V226_DEFICIT_ABS_CAP = 6.0
V226_TARGET_OVERSHOOT_MARGIN_AB = 1.0


def load_v226_projector_checkpoint(checkpoint_path, device):
    """Restore and validate the frozen V2.26 gamma=.25 projector."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("arch") != FULL_COLOR_ARCH_V8_9 or checkpoint.get("version") != "v2.26":
        raise RuntimeError("V227_CHECKPOINT_CONFIG_MISMATCH: expected a V2.26 checkpoint")
    config = checkpoint.get("projector_config", {})
    gamma = float(config.get("compensation_gamma", -1.0))
    if gamma != 0.25:
        raise RuntimeError(
            f"V227_CHECKPOINT_CONFIG_MISMATCH: compensation_gamma={gamma}, expected 0.25"
        )
    projector = BoundaryTargetAlignedProjectorV826(
        target_dir_min_ab=config.get("target_dir_min_ab", 1.5),
        luma_low_radius=config.get("luma_low_radius", 9),
        max_low_l_shift=config.get("max_low_l_shift", 35.0),
        edge_chroma_strength=config.get("edge_chroma_strength", 0.70),
        edge_luma_strength=config.get("edge_luma_strength", 0.65),
        edge_luma_margin=config.get("edge_luma_margin", 2.5),
        edge_target_l_margin=config.get("edge_target_l_margin", 4.0),
        orth_keep=config.get("orth_keep", 0.10),
        hard_protect_threshold=config.get("hard_protect_threshold", 0.50),
        compensation_gamma=gamma,
    ).to(device).eval()
    projector.load_state_dict(checkpoint.get("projector_state_dict", {}), strict=True)
    return projector, checkpoint


def apply_v226_compensation_from_cached_v225(
    *,
    base_rgb: torch.Tensor,
    v225_rgb: torch.Tensor,
    v225_luma_output: torch.Tensor,
    v225_ab_output: torch.Tensor,
    target_ref_ab: torch.Tensor,
    hair_edge: torch.Tensor,
    hair_membership: torch.Tensor,
    hard_protect: torch.Tensor,
    edge_chroma_strength: float = 0.70,
    compensation_gamma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the V2.26 residual without rerunning preprocessing or StyleGAN."""
    gamma = float(compensation_gamma)
    if gamma not in V226_COMPENSATION_GAMMAS:
        raise ValueError(f"compensation_gamma must be one of {V226_COMPENSATION_GAMMAS}")
    base_rgb = base_rgb.float().clamp(0, 1)
    base_ab = rgb_to_lab(base_rgb)[:, 1:]
    desired_ab = build_desired_edge_ab_target(
        base_ab, target_ref_ab.float(), edge_chroma_strength
    )
    desired_delta = desired_ab - base_ab
    desired_norm = torch.linalg.vector_norm(desired_delta, dim=1, keepdim=True)
    desired_unit = desired_delta / desired_norm.clamp_min(1e-6)
    current_delta = v225_ab_output - base_ab
    current_parallel = torch.relu((current_delta * desired_unit).sum(dim=1, keepdim=True))
    raw_deficit = torch.relu(desired_norm - current_parallel)
    bounded_deficit = torch.minimum(
        raw_deficit,
        torch.minimum(
            V226_DEFICIT_CAP_RATIO * desired_norm,
            raw_deficit.new_tensor(V226_DEFICIT_ABS_CAP),
        ),
    )
    compensation_mag = gamma * bounded_deficit
    compensated_ab = v225_ab_output + hair_edge * compensation_mag * desired_unit
    new_parallel = ((compensated_ab - base_ab) * desired_unit).sum(dim=1, keepdim=True)
    max_parallel = desired_norm + V226_TARGET_OVERSHOOT_MARGIN_AB
    compensated_ab = compensated_ab - hair_edge * torch.relu(
        new_parallel - max_parallel
    ) * desired_unit
    if gamma == 0.0:
        output = v225_rgb
    else:
        candidate = lab_to_rgb(torch.cat((v225_luma_output, compensated_ab), dim=1))
        active = hair_membership > 1e-6
        output = torch.where(active.expand_as(candidate), candidate, base_rgb)
        output = (hard_protect * base_rgb + (1.0 - hard_protect) * output).clamp(0, 1)
    if not torch.isfinite(output).all():
        raise ValueError("V2.26 cached compensation produced NaN or Inf")
    return output, {
        "desired_edge_ab": desired_ab,
        "desired_edge_ab_norm": desired_norm,
        "edge_anchor_deficit_mag": raw_deficit,
        "edge_bounded_deficit_mag": bounded_deficit,
        "edge_compensation_mag": compensation_mag,
        "edge_compensation_ab": compensated_ab - v225_ab_output,
        "ab_output": compensated_ab,
    }


class BoundaryTargetAlignedProjectorV826(ReferenceConditionedBoundaryProjectorV825):
    """V2.25 output plus optional, bounded edge-only AB residual."""

    def __init__(self, *, compensation_gamma: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.compensation_gamma = float(compensation_gamma)
        if self.compensation_gamma not in V226_COMPENSATION_GAMMAS:
            raise ValueError(
                f"compensation_gamma must be one of {V226_COMPENSATION_GAMMAS}, "
                f"got {self.compensation_gamma}"
            )

    def config_dict(self):
        config = dict(super().config_dict())
        config.update({
            "compensation_gamma": self.compensation_gamma,
            "deficit_cap_ratio": V226_DEFICIT_CAP_RATIO,
            "deficit_abs_cap": V226_DEFICIT_ABS_CAP,
            "target_overshoot_margin_ab": V226_TARGET_OVERSHOOT_MARGIN_AB,
        })
        return config

    @torch.no_grad()
    def forward(self, *, base_rgb, anchor_rgb, pseudo_lab, target_ref_ab,
                reference_delta_l, target_hair_mask, target_hair_eroded,
                outer_background_guard, face_keep_mask, skin_protect_mask,
                satd_protect_mask, remove_mask, return_aux=False):
        # The parent is the canonical V2.25 path.  Returning its tensor directly
        # for gamma=0 is what makes the parity contract exact.
        selective_rgb, aux = super().forward(
            base_rgb=base_rgb, anchor_rgb=anchor_rgb, pseudo_lab=pseudo_lab,
            target_ref_ab=target_ref_ab, reference_delta_l=reference_delta_l,
            target_hair_mask=target_hair_mask, target_hair_eroded=target_hair_eroded,
            outer_background_guard=outer_background_guard, face_keep_mask=face_keep_mask,
            skin_protect_mask=skin_protect_mask, satd_protect_mask=satd_protect_mask,
            remove_mask=remove_mask, return_aux=True,
        )
        output, compensation_aux = apply_v226_compensation_from_cached_v225(
            base_rgb=base_rgb,
            v225_rgb=selective_rgb,
            v225_luma_output=aux["luma_output"],
            v225_ab_output=aux["ab_output"],
            target_ref_ab=target_ref_ab,
            hair_edge=aux["hair_edge"],
            hair_membership=aux["hair_membership"],
            hard_protect=aux["hard_protect"],
            edge_chroma_strength=self.edge_chroma_strength,
            compensation_gamma=self.compensation_gamma,
        )
        aux = dict(aux)
        aux.update({
            "v225_rgb": selective_rgb,
            "compensation_gamma": output.new_full((output.size(0),), self.compensation_gamma),
            **compensation_aux,
        })
        return (output, aux) if return_aux else output


__all__ = [
    "FULL_COLOR_ARCH_V8_9",
    "V226_COMPENSATION_GAMMAS",
    "BoundaryTargetAlignedProjectorV826",
    "apply_v226_compensation_from_cached_v225",
    "load_v226_projector_checkpoint",
    "fixed_direct_anchor_tail",
    "fixed_direct_anchor_tail_v826",
]
