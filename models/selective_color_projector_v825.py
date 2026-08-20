"""Reference-conditioned boundary target projector for V2.25."""

from __future__ import annotations

import torch
from torch import nn

from models.SG_IDCT_v16 import gaussian_blur2d, lab_to_rgb, rgb_to_lab
from models.boundary_masks_v824 import build_boundary_masks_v824
from models.color_condition_v8 import masked_robust_stats
from models.boundary_target_v826 import (
    build_desired_edge_l_target,
    compute_reference_relative_halo_limit,
)


FULL_COLOR_ARCH_V8_8 = "reference_conditioned_boundary_target_v825"


def fixed_direct_anchor_tail(
    latent_face_tail: torch.Tensor,
    latent_color_tail: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if latent_face_tail.shape != latent_color_tail.shape:
        raise ValueError("latent_face_tail and latent_color_tail must have the same shape")
    if latent_face_tail.dim() != 3 or latent_face_tail.size(1) != 12:
        raise ValueError(
            "fixed_direct_anchor_tail expects [B,12,512], "
            f"got {tuple(latent_face_tail.shape)}"
        )
    alpha_tensor = torch.as_tensor(
        alpha, device=latent_face_tail.device, dtype=latent_face_tail.dtype
    )
    if alpha_tensor.numel() != 1 or not torch.isfinite(alpha_tensor):
        raise ValueError(f"alpha must be one finite scalar, got {alpha!r}")
    if not 0.0 <= float(alpha_tensor) <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {float(alpha_tensor)}")
    return latent_face_tail + alpha_tensor * (latent_color_tail - latent_face_tail)


def _as_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(mask, device=reference.device, dtype=reference.dtype)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    expected = (reference.size(0), 1, reference.size(2), reference.size(3))
    if tuple(mask.shape) != expected:
        raise ValueError(f"mask must have shape {expected}, got {tuple(mask.shape)}")
    return mask.float().clamp(0, 1)


def _ensure_scalar_map(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.dim() == 3:
        value = value.unsqueeze(1)
    expected = (reference.size(0), 1, reference.size(2), reference.size(3))
    if tuple(value.shape) != expected:
        raise ValueError(f"scalar map must have shape {expected}, got {tuple(value.shape)}")
    return value


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value = _ensure_scalar_map(value, mask)
    if value.shape != mask.shape:
        raise ValueError(
            f"masked scalar map and mask must match, got {value.shape} and {mask.shape}"
        )
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (value * mask).flatten(1).sum(dim=1) / denominator


def _masked_channel_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.dim() != 4 or value.size(0) != mask.size(0):
        raise ValueError("masked channel mean expects matching [B,C,H,W] and [B,1,H,W]")
    if value.shape[-2:] != mask.shape[-2:]:
        raise ValueError("masked channel mean spatial shapes must match")
    denominator = mask.sum(dim=(2, 3)).clamp_min(1.0)
    return (value * mask).sum(dim=(2, 3)) / denominator


class ReferenceConditionedBoundaryProjectorV825(nn.Module):
    """Transfer low-frequency full color tone while preserving base detail."""

    def __init__(
        self,
        *,
        target_dir_min_ab: float = 1.5,
        luma_low_radius: int = 9,
        max_low_l_shift: float = 35.0,
        edge_chroma_strength: float = 0.70,
        edge_luma_strength: float = 0.65,
        edge_luma_margin: float = 2.5,
        edge_target_l_margin: float = 4.0,
        orth_keep: float = 0.10,
        hard_protect_threshold: float = 0.50,
    ):
        super().__init__()
        self.target_dir_min_ab = float(target_dir_min_ab)
        self.luma_low_radius = int(luma_low_radius)
        self.max_low_l_shift = float(max_low_l_shift)
        self.edge_chroma_strength = float(edge_chroma_strength)
        self.edge_luma_strength = float(edge_luma_strength)
        self.edge_luma_margin = float(edge_luma_margin)
        self.edge_target_l_margin = float(edge_target_l_margin)
        self.orth_keep = float(orth_keep)
        self.hard_protect_threshold = float(hard_protect_threshold)
        if self.target_dir_min_ab <= 0:
            raise ValueError("target_dir_min_ab must be positive")
        if self.luma_low_radius < 1 or self.max_low_l_shift <= 0:
            raise ValueError("luma_low_radius and max_low_l_shift must be positive")
        if not 0.60 <= self.edge_chroma_strength <= 0.80:
            raise ValueError("edge_chroma_strength must be in [0.60,0.80]")
        if not 0.0 <= self.edge_luma_strength <= 1.0:
            raise ValueError("edge_luma_strength must be in [0,1]")
        if not 0.0 <= self.orth_keep <= 0.25:
            raise ValueError("orth_keep must be in [0,0.25]")
        if not 0.0 < self.hard_protect_threshold <= 1.0:
            raise ValueError("hard_protect_threshold must be in (0,1]")

    def config_dict(self) -> dict[str, float | int]:
        return {
            "target_dir_min_ab": self.target_dir_min_ab,
            "luma_low_radius": self.luma_low_radius,
            "max_low_l_shift": self.max_low_l_shift,
            "edge_chroma_strength": self.edge_chroma_strength,
            "edge_luma_strength": self.edge_luma_strength,
            "edge_luma_margin": self.edge_luma_margin,
            "edge_target_l_margin": self.edge_target_l_margin,
            "orth_keep": self.orth_keep,
            "hard_protect_threshold": self.hard_protect_threshold,
            "core_luma_gain": 1.0,
            "core_chroma_gain": 1.0,
        }

    def forward(
        self,
        *,
        base_rgb: torch.Tensor,
        anchor_rgb: torch.Tensor,
        pseudo_lab: torch.Tensor,
        target_ref_ab: torch.Tensor,
        reference_delta_l: torch.Tensor,
        target_hair_mask: torch.Tensor,
        target_hair_eroded: torch.Tensor,
        outer_background_guard: torch.Tensor,
        face_keep_mask: torch.Tensor,
        skin_protect_mask: torch.Tensor,
        satd_protect_mask: torch.Tensor,
        remove_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        if base_rgb.shape != anchor_rgb.shape or base_rgb.dim() != 4 or base_rgb.size(1) != 3:
            raise ValueError("base_rgb and anchor_rgb must match [B,3,H,W]")
        if pseudo_lab.shape != base_rgb.shape:
            raise ValueError("pseudo_lab must match the RGB image shape")
        expected_target_ref = (base_rgb.size(0), 2, base_rgb.size(2), base_rgb.size(3))
        if tuple(target_ref_ab.shape) != expected_target_ref:
            raise ValueError(
                f"target_ref_ab must have shape {expected_target_ref}, got {tuple(target_ref_ab.shape)}"
            )
        base_rgb = base_rgb.float().clamp(0, 1)
        anchor_rgb = anchor_rgb.float().clamp(0, 1)
        pseudo_lab = pseudo_lab.float()
        target_ref_ab = target_ref_ab.float()
        masks = {
            "target_hair_mask": target_hair_mask,
            "target_hair_eroded": target_hair_eroded,
            "outer_background_guard": outer_background_guard,
            "face_keep_mask": face_keep_mask,
            "skin_protect_mask": skin_protect_mask,
            "satd_protect_mask": satd_protect_mask,
            "remove_mask": remove_mask,
        }
        masks = {name: _as_mask(value, base_rgb) for name, value in masks.items()}
        target_hair = masks["target_hair_mask"]
        target_support = target_hair > 1e-6
        threshold = self.hard_protect_threshold
        face_protect = masks["face_keep_mask"] * masks["satd_protect_mask"] >= threshold
        skin_protect = masks["skin_protect_mask"] >= threshold
        remove_protect = (masks["remove_mask"] >= threshold) & (~target_support)
        background_protect = masks["outer_background_guard"] >= threshold
        hard_protect = (
            face_protect | skin_protect | remove_protect | background_protect
        ).to(base_rgb.dtype)
        boundary_masks = build_boundary_masks_v824(
            target_hair_mask=target_hair,
            target_hair_eroded=masks["target_hair_eroded"],
            hard_protect=hard_protect,
        )
        hair_core = boundary_masks["core_membership"]
        hair_edge = boundary_masks["edge_membership"]
        hair_membership = boundary_masks["hair_membership"]
        hair_support = hair_membership

        lab_base = rgb_to_lab(base_rgb)
        lab_anchor = rgb_to_lab(anchor_rgb)
        if not torch.isfinite(lab_base).all() or not torch.isfinite(lab_anchor).all():
            raise ValueError("RGB to Lab produced NaN or Inf")
        base_l, base_ab = lab_base[:, :1], lab_base[:, 1:]
        anchor_l, anchor_ab = lab_anchor[:, :1], lab_anchor[:, 1:]
        anchor_delta_ab = anchor_ab - base_ab
        pseudo_delta_ab = pseudo_lab[:, 1:] - base_ab
        reference_target_delta_ab = target_ref_ab - base_ab

        target_vec = _masked_channel_mean(pseudo_delta_ab, hair_core).view(-1, 2, 1, 1)
        target_norm = torch.linalg.vector_norm(target_vec, dim=1, keepdim=True)
        target_valid = (target_norm >= self.target_dir_min_ab).float()
        target_unit_core = target_vec / target_norm.clamp_min(1e-6)
        reference_norm = torch.linalg.vector_norm(
            reference_target_delta_ab, dim=1, keepdim=True
        )
        reference_unit = reference_target_delta_ab / reference_norm.clamp_min(1e-6)
        edge_target_valid = (reference_norm >= self.target_dir_min_ab).float()
        target_unit = torch.where(
            (hair_edge > 1e-6).expand_as(reference_unit), reference_unit, target_unit_core
        )
        target_valid_map = torch.where(
            hair_edge > 1e-6, edge_target_valid, target_valid
        )
        raw_parallel = (anchor_delta_ab * target_unit).sum(dim=1, keepdim=True)
        parallel_mag = torch.relu(raw_parallel)
        parallel_delta = parallel_mag * target_unit
        orth_delta = anchor_delta_ab - parallel_delta
        target_parallel_mag = torch.relu(
            (pseudo_delta_ab * target_unit_core).sum(dim=1, keepdim=True)
        )
        core_parallel_cap = 1.15 * target_parallel_mag + 1.0
        reference_parallel_mag = torch.relu(
            (reference_target_delta_ab * target_unit).sum(dim=1, keepdim=True)
        )
        edge_parallel_cap = 1.15 * reference_parallel_mag + 1.0
        safe_parallel_core = torch.minimum(parallel_mag, core_parallel_cap)
        safe_parallel_edge = torch.minimum(parallel_mag, edge_parallel_cap)
        safe_parallel_mag = hair_core * safe_parallel_core + hair_edge * safe_parallel_edge
        safe_delta_ab = (
            safe_parallel_mag * target_unit + self.orth_keep * orth_delta
        ) * target_valid_map

        delta_l_anchor_low = gaussian_blur2d(
            anchor_l - base_l, radius=self.luma_low_radius
        )
        anchor_low_median = masked_robust_stats(
            delta_l_anchor_low, hair_core
        )["median"].view(-1, 1, 1, 1)
        reference_delta_l = torch.as_tensor(
            reference_delta_l, device=base_rgb.device, dtype=base_rgb.dtype
        ).reshape(-1)
        if reference_delta_l.numel() != base_rgb.size(0):
            raise ValueError(
                "reference_delta_l must contain one value per sample, "
                f"got {reference_delta_l.numel()} for batch {base_rgb.size(0)}"
            )
        if not torch.isfinite(reference_delta_l).all():
            raise ValueError("reference_delta_l contains NaN or Inf")
        calibrated_delta_l = (
            delta_l_anchor_low
            - anchor_low_median
            + reference_delta_l.view(-1, 1, 1, 1)
        ).clamp(-self.max_low_l_shift, self.max_low_l_shift)

        chroma_weight = (
            hair_core + self.edge_chroma_strength * hair_edge
        ).clamp(0, 1)
        luma_weight = (
            hair_core + self.edge_luma_strength * hair_edge
        ).clamp(0, 1)
        ab_out = base_ab + chroma_weight * safe_delta_ab
        l_out = base_l + luma_weight * calibrated_delta_l
        expected_edge_l = build_desired_edge_l_target(
            base_l, reference_delta_l, self.edge_luma_strength
        )
        allowed_edge_l = compute_reference_relative_halo_limit(
            base_l, reference_delta_l, self.edge_luma_strength, self.edge_target_l_margin
        )
        edge_luma_excess_before_cap = torch.relu(l_out - allowed_edge_l) * hair_edge
        l_out = l_out - hair_edge * torch.relu(l_out - allowed_edge_l)
        edge_luma_excess_after_cap = torch.relu(l_out - allowed_edge_l) * hair_edge

        candidate_rgb = lab_to_rgb(torch.cat((l_out, ab_out), dim=1))
        active = hair_membership > 1e-6
        selective_rgb = torch.where(active.expand_as(candidate_rgb), candidate_rgb, base_rgb)
        selective_rgb = (
            hard_protect * base_rgb + (1.0 - hard_protect) * selective_rgb
        ).clamp(0, 1)
        if not torch.isfinite(selective_rgb).all():
            raise ValueError("Full-color projector produced NaN or Inf")
        if not return_aux:
            return selective_rgb

        outside_hair = boundary_masks["outside_hair"]
        rgb_delta_scalar = (selective_rgb - base_rgb).abs().mean(dim=1, keepdim=True)
        anchor_norm = torch.linalg.vector_norm(anchor_delta_ab, dim=1, keepdim=True)
        parallel_norm = torch.linalg.vector_norm(parallel_delta, dim=1, keepdim=True)
        orth_norm = torch.linalg.vector_norm(orth_delta, dim=1, keepdim=True)
        aux = {
            "hard_protect": hard_protect,
            "hair_support": hair_support,
            "hair_core": hair_core,
            "hair_edge": hair_edge,
            "hair_membership": hair_membership,
            "luma_transfer_weight": luma_weight,
            "chroma_transfer_weight": chroma_weight,
            "calibrated_delta_l": calibrated_delta_l,
            "anchor_low_median": anchor_low_median.flatten(),
            "reference_delta_l": reference_delta_l,
            "target_ref_ab": target_ref_ab,
            "edge_reference_target_ab_norm": reference_norm,
            "edge_anchor_delta_ab_norm": torch.linalg.vector_norm(anchor_delta_ab, dim=1, keepdim=True),
            "edge_reference_parallel_mag": reference_parallel_mag,
            "edge_anchor_parallel_mag": parallel_mag,
            "edge_parallel_cap": edge_parallel_cap,
            "edge_parallel_cap_saturation_fraction": (
                (parallel_mag >= edge_parallel_cap - 1e-6).float()
            ),
            "edge_target_direction_valid_fraction": edge_target_valid,
            "edge_negative_parallel_fraction": (raw_parallel < 0).float(),
            "edge_pre_weight_safe_ab_norm": torch.linalg.vector_norm(
                safe_delta_ab, dim=1, keepdim=True
            ),
            "edge_post_weight_ab_norm": torch.linalg.vector_norm(
                ab_out - base_ab, dim=1, keepdim=True
            ),
            "expected_edge_l": expected_edge_l,
            "luma_output": l_out,
            "ab_output": ab_out,
            "base_lab": lab_base,
            "edge_luma_excess_before_cap": edge_luma_excess_before_cap,
            "edge_luma_excess_after_guard": edge_luma_excess_after_cap,
            "parallel_fraction": _masked_mean(
                parallel_norm / anchor_norm.clamp_min(1e-6), hair_support
            ),
            "orthogonal_fraction": _masked_mean(
                orth_norm / anchor_norm.clamp_min(1e-6), hair_support
            ),
            "negative_parallel_fraction": _masked_mean(
                (raw_parallel < 0).float(), hair_support
            ),
            "overshoot_fraction": _masked_mean(
                (
                    parallel_mag
                    > (hair_core * core_parallel_cap + hair_edge * edge_parallel_cap)
                ).float(), hair_support
            ),
            "outside_hair_max_abs_delta": (
                (selective_rgb - base_rgb).abs() * outside_hair
            ).flatten(1).amax(dim=1),
            "hard_protect_max_abs_delta": (
                (selective_rgb - base_rgb).abs() * hard_protect
            ).flatten(1).amax(dim=1),
            "outer_bg_keep_l1": _masked_mean(
                rgb_delta_scalar, masks["outer_background_guard"]
            ),
            "mean_edge_membership": _masked_mean(hair_edge, hair_edge),
            "mean_edge_chroma_weight": _masked_mean(chroma_weight, hair_edge),
            "mean_edge_luma_weight": _masked_mean(luma_weight, hair_edge),
            "mean_core_chroma_weight": _masked_mean(chroma_weight, hair_core),
            "mean_core_luma_weight": _masked_mean(luma_weight, hair_core),
            "face_keep_l1": _masked_mean(
                rgb_delta_scalar,
                (masks["face_keep_mask"] * masks["satd_protect_mask"]).clamp(0, 1),
            ),
        }
        return selective_rgb, aux
