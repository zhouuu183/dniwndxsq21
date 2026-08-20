"""Spatially selective chroma projection for Direct Anchor v2.22."""

from __future__ import annotations

import math

import torch
from torch import nn

from models.SG_IDCT_v16 import gaussian_blur2d, lab_to_rgb, rgb_to_lab


DIRECT_COLOR_ARCH_V8_5 = "direct_color_anchor_v8_5_selective_chroma"


def fixed_direct_anchor_tail(
    latent_face_tail: torch.Tensor,
    latent_color_tail: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Build the fixed strong anchor without any learned strength path."""
    if latent_face_tail.shape != latent_color_tail.shape:
        raise ValueError(
            "latent_face_tail and latent_color_tail must have the same shape, "
            f"got {tuple(latent_face_tail.shape)} and {tuple(latent_color_tail.shape)}"
        )
    if latent_face_tail.dim() != 3 or latent_face_tail.size(1) != 12:
        raise ValueError(
            "fixed_direct_anchor_tail expects [B,12,512] tails, "
            f"got {tuple(latent_face_tail.shape)}"
        )
    alpha_tensor = torch.as_tensor(alpha, device=latent_face_tail.device, dtype=latent_face_tail.dtype)
    if alpha_tensor.numel() != 1 or not torch.isfinite(alpha_tensor):
        raise ValueError(f"alpha must be one finite scalar, got {alpha!r}")
    if not 0.0 <= float(alpha_tensor) <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {float(alpha_tensor)}")
    return latent_face_tail + alpha_tensor * (latent_color_tail - latent_face_tail)


def _as_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(mask, device=reference.device, dtype=reference.dtype)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    expected_shape = (reference.size(0), 1, reference.size(2), reference.size(3))
    if mask.shape != expected_shape:
        raise ValueError(
            f"mask shape must be [B,1,H,W] or [B,H,W], got {tuple(mask.shape)} "
            f"for image {tuple(reference.shape)}"
        )
    if mask.size(1) != 1:
        raise ValueError(f"mask must have one channel, got {tuple(mask.shape)}")
    return mask.float().clamp(0, 1)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (value * mask).flatten(1).sum(dim=1) / denominator


def _masked_channel_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum(dim=(2, 3)).clamp_min(1.0)
    return (value * mask).sum(dim=(2, 3)) / denominator


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


class SelectiveHairColorProjectorV822(nn.Module):
    """Project a strong anchor's useful chroma into safe hair regions.

    The module deliberately has only four trainable scalar calibration values.
    Spatial protection and luminance suppression are architectural, so losses
    cannot solve preservation by shrinking a global latent alpha.
    """

    PARAMETER_BOUNDS = {
        "parallel_gain": (0.85, 1.10),
        "orth_keep": (0.00, 0.25),
        "boundary_strength": (0.05, 0.50),
        "luma_strength": (0.00, 0.35),
    }

    def __init__(
        self,
        *,
        target_dir_min_ab: float = 1.5,
        edge_luma_margin: float = 2.5,
        halo_scale: float = 4.0,
        parallel_gain_init: float = 1.0,
        orth_keep_init: float = 0.10,
        boundary_strength_init: float = 0.22,
        luma_strength_init: float = 0.12,
    ):
        super().__init__()
        self.target_dir_min_ab = float(target_dir_min_ab)
        self.edge_luma_margin = float(edge_luma_margin)
        self.halo_scale = float(halo_scale)
        if self.target_dir_min_ab <= 0 or self.halo_scale <= 0:
            raise ValueError("target_dir_min_ab and halo_scale must be positive")
        initial_values = {
            "parallel_gain": parallel_gain_init,
            "orth_keep": orth_keep_init,
            "boundary_strength": boundary_strength_init,
            "luma_strength": luma_strength_init,
        }
        for name, value in initial_values.items():
            low, high = self.PARAMETER_BOUNDS[name]
            if not low <= float(value) <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
            raw = nn.Parameter(torch.tensor(_inverse_sigmoid(
                (float(value) - low) / (high - low)
            )))
            setattr(self, f"raw_{name}", raw)

    @staticmethod
    def _bounded(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.sigmoid(raw)

    def parameter_values(self) -> dict[str, torch.Tensor]:
        return {
            name: self._bounded(getattr(self, f"raw_{name}"), *bounds)
            for name, bounds in self.PARAMETER_BOUNDS.items()
        }

    def parameter_values_float(self) -> dict[str, float]:
        return {name: float(value.detach().item()) for name, value in self.parameter_values().items()}

    def forward(
        self,
        *,
        base_rgb: torch.Tensor,
        anchor_rgb: torch.Tensor,
        pseudo_lab: torch.Tensor,
        target_hair_mask: torch.Tensor,
        color_supervision_mask: torch.Tensor,
        transition_ring: torch.Tensor,
        outer_background_guard: torch.Tensor,
        face_keep_mask: torch.Tensor,
        skin_protect_mask: torch.Tensor,
        satd_protect_mask: torch.Tensor,
        remove_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        if base_rgb.shape != anchor_rgb.shape or base_rgb.dim() != 4 or base_rgb.size(1) != 3:
            raise ValueError(
                "base_rgb and anchor_rgb must have matching [B,3,H,W] shapes, "
                f"got {tuple(base_rgb.shape)} and {tuple(anchor_rgb.shape)}"
            )
        if pseudo_lab.shape != base_rgb.shape:
            raise ValueError(
                f"pseudo_lab must match RGB shape {tuple(base_rgb.shape)}, got {tuple(pseudo_lab.shape)}"
            )
        base_rgb = base_rgb.float().clamp(0, 1)
        anchor_rgb = anchor_rgb.float().clamp(0, 1)
        pseudo_lab = pseudo_lab.float()
        masks = {
            "target_hair_mask": target_hair_mask,
            "color_supervision_mask": color_supervision_mask,
            "transition_ring": transition_ring,
            "outer_background_guard": outer_background_guard,
            "face_keep_mask": face_keep_mask,
            "skin_protect_mask": skin_protect_mask,
            "satd_protect_mask": satd_protect_mask,
            "remove_mask": remove_mask,
        }
        masks = {name: _as_mask(value, base_rgb) for name, value in masks.items()}
        target_hair = masks["target_hair_mask"]
        target_support = target_hair > 1e-6
        face_protect = (
            masks["face_keep_mask"] * masks["satd_protect_mask"] > 1e-6
        )
        skin_protect = masks["skin_protect_mask"] > 1e-6
        remove_protect = (masks["remove_mask"] > 1e-6) & (~target_support)
        background_protect = masks["outer_background_guard"] > 1e-6
        hard_protect = (
            face_protect | skin_protect | remove_protect | background_protect
        ).to(base_rgb.dtype)
        hair_support = (target_hair * (1.0 - hard_protect)).clamp(0, 1)
        hair_core = (
            masks["color_supervision_mask"]
            * (1.0 - hard_protect)
        ).clamp(0, 1)
        hair_edge = (masks["transition_ring"] * hair_support).clamp(0, 1)

        lab_base = rgb_to_lab(base_rgb)
        lab_anchor = rgb_to_lab(anchor_rgb)
        if not torch.isfinite(lab_base).all() or not torch.isfinite(lab_anchor).all():
            raise ValueError("base/anchor RGB to Lab produced NaN or Inf")
        base_l, base_ab = lab_base[:, :1], lab_base[:, 1:]
        anchor_l, anchor_ab = lab_anchor[:, :1], lab_anchor[:, 1:]
        anchor_delta_ab = anchor_ab - base_ab
        anchor_delta_l = anchor_l - base_l
        pseudo_delta_ab = pseudo_lab[:, 1:] - base_ab

        target_vec = _masked_channel_mean(pseudo_delta_ab, hair_core).view(-1, 2, 1, 1)
        target_norm = torch.linalg.vector_norm(target_vec, dim=1, keepdim=True)
        target_valid = (target_norm >= self.target_dir_min_ab).float()
        target_unit = target_vec / target_norm.clamp_min(1e-6)

        raw_parallel = (anchor_delta_ab * target_unit).sum(dim=1, keepdim=True)
        parallel_mag = torch.relu(raw_parallel)
        parallel_delta = parallel_mag * target_unit
        orth_delta = anchor_delta_ab - parallel_delta
        target_parallel_mag = torch.relu(
            (pseudo_delta_ab * target_unit).sum(dim=1, keepdim=True)
        )
        parallel_cap = 1.15 * target_parallel_mag + 1.0
        safe_parallel_mag = torch.minimum(parallel_mag, parallel_cap)
        overshoot = (parallel_mag > parallel_cap).float()

        values = self.parameter_values()
        safe_delta_ab = (
            safe_parallel_mag * target_unit
            + values["orth_keep"] * orth_delta
        )
        safe_delta_ab = safe_delta_ab * values["parallel_gain"] * target_valid
        halo_excess = torch.relu(
            anchor_l - torch.maximum(base_l, pseudo_lab[:, :1]) - self.edge_luma_margin
        )
        halo_gate = torch.exp(-halo_excess / self.halo_scale).clamp(0, 1)
        a_core = hair_core
        a_edge = hair_edge * values["boundary_strength"] * halo_gate
        a_chroma = (
            (a_core + a_edge).clamp(0, 1)
            * (1.0 - hard_protect)
            * target_hair
        ).clamp(0, 1)

        anchor_delta_l_low = gaussian_blur2d(anchor_delta_l, radius=7).clamp(-8.0, 8.0)
        l_out = base_l + hair_core * values["luma_strength"] * anchor_delta_l_low * target_valid
        selective_rgb = lab_to_rgb(torch.cat((l_out, base_ab + a_chroma * safe_delta_ab), dim=1))
        selective_rgb = (
            hair_support * selective_rgb + (1.0 - hair_support) * base_rgb
        ).clamp(0, 1)
        selective_rgb = (
            hard_protect * base_rgb + (1.0 - hard_protect) * selective_rgb
        ).clamp(0, 1)
        if not torch.isfinite(selective_rgb).all():
            raise ValueError("Selective projector produced NaN or Inf")
        if not return_aux:
            return selective_rgb

        anchor_norm = torch.linalg.vector_norm(anchor_delta_ab, dim=1, keepdim=True)
        parallel_norm = torch.linalg.vector_norm(parallel_delta, dim=1, keepdim=True)
        orth_norm = torch.linalg.vector_norm(orth_delta, dim=1, keepdim=True)
        # A fractional target mask is the intended transition support, not an
        # outside-hair leak. Measure only pixels with no target support.
        outside_hair = (~target_support).to(target_hair.dtype)
        aux = {
            "hard_protect": hard_protect,
            "hair_support": hair_support,
            "hair_core": hair_core,
            "hair_edge": hair_edge,
            "halo_gate": halo_gate,
            "a_core": a_core,
            "a_edge": a_edge,
            "parallel_gain": values["parallel_gain"].expand(base_rgb.size(0)),
            "orth_keep": values["orth_keep"].expand(base_rgb.size(0)),
            "boundary_strength": values["boundary_strength"].expand(base_rgb.size(0)),
            "luma_strength": values["luma_strength"].expand(base_rgb.size(0)),
            "parallel_fraction": _masked_mean(
                parallel_norm / anchor_norm.clamp_min(1e-6), hair_support
            ),
            "orthogonal_fraction": _masked_mean(
                orth_norm / anchor_norm.clamp_min(1e-6), hair_support
            ),
            "negative_parallel_fraction": _masked_mean(
                (raw_parallel < 0).float(), hair_support
            ),
            "overshoot_fraction": _masked_mean(overshoot, hair_support),
            "mean_A_core": _masked_mean(a_core, target_hair),
            "mean_A_edge": _masked_mean(a_edge, target_hair),
            "mean_halo_gate": _masked_mean(halo_gate, hair_edge),
            "target_direction_norm": target_norm.flatten(),
            "target_direction_valid": target_valid.flatten(),
            "outside_hair_max_abs_delta": (
                (selective_rgb - base_rgb).abs() * outside_hair
            ).flatten(1).amax(dim=1),
            "hard_protect_max_abs_delta": (
                (selective_rgb - base_rgb).abs() * hard_protect
            ).flatten(1).amax(dim=1),
            "outer_bg_keep_l1": _masked_mean(
                (selective_rgb - base_rgb).abs().mean(dim=1), masks["outer_background_guard"]
            ),
            "face_keep_l1": _masked_mean(
                (selective_rgb - base_rgb).abs().mean(dim=1),
                masks["face_keep_mask"] * masks["satd_protect_mask"],
            ),
        }
        return selective_rgb, aux
