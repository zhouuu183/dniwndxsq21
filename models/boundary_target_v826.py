"""Shared V2.26 boundary target algebra.

The helper functions intentionally contain no projector state.  Keeping the
target construction here prevents train and inference code from drifting apart.
"""

from __future__ import annotations

import torch


def _scalar(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.dim() == 0:
        value = value.view(1, 1, 1, 1)
    elif value.dim() == 1:
        value = value.view(-1, 1, 1, 1)
    elif value.dim() == 3:
        value = value.unsqueeze(1)
    if value.shape[0] not in (1, reference.shape[0]):
        raise ValueError("scalar boundary condition has an incompatible batch size")
    return value


def build_desired_edge_ab_target(
    base_ab: torch.Tensor,
    target_ref_ab: torch.Tensor,
    edge_chroma_strength: float = 0.70,
) -> torch.Tensor:
    """Return Base + strength * (reference - Base) in Lab AB space."""
    if base_ab.shape != target_ref_ab.shape or base_ab.dim() != 4 or base_ab.size(1) != 2:
        raise ValueError("base_ab and target_ref_ab must match [B,2,H,W]")
    strength = float(edge_chroma_strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("edge_chroma_strength must be in [0,1]")
    return base_ab + strength * (target_ref_ab - base_ab)


def build_desired_edge_l_target(
    base_l: torch.Tensor,
    reference_delta_l: torch.Tensor | float,
    edge_luma_strength: float = 0.65,
) -> torch.Tensor:
    """Return the desired signed global edge lightness target."""
    if base_l.dim() != 4 or base_l.size(1) != 1:
        raise ValueError("base_l must be [B,1,H,W]")
    delta = _scalar(reference_delta_l, base_l)
    if delta.shape[0] == 1 and base_l.shape[0] != 1:
        delta = delta.expand(base_l.shape[0], -1, -1, -1)
    return base_l + float(edge_luma_strength) * delta


def compute_reference_relative_halo_limit(
    base_l: torch.Tensor,
    reference_delta_l: torch.Tensor | float,
    edge_luma_strength: float = 0.65,
    edge_target_l_margin: float = 4.0,
) -> torch.Tensor:
    """Return the allowed edge L value under the reference-relative guard."""
    return build_desired_edge_l_target(base_l, reference_delta_l, edge_luma_strength) + float(
        edge_target_l_margin
    )

