"""Canonical raw-hair boundary masks shared by V2.24 train and inference."""

from __future__ import annotations

import torch


def _normalise_mask(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if reference.dim() == 3:
        reference = reference.unsqueeze(1)
    if value.dim() == 3:
        value = value.unsqueeze(1)
    expected = (reference.size(0), 1, reference.size(2), reference.size(3))
    if tuple(value.shape) != expected:
        raise ValueError(f"mask must have shape {expected}, got {tuple(value.shape)}")
    return value.float().clamp(0, 1)


def build_boundary_masks_v824(
    *,
    target_hair_mask: torch.Tensor,
    target_hair_eroded: torch.Tensor,
    hard_protect: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build non-overlapping core/inner-edge membership from raw hair masks.

    The returned edge is the *inside-hair* ring ``raw_hair - eroded_hair``.
    No transfer strength is applied here; callers apply it exactly once.
    """
    raw_hair = _normalise_mask(target_hair_mask, target_hair_mask)
    eroded = _normalise_mask(target_hair_eroded, raw_hair)
    core_membership = torch.minimum(eroded, raw_hair)
    edge_membership = (raw_hair - core_membership).clamp(0, 1)
    if hard_protect is not None:
        protect = _normalise_mask(hard_protect, raw_hair)
        editable = (1.0 - protect).clamp(0, 1)
        core_membership = core_membership * editable
        edge_membership = edge_membership * editable
    edge_membership = edge_membership * (1.0 - core_membership)
    return {
        "raw_hair": raw_hair,
        "eroded": eroded,
        "core_membership": core_membership.clamp(0, 1),
        "edge_membership": edge_membership.clamp(0, 1),
        "hair_membership": (core_membership + edge_membership).clamp(0, 1),
        "outside_hair": (1.0 - raw_hair).clamp(0, 1),
    }
