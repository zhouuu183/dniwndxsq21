"""Synthetic contract tests for V2.24 boundary single-alpha algebra."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.boundary_masks_v824 import build_boundary_masks_v824
from models.selective_color_projector_v824 import BoundaryStableFullColorProjectorV824


def _inputs(raw: torch.Tensor, eroded: torch.Tensor, protect=None):
    base = torch.full((1, 3, 32, 32), 0.45)
    anchor = torch.full((1, 3, 32, 32), 0.62)
    pseudo = rgb_to_lab(torch.full_like(base, 0.62))
    zeros = torch.zeros_like(raw)
    return dict(
        base_rgb=base,
        anchor_rgb=anchor,
        pseudo_lab=pseudo,
        reference_delta_l=torch.tensor([8.0]),
        target_hair_mask=raw,
        target_hair_eroded=eroded,
        outer_background_guard=zeros,
        face_keep_mask=protect if protect is not None else zeros,
        skin_protect_mask=zeros,
        satd_protect_mask=torch.ones_like(raw) if protect is not None else zeros,
        remove_mask=zeros,
        return_aux=True,
    )


def main() -> None:
    projector = BoundaryStableFullColorProjectorV824()
    zero = torch.zeros(1, 1, 32, 32)
    one = torch.ones_like(zero)

    masks = build_boundary_masks_v824(target_hair_mask=one, target_hair_eroded=one)
    assert torch.allclose(masks["core_membership"], one)
    assert torch.allclose(masks["edge_membership"], zero)
    _, aux = projector(**_inputs(one, one))
    assert torch.allclose(aux["chroma_transfer_weight"], one)
    assert torch.allclose(aux["luma_transfer_weight"], one)

    _, aux = projector(**_inputs(one, zero))
    assert torch.allclose(aux["chroma_transfer_weight"], torch.full_like(one, 0.70))
    assert torch.allclose(aux["luma_transfer_weight"], torch.full_like(one, 0.65))

    soft = torch.full_like(one, 0.40)
    masks = build_boundary_masks_v824(target_hair_mask=soft, target_hair_eroded=zero)
    assert torch.allclose(masks["edge_membership"], soft)
    _, aux = projector(**_inputs(soft, zero))
    assert torch.allclose(aux["chroma_transfer_weight"], torch.full_like(one, 0.28))
    assert torch.allclose(aux["luma_transfer_weight"], torch.full_like(one, 0.26))

    outside = torch.zeros_like(one)
    output, _ = projector(**_inputs(outside, outside))
    assert (output - _inputs(outside, outside)["base_rgb"]).abs().max() <= 1e-6

    protected = torch.ones_like(one)
    output, aux = projector(**_inputs(one, one, protected))
    assert (output - _inputs(one, one, protected)["base_rgb"]).abs().max() <= 1e-6
    assert aux["hard_protect_max_abs_delta"].max() <= 1e-6

    bright_base_l = torch.full((1, 1, 8, 8), 30.0)
    bright_pseudo_l = torch.full_like(bright_base_l, 70.0)
    candidate_l = torch.full_like(bright_base_l, 68.0)
    allowed = torch.maximum(bright_base_l, bright_pseudo_l) + projector.edge_luma_margin
    assert torch.all(candidate_l <= allowed)

    train = build_boundary_masks_v824(target_hair_mask=soft, target_hair_eroded=zero)
    infer = build_boundary_masks_v824(target_hair_mask=soft, target_hair_eroded=zero)
    assert max((train[key] - infer[key]).abs().max().item() for key in train) <= 1e-6
    print("test_v824_boundary_algebra: PASS")


if __name__ == "__main__":
    main()
