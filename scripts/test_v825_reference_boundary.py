"""Synthetic tests for V2.25 reference-conditioned boundary targets."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.selective_color_projector_v825 import ReferenceConditionedBoundaryProjectorV825


def render(*, delta_ab=(10.0, 0.0), anchor_l_shift=0.0, reference_l_shift=0.0, target_ab=(8.0, 0.0)):
    projector = ReferenceConditionedBoundaryProjectorV825()
    base = torch.full((1, 3, 32, 32), 0.45)
    base_lab = rgb_to_lab(base)
    anchor_lab = base_lab.clone()
    anchor_lab[:, :1] += anchor_l_shift
    anchor_lab[:, 1:2] += delta_ab[0]
    anchor_lab[:, 2:3] += delta_ab[1]
    anchor = lab_to_rgb(anchor_lab).clamp(0, 1)
    target_ref_ab = base_lab[:, 1:].clone()
    target_ref_ab[:, :1] += target_ab[0]
    target_ref_ab[:, 1:] += target_ab[1]
    one = torch.ones(1, 1, 32, 32)
    zero = torch.zeros_like(one)
    return projector(
        base_rgb=base, anchor_rgb=anchor, pseudo_lab=base_lab,
        target_ref_ab=target_ref_ab, reference_delta_l=torch.tensor([reference_l_shift]),
        target_hair_mask=one, target_hair_eroded=zero,
        outer_background_guard=zero, face_keep_mask=zero, skin_protect_mask=zero,
        satd_protect_mask=zero, remove_mask=zero, return_aux=True,
    )


def main():
    _, aux = render()
    delta_ab = (aux["ab_output"] - aux["base_lab"][:, 1:]).norm(dim=1).mean()
    assert delta_ab >= 4.0, delta_ab

    _, aux = render(anchor_l_shift=20.0, reference_l_shift=20.0)
    bright_shift = (aux["luma_output"] - aux["base_lab"][:, :1]).mean()
    assert bright_shift >= 9.0, bright_shift

    _, aux = render(anchor_l_shift=-20.0, reference_l_shift=-20.0)
    dark_shift = (aux["luma_output"] - aux["base_lab"][:, :1]).mean()
    assert dark_shift <= -9.0, dark_shift

    _, aux = render(delta_ab=(0.0, 0.0), target_ab=(0.0, 0.0))
    no_edit_ab = (aux["ab_output"] - aux["base_lab"][:, 1:]).abs().max()
    assert no_edit_ab <= 1e-6, no_edit_ab
    print("test_v825_reference_boundary: PASS")


if __name__ == "__main__":
    main()
