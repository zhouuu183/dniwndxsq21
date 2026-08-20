import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.selective_color_projector_v822 import (
    SelectiveHairColorProjectorV822,
    fixed_direct_anchor_tail,
)


def build_inputs(size=32):
    base = torch.full((1, 3, size, size), 0.35)
    anchor = base.clone()
    anchor[:, 0, 7:25, 7:25] = 0.70
    anchor[:, 1, 7:25, 7:25] = 0.24
    anchor[:, 2, 7:25, 7:25] = 0.30
    pseudo = base.clone()
    pseudo[:, 0, 8:24, 8:24] = 0.66
    pseudo[:, 1, 8:24, 8:24] = 0.22
    pseudo[:, 2, 8:24, 8:24] = 0.28
    target = torch.zeros(1, 1, size, size)
    target[:, :, 6:26, 6:26] = 1.0
    core = torch.zeros_like(target)
    core[:, :, 9:23, 9:23] = 1.0
    ring = (target - core).clamp(0, 1)
    zeros = torch.zeros_like(target)
    return {
        "base_rgb": base,
        "anchor_rgb": anchor,
        "pseudo_lab": rgb_to_lab(pseudo),
        "target_hair_mask": target,
        "color_supervision_mask": core,
        "transition_ring": ring,
        "outer_background_guard": zeros.clone(),
        "face_keep_mask": zeros.clone(),
        "skin_protect_mask": zeros.clone(),
        "satd_protect_mask": zeros.clone(),
        "remove_mask": zeros.clone(),
    }


def main():
    torch.manual_seed(222)
    clipped_lab = torch.zeros(1, 3, 2, 2, requires_grad=True)
    lab_to_rgb(clipped_lab).sum().backward()
    assert clipped_lab.grad is not None
    assert torch.isfinite(clipped_lab.grad).all()

    face_tail = torch.randn(2, 12, 512)
    color_tail = torch.randn(2, 12, 512)
    strong = fixed_direct_anchor_tail(face_tail, color_tail, 0.90)
    expected = face_tail + 0.90 * (color_tail - face_tail)
    assert torch.allclose(strong, expected)
    assert torch.equal(fixed_direct_anchor_tail(face_tail, color_tail, 0.0), face_tail)

    projector = SelectiveHairColorProjectorV822()
    inputs = build_inputs()
    identity_inputs = dict(inputs)
    identity_inputs["anchor_rgb"] = identity_inputs["base_rgb"].clone()
    identity, _ = projector(**identity_inputs, return_aux=True)
    assert float((identity - inputs["base_rgb"]).abs().max()) < 2e-4

    output, aux = projector(**inputs, return_aux=True)
    assert torch.isfinite(output).all()
    outside = 1.0 - inputs["target_hair_mask"]
    assert float(((output - inputs["base_rgb"]).abs() * outside).max()) <= 1e-7
    assert float(aux["outside_hair_max_abs_delta"].max()) <= 1e-7

    soft_inputs = build_inputs()
    soft_inputs["target_hair_mask"][:, :, 6:9, 6:26] = 0.28
    soft_inputs["color_supervision_mask"][:, :, 6:9, 6:26] = 0.0
    soft_inputs["transition_ring"][:, :, 6:9, 6:26] = 0.28
    soft_output, soft_aux = projector(**soft_inputs, return_aux=True)
    strict_outside = (soft_inputs["target_hair_mask"] <= 1e-6).float()
    soft_boundary = soft_inputs["transition_ring"] > 0
    assert float(
        ((soft_output - soft_inputs["base_rgb"]).abs() * strict_outside).max()
    ) <= 1e-7
    assert float(soft_aux["outside_hair_max_abs_delta"].max()) <= 1e-7
    assert float(
        (soft_output - soft_inputs["base_rgb"]).abs()[soft_boundary.expand_as(soft_output)].max()
    ) > 1e-6

    protected_inputs = build_inputs()
    protected_inputs["face_keep_mask"][:, :, 12:18, 12:18] = 1.0
    protected_inputs["satd_protect_mask"][:, :, 12:18, 12:18] = 1.0
    protected_inputs["anchor_rgb"][:, :, 12:18, 12:18] = 1.0
    protected, protected_aux = projector(**protected_inputs, return_aux=True)
    protected_region = protected_aux["hard_protect"]
    assert float(((protected - protected_inputs["base_rgb"]).abs() * protected_region).max()) <= 1e-7
    assert float(protected_aux["hard_protect_max_abs_delta"].max()) <= 1e-7

    soft_protected_inputs = build_inputs()
    soft_protected_inputs["skin_protect_mask"][:, :, 12:18, 12:18] = 0.45
    soft_protected_inputs["anchor_rgb"][:, :, 12:18, 12:18] = 1.0
    soft_protected, soft_protected_aux = projector(
        **soft_protected_inputs, return_aux=True
    )
    soft_protected_region = soft_protected_aux["hard_protect"]
    assert float(soft_protected_region[:, :, 12:18, 12:18].min()) == 1.0
    assert float(
        (
            (soft_protected - soft_protected_inputs["base_rgb"]).abs()
            * soft_protected_region
        ).max()
    ) <= 1e-7
    assert float(soft_protected_aux["hard_protect_max_abs_delta"].max()) <= 1e-7

    remove_inside_hair_inputs = build_inputs()
    remove_inside_hair_inputs["target_hair_mask"][:, :, 7:9, 7:25] = 0.28
    remove_inside_hair_inputs["color_supervision_mask"][:, :, 7:9, 7:25] = 0.0
    remove_inside_hair_inputs["transition_ring"][:, :, 7:9, 7:25] = 0.28
    remove_inside_hair_inputs["remove_mask"][:, :, 7:9, 7:25] = 1.0
    remove_inside_output, remove_inside_aux = projector(
        **remove_inside_hair_inputs, return_aux=True
    )
    remove_inside_region = remove_inside_hair_inputs["transition_ring"] > 0
    assert float(
        remove_inside_aux["hard_protect"][remove_inside_region].max()
    ) == 0.0
    assert float(
        (remove_inside_output - remove_inside_hair_inputs["base_rgb"])
        .abs()[remove_inside_region.expand_as(remove_inside_output)]
        .max()
    ) > 1e-6

    halo_inputs = build_inputs()
    ring = halo_inputs["transition_ring"].bool().expand_as(halo_inputs["anchor_rgb"])
    halo_inputs["anchor_rgb"] = torch.where(
        ring, torch.ones_like(halo_inputs["anchor_rgb"]), halo_inputs["anchor_rgb"]
    )
    _, halo_aux = projector(**halo_inputs, return_aux=True)
    ring_gate = halo_aux["halo_gate"][halo_inputs["transition_ring"].bool()]
    assert float(ring_gate.mean()) < 0.75

    projector.zero_grad(set_to_none=True)
    train_output, _ = projector(**inputs, return_aux=True)
    (train_output - inputs["base_rgb"]).square().mean().backward()
    trainable = {name for name, parameter in projector.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "raw_parallel_gain",
        "raw_orth_keep",
        "raw_boundary_strength",
        "raw_luma_strength",
    }
    assert all(parameter.grad is not None for parameter in projector.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in projector.parameters())
    values = projector.parameter_values_float()
    assert 0.85 <= values["parallel_gain"] <= 1.10
    assert 0.00 <= values["orth_keep"] <= 0.25
    assert 0.05 <= values["boundary_strength"] <= 0.50
    assert 0.00 <= values["luma_strength"] <= 0.35
    print(f"v2.22 selective projector tests passed: {values}")


if __name__ == "__main__":
    main()
