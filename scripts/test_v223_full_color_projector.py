import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.SG_IDCT_v16 import gaussian_blur2d, lab_to_rgb, rgb_to_lab
from models.selective_color_projector_v823 import (
    FullColorToneSelectiveProjectorV823,
    fixed_direct_anchor_tail,
)


def build_masks(batch=1, size=48):
    target = torch.zeros(batch, 1, size, size)
    target[:, :, 6:42, 6:42] = 1.0
    core = torch.zeros_like(target)
    core[:, :, 11:37, 11:37] = 1.0
    ring = (target - core).clamp(0, 1)
    zeros = torch.zeros_like(target)
    return target, core, ring, zeros


def lab_image(lightness, ab=(8.0, 12.0), batch=1, size=48):
    if not isinstance(lightness, torch.Tensor):
        lightness = torch.full((batch, 1, size, size), float(lightness))
    a = torch.full_like(lightness, float(ab[0]))
    b = torch.full_like(lightness, float(ab[1]))
    return torch.cat((lightness, a, b), dim=1)


def run_projector(projector, base_lab, anchor_lab, pseudo_lab, delta_l, masks=None):
    batch, _, size, _ = base_lab.shape
    target, core, ring, zeros = masks or build_masks(batch, size)
    return projector(
        base_rgb=lab_to_rgb(base_lab),
        anchor_rgb=lab_to_rgb(anchor_lab),
        pseudo_lab=pseudo_lab,
        reference_delta_l=torch.as_tensor(delta_l).expand(batch),
        target_hair_mask=target,
        color_supervision_mask=core,
        transition_ring=ring,
        outer_background_guard=zeros.clone(),
        face_keep_mask=zeros.clone(),
        skin_protect_mask=zeros.clone(),
        satd_protect_mask=zeros.clone(),
        remove_mask=zeros.clone(),
        return_aux=True,
    )


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def main():
    torch.manual_seed(223)
    projector = FullColorToneSelectiveProjectorV823()
    assert sum(parameter.numel() for parameter in projector.parameters()) == 0

    face_tail = torch.randn(2, 12, 512)
    color_tail = torch.randn(2, 12, 512)
    expected = face_tail + 0.9 * (color_tail - face_tail)
    assert torch.allclose(fixed_direct_anchor_tail(face_tail, color_tail, 0.9), expected)

    base_lab = lab_image(55.0)
    base_rgb = lab_to_rgb(base_lab)
    identity, identity_aux = run_projector(
        projector, base_lab, base_lab, base_lab, 0.0
    )
    assert float((identity - base_rgb).abs().max()) <= 1e-4
    assert float(identity_aux["outside_hair_max_abs_delta"].max()) <= 1e-7

    dark_base = lab_image(70.0, ab=(5.0, 8.0))
    dark_anchor = lab_image(30.0, ab=(18.0, 20.0))
    dark_pseudo = lab_image(30.0, ab=(18.0, 20.0))
    dark_output, dark_aux = run_projector(
        projector, dark_base, dark_anchor, dark_pseudo, -40.0
    )
    dark_l = rgb_to_lab(dark_output)[:, :1]
    assert float(masked_mean(dark_l, dark_aux["hair_core"])) < 42.0

    light_base = lab_image(30.0, ab=(18.0, 20.0))
    light_anchor = lab_image(70.0, ab=(5.0, 8.0))
    light_pseudo = lab_image(70.0, ab=(5.0, 8.0))
    light_output, light_aux = run_projector(
        projector, light_base, light_anchor, light_pseudo, 40.0
    )
    light_l = rgb_to_lab(light_output)[:, :1]
    assert float(masked_mean(light_l, light_aux["hair_core"])) > 58.0

    stripe = ((torch.arange(48) % 2) * 2 - 1).float().view(1, 1, 1, 48) * 4.0
    textured_l = torch.full((1, 1, 48, 48), 60.0) + stripe
    texture_base = lab_image(textured_l, ab=(6.0, 10.0))
    texture_anchor = lab_image(35.0, ab=(15.0, 18.0))
    texture_pseudo = lab_image(35.0, ab=(15.0, 18.0))
    texture_output, texture_aux = run_projector(
        projector, texture_base, texture_anchor, texture_pseudo, -25.0
    )
    output_l = rgb_to_lab(texture_output)[:, :1]
    base_l = rgb_to_lab(lab_to_rgb(texture_base))[:, :1]
    hp_output = output_l - gaussian_blur2d(output_l, radius=3)
    hp_base = base_l - gaussian_blur2d(base_l, radius=3)
    selected = texture_aux["hair_core"] > 0.5
    correlation = torch.corrcoef(
        torch.stack((hp_output[selected], hp_base[selected]))
    )[0, 1]
    assert float(correlation) > 0.90

    target, core, ring, zeros = build_masks()
    halo_anchor = lab_image(40.0, ab=(22.0, 18.0))
    halo_anchor[:, :1] = torch.where(
        ring.bool(), torch.full_like(halo_anchor[:, :1], 95.0), halo_anchor[:, :1]
    )
    halo_output, halo_aux = run_projector(
        projector,
        lab_image(50.0, ab=(5.0, 8.0)),
        halo_anchor,
        lab_image(40.0, ab=(22.0, 18.0)),
        -10.0,
        (target, core, ring, zeros),
    )
    halo_lab = rgb_to_lab(halo_output)
    assert float(halo_lab[:, :1][ring.bool()].max()) <= 53.0
    edge_chroma_delta = torch.linalg.vector_norm(
        halo_lab[:, 1:] - lab_image(50.0, ab=(5.0, 8.0))[:, 1:],
        dim=1,
        keepdim=True,
    )
    assert float(masked_mean(edge_chroma_delta, ring)) > 0.5
    assert float(halo_aux["chroma_transfer_weight"][ring.bool()].mean()) > 0.0

    protected_masks = list(build_masks())
    protected = torch.zeros_like(protected_masks[0])
    protected[:, :, 14:20, 14:20] = 0.75
    protected_output, protected_aux = projector(
        base_rgb=lab_to_rgb(light_base),
        anchor_rgb=torch.ones_like(light_base),
        pseudo_lab=light_pseudo,
        reference_delta_l=torch.tensor([40.0]),
        target_hair_mask=protected_masks[0],
        color_supervision_mask=protected_masks[1],
        transition_ring=protected_masks[2],
        outer_background_guard=protected_masks[3],
        face_keep_mask=protected_masks[3],
        skin_protect_mask=protected,
        satd_protect_mask=protected_masks[3],
        remove_mask=protected_masks[3],
        return_aux=True,
    )
    protected_base = lab_to_rgb(light_base)
    assert float(
        ((protected_output - protected_base).abs() * protected_aux["hard_protect"]).max()
    ) <= 1e-7
    assert float(protected_aux["hard_protect_max_abs_delta"].max()) <= 1e-7
    assert float(protected_aux["outside_hair_max_abs_delta"].max()) <= 1e-7

    soft_edge_protected = build_masks()
    soft_edge_protected[2][:, :, 6:11, 6:42] = 0.28
    soft_skin = torch.zeros_like(soft_edge_protected[0])
    soft_skin[:, :, 6:11, 6:42] = 0.45
    soft_edge_output, soft_edge_aux = projector(
        base_rgb=lab_to_rgb(light_base),
        anchor_rgb=lab_to_rgb(light_anchor),
        pseudo_lab=light_pseudo,
        reference_delta_l=torch.tensor([40.0]),
        target_hair_mask=soft_edge_protected[0],
        color_supervision_mask=soft_edge_protected[1],
        transition_ring=soft_edge_protected[2],
        outer_background_guard=soft_edge_protected[3],
        face_keep_mask=soft_edge_protected[3],
        skin_protect_mask=soft_skin,
        satd_protect_mask=soft_edge_protected[3],
        remove_mask=soft_edge_protected[3],
        return_aux=True,
    )
    soft_edge_region = soft_edge_protected[2] > 0
    assert float(soft_edge_aux["hard_protect"][soft_edge_region].max()) == 0.0
    assert float(
        (soft_edge_output - lab_to_rgb(light_base)).abs()
        .expand_as(soft_edge_output)[soft_edge_region.expand_as(soft_edge_output)]
        .max()
    ) > 1e-6
    print("v2.23 full-color projector tests passed")


if __name__ == "__main__":
    main()
