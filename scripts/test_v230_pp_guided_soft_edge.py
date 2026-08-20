"""Synthetic contracts for V2.30 PP-guided soft edge recoloring."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.achromatic_hair_carrier_v830 import AchromaticHairCarrierV830
from models.hair_topology_repair_v830 import HairTopologyRepairV830
from models.pp_guided_edge_recolor_v830 import PPGuidedEdgeRecolorV830
from models.pp_guided_final_v830 import PPGuidedFinalV830, build_soft_core_weight_v830
from models.v830_runtime_inputs import (
    PARSER_LABELS_V830,
    build_parser_regions_v830,
    build_v830_runtime_inputs,
    parser_region_audit_v830,
)
from utils.v230_metrics import aggregate_v230, classify_v230


def _chroma_ratio(image: torch.Tensor) -> float:
    return float((image[:, 0] - image[:, 1]).abs().mean().item())


def test_achromatic_hf_and_core_color():
    size = 41
    columns = torch.arange(size).view(1, 1, 1, size)
    luminance_stripes = ((columns % 2) * 2 - 1).float() * 0.08
    red_chroma = ((columns % 4 < 2) * 2 - 1).float() * 0.08
    anchor = torch.full((1, 3, size, size), 0.45)
    anchor = anchor + luminance_stripes
    anchor[:, 0:1] += red_chroma
    anchor = anchor.clamp(0, 1)
    v226 = torch.zeros_like(anchor)
    v226[:, 0] = 0.55
    v226[:, 1] = 0.35
    v226[:, 2] = 0.20
    core = torch.ones(1, 1, size, size)
    result, aux = AchromaticHairCarrierV830(low_radius=4, detail_radius=4)(
        anchor_rgb=anchor, v226_rgb=v226, repaired_hair_core=core, return_aux=True
    )
    interior = result[..., 8:-8, 8:-8]
    assert torch.allclose(interior.mean(dim=(2, 3)), v226.mean(dim=(2, 3)), atol=0.03)
    assert aux["achromatic_detail_gain"].std().item() > 0.01
    assert _chroma_ratio(interior) < _chroma_ratio(anchor[..., 8:-8, 8:-8]) + 0.25
    channel_gain = result / v226.clamp_min(1e-4)
    assert (channel_gain[:, 0] - channel_gain[:, 1]).abs().mean().item() < 1e-5


def test_topology_interior_hole_and_face_rejection():
    hair = torch.zeros(1, 1, 17, 17)
    hair[:, :, 3:14, 3:14] = 1
    hair[:, :, 8, 8] = 0
    face = torch.zeros_like(hair)
    repaired, aux = HairTopologyRepairV830(hole_radius=2, neighbor_threshold=0.70)(
        target_hair_mask=hair, source_face_mask=face, return_aux=True
    )
    assert repaired[0, 0, 8, 8].item() == 1.0
    face[:, :, 8, 8] = 1
    rejected, rejected_aux = HairTopologyRepairV830(
        hole_radius=2, neighbor_threshold=0.70
    )(target_hair_mask=hair, source_face_mask=face, return_aux=True)
    assert rejected[0, 0, 8, 8].item() == 0.0
    assert rejected_aux["rejected_face_holes"][0, 0, 8, 8].item() == 1.0
    assert aux["accepted_hole_fraction"].item() > 0


def test_pp_edge_geometry_and_far_outside_exact():
    size = 65
    pp = torch.full((1, 3, size, size), 0.2)
    ramp = torch.linspace(0, 1, 9).view(1, 1, 1, 9)
    pp[..., 24:41, 28:37] = 0.2 + 0.5 * ramp
    base = torch.full((1, 3, size, size), 0.2)
    target = torch.full((1, 3, size, size), 0.6)
    hair = torch.zeros(1, 1, size, size)
    hair[..., 20:45, 28:37] = 1
    corrected, aux = PPGuidedEdgeRecolorV830(face_contact_width=4)(
        pp_rgb=pp, base_rgb=base, target_low_rgb=target,
        repaired_hair_mask=hair, source_face_mask=torch.zeros_like(hair),
        return_aux=True,
    )
    assert aux["outside_exact_pp_max_delta"].item() == 0.0
    assert (corrected[..., :12, :] - pp[..., :12, :]).abs().max().item() == 0.0
    before_edge = (pp[..., 1:] - pp[..., :-1]).abs().sum(dim=1)
    after_edge = (corrected[..., 1:] - corrected[..., :-1]).abs().sum(dim=1)
    assert abs(int(before_edge.argmax()) - int(after_edge.argmax())) <= 1


def test_face_contact_arbitration_and_true_bang_core():
    size = 81
    base = torch.full((1, 3, size, size), 0.72)
    pp = base.clone()
    target = torch.zeros_like(pp)
    target[:, 0] = 0.20
    target[:, 1] = 0.08
    target[:, 2] = 0.05
    hair = torch.zeros(1, 1, size, size)
    hair[..., 15:66, 20:61] = 1
    pp[..., 20:61, 20:61] = target[..., 20:61, 20:61]
    face = torch.ones_like(hair)
    _, aux = PPGuidedEdgeRecolorV830(face_contact_width=4)(
        pp_rgb=pp, base_rgb=base, target_low_rgb=target,
        repaired_hair_mask=hair, source_face_mask=face, return_aux=True,
    )
    face_like_contact = aux["contact_inside"] * (pp[:, :1] > 0.6)
    hair_like_contact = aux["contact_inside"] * (pp[:, :1] < 0.3)
    assert aux["hair_tone_confidence"][face_like_contact > 0.5].mean().item() < 0.4
    assert aux["hair_tone_confidence"][hair_like_contact > 0.5].mean().item() > 0.6

    final, final_aux = PPGuidedFinalV830(core_seam_width=5)(
        core_carrier_rgb=target, pp_original_rgb=pp, base_rgb=base,
        target_low_rgb=target, repaired_hair_mask=hair,
        source_face_mask=face, return_aux=True,
    )
    assert final_aux["soft_core_weight"][0, 0, 40, 40].item() == 1.0
    assert torch.allclose(final[0, :, 40, 40], target[0, :, 40, 40], atol=1e-6)


def test_no_binary_silhouette_and_soft_weight_contract():
    size = 64
    hair = torch.zeros(1, 1, size, size)
    for row in range(12, 52):
        hair[..., row, 16:48 + (row % 2)] = 1
    weight = build_soft_core_weight_v830(hair, width=5)
    contour = hair - (1.0 - F.max_pool2d(1.0 - hair, 3, stride=1, padding=1))
    assert (weight * contour).max().item() == 0.0
    assert weight.min().item() == 0.0 and weight.max().item() == 1.0
    assert torch.unique(weight).numel() > 2

    pp = torch.zeros(1, 3, size, size)
    pp[..., :, 31:] = 0.6
    core = torch.ones_like(pp)
    final, aux = PPGuidedFinalV830(core_seam_width=5)(
        core_carrier_rgb=core, pp_original_rgb=pp, base_rgb=pp,
        target_low_rgb=core, repaired_hair_mask=hair,
        source_face_mask=torch.zeros_like(hair), return_aux=True,
    )
    visible_contour = aux["visible_contour_band"]
    assert aux["visible_contour_core_weight_max"].max().item() == 0.0
    assert (final - aux["corrected_pp_rgb"])[visible_contour.expand_as(final) > 0.5].abs().max().item() == 0.0
    assert aux["completed_image_weight_sum_error"].item() == 0.0


def test_parser_regions_and_runtime_parity():
    labels = torch.tensor([[[0, 1, 2, 8, 13, 16, 17, 18]]])
    reference = torch.zeros(1, 1, 1, 8)
    regions = build_parser_regions_v830(labels, reference)
    assert regions["source_skin_mask"][0, 0, 0, 1].item() == 1
    assert regions["source_skin_mask"][0, 0, 0, 2].item() == 0
    assert regions["source_face_mask"][0, 0, 0, 2].item() == 1
    assert regions["source_ear_mask"][0, 0, 0, 3].item() == 1
    assert regions["source_neck_mask"][0, 0, 0, 5:7].min().item() == 1
    assert regions["source_cloth_mask"][0, 0, 0, 7].item() == 1
    assert parser_region_audit_v830()["hair_label"] == PARSER_LABELS_V830["hair"]

    image = torch.zeros(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    kwargs = dict(
        base_rgb=image, anchor_rgb=image, v226_rgb=image, v229_prepp_rgb=image,
        pp_original_rgb=torch.zeros(1, 3, 32, 32), target_hair_mask=mask,
        target_hair_eroded=mask, target_hair_dilated=mask,
        parser_labels=torch.zeros(1, 1, 8, 8),
    )
    left = build_v830_runtime_inputs(**kwargs)
    right = build_v830_runtime_inputs(**kwargs)
    assert all(torch.equal(left[key], right[key]) for key in left)


def test_metric_gate_order():
    summary = {
        "median_core_ref_full": 1.0, "median_v226_core_ref_full": 1.0,
        "median_core_ref_ab": 1.0, "median_v226_core_ref_ab": 1.0,
        "median_core_luma_hf_to_anchor": 0.1, "median_v226_luma_hf_to_anchor": 1.0,
        "median_base_color_rim_fraction": 0.1, "p90_base_color_rim_fraction": 0.2,
        "median_base_color_edge_progress": 0.9,
        "median_face_boundary_contamination": 0.05, "p90_face_boundary_contamination": 0.1,
        "median_face_contact_color_leak_rgb": 0.04, "median_v229_face_contact_color_leak_rgb": 0.08,
        "median_reference_background_color_retention": 0.2,
        "median_pp_reference_background_color_retention": 0.2,
        "median_seam_hf_energy": 0.1, "median_v229_seam_hf_energy": 0.2,
        "max_visible_contour_core_weight": 0.0, "max_accepted_face_hole_fraction": 0.0,
        "max_outside_exact_pp_max_delta": 0.0,
    }
    assert classify_v230(
        summary, parser_audit_passed=True, parity_max_diff=0.0
    ) == "V230_READY_FOR_VISUAL_REVIEW"
    assert classify_v230(
        summary, parser_audit_passed=False, parity_max_diff=0.0
    ) == "V230_PARSER_REGION_AUDIT_FAIL"
    aggregated = aggregate_v230([{"value": 1.0}, {"value": 3.0}])
    assert aggregated["median_value"] == 2.0 and aggregated["max_value"] == 3.0


def main():
    test_achromatic_hf_and_core_color()
    test_topology_interior_hole_and_face_rejection()
    test_pp_edge_geometry_and_far_outside_exact()
    test_face_contact_arbitration_and_true_bang_core()
    test_no_binary_silhouette_and_soft_weight_contract()
    test_parser_regions_and_runtime_parity()
    test_metric_gate_order()
    print("V2.30 PP-guided soft-edge tests passed")


if __name__ == "__main__":
    main()
