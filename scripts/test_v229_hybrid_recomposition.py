"""Synthetic algebra and ownership contracts for Blending V8.29."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.boundary_recomposition_v829 import (
    BoundaryRecompositionV829,
    recompose_observed_boundary_v829,
)
from models.hair_ownership_v829 import (
    HairOwnershipV829,
    apply_pp_hair_ownership_lock_v829,
)
from models.hybrid_hair_carrier_v829 import HybridHairCarrierV829, gaussian_blur_v829
from models.v829_runtime_inputs import build_v829_runtime_inputs
from utils.v229_metrics import (
    aggregate_v229,
    classify_v229,
    pp_ownership_metrics,
    v229_metric_tensors,
)


def test_double_alpha_and_exact_background_replacement():
    alpha = torch.tensor([[[[0.5]]]])
    foreground = torch.ones(1, 3, 1, 1)
    anchor_background = torch.zeros_like(foreground)
    base_background = torch.full_like(foreground, 0.2)
    anchor = alpha * foreground + (1.0 - alpha) * anchor_background
    expected = alpha * foreground + (1.0 - alpha) * base_background
    recomposed = recompose_observed_boundary_v829(
        anchor, alpha, anchor_background, base_background
    )
    old_double_alpha = base_background + alpha * (anchor - base_background)
    assert (recomposed - expected).abs().max().item() <= 1e-6
    assert (old_double_alpha - expected).abs().max().item() > 0.1


def test_tone_correction_is_alpha_delta_once():
    anchor = torch.full((1, 3, 1, 1), 0.4)
    alpha = torch.tensor([[[[0.25]]]])
    delta = torch.full_like(anchor, 0.2)
    output = recompose_observed_boundary_v829(
        anchor, alpha, torch.zeros_like(anchor), torch.zeros_like(anchor), delta
    )
    assert (output - (anchor + 0.25 * delta)).abs().max().item() <= 1e-6


def _ownership_fixture():
    base = torch.zeros(1, 3, 9, 9)
    anchor = base.clone()
    anchor[:, 0] = 0.4
    hair = torch.zeros(1, 1, 9, 9)
    hair[:, :, 3:6, 3:6] = 1
    eroded = torch.zeros_like(hair)
    eroded[:, :, 4:5, 4:5] = 1
    dilated = torch.zeros_like(hair)
    dilated[:, :, 2:7, 2:7] = 1
    return base, anchor, hair, eroded, dilated


def test_hair_over_skin_and_dilation_over_skin():
    base, anchor, hair, eroded, dilated = _ownership_fixture()
    skin = torch.ones_like(hair)
    owner, aux = HairOwnershipV829(outer_strand_recovery=False)(
        base_rgb=base,
        anchor_rgb=anchor,
        target_hair_mask=hair,
        target_hair_eroded=eroded,
        target_hair_dilated=dilated,
        source_subject_mask=skin,
        source_skin_mask=skin,
        return_aux=True,
    )
    assert owner[hair > 0.5].min().item() == 1.0
    outer_skin = (dilated > 0.5) & (hair < 0.5)
    assert owner[outer_skin].max().item() == 0.0
    assert aux["face_side_outer"][outer_skin].min().item() == 1.0
    assert aux["partition_error"].item() == 0.0


def test_pp_ownership_ignores_coverage_and_preserves_nonhair():
    prepp = torch.zeros(1, 3, 4, 4)
    pp = torch.ones_like(prepp)
    owner = torch.zeros(1, 1, 4, 4)
    owner[:, :, 1:3, 1:3] = 1
    for coverage in (0.1, 0.5, 0.9):
        del coverage
        final, effective = apply_pp_hair_ownership_lock_v829(prepp, pp, owner)
        assert final[:, :, 1:3, 1:3].abs().max().item() == 0.0
        assert (final[:, :, 0] - pp[:, :, 0]).abs().max().item() == 0.0
        assert torch.equal(effective, owner)


def test_hybrid_core_uses_v226_low_and_anchor_high():
    size = 31
    columns = torch.arange(size).view(1, 1, 1, size)
    stripes = ((columns % 2) * 2 - 1).float().expand(1, 3, size, size) * 0.08
    anchor = (0.45 + stripes).clamp(0, 1)
    v226 = torch.full_like(anchor, 0.72)
    core = torch.ones(1, 1, size, size)
    hybrid, aux = HybridHairCarrierV829(low_radius=3)(
        anchor_rgb=anchor,
        v226_rgb=v226,
        base_rgb=torch.zeros_like(anchor),
        hair_core=core,
        return_aux=True,
    )
    inner = hybrid[:, :, 5:-5, 5:-5]
    assert abs(inner.mean().item() - 0.72) < 0.02
    hybrid_hf = hybrid - gaussian_blur_v829(hybrid, 3)
    anchor_hf = aux["anchor_hf"]
    assert (hybrid_hf[:, :, 5:-5, 5:-5] - anchor_hf[:, :, 5:-5, 5:-5]).abs().mean().item() < 0.02


def test_no_background_estimate_falls_back_to_anchor_composite():
    base, anchor, hair, eroded, dilated = _ownership_fixture()
    # Full source subject support plus an all-image target means no far samples.
    hair.fill_(1)
    eroded.fill_(0)
    eroded[:, :, 1:-1, 1:-1] = 1
    dilated.fill_(1)
    source = torch.ones_like(hair)
    v226 = anchor.clone()
    carrier = HybridHairCarrierV829(low_radius=2)
    hybrid, _ = carrier(
        anchor_rgb=anchor, v226_rgb=v226, base_rgb=base, hair_core=eroded, return_aux=True
    )
    runtime = build_v829_runtime_inputs(
        base_rgb=base,
        anchor_rgb=anchor,
        v226_rgb=v226,
        target_hair_mask=hair,
        target_hair_eroded=eroded,
        target_hair_dilated=dilated,
        source_subject_mask=source,
        source_skin_mask=source,
    )
    output, aux = BoundaryRecompositionV829(
        coverage_max_distance=4,
        carrier_low_radius=2,
        tone_propagation_radius=2,
        background_radius=2,
        background_ring_outer=2,
    )(**runtime, hybrid_core_rgb=hybrid, return_aux=True)
    assert torch.isfinite(output).all()
    assert aux["background_recompose_valid"].max().item() == 0.0
    edge = aux["inner_edge_owner"] > 0.5
    expected = anchor + aux["coverage_alpha"] * aux["tone_delta_edge"]
    assert (output[edge.expand_as(output)] - expected[edge.expand_as(expected)]).abs().max().item() <= 1e-6


def test_face_and_background_contexts_do_not_cross():
    size = 41
    base = torch.zeros(1, 3, size, size)
    base[:, 0, :, :size // 2] = 0.65
    base[:, 1, :, size // 2:] = 0.70
    anchor = base.clone()
    anchor[:, 0, :, :size // 2] += 0.10
    anchor[:, 2, :, size // 2:] += 0.15
    hair = torch.zeros(1, 1, size, size)
    hair[:, :, 12:29, 12:29] = 1
    eroded = torch.zeros_like(hair)
    eroded[:, :, 15:26, 15:26] = 1
    dilated = torch.zeros_like(hair)
    dilated[:, :, 10:31, 10:31] = 1
    source = torch.zeros_like(hair)
    source[:, :, :, :size // 2] = 1
    hybrid = anchor.clone()
    runtime = build_v829_runtime_inputs(
        base_rgb=base, anchor_rgb=anchor, v226_rgb=anchor,
        target_hair_mask=hair, target_hair_eroded=eroded,
        target_hair_dilated=dilated, source_subject_mask=source,
        source_skin_mask=source,
    )
    _, aux = BoundaryRecompositionV829(
        coverage_max_distance=5, background_radius=5,
        background_ring_inner=2, background_ring_outer=10,
    )(**runtime, hybrid_core_rgb=hybrid, return_aux=True)
    left_edge = aux["inner_edge_owner"].clone()
    left_edge[:, :, :15, :] = 0
    left_edge[:, :, 26:, :] = 0
    left_edge[:, :, :, 15:] = 0
    right_edge = aux["inner_edge_owner"].clone()
    right_edge[:, :, :15, :] = 0
    right_edge[:, :, 26:, :] = 0
    right_edge[:, :, :, :26] = 0
    assert aux["face_context_map"][left_edge > 0.5].mean().item() > 0.9
    assert aux["background_context_map"][right_edge > 0.5].mean().item() > 0.9


def test_v229_metric_contract_and_gate_order():
    base, anchor, hair, eroded, dilated = _ownership_fixture()
    output = anchor.clone()
    metrics = v229_metric_tensors(
        base_rgb=base, anchor_rgb=anchor, v226_rgb=anchor, v228_rgb=anchor,
        output_rgb=output, target_lab=torch.zeros(1, 3, 9, 9), core=eroded,
        inner_edge=(hair - eroded).clamp(0, 1), ownership=hair,
        nonhair=1.0 - hair, source_skin=torch.zeros_like(hair),
        face_context=torch.zeros_like(hair), background_context=(hair - eroded).clamp(0, 1),
    )
    assert all(value.shape == (1,) for value in metrics.values())
    pp = pp_ownership_metrics(output, base, output, hair, (hair - eroded).clamp(0, 1))
    assert pp["pp_owner_max_delta"].item() == 0.0
    summary = {
        "median_core_ref_full": 1.0, "median_v226_core_ref_full": 1.0,
        "median_anchor_hf_l1": 0.1, "median_v226_anchor_hf_l1": 1.0,
        "max_outside_max_delta": 0.0, "median_background_contamination_retention": 0.0,
        "max_face_side_outer_hair_fraction": 0.0, "median_face_side_color_leak_rgb": 0.0,
        "median_base_color_rim_fraction": 0.1, "median_v228_base_color_rim_fraction": 0.3,
        "median_edge_to_nearcore_delta_e": 1.0, "median_v228_edge_to_nearcore_delta_e": 2.0,
        "max_pp_owner_max_delta": 0.0, "max_nonhair_final_to_pp_max_delta": 0.0,
        "mean_clip_magnitude": 0.0,
    }
    assert classify_v229(summary, parity_max_diff=0.0, phase_b=False) == "V229_PHASE_A_READY_FOR_VISUAL_REVIEW"
    phase_a = {
        **summary,
        "median_background_contamination_retention": 0.02,
        "median_face_side_color_leak_rgb": 0.0,
        "mean_outer_strand_small_component_fraction": 0.0,
    }
    phase_b = {
        **phase_a,
        "max_outer_strand_accepted_fraction": 0.01,
        "min_outer_strand_min_residual_alignment": 0.85,
    }
    assert classify_v229(
        phase_b, parity_max_diff=0.0, phase_b=True, phase_a_summary=phase_a
    ) == "V229_PHASE_B_READY_FOR_VISUAL_REVIEW"
    phase_b["median_background_contamination_retention"] = 0.03
    assert classify_v229(
        phase_b, parity_max_diff=0.0, phase_b=True, phase_a_summary=phase_a
    ) == "V229_BACKGROUND_RECOMPOSITION_FAIL"
    aggregated = aggregate_v229([{"metric": 3.0}, {"metric": 1.0}])
    assert aggregated["min_metric"] == 1.0 and aggregated["max_metric"] == 3.0


def main():
    test_double_alpha_and_exact_background_replacement()
    test_tone_correction_is_alpha_delta_once()
    test_hair_over_skin_and_dilation_over_skin()
    test_pp_ownership_ignores_coverage_and_preserves_nonhair()
    test_hybrid_core_uses_v226_low_and_anchor_high()
    test_no_background_estimate_falls_back_to_anchor_composite()
    test_face_and_background_contexts_do_not_cross()
    test_v229_metric_contract_and_gate_order()
    print("V2.29 hybrid recomposition tests passed")


if __name__ == "__main__":
    main()
