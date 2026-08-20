"""Synthetic contracts for V2.27 diagnostic hardening."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.selective_color_projector_v826 import (
    FULL_COLOR_ARCH_V8_9,
    BoundaryTargetAlignedProjectorV826,
    apply_v226_compensation_from_cached_v225,
    load_v226_projector_checkpoint,
)
from utils.v227_metrics import independent_edge_artifact_metrics, meaningful_ab_metrics


def test_meaningful_ab_denominator_and_progress():
    base = torch.zeros(1, 3, 10, 10)
    output = base.clone()
    target = torch.zeros(1, 2, 10, 10)
    target[:, 0, :, :5] = 10.0
    target[:, 0, :, 5:] = 0.1
    output[:, 1, :, :5] = 7.0
    output[:, 1, :, 5:] = -50.0
    metrics = meaningful_ab_metrics(
        base_lab=base, output_lab=output, target_ref_ab=target,
        hair_edge=torch.ones(1, 1, 10, 10),
    )
    assert torch.allclose(metrics["meaningful_ab_pixel_fraction"], torch.tensor([0.5]))
    assert torch.allclose(metrics["edge_desired_ab_direction_meaningful"], torch.tensor([1.0]))
    assert torch.allclose(metrics["edge_desired_ab_progress_meaningful"], torch.tensor([1.0]))


def _projector_inputs():
    torch.manual_seed(17)
    base = torch.rand(1, 3, 16, 16)
    anchor = torch.rand_like(base)
    hair = torch.zeros(1, 1, 16, 16); hair[:, :, 2:14, 2:14] = 1
    eroded = torch.zeros_like(hair); eroded[:, :, 5:11, 5:11] = 1
    outside = 1 - hair
    return {
        "base_rgb": base, "anchor_rgb": anchor,
        "pseudo_lab": torch.rand_like(base) * 50,
        "target_ref_ab": torch.rand(1, 2, 16, 16) * 30 - 15,
        "reference_delta_l": torch.tensor([7.0]),
        "target_hair_mask": hair, "target_hair_eroded": eroded,
        "outer_background_guard": torch.zeros_like(hair),
        "face_keep_mask": outside, "skin_protect_mask": outside,
        "satd_protect_mask": outside, "remove_mask": torch.zeros_like(hair),
    }


def test_cached_gamma_parity_and_invariants():
    inputs = _projector_inputs()
    model = BoundaryTargetAlignedProjectorV826(compensation_gamma=0.0)
    gamma0, aux = model(return_aux=True, **inputs)
    cached0, _ = apply_v226_compensation_from_cached_v225(
        base_rgb=inputs["base_rgb"], v225_rgb=aux["v225_rgb"],
        v225_luma_output=aux["luma_output"], v225_ab_output=aux["ab_output"],
        target_ref_ab=inputs["target_ref_ab"], hair_edge=aux["hair_edge"],
        hair_membership=aux["hair_membership"], hard_protect=aux["hard_protect"],
        compensation_gamma=0.0,
    )
    cached25, _ = apply_v226_compensation_from_cached_v225(
        base_rgb=inputs["base_rgb"], v225_rgb=aux["v225_rgb"],
        v225_luma_output=aux["luma_output"], v225_ab_output=aux["ab_output"],
        target_ref_ab=inputs["target_ref_ab"], hair_edge=aux["hair_edge"],
        hair_membership=aux["hair_membership"], hard_protect=aux["hard_protect"],
        compensation_gamma=0.25,
    )
    assert (cached0 - gamma0).abs().max().item() <= 1e-6
    assert torch.equal(inputs["base_rgb"], inputs["base_rgb"].clone())
    assert torch.isfinite(cached25).all()


def test_checkpoint_restore_gamma():
    model = BoundaryTargetAlignedProjectorV826(compensation_gamma=0.25)
    payload = {
        "arch": FULL_COLOR_ARCH_V8_9, "version": "v2.26",
        "projector_config": model.config_dict(),
        "projector_state_dict": model.state_dict(),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "v226.pth"
        torch.save(payload, path)
        restored, checkpoint = load_v226_projector_checkpoint(path, "cpu")
    assert restored.compensation_gamma == 0.25
    assert checkpoint["projector_config"]["compensation_gamma"] == 0.25


def test_independent_fringe_metrics():
    edge = torch.zeros(1, 1, 12, 12); edge[:, :, :, :4] = 1
    core = 1 - edge
    base = torch.zeros(1, 3, 12, 12); base[:, :1] = 40; base[:, 1] = 12

    white = base.clone(); white[:, :1, :, :4] = 60; white[:, 1:, :, :4] = 0
    white_metrics = independent_edge_artifact_metrics(
        base_lab=base, output_lab=white, hair_core=core, hair_edge=edge
    )
    assert white_metrics["white_fringe_fraction"].item() > 0

    legitimate = base.clone(); legitimate[:, :1] = 55
    legitimate_metrics = independent_edge_artifact_metrics(
        base_lab=base, output_lab=legitimate, hair_core=core, hair_edge=edge
    )
    assert legitimate_metrics["white_fringe_fraction"].item() == 0

    saturated = base.clone(); saturated[:, 1, :, :4] = 30
    saturated_metrics = independent_edge_artifact_metrics(
        base_lab=base, output_lab=saturated, hair_core=core, hair_edge=edge
    )
    assert saturated_metrics["oversaturation_fringe_fraction"].item() > 0


def main():
    test_meaningful_ab_denominator_and_progress()
    test_cached_gamma_parity_and_invariants()
    test_checkpoint_restore_gamma()
    test_independent_fringe_metrics()
    print("test_v227_diagnostic_hardening: PASS")


if __name__ == "__main__":
    main()
