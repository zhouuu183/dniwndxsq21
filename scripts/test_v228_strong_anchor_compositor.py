"""Synthetic contracts for the deterministic Blending V8.28 compositor."""

from __future__ import annotations

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.strong_anchor_compositor_v828 import (
    StrongAnchorAppearanceCompositorV828,
    apply_pp_hair_lock_v828,
)
from models.v828_runtime_inputs import build_v828_runtime_inputs
from utils.v228_metrics import appearance_metric_tensors, pp_lock_metric_tensors


def _fixture(size: int = 32):
    base = torch.zeros(1, 3, size, size)
    anchor = base.clone()
    anchor[:, 0] = 0.20
    anchor[:, 1] = 0.05
    hair = torch.zeros(1, 1, size, size)
    hair[:, :, 8:24, 8:24] = 1
    eroded = torch.zeros_like(hair)
    eroded[:, :, 11:21, 11:21] = 1
    dilated = torch.zeros_like(hair)
    dilated[:, :, 5:27, 5:27] = 1
    inputs = build_v828_runtime_inputs(
        base_rgb=base,
        anchor_rgb=anchor,
        target_hair_mask=hair,
        target_hair_eroded=eroded,
        target_hair_dilated=dilated,
    )
    return base, anchor, inputs


def test_core_outside_and_fractional_alpha_contract():
    base, anchor, inputs = _fixture()
    model = StrongAnchorAppearanceCompositorV828(bg_residual_radius=2, outer_ring_width=3)
    output, aux = model(**inputs, return_aux=True)
    core = aux["sure_fg"].expand_as(output) > 0.5
    outside = aux["outside"].expand_as(output) > 0.5
    unknown = aux["unknown_band"] > 0.5
    assert (output[core] - anchor[core]).abs().max().item() <= 1e-6
    assert (output[outside] - base[outside]).abs().max().item() <= 1e-6
    assert ((aux["hair_alpha"][unknown] > 0) & (aux["hair_alpha"][unknown] < 1)).any()
    assert aux["coverage_alpha_multiply_count"].item() == 1
    assert aux["partition_error"].item() == 0


def test_background_residual_removal_and_no_sample_fallback():
    base, anchor, inputs = _fixture()
    # Red drift exists throughout Anchor; a separate green hair residual remains.
    anchor = anchor.clone()
    anchor[:, 0] += 0.15
    anchor[:, 1, 8:24, 8:24] += 0.25
    inputs["anchor_rgb"] = anchor.clamp(0, 1)
    model = StrongAnchorAppearanceCompositorV828(bg_residual_radius=3, outer_ring_width=4)
    _, aux = model(**inputs, return_aux=True)
    valid = aux["background_est_valid"] > 0.5
    assert valid.any()
    estimated_red = aux["background_residual_estimate"][:, 0:1][valid]
    assert estimated_red.mean().item() > 0.10
    # With all surrounding pixels protected, background estimation must safely disable.
    inputs["hard_protect_mask"] = torch.ones_like(inputs["hard_protect_mask"])
    output, protected_aux = model(**inputs, return_aux=True)
    assert torch.isfinite(output).all()
    assert protected_aux["background_est_valid"].max().item() == 0
    assert protected_aux["background_residual_estimate"].abs().max().item() == 0


def test_pp_strict_lock_and_unlock_contract():
    prepp = torch.zeros(1, 3, 8, 8)
    pp = torch.ones_like(prepp)
    alpha = torch.zeros(1, 1, 8, 8)
    alpha[:, :, 2:6, 2:6] = 1
    final, lock = apply_pp_hair_lock_v828(prepp, pp, alpha)
    assert final[:, :, 2:6, 2:6].abs().max().item() == 0
    assert (final[:, :, :2] - pp[:, :, :2]).abs().max().item() == 0
    unlock = torch.zeros_like(alpha)
    unlock[:, :, 3:5, 3:5] = 1
    unlocked, effective = apply_pp_hair_lock_v828(prepp, pp, alpha, unlock)
    assert (unlocked[:, :, 3:5, 3:5] - pp[:, :, 3:5, 3:5]).abs().max().item() == 0
    assert effective[:, :, 3:5, 3:5].max().item() == 0


def test_soft_probability_is_limited_to_trimap():
    base, anchor, inputs = _fixture()
    soft = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 32, 32)
    inputs["soft_hair_probability"] = soft
    model = StrongAnchorAppearanceCompositorV828(bg_residual_radius=2)
    _, aux = model(**inputs, return_aux=True)
    assert aux["hair_alpha"][aux["sure_fg"] > 0.5].min().item() == 1
    assert aux["hair_alpha"][aux["outside"] > 0.5].max().item() == 0
    unknown_values = aux["hair_alpha"][aux["unknown_band"] > 0.5]
    assert unknown_values.unique().numel() > 2


def test_metric_shapes_and_strict_pp_drift():
    base, anchor, inputs = _fixture()
    model = StrongAnchorAppearanceCompositorV828(bg_residual_radius=2)
    output, aux = model(**inputs, return_aux=True)
    metrics = appearance_metric_tensors(
        base_rgb=base,
        anchor_rgb=anchor,
        v226_rgb=base,
        output_rgb=output,
        core=aux["sure_fg"],
        edge=aux["unknown_band"],
        outer_ring=aux["background_sample_ring"],
        face_mask=torch.zeros_like(aux["sure_fg"]),
    )
    assert all(value.shape == (1,) for value in metrics.values())
    final, _ = apply_pp_hair_lock_v828(output, torch.ones_like(output), aux["hair_alpha"])
    pp_metrics = pp_lock_metric_tensors(output, final, aux["sure_fg"], aux["unknown_band"])
    assert pp_metrics["pp_core_ab_shift"].item() <= 1e-4
    assert pp_metrics["pp_core_l_shift"].item() <= 1e-4


if __name__ == "__main__":
    test_core_outside_and_fractional_alpha_contract()
    test_background_residual_removal_and_no_sample_fallback()
    test_pp_strict_lock_and_unlock_contract()
    test_soft_probability_is_limited_to_trimap()
    test_metric_shapes_and_strict_pp_drift()
    print("V2.28 strong-anchor compositor tests passed")
