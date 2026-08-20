"""Synthetic contracts for V2.31 high-resolution matting and PP recolor."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.hair_matting_v831 import HairMattingV831
from models.local_background_replacement_v831 import LocalBackgroundReplacementV831
from models.pp_unified_final_v831 import PPUnifiedFinalV831
from models.pp_unified_recolor_v831 import PPUnifiedRecolorV831
from models.trimap_builder_v831 import TrimapBuilderV831
from models.v831_runtime_inputs import build_v831_runtime_inputs
from models.SG_IDCT_v16 import rgb_to_lab
from models.color_condition_v8 import compute_intrinsic_hair_color_stats
from utils.v231_metrics import classify_matte_v231, classify_v231, v231_metric_tensors


class FakeProcessor:
    def __call__(self, images=None, trimaps=None, return_tensors=None):
        del return_tensors
        image = torch.from_numpy(__import__("numpy").stack(images)).permute(0, 3, 1, 2).float() / 255
        trimap_bytes = torch.from_numpy(__import__("numpy").stack(trimaps))
        assert set(torch.unique(trimap_bytes).tolist()).issubset({0, 128, 255})
        trimap = trimap_bytes.unsqueeze(1).float() / 255
        return {"pixel_values": image, "trimap": trimap}


class FakeMatte(nn.Module):
    def forward(self, pixel_values, trimap):
        del pixel_values
        ramp = torch.linspace(0.1, 0.9, trimap.shape[-1], device=trimap.device)
        return SimpleNamespace(alphas=ramp.view(1, 1, 1, -1).expand_as(trimap))


def _mask(size=64):
    hair = torch.zeros(1, 1, size // 4, size // 4)
    hair[..., 3:-3, 4:-4] = 1
    return hair


def test_trimap_semantics_and_face_contact():
    hair = _mask()
    face = torch.zeros_like(hair)
    face[..., 7:, :] = 1
    trimap, aux = TrimapBuilderV831(2, 2, 2)(
        hair, face, output_size=(64, 64), return_aux=True
    )
    assert torch.equal(torch.unique(trimap).sort().values, torch.tensor([0.0, 0.5, 1.0]))
    assert (aux["sure_fg"] * aux["sure_bg"]).max().item() == 0
    assert (aux["unknown"] * aux["face_contact_zone"]).sum().item() > 0
    assert aux["sure_bg"][..., 0, 0].item() == 1


def test_only_small_enclosed_trimap_holes_are_filled():
    hair = torch.ones(1, 1, 32, 32)
    hair[..., 8, 8] = 0
    hair[..., 14:19, 14:19] = 0
    hair[..., 0, 24] = 0
    _, aux = TrimapBuilderV831(1, 1, 0, max_trimap_hole_area=16)(
        hair, torch.zeros_like(hair), output_size=(32, 32), return_aux=True
    )
    assert aux["filled_trimap_holes"][0, 0, 8, 8].item() == 1
    assert aux["filled_trimap_holes"][..., 14:19, 14:19].sum().item() == 0
    assert aux["filled_trimap_holes"][0, 0, 0, 24].item() == 0


def test_matte_clamp_continuity_and_true_bang():
    size = 64
    pp = torch.rand(1, 3, size, size)
    hair = _mask(size)
    face = torch.zeros_like(hair)
    face[..., hair.shape[-2] // 2 :, :] = 1
    matte = HairMattingV831(
        "injected", device="cpu", inner_width=2, outer_width=2,
        face_contact_extra_inner=1, processor=FakeProcessor(), model=FakeMatte(),
    )
    alpha, aux = matte(
        pp_rgb_1024=pp, target_hair_mask_256=hair,
        source_face_mask_256=face, source_skin_mask_256=face, return_aux=True,
    )
    assert aux["sure_fg_error"].item() == 0
    assert aux["sure_bg_error"].item() == 0
    unknown_values = alpha[aux["unknown"] > 0.5]
    assert ((unknown_values > 0) & (unknown_values < 1)).any()
    deep_hair = aux["sure_fg"] > 0.5
    assert deep_hair.any() and alpha[deep_hair].min().item() == 1


def _highpass(value):
    return value - F.avg_pool2d(value, 5, stride=1, padding=2)


def test_low_frequency_recolor_preserves_pp_texture():
    size = 65
    x = torch.arange(size).view(1, 1, 1, size)
    texture = ((x % 2) * 2 - 1).float() * 0.04
    pp = torch.full((1, 3, size, size), 0.35) + texture
    target = torch.zeros_like(pp)
    target[:, 0] = 0.65
    target[:, 1] = 0.30
    target[:, 2] = 0.20
    alpha = torch.ones(1, 1, size, size)
    result, aux = PPUnifiedRecolorV831(5, 7)(
        pp_rgb=pp, v226_rgb=target, alpha_hr=alpha,
        sure_fg=alpha, return_aux=True,
    )
    assert (result.mean(dim=(2, 3)) - target.mean(dim=(2, 3))).abs().max().item() < 0.08
    assert (_highpass(result) - _highpass(pp)).abs().mean().item() < 0.015
    assert aux["dense_core"].min().item() == 1


def test_face_contact_is_alpha_scaled():
    size = 33
    pp = torch.full((1, 3, size, size), 0.3)
    target = torch.full_like(pp, 0.7)
    sure = torch.ones(1, 1, size, size)
    full = PPUnifiedRecolorV831(3, 3)(
        pp_rgb=pp, v226_rgb=target, alpha_hr=sure, sure_fg=sure
    )
    partial_alpha = sure.clone()
    partial_alpha[..., 16, 16] = 0.2
    partial = PPUnifiedRecolorV831(3, 3)(
        pp_rgb=pp, v226_rgb=target, alpha_hr=partial_alpha, sure_fg=sure
    )
    full_delta = (full[..., 16, 16] - pp[..., 16, 16]).abs().mean()
    partial_delta = (partial[..., 16, 16] - pp[..., 16, 16]).abs().mean()
    assert 0.12 < (partial_delta / full_delta).item() < 0.30


def test_background_equation_and_far_outside():
    size = 41
    pp = torch.full((1, 3, size, size), 0.6)
    base = torch.full_like(pp, 0.2)
    alpha = torch.zeros(1, 1, size, size)
    alpha[..., 18:23, 18:23] = 0.25
    unknown = torch.zeros_like(alpha)
    unknown[..., 18:23, 18:23] = 1
    face = torch.ones_like(alpha)
    subject = torch.ones_like(alpha)
    phase_c, aux = LocalBackgroundReplacementV831(3, 0)(
        phase_b_rgb=pp, pp_rgb=pp, base_rgb=base, alpha_hr=alpha,
        unknown=unknown, source_face_mask=face, source_subject_mask=subject,
        return_aux=True,
    )
    expected = 0.6 + (1 - 0.25) * (0.2 - 0.6)
    assert abs(phase_c[0, 0, 20, 20].item() - expected) < 1e-5
    assert torch.equal(phase_c[..., :10, :], pp[..., :10, :])
    assert aux["far_outside_max_delta"].max().item() == 0


def test_no_binary_final_and_runtime_contract():
    source = (Path(__file__).parents[1] / "models" / "pp_unified_final_v831.py").read_text()
    assert "alpha_hr > 0.5" not in source
    assert "torch.where" not in source
    image = torch.zeros(1, 3, 16, 16)
    pp = torch.zeros(1, 3, 64, 64)
    labels = torch.zeros(1, 1, 16, 16)
    runtime = build_v831_runtime_inputs(
        pp_original_rgb=pp, base_rgb=image, v226_rgb=image,
        target_hair_mask=image[:, :1], parser_labels=labels,
    )
    assert runtime["pp_original_rgb"].shape[-2:] == (64, 64)
    finalizer = PPUnifiedFinalV831(enable_phase_c=True)
    config = finalizer.config_dict()
    assert config["final_binary_hair_mask"] is False
    assert config["strong_anchor_final_injection"] is False


def test_all_failed_gates_are_reported():
    matte = classify_matte_v231({
        "max_sure_fg_error": 0.1, "max_sure_bg_error": 0.0,
        "median_unknown_fractional_fraction": 0.0,
        "median_alpha_boundary_deviation": 0.0,
        "p90_face_false_positive_alpha": 0.8,
    })
    assert matte["failed_gates"] == [
        "V231_MATTING_BACKEND_FAIL", "V231_MATTE_NOT_REFINING", "V231_MATTE_FACE_FP"
    ]
    failed = classify_v231({}, parity_rgb=1.0, parity_alpha=1.0)
    assert failed["primary_failure"] == "V231_RUNTIME_PARITY_FAIL"
    assert "V231_CORE_COLOR_FAIL" in failed["failed_gates"]
    assert "V231_BACKGROUND_RETENTION_FAIL" in failed["failed_gates"]


def test_metric_tensor_contract():
    image = torch.full((2, 3, 16, 16), 0.4)
    target = image.clone()
    target[:, 0] = 0.55
    mask = torch.ones(2, 1, 16, 16)
    alpha = F.interpolate(mask, size=(64, 64), mode="nearest")
    ref_stats = compute_intrinsic_hair_color_stats(rgb_to_lab(target), mask)
    metrics = v231_metric_tensors(
        base_rgb=image, v226_rgb=target, v230_rgb=image,
        pp_rgb=F.interpolate(image, size=(64, 64), mode="nearest"),
        phase_b_rgb=F.interpolate(target, size=(64, 64), mode="nearest"),
        phase_c_rgb=F.interpolate(target, size=(64, 64), mode="nearest"),
        target_lab=rgb_to_lab(target), alpha_hr=alpha,
        source_face_mask=torch.zeros_like(mask),
        source_subject_mask=torch.zeros_like(mask), ref_stats=ref_stats,
    )
    assert metrics["core_ref_full_pseudo"].shape == (2,)
    assert metrics["phase_c_direct_ref_ab_stat_error"].max().item() < 1e-4


def main():
    test_trimap_semantics_and_face_contact()
    test_only_small_enclosed_trimap_holes_are_filled()
    test_matte_clamp_continuity_and_true_bang()
    test_low_frequency_recolor_preserves_pp_texture()
    test_face_contact_is_alpha_scaled()
    test_background_equation_and_far_outside()
    test_no_binary_final_and_runtime_contract()
    test_all_failed_gates_are_reported()
    test_metric_tensor_contract()
    print("V2.31 high-resolution matting/recolor tests passed")


if __name__ == "__main__":
    main()
