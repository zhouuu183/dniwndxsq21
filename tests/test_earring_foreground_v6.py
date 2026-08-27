import torch

from models.earring_foreground_v6 import (
    EarringNativeInstanceV6,
    align_earring_instance_v6,
    composite_earring_v6,
    extract_source_native_earring_v6,
    retain_single_earring_group_v6,
)


def mask(h=64, w=64):
    return torch.zeros(1, 1, h, w)


def test_empty_source_has_no_group():
    value = mask()
    anchor = mask()
    assert retain_single_earring_group_v6(value, anchor).sum().item() == 0


def test_only_lobe_near_component_is_kept():
    value = mask()
    value[:, :, 29:34, 30:35] = 1.0
    value[:, :, 50:53, 5:20] = 1.0
    anchor = mask()
    anchor[:, :, 25:31, 29:36] = 1.0
    result = retain_single_earring_group_v6(value, anchor, max_gap=4)
    assert result[:, :, 29:34, 30:35].sum().item() > 0
    assert result[:, :, 50:53, 5:20].sum().item() == 0


def test_long_pendant_accepts_short_downward_segments_only():
    value = mask()
    value[:, :, 28:33, 30:35] = 1.0
    value[:, :, 36:40, 31:34] = 1.0
    value[:, :, 45:49, 46:50] = 1.0
    anchor = mask()
    anchor[:, :, 24:30, 29:36] = 1.0
    result = retain_single_earring_group_v6(value, anchor, max_gap=5)
    assert result[:, :, 28:33, 30:35].sum().item() > 0
    assert result[:, :, 36:40, 31:34].sum().item() > 0
    assert result[:, :, 45:49, 46:50].sum().item() == 0


def test_alignment_rejects_out_of_range_raw_shift():
    h = w = 64
    rgb = torch.full((1, 3, h, w), 0.7)
    alpha = mask(h, w)
    alpha[:, :, 14:18, 14:18] = 1.0
    instance = EarringNativeInstanceV6(
        alpha=alpha,
        rgb=rgb,
        hole_alpha=mask(h, w),
        left_alpha=alpha,
        right_alpha=mask(h, w),
        left_hole_alpha=mask(h, w),
        right_hole_alpha=mask(h, w),
    )
    source_ear = mask(h, w)
    source_ear[:, :, 8:15, 12:20] = 1.0
    target_ear = mask(h, w)
    target_ear[:, :, 42:50, 43:51] = 1.0
    aligned = align_earring_instance_v6(
        instance, source_ear, mask(h, w), target_ear, mask(h, w), max_shift=3
    )
    assert aligned["left_alignment_valid"].item() == 0
    assert aligned["target_aligned_left_alpha"].sum().item() == 0


def test_composite_failure_and_hole_leave_base_unchanged():
    base = torch.full((1, 3, 32, 32), 0.2)
    aligned = {
        "target_aligned_left_alpha": mask(32, 32),
        "target_aligned_right_alpha": mask(32, 32),
        "target_aligned_hole_alpha": mask(32, 32),
        "target_aligned_earring_rgb": torch.full((1, 3, 32, 32), 0.9),
    }
    open_ear = torch.ones(1, 1, 32, 32)
    result, alpha = composite_earring_v6(base, aligned, open_ear, open_ear)
    assert torch.equal(result, base)
    assert alpha.sum().item() == 0

    aligned["target_aligned_left_alpha"][:, :, 10:22, 10:22] = 1.0
    aligned["target_aligned_hole_alpha"][:, :, 14:18, 14:18] = 1.0
    result, alpha = composite_earring_v6(base, aligned, open_ear, open_ear)
    assert alpha[:, :, 14:18, 14:18].sum().item() == 0
    assert torch.equal(result[:, :, 14:18, 14:18], base[:, :, 14:18, 14:18])


def test_same_side_second_component_is_rejected():
    value = mask()
    value[:, :, 29:34, 30:35] = 1.0
    value[:, :, 36:40, 40:45] = 1.0
    anchor = mask()
    anchor[:, :, 25:31, 29:36] = 1.0
    result = retain_single_earring_group_v6(value, anchor, max_gap=4)
    assert result[:, :, 29:34, 30:35].sum().item() > 0
    assert result[:, :, 36:40, 40:45].sum().item() == 0


def test_composite_keeps_subpixel_stud_alpha_without_rescaling():
    base = torch.full((1, 3, 16, 16), 0.2)
    alpha = mask(16, 16)
    alpha[:, :, 7:9, 7:9] = 1.0
    aligned = {
        "target_aligned_left_alpha": alpha,
        "target_aligned_right_alpha": mask(16, 16),
        "target_aligned_hole_alpha": mask(16, 16),
        "target_aligned_earring_rgb": torch.full((1, 3, 16, 16), 0.9),
    }
    gate = mask(16, 16)
    gate[:, :, 4:12, 4:12] = 1.0
    result, output_alpha = composite_earring_v6(base, aligned, gate, mask(16, 16))
    assert torch.equal(output_alpha, alpha)
    assert torch.allclose(result[:, :, 7:9, 7:9], torch.full((1, 3, 2, 2), 0.9))


def test_composite_closed_target_side_is_identity():
    base = torch.rand(1, 3, 24, 24)
    alpha = mask(24, 24)
    alpha[:, :, 8:16, 8:16] = 1.0
    aligned = {
        "target_aligned_left_alpha": alpha,
        "target_aligned_right_alpha": mask(24, 24),
        "target_aligned_hole_alpha": mask(24, 24),
        "target_aligned_earring_rgb": torch.zeros(1, 3, 24, 24),
    }
    result, output_alpha = composite_earring_v6(
        base, aligned, mask(24, 24), mask(24, 24)
    )
    assert torch.equal(result, base)
    assert output_alpha.sum().item() == 0


def test_alignment_allows_only_one_pixel_zero_shift_fallback():
    h = w = 32
    rgb = torch.full((1, 3, h, w), 0.7)
    alpha = mask(h, w)
    alpha[:, :, 12:16, 12:16] = 1.0
    instance = EarringNativeInstanceV6(
        alpha=alpha,
        rgb=rgb,
        hole_alpha=mask(h, w),
        left_alpha=alpha,
        right_alpha=mask(h, w),
        left_hole_alpha=mask(h, w),
        right_hole_alpha=mask(h, w),
    )
    source_ear = mask(h, w)
    source_ear[:, :, 8:15, 10:20] = 1.0
    target_ear = mask(h, w)
    target_ear[:, :, 9:16, 10:20] = 1.0
    aligned = align_earring_instance_v6(
        instance, source_ear, mask(h, w), target_ear, mask(h, w), max_shift=0
    )
    assert aligned["fallback_zero_shift_used_left"].item() == 1
    assert aligned["target_aligned_left_alpha"].sum().item() > 0


def test_parser_missed_skin_label_can_use_verified_native_seed():
    source = torch.full((1, 3, 64, 64), 0.5)
    parsing = torch.zeros(1, 1, 64, 64, dtype=torch.long)
    parsing[:, :, 20:43, 28:40] = 7
    source[:, :, 20:43, 28:40] = 0.65
    parsing[:, :, 40:46, 32:36] = 7  # parser calls the pendant ear skin
    source[:, :, 40:46, 32:36] = 0.05
    seed = mask(64, 64)
    seed[:, :, 40:46, 32:36] = 1.0
    extracted = extract_source_native_earring_v6(
        source, parsing, source_native_seed=seed, allow_long_continuation=True
    )
    assert extracted["source_native_left_alpha"].sum().item() > 0


def test_unseeded_uniform_background_does_not_become_earring():
    source = torch.full((1, 3, 64, 64), 0.5)
    parsing = torch.zeros(1, 1, 64, 64, dtype=torch.long)
    parsing[:, :, 20:43, 28:40] = 7
    extracted = extract_source_native_earring_v6(source, parsing)
    assert extracted["source_native_earring_alpha"].sum().item() == 0


def test_hole_alpha_never_changes_base_even_with_valid_object():
    base = torch.full((1, 3, 20, 20), 0.35)
    alpha = mask(20, 20)
    alpha[:, :, 6:15, 6:15] = 1.0
    hole = mask(20, 20)
    hole[:, :, 9:12, 9:12] = 1.0
    aligned = {
        "target_aligned_left_alpha": alpha,
        "target_aligned_right_alpha": mask(20, 20),
        "target_aligned_hole_alpha": hole,
        "target_aligned_earring_rgb": torch.full((1, 3, 20, 20), 0.95),
    }
    result, output_alpha = composite_earring_v6(
        base, aligned, torch.ones_like(hole), mask(20, 20)
    )
    assert torch.equal(result[:, :, 9:12, 9:12], base[:, :, 9:12, 9:12])
    assert output_alpha[:, :, 9:12, 9:12].sum().item() == 0
