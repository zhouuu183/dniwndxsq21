from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.SG_IDCT_v16 import gaussian_blur2d, lab_to_rgb, rgb_to_lab


COLOR_DESCRIPTOR_DIM = 41


@dataclass(frozen=True)
class ColorConditionConfigV8:
    ab_no_edit_threshold: float = 1.5
    ab_full_edit_threshold: float = 15.0
    hue_no_edit_deg: float = 4.0
    hue_full_edit_deg: float = 30.0
    chroma_mag_no_edit: float = 2.0
    chroma_mag_full_edit: float = 15.0
    color_dist_no_edit: float = 2.0
    color_dist_full_edit: float = 15.0
    lightness_no_edit_threshold: float = 3.0
    lightness_full_edit_threshold: float = 15.0
    max_global_l_shift: float = 40.0
    hue_valid_chroma_center: float = 5.0
    hue_valid_chroma_softness: float = 2.0
    relative_luma_bins: int = 8
    relative_luma_min_scale: float = 3.0
    global_ab_fallback_min_reliability: float = 0.5
    pseudo_fidelity_ab_bad: float = 8.0
    pseudo_fidelity_hue_bad: float = 15.0
    min_safe_fraction: float = 0.35
    highlight_mad_scale: float = 1.8
    highlight_global_min_margin: float = 3.0
    highlight_local_l_margin: float = 2.5
    highlight_local_c_margin: float = 1.5
    highlight_chroma_ratio: float = 0.82
    highlight_near_clip: float = 0.94
    highlight_clip_chroma: float = 12.0
    highlight_softness_l: float = 2.0
    highlight_softness_c: float = 1.5
    highlight_local_radius: int = 7
    reference_mask_erode: int = 2


def _as_image4d(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.size(1) != 3:
        raise ValueError(f"Expected RGB [B,3,H,W], got shape={tuple(image.shape)}")
    return image.float()


def _as_mask4d(mask: torch.Tensor, size: tuple[int, int], batch: int) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4 or mask.size(1) != 1:
        raise ValueError(f"Expected mask [B,1,H,W], got shape={tuple(mask.shape)}")
    if mask.size(0) == 1 and batch != 1:
        mask = mask.expand(batch, -1, -1, -1)
    if mask.size(0) != batch:
        raise ValueError(f"Mask batch {mask.size(0)} does not match image batch {batch}")
    mask = mask.float().clamp(0, 1)
    if mask.shape[-2:] != size:
        mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    return mask.clamp(0, 1)


def _to_rgb01(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    image = _as_image4d(image)
    normalized = bool((image.detach().amin() < -0.05).item())
    if normalized:
        image = image * 0.5 + 0.5
    return image.clamp(0, 1), normalized


def _from_rgb01(image: torch.Tensor, normalized: bool) -> torch.Tensor:
    image = image.clamp(0, 1)
    return image * 2.0 - 1.0 if normalized else image


def _weighted_quantile(values: torch.Tensor, weights: torch.Tensor, quantile: float) -> torch.Tensor:
    values = values.flatten(2)
    weights = weights.flatten(2)
    if weights.size(1) == 1 and values.size(1) != 1:
        weights = weights.expand(-1, values.size(1), -1)

    sorted_values, indices = values.sort(dim=-1)
    sorted_weights = weights.gather(-1, indices)
    cumulative = sorted_weights.cumsum(dim=-1)
    threshold = float(quantile) * sorted_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    quantile_index = (cumulative >= threshold).to(torch.int64).argmax(dim=-1, keepdim=True)
    return sorted_values.gather(-1, quantile_index).squeeze(-1)


def masked_robust_stats(features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return per-sample robust statistics for [B,C,H,W] features."""
    if features.dim() == 3:
        features = features.unsqueeze(0)
    if features.dim() != 4:
        raise ValueError(f"Expected features [B,C,H,W], got shape={tuple(features.shape)}")
    mask = _as_mask4d(mask, features.shape[-2:], features.size(0))
    valid = mask.sum(dim=(-2, -1), keepdim=True) >= 1.0
    mask = torch.where(valid, mask, torch.ones_like(mask))

    median = _weighted_quantile(features, mask, 0.5)
    absolute_deviation = (features - median[:, :, None, None]).abs()
    mad = _weighted_quantile(absolute_deviation, mask, 0.5)
    robust_scale = (1.4826 * mad).clamp_min(1e-3)
    normalized_deviation = absolute_deviation / (2.5 * robust_scale[:, :, None, None])
    robust_weight = mask / (1.0 + normalized_deviation.square())
    if robust_weight.size(1) == 1 and features.size(1) != 1:
        robust_weight = robust_weight.expand(-1, features.size(1), -1, -1)
    denominator = robust_weight.sum(dim=(-2, -1)).clamp_min(1e-6)
    mean = (features * robust_weight).sum(dim=(-2, -1)) / denominator
    variance = (
        (features - mean[:, :, None, None]).square() * robust_weight
    ).sum(dim=(-2, -1)) / denominator
    return {
        "mean": mean,
        "std": torch.sqrt(variance.clamp_min(1e-8)),
        "median": median,
        "mad": mad,
    }


def compute_intrinsic_hair_color_stats(
    lab: torch.Tensor,
    mask: torch.Tensor,
    config: ColorConditionConfigV8 | None = None,
) -> dict[str, torch.Tensor]:
    """Compute alignment-free, robust per-sample hair color statistics."""
    config = config or ColorConditionConfigV8()
    if lab.dim() == 3:
        lab = lab.unsqueeze(0)
    if lab.dim() != 4 or lab.size(1) != 3:
        raise ValueError(f"Expected Lab [B,3,H,W], got shape={tuple(lab.shape)}")
    mask = _as_mask4d(mask, lab.shape[-2:], lab.size(0))
    ab = lab[:, 1:3]
    lightness = lab[:, 0:1]
    chroma = torch.linalg.vector_norm(ab, dim=1, keepdim=True)
    ab_stats = masked_robust_stats(ab, mask)
    l_stats = masked_robust_stats(lightness, mask)
    c_stats = masked_robust_stats(chroma, mask)
    mean_ab = ab_stats["mean"]
    hue_norm = torch.linalg.vector_norm(mean_ab, dim=1, keepdim=True)
    hue_unit = mean_ab / hue_norm.clamp_min(1e-6)
    neutral_hue = torch.tensor([1.0, 0.0], device=lab.device, dtype=lab.dtype)[None]
    hue_unit = torch.where(hue_norm > 1e-6, hue_unit, neutral_hue)
    median_chroma = c_stats["median"].squeeze(1)
    hue_validity = torch.sigmoid(
        (median_chroma - config.hue_valid_chroma_center)
        / max(config.hue_valid_chroma_softness, 1e-4)
    )
    return {
        "mean_ab": mean_ab,
        "median_ab": ab_stats["median"],
        "mean_chroma": c_stats["mean"].squeeze(1),
        "median_chroma": median_chroma,
        "std_ab": ab_stats["std"],
        "hue_unit": hue_unit,
        "hue_validity": hue_validity,
        "l_median": l_stats["median"].squeeze(1),
        "l_q25": _weighted_quantile(lightness, mask, 0.25).squeeze(1),
        "l_q75": _weighted_quantile(lightness, mask, 0.75).squeeze(1),
        "c_q25": _weighted_quantile(chroma, mask, 0.25).squeeze(1),
        "c_q50": median_chroma,
        "c_q75": _weighted_quantile(chroma, mask, 0.75).squeeze(1),
    }


def compute_composite_color_distance(
    ref_stats: dict[str, torch.Tensor],
    base_stats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    delta_ab = ref_stats["mean_ab"] - base_stats["mean_ab"]
    distance_ab = torch.linalg.vector_norm(delta_ab, dim=1)
    hue_cosine = (ref_stats["hue_unit"] * base_stats["hue_unit"]).sum(dim=1).clamp(-1, 1)
    hue_distance_deg = torch.rad2deg(torch.acos(hue_cosine))
    hue_reliability = torch.minimum(
        ref_stats["hue_validity"], base_stats["hue_validity"]
    )
    chroma_distance = (
        ref_stats["median_chroma"] - base_stats["median_chroma"]
    ).abs()
    distribution_distance = (
        torch.linalg.vector_norm(ref_stats["std_ab"] - base_stats["std_ab"], dim=1)
        + 0.5 * (ref_stats["c_q25"] - base_stats["c_q25"]).abs()
        + 0.5 * (ref_stats["c_q75"] - base_stats["c_q75"]).abs()
    )
    return {
        "delta_ab": delta_ab,
        "distance_ab": distance_ab,
        "hue_cosine": hue_cosine,
        "hue_distance_deg": hue_distance_deg,
        "hue_reliability": hue_reliability,
        "chroma_distance": chroma_distance,
        "distribution_distance": distribution_distance,
    }


def compute_composite_color_gate(
    distances: dict[str, torch.Tensor],
    config: ColorConditionConfigV8 | None = None,
) -> dict[str, torch.Tensor]:
    config = config or ColorConditionConfigV8()
    g_ab = _smoothstep(
        distances["distance_ab"], config.ab_no_edit_threshold, config.ab_full_edit_threshold
    )
    g_hue = _smoothstep(
        distances["hue_distance_deg"], config.hue_no_edit_deg, config.hue_full_edit_deg
    ) * distances["hue_reliability"]
    g_chroma = _smoothstep(
        distances["chroma_distance"], config.chroma_mag_no_edit, config.chroma_mag_full_edit
    )
    g_distribution = _smoothstep(
        distances["distribution_distance"],
        config.color_dist_no_edit,
        config.color_dist_full_edit,
    )
    chroma_gate = 1.0 - (1.0 - g_ab) * (1.0 - g_hue) * (1.0 - g_chroma) * (
        1.0 - g_distribution
    )
    return {
        "chroma_need_gate": chroma_gate.clamp(0, 1),
        "gate_ab": g_ab,
        "gate_hue": g_hue,
        "gate_chroma": g_chroma,
        "gate_distribution": g_distribution,
    }


def compute_reference_fidelity_metrics(
    candidate_lab: torch.Tensor,
    candidate_mask: torch.Tensor,
    ref_stats: dict[str, torch.Tensor],
    config: ColorConditionConfigV8 | None = None,
) -> dict[str, torch.Tensor]:
    candidate_stats = compute_intrinsic_hair_color_stats(candidate_lab, candidate_mask, config)
    hue_cosine = (
        candidate_stats["hue_unit"] * ref_stats["hue_unit"]
    ).sum(dim=1).clamp(-1, 1)
    hue_reliability = torch.minimum(
        candidate_stats["hue_validity"], ref_stats["hue_validity"]
    )
    return {
        "mean_ab_error": torch.linalg.vector_norm(
            candidate_stats["mean_ab"] - ref_stats["mean_ab"], dim=1
        ),
        "hue_error": torch.rad2deg(torch.acos(hue_cosine)) * hue_reliability,
        "chroma_error": (
            candidate_stats["median_chroma"] - ref_stats["median_chroma"]
        ).abs(),
        "median_l_error": (
            candidate_stats["l_median"] - ref_stats["l_median"]
        ).abs(),
        "candidate_stats": candidate_stats,
    }


def reference_color_score(metrics: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        metrics["mean_ab_error"]
        + 0.10 * metrics["hue_error"]
        + 0.50 * metrics["chroma_error"]
    )


def correction_hue_regression_loss(
    anchor_hue_error: torch.Tensor,
    final_hue_error: torch.Tensor,
    tolerance_deg: float = 1.5,
) -> torch.Tensor:
    return torch.relu(final_hue_error - anchor_hue_error - float(tolerance_deg)).mean()


def correction_reference_regression_loss(
    anchor_metrics: dict[str, torch.Tensor],
    final_metrics: dict[str, torch.Tensor],
    tolerance: float = 0.2,
) -> torch.Tensor:
    return torch.relu(
        reference_color_score(final_metrics)
        - reference_color_score(anchor_metrics)
        - float(tolerance)
    ).mean()


def compute_soft_l_shift(delta_l_unclamped: torch.Tensor, max_global_l_shift: float = 40.0) -> torch.Tensor:
    """Bound a global lightness correction smoothly without a hard +/-20 cap."""
    max_shift = max(float(max_global_l_shift), 1e-4)
    return max_shift * torch.tanh(delta_l_unclamped / max_shift)


def _masked_gaussian_mean(value: torch.Tensor, mask: torch.Tensor, radius: int) -> torch.Tensor:
    max_radius = max(min(value.shape[-2:]) - 1, 0)
    radius = min(max(int(radius), 0), max_radius)
    if radius == 0:
        return value
    numerator = gaussian_blur2d(value * mask, radius=radius)
    denominator = gaussian_blur2d(mask, radius=radius).clamp_min(1e-4)
    return numerator / denominator


def _erode_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=2 * width + 1, stride=1, padding=width)


def build_reference_color_safe_mask(
    reference_image: torch.Tensor,
    reference_hair_mask: torch.Tensor,
    config: ColorConditionConfigV8 | None = None,
) -> dict[str, torch.Tensor]:
    config = config or ColorConditionConfigV8()
    reference_rgb, _ = _to_rgb01(reference_image)
    hair_mask = _as_mask4d(
        reference_hair_mask,
        reference_rgb.shape[-2:],
        reference_rgb.size(0),
    )
    eroded = _erode_mask(hair_mask, config.reference_mask_erode)
    eroded_valid = eroded.sum(dim=(-2, -1), keepdim=True) >= 16.0
    stats_mask = torch.where(eroded_valid, eroded, hair_mask)

    reference_lab = rgb_to_lab(reference_rgb)
    lightness = reference_lab[:, 0:1]
    chroma = torch.linalg.vector_norm(reference_lab[:, 1:3], dim=1, keepdim=True)
    l_stats = masked_robust_stats(lightness, stats_mask)
    c_stats = masked_robust_stats(chroma, stats_mask)
    median_l = l_stats["median"][:, :, None, None]
    robust_mad_l = (1.4826 * l_stats["mad"])[:, :, None, None]

    local_l = _masked_gaussian_mean(lightness, hair_mask, config.highlight_local_radius)
    local_c = _masked_gaussian_mean(chroma, hair_mask, config.highlight_local_radius)
    global_margin = torch.maximum(
        config.highlight_mad_scale * robust_mad_l,
        torch.full_like(robust_mad_l, config.highlight_global_min_margin),
    )
    global_bright = torch.sigmoid(
        (lightness - median_l - global_margin) / max(config.highlight_softness_l, 1e-4)
    )
    local_bright = torch.sigmoid(
        (lightness - local_l - config.highlight_local_l_margin)
        / max(config.highlight_softness_l, 1e-4)
    )
    local_desaturation = torch.sigmoid(
        (local_c - chroma - config.highlight_local_c_margin)
        / max(config.highlight_softness_c, 1e-4)
    )
    chroma_ratio = chroma / local_c.clamp_min(1.0)
    ratio_desaturation = torch.sigmoid(
        (config.highlight_chroma_ratio - chroma_ratio) / 0.08
    )
    desaturation = torch.maximum(local_desaturation, ratio_desaturation)

    near_clip = torch.sigmoid((reference_rgb.amax(dim=1, keepdim=True) - config.highlight_near_clip) / 0.015)
    low_clip_chroma = torch.sigmoid(
        (config.highlight_clip_chroma - chroma) / max(config.highlight_softness_c, 1e-4)
    )
    brightness_confidence = torch.maximum(global_bright, local_bright)
    specular_confidence = brightness_confidence * desaturation
    clip_confidence = near_clip * low_clip_chroma * local_bright
    highlight_confidence = torch.maximum(
        specular_confidence,
        clip_confidence,
    ) * hair_mask

    raw_safe_mask = hair_mask * (1.0 - highlight_confidence.clamp(0, 1))
    hair_area = hair_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    raw_safe_fraction = raw_safe_mask.sum(dim=(-2, -1), keepdim=True) / hair_area
    fallback_mix = (
        (config.min_safe_fraction - raw_safe_fraction)
        / (1.0 - raw_safe_fraction).clamp_min(1e-6)
    ).clamp(0, 1)
    safe_mask = raw_safe_mask * (1.0 - fallback_mix) + hair_mask * fallback_mix
    safe_fraction = safe_mask.sum(dim=(-2, -1), keepdim=True) / hair_area
    rejected_mask = (hair_mask - safe_mask).clamp(0, 1)

    return {
        "safe_ref_mask": safe_mask.clamp(0, 1),
        "rejected_highlight_mask": rejected_mask,
        "highlight_confidence": highlight_confidence.clamp(0, 1),
        "safe_fraction": safe_fraction.flatten(1).mean(dim=1),
        "raw_safe_fraction": raw_safe_fraction.flatten(1).mean(dim=1),
        "reference_lab": reference_lab,
        "reference_rgb01": reference_rgb,
        "median_chroma": c_stats["median"].squeeze(1),
    }


def _smoothstep(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    if upper <= lower:
        raise ValueError(f"Smoothstep upper={upper} must be greater than lower={lower}")
    x = ((value - lower) / (upper - lower)).clamp(0, 1)
    return x.square() * (3.0 - 2.0 * x)


def compute_color_need_gates(
    ref_stats: dict[str, torch.Tensor],
    base_stats: dict[str, torch.Tensor],
    delta_l_global: torch.Tensor,
    config: ColorConditionConfigV8 | None = None,
) -> dict[str, torch.Tensor]:
    config = config or ColorConditionConfigV8()
    distances = compute_composite_color_distance(ref_stats, base_stats)
    gate_parts = compute_composite_color_gate(distances, config)
    lightness_need_gate = _smoothstep(
        delta_l_global.abs(),
        config.lightness_no_edit_threshold,
        config.lightness_full_edit_threshold,
    )
    chroma_need_gate = gate_parts["chroma_need_gate"]
    edit_need_gate = 1.0 - (1.0 - chroma_need_gate) * (1.0 - lightness_need_gate)
    return {
        **distances,
        **gate_parts,
        "chroma_need_gate": chroma_need_gate,
        "lightness_need_gate": lightness_need_gate,
        "edit_need_gate": edit_need_gate,
    }


def compute_color_descriptor(
    ref_stats: dict[str, torch.Tensor],
    base_stats: dict[str, torch.Tensor],
    gate_info: dict[str, torch.Tensor],
    delta_l_global: torch.Tensor,
    delta_l_unclamped: torch.Tensor,
    safe_fraction: torch.Tensor,
    relative_luma_reliability: torch.Tensor,
    pseudo_reference_fidelity: torch.Tensor,
) -> torch.Tensor:
    descriptor = torch.cat(
        [
            ref_stats["mean_ab"] / 110.0,
            ref_stats["median_ab"] / 110.0,
            (ref_stats["mean_chroma"] / 110.0).unsqueeze(1),
            (ref_stats["median_chroma"] / 110.0).unsqueeze(1),
            (ref_stats["l_median"] / 100.0).unsqueeze(1),
            ref_stats["hue_unit"],
            ref_stats["std_ab"] / 110.0,
            (ref_stats["c_q25"] / 110.0).unsqueeze(1),
            (ref_stats["c_q75"] / 110.0).unsqueeze(1),
            base_stats["mean_ab"] / 110.0,
            base_stats["median_ab"] / 110.0,
            (base_stats["mean_chroma"] / 110.0).unsqueeze(1),
            (base_stats["median_chroma"] / 110.0).unsqueeze(1),
            (base_stats["l_median"] / 100.0).unsqueeze(1),
            base_stats["hue_unit"],
            base_stats["std_ab"] / 110.0,
            (base_stats["c_q25"] / 110.0).unsqueeze(1),
            (base_stats["c_q75"] / 110.0).unsqueeze(1),
            gate_info["delta_ab"] / 110.0,
            gate_info["hue_cosine"].unsqueeze(1),
            (gate_info["hue_distance_deg"] / 180.0).unsqueeze(1),
            ((ref_stats["median_chroma"] - base_stats["median_chroma"]) / 110.0).unsqueeze(1),
            ((ref_stats["c_q25"] - base_stats["c_q25"]) / 110.0).unsqueeze(1),
            ((ref_stats["c_q75"] - base_stats["c_q75"]) / 110.0).unsqueeze(1),
            (gate_info["distribution_distance"] / 110.0).unsqueeze(1),
            (delta_l_global / 100.0).unsqueeze(1),
            (delta_l_unclamped / 100.0).unsqueeze(1),
            relative_luma_reliability.unsqueeze(1),
            pseudo_reference_fidelity.unsqueeze(1),
            safe_fraction.unsqueeze(1),
            gate_info["chroma_need_gate"].unsqueeze(1),
            gate_info["lightness_need_gate"].unsqueeze(1),
        ],
        dim=1,
    )
    if descriptor.size(1) != COLOR_DESCRIPTOR_DIM:
        raise RuntimeError(f"Expected descriptor dim={COLOR_DESCRIPTOR_DIM}, got {descriptor.size(1)}")

    return descriptor


def relative_luma_conditional_ab(
    target_luma: torch.Tensor,
    ref_luma: torch.Tensor,
    ref_ab: torch.Tensor,
    ref_mask: torch.Tensor,
    target_mask: torch.Tensor,
    ref_stats: dict[str, torch.Tensor],
    base_stats: dict[str, torch.Tensor],
    config: ColorConditionConfigV8 | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config = config or ColorConditionConfigV8()
    bins = max(int(config.relative_luma_bins), 3)
    centers = torch.linspace(-2.5, 2.5, bins, device=ref_ab.device, dtype=ref_ab.dtype).view(
        1, bins, 1, 1
    )
    sigma = 3.0 / max(bins - 1, 1)
    ref_mask = _as_mask4d(ref_mask, ref_ab.shape[-2:], ref_ab.size(0))
    target_mask = _as_mask4d(target_mask, target_luma.shape[-2:], target_luma.size(0))
    ref_scale = (
        0.7413 * (ref_stats["l_q75"] - ref_stats["l_q25"])
    ).clamp_min(config.relative_luma_min_scale)
    base_scale = (
        0.7413 * (base_stats["l_q75"] - base_stats["l_q25"])
    ).clamp_min(config.relative_luma_min_scale)
    ref_z = (ref_luma - ref_stats["l_median"][:, None, None, None]) / ref_scale[
        :, None, None, None
    ]
    target_z = (target_luma - base_stats["l_median"][:, None, None, None]) / base_scale[
        :, None, None, None
    ]
    ref_weights = torch.exp(-0.5 * ((ref_z - centers) / sigma).square()) * ref_mask
    ref_weights_5d = ref_weights.unsqueeze(2)
    denominator_raw = ref_weights_5d.sum(dim=(-2, -1), keepdim=True)
    denominator = denominator_raw.clamp_min(1e-6)
    bin_ab = (ref_ab.unsqueeze(1) * ref_weights_5d).sum(dim=(-2, -1), keepdim=True) / denominator
    global_ab_value = 0.5 * (ref_stats["mean_ab"] + ref_stats["median_ab"])
    global_ab = global_ab_value[:, None, :, None, None]
    valid_bins = (denominator_raw > 8.0).to(ref_ab.dtype)
    bin_ab = torch.where(valid_bins.bool(), bin_ab, global_ab)
    target_weights = torch.exp(-0.5 * ((target_z - centers) / sigma).square()).unsqueeze(2)
    target_weight_sum = target_weights.sum(dim=1).clamp_min(1e-6)
    conditional_ab = (bin_ab * target_weights).sum(dim=1) / target_weight_sum
    bin_reliability = (valid_bins * target_weights).sum(dim=1) / target_weight_sum
    ref_z_q05 = _weighted_quantile(ref_z, ref_mask, 0.05)[:, :, None, None]
    ref_z_q95 = _weighted_quantile(ref_z, ref_mask, 0.95)[:, :, None, None]
    coverage = torch.sigmoid((target_z - ref_z_q05) / 0.25) * torch.sigmoid(
        (ref_z_q95 - target_z) / 0.25
    )
    reliability_map = (bin_reliability * coverage).clamp(0, 1)
    reliability = (
        (reliability_map * target_mask).sum(dim=(-2, -1))
        / target_mask.sum(dim=(-2, -1)).clamp_min(1.0)
    ).squeeze(1)
    fallback_threshold = max(config.global_ab_fallback_min_reliability, 1e-4)
    sample_conditional_weight = (reliability / fallback_threshold).clamp(0, 1)
    effective_reliability_map = (
        reliability_map * sample_conditional_weight[:, None, None, None]
    )
    fallback_ab = global_ab_value[:, :, None, None]
    target_ref_ab = (
        effective_reliability_map * conditional_ab
        + (1.0 - effective_reliability_map) * fallback_ab
    )
    return target_ref_ab, reliability, effective_reliability_map


def build_pseudo_color_target(
    reference_lab: torch.Tensor,
    safe_ref_mask: torch.Tensor,
    base_lab: torch.Tensor,
    target_hair_mask: torch.Tensor,
    delta_l_global: torch.Tensor,
    ref_stats: dict[str, torch.Tensor],
    base_stats: dict[str, torch.Tensor],
    config: ColorConditionConfigV8 | None = None,
) -> dict[str, torch.Tensor]:
    config = config or ColorConditionConfigV8()
    target_hair_mask = _as_mask4d(target_hair_mask, base_lab.shape[-2:], base_lab.size(0))
    target_ref_ab, relative_reliability, reliability_map = relative_luma_conditional_ab(
        target_luma=base_lab[:, 0:1],
        ref_luma=reference_lab[:, 0:1],
        ref_ab=reference_lab[:, 1:3],
        ref_mask=safe_ref_mask,
        target_mask=target_hair_mask,
        ref_stats=ref_stats,
        base_stats=base_stats,
        config=config,
    )
    delta_l = delta_l_global[:, None, None, None]
    candidate_ab = base_lab[:, 1:3] + target_hair_mask * (target_ref_ab - base_lab[:, 1:3])
    pseudo_l = base_lab[:, 0:1] + target_hair_mask * delta_l
    candidate_lab = torch.cat([pseudo_l.clamp(0, 100), candidate_ab], dim=1)
    candidate_fidelity = compute_reference_fidelity_metrics(
        candidate_lab, target_hair_mask, ref_stats, config
    )
    analytic_fidelity = torch.exp(
        -candidate_fidelity["mean_ab_error"] / max(config.pseudo_fidelity_ab_bad, 1e-4)
        -candidate_fidelity["hue_error"] / max(config.pseudo_fidelity_hue_bad, 1e-4)
    ).detach().clamp(0, 1)
    global_ab = 0.5 * (ref_stats["mean_ab"] + ref_stats["median_ab"])
    guarded_target_ab = (
        analytic_fidelity[:, None, None, None] * target_ref_ab
        + (1.0 - analytic_fidelity[:, None, None, None]) * global_ab[:, :, None, None]
    )
    pseudo_ab = base_lab[:, 1:3] + target_hair_mask * (
        guarded_target_ab - base_lab[:, 1:3]
    )
    pseudo_lab = torch.cat([pseudo_l.clamp(0, 100), pseudo_ab], dim=1)
    final_fidelity = compute_reference_fidelity_metrics(
        pseudo_lab, target_hair_mask, ref_stats, config
    )
    pseudo_reference_fidelity = torch.exp(
        -final_fidelity["mean_ab_error"] / max(config.pseudo_fidelity_ab_bad, 1e-4)
        -final_fidelity["hue_error"] / max(config.pseudo_fidelity_hue_bad, 1e-4)
    ).clamp(0, 1)
    return {
        "pseudo_lab": pseudo_lab,
        "pseudo_rgb01": lab_to_rgb(pseudo_lab),
        "target_ref_ab": guarded_target_ab,
        "relative_luma_reliability": relative_reliability,
        "relative_luma_reliability_map": reliability_map,
        "pseudo_reference_fidelity": pseudo_reference_fidelity,
        "fidelity_metrics": final_fidelity,
    }


def build_reference_color_proxy(
    reference_image: torch.Tensor,
    reference_hair_mask: torch.Tensor,
    reference_lab: torch.Tensor,
    safe_ref_mask: torch.Tensor,
) -> torch.Tensor:
    _, normalized = _to_rgb01(reference_image)
    hair_mask = _as_mask4d(
        reference_hair_mask,
        reference_lab.shape[-2:],
        reference_lab.size(0),
    )
    l_stats = masked_robust_stats(reference_lab[:, 0:1], safe_ref_mask)
    ab_stats = masked_robust_stats(reference_lab[:, 1:3], safe_ref_mask)
    proxy_lab = torch.cat(
        [
            l_stats["median"][:, :, None, None].expand(-1, -1, *reference_lab.shape[-2:]),
            ab_stats["mean"][:, :, None, None].expand(-1, -1, *reference_lab.shape[-2:]),
        ],
        dim=1,
    )
    flat_hair_rgb = lab_to_rgb(proxy_lab)
    neutral_background = torch.full_like(flat_hair_rgb, 0.5)
    proxy_rgb01 = flat_hair_rgb * hair_mask + neutral_background * (1.0 - hair_mask)
    return _from_rgb01(proxy_rgb01, normalized)


def build_color_condition_bundle(
    reference_image: torch.Tensor,
    reference_hair_mask: torch.Tensor,
    base_image: torch.Tensor,
    target_hair_mask: torch.Tensor,
    config: ColorConditionConfigV8 | None = None,
) -> dict[str, object]:
    """Build the shared training/inference color condition for BlendingV8."""
    config = config or ColorConditionConfigV8()
    reference_image = _as_image4d(reference_image)
    base_image = _as_image4d(base_image)
    if reference_image.size(0) != base_image.size(0):
        raise ValueError("Reference and base batch sizes must match")
    if reference_image.shape[-2:] != base_image.shape[-2:]:
        reference_image = F.interpolate(reference_image, size=base_image.shape[-2:], mode="bilinear", align_corners=False)

    safe_info = build_reference_color_safe_mask(reference_image, reference_hair_mask, config)
    base_rgb01, base_normalized = _to_rgb01(base_image)
    base_lab = rgb_to_lab(base_rgb01)
    target_hair_mask = _as_mask4d(target_hair_mask, base_lab.shape[-2:], base_lab.size(0))
    ref_stats = compute_intrinsic_hair_color_stats(
        safe_info["reference_lab"], safe_info["safe_ref_mask"], config
    )
    base_stats = compute_intrinsic_hair_color_stats(base_lab, target_hair_mask, config)
    delta_l_unclamped = ref_stats["l_median"] - base_stats["l_median"]
    delta_l_global = compute_soft_l_shift(delta_l_unclamped, config.max_global_l_shift)
    gate_info = compute_color_need_gates(ref_stats, base_stats, delta_l_global, config)
    pseudo = build_pseudo_color_target(
        reference_lab=safe_info["reference_lab"],
        safe_ref_mask=safe_info["safe_ref_mask"],
        base_lab=base_lab,
        target_hair_mask=target_hair_mask,
        delta_l_global=delta_l_global,
        ref_stats=ref_stats,
        base_stats=base_stats,
        config=config,
    )
    descriptor = compute_color_descriptor(
        ref_stats=ref_stats,
        base_stats=base_stats,
        gate_info=gate_info,
        delta_l_global=delta_l_global,
        delta_l_unclamped=delta_l_unclamped,
        safe_fraction=safe_info["safe_fraction"],
        relative_luma_reliability=pseudo["relative_luma_reliability"],
        pseudo_reference_fidelity=pseudo["pseudo_reference_fidelity"],
    )
    color_proxy = build_reference_color_proxy(
        reference_image,
        reference_hair_mask,
        safe_info["reference_lab"],
        safe_info["safe_ref_mask"],
    )
    composite_color_distance = (
        gate_info["distance_ab"]
        + 0.10 * gate_info["hue_distance_deg"] * gate_info["hue_reliability"]
        + 0.50 * gate_info["chroma_distance"]
        + gate_info["distribution_distance"]
    )
    metrics = {
        "ref_mean_ab": ref_stats["mean_ab"],
        "base_mean_ab": base_stats["mean_ab"],
        "ref_median_l": ref_stats["l_median"],
        "base_median_l": base_stats["l_median"],
        "delta_l_global": delta_l_global,
        "delta_l_unclamped": delta_l_unclamped,
        "ref_base_ab_distance": gate_info["distance_ab"],
        "ref_base_global_l_distance": delta_l_global.abs(),
        "composite_color_distance": composite_color_distance,
        "hue_distance_deg": gate_info["hue_distance_deg"],
        "chroma_distance": gate_info["chroma_distance"],
        "distribution_distance": gate_info["distribution_distance"],
        "relative_luma_reliability": pseudo["relative_luma_reliability"],
        "pseudo_reference_fidelity": pseudo["pseudo_reference_fidelity"],
        "pseudo_to_reference_mean_ab": pseudo["fidelity_metrics"]["mean_ab_error"],
        "pseudo_to_reference_hue_error": pseudo["fidelity_metrics"]["hue_error"],
        "pseudo_to_reference_chroma_error": pseudo["fidelity_metrics"]["chroma_error"],
        "pseudo_to_reference_median_l_error": pseudo["fidelity_metrics"]["median_l_error"],
        "chroma_need_gate": gate_info["chroma_need_gate"],
        "lightness_need_gate": gate_info["lightness_need_gate"],
        "edit_need_gate": gate_info["edit_need_gate"],
        "safe_fraction": safe_info["safe_fraction"],
        "raw_safe_fraction": safe_info["raw_safe_fraction"],
        "rejected_fraction": 1.0 - safe_info["safe_fraction"],
    }
    return {
        "safe_ref_mask": safe_info["safe_ref_mask"],
        "rejected_highlight_mask": safe_info["rejected_highlight_mask"],
        "descriptor": descriptor,
        "chroma_need_gate": metrics["chroma_need_gate"],
        "lightness_need_gate": metrics["lightness_need_gate"],
        "edit_need_gate": metrics["edit_need_gate"],
        "pseudo_lab": pseudo["pseudo_lab"],
        "pseudo_rgb": _from_rgb01(pseudo["pseudo_rgb01"], base_normalized),
        "color_proxy": color_proxy,
        "target_ref_ab": pseudo["target_ref_ab"],
        "relative_luma_reliability": pseudo["relative_luma_reliability"],
        "relative_luma_reliability_map": pseudo["relative_luma_reliability_map"],
        "pseudo_reference_fidelity": pseudo["pseudo_reference_fidelity"],
        "ref_stats": ref_stats,
        "base_stats": base_stats,
        "composite_color_distance": composite_color_distance,
        "hue_distance_deg": gate_info["hue_distance_deg"],
        "chroma_distance": gate_info["chroma_distance"],
        "distribution_distance": gate_info["distribution_distance"],
        "base_lab": base_lab,
        "metrics": metrics,
    }
