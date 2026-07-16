from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.pp_losses import LossBuilderMulti
from models.ear_modules_v5 import (
    RAW_DETAIL_LABELS,
    RAW_FACE_SURFACE_LABELS,
    build_weak_earring_mask,
    dilate_mask,
    erode_mask,
    ensure_mask_4d,
    high_pass_filter,
    low_pass_filter,
    resize_mask,
)

CLEANUP_MASK_KEYS = ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    gray = rgb_to_gray(image)
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=image.device, dtype=image.dtype)
    kernel_y = kernel_x.t()
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    if value.ndim == 4 and mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = resize_mask(mask, pred.shape[-2:])
    return masked_mean((pred - target).abs(), mask)


def masked_ratio(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = resize_mask(mask, value.shape[-2:])
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def masked_non_darker(pred_gray: torch.Tensor, anchor_gray: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = resize_mask(mask, pred_gray.shape[-2:])
    return masked_mean(F.relu(anchor_gray - pred_gray), mask)


def parsing_label_mask(parsing: torch.Tensor | None, labels: tuple[int, ...]) -> torch.Tensor | None:
    if parsing is None:
        return None
    parsing = ensure_mask_4d(parsing).long()
    mask = torch.zeros_like(parsing, dtype=torch.bool)
    for label in labels:
        mask |= parsing == label
    return mask.float()


def combine_aux_masks(aux: dict, keys: tuple[str, ...]) -> torch.Tensor | None:
    masks = []
    for key in keys:
        value = aux.get(key)
        if value is not None:
            masks.append(ensure_mask_4d(value).float())
    if not masks:
        return None
    return torch.stack(masks, dim=0).amax(dim=0).clamp(0, 1)


def source_direction_leak_loss(
    source: torch.Tensor,
    anchor: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = resize_mask(mask, pred.shape[-2:])
    if source.shape[-2:] != pred.shape[-2:]:
        source = F.interpolate(source, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    if anchor.shape[-2:] != pred.shape[-2:]:
        anchor = F.interpolate(anchor, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    source_delta = (source - anchor).detach()
    pred_delta = pred - anchor
    source_norm = source_delta.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-4)
    source_dir = source_delta / source_norm
    copied_toward_source = F.relu((pred_delta * source_dir).sum(dim=1, keepdim=True))
    return masked_mean(copied_toward_source * source_norm.clamp(max=1.0), mask)


def masked_scalar_stats(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = resize_mask(mask, value.shape[-2:])
    if value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    denom = mask.sum().clamp_min(1.0)
    mean = (value * mask).sum() / denom
    var = ((value - mean).pow(2) * mask).sum() / denom
    return mean, torch.sqrt(var + 1e-6)


def masked_channel_stats(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = resize_mask(mask, value.shape[-2:])
    if mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = (value * mask).sum(dim=(2, 3), keepdim=True) / denom
    var = ((value - mean).pow(2) * mask).sum(dim=(2, 3), keepdim=True) / denom
    return mean, torch.sqrt(var + 1e-6)


def masked_channel_stats_per_sample(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_pixels: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = resize_mask(mask, value.shape[-2:])
    spatial_mask = mask[:, :1] if mask.size(1) == 1 else mask.mean(dim=1, keepdim=True)
    valid = (spatial_mask.sum(dim=(2, 3), keepdim=True) >= min_pixels).float()
    if mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = (value * mask).sum(dim=(2, 3), keepdim=True) / denom
    var = ((value - mean).pow(2) * mask).sum(dim=(2, 3), keepdim=True) / denom
    return mean, torch.sqrt(var + 1e-6), valid


def texture_stat_loss(
    pred: torch.Tensor,
    reference: torch.Tensor,
    pred_mask: torch.Tensor,
    reference_mask: torch.Tensor,
) -> torch.Tensor:
    pred_mask = resize_mask(pred_mask, pred.shape[-2:])
    reference_mask = resize_mask(reference_mask, reference.shape[-2:])
    if pred_mask.detach().sum().item() < 1 or reference_mask.detach().sum().item() < 1:
        return pred.sum() * 0.0

    pred_high_energy = high_pass_filter(pred).abs().mean(dim=1, keepdim=True)
    ref_high_energy = high_pass_filter(reference).abs().mean(dim=1, keepdim=True)
    pred_edge = sobel_edges(pred)
    ref_edge = sobel_edges(reference)

    pred_high_mean, pred_high_std = masked_scalar_stats(pred_high_energy, pred_mask)
    ref_high_mean, ref_high_std = masked_scalar_stats(ref_high_energy, reference_mask)
    pred_edge_mean, pred_edge_std = masked_scalar_stats(pred_edge, pred_mask)
    ref_edge_mean, ref_edge_std = masked_scalar_stats(ref_edge, reference_mask)
    return (
        (pred_high_mean - ref_high_mean.detach()).abs()
        + (pred_high_std - ref_high_std.detach()).abs()
        + 0.5 * (pred_edge_mean - ref_edge_mean.detach()).abs()
        + 0.5 * (pred_edge_std - ref_edge_std.detach()).abs()
    )


def low_color_stat_loss(
    pred: torch.Tensor,
    reference: torch.Tensor,
    pred_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    kernel_size: int = 31,
) -> torch.Tensor:
    pred_mask = resize_mask(pred_mask, pred.shape[-2:])
    reference_mask = resize_mask(reference_mask, reference.shape[-2:])
    if pred_mask.detach().sum().item() < 1 or reference_mask.detach().sum().item() < 1:
        return pred.sum() * 0.0
    pred_low = low_pass_filter(pred, kernel_size=kernel_size)
    ref_low = low_pass_filter(reference, kernel_size=kernel_size)
    pred_mean, pred_std, pred_valid = masked_channel_stats_per_sample(pred_low, pred_mask)
    ref_mean, ref_std, ref_valid = masked_channel_stats_per_sample(ref_low, reference_mask)
    valid = (pred_valid * ref_valid).detach()
    if valid.sum().item() < 1:
        return pred.sum() * 0.0
    loss_per_sample = (
        (pred_mean - ref_mean.detach()).abs().mean(dim=1, keepdim=True)
        + 0.25 * (pred_std - ref_std.detach()).abs().mean(dim=1, keepdim=True)
    )
    return (loss_per_sample * valid).sum() / valid.sum().clamp_min(1.0)


def low_gray_floor_loss(
    pred: torch.Tensor,
    reference: torch.Tensor,
    pred_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    kernel_size: int = 31,
    std_scale: float = 0.75,
) -> torch.Tensor:
    pred_mask = resize_mask(pred_mask, pred.shape[-2:])
    reference_mask = resize_mask(reference_mask, reference.shape[-2:])
    if pred_mask.detach().sum().item() < 1 or reference_mask.detach().sum().item() < 1:
        return pred.sum() * 0.0
    pred_gray = rgb_to_gray(low_pass_filter(pred, kernel_size=kernel_size))
    ref_gray = rgb_to_gray(low_pass_filter(reference, kernel_size=kernel_size))
    _, _, pred_valid = masked_channel_stats_per_sample(pred_gray, pred_mask)
    ref_mean, ref_std, ref_valid = masked_channel_stats_per_sample(ref_gray, reference_mask)
    valid = (pred_valid * ref_valid).detach()
    if valid.sum().item() < 1:
        return pred.sum() * 0.0
    floor = (ref_mean - std_scale * ref_std).detach()
    return masked_mean(F.relu(floor - pred_gray), pred_mask * valid)


def source_dark_high_reject_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float = 0.006,
    source_threshold: float = 0.018,
    kernel_size: int = 11,
) -> torch.Tensor:
    mask = resize_mask(mask, pred.shape[-2:])
    if source.shape[-2:] != pred.shape[-2:]:
        source = F.interpolate(source, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    source_gray = rgb_to_gray(source)
    target_gray = rgb_to_gray(target)
    pred_gray = rgb_to_gray(pred)
    source_dark = (low_pass_filter(source_gray, kernel_size=kernel_size) - source_gray).clamp(0, 1).detach()
    target_dark = (low_pass_filter(target_gray, kernel_size=kernel_size) - target_gray).clamp(0, 1).detach()
    pred_dark = (low_pass_filter(pred_gray, kernel_size=kernel_size) - pred_gray).clamp(0, 1)
    source_gate = (source_dark > source_threshold).float()
    excess_dark = F.relu(pred_dark - target_dark - margin)
    return masked_mean(excess_dark * source_gate, mask)


def build_weak_ear_pseudo_mask(
    source: torch.Tensor,
    query_mask: torch.Tensor,
    parser_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = build_weak_earring_mask(source, query_mask, parser_mask)
    return outputs["weak_earring_mask"], outputs["weak_high_energy"]


def build_weak_presence_target(pseudo_mask: torch.Tensor, parser_mask: torch.Tensor | None = None) -> torch.Tensor:
    pseudo_mask = ensure_mask_4d(pseudo_mask).float()
    width = pseudo_mask.size(-1)
    x_coords = torch.linspace(0, 1, width, device=pseudo_mask.device, dtype=pseudo_mask.dtype).view(1, 1, 1, width)
    left_half = (x_coords <= 0.5).float()
    right_half = 1 - left_half

    left_presence = (pseudo_mask * left_half).flatten(1).amax(dim=1)
    right_presence = (pseudo_mask * right_half).flatten(1).amax(dim=1)

    if parser_mask is not None:
        parser_mask = resize_mask(parser_mask, pseudo_mask.shape[-2:])
        left_presence = torch.maximum(left_presence, (parser_mask * left_half).flatten(1).amax(dim=1))
        right_presence = torch.maximum(right_presence, (parser_mask * right_half).flatten(1).amax(dim=1))

    any_presence = torch.maximum(left_presence, right_presence)
    return torch.stack([left_presence, right_presence, any_presence], dim=1).float()


class EarAwareLossBuilder(LossBuilderMulti):
    def __init__(self, losses_dict, device: str = "cuda"):
        super().__init__(losses_dict, device=device)
        self.mask_bce = nn.BCEWithLogitsLoss(reduction="none")
        self.presence_bce = nn.BCEWithLogitsLoss()

    def __call__(self, source, target, target_mask, HT_E, gen_w, F_w, gen_F, F_gen, aux=None, **kwargs):
        cleanup_mask = combine_aux_masks(aux, CLEANUP_MASK_KEYS) if aux is not None else None
        base_target_mask = target_mask
        base_exclude = None
        if cleanup_mask is not None:
            cleanup_for_base = resize_mask(cleanup_mask, target_mask.shape[-2:])
            exclude_strength = max(0.0, min(1.0, float(self.losses_dict.get("base_cleanup_exclude", 1.0))))
            base_exclude = exclude_strength * cleanup_for_base
        if aux is not None:
            source_hair_for_base = aux.get("source_hair_mask")
            target_face_for_base = aux.get("target_face_surface_mask")
            if source_hair_for_base is not None and target_face_for_base is not None:
                base_size = target_mask.shape[-2:]
                source_hair_risk = resize_mask(source_hair_for_base, base_size)
                dilate = int(self.losses_dict.get("base_source_hair_exclude_dilate", 9))
                if dilate > 0:
                    source_hair_risk = dilate_mask(source_hair_risk, dilate)
                source_hair_risk = source_hair_risk * resize_mask(target_face_for_base, base_size)

                max_y = max(0.0, min(1.0, float(self.losses_dict.get("base_source_hair_exclude_max_y", 0.55))))
                if max_y < 1.0:
                    y_coords = torch.linspace(
                        0,
                        1,
                        base_size[0],
                        device=source_hair_risk.device,
                        dtype=source_hair_risk.dtype,
                    ).view(1, 1, base_size[0], 1)
                    source_hair_risk = source_hair_risk * (y_coords <= max_y).float()

                target_hair_for_base = aux.get("target_hair_mask")
                if target_hair_for_base is not None:
                    source_hair_risk = source_hair_risk * (
                        1.0 - resize_mask(target_hair_for_base, base_size)
                    ).clamp(0, 1)

                ear_exclude = torch.zeros_like(source_hair_risk)
                for key in ("ear_roi", "visible_ear_roi", "earring_confident_mask", "source_earring_mask"):
                    value = aux.get(key)
                    if value is not None:
                        ear_exclude = torch.clamp(ear_exclude + resize_mask(value, base_size), 0, 1)
                ear_dilate = int(self.losses_dict.get("base_source_hair_exclude_ear_dilate", 9))
                if ear_dilate > 0:
                    ear_exclude = dilate_mask(ear_exclude, ear_dilate)
                source_hair_risk = source_hair_risk * (1.0 - ear_exclude).clamp(0, 1)

                source_hair_strength = max(
                    0.0,
                    min(1.0, float(self.losses_dict.get("base_source_hair_exclude_strength", 1.0))),
                )
                source_hair_exclude = source_hair_strength * source_hair_risk
                base_exclude = source_hair_exclude if base_exclude is None else torch.clamp(base_exclude + source_hair_exclude, 0, 1)
        if base_exclude is not None:
            base_target_mask = ensure_mask_4d(target_mask).float() * (1.0 - base_exclude).clamp(0, 1)

        losses = super().__call__(source, target, base_target_mask, HT_E, gen_w, F_w, gen_F, F_gen, **kwargs)
        if aux is None:
            return losses

        gen_F_256 = self.downsample_256(gen_F)
        gen_F_256_01 = ((gen_F_256 + 1) / 2).clamp(0, 1)

        uses_safe_query = aux.get("ear_detail_query_mask") is not None or aux.get("hair_safe_query_mask") is not None
        query_mask = aux.get("ear_detail_query_mask", aux.get("hair_safe_query_mask", aux.get("query_mask")))
        source_ear_mask = aux.get("source_earring_mask")
        earring_confident_mask = aux.get("earring_confident_mask", source_ear_mask)
        earring_highlight_mask = aux.get("earring_highlight_mask")
        earring_reference = aux.get("earring_reference", source)
        revealed_skin_mask = aux.get("revealed_skin_mask")
        source_skin_valid_mask = aux.get("source_skin_valid_mask")
        fine_mask = aux.get("fine_mask")
        fine_mask_logits = aux.get("fine_mask_logits")
        presence_logits = aux.get("presence_logits")
        presence_target = aux.get("presence_target")
        visible_ear_roi = aux.get("visible_ear_roi")
        earring_valid_roi = aux.get("earring_valid_roi", aux.get("ear_roi", visible_ear_roi))
        source_hair_block_mask = aux.get("source_hair_block_mask")
        target_earring_suppress_mask = aux.get("target_earring_suppress_mask")
        target_clean_01 = aux.get("target_clean_01")
        gamma = aux.get("gamma")
        beta = aux.get("beta")
        target_hair_ear_block_mask = aux.get("target_hair_ear_block_mask")
        earring_mask_is_dataset = aux.get("earring_mask_is_dataset")

        if query_mask is None or source_ear_mask is None:
            return losses

        if torch.is_tensor(earring_mask_is_dataset):
            dataset_gate = earring_mask_is_dataset.float().view(-1, 1, 1, 1) > 0.5
        else:
            dataset_gate = None

        if earring_valid_roi is not None:
            earring_valid_roi = resize_mask(earring_valid_roi, query_mask.shape[-2:])
            dataset_valid_roi = torch.ones_like(earring_valid_roi)
            query_mask = ensure_mask_4d(query_mask).float() * earring_valid_roi
            source_ear_mask_raw = ensure_mask_4d(source_ear_mask).float()
            earring_confident_raw = ensure_mask_4d(earring_confident_mask).float()
            if dataset_gate is None:
                source_ear_mask = source_ear_mask_raw * earring_valid_roi
                earring_confident_mask = earring_confident_raw * earring_valid_roi
            else:
                source_ear_mask = torch.where(
                    dataset_gate,
                    source_ear_mask_raw * dataset_valid_roi,
                    source_ear_mask_raw * earring_valid_roi,
                )
                earring_confident_mask = torch.where(
                    dataset_gate,
                    earring_confident_raw * dataset_valid_roi,
                    earring_confident_raw * earring_valid_roi,
                )

        if source_hair_block_mask is not None and not uses_safe_query:
            source_hair_block_mask = resize_mask(source_hair_block_mask, query_mask.shape[-2:])
            block_strength = max(0.0, min(1.0, float(self.losses_dict.get("ear_block_strength", 0.95))))
            query_mask = query_mask * (1 - block_strength * source_hair_block_mask).clamp(0, 1)

        cleanup_regular_mask = cleanup_mask
        revealed_skin_mask_256 = None
        if revealed_skin_mask is not None:
            revealed_skin_mask_256 = resize_mask(revealed_skin_mask, gen_F_256_01.shape[-2:])
            if cleanup_regular_mask is not None:
                cleanup_regular_mask = resize_mask(cleanup_regular_mask, gen_F_256_01.shape[-2:])
                cleanup_regular_mask = cleanup_regular_mask * (1.0 - revealed_skin_mask_256).clamp(0, 1)

        source_face_surface = parsing_label_mask(aux.get("source_parsing"), RAW_FACE_SURFACE_LABELS)
        target_face_surface = parsing_label_mask(aux.get("target_parsing"), RAW_FACE_SURFACE_LABELS)
        source_hair_for_skin = aux.get("source_hair_mask")
        target_hair_for_skin = aux.get("target_hair_mask")
        source_visible_skin_ref = source_face_surface
        if source_visible_skin_ref is not None:
            source_visible_skin_ref = resize_mask(source_visible_skin_ref, gen_F_256_01.shape[-2:])
            if source_hair_for_skin is not None:
                source_visible_skin_ref = source_visible_skin_ref * (
                    1.0 - resize_mask(source_hair_for_skin, gen_F_256_01.shape[-2:])
                ).clamp(0, 1)
            if target_hair_for_skin is not None:
                source_visible_skin_ref = source_visible_skin_ref * (
                    1.0 - resize_mask(target_hair_for_skin, gen_F_256_01.shape[-2:])
                ).clamp(0, 1)
            if cleanup_mask is not None:
                source_visible_skin_ref = source_visible_skin_ref * (
                    1.0 - resize_mask(cleanup_mask, gen_F_256_01.shape[-2:])
                ).clamp(0, 1)
            if revealed_skin_mask_256 is not None:
                source_visible_skin_ref = source_visible_skin_ref * (1.0 - revealed_skin_mask_256).clamp(0, 1)

        if cleanup_mask is not None:
            cleanup_mask = resize_mask(cleanup_mask, gen_F_256_01.shape[-2:])
            if cleanup_regular_mask is None:
                cleanup_regular_mask = cleanup_mask
            cleanup_low_anchor_weight = self.losses_dict.get(
                "cleanup_low_anchor",
                self.losses_dict.get("cleanup_anchor", 0.0),
            )
            face_surface = source_face_surface
            source_hair_for_texture = source_hair_for_skin
            target_hair_for_texture = target_hair_for_skin
            cleanup_source_skin = cleanup_regular_mask
            if face_surface is not None:
                cleanup_source_skin = cleanup_source_skin * resize_mask(face_surface, gen_F_256_01.shape[-2:])
            if source_hair_for_texture is not None:
                cleanup_source_skin = cleanup_source_skin * (
                    1.0 - resize_mask(source_hair_for_texture, gen_F_256_01.shape[-2:])
                ).clamp(0, 1)
            if target_hair_for_texture is not None:
                cleanup_source_skin = cleanup_source_skin * (
                    1.0 - resize_mask(target_hair_for_texture, gen_F_256_01.shape[-2:])
                ).clamp(0, 1)
            cleanup_invalid_skin = cleanup_regular_mask * (1.0 - cleanup_source_skin).clamp(0, 1)
            if cleanup_low_anchor_weight > 0:
                cleanup_low_loss = masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(source),
                    cleanup_source_skin,
                )
                if source_visible_skin_ref is not None:
                    cleanup_low_loss = cleanup_low_loss + low_color_stat_loss(
                        gen_F_256_01,
                        source,
                        cleanup_invalid_skin,
                        source_visible_skin_ref,
                    )
                losses["cleanup_low_anchor"] = cleanup_low_anchor_weight * cleanup_low_loss

            cleanup_source_reject_weight = self.losses_dict.get("cleanup_source_reject", 0.0)
            if cleanup_source_reject_weight > 0:
                cleanup_source_reject_mask = cleanup_invalid_skin
                if source_hair_for_texture is not None:
                    cleanup_source_reject_mask = cleanup_source_reject_mask * dilate_mask(
                        resize_mask(source_hair_for_texture, gen_F_256_01.shape[-2:]),
                        int(self.losses_dict.get("face_source_dark_source_hair_dilate", 7)),
                    )
                losses["cleanup_source_reject"] = cleanup_source_reject_weight * source_direction_leak_loss(
                    source,
                    target,
                    gen_F_256_01,
                    cleanup_source_reject_mask,
                )

            cleanup_non_dark_weight = self.losses_dict.get("cleanup_non_dark", 0.0)
            if cleanup_non_dark_weight > 0:
                cleanup_non_dark_loss = masked_non_darker(
                    rgb_to_gray(gen_F_256_01),
                    rgb_to_gray(source),
                    cleanup_source_skin,
                )
                if source_visible_skin_ref is not None:
                    cleanup_non_dark_loss = cleanup_non_dark_loss + low_gray_floor_loss(
                        gen_F_256_01,
                        source,
                        cleanup_invalid_skin,
                        source_visible_skin_ref,
                    )
                losses["cleanup_non_dark"] = cleanup_non_dark_weight * cleanup_non_dark_loss

            if face_surface is not None:
                visible_skin_ref = resize_mask(face_surface, gen_F_256_01.shape[-2:])
                if source_hair_for_texture is not None:
                    visible_skin_ref = visible_skin_ref * (
                        1.0 - resize_mask(source_hair_for_texture, gen_F_256_01.shape[-2:])
                    ).clamp(0, 1)
                if target_hair_for_texture is not None:
                    visible_skin_ref = visible_skin_ref * (
                        1.0 - resize_mask(target_hair_for_texture, gen_F_256_01.shape[-2:])
                    ).clamp(0, 1)
                visible_skin_ref = visible_skin_ref * (1.0 - cleanup_mask).clamp(0, 1)

                cleanup_texture_weight = self.losses_dict.get("cleanup_texture_stat", 0.0)
                if cleanup_texture_weight > 0:
                    losses["cleanup_texture_stat"] = cleanup_texture_weight * texture_stat_loss(
                        gen_F_256_01,
                        source,
                        cleanup_regular_mask,
                        visible_skin_ref,
                    )

                cleanup_high_weight = self.losses_dict.get("cleanup_high", 0.0)
                if cleanup_high_weight > 0:
                    losses["cleanup_high"] = cleanup_high_weight * masked_l1(
                        high_pass_filter(gen_F_256_01),
                        high_pass_filter(source),
                        cleanup_source_skin,
                    )

        if revealed_skin_mask_256 is not None:
            revealed_low_weight = self.losses_dict.get("revealed_skin_low_anchor", 0.0)
            if revealed_low_weight > 0:
                revealed_low_loss = gen_F_256_01.sum() * 0.0
                if source_visible_skin_ref is not None:
                    revealed_low_loss = revealed_low_loss + low_color_stat_loss(
                        gen_F_256_01,
                        source,
                        revealed_skin_mask_256,
                        source_visible_skin_ref,
                    )
                losses["revealed_skin_low_anchor"] = revealed_low_weight * revealed_low_loss

            face_surface = source_face_surface
            visible_skin_ref = source_visible_skin_ref
            if visible_skin_ref is not None:
                revealed_texture_weight = self.losses_dict.get("revealed_skin_texture", 0.0)
                if revealed_texture_weight > 0:
                    losses["revealed_skin_texture"] = revealed_texture_weight * texture_stat_loss(
                        gen_F_256_01,
                        source,
                        revealed_skin_mask_256,
                        visible_skin_ref,
                    )

            revealed_source_skin = revealed_skin_mask_256
            if source_skin_valid_mask is not None:
                revealed_source_skin = revealed_source_skin * resize_mask(source_skin_valid_mask, gen_F_256_01.shape[-2:])
            elif face_surface is not None:
                revealed_source_skin = revealed_source_skin * resize_mask(face_surface, gen_F_256_01.shape[-2:])
            source_hair_for_texture = source_hair_for_skin
            if source_hair_for_texture is not None:
                revealed_source_skin = revealed_source_skin * (
                    1.0 - resize_mask(source_hair_for_texture, gen_F_256_01.shape[-2:])
                ).clamp(0, 1)

            revealed_high_weight = self.losses_dict.get("revealed_skin_high", 0.0)
            if revealed_high_weight > 0:
                losses["revealed_skin_high"] = revealed_high_weight * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(source),
                    revealed_source_skin,
                )

            revealed_energy_weight = self.losses_dict.get("revealed_skin_energy", 0.0)
            if revealed_energy_weight > 0 and visible_skin_ref is not None:
                pred_energy = high_pass_filter(gen_F_256_01).abs().mean(dim=1, keepdim=True)
                ref_energy = high_pass_filter(source).abs().mean(dim=1, keepdim=True)
                pred_mean, _ = masked_scalar_stats(pred_energy, revealed_skin_mask_256)
                ref_mean, _ = masked_scalar_stats(ref_energy, visible_skin_ref)
                losses["revealed_skin_energy"] = revealed_energy_weight * F.relu(ref_mean.detach() - pred_mean)

            revealed_seam_weight = self.losses_dict.get("revealed_skin_seam", 0.0)
            if revealed_seam_weight > 0:
                seam_mask = (dilate_mask(revealed_skin_mask_256, 5) - erode_mask(revealed_skin_mask_256, 5)).clamp(0, 1)
                if source_visible_skin_ref is not None:
                    seam_mask = seam_mask * source_visible_skin_ref
                losses["revealed_skin_seam"] = revealed_seam_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(source),
                    seam_mask,
                )

        earring_confident_mask = resize_mask(earring_confident_mask, gen_F_256_01.shape[-2:])
        earring_reference = F.interpolate(earring_reference, size=gen_F_256_01.shape[-2:], mode="bilinear", align_corners=False)
        earring_supervision_hint = aux.get("earring_supervision_mask")
        if earring_supervision_hint is not None:
            earring_supervision_hint = resize_mask(earring_supervision_hint, gen_F_256_01.shape[-2:])
        else:
            earring_supervision_hint = earring_confident_mask
        earring_keep_support = torch.zeros_like(earring_confident_mask)
        for key, weight in (
            ("source_earring_mask", 1.0),
            ("source_earring_clean_mask", 1.0),
            ("earring_query_recall_mask", 0.35),
        ):
            value = aux.get(key)
            if value is not None:
                earring_keep_support = torch.clamp(
                    earring_keep_support + weight * resize_mask(value, gen_F_256_01.shape[-2:]),
                    0,
                    1,
                )
        earring_target_exclude_dilate = int(self.losses_dict.get("earring_target_exclude_dilate", 9))
        earring_object_protect_mask = torch.clamp(earring_supervision_hint + earring_keep_support, 0, 1)
        if earring_target_exclude_dilate > 0:
            earring_object_protect_mask = dilate_mask(earring_object_protect_mask, earring_target_exclude_dilate)

        local_edit_mask = earring_object_protect_mask
        if query_mask is not None:
            local_edit_mask = torch.clamp(local_edit_mask + resize_mask(query_mask, gen_F_256_01.shape[-2:]), 0, 1)
        if fine_mask is not None:
            local_edit_mask = torch.clamp(local_edit_mask + resize_mask(fine_mask, gen_F_256_01.shape[-2:]), 0, 1)
        if earring_highlight_mask is not None:
            local_edit_mask = torch.clamp(local_edit_mask + resize_mask(earring_highlight_mask, gen_F_256_01.shape[-2:]), 0, 1)
        preserve_mask = (1.0 - dilate_mask(local_edit_mask, int(self.losses_dict.get("target_preserve_exclude_dilate", 9)))).clamp(0, 1)
        target_face_for_preserve = parsing_label_mask(aux.get("target_parsing"), RAW_FACE_SURFACE_LABELS)
        if target_face_for_preserve is not None:
            preserve_mask = preserve_mask * (
                1.0 - resize_mask(target_face_for_preserve, gen_F_256_01.shape[-2:])
            ).clamp(0, 1)
        if revealed_skin_mask_256 is not None:
            preserve_mask = preserve_mask * (1.0 - revealed_skin_mask_256).clamp(0, 1)
        if cleanup_mask is not None:
            preserve_mask = preserve_mask * (1.0 - cleanup_mask).clamp(0, 1)

        target_preserve_weight = self.losses_dict.get("target_preserve", 0.0)
        if target_preserve_weight > 0:
            losses["target_preserve"] = target_preserve_weight * (
                masked_l1(gen_F_256_01, target, preserve_mask)
                + 0.75 * masked_l1(low_pass_filter(gen_F_256_01), low_pass_filter(target), preserve_mask)
            )

        face_dark_reject_weight = self.losses_dict.get("face_source_dark_reject", 0.0)
        target_face_for_dark_reject = parsing_label_mask(aux.get("target_parsing"), RAW_FACE_SURFACE_LABELS)
        if face_dark_reject_weight > 0 and target_face_for_dark_reject is not None:
            face_dark_reject_mask = resize_mask(target_face_for_dark_reject, gen_F_256_01.shape[-2:])
            target_hair_for_dark = aux.get("target_hair_mask")
            if target_hair_for_dark is not None:
                face_dark_reject_mask = face_dark_reject_mask * (
                    1.0 - resize_mask(target_hair_for_dark, gen_F_256_01.shape[-2:])
                ).clamp(0, 1)
            source_hair_for_dark = aux.get("source_hair_mask")
            source_hair_risk = None
            if source_hair_for_dark is not None:
                source_hair_risk = resize_mask(source_hair_for_dark, gen_F_256_01.shape[-2:])
                source_hair_risk = dilate_mask(
                    source_hair_risk,
                    int(self.losses_dict.get("face_source_dark_source_hair_dilate", 7)),
                )
                face_dark_reject_mask = face_dark_reject_mask * source_hair_risk
            target_detail_for_dark = parsing_label_mask(aux.get("target_parsing"), RAW_DETAIL_LABELS)
            if target_detail_for_dark is not None:
                detail_exclude = dilate_mask(
                    resize_mask(target_detail_for_dark, gen_F_256_01.shape[-2:]),
                    int(self.losses_dict.get("face_source_dark_detail_exclude_dilate", 5)),
                )
                face_dark_reject_mask = face_dark_reject_mask * (1.0 - detail_exclude).clamp(0, 1)

            source_copy_exclude = local_edit_mask
            for key in ("ear_roi", "visible_ear_roi", "target_hair_ear_block_mask", "target_covered_ear_block_mask"):
                value = aux.get(key)
                if value is not None:
                    source_copy_exclude = torch.clamp(
                        source_copy_exclude + resize_mask(value, gen_F_256_01.shape[-2:]),
                        0,
                        1,
                    )
            source_copy_exclude = dilate_mask(source_copy_exclude, int(self.losses_dict.get("face_source_dark_ear_exclude_dilate", 7)))
            face_dark_reject_mask = face_dark_reject_mask * (1.0 - source_copy_exclude).clamp(0, 1)

            losses["face_source_dark_reject"] = face_dark_reject_weight * source_dark_high_reject_loss(
                source,
                target,
                gen_F_256_01,
                face_dark_reject_mask,
                margin=float(self.losses_dict.get("face_source_dark_reject_margin", 0.006)),
                source_threshold=float(self.losses_dict.get("face_source_dark_reject_source_threshold", 0.018)),
                kernel_size=int(self.losses_dict.get("face_source_dark_reject_kernel", 11)),
            )

        target_hair_preserve_weight = self.losses_dict.get("target_hair_preserve", 0.0)
        target_hair_mask_for_preserve = aux.get("target_hair_mask")
        if target_hair_preserve_weight > 0 and target_hair_mask_for_preserve is not None:
            target_hair_preserve_mask = (
                resize_mask(target_hair_mask_for_preserve, gen_F_256_01.shape[-2:])
                * (1.0 - local_edit_mask).clamp(0, 1)
            )
            losses["target_hair_preserve"] = target_hair_preserve_weight * (
                masked_l1(gen_F_256_01, target, target_hair_preserve_mask)
                + masked_l1(low_pass_filter(gen_F_256_01), low_pass_filter(target), target_hair_preserve_mask)
            )

        if target_hair_ear_block_mask is not None:
            target_hair_ear_block_mask = resize_mask(target_hair_ear_block_mask, gen_F_256_01.shape[-2:])
            if target_hair_mask_for_preserve is not None:
                target_hair_ear_support = resize_mask(target_hair_mask_for_preserve, gen_F_256_01.shape[-2:])
                support_dilate = int(self.losses_dict.get("target_hair_ear_anchor_hair_dilate", 3))
                if support_dilate > 0:
                    target_hair_ear_support = dilate_mask(target_hair_ear_support, support_dilate)
                target_hair_ear_block_mask = target_hair_ear_block_mask * target_hair_ear_support.clamp(0, 1)
            else:
                target_hair_ear_block_mask = torch.zeros_like(target_hair_ear_block_mask)

            target_hair_anchor_weight = self.losses_dict.get("target_hair_ear_anchor", 0.0)
            if target_hair_anchor_weight > 0:
                losses["target_hair_ear_anchor"] = target_hair_anchor_weight * (
                    masked_l1(gen_F_256_01, target, target_hair_ear_block_mask)
                    + 0.5 * masked_l1(
                        low_pass_filter(gen_F_256_01),
                        low_pass_filter(target),
                        target_hair_ear_block_mask,
                    )
                )

            target_hair_reject_weight = self.losses_dict.get("target_hair_ear_source_reject", 0.0)
            if target_hair_reject_weight > 0:
                losses["target_hair_ear_source_reject"] = target_hair_reject_weight * source_direction_leak_loss(
                    source,
                    target,
                    gen_F_256_01,
                    target_hair_ear_block_mask,
                )

        if target_earring_suppress_mask is not None:
            target_earring_suppress_mask = resize_mask(target_earring_suppress_mask, gen_F_256_01.shape[-2:])
            if target_clean_01 is None:
                target_clean_01 = low_pass_filter(target)
            else:
                target_clean_01 = F.interpolate(
                    target_clean_01,
                    size=gen_F_256_01.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0, 1)

            suppress_low_weight = self.losses_dict.get("target_earring_suppress_low", 0.0)
            if suppress_low_weight > 0:
                losses["target_earring_suppress_low"] = suppress_low_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target_clean_01),
                    target_earring_suppress_mask,
                )

            suppress_high_weight = self.losses_dict.get("target_earring_suppress_high", 0.0)
            if suppress_high_weight > 0:
                losses["target_earring_suppress_high"] = suppress_high_weight * masked_mean(
                    high_pass_filter(gen_F_256_01).abs(),
                    target_earring_suppress_mask,
                )

        source_ear_for_detail = torch.clamp(earring_supervision_hint + earring_keep_support, 0, 1)
        detail_mask = parsing_label_mask(aux.get("source_parsing"), RAW_DETAIL_LABELS)
        if detail_mask is not None:
            detail_mask = resize_mask(detail_mask, gen_F_256_01.shape[-2:])
            target_hair_mask = aux.get("target_hair_mask")
            if target_hair_mask is not None:
                detail_mask = detail_mask * (1 - resize_mask(target_hair_mask, gen_F_256_01.shape[-2:])).clamp(0, 1)
            if source_hair_block_mask is not None:
                detail_mask = detail_mask * (1 - resize_mask(source_hair_block_mask, gen_F_256_01.shape[-2:])).clamp(0, 1)
            if cleanup_mask is not None:
                detail_mask = detail_mask * (1 - cleanup_mask).clamp(0, 1)
            detail_mask = torch.clamp(detail_mask + source_ear_for_detail, 0, 1)

            detail_high_weight = self.losses_dict.get("detail_high", 0.0)
            if detail_high_weight > 0:
                losses["detail_high"] = detail_high_weight * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(source),
                    detail_mask,
                )

            detail_low_anchor_weight = self.losses_dict.get("detail_low_anchor", 0.0)
            if detail_low_anchor_weight > 0:
                detail_low_mask = detail_mask * (1.0 - earring_object_protect_mask).clamp(0, 1)
                losses["detail_low_anchor"] = detail_low_anchor_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target),
                    detail_low_mask,
                )

        earring_supervision_dilate = int(self.losses_dict.get("earring_supervision_dilate", 5))
        earring_supervision_seed = torch.clamp(earring_supervision_hint + earring_keep_support, 0, 1)
        earring_supervision = dilate_mask(earring_supervision_seed, earring_supervision_dilate) * query_mask
        weak_pseudo_mask = torch.zeros_like(earring_confident_mask)
        if earring_supervision_seed.detach().sum().item() < 1:
            weak_pseudo_mask, _ = build_weak_ear_pseudo_mask(earring_reference, query_mask, earring_supervision_seed)
        mask_target = torch.clamp(weak_pseudo_mask + earring_supervision + earring_supervision_seed, 0, 1)
        query_expand = float(self.losses_dict.get("ear_query_expand", 0.02))
        supervision_mask = torch.clamp(mask_target + query_expand * query_mask, 0, 1)
        if fine_mask is not None:
            supervision_mask = torch.clamp(supervision_mask + 0.5 * fine_mask.detach(), 0, 1)
        earring_detail_supervision_mask = supervision_mask

        mask_weight = self.losses_dict.get("ear_mask", 0.0)
        if mask_weight > 0 and fine_mask_logits is not None:
            valid_mask = torch.clamp(query_mask + mask_target, 0, 1)
            bce_map = self.mask_bce(fine_mask_logits, mask_target.float())
            losses["ear_mask"] = mask_weight * masked_mean(bce_map, valid_mask)

        area_weight = self.losses_dict.get("ear_mask_area", 0.0)
        if area_weight > 0 and fine_mask is not None:
            overflow = F.relu(fine_mask - mask_target)
            losses["ear_mask_area"] = area_weight * masked_ratio(overflow, query_mask)

        high_weight = self.losses_dict.get("ear_high", 0.0)
        if high_weight > 0:
            source_high = high_pass_filter(earring_reference)
            gen_high = high_pass_filter(gen_F_256_01)
            losses["ear_high"] = high_weight * masked_l1(gen_high, source_high, earring_detail_supervision_mask)

        color_weight = self.losses_dict.get("ear_color", 0.0)
        if color_weight > 0:
            earring_color_mask = earring_supervision_seed
            losses["ear_color"] = color_weight * (
                masked_l1(gen_F_256_01, earring_reference, earring_color_mask)
                + 0.5 * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(earring_reference),
                    earring_color_mask,
                )
            )

        highlight_weight = self.losses_dict.get("ear_highlight", 0.0)
        if highlight_weight > 0 and earring_highlight_mask is not None:
            earring_highlight_mask = resize_mask(earring_highlight_mask, gen_F_256_01.shape[-2:])
            highlight_valid = torch.clamp(
                resize_mask(query_mask, gen_F_256_01.shape[-2:]) + earring_supervision_seed,
                0,
                1,
            )
            if target_hair_ear_block_mask is not None:
                highlight_valid = highlight_valid * (1.0 - target_hair_ear_block_mask).clamp(0, 1)
            earring_highlight_mask = torch.clamp(
                earring_highlight_mask + 0.35 * earring_supervision_seed,
                0,
                1,
            ) * highlight_valid
            losses["ear_highlight"] = highlight_weight * (
                masked_l1(gen_F_256_01, earring_reference, earring_highlight_mask)
                + masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(earring_reference),
                    earring_highlight_mask,
                )
                + masked_non_darker(
                    rgb_to_gray(gen_F_256_01),
                    rgb_to_gray(earring_reference),
                    earring_highlight_mask,
                )
            )

        lighting_weight = self.losses_dict.get("ear_lighting", 0.0)
        if lighting_weight > 0:
            target_low = low_pass_filter(target)
            gen_low = low_pass_filter(gen_F_256_01)
            lighting_mask = torch.clamp(query_mask + (fine_mask if fine_mask is not None else 0), 0, 1)
            lighting_mask = lighting_mask * (1.0 - earring_object_protect_mask).clamp(0, 1)
            losses["ear_lighting"] = lighting_weight * masked_l1(gen_low, target_low, lighting_mask)

        leak_weight = self.losses_dict.get("ear_hair_leak", 0.0)
        if leak_weight > 0 and source_hair_block_mask is not None:
            source_hair_block_mask = ensure_mask_4d(source_hair_block_mask).float()
            block_strength = max(0.0, min(1.0, float(self.losses_dict.get("ear_block_strength", 0.95))))
            leak_mask = source_hair_block_mask * block_strength * (1.0 - source_ear_for_detail).clamp(0, 1)
            losses["ear_hair_leak"] = leak_weight * source_direction_leak_loss(
                source,
                target,
                gen_F_256_01,
                leak_mask,
            )

            anchor_weight = self.losses_dict.get("ear_hair_anchor", 0.0)
            if anchor_weight > 0:
                losses["ear_hair_anchor"] = anchor_weight * (
                    masked_l1(gen_F_256_01, target, leak_mask)
                    + 0.5 * masked_l1(low_pass_filter(gen_F_256_01), low_pass_filter(target), leak_mask)
                )

        edge_weight = self.losses_dict.get("ear_edge", 0.0)
        if edge_weight > 0:
            source_edge = sobel_edges(earring_reference)
            gen_edge = sobel_edges(gen_F_256_01)
            losses["ear_edge"] = edge_weight * masked_l1(gen_edge, source_edge, earring_detail_supervision_mask)

        presence_weight = self.losses_dict.get("ear_presence", 0.0)
        if presence_weight > 0 and presence_logits is not None:
            weak_presence_target = build_weak_presence_target(mask_target, earring_supervision_seed * query_mask)
            if presence_target is not None:
                weak_presence_target = torch.maximum(weak_presence_target, presence_target.float())
            losses["ear_presence"] = presence_weight * self.presence_bce(presence_logits, weak_presence_target)

        brightness_weight = self.losses_dict.get("ear_brightness_reg", 0.0)
        if brightness_weight > 0 and gamma is not None and beta is not None:
            losses["ear_brightness_reg"] = brightness_weight * ((gamma - 1).abs().mean() + beta.abs().mean())

        return losses
