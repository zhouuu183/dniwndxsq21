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


def masked_non_brighter(
    pred_gray: torch.Tensor,
    anchor_gray: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.02,
) -> torch.Tensor:
    mask = resize_mask(mask, pred_gray.shape[-2:])
    return masked_mean(F.relu(pred_gray - anchor_gray - margin), mask)


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
        if cleanup_mask is not None:
            cleanup_for_base = resize_mask(cleanup_mask, target_mask.shape[-2:])
            exclude_strength = max(0.0, min(1.0, float(self.losses_dict.get("base_cleanup_exclude", 1.0))))
            base_target_mask = ensure_mask_4d(target_mask).float() * (1 - exclude_strength * cleanup_for_base).clamp(0, 1)

        losses = super().__call__(source, target, base_target_mask, HT_E, gen_w, F_w, gen_F, F_gen, **kwargs)
        if aux is None:
            return losses

        gen_F_256 = self.downsample_256(gen_F)
        gen_F_256_01 = ((gen_F_256 + 1) / 2).clamp(0, 1)

        uses_safe_query = aux.get("ear_detail_query_mask") is not None or aux.get("hair_safe_query_mask") is not None
        query_mask = aux.get("ear_detail_query_mask", aux.get("hair_safe_query_mask", aux.get("query_mask")))
        source_ear_mask = aux.get("source_earring_mask")
        earring_confident_mask = aux.get("earring_confident_mask", source_ear_mask)
        earring_detail_mask = aux.get("earring_detail_mask", earring_confident_mask)
        earring_highlight_mask = aux.get("earring_highlight_mask")
        earring_reference = aux.get("earring_reference", source)
        revealed_skin_mask = aux.get("revealed_skin_mask")
        source_skin_valid_mask = aux.get("source_skin_valid_mask")
        fine_mask = aux.get("fine_mask")
        fine_mask_logits = aux.get("fine_mask_logits")
        presence_logits = aux.get("presence_logits")
        presence_target = aux.get("presence_target")
        visible_ear_roi = aux.get("visible_ear_roi")
        source_hair_block_mask = aux.get("source_hair_block_mask")
        gamma = aux.get("gamma")
        beta = aux.get("beta")

        if query_mask is None or source_ear_mask is None:
            return losses

        if visible_ear_roi is not None:
            visible_ear_roi = ensure_mask_4d(visible_ear_roi).float()
            query_mask = ensure_mask_4d(query_mask).float() * (0.25 + 0.75 * visible_ear_roi).clamp(0, 1)
            source_ear_mask = ensure_mask_4d(source_ear_mask).float() * (0.35 + 0.65 * visible_ear_roi).clamp(0, 1)
            earring_confident_mask = ensure_mask_4d(earring_confident_mask).float() * (0.25 + 0.75 * visible_ear_roi).clamp(0, 1)
            earring_detail_mask = ensure_mask_4d(earring_detail_mask).float() * (0.35 + 0.65 * visible_ear_roi).clamp(0, 1)

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

        if cleanup_mask is not None:
            cleanup_mask = resize_mask(cleanup_mask, gen_F_256_01.shape[-2:])
            if cleanup_regular_mask is None:
                cleanup_regular_mask = cleanup_mask
            cleanup_low_anchor_weight = self.losses_dict.get(
                "cleanup_low_anchor",
                self.losses_dict.get("cleanup_anchor", 0.0),
            )
            if cleanup_low_anchor_weight > 0:
                losses["cleanup_low_anchor"] = cleanup_low_anchor_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target),
                    cleanup_regular_mask,
                )

            cleanup_source_reject_weight = self.losses_dict.get("cleanup_source_reject", 0.0)
            if cleanup_source_reject_weight > 0:
                losses["cleanup_source_reject"] = cleanup_source_reject_weight * source_direction_leak_loss(
                    source,
                    target,
                    gen_F_256_01,
                    cleanup_regular_mask,
                )

            cleanup_non_dark_weight = self.losses_dict.get("cleanup_non_dark", 0.0)
            if cleanup_non_dark_weight > 0:
                losses["cleanup_non_dark"] = cleanup_non_dark_weight * masked_non_darker(
                    rgb_to_gray(gen_F_256_01),
                    rgb_to_gray(target),
                    cleanup_regular_mask,
                )

            face_surface = parsing_label_mask(aux.get("source_parsing"), RAW_FACE_SURFACE_LABELS)
            source_hair_for_texture = aux.get("source_hair_mask")
            target_hair_for_texture = aux.get("target_hair_mask")
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
                cleanup_source_skin = cleanup_regular_mask * resize_mask(face_surface, gen_F_256_01.shape[-2:])
                if source_hair_for_texture is not None:
                    cleanup_source_skin = cleanup_source_skin * (
                        1.0 - resize_mask(source_hair_for_texture, gen_F_256_01.shape[-2:])
                    ).clamp(0, 1)
                if cleanup_high_weight > 0:
                    losses["cleanup_high"] = cleanup_high_weight * masked_l1(
                        high_pass_filter(gen_F_256_01),
                        high_pass_filter(source),
                        cleanup_source_skin,
                    )

        if revealed_skin_mask_256 is not None:
            revealed_low_weight = self.losses_dict.get("revealed_skin_low_anchor", 0.0)
            if revealed_low_weight > 0:
                losses["revealed_skin_low_anchor"] = revealed_low_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target),
                    revealed_skin_mask_256,
                )

            face_surface = parsing_label_mask(aux.get("source_parsing"), RAW_FACE_SURFACE_LABELS)
            visible_skin_ref = parsing_label_mask(aux.get("target_parsing"), RAW_FACE_SURFACE_LABELS)
            if visible_skin_ref is not None:
                visible_skin_ref = resize_mask(visible_skin_ref, gen_F_256_01.shape[-2:])
                target_hair_for_texture = aux.get("target_hair_mask")
                if target_hair_for_texture is not None:
                    visible_skin_ref = visible_skin_ref * (
                        1.0 - resize_mask(target_hair_for_texture, gen_F_256_01.shape[-2:])
                    ).clamp(0, 1)
                visible_skin_ref = visible_skin_ref * (1.0 - revealed_skin_mask_256).clamp(0, 1)

                revealed_texture_weight = self.losses_dict.get("revealed_skin_texture", 0.0)
                if revealed_texture_weight > 0:
                    losses["revealed_skin_texture"] = revealed_texture_weight * texture_stat_loss(
                        gen_F_256_01,
                        target,
                        revealed_skin_mask_256,
                        visible_skin_ref,
                    )

            revealed_source_skin = revealed_skin_mask_256
            if source_skin_valid_mask is not None:
                revealed_source_skin = revealed_source_skin * resize_mask(source_skin_valid_mask, gen_F_256_01.shape[-2:])
            elif face_surface is not None:
                revealed_source_skin = revealed_source_skin * resize_mask(face_surface, gen_F_256_01.shape[-2:])
            source_hair_for_texture = aux.get("source_hair_mask")
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
                ref_energy = high_pass_filter(target).abs().mean(dim=1, keepdim=True)
                pred_mean, _ = masked_scalar_stats(pred_energy, revealed_skin_mask_256)
                ref_mean, _ = masked_scalar_stats(ref_energy, visible_skin_ref)
                losses["revealed_skin_energy"] = revealed_energy_weight * F.relu(ref_mean.detach() - pred_mean)

            revealed_seam_weight = self.losses_dict.get("revealed_skin_seam", 0.0)
            if revealed_seam_weight > 0:
                seam_mask = (dilate_mask(revealed_skin_mask_256, 5) - erode_mask(revealed_skin_mask_256, 5)).clamp(0, 1)
                losses["revealed_skin_seam"] = revealed_seam_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target),
                    seam_mask,
                )

        non_edit_low_weight = self.losses_dict.get("non_edit_low_anchor", 0.0)
        non_edit_high_weight = self.losses_dict.get("non_edit_high_anchor", 0.0)
        non_edit_non_dark_weight = self.losses_dict.get("non_edit_non_dark", 0.0)
        if non_edit_low_weight > 0 or non_edit_high_weight > 0 or non_edit_non_dark_weight > 0:
            query_mask_256 = resize_mask(query_mask, gen_F_256_01.shape[-2:])
            edit_mask = query_mask_256
            if cleanup_mask is not None:
                edit_mask = torch.clamp(edit_mask + resize_mask(cleanup_mask, gen_F_256_01.shape[-2:]), 0, 1)
            if revealed_skin_mask_256 is not None:
                edit_mask = torch.clamp(edit_mask + revealed_skin_mask_256, 0, 1)
            edit_dilate = int(self.losses_dict.get("non_edit_edit_dilate", 17))
            non_edit_mask = (1.0 - dilate_mask(edit_mask, edit_dilate)).clamp(0, 1)
            non_edit_erode = int(self.losses_dict.get("non_edit_erode", 9))
            if non_edit_erode > 1:
                non_edit_mask = erode_mask(non_edit_mask, non_edit_erode)

            if non_edit_low_weight > 0:
                losses["non_edit_low_anchor"] = non_edit_low_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target),
                    non_edit_mask,
                )
            if non_edit_high_weight > 0:
                losses["non_edit_high_anchor"] = non_edit_high_weight * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(target),
                    non_edit_mask,
                )
            if non_edit_non_dark_weight > 0:
                losses["non_edit_non_dark"] = non_edit_non_dark_weight * masked_non_darker(
                    rgb_to_gray(gen_F_256_01),
                    rgb_to_gray(target),
                    non_edit_mask,
                )

        earring_confident_mask = resize_mask(earring_confident_mask, gen_F_256_01.shape[-2:])
        earring_detail_mask = resize_mask(earring_detail_mask, gen_F_256_01.shape[-2:])
        earring_reference = F.interpolate(earring_reference, size=gen_F_256_01.shape[-2:], mode="bilinear", align_corners=False)
        earring_detail_support = torch.clamp(dilate_mask(earring_detail_mask, 5) + 0.18 * earring_confident_mask, 0, 1)
        non_earring_ear_mask = torch.clamp(query_mask * (1.0 - dilate_mask(earring_detail_mask, 7)), 0, 1)
        source_ear_for_detail = earring_detail_support
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
            detail_low_mask = detail_mask * (1.0 - dilate_mask(earring_detail_support, 5)).clamp(0, 1)

            detail_high_weight = self.losses_dict.get("detail_high", 0.0)
            if detail_high_weight > 0:
                losses["detail_high"] = detail_high_weight * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(source),
                    detail_mask,
                )

            detail_low_anchor_weight = self.losses_dict.get("detail_low_anchor", 0.0)
            if detail_low_anchor_weight > 0:
                losses["detail_low_anchor"] = detail_low_anchor_weight * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(target),
                    detail_low_mask,
                )

        earring_supervision_dilate = int(self.losses_dict.get("earring_supervision_dilate", 5))
        earring_supervision = torch.clamp(
            dilate_mask(earring_detail_mask, earring_supervision_dilate)
            + 0.18 * earring_confident_mask,
            0,
            1,
        ) * query_mask
        weak_pseudo_mask = torch.zeros_like(earring_confident_mask)
        if earring_detail_mask.detach().sum().item() < 1:
            weak_pseudo_mask, _ = build_weak_ear_pseudo_mask(earring_reference, query_mask, earring_detail_mask)
        mask_target = torch.clamp(weak_pseudo_mask + earring_supervision + earring_detail_mask, 0, 1)
        query_expand = float(self.losses_dict.get("ear_query_expand", 0.05))
        supervision_mask = torch.clamp(mask_target + query_expand * query_mask, 0, 1)
        if fine_mask is not None:
            supervision_mask = torch.clamp(supervision_mask + 0.5 * fine_mask.detach(), 0, 1)

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
            losses["ear_high"] = high_weight * masked_l1(gen_high, source_high, supervision_mask)

        color_weight = self.losses_dict.get("ear_color", 0.0)
        if color_weight > 0:
            color_mask = torch.clamp(earring_detail_mask + 0.35 * earring_highlight_mask, 0, 1) if earring_highlight_mask is not None else earring_detail_mask
            losses["ear_color"] = color_weight * (
                masked_l1(gen_F_256_01, earring_reference, color_mask)
                + 0.5 * masked_l1(
                    low_pass_filter(gen_F_256_01),
                    low_pass_filter(earring_reference),
                    color_mask,
                )
            )

        highlight_weight = self.losses_dict.get("ear_highlight", 0.0)
        if highlight_weight > 0 and earring_highlight_mask is not None:
            earring_highlight_mask = resize_mask(earring_highlight_mask, gen_F_256_01.shape[-2:])
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
            lighting_mask = lighting_mask * (1.0 - earring_detail_support).clamp(0, 1)
            losses["ear_lighting"] = lighting_weight * masked_l1(gen_low, target_low, lighting_mask)

        non_ear_brightness_weight = self.losses_dict.get("ear_non_earring_brightness", 0.0)
        if non_ear_brightness_weight > 0:
            losses["ear_non_earring_brightness"] = non_ear_brightness_weight * masked_non_brighter(
                rgb_to_gray(gen_F_256_01),
                rgb_to_gray(target),
                non_earring_ear_mask,
                margin=float(self.losses_dict.get("ear_non_earring_brightness_margin", 0.025)),
            )

        leak_weight = self.losses_dict.get("ear_hair_leak", 0.0)
        if leak_weight > 0 and source_hair_block_mask is not None:
            source_hair_block_mask = ensure_mask_4d(source_hair_block_mask).float()
            block_strength = max(0.0, min(1.0, float(self.losses_dict.get("ear_block_strength", 0.95))))
            leak_mask = source_hair_block_mask * block_strength * (1.0 - earring_detail_support).clamp(0, 1)
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
            losses["ear_edge"] = edge_weight * masked_l1(gen_edge, source_edge, supervision_mask)

        presence_weight = self.losses_dict.get("ear_presence", 0.0)
        if presence_weight > 0 and presence_logits is not None:
            weak_presence_target = build_weak_presence_target(mask_target, earring_confident_mask * query_mask)
            if presence_target is not None:
                weak_presence_target = torch.maximum(weak_presence_target, presence_target.float())
            losses["ear_presence"] = presence_weight * self.presence_bce(presence_logits, weak_presence_target)

        brightness_weight = self.losses_dict.get("ear_brightness_reg", 0.0)
        if brightness_weight > 0 and gamma is not None and beta is not None:
            losses["ear_brightness_reg"] = brightness_weight * ((gamma - 1).abs().mean() + beta.abs().mean())

        return losses
