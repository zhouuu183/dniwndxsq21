from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.pp_losses import LossBuilderMulti
from models.ear_modules_v51 import (
    RAW_DETAIL_LABELS,
    dilate_mask,
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


def build_weak_ear_pseudo_mask(
    source: torch.Tensor,
    query_mask: torch.Tensor,
    parser_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_mask = ensure_mask_4d(query_mask).float()
    energy = high_pass_filter(source).abs().mean(dim=1, keepdim=True) * query_mask

    flat_energy = energy.flatten(2)
    flat_query = query_mask.flatten(2)
    energy_max = flat_energy.amax(dim=2, keepdim=True).view(-1, 1, 1, 1)
    energy_mean = (flat_energy.sum(dim=2, keepdim=True) / flat_query.sum(dim=2, keepdim=True).clamp_min(1.0))
    energy_mean = energy_mean.view(-1, 1, 1, 1)
    threshold = torch.maximum(0.6 * energy_max, 1.5 * energy_mean)

    pseudo_mask = (energy >= threshold).float() * query_mask
    pseudo_mask = dilate_mask(pseudo_mask, 3) * query_mask

    if parser_mask is not None:
        parser_mask = resize_mask(parser_mask, source.shape[-2:])
        pseudo_mask = torch.clamp(pseudo_mask + 0.2 * parser_mask * query_mask, 0, 1)

    pseudo_mask = pseudo_mask * query_mask

    return pseudo_mask, energy


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
        losses = super().__call__(source, target, target_mask, HT_E, gen_w, F_w, gen_F, F_gen, **kwargs)
        if aux is None:
            return losses

        gen_F_256 = self.downsample_256(gen_F)
        gen_F_256_01 = ((gen_F_256 + 1) / 2).clamp(0, 1)

        uses_safe_query = aux.get("hair_safe_query_mask") is not None
        query_mask = aux.get("ear_detail_query_mask", aux.get("hair_safe_query_mask", aux.get("query_mask")))
        source_ear_mask = aux.get("source_earring_mask")
        recall_mask = aux.get("source_earring_recall_mask")
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
            query_mask = ensure_mask_4d(query_mask).float() * visible_ear_roi
            source_ear_mask = (ensure_mask_4d(source_ear_mask).float() > 0.05).float() * visible_ear_roi
        else:
            source_ear_mask = (ensure_mask_4d(source_ear_mask).float() > 0.05).float()

        recall_mask_for_loss = None
        if recall_mask is not None:
            recall_mask = (ensure_mask_4d(recall_mask).float() > 0.05).float()
            recall_mask = resize_mask(recall_mask, query_mask.shape[-2:])
            if visible_ear_roi is not None:
                recall_mask = recall_mask * visible_ear_roi
            source_ear_mask = torch.clamp(source_ear_mask + recall_mask, 0, 1)
            recall_mask_for_loss = resize_mask(recall_mask, gen_F_256_01.shape[-2:])

        if source_hair_block_mask is not None:
            source_hair_block_mask = resize_mask(source_hair_block_mask, query_mask.shape[-2:])
            if recall_mask is not None:
                source_hair_block_mask = source_hair_block_mask * (1 - dilate_mask(recall_mask, 3)).clamp(0, 1)

        if source_hair_block_mask is not None and not uses_safe_query:
            block_strength = max(0.0, min(1.0, float(self.losses_dict.get("ear_block_strength", 0.95))))
            query_mask = query_mask * (1 - block_strength * source_hair_block_mask).clamp(0, 1)

        if recall_mask_for_loss is not None and recall_mask_for_loss.sum().item() > 0:
            recall_rgb_weight = self.losses_dict.get("earring_recall_rgb", 0.0)
            if recall_rgb_weight > 0:
                losses["earring_recall_rgb"] = recall_rgb_weight * masked_l1(
                    gen_F_256_01,
                    source,
                    recall_mask_for_loss,
                )

            recall_high_weight = self.losses_dict.get("earring_recall_high", 0.0)
            if recall_high_weight > 0:
                losses["earring_recall_high"] = recall_high_weight * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(source),
                    recall_mask_for_loss,
                )

            recall_edge_weight = self.losses_dict.get("earring_recall_edge", 0.0)
            if recall_edge_weight > 0:
                losses["earring_recall_edge"] = recall_edge_weight * masked_l1(
                    sobel_edges(gen_F_256_01),
                    sobel_edges(source),
                    recall_mask_for_loss,
                )

            no_halo_weight = self.losses_dict.get("earring_recall_no_halo", 0.0)
            if no_halo_weight > 0:
                no_halo_mask = (dilate_mask(recall_mask_for_loss, 11) - dilate_mask(recall_mask_for_loss, 3)).clamp(0, 1)
                if no_halo_mask.sum().item() > 0:
                    losses["earring_recall_no_halo"] = no_halo_weight * (
                        masked_l1(gen_F_256_01, target, no_halo_mask)
                        + 0.5 * masked_l1(low_pass_filter(gen_F_256_01), low_pass_filter(target), no_halo_mask)
                    )

        cleanup_mask = combine_aux_masks(aux, CLEANUP_MASK_KEYS)
        if cleanup_mask is not None:
            cleanup_mask = resize_mask(cleanup_mask, gen_F_256_01.shape[-2:])
            cleanup_anchor_weight = self.losses_dict.get("cleanup_anchor", 0.0)
            if cleanup_anchor_weight > 0:
                losses["cleanup_anchor"] = cleanup_anchor_weight * (
                    masked_l1(gen_F_256_01, target, cleanup_mask)
                    + 0.5 * masked_l1(low_pass_filter(gen_F_256_01), low_pass_filter(target), cleanup_mask)
                )

            cleanup_non_dark_weight = self.losses_dict.get("cleanup_non_dark", 0.0)
            if cleanup_non_dark_weight > 0:
                losses["cleanup_non_dark"] = cleanup_non_dark_weight * masked_non_darker(
                    rgb_to_gray(gen_F_256_01),
                    rgb_to_gray(target),
                    cleanup_mask,
                )

        detail_mask = parsing_label_mask(aux.get("source_parsing"), RAW_DETAIL_LABELS)
        if detail_mask is not None:
            detail_mask = resize_mask(detail_mask, gen_F_256_01.shape[-2:])
            detail_mask = torch.clamp(detail_mask + resize_mask(source_ear_mask, gen_F_256_01.shape[-2:]), 0, 1)
            target_hair_mask = aux.get("target_hair_mask")
            if target_hair_mask is not None:
                detail_mask = detail_mask * (1 - resize_mask(target_hair_mask, gen_F_256_01.shape[-2:])).clamp(0, 1)
            if source_hair_block_mask is not None:
                detail_mask = detail_mask * (1 - resize_mask(source_hair_block_mask, gen_F_256_01.shape[-2:])).clamp(0, 1)

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
                    detail_mask,
                )

        weak_pseudo_mask, _ = build_weak_ear_pseudo_mask(source, query_mask, source_ear_mask)
        query_expand = float(self.losses_dict.get("ear_query_expand", 0.05))
        supervision_mask = torch.clamp(weak_pseudo_mask + query_expand * query_mask, 0, 1)
        if fine_mask is not None:
            supervision_mask = torch.clamp(supervision_mask + 0.5 * fine_mask.detach(), 0, 1)

        mask_weight = self.losses_dict.get("ear_mask", 0.0)
        if mask_weight > 0 and fine_mask_logits is not None:
            valid_mask = torch.clamp(query_mask + weak_pseudo_mask, 0, 1)
            bce_map = self.mask_bce(fine_mask_logits, weak_pseudo_mask.float())
            losses["ear_mask"] = mask_weight * masked_mean(bce_map, valid_mask)

        area_weight = self.losses_dict.get("ear_mask_area", 0.0)
        if area_weight > 0 and fine_mask is not None:
            overflow = F.relu(fine_mask - weak_pseudo_mask)
            losses["ear_mask_area"] = area_weight * masked_ratio(overflow, query_mask)

        high_weight = self.losses_dict.get("ear_high", 0.0)
        if high_weight > 0:
            source_high = high_pass_filter(source)
            gen_high = high_pass_filter(gen_F_256_01)
            losses["ear_high"] = high_weight * masked_l1(gen_high, source_high, supervision_mask)

        lighting_weight = self.losses_dict.get("ear_lighting", 0.0)
        if lighting_weight > 0:
            target_low = low_pass_filter(target)
            gen_low = low_pass_filter(gen_F_256_01)
            lighting_mask = torch.clamp(query_mask + (fine_mask if fine_mask is not None else 0), 0, 1)
            losses["ear_lighting"] = lighting_weight * masked_l1(gen_low, target_low, lighting_mask)

        leak_weight = self.losses_dict.get("ear_hair_leak", 0.0)
        if leak_weight > 0 and source_hair_block_mask is not None:
            source_hair_block_mask = ensure_mask_4d(source_hair_block_mask).float()
            block_strength = max(0.0, min(1.0, float(self.losses_dict.get("ear_block_strength", 0.95))))
            leak_mask = source_hair_block_mask * block_strength
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
            source_edge = sobel_edges(source)
            gen_edge = sobel_edges(gen_F_256_01)
            losses["ear_edge"] = edge_weight * masked_l1(gen_edge, source_edge, supervision_mask)

        presence_weight = self.losses_dict.get("ear_presence", 0.0)
        if presence_weight > 0 and presence_logits is not None:
            weak_presence_target = build_weak_presence_target(weak_pseudo_mask, source_ear_mask * query_mask)
            if presence_target is not None:
                weak_presence_target = torch.maximum(weak_presence_target, presence_target.float())
            losses["ear_presence"] = presence_weight * self.presence_bce(presence_logits, weak_presence_target)

        brightness_weight = self.losses_dict.get("ear_brightness_reg", 0.0)
        if brightness_weight > 0 and gamma is not None and beta is not None:
            losses["ear_brightness_reg"] = brightness_weight * ((gamma - 1).abs().mean() + beta.abs().mean())

        return losses

