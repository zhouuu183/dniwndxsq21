from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.pp_losses import LossBuilderMulti
from models.ear_modules_v54 import (
    RAW_DETAIL_LABELS,
    build_target_earring_safe_zone,
    dilate_mask,
    erode_mask,
    ensure_mask_4d,
    high_pass_filter,
    low_pass_filter,
    resize_mask,
)

CLEANUP_MASK_KEYS = ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")
FACE_SURFACE_LABELS = (1, 2)


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


def masked_channel_stats(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = resize_mask(mask, value.shape[-2:])
    if mask.size(1) == 1 and value.size(1) != 1:
        mask = mask.expand(-1, value.size(1), -1, -1)
    denom = mask.sum(dim=(0, 2, 3)).clamp_min(1.0)
    mean = (value * mask).sum(dim=(0, 2, 3)) / denom
    var = ((value - mean.view(1, -1, 1, 1)).pow(2) * mask).sum(dim=(0, 2, 3)) / denom
    return mean, torch.sqrt(var + 1e-6)


def masked_stats_l1(
    pred: torch.Tensor,
    pred_mask: torch.Tensor,
    ref: torch.Tensor,
    ref_mask: torch.Tensor,
) -> torch.Tensor:
    pred_mask = resize_mask(pred_mask, pred.shape[-2:])
    ref_mask = resize_mask(ref_mask, ref.shape[-2:])
    if pred_mask.sum().item() <= 0 or ref_mask.sum().item() <= 0:
        return pred.sum() * 0
    pred_mean, pred_std = masked_channel_stats(pred, pred_mask)
    ref_mean, ref_std = masked_channel_stats(ref.detach(), ref_mask)
    return (pred_mean - ref_mean).abs().mean() + (pred_std - ref_std).abs().mean()


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


def build_boundary_ring(mask: torch.Tensor, outer: int = 13, inner: int = 5) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    return (dilate_mask(mask, outer) - erode_mask(mask, inner)).clamp(0, 1)


def build_face_structure_mask(
    source_parsing: torch.Tensor | None,
    target_hair_mask: torch.Tensor | None = None,
    source_hair_block_mask: torch.Tensor | None = None,
    cleanup_mask: torch.Tensor | None = None,
    earring_mask: torch.Tensor | None = None,
    out_size: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    detail_mask = parsing_label_mask(source_parsing, RAW_DETAIL_LABELS)
    if detail_mask is None:
        return None

    if out_size is not None:
        detail_mask = resize_mask(detail_mask, out_size)
    if target_hair_mask is not None:
        detail_mask = detail_mask * (1 - resize_mask(target_hair_mask, detail_mask.shape[-2:])).clamp(0, 1)
    if source_hair_block_mask is not None:
        detail_mask = detail_mask * (1 - resize_mask(source_hair_block_mask, detail_mask.shape[-2:])).clamp(0, 1)
    if cleanup_mask is not None:
        detail_mask = detail_mask * (1 - resize_mask(cleanup_mask, detail_mask.shape[-2:])).clamp(0, 1)
    if earring_mask is not None:
        detail_mask = detail_mask * (1 - resize_mask(earring_mask, detail_mask.shape[-2:])).clamp(0, 1)
    return detail_mask.clamp(0, 1)


def build_face_surface_texture_mask(
    source_parsing: torch.Tensor | None,
    target_parsing: torch.Tensor | None,
    source_hair_mask: torch.Tensor | None = None,
    target_hair_mask: torch.Tensor | None = None,
    source_hair_block_mask: torch.Tensor | None = None,
    ear_exclude_mask: torch.Tensor | None = None,
    out_size: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    source_face = parsing_label_mask(source_parsing, FACE_SURFACE_LABELS)
    target_face = parsing_label_mask(target_parsing, FACE_SURFACE_LABELS)
    if source_face is None or target_face is None:
        return None

    face_mask = source_face * resize_mask(target_face, source_face.shape[-2:])
    if out_size is not None:
        face_mask = resize_mask(face_mask, out_size)
    if source_hair_mask is not None:
        face_mask = face_mask * (1 - resize_mask(source_hair_mask, face_mask.shape[-2:])).clamp(0, 1)
    if target_hair_mask is not None:
        face_mask = face_mask * (1 - resize_mask(target_hair_mask, face_mask.shape[-2:])).clamp(0, 1)
    if source_hair_block_mask is not None:
        face_mask = face_mask * (1 - resize_mask(source_hair_block_mask, face_mask.shape[-2:])).clamp(0, 1)
    if ear_exclude_mask is not None:
        face_mask = face_mask * (1 - dilate_mask(resize_mask(ear_exclude_mask, face_mask.shape[-2:]), 7)).clamp(0, 1)
    return face_mask.clamp(0, 1)


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

        uses_safe_query = aux.get("hair_safe_query_mask") is not None
        query_mask = aux.get("ear_detail_query_mask", aux.get("hair_safe_query_mask", aux.get("query_mask")))
        source_ear_mask = aux.get("source_earring_mask")
        fine_mask = aux.get("fine_mask")
        fine_mask_logits = aux.get("fine_mask_logits")
        presence_logits = aux.get("presence_logits")
        presence_target = aux.get("presence_target")
        visible_ear_roi = aux.get("visible_ear_roi")
        source_hair_block_mask = aux.get("source_hair_block_mask")
        gamma = aux.get("gamma")
        beta = aux.get("beta")
        cleanup_face_stage = self.losses_dict.get("training_stage") == "cleanup_face"

        if query_mask is None or source_ear_mask is None:
            return losses

        query_mask = ensure_mask_4d(query_mask).float()
        source_ear_mask = ensure_mask_4d(source_ear_mask).float()
        reference_mask = aux.get("earring_reference_mask")
        if reference_mask is not None:
            reference_mask = resize_mask(reference_mask, query_mask.shape[-2:])
        else:
            reference_mask = torch.zeros_like(source_ear_mask)
        earring_confident_mask = aux.get("earring_confident_mask")
        if earring_confident_mask is not None:
            earring_confident_mask = resize_mask(earring_confident_mask, query_mask.shape[-2:])
        else:
            earring_confident_mask = torch.zeros_like(source_ear_mask)
        earring_object_mask = aux.get("earring_object_mask")
        if earring_object_mask is not None:
            earring_object_mask = resize_mask(earring_object_mask, query_mask.shape[-2:])
        else:
            earring_object_mask = torch.zeros_like(source_ear_mask)
        earring_locator_mask = aux.get("earring_locator_mask")
        if earring_locator_mask is not None:
            earring_locator_mask = resize_mask(earring_locator_mask, query_mask.shape[-2:])
        else:
            earring_locator_mask = torch.zeros_like(source_ear_mask)
        earring_highlight_mask = aux.get("earring_highlight_mask")
        if earring_highlight_mask is not None:
            earring_highlight_mask = resize_mask(earring_highlight_mask, query_mask.shape[-2:])
        else:
            earring_highlight_mask = torch.zeros_like(source_ear_mask)

        target_safe_zone = aux.get("target_earring_safe_zone")
        if target_safe_zone is None:
            target_safe_zone = build_target_earring_safe_zone(
                aux.get("target_parsing"),
                query_mask.shape[-2:],
                placement_roi=aux.get("earring_search_roi", visible_ear_roi),
            )
        if target_safe_zone is not None:
            target_safe_zone = resize_mask(target_safe_zone, query_mask.shape[-2:])
            target_safe_query = dilate_mask(
                target_safe_zone,
                int(self.losses_dict.get("target_safe_query_dilate", 3)),
            )
            reference_mask = (reference_mask * target_safe_zone).clamp(0, 1)
            earring_confident_mask = (earring_confident_mask * target_safe_zone).clamp(0, 1)
            earring_object_mask = (earring_object_mask * target_safe_zone).clamp(0, 1)
            earring_locator_mask = (earring_locator_mask * target_safe_zone).clamp(0, 1)
            earring_highlight_mask = (earring_highlight_mask * target_safe_zone).clamp(0, 1)
            query_mask = (query_mask * target_safe_query).clamp(0, 1)

        earring_confident_mask = torch.clamp(
            earring_confident_mask + reference_mask + earring_object_mask + earring_locator_mask,
            0,
            1,
        )
        earring_highlight_mask = torch.clamp(
            earring_highlight_mask + 0.35 * earring_confident_mask + 0.35 * earring_locator_mask,
            0,
            1,
        )
        aligned_positive_mask = aux.get("aligned_earring_positive_mask")
        if aligned_positive_mask is not None:
            aligned_positive_mask = resize_mask(aligned_positive_mask, query_mask.shape[-2:])
        else:
            aligned_positive_mask = torch.clamp(reference_mask + earring_locator_mask + earring_object_mask, 0, 1)
        earring_write_gate = aux.get("earring_write_gate")
        if earring_write_gate is not None:
            earring_write_gate = resize_mask(earring_write_gate, query_mask.shape[-2:])
        else:
            earring_write_gate = torch.clamp(query_mask + earring_confident_mask, 0, 1)

        if visible_ear_roi is not None:
            visible_ear_roi = ensure_mask_4d(visible_ear_roi).float()
            transfer_roi = torch.clamp(
                visible_ear_roi + reference_mask + earring_confident_mask + earring_object_mask + earring_locator_mask,
                0,
                1,
            )
            query_mask = query_mask * transfer_roi

        if source_hair_block_mask is not None and not uses_safe_query:
            source_hair_block_mask = resize_mask(source_hair_block_mask, query_mask.shape[-2:])
            block_strength = max(0.0, min(1.0, float(self.losses_dict.get("ear_block_strength", 0.95))))
            query_mask = query_mask * (1 - block_strength * source_hair_block_mask).clamp(0, 1)

        source_for_earring = aux.get("earring_reference_image", source)
        if source_for_earring.shape[-2:] != source.shape[-2:]:
            source_for_earring = F.interpolate(
                source_for_earring,
                size=source.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        earring_confident_for_detail = resize_mask(earring_confident_mask, gen_F_256_01.shape[-2:])
        ear_exclude_for_face = torch.clamp(
            resize_mask(query_mask, gen_F_256_01.shape[-2:])
            + earring_confident_for_detail,
            0,
            1,
        )
        face_texture_mask = build_face_surface_texture_mask(
            aux.get("source_parsing"),
            aux.get("target_parsing"),
            source_hair_mask=aux.get("source_hair_mask"),
            target_hair_mask=aux.get("target_hair_mask"),
            source_hair_block_mask=source_hair_block_mask,
            ear_exclude_mask=ear_exclude_for_face,
            out_size=gen_F_256_01.shape[-2:],
        )
        pollution_anchor_weight = self.losses_dict.get("earring_pollution_anchor", 0.0)
        pollution_dark_weight = self.losses_dict.get("earring_pollution_dark", 0.0)
        if pollution_anchor_weight > 0 or pollution_dark_weight > 0:
            pollution_gate = resize_mask(earring_write_gate, gen_F_256_01.shape[-2:])
            aligned_positive_for_pollution = resize_mask(aligned_positive_mask, gen_F_256_01.shape[-2:])
            exclude_dilate = int(self.losses_dict.get("earring_pollution_guard_dilate", 13))
            pollution_guard = pollution_gate * (1 - dilate_mask(aligned_positive_for_pollution, exclude_dilate)).clamp(0, 1)
            target_face_surface = parsing_label_mask(aux.get("target_parsing"), FACE_SURFACE_LABELS)
            if target_face_surface is not None:
                face_near = dilate_mask(resize_mask(target_face_surface, gen_F_256_01.shape[-2:]), 9)
                pollution_guard = pollution_guard * face_near
            target_hair_mask = aux.get("target_hair_mask")
            if target_hair_mask is not None:
                pollution_guard = pollution_guard * (
                    1 - resize_mask(target_hair_mask, gen_F_256_01.shape[-2:])
                ).clamp(0, 1)
            if cleanup_mask is not None:
                pollution_guard = pollution_guard * (
                    1 - resize_mask(cleanup_mask, gen_F_256_01.shape[-2:])
                ).clamp(0, 1)
            if pollution_guard.sum().item() > 0:
                if pollution_anchor_weight > 0:
                    losses["earring_pollution_anchor"] = pollution_anchor_weight * masked_l1(
                        gen_F_256_01,
                        target,
                        pollution_guard,
                    )
                if pollution_dark_weight > 0:
                    losses["earring_pollution_dark"] = pollution_dark_weight * masked_non_darker(
                        rgb_to_gray(gen_F_256_01),
                        rgb_to_gray(target),
                        pollution_guard,
                    )
        cheek_dark_weight = self.losses_dict.get("cheek_dark_reject", 0.0)
        cheek_low_weight = self.losses_dict.get("cheek_low_anchor", 0.0)
        if cheek_dark_weight > 0 or cheek_low_weight > 0:
            target_face_guard = parsing_label_mask(aux.get("target_parsing"), FACE_SURFACE_LABELS)
            if target_face_guard is not None:
                cheek_guard = resize_mask(target_face_guard, gen_F_256_01.shape[-2:])
                target_hair_mask = aux.get("target_hair_mask")
                if target_hair_mask is not None:
                    cheek_guard = cheek_guard * (
                        1 - resize_mask(target_hair_mask, gen_F_256_01.shape[-2:])
                    ).clamp(0, 1)
                if target_safe_zone is not None:
                    cheek_guard = cheek_guard * (
                        1 - dilate_mask(resize_mask(target_safe_zone, gen_F_256_01.shape[-2:]), 15)
                    ).clamp(0, 1)
                cheek_guard = cheek_guard * (
                    1 - dilate_mask(ear_exclude_for_face, 9)
                ).clamp(0, 1)
                if cleanup_mask is not None:
                    cheek_guard = cheek_guard * (
                        1 - resize_mask(cleanup_mask, gen_F_256_01.shape[-2:])
                    ).clamp(0, 1)

                if cheek_guard.sum().item() > 0:
                    if cheek_dark_weight > 0:
                        losses["cheek_dark_reject"] = cheek_dark_weight * masked_non_darker(
                            rgb_to_gray(gen_F_256_01),
                            rgb_to_gray(target),
                            cheek_guard,
                        )
                    if cheek_low_weight > 0:
                        losses["cheek_low_anchor"] = cheek_low_weight * masked_l1(
                            low_pass_filter(gen_F_256_01),
                            low_pass_filter(target),
                            cheek_guard,
                        )
        cleanup_face_mask = aux.get("cleanup_face_mask")
        if cleanup_face_stage and cleanup_face_mask is not None:
            cleanup_face_mask = resize_mask(cleanup_face_mask, gen_F_256_01.shape[-2:])
            target_face_surface = parsing_label_mask(aux.get("target_parsing"), FACE_SURFACE_LABELS)
            if target_face_surface is not None:
                cleanup_face_mask = cleanup_face_mask * resize_mask(target_face_surface, gen_F_256_01.shape[-2:])

            target_hair_mask = aux.get("target_hair_mask")
            if target_hair_mask is not None:
                cleanup_face_mask = cleanup_face_mask * (
                    1 - resize_mask(target_hair_mask, gen_F_256_01.shape[-2:])
                ).clamp(0, 1)

            ear_exclude = torch.clamp(
                resize_mask(query_mask, gen_F_256_01.shape[-2:])
                + earring_confident_for_detail,
                0,
                1,
            )
            target_earring_mask = aux.get("target_earring_mask")
            if target_earring_mask is not None:
                ear_exclude = torch.clamp(ear_exclude + resize_mask(target_earring_mask, gen_F_256_01.shape[-2:]), 0, 1)
            cleanup_face_mask = cleanup_face_mask * (1 - dilate_mask(ear_exclude, 9)).clamp(0, 1)

            if cleanup_face_mask.sum().item() > 0:
                gen_low = low_pass_filter(gen_F_256_01)
                target_low = low_pass_filter(target)
                gen_high = high_pass_filter(gen_F_256_01)
                target_high = high_pass_filter(target)

                cleanup_face_low_weight = self.losses_dict.get("cleanup_face_low", 0.0)
                if cleanup_face_low_weight > 0:
                    losses["cleanup_face_low"] = cleanup_face_low_weight * masked_l1(
                        gen_low,
                        target_low,
                        cleanup_face_mask,
                    )

                cleanup_face_texture_weight = self.losses_dict.get("cleanup_face_texture", 0.0)
                if cleanup_face_texture_weight > 0:
                    texture_ref_mask = face_texture_mask
                    if texture_ref_mask is None:
                        texture_ref_mask = cleanup_face_mask
                    losses["cleanup_face_texture"] = cleanup_face_texture_weight * masked_stats_l1(
                        gen_high,
                        cleanup_face_mask,
                        high_pass_filter(source),
                        texture_ref_mask,
                    )

                cleanup_face_seam_weight = self.losses_dict.get("cleanup_face_seam", 0.0)
                if cleanup_face_seam_weight > 0:
                    seam_mask = build_boundary_ring(cleanup_face_mask, outer=13, inner=5)
                    if target_face_surface is not None:
                        seam_mask = seam_mask * resize_mask(target_face_surface, gen_F_256_01.shape[-2:])
                    if target_hair_mask is not None:
                        seam_mask = seam_mask * (
                            1 - resize_mask(target_hair_mask, gen_F_256_01.shape[-2:])
                        ).clamp(0, 1)
                    losses["cleanup_face_seam"] = cleanup_face_seam_weight * (
                        masked_l1(gen_low, target_low, seam_mask)
                        + 0.5 * masked_l1(gen_high, target_high, seam_mask)
                    )

                cleanup_face_shadow_reject_weight = self.losses_dict.get("cleanup_face_shadow_reject", 0.0)
                if cleanup_face_shadow_reject_weight > 0:
                    losses["cleanup_face_shadow_reject"] = cleanup_face_shadow_reject_weight * source_direction_leak_loss(
                        source,
                        target,
                        gen_F_256_01,
                        cleanup_face_mask,
                    )

            return losses

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

        revealed_skin_mask = aux.get("revealed_skin_mask")
        if revealed_skin_mask is not None:
            revealed_skin_mask = resize_mask(revealed_skin_mask, gen_F_256_01.shape[-2:])
            target_face_surface = parsing_label_mask(aux.get("target_parsing"), FACE_SURFACE_LABELS)
            if target_face_surface is not None:
                revealed_skin_mask = revealed_skin_mask * resize_mask(target_face_surface, gen_F_256_01.shape[-2:])
            target_hair_mask = aux.get("target_hair_mask")
            if target_hair_mask is not None:
                revealed_skin_mask = revealed_skin_mask * (
                    1 - resize_mask(target_hair_mask, gen_F_256_01.shape[-2:])
                ).clamp(0, 1)
            revealed_skin_mask = revealed_skin_mask * (1 - dilate_mask(ear_exclude_for_face, 9)).clamp(0, 1)

            if revealed_skin_mask.sum().item() > 0 and face_texture_mask is not None:
                texture_context_mask = face_texture_mask * (1 - dilate_mask(revealed_skin_mask, 17)).clamp(0, 1)
                source_high = high_pass_filter(source)
                gen_high = high_pass_filter(gen_F_256_01)

                texture_weight = self.losses_dict.get("revealed_skin_texture", 0.0)
                if texture_weight > 0:
                    losses["revealed_skin_texture"] = texture_weight * masked_stats_l1(
                        gen_high,
                        revealed_skin_mask,
                        source_high,
                        texture_context_mask,
                    )

                energy_weight = self.losses_dict.get("revealed_skin_energy", 0.0)
                if energy_weight > 0:
                    losses["revealed_skin_energy"] = energy_weight * masked_stats_l1(
                        gen_high.abs().mean(dim=1, keepdim=True),
                        revealed_skin_mask,
                        source_high.abs().mean(dim=1, keepdim=True),
                        texture_context_mask,
                    )

                high_weight = self.losses_dict.get("revealed_skin_high", 0.0)
                if high_weight > 0:
                    losses["revealed_skin_high"] = high_weight * masked_stats_l1(
                        sobel_edges(gen_F_256_01),
                        revealed_skin_mask,
                        sobel_edges(source),
                        texture_context_mask,
                    )

                seam_weight = self.losses_dict.get("revealed_skin_seam", 0.0)
                if seam_weight > 0:
                    seam_mask = build_boundary_ring(revealed_skin_mask, outer=17, inner=5)
                    if target_face_surface is not None:
                        seam_mask = seam_mask * resize_mask(target_face_surface, gen_F_256_01.shape[-2:])
                    seam_mask = seam_mask * (1 - dilate_mask(ear_exclude_for_face, 9)).clamp(0, 1)
                    losses["revealed_skin_seam"] = seam_weight * (
                        masked_l1(low_pass_filter(gen_F_256_01), low_pass_filter(target), seam_mask)
                        + 0.5 * masked_stats_l1(gen_high, seam_mask, source_high, texture_context_mask)
                    )

        if face_texture_mask is not None:
            face_texture_weight = self.losses_dict.get("face_texture_consistency", 0.0)
            if face_texture_weight > 0:
                losses["face_source_texture"] = face_texture_weight * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(source),
                    face_texture_mask,
                )

        detail_mask = build_face_structure_mask(
            aux.get("source_parsing"),
            target_hair_mask=aux.get("target_hair_mask"),
            source_hair_block_mask=source_hair_block_mask,
            cleanup_mask=cleanup_mask,
            earring_mask=earring_confident_mask,
            out_size=gen_F_256_01.shape[-2:],
        )
        if detail_mask is not None:
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

            detail_structure_weight = self.losses_dict.get("detail_structure", 0.0)
            if detail_structure_weight > 0:
                losses["detail_structure"] = detail_structure_weight * masked_l1(
                    sobel_edges(gen_F_256_01),
                    sobel_edges(source),
                    detail_mask,
                )

        earring_supervision_dilate = int(self.losses_dict.get("earring_supervision_dilate", 5))
        earring_supervision = torch.clamp(
            dilate_mask(earring_confident_mask, earring_supervision_dilate) + earring_object_mask + earring_locator_mask,
            0,
            1,
        )
        weak_pseudo_mask, _ = build_weak_ear_pseudo_mask(source_for_earring, query_mask, earring_confident_mask)
        mask_target = torch.clamp(weak_pseudo_mask + earring_supervision + earring_object_mask + earring_locator_mask, 0, 1)
        query_expand = float(self.losses_dict.get("ear_query_expand", 0.05))
        supervision_mask = torch.clamp(
            mask_target
            + query_expand * query_mask
            + 0.5 * earring_highlight_mask
            + 0.35 * earring_object_mask
            + earring_locator_mask,
            0,
            1,
        )
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
            source_high = high_pass_filter(source_for_earring)
            gen_high = high_pass_filter(gen_F_256_01)
            losses["ear_high"] = high_weight * masked_l1(gen_high, source_high, supervision_mask)

        earring_confident_high_weight = self.losses_dict.get("earring_confident_high", 0.0)
        if earring_confident_high_weight > 0:
            losses["earring_confident_high"] = earring_confident_high_weight * masked_l1(
                high_pass_filter(gen_F_256_01),
                high_pass_filter(source_for_earring),
                earring_confident_mask,
            )

        earring_color_weight = self.losses_dict.get("earring_color", 0.0)
        if earring_color_weight > 0:
            losses["earring_color"] = earring_color_weight * masked_l1(
                gen_F_256_01,
                source_for_earring,
                earring_confident_mask,
            )

        locator_color_weight = self.losses_dict.get("earring_locator_color", 0.0)
        if locator_color_weight > 0 and earring_locator_mask.flatten(1).sum().item() > 0:
            losses["earring_locator_color"] = locator_color_weight * masked_l1(
                gen_F_256_01,
                source_for_earring,
                earring_locator_mask,
            )

        locator_high_weight = self.losses_dict.get("earring_locator_high", 0.0)
        if locator_high_weight > 0 and earring_locator_mask.flatten(1).sum().item() > 0:
            losses["earring_locator_high"] = locator_high_weight * masked_l1(
                high_pass_filter(gen_F_256_01),
                high_pass_filter(source_for_earring),
                earring_locator_mask,
            )

        earring_highlight_weight = self.losses_dict.get("earring_highlight", 0.0)
        if earring_highlight_weight > 0:
            highlight_mask = torch.clamp(
                earring_highlight_mask + 0.35 * dilate_mask(earring_highlight_mask, 3) * earring_confident_mask,
                0,
                1,
            )
            losses["earring_highlight"] = earring_highlight_weight * (
                masked_l1(gen_F_256_01, source_for_earring, highlight_mask)
                + 0.65 * masked_l1(
                    high_pass_filter(gen_F_256_01),
                    high_pass_filter(source_for_earring),
                    highlight_mask,
                )
                + 0.35 * masked_non_darker(
                    rgb_to_gray(gen_F_256_01),
                    rgb_to_gray(source_for_earring),
                    highlight_mask,
                )
            )

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
            source_edge = sobel_edges(source_for_earring)
            gen_edge = sobel_edges(gen_F_256_01)
            losses["ear_edge"] = edge_weight * masked_l1(gen_edge, source_edge, supervision_mask)

        locator_edge_weight = self.losses_dict.get("earring_locator_edge", 0.0)
        if locator_edge_weight > 0 and earring_locator_mask.flatten(1).sum().item() > 0:
            losses["earring_locator_edge"] = locator_edge_weight * masked_l1(
                sobel_edges(gen_F_256_01),
                sobel_edges(source_for_earring),
                earring_locator_mask,
            )

        presence_weight = self.losses_dict.get("ear_presence", 0.0)
        if presence_weight > 0 and presence_logits is not None:
            weak_presence_target = build_weak_presence_target(mask_target, earring_confident_mask)
            if presence_target is not None:
                weak_presence_target = torch.maximum(weak_presence_target, presence_target.float())
            losses["ear_presence"] = presence_weight * self.presence_bce(presence_logits, weak_presence_target)

        brightness_weight = self.losses_dict.get("ear_brightness_reg", 0.0)
        if brightness_weight > 0 and gamma is not None and beta is not None:
            losses["ear_brightness_reg"] = brightness_weight * ((gamma - 1).abs().mean() + beta.abs().mean())

        return losses
