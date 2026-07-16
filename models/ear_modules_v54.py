from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CtrlHair.external_code.face_parsing.my_parsing_util import FaceParsing_tensor
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion

RAW_LEFT_EAR = 7
RAW_RIGHT_EAR = 8
RAW_EARRING = 9
RAW_HAIR = 17
RAW_HAT = 18
RAW_FACE_SURFACE_LABELS = (1, 2)
RAW_DETAIL_LABELS = (2, 3, 4, 5, 6, 11, 12, 13)


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def ensure_mask_4d(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    return mask


def normalized_to_01(images: torch.Tensor) -> torch.Tensor:
    if images.dtype == torch.uint8:
        return images.float().div(255)
    if images.min().item() < -0.05:
        return ((images + 1) / 2).clamp(0, 1)
    return images.clamp(0, 1)


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    return F.interpolate(mask, size=size, mode="bilinear", align_corners=False).clamp(0, 1)


def dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return (mask > 0).float()
    return F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size // 2)


def erode_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    return 1 - dilate_mask(1 - mask, kernel_size)


def shift_mask(mask: torch.Tensor, down: int = 0, right: int = 0) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    return shift_tensor(mask, down=down, right=right)


def shift_tensor(value: torch.Tensor, down: int = 0, right: int = 0) -> torch.Tensor:
    shifted = torch.zeros_like(value)

    h_slice_src = slice(0, value.size(-2) - max(0, down))
    h_slice_dst = slice(max(0, down), value.size(-2))
    if down < 0:
        h_slice_src = slice(-down, value.size(-2))
        h_slice_dst = slice(0, value.size(-2) + down)

    w_slice_src = slice(0, value.size(-1) - max(0, right))
    w_slice_dst = slice(max(0, right), value.size(-1))
    if right < 0:
        w_slice_src = slice(-right, value.size(-1))
        w_slice_dst = slice(0, value.size(-1) + right)

    shifted[..., h_slice_dst, w_slice_dst] = value[..., h_slice_src, w_slice_src]
    return shifted


def gaussian_blur(image: torch.Tensor, kernel_size: int = 11, sigma: float = 3.0) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return image

    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / max(2 * sigma ** 2, 1e-6))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d[None, None, ...].expand(image.size(1), 1, -1, -1)
    return F.conv2d(image, kernel_2d, padding=kernel_size // 2, groups=image.size(1))


def low_pass_filter(image: torch.Tensor, kernel_size: int = 11, sigma: float = 3.0) -> torch.Tensor:
    return gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)


def high_pass_filter(image: torch.Tensor, kernel_size: int = 11, sigma: float = 3.0) -> torch.Tensor:
    return image - low_pass_filter(image, kernel_size=kernel_size, sigma=sigma)


def masked_adaptive_threshold(
    value: torch.Tensor,
    mask: torch.Tensor,
    std_scale: float = 0.75,
    min_delta: float = 0.02,
    min_area: float = 8.0,
) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    if value.size(1) != 1:
        value = value.mean(dim=1, keepdim=True)

    flat_mask = mask.flatten(2)
    denom = flat_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    mean = (value * mask).flatten(2).sum(dim=2, keepdim=True) / denom
    mean = mean.view(value.size(0), 1, 1, 1)
    centered = value - mean
    var = (centered.pow(2) * mask).flatten(2).sum(dim=2, keepdim=True) / denom
    std = var.sqrt().view(value.size(0), 1, 1, 1)
    enough_area = (flat_mask.sum(dim=2, keepdim=True) >= min_area).view(value.size(0), 1, 1, 1).float()
    threshold = mean + std_scale * std + min_delta
    return (value > threshold).float() * mask * enough_area


def parsing_label_mask(parsing: torch.Tensor | None, labels: tuple[int, ...], size: tuple[int, int]) -> torch.Tensor | None:
    if parsing is None:
        return None
    parsing = ensure_mask_4d(parsing).long()
    if parsing.shape[-2:] != size:
        parsing = F.interpolate(parsing.float(), size=size, mode="nearest").long()
    mask = torch.zeros_like(parsing, dtype=torch.bool)
    for label in labels:
        mask |= parsing == label
    return mask.float()


def build_source_earring_spatial_gate(
    source_parsing: torch.Tensor | None,
    source_earring_mask: torch.Tensor | None,
    size: tuple[int, int],
    outside_margin: int = 46,
    inside_margin: int = 18,
    top_ratio: float = 0.46,
    bottom_ratio: float = 0.94,
) -> torch.Tensor | None:
    if source_parsing is None:
        return None

    parsing_4d = ensure_mask_4d(source_parsing).long()
    if parsing_4d.shape[-2:] != size:
        parsing_4d = F.interpolate(parsing_4d.float(), size=size, mode="nearest").long()

    if source_earring_mask is None:
        source_earring_mask = torch.zeros(
            parsing_4d.size(0),
            1,
            size[0],
            size[1],
            device=parsing_4d.device,
            dtype=torch.float32,
        )
    else:
        source_earring_mask = resize_mask(source_earring_mask, size)

    face_surface = parsing_label_mask(parsing_4d, RAW_FACE_SURFACE_LABELS, size)
    left_ear = parsing_label_mask(parsing_4d, (RAW_LEFT_EAR,), size)
    right_ear = parsing_label_mask(parsing_4d, (RAW_RIGHT_EAR,), size)
    if face_surface is None:
        return None
    if left_ear is None:
        left_ear = torch.zeros_like(face_surface)
    if right_ear is None:
        right_ear = torch.zeros_like(face_surface)

    batch, _, height, width = face_surface.shape
    gate = torch.zeros_like(face_surface)
    outside_margin = max(4, int(outside_margin))
    inside_margin = max(2, int(inside_margin))
    top_ratio = max(0.20, min(0.70, float(top_ratio)))
    bottom_ratio = max(top_ratio + 0.05, min(0.98, float(bottom_ratio)))

    for batch_idx in range(batch):
        for left_side in (True, False):
            side = _half_plane(face_surface[batch_idx:batch_idx + 1], left_side)
            face_side = face_surface[batch_idx:batch_idx + 1] * side
            points = torch.nonzero(face_side[0, 0] > 0.05, as_tuple=False)
            if points.size(0) < 16:
                continue

            y_min = int(points[:, 0].min().item())
            y_max = int(points[:, 0].max().item())
            y0 = max(0, int(round(y_min + top_ratio * max(1, y_max - y_min))))
            y1 = min(height - 1, int(round(y_min + bottom_ratio * max(1, y_max - y_min))))
            for y in range(y0, y1 + 1):
                row = torch.nonzero(face_side[0, 0, y] > 0.05, as_tuple=False).flatten()
                if row.numel() == 0:
                    band = points[(points[:, 0].float() - float(y)).abs() <= 4]
                    if band.size(0) == 0:
                        continue
                    row = band[:, 1]
                if left_side:
                    edge_x = int(row.min().item())
                    x0 = max(0, edge_x - outside_margin)
                    x1 = min(width - 1, edge_x + inside_margin)
                else:
                    edge_x = int(row.max().item())
                    x0 = max(0, edge_x - inside_margin)
                    x1 = min(width - 1, edge_x + outside_margin)
                gate[batch_idx, 0, y, x0:x1 + 1] = 1.0

    ear_context = torch.clamp(left_ear + right_ear + dilate_mask(source_earring_mask, 7), 0, 1)
    lower_context = torch.clamp(
        dilate_mask(ear_context, 27)
        + shift_mask(dilate_mask(ear_context, 23), down=16)
        + 0.75 * shift_mask(dilate_mask(ear_context, 21), down=32)
        + 0.45 * shift_mask(dilate_mask(ear_context, 19), down=48),
        0,
        1,
    )
    fallback_gate = gate * lower_context
    source_parser_keep = dilate_mask(source_earring_mask, 13)
    gated = torch.clamp(fallback_gate + source_parser_keep, 0, 1)
    if face_surface is not None:
        face_inner = erode_mask(face_surface, 9)
        gated = gated * (1 - 0.85 * face_inner).clamp(0, 1)
    empty = gated.flatten(1).amax(dim=1).view(-1, 1, 1, 1) <= 0
    gated = torch.where(empty, gate, gated)
    return gated.clamp(0, 1)


def build_earring_evidence_score(source_01: torch.Tensor) -> dict[str, torch.Tensor]:
    source_01 = normalized_to_01(source_01)
    high_energy = high_pass_filter(source_01).abs().mean(dim=1, keepdim=True)
    chroma = source_01.amax(dim=1, keepdim=True) - source_01.amin(dim=1, keepdim=True)
    gray = rgb_to_gray(source_01)
    grad_x = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    edge_energy = torch.maximum(grad_x, grad_y)
    local_gray = low_pass_filter(gray, kernel_size=15, sigma=4.0)
    bright_spike = F.relu(gray - local_gray)
    dark_spike = F.relu(local_gray - gray)
    contrast_spike = torch.maximum(bright_spike, dark_spike)
    score = (
        1.40 * high_energy
        + 0.90 * chroma
        + 0.85 * bright_spike
        + 0.85 * dark_spike
        + 0.95 * edge_energy
        + 0.65 * contrast_spike
    )
    return {
        "score": score,
        "high_energy": high_energy,
        "chroma": chroma,
        "gray": gray,
        "edge_energy": edge_energy,
        "bright_spike": bright_spike,
        "dark_spike": dark_spike,
        "contrast_spike": contrast_spike,
    }


def clean_earring_candidate_mask(
    candidate_mask: torch.Tensor,
    source_01: torch.Tensor,
    support_mask: torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
    source_parsing: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    max_area_frac: float = 0.018,
    min_area: int = 4,
    dilate: int = 3,
) -> torch.Tensor:
    source_01 = normalized_to_01(source_01)
    size = source_01.shape[-2:]
    candidate_mask = resize_mask(candidate_mask, size)
    if support_mask is None:
        support = torch.ones_like(candidate_mask)
    else:
        support = resize_mask(support_mask, size)
    if query_mask is not None:
        support = torch.clamp(support + dilate_mask(resize_mask(query_mask, size), 9), 0, 1)
    support = torch.clamp(support + candidate_mask, 0, 1)

    evidence = build_earring_evidence_score(source_01)
    score = evidence["score"]

    face_surface = parsing_label_mask(source_parsing, RAW_FACE_SURFACE_LABELS, size)
    if face_surface is not None:
        score = score * (1 - 0.70 * face_surface).clamp(0, 1)
        support = support * (1 - 0.35 * face_surface).clamp(0, 1)
    ear_region = None
    if source_parsing is not None:
        left_ear = parsing_label_mask(source_parsing, (RAW_LEFT_EAR,), size)
        right_ear = parsing_label_mask(source_parsing, (RAW_RIGHT_EAR,), size)
        if left_ear is None:
            left_ear = torch.zeros_like(candidate_mask)
        if right_ear is None:
            right_ear = torch.zeros_like(candidate_mask)
        ear_region = torch.clamp(left_ear + right_ear, 0, 1)
    if source_hair_mask is not None:
        hair = resize_mask(source_hair_mask, size)
        score = score * (1 - 0.18 * hair).clamp(0, 1)
    else:
        hair = None
    parser_earring_mask = parsing_label_mask(source_parsing, (RAW_EARRING,), size)
    spatial_gate = build_source_earring_spatial_gate(source_parsing, parser_earring_mask, size)
    if spatial_gate is not None:
        parser_keep = dilate_mask(parser_earring_mask, 9) if parser_earring_mask is not None else torch.zeros_like(candidate_mask)
        support = support * torch.clamp(spatial_gate + parser_keep, 0, 1)

    candidate = (candidate_mask * support > 0.05).float()
    if candidate.flatten(1).sum().item() <= 0:
        return torch.zeros_like(candidate_mask)

    height, width = size
    max_total = max(int(min_area), int(round(height * width * max_area_frac)))
    max_side = max(int(min_area), max_total // 2)
    max_component = max(max_side, int(round(height * width * max_area_frac * 0.75)))
    cleaned = torch.zeros_like(candidate)

    for batch_idx in range(candidate.size(0)):
        for left_side in (True, False):
            side = _half_plane(candidate[batch_idx:batch_idx + 1], left_side)
            side_candidate = candidate[batch_idx:batch_idx + 1] * side
            count = int((side_candidate > 0.05).sum().item())
            if count < min_area:
                continue
            remaining = (side_candidate > 0.05).float()
            side_keep = torch.zeros_like(side_candidate)
            components: list[tuple[float, torch.Tensor]] = []
            max_components = 32
            side_score_map = score[batch_idx:batch_idx + 1] * side
            for _ in range(max_components):
                if remaining.sum().item() < min_area:
                    break
                seed_scores = side_score_map * remaining
                seed_index = int(seed_scores.flatten().argmax().item())
                if float(seed_scores.flatten()[seed_index].item()) <= 0:
                    seed_index = int(remaining.flatten().argmax().item())
                component = torch.zeros_like(remaining).flatten()
                component[seed_index] = 1.0
                component = component.view_as(remaining) * remaining
                previous_area = -1.0
                for _grow in range(max(height, width)):
                    component = dilate_mask(component, 3) * remaining
                    area_now = float(component.sum().item())
                    if abs(area_now - previous_area) < 0.5:
                        break
                    previous_area = area_now
                remaining = (remaining * (1 - component)).clamp(0, 1)

                area = float(component.sum().item())
                if area < min_area:
                    continue
                points = torch.nonzero(component[0, 0] > 0.05, as_tuple=False)
                if points.numel() == 0:
                    continue
                y0 = int(points[:, 0].min().item())
                y1 = int(points[:, 0].max().item())
                x0 = int(points[:, 1].min().item())
                x1 = int(points[:, 1].max().item())
                box_h = max(1, y1 - y0 + 1)
                box_w = max(1, x1 - x0 + 1)
                aspect = max(box_h / box_w, box_w / box_h)
                fill = area / max(float(box_h * box_w), 1.0)
                component_score = float((side_score_map * component).sum().item() / max(area, 1.0))
                face_overlap = 0.0
                if face_surface is not None:
                    face_overlap = float((face_surface[batch_idx:batch_idx + 1] * component).sum().item() / max(area, 1.0))
                hair_overlap = 0.0
                if hair is not None:
                    hair_overlap = float((hair[batch_idx:batch_idx + 1] * component).sum().item() / max(area, 1.0))
                ear_overlap = 0.0
                ear_near_overlap = 0.0
                if ear_region is not None:
                    ear_side = ear_region[batch_idx:batch_idx + 1] * side
                    ear_overlap = float((ear_side * component).sum().item() / max(area, 1.0))
                    ear_near = torch.clamp(
                        dilate_mask(ear_side, 25)
                        + shift_mask(dilate_mask(ear_side, 21), down=16)
                        + 0.70 * shift_mask(dilate_mask(ear_side, 19), down=32),
                        0,
                        1,
                    )
                    ear_near_overlap = float((ear_near * component).sum().item() / max(area, 1.0))
                slender_ok = aspect <= 12.0 and fill >= 0.12 and component_score >= 0.040 and face_overlap <= 0.62
                enough_object_evidence = (
                    component_score >= 0.055
                    or (component_score >= 0.040 and ear_near_overlap >= 0.12)
                    or ear_overlap >= 0.08
                )
                if face_overlap > 0.78 and ear_overlap < 0.04 and component_score < 0.090:
                    continue
                if hair_overlap > 0.70 and ear_overlap < 0.04 and component_score < 0.100:
                    continue
                if ear_region is not None and ear_near_overlap < 0.04 and ear_overlap < 0.03 and component_score < 0.100:
                    continue
                if not enough_object_evidence:
                    continue
                if aspect > 7.5 and fill < 0.28 and not slender_ok:
                    continue
                needs_trim = area > max_component or (
                    aspect > 10.5 and not slender_ok
                ) or (face_overlap > 0.55 and component_score < 0.055)
                if needs_trim:
                    valid = component.flatten() > 0.05
                    component_scores = (side_score_map * component).flatten()
                    component_scores = torch.where(valid, component_scores, torch.full_like(component_scores, -1.0))
                    keep_count = min(int(area), max_side)
                    top_indices = torch.topk(component_scores, keep_count).indices
                    trimmed = torch.zeros_like(component).flatten()
                    trimmed[top_indices] = 1.0
                    component = trimmed.view_as(component) * component
                    area = float(component.sum().item())
                    if area < min_area:
                        continue
                    points = torch.nonzero(component[0, 0] > 0.05, as_tuple=False)
                    y0 = int(points[:, 0].min().item())
                    y1 = int(points[:, 0].max().item())
                    x0 = int(points[:, 1].min().item())
                    x1 = int(points[:, 1].max().item())
                    box_h = max(1, y1 - y0 + 1)
                    box_w = max(1, x1 - x0 + 1)
                    aspect = max(box_h / box_w, box_w / box_h)
                    fill = area / max(float(box_h * box_w), 1.0)
                    component_score = float((side_score_map * component).sum().item() / max(area, 1.0))
                    slender_ok = aspect <= 12.0 and fill >= 0.10 and component_score >= 0.040 and face_overlap <= 0.62
                    if aspect > 10.5 and fill < 0.25 and not slender_ok:
                        continue
                rank = (
                    component_score
                    + 0.030 * min(area, max_side) / max_side
                    + 0.070 * ear_overlap
                    + 0.040 * ear_near_overlap
                    - 0.080 * face_overlap
                    - 0.070 * hair_overlap
                )
                components.append((rank, component))

            if components:
                components.sort(key=lambda item: item[0], reverse=True)
                kept_area = 0.0
                for _, component in components[:2]:
                    area = float(component.sum().item())
                    if kept_area + area > max_side * 0.85 and kept_area > 0:
                        continue
                    side_keep = torch.clamp(side_keep + component, 0, 1)
                    kept_area += area
            cleaned[batch_idx:batch_idx + 1] = torch.clamp(cleaned[batch_idx:batch_idx + 1] + side_keep, 0, 1)

    if dilate > 1:
        cleaned = dilate_mask(cleaned, dilate)
    return (cleaned * torch.clamp(dilate_mask(candidate_mask, 3) + candidate_mask, 0, 1) * support).clamp(0, 1)


def build_source_earring_object_mask(
    source_01: torch.Tensor,
    source_parsing: torch.Tensor | None,
    source_earring_mask: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    max_area_frac: float = 0.018,
    min_area: int = 4,
) -> torch.Tensor:
    source_01 = normalized_to_01(source_01)
    size = source_01.shape[-2:]
    if source_earring_mask is None:
        source_earring_mask = torch.zeros(source_01.size(0), 1, size[0], size[1], device=source_01.device, dtype=source_01.dtype)
    else:
        source_earring_mask = resize_mask(source_earring_mask, size)

    spatial_gate = build_source_earring_spatial_gate(source_parsing, source_earring_mask, size)
    if spatial_gate is None:
        spatial_gate = torch.ones_like(source_earring_mask)

    evidence = build_earring_evidence_score(source_01)
    high_energy = evidence["high_energy"]
    edge_energy = evidence["edge_energy"]
    chroma = evidence["chroma"]
    gray = evidence["gray"]
    bright_spike = evidence["bright_spike"]
    dark_spike = evidence["dark_spike"]
    contrast_spike = evidence["contrast_spike"]

    face_surface = parsing_label_mask(source_parsing, RAW_FACE_SURFACE_LABELS, size)
    left_ear = parsing_label_mask(source_parsing, (RAW_LEFT_EAR,), size)
    right_ear = parsing_label_mask(source_parsing, (RAW_RIGHT_EAR,), size)
    if left_ear is None:
        left_ear = torch.zeros_like(source_earring_mask)
    if right_ear is None:
        right_ear = torch.zeros_like(source_earring_mask)
    ear_region = torch.clamp(left_ear + right_ear, 0, 1)
    ear_near = torch.clamp(
        dilate_mask(ear_region, 27)
        + shift_mask(dilate_mask(ear_region, 23), down=14)
        + 0.75 * shift_mask(dilate_mask(ear_region, 19), down=28),
        0,
        1,
    )

    if source_hair_mask is not None:
        hair = resize_mask(source_hair_mask, size)
    else:
        hair = torch.zeros_like(source_earring_mask)

    local_bg = low_pass_filter(source_01, kernel_size=21, sigma=6.0)
    local_gray_bg = rgb_to_gray(local_bg)
    color_delta = (source_01 - local_bg).abs().mean(dim=1, keepdim=True)
    gray_delta = (gray - local_gray_bg).abs()
    local_objectness = torch.clamp(color_delta + gray_delta + 0.5 * chroma, 0, 1)

    object_score = (
        1.25 * edge_energy
        + 1.05 * high_energy
        + 1.05 * local_objectness
        + 0.90 * contrast_spike
        + 0.85 * chroma
        + 0.75 * bright_spike
        + 0.75 * dark_spike
    )
    if face_surface is not None:
        object_score = object_score * (1 - 0.55 * erode_mask(face_surface, 7)).clamp(0, 1)
    object_score = object_score * (1 - 0.45 * hair).clamp(0, 1)

    score_gate = masked_adaptive_threshold(object_score, spatial_gate, std_scale=0.14, min_delta=0.004, min_area=12)
    edge_gate = masked_adaptive_threshold(edge_energy + high_energy, spatial_gate, std_scale=0.20, min_delta=0.006, min_area=12)
    contrast_gate = masked_adaptive_threshold(
        contrast_spike + chroma + local_objectness,
        spatial_gate,
        std_scale=0.14,
        min_delta=0.006,
        min_area=12,
    )
    parser_keep = dilate_mask(source_earring_mask, 5)
    candidate = torch.clamp(
        (score_gate * torch.clamp(edge_gate + contrast_gate, 0, 1))
        + (edge_gate * contrast_gate)
        + (score_gate * masked_adaptive_threshold(local_objectness, spatial_gate, std_scale=0.12, min_delta=0.004, min_area=12))
        + parser_keep,
        0,
        1,
    ) * spatial_gate
    object_hull = erode_mask(dilate_mask(candidate, 9), 5) * spatial_gate
    fill_seed = masked_adaptive_threshold(local_objectness + contrast_spike, object_hull, std_scale=0.10, min_delta=0.004, min_area=6)
    candidate = torch.clamp(candidate + object_hull * torch.clamp(fill_seed + dilate_mask(candidate, 5), 0, 1), 0, 1) * spatial_gate

    support = torch.clamp(spatial_gate * torch.clamp(ear_near + parser_keep, 0, 1), 0, 1)
    object_mask = clean_earring_candidate_mask(
        candidate,
        source_01,
        support_mask=support,
        query_mask=support,
        source_parsing=source_parsing,
        source_hair_mask=source_hair_mask,
        max_area_frac=max_area_frac,
        min_area=min_area,
        dilate=1,
    )
    object_mask = torch.clamp(erode_mask(dilate_mask(object_mask, 7), 3) * spatial_gate, 0, 1)
    if parser_keep.flatten(1).sum().item() > 0:
        parser_object = clean_earring_candidate_mask(
            parser_keep * spatial_gate,
            source_01,
            support_mask=support,
            query_mask=support,
            source_parsing=source_parsing,
            source_hair_mask=source_hair_mask,
            max_area_frac=max_area_frac,
            min_area=min_area,
            dilate=1,
        )
        object_mask = torch.clamp(object_mask + parser_object, 0, 1)
    return object_mask.clamp(0, 1)


def _scaled_value(value: int | float, size: tuple[int, int], base: int = 256, minimum: int = 0) -> int:
    scale = max(size) / float(base)
    return max(minimum, int(round(float(value) * scale)))


def _scaled_area(value: int | float, size: tuple[int, int], base: int = 256) -> int:
    scale = max(size) / float(base)
    return max(1, int(round(float(value) * scale * scale)))


def _filter_locator_components(
    mask: torch.Tensor,
    min_area: int,
    max_area_ratio: float,
    keep_top: int,
) -> torch.Tensor:
    mask = (ensure_mask_4d(mask).float() > 0.5).float()
    kept = torch.zeros_like(mask)
    if mask.flatten(1).amax(dim=1).sum().item() <= 0:
        return kept

    batch, _, height, width = mask.shape
    max_area = int(float(max_area_ratio) * height * width) if max_area_ratio > 0 else height * width
    max_area = max(max_area, int(min_area))
    keep_top = max(1, int(keep_top))

    for batch_idx in range(batch):
        arr = mask[batch_idx, 0].detach().cpu().numpy().astype(np.bool_)
        visited = np.zeros_like(arr, dtype=np.bool_)
        components: list[list[tuple[int, int]]] = []
        ys, xs = np.nonzero(arr)

        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        if not visited[ny, nx] and arr[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            area = len(comp)
            if int(min_area) <= area <= max_area:
                components.append(comp)

        components.sort(key=len, reverse=True)
        for comp in components[:keep_top]:
            comp_y, comp_x = zip(*comp)
            kept[batch_idx, 0, list(comp_y), list(comp_x)] = 1.0

    return kept


def _masked_rgb_mean_and_valid(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    min_area: float = 32.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = normalized_to_01(image)
    if mask is None:
        mask = torch.zeros(image.size(0), 1, image.size(2), image.size(3), device=image.device, dtype=image.dtype)
    else:
        mask = resize_mask(mask, image.shape[-2:])

    area = mask.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
    valid = (area >= float(min_area)).float()
    global_mean = image.mean(dim=(2, 3), keepdim=True)
    denom = area.clamp_min(1.0)
    local_mean = (image * mask).sum(dim=(2, 3), keepdim=True) / denom
    return torch.where(valid > 0, local_mean, global_mean), valid


def _rgb_distance_to_mean(image: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return (image - mean).pow(2).mean(dim=1, keepdim=True).clamp_min(1e-6).sqrt()


def build_lobe_cheek_accessory_zone(
    source_parsing: torch.Tensor | None,
    source_earring_mask: torch.Tensor | None,
    size: tuple[int, int],
    outside_margin: int = 56,
    inside_margin: int = 14,
    top_ratio: float = 0.42,
    bottom_extra_ratio: float = 0.14,
) -> torch.Tensor | None:
    if source_parsing is None:
        return None

    parsing_4d = ensure_mask_4d(source_parsing).long()
    if parsing_4d.shape[-2:] != size:
        parsing_4d = F.interpolate(parsing_4d.float(), size=size, mode="nearest").long()

    face_surface = parsing_label_mask(parsing_4d, RAW_FACE_SURFACE_LABELS, size)
    if face_surface is None:
        return None

    device = face_surface.device
    dtype = face_surface.dtype
    batch, _, height, width = face_surface.shape
    zone = torch.zeros_like(face_surface)

    outside_margin = max(8, int(outside_margin))
    inside_margin = max(2, int(inside_margin))
    top_ratio = max(0.25, min(0.70, float(top_ratio)))
    bottom_extra_ratio = max(0.0, min(0.30, float(bottom_extra_ratio)))

    x_coords = torch.linspace(0, 1, width, device=device, dtype=dtype).view(1, 1, 1, width)
    left_half = (x_coords <= 0.5).float()
    right_half = 1 - left_half

    for batch_idx in range(batch):
        for left_side, half in ((True, left_half), (False, right_half)):
            face_side = face_surface[batch_idx:batch_idx + 1] * half
            points = torch.nonzero(face_side[0, 0] > 0.05, as_tuple=False)
            if points.size(0) < 16:
                continue

            y_min = int(points[:, 0].min().item())
            y_max = int(points[:, 0].max().item())
            span_y = max(1, y_max - y_min)
            y0 = max(0, int(round(y_min + top_ratio * span_y)))
            y1 = min(height - 1, int(round(y_max + bottom_extra_ratio * span_y)))
            for y in range(y0, y1 + 1):
                row = torch.nonzero(face_side[0, 0, y] > 0.05, as_tuple=False).flatten()
                if row.numel() == 0:
                    band = points[(points[:, 0].float() - float(y)).abs() <= 5]
                    if band.size(0) == 0:
                        continue
                    row = band[:, 1]
                if left_side:
                    edge_x = int(row.min().item())
                    x0 = max(0, edge_x - outside_margin)
                    x1 = min(width - 1, edge_x + inside_margin)
                else:
                    edge_x = int(row.max().item())
                    x0 = max(0, edge_x - inside_margin)
                    x1 = min(width - 1, edge_x + outside_margin)
                zone[batch_idx, 0, y, x0:x1 + 1] = 1.0

    left_ear = parsing_label_mask(parsing_4d, (RAW_LEFT_EAR,), size)
    right_ear = parsing_label_mask(parsing_4d, (RAW_RIGHT_EAR,), size)
    ear_region = torch.zeros_like(face_surface)
    if left_ear is not None:
        ear_region = torch.clamp(ear_region + left_ear, 0, 1)
    if right_ear is not None:
        ear_region = torch.clamp(ear_region + right_ear, 0, 1)
    if source_earring_mask is not None:
        source_earring_mask = resize_mask(source_earring_mask, size)
    else:
        source_earring_mask = torch.zeros_like(face_surface)

    lobe_zone = torch.clamp(
        dilate_mask(ear_region, 17)
        + shift_mask(dilate_mask(ear_region, 15), down=10)
        + shift_mask(dilate_mask(ear_region, 13), down=24)
        + 0.70 * shift_mask(dilate_mask(ear_region, 11), down=40)
        + dilate_mask(source_earring_mask, 11),
        0,
        1,
    )
    combined = torch.clamp(zone + lobe_zone, 0, 1)
    if face_surface is not None:
        face_inner = erode_mask(face_surface, 13)
        combined = combined * (1 - 0.75 * face_inner).clamp(0, 1)
    return combined.clamp(0, 1)


def build_target_earring_safe_zone(
    target_parsing: torch.Tensor | None,
    size: tuple[int, int],
    placement_roi: torch.Tensor | None = None,
    outside_margin: int = 34,
    inside_margin: int = 5,
    top_ratio: float = 0.40,
    bottom_extra_ratio: float = 0.14,
    ear_dilate: int = 19,
    lobe_down_shift: int = 18,
    lobe_extra_down_shift: int = 36,
    face_inner_erode: int = 15,
    placement_dilate: int = 5,
) -> torch.Tensor | None:
    """Target-side write gate for earrings.

    The search/query ROI is intentionally broad. This gate is the stricter
    region where the model is allowed to write accessory detail: visible ears,
    lobe/below-ear extensions, and a narrow band along the outer cheek contour.
    """
    if target_parsing is None:
        return resize_mask(placement_roi, size) if placement_roi is not None else None

    parsing_4d = ensure_mask_4d(target_parsing).long()
    if parsing_4d.shape[-2:] != size:
        parsing_4d = F.interpolate(parsing_4d.float(), size=size, mode="nearest").long()

    face_surface = parsing_label_mask(parsing_4d, RAW_FACE_SURFACE_LABELS, size)
    if face_surface is None:
        return resize_mask(placement_roi, size) if placement_roi is not None else None

    device = face_surface.device
    dtype = face_surface.dtype
    batch, _, height, width = face_surface.shape
    contour_zone = torch.zeros_like(face_surface)

    outside_margin = max(6, int(outside_margin))
    inside_margin = max(1, int(inside_margin))
    top_ratio = max(0.25, min(0.70, float(top_ratio)))
    bottom_extra_ratio = max(0.0, min(0.30, float(bottom_extra_ratio)))

    x_coords = torch.linspace(0, 1, width, device=device, dtype=dtype).view(1, 1, 1, width)
    left_half = (x_coords <= 0.5).float()
    right_half = 1 - left_half

    for batch_idx in range(batch):
        for left_side, half in ((True, left_half), (False, right_half)):
            face_side = face_surface[batch_idx:batch_idx + 1] * half
            points = torch.nonzero(face_side[0, 0] > 0.05, as_tuple=False)
            if points.size(0) < 16:
                continue

            y_min = int(points[:, 0].min().item())
            y_max = int(points[:, 0].max().item())
            span_y = max(1, y_max - y_min)
            y0 = max(0, int(round(y_min + top_ratio * span_y)))
            y1 = min(height - 1, int(round(y_max + bottom_extra_ratio * span_y)))
            for y in range(y0, y1 + 1):
                row = torch.nonzero(face_side[0, 0, y] > 0.05, as_tuple=False).flatten()
                if row.numel() == 0:
                    band = points[(points[:, 0].float() - float(y)).abs() <= 5]
                    if band.size(0) == 0:
                        continue
                    row = band[:, 1]
                if left_side:
                    edge_x = int(row.min().item())
                    x0 = max(0, edge_x - outside_margin)
                    x1 = min(width - 1, edge_x + inside_margin)
                else:
                    edge_x = int(row.max().item())
                    x0 = max(0, edge_x - inside_margin)
                    x1 = min(width - 1, edge_x + outside_margin)
                contour_zone[batch_idx, 0, y, x0:x1 + 1] = 1.0

    left_ear = parsing_label_mask(parsing_4d, (RAW_LEFT_EAR,), size)
    right_ear = parsing_label_mask(parsing_4d, (RAW_RIGHT_EAR,), size)
    ear_region = torch.zeros_like(face_surface)
    if left_ear is not None:
        ear_region = torch.clamp(ear_region + left_ear, 0, 1)
    if right_ear is not None:
        ear_region = torch.clamp(ear_region + right_ear, 0, 1)

    ear_dilate = _scaled_value(ear_dilate, size, minimum=3)
    lobe_down_shift = _scaled_value(lobe_down_shift, size)
    lobe_extra_down_shift = _scaled_value(lobe_extra_down_shift, size)
    lobe_zone = torch.clamp(
        dilate_mask(ear_region, ear_dilate)
        + shift_mask(dilate_mask(ear_region, max(3, ear_dilate - 2)), down=lobe_down_shift)
        + 0.70 * shift_mask(dilate_mask(ear_region, max(3, ear_dilate - 4)), down=lobe_extra_down_shift),
        0,
        1,
    )

    safe_zone = torch.clamp(contour_zone + lobe_zone, 0, 1)
    placement = resize_mask(placement_roi, size) if placement_roi is not None else None
    if placement is not None:
        placement_select = dilate_mask(placement, _scaled_value(placement_dilate, size, minimum=1))
        gated = safe_zone * placement_select
        gated_present = (gated.flatten(1).sum(dim=1) >= 3).view(-1, 1, 1, 1)
        safe_zone = torch.where(gated_present, gated, safe_zone)

    face_inner = erode_mask(face_surface, _scaled_value(face_inner_erode, size, minimum=3))
    ear_keep = torch.clamp(dilate_mask(lobe_zone + ear_region, _scaled_value(9, size, minimum=3)), 0, 1)
    safe_zone = safe_zone * (1 - face_inner * (1 - ear_keep).clamp(0, 1)).clamp(0, 1)

    if placement is not None:
        fallback = dilate_mask(placement, 3) * (1 - face_inner).clamp(0, 1)
        empty = (safe_zone.flatten(1).sum(dim=1) < 3).view(-1, 1, 1, 1)
        safe_zone = torch.where(empty, fallback, safe_zone)

    return safe_zone.clamp(0, 1)


def build_ppe1_earring_locator_masks(
    source_01: torch.Tensor,
    source_parsing: torch.Tensor | None,
    source_earring_mask: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    ear_roi_dilate: int = 11,
    ear_lobe_shift: int = 10,
    source_hair_exclude_dilate: int = 2,
    weak_high_threshold: float = 0.018,
    weak_chroma_threshold: float = 0.030,
    weak_contrast_threshold: float = 0.016,
    support_high_threshold: float = 0.006,
    support_chroma_threshold: float = 0.018,
    support_contrast_threshold: float = 0.009,
    object_seed_dilate: int = 1,
    object_support_dilate: int = 2,
    object_grow_iters: int = 1,
    parser_weak_support_dilate: int = 5,
    component_min_area: int = 3,
    component_max_area_ratio: float = 0.025,
    component_keep_top: int = 8,
    weak_fallback: bool = True,
) -> dict[str, torch.Tensor]:
    """
    PPE1-style deterministic source-side earring locator for V54 raw parser labels.

    It builds the target used for training. Parser earring labels are treated as
    seeds when available; otherwise a local high-frequency/chroma/contrast weak
    detector searches only around the ear/lobe ROI.
    """
    source_01 = normalized_to_01(source_01)
    size = source_01.shape[-2:]
    empty = torch.zeros(source_01.size(0), 1, size[0], size[1], device=source_01.device, dtype=source_01.dtype)

    parser_earring = parsing_label_mask(source_parsing, (RAW_EARRING,), size)
    source_ear = None
    left_ear = parsing_label_mask(source_parsing, (RAW_LEFT_EAR,), size)
    right_ear = parsing_label_mask(source_parsing, (RAW_RIGHT_EAR,), size)
    if left_ear is not None or right_ear is not None:
        source_ear = torch.clamp((left_ear if left_ear is not None else empty) + (right_ear if right_ear is not None else empty), 0, 1)
    if parser_earring is None:
        parser_earring = empty
    if source_earring_mask is not None:
        parser_earring = torch.clamp(parser_earring + resize_mask(source_earring_mask, size), 0, 1)

    if source_ear is None:
        source_ear = empty

    lobe_shift = _scaled_value(ear_lobe_shift, size)
    ear_base = torch.clamp(source_ear + parser_earring, 0, 1)
    lobe_probe = shift_mask(source_ear if source_ear.flatten(1).amax().item() > 0 else ear_base, down=lobe_shift)
    source_ear_roi = torch.clamp(ear_base + 0.75 * lobe_probe, 0, 1)
    source_ear_roi = dilate_mask(source_ear_roi, _scaled_value(ear_roi_dilate, size, minimum=1))

    spatial_gate = build_source_earring_spatial_gate(source_parsing, parser_earring, size)
    anatomy_zone = build_lobe_cheek_accessory_zone(source_parsing, parser_earring, size)
    if spatial_gate is not None:
        source_ear_roi = source_ear_roi * torch.clamp(spatial_gate + dilate_mask(parser_earring, 9), 0, 1)
        empty_roi = source_ear_roi.flatten(1).amax(dim=1).view(-1, 1, 1, 1) <= 0
        source_ear_roi = torch.where(empty_roi, spatial_gate, source_ear_roi)
    if anatomy_zone is not None:
        source_ear_roi = torch.clamp(source_ear_roi + dilate_mask(parser_earring, 7), 0, 1) * anatomy_zone
        empty_roi = source_ear_roi.flatten(1).amax(dim=1).view(-1, 1, 1, 1) <= 0
        source_ear_roi = torch.where(empty_roi, anatomy_zone, source_ear_roi)
    source_ear_roi = source_ear_roi.clamp(0, 1)

    gray = rgb_to_gray(source_01)
    local_small = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
    local_large = F.avg_pool2d(gray, kernel_size=21, stride=1, padding=10)
    high = (gray - local_small).abs()
    contrast = (gray - local_large).abs()
    chroma = source_01.amax(dim=1, keepdim=True) - source_01.amin(dim=1, keepdim=True)

    weak_seed = (
        (high > float(weak_high_threshold)).float()
        * torch.clamp(
            (chroma > float(weak_chroma_threshold)).float()
            + (contrast > float(weak_contrast_threshold)).float(),
            0,
            1,
        )
    )
    weak_support = (
        (high > float(support_high_threshold)).float()
        * torch.clamp(
            (chroma > float(support_chroma_threshold)).float()
            + (contrast > float(support_contrast_threshold)).float(),
            0,
            1,
        )
    )

    hair = resize_mask(source_hair_mask, size) if source_hair_mask is not None else parsing_label_mask(source_parsing, (RAW_HAIR,), size)
    face_surface = parsing_label_mask(source_parsing, RAW_FACE_SURFACE_LABELS, size)
    background = None
    if source_parsing is not None:
        parsing_4d = ensure_mask_4d(source_parsing).long()
        if parsing_4d.shape[-2:] != size:
            parsing_4d = F.interpolate(parsing_4d.float(), size=size, mode="nearest").long()
        background = (parsing_4d == 0).float()
    skin_sample = erode_mask(face_surface, 9) if face_surface is not None else None
    hair_sample = erode_mask(hair, 7) if hair is not None else None
    bg_sample = background
    local_bg_sample = None
    if background is not None:
        local_bg_sample = background * dilate_mask(source_ear_roi, 39)
        bg_sample = torch.where(
            (local_bg_sample.flatten(1).sum(dim=1).view(-1, 1, 1, 1) >= 16),
            local_bg_sample,
            background,
        )
    skin_mean, skin_valid = _masked_rgb_mean_and_valid(source_01, skin_sample, min_area=48)
    hair_mean, hair_valid = _masked_rgb_mean_and_valid(source_01, hair_sample, min_area=48)
    bg_mean, bg_valid = _masked_rgb_mean_and_valid(source_01, bg_sample, min_area=48)
    dist_skin = _rgb_distance_to_mean(source_01, skin_mean)
    dist_hair = _rgb_distance_to_mean(source_01, hair_mean)
    dist_bg = _rgb_distance_to_mean(source_01, bg_mean)
    dist_hair_valid = torch.where(hair_valid > 0, dist_hair, torch.ones_like(dist_hair))
    dist_bg_valid = torch.where(bg_valid > 0, dist_bg, torch.ones_like(dist_bg))
    dist_base = torch.minimum(dist_skin, torch.minimum(dist_hair_valid, dist_bg_valid))
    color_score = torch.clamp(0.95 * dist_skin + 0.55 * dist_base + 0.35 * chroma, 0, 1) * source_ear_roi
    texture_score = torch.clamp(high + contrast + 0.75 * chroma, 0, 1) * source_ear_roi
    color_seed = masked_adaptive_threshold(color_score, source_ear_roi, std_scale=0.20, min_delta=0.010, min_area=8)
    color_support = masked_adaptive_threshold(color_score, source_ear_roi, std_scale=0.06, min_delta=0.004, min_area=8)
    texture_seed = masked_adaptive_threshold(texture_score, source_ear_roi, std_scale=0.12, min_delta=0.004, min_area=8)
    texture_support = masked_adaptive_threshold(texture_score, source_ear_roi, std_scale=0.04, min_delta=0.002, min_area=8)
    color_seed = color_seed * torch.clamp(texture_support + dilate_mask(parser_earring, 5), 0, 1)
    color_support = color_support * torch.clamp(texture_support + texture_seed + dilate_mask(parser_earring, 7), 0, 1)

    if hair is not None:
        hair_block = dilate_mask(hair, _scaled_value(source_hair_exclude_dilate, size, minimum=1))
        weak_seed = weak_seed * (1 - 0.70 * hair_block).clamp(0, 1)
        weak_support = weak_support * (1 - 0.45 * hair_block).clamp(0, 1)
    if face_surface is not None:
        face_core = erode_mask(face_surface, 7)
        weak_seed = weak_seed * (1 - 0.55 * face_core).clamp(0, 1)
        weak_support = weak_support * (1 - 0.35 * erode_mask(face_surface, 9)).clamp(0, 1)
        color_seed = color_seed * (1 - 0.35 * face_core).clamp(0, 1)
        color_support = color_support * (1 - 0.20 * erode_mask(face_surface, 9)).clamp(0, 1)
    weak_seed = torch.clamp(weak_seed + color_seed + parser_earring, 0, 1) * source_ear_roi
    weak_support = torch.clamp(weak_support + color_support + dilate_mask(parser_earring, 3), 0, 1) * source_ear_roi

    min_area = _scaled_area(component_min_area, size)
    has_parser = (parser_earring.flatten(1).sum(dim=1) >= min_area).view(-1, 1, 1, 1)
    parser_near = dilate_mask(parser_earring, _scaled_value(parser_weak_support_dilate, size, minimum=1)) * source_ear_roi
    parser_seed = dilate_mask(parser_earring, _scaled_value(object_seed_dilate, size, minimum=1)) * source_ear_roi
    parser_support = (
        parser_near
        * torch.clamp(weak_support + dilate_mask(parser_earring, _scaled_value(object_support_dilate, size, minimum=1)), 0, 1)
    )

    if weak_fallback:
        weak_object_seed = dilate_mask(weak_seed, _scaled_value(object_seed_dilate, size, minimum=1)) * source_ear_roi
        weak_object_support = weak_support * source_ear_roi
    else:
        weak_object_seed = empty
        weak_object_support = empty

    seed = torch.where(has_parser, parser_seed, weak_object_seed)
    support = torch.where(has_parser, parser_support, weak_object_support)
    object_mask = seed
    for _ in range(max(0, int(object_grow_iters))):
        object_mask = dilate_mask(object_mask, 3) * support
    object_mask = torch.clamp(object_mask + seed, 0, 1) * source_ear_roi
    object_mask = _filter_locator_components(
        object_mask,
        min_area=min_area,
        max_area_ratio=float(component_max_area_ratio),
        keep_top=int(component_keep_top),
    )
    color_object = torch.clamp(dilate_mask(color_seed, 3) * color_support * source_ear_roi, 0, 1)
    color_object = clean_earring_candidate_mask(
        color_object,
        source_01,
        support_mask=source_ear_roi,
        query_mask=source_ear_roi,
        source_parsing=source_parsing,
        source_hair_mask=hair,
        max_area_frac=min(float(component_max_area_ratio), 0.022),
        min_area=max(2, min_area),
        dilate=1,
    )
    object_mask = torch.clamp(object_mask + color_object, 0, 1) * source_ear_roi

    return {
        "earring_locator_mask": object_mask.clamp(0, 1),
        "earring_locator_seed": weak_seed.clamp(0, 1),
        "earring_locator_support": weak_support.clamp(0, 1),
        "earring_locator_roi": source_ear_roi.clamp(0, 1),
        "earring_locator_color_seed": color_seed.clamp(0, 1),
        "earring_locator_color_object": color_object.clamp(0, 1),
        "earring_locator_color_score": color_score.clamp(0, 1),
    }


def _half_plane(mask: torch.Tensor, left_side: bool) -> torch.Tensor:
    width = mask.size(-1)
    x_coords = torch.linspace(0, 1, width, device=mask.device, dtype=mask.dtype).view(1, 1, 1, width)
    return (x_coords <= 0.5).float() if left_side else (x_coords > 0.5).float()


def _top_anchor(mask_2d: torch.Tensor, min_area: float) -> tuple[float, float] | None:
    points = torch.nonzero(mask_2d > 0.05, as_tuple=False)
    if points.size(0) < min_area:
        return None

    y_values = points[:, 0]
    top_y = y_values.min()
    band = points[y_values <= top_y + 5]
    weights = mask_2d[band[:, 0], band[:, 1]].float().clamp_min(1e-4)
    y = (band[:, 0].float() * weights).sum() / weights.sum()
    x = (band[:, 1].float() * weights).sum() / weights.sum()
    return float(y.item()), float(x.item())


def _mask_centroid_anchor(mask_2d: torch.Tensor, min_area: float) -> tuple[float, float] | None:
    points = torch.nonzero(mask_2d > 0.05, as_tuple=False)
    if points.size(0) < min_area:
        return None
    weights = mask_2d[points[:, 0], points[:, 1]].float().clamp_min(1e-4)
    y = (points[:, 0].float() * weights).sum() / weights.sum()
    x = (points[:, 1].float() * weights).sum() / weights.sum()
    return float(y.item()), float(x.item())


def _side_outer_lobe_anchor(mask_2d: torch.Tensor, left_side: bool, attach_y_ratio: float, min_area: float) -> tuple[float, float] | None:
    points = torch.nonzero(mask_2d > 0.05, as_tuple=False)
    if points.size(0) < min_area:
        return None

    y_min = points[:, 0].float().min()
    y_max = points[:, 0].float().max()
    span_y = (y_max - y_min).clamp_min(1.0)
    target_y = y_min + max(0.45, min(0.92, float(attach_y_ratio))) * span_y
    band = points[(points[:, 0].float() - target_y).abs() <= 6]
    if band.size(0) == 0:
        band = points[points[:, 0].float() >= y_min + 0.50 * span_y]
    if band.size(0) == 0:
        band = points

    weights = mask_2d[band[:, 0], band[:, 1]].float().clamp_min(1e-4)
    y = (band[:, 0].float() * weights).sum() / weights.sum()
    x_values = band[:, 1].float()
    if left_side:
        edge_x = x_values.min()
    else:
        edge_x = x_values.max()
    centroid_x = (x_values * weights).sum() / weights.sum()
    x = 0.70 * edge_x + 0.30 * centroid_x
    return float(y.item()), float(x.item())


def _target_lobe_anchor(
    ear_2d: torch.Tensor,
    fallback_2d: torch.Tensor | None,
    attach_y_ratio: float,
    min_area: float,
    fallback_is_anchor: bool = False,
) -> tuple[float, float] | None:
    candidate = ear_2d
    using_fallback = False
    if torch.count_nonzero(candidate > 0.05).item() < min_area and fallback_2d is not None:
        candidate = fallback_2d
        using_fallback = True

    points = torch.nonzero(candidate > 0.05, as_tuple=False)
    if points.size(0) < min_area:
        return None

    if using_fallback and fallback_is_anchor:
        weights = candidate[points[:, 0], points[:, 1]].float().clamp_min(1e-4)
        y = (points[:, 0].float() * weights).sum() / weights.sum()
        x = (points[:, 1].float() * weights).sum() / weights.sum()
        return float(y.item()), float(x.item())

    y_min = points[:, 0].float().min()
    y_max = points[:, 0].float().max()
    target_y = y_min + attach_y_ratio * (y_max - y_min).clamp_min(1.0)
    lower_band = points[points[:, 0].float() >= y_min + 0.55 * (y_max - y_min).clamp_min(1.0)]
    if lower_band.size(0) == 0:
        lower_band = points

    weights = candidate[lower_band[:, 0], lower_band[:, 1]].float().clamp_min(1e-4)
    target_x = (lower_band[:, 1].float() * weights).sum() / weights.sum()
    return float(target_y.item()), float(target_x.item())


def _build_anchor_window_roi(
    primary_mask: torch.Tensor,
    fallback_mask: torch.Tensor | None,
    left_side: bool,
    down_shift: int,
    outer_shift: int,
    radius_y: int,
    radius_x: int,
    attach_y_ratio: float,
    min_area: float = 4.0,
    fallback_is_anchor: bool = False,
) -> torch.Tensor:
    primary_mask = ensure_mask_4d(primary_mask).float()
    fallback_mask = ensure_mask_4d(fallback_mask).float() if fallback_mask is not None else None
    output = torch.zeros_like(primary_mask)
    batch, _, height, width = primary_mask.shape
    yy = torch.arange(height, device=primary_mask.device, dtype=primary_mask.dtype).view(1, height, 1)
    xx = torch.arange(width, device=primary_mask.device, dtype=primary_mask.dtype).view(1, 1, width)
    radius_y = max(1, int(radius_y))
    radius_x = max(1, int(radius_x))
    side = _half_plane(primary_mask, left_side)
    side_offset = -int(outer_shift) if left_side else int(outer_shift)

    for batch_idx in range(batch):
        primary_2d = primary_mask[batch_idx:batch_idx + 1] * side
        fallback_2d = fallback_mask[batch_idx:batch_idx + 1] * side if fallback_mask is not None else None
        anchor = _target_lobe_anchor(
            primary_2d[0, 0],
            fallback_2d[0, 0] if fallback_2d is not None else None,
            attach_y_ratio,
            min_area,
            fallback_is_anchor=fallback_is_anchor,
        )
        if anchor is None:
            continue
        center_y = max(0.0, min(float(height - 1), anchor[0] + int(down_shift)))
        center_x = max(0.0, min(float(width - 1), anchor[1] + side_offset))
        ellipse = (
            ((yy - center_y) / float(radius_y)).pow(2)
            + ((xx - center_x) / float(radius_x)).pow(2)
            <= 1.0
        ).float().unsqueeze(0)
        output[batch_idx:batch_idx + 1] = ellipse * side
    return output.clamp(0, 1)


def _build_face_side_lobe_fallback(
    target_face_mask: torch.Tensor,
    side_seed: torch.Tensor,
    left_side: bool,
    vertical_ratio: float = 0.58,
    outer_offset: int = 10,
    min_area: float = 16.0,
) -> torch.Tensor:
    target_face_mask = ensure_mask_4d(target_face_mask).float()
    side_seed = ensure_mask_4d(side_seed).float()
    output = torch.zeros_like(target_face_mask)
    batch, _, height, width = target_face_mask.shape
    side = _half_plane(target_face_mask, left_side)
    face_side = target_face_mask * side

    for batch_idx in range(batch):
        points = torch.nonzero(face_side[batch_idx, 0] > 0.05, as_tuple=False)
        if points.size(0) < min_area:
            points = torch.nonzero((side_seed[batch_idx:batch_idx + 1] * side)[0, 0] > 0.05, as_tuple=False)
        if points.size(0) < min_area:
            continue

        y_min = points[:, 0].float().min()
        y_max = points[:, 0].float().max()
        target_y = y_min + max(0.35, min(0.85, float(vertical_ratio))) * (y_max - y_min).clamp_min(1.0)
        band = points[(points[:, 0].float() - target_y).abs() <= 8]
        if band.size(0) == 0:
            band = points
        if left_side:
            edge_x = band[:, 1].float().min()
            target_x = edge_x - int(outer_offset)
        else:
            edge_x = band[:, 1].float().max()
            target_x = edge_x + int(outer_offset)
        target_y = max(0.0, min(float(height - 1), float(target_y.item())))
        target_x = max(0.0, min(float(width - 1), float(target_x.item())))
        output[batch_idx, 0, int(round(target_y)), int(round(target_x))] = 1.0
    return dilate_mask(output, 5)


def build_aligned_earring_reference(
    source_01: torch.Tensor,
    earring_mask: torch.Tensor,
    target_parsing: torch.Tensor | None,
    visible_ear_roi: torch.Tensor | None = None,
    align_strength: float = 1.0,
    max_shift: int = 26,
    attach_y_ratio: float = 0.78,
    min_area: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_01 = normalized_to_01(source_01)
    earring_mask = resize_mask(earring_mask, source_01.shape[-2:])
    if target_parsing is None or align_strength <= 0:
        return source_01, earring_mask.clamp(0, 1)

    target_masks = build_raw_face_masks(target_parsing)
    left_target_ear = resize_mask(target_masks["left_ear"], source_01.shape[-2:])
    right_target_ear = resize_mask(target_masks["right_ear"], source_01.shape[-2:])
    visible_ear_roi = (
        resize_mask(visible_ear_roi, source_01.shape[-2:])
        if visible_ear_roi is not None
        else torch.clamp(left_target_ear + right_target_ear, 0, 1)
    )

    numer = torch.zeros_like(source_01)
    denom = torch.zeros_like(earring_mask)
    carried_mask = torch.zeros_like(earring_mask)
    consumed_source_mask = torch.zeros_like(earring_mask)
    max_shift = max(0, int(max_shift))
    attach_y_ratio = max(0.45, min(0.95, float(attach_y_ratio)))
    align_strength = max(0.0, min(1.0, float(align_strength)))

    for batch_idx in range(source_01.size(0)):
        for left_side, target_ear in ((True, left_target_ear), (False, right_target_ear)):
            half = _half_plane(earring_mask[batch_idx:batch_idx + 1], left_side)
            side_mask = earring_mask[batch_idx:batch_idx + 1] * half
            if side_mask.sum().item() < min_area:
                continue
            consumed_source_mask[batch_idx:batch_idx + 1] = torch.clamp(
                consumed_source_mask[batch_idx:batch_idx + 1] + side_mask,
                0,
                1,
            )

            fallback = visible_ear_roi[batch_idx:batch_idx + 1] * half
            source_anchor = _top_anchor(side_mask[0, 0], min_area)
            target_anchor = _target_lobe_anchor(
                target_ear[batch_idx, 0],
                None,
                attach_y_ratio,
                min_area,
            )
            if target_anchor is None:
                target_anchor = _side_outer_lobe_anchor(
                    fallback[0, 0],
                    left_side,
                    attach_y_ratio,
                    min_area,
                )

            if source_anchor is None or target_anchor is None:
                shifted_mask = side_mask
                shifted_patch = source_01[batch_idx:batch_idx + 1] * side_mask
            else:
                dy = int(round((target_anchor[0] - source_anchor[0]) * align_strength))
                dx = int(round((target_anchor[1] - source_anchor[1]) * align_strength))
                dy = max(-max_shift, min(max_shift, dy))
                dx = max(-max_shift, min(max_shift, dx))
                shifted_mask = shift_mask(side_mask, down=dy, right=dx)
                shifted_patch = shift_tensor(source_01[batch_idx:batch_idx + 1] * side_mask, down=dy, right=dx)

            numer[batch_idx:batch_idx + 1] += shifted_patch
            denom[batch_idx:batch_idx + 1] += shifted_mask
            carried_mask[batch_idx:batch_idx + 1] = torch.clamp(
                carried_mask[batch_idx:batch_idx + 1] + shifted_mask,
                0,
                1,
            )

    unchanged_mask = (
        earring_mask
        * (1 - consumed_source_mask).clamp(0, 1)
        * (1 - dilate_mask(carried_mask, 3)).clamp(0, 1)
    ).clamp(0, 1)
    numer = numer + source_01 * unchanged_mask
    denom = (denom + unchanged_mask).clamp_min(1e-6)
    reference_patch = (numer / denom).clamp(0, 1)
    reference_mask = torch.clamp(carried_mask + unchanged_mask, 0, 1)
    reference_image = source_01 * (1 - reference_mask) + reference_patch * reference_mask
    return reference_image.clamp(0, 1), reference_mask.clamp(0, 1)


def build_source_earring_detection_roi(
    source_parsing: torch.Tensor | None,
    source_earring_mask: torch.Tensor | None,
    size: tuple[int, int],
    ear_dilate: int = 11,
    earring_dilate: int = 19,
) -> torch.Tensor:
    if source_earring_mask is None:
        if source_parsing is None:
            raise ValueError("source_parsing or source_earring_mask is required")
        source_parsing_4d = ensure_mask_4d(source_parsing)
        source_earring_mask = torch.zeros(
            source_parsing_4d.size(0),
            1,
            size[0],
            size[1],
            device=source_parsing_4d.device,
            dtype=torch.float32,
        )
    else:
        source_earring_mask = resize_mask(source_earring_mask, size)

    if source_parsing is None:
        source_left_ear = torch.zeros_like(source_earring_mask)
        source_right_ear = torch.zeros_like(source_earring_mask)
        face_surface = None
    else:
        source_left_ear = parsing_label_mask(source_parsing, (RAW_LEFT_EAR,), size)
        source_right_ear = parsing_label_mask(source_parsing, (RAW_RIGHT_EAR,), size)
        face_surface = parsing_label_mask(source_parsing, RAW_FACE_SURFACE_LABELS, size)
        if source_left_ear is None:
            source_left_ear = torch.zeros_like(source_earring_mask)
        if source_right_ear is None:
            source_right_ear = torch.zeros_like(source_earring_mask)

    source_ear_roi = torch.clamp(
        dilate_mask(source_left_ear + source_right_ear, ear_dilate)
        + dilate_mask(source_earring_mask, earring_dilate),
        0,
        1,
    )
    if face_surface is not None:
        seed = torch.clamp(source_ear_roi + source_earring_mask, 0, 1)
        left_fallback = _build_face_side_lobe_fallback(
            face_surface,
            seed,
            True,
            vertical_ratio=0.58,
            outer_offset=7,
            min_area=16.0,
        )
        right_fallback = _build_face_side_lobe_fallback(
            face_surface,
            seed,
            False,
            vertical_ratio=0.58,
            outer_offset=7,
            min_area=16.0,
        )
        source_ear_roi = torch.clamp(source_ear_roi + dilate_mask(left_fallback + right_fallback, 9), 0, 1)

    detection_roi = torch.clamp(
        source_ear_roi
        + dilate_mask(source_earring_mask, 17)
        + shift_mask(source_ear_roi, down=12)
        + 0.60 * shift_mask(source_ear_roi, down=24)
        + 0.35 * shift_mask(source_ear_roi, down=36),
        0,
        1,
    )
    spatial_gate = build_source_earring_spatial_gate(source_parsing, source_earring_mask, size)
    if spatial_gate is not None:
        detection_roi = detection_roi * torch.clamp(spatial_gate + dilate_mask(source_earring_mask, 13), 0, 1)
    return detection_roi


def build_weak_earring_masks(
    source_01: torch.Tensor,
    visible_ear_roi: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    source_earring_mask: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    source_hair_block_mask: torch.Tensor | None = None,
    source_parsing: torch.Tensor | None = None,
    confident_dilate: int = 3,
    highlight_dilate: int = 3,
) -> dict[str, torch.Tensor]:
    source_01 = normalized_to_01(source_01)
    visible_ear_roi = resize_mask(visible_ear_roi, source_01.shape[-2:])
    query_mask = visible_ear_roi if query_mask is None else resize_mask(query_mask, source_01.shape[-2:])

    if source_earring_mask is None:
        source_earring_mask = torch.zeros_like(visible_ear_roi)
    else:
        source_earring_mask = resize_mask(source_earring_mask, source_01.shape[-2:])

    spatial_gate = build_source_earring_spatial_gate(source_parsing, source_earring_mask, source_01.shape[-2:])
    if spatial_gate is None:
        spatial_gate = torch.ones_like(visible_ear_roi)
    source_earring_mask = (source_earring_mask * torch.clamp(spatial_gate + dilate_mask(source_earring_mask, 5), 0, 1)).clamp(0, 1)

    lower_ear_roi = torch.clamp(
        visible_ear_roi
        + shift_mask(visible_ear_roi, down=10)
        + 0.55 * shift_mask(visible_ear_roi, down=20)
        + 0.25 * shift_mask(visible_ear_roi, down=30),
        0,
        1,
    )
    candidate_roi = torch.clamp(
        dilate_mask(lower_ear_roi, 5) + dilate_mask(query_mask, 5) + dilate_mask(source_earring_mask, 9),
        0,
        1,
    )
    transfer_roi = torch.clamp(candidate_roi + visible_ear_roi + dilate_mask(source_earring_mask, 15), 0, 1)
    texture_roi = candidate_roi
    if source_hair_mask is not None:
        source_hair_mask = resize_mask(source_hair_mask, source_01.shape[-2:])
        texture_roi = texture_roi * (1 - 0.35 * source_hair_mask).clamp(0, 1)
    if source_hair_block_mask is not None:
        source_hair_block_mask = resize_mask(source_hair_block_mask, source_01.shape[-2:])
        texture_roi = texture_roi * (1 - 0.25 * source_hair_block_mask).clamp(0, 1)
    texture_roi = texture_roi * spatial_gate

    evidence = build_earring_evidence_score(source_01)
    high_energy = evidence["high_energy"]
    chroma = evidence["chroma"]
    bright_spike = evidence["bright_spike"]
    dark_spike = evidence["dark_spike"]
    contrast_spike = evidence["contrast_spike"]

    edge_energy = evidence["edge_energy"]

    energy_mask = masked_adaptive_threshold(high_energy, texture_roi, std_scale=0.55, min_delta=0.008)
    edge_mask = masked_adaptive_threshold(edge_energy, texture_roi, std_scale=0.35, min_delta=0.010)
    chroma_mask = masked_adaptive_threshold(chroma, texture_roi, std_scale=0.70, min_delta=0.035)
    bright_mask = masked_adaptive_threshold(bright_spike, texture_roi, std_scale=0.30, min_delta=0.014)
    dark_mask = masked_adaptive_threshold(dark_spike, texture_roi, std_scale=0.34, min_delta=0.012)
    contrast_mask = masked_adaptive_threshold(contrast_spike, texture_roi, std_scale=0.32, min_delta=0.012)

    evidence_votes = (
        energy_mask
        + chroma_mask
        + bright_mask
        + dark_mask
        + edge_mask
        + contrast_mask
    )
    dark_contour = dark_mask * torch.clamp(edge_mask + contrast_mask + energy_mask, 0, 1)
    line_contour = edge_mask * torch.clamp(dark_mask + contrast_mask + chroma_mask, 0, 1) * torch.clamp(
        dark_mask + chroma_mask + source_earring_mask,
        0,
        1,
    )
    strong_texture = (
        (energy_mask * torch.clamp(chroma_mask + bright_mask + dark_mask + contrast_mask, 0, 1))
        + dark_contour
        + line_contour * torch.clamp(query_mask + dilate_mask(query_mask, 5), 0, 1)
        + ((evidence_votes >= 2.0).float() * torch.clamp(query_mask + dilate_mask(query_mask, 3), 0, 1))
        + ((evidence_votes >= 3.0).float() * candidate_roi)
    )
    texture_candidate = torch.clamp(
        strong_texture + source_earring_mask,
        0,
        1,
    ) * candidate_roi
    object_candidate = build_source_earring_object_mask(
        source_01,
        source_parsing,
        source_earring_mask,
        source_hair_mask,
        max_area_frac=0.018,
        min_area=4,
    )

    source_earring_clean = clean_earring_candidate_mask(
        torch.clamp(source_earring_mask + object_candidate, 0, 1),
        source_01,
        support_mask=transfer_roi,
        query_mask=query_mask,
        source_parsing=source_parsing,
        source_hair_mask=source_hair_mask,
        max_area_frac=0.026,
        min_area=4,
        dilate=3,
    )
    texture_candidate = clean_earring_candidate_mask(
        texture_candidate,
        source_01,
        support_mask=transfer_roi,
        query_mask=query_mask,
        source_parsing=source_parsing,
        source_hair_mask=source_hair_mask,
        max_area_frac=0.026,
        min_area=4,
        dilate=3,
    )

    earring_confident = torch.clamp(source_earring_clean + texture_candidate + object_candidate, 0, 1)
    earring_confident = dilate_mask(earring_confident, confident_dilate) * transfer_roi
    earring_confident = clean_earring_candidate_mask(
        earring_confident,
        source_01,
        support_mask=transfer_roi,
        query_mask=query_mask,
        source_parsing=source_parsing,
        source_hair_mask=source_hair_mask,
        max_area_frac=0.026,
        min_area=4,
        dilate=3,
    )
    sparkle_core = bright_mask * torch.clamp(energy_mask + chroma_mask + source_earring_mask, 0, 1)
    color_core = chroma_mask * energy_mask * torch.clamp(query_mask + source_earring_mask + earring_confident, 0, 1)
    dark_core = dark_mask * torch.clamp(energy_mask + contrast_mask + source_earring_mask, 0, 1)
    highlight_core = torch.clamp(sparkle_core + 0.5 * color_core + 0.25 * dark_core, 0, 1) * earring_confident
    earring_highlight = dilate_mask(highlight_core, highlight_dilate) * earring_confident

    return {
        "earring_confident_mask": earring_confident.clamp(0, 1),
        "earring_highlight_mask": earring_highlight.clamp(0, 1),
        "source_earring_clean_mask": source_earring_clean.clamp(0, 1),
        "source_earring_object_mask": object_candidate.clamp(0, 1),
        "earring_energy_mask": energy_mask.clamp(0, 1),
    }


def build_raw_face_masks(parsing: torch.Tensor) -> dict[str, torch.Tensor]:
    parsing = ensure_mask_4d(parsing).long()
    return {
        "left_ear": (parsing == RAW_LEFT_EAR).float(),
        "right_ear": (parsing == RAW_RIGHT_EAR).float(),
        "earring": (parsing == RAW_EARRING).float(),
        "face_surface": torch.stack([(parsing == label).float() for label in RAW_FACE_SURFACE_LABELS], dim=0).amax(dim=0),
        "detail": torch.stack([(parsing == label).float() for label in RAW_DETAIL_LABELS], dim=0).amax(dim=0),
        "hair": (parsing == RAW_HAIR).float(),
        "hat": (parsing == RAW_HAT).float(),
    }


class FaceParsingHelperV54:
    def __init__(self, parse_size: int = 512):
        self.parse_size = parse_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def parse(self, images: torch.Tensor, out_size: tuple[int, int] | None = None) -> torch.Tensor:
        images = normalized_to_01(images)
        out_size = out_size or tuple(images.shape[-2:])
        parse_images = F.interpolate(images, size=(self.parse_size, self.parse_size), mode="bilinear",
                                     align_corners=False)

        device = parse_images.device
        norm_mean = self.mean.to(device=device, dtype=parse_images.dtype)
        norm_std = self.std.to(device=device, dtype=parse_images.dtype)
        parse_images = (parse_images - norm_mean) / norm_std

        if device.type != "cuda":
            parse_images = parse_images.to("cuda")

        parsings = []
        for image in parse_images:
            parsing, _ = FaceParsing_tensor.parsing_img(image.unsqueeze(0))
            parsings.append(parsing)

        parsing = torch.stack(parsings, dim=0).unsqueeze(1).float()
        parsing = F.interpolate(parsing, size=out_size, mode="nearest")
        return parsing.long().to(images.device)


class HairMaskExtractorV54(nn.Module):
    def __init__(self, device: str = "cuda", dilate_erosion: int = 5):
        super().__init__()
        self.device = device
        self.seg = BiSeNet(n_classes=16).to(device)
        self.seg.load_state_dict(torch.load("pretrained_models/BiSeNet/seg.pth", map_location=device))
        self.seg.eval()
        for param in self.seg.parameters():
            param.requires_grad = False

        self.downsample_512 = BicubicDownSample(factor=2)
        self.dilate_erosion = DilateErosion(dilate_erosion=dilate_erosion, device=device)

    @torch.no_grad()
    def generate_mask(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = normalized_to_01(images).to(self.device)
        seg_in = (self.downsample_512(images) - seg_mean) / seg_std
        seg_logits, _, _ = self.seg(seg_in)
        current_mask = torch.argmax(seg_logits, dim=1).float()
        hair_mask = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
        hair_mask = F.interpolate(hair_mask.unsqueeze(1), size=(256, 256), mode="nearest")
        return self.dilate_erosion.mask(hair_mask)


class EarAnchoredQueryBuilder(nn.Module):
    def __init__(
        self,
        ear_dilate: int = 21,
        hair_change_dilate: int = 25,
        earring_expand: int = 15,
        downward_shift: int = 10,
        target_hair_dilate: int = 11,
        source_hair_block_dilate: int = 5,
        source_hair_block_strength: float = 0.95,
        target_visibility_expand: int = 5,
        max_target_hair_overlap: float = 0.55,
        earring_lobe_dilate: int = 5,
        earring_lobe_down_shift: int = 18,
        earring_outer_shift: int = 6,
        earring_query_floor: float = 0.45,
        earring_search_radius_y: int = 28,
        earring_search_radius_x: int = 18,
        earring_search_attach_y_ratio: float = 0.78,
    ):
        super().__init__()
        self.ear_dilate = ear_dilate
        self.hair_change_dilate = hair_change_dilate
        self.earring_expand = earring_expand
        self.downward_shift = downward_shift
        self.target_hair_dilate = target_hair_dilate
        self.source_hair_block_dilate = source_hair_block_dilate
        self.source_hair_block_strength = max(0.0, min(1.0, float(source_hair_block_strength)))
        self.target_visibility_expand = target_visibility_expand
        self.max_target_hair_overlap = max_target_hair_overlap
        self.earring_lobe_dilate = earring_lobe_dilate
        self.earring_lobe_down_shift = earring_lobe_down_shift
        self.earring_outer_shift = earring_outer_shift
        self.earring_query_floor = max(0.0, min(1.0, float(earring_query_floor)))
        self.earring_search_radius_y = earring_search_radius_y
        self.earring_search_radius_x = earring_search_radius_x
        self.earring_search_attach_y_ratio = max(0.45, min(0.95, float(earring_search_attach_y_ratio)))
        self.earring_face_fallback_y_ratio = 0.58

    def _split_by_side(self, mask: torch.Tensor, left_hint: torch.Tensor, right_hint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        width = mask.size(-1)
        x_coords = torch.linspace(0, 1, width, device=mask.device, dtype=mask.dtype).view(1, 1, 1, width)
        left_half = (x_coords <= 0.5).float()
        right_half = 1 - left_half

        left_mask = mask * torch.clamp(left_hint + left_half, 0, 1)
        right_mask = mask * torch.clamp(right_hint + right_half, 0, 1)
        return left_mask, right_mask

    @staticmethod
    def _roi_overlap(mask: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        mask = ensure_mask_4d(mask).float()
        roi = ensure_mask_4d(roi).float()
        numer = (mask * roi).flatten(1).sum(dim=1)
        denom = roi.flatten(1).sum(dim=1).clamp_min(1.0)
        return numer / denom

    def forward(
        self,
        source_parsing: torch.Tensor,
        target_parsing: torch.Tensor,
        source_hair_mask: torch.Tensor | None = None,
        target_hair_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        source_masks = build_raw_face_masks(source_parsing)
        target_masks = build_raw_face_masks(target_parsing)

        source_hair_mask = source_masks["hair"] if source_hair_mask is None else ensure_mask_4d(source_hair_mask).float()
        target_hair_mask = target_masks["hair"] if target_hair_mask is None else ensure_mask_4d(target_hair_mask).float()

        left_hint = dilate_mask(source_masks["left_ear"] + target_masks["left_ear"], max(3, self.ear_dilate // 2))
        right_hint = dilate_mask(source_masks["right_ear"] + target_masks["right_ear"], max(3, self.ear_dilate // 2))
        source_left_ring, source_right_ring = self._split_by_side(source_masks["earring"], left_hint, right_hint)
        target_left_ring, target_right_ring = self._split_by_side(target_masks["earring"], left_hint, right_hint)

        left_roi = dilate_mask(source_masks["left_ear"] + target_masks["left_ear"] + source_left_ring, self.ear_dilate)
        right_roi = dilate_mask(source_masks["right_ear"] + target_masks["right_ear"] + source_right_ring, self.ear_dilate)
        left_roi = torch.clamp(left_roi + shift_mask(left_roi, down=self.downward_shift), 0, 1)
        right_roi = torch.clamp(right_roi + shift_mask(right_roi, down=self.downward_shift), 0, 1)
        ear_roi = torch.clamp(left_roi + right_roi, 0, 1)

        hair_change = dilate_mask((source_hair_mask - target_hair_mask).abs(), self.hair_change_dilate)
        ring_hint = dilate_mask(source_masks["earring"], self.earring_expand)
        hat_mask = torch.clamp(source_masks["hat"] + target_masks["hat"], 0, 1)
        target_hair_context = dilate_mask(target_hair_mask, self.target_hair_dilate)
        source_hair_context = dilate_mask(source_hair_mask, self.source_hair_block_dilate)
        source_earring_keep = dilate_mask(source_masks["earring"], max(3, self.earring_expand // 2))

        left_target_visible = target_masks["left_ear"] + target_left_ring
        right_target_visible = target_masks["right_ear"] + target_right_ring
        left_target_visible = dilate_mask(left_target_visible * left_roi, self.target_visibility_expand) * left_roi
        right_target_visible = dilate_mask(right_target_visible * right_roi, self.target_visibility_expand) * right_roi

        left_hair_overlap = self._roi_overlap(target_hair_context, left_roi)
        right_hair_overlap = self._roi_overlap(target_hair_context, right_roi)
        left_has_visible_target = left_target_visible.flatten(1).amax(dim=1) > 0
        right_has_visible_target = right_target_visible.flatten(1).amax(dim=1) > 0

        left_side_visible = ((left_hair_overlap < self.max_target_hair_overlap) | left_has_visible_target).float()
        right_side_visible = ((right_hair_overlap < self.max_target_hair_overlap) | right_has_visible_target).float()
        left_side_visible_map = left_side_visible.view(-1, 1, 1, 1)
        right_side_visible_map = right_side_visible.view(-1, 1, 1, 1)

        left_open_region = left_roi * (1 - target_hair_context)
        right_open_region = right_roi * (1 - target_hair_context)
        left_visible_roi = torch.clamp(left_open_region + left_target_visible, 0, 1) * left_side_visible_map
        right_visible_roi = torch.clamp(right_open_region + right_target_visible, 0, 1) * right_side_visible_map

        target_face_surface = target_masks["face_surface"]
        left_target_lobe_anchor = torch.clamp(target_masks["left_ear"] + target_left_ring, 0, 1)
        right_target_lobe_anchor = torch.clamp(target_masks["right_ear"] + target_right_ring, 0, 1)
        left_face_fallback = _build_face_side_lobe_fallback(
            target_face_surface,
            left_roi + left_target_visible,
            True,
            vertical_ratio=self.earring_face_fallback_y_ratio,
            outer_offset=max(4, self.earring_outer_shift),
        )
        right_face_fallback = _build_face_side_lobe_fallback(
            target_face_surface,
            right_roi + right_target_visible,
            False,
            vertical_ratio=self.earring_face_fallback_y_ratio,
            outer_offset=max(4, self.earring_outer_shift),
        )
        left_fallback_anchor = torch.clamp(left_target_visible + left_face_fallback, 0, 1)
        right_fallback_anchor = torch.clamp(right_target_visible + right_face_fallback, 0, 1)
        left_lobe_roi = _build_anchor_window_roi(
            left_target_lobe_anchor,
            left_fallback_anchor,
            True,
            self.earring_lobe_down_shift,
            self.earring_outer_shift,
            self.earring_search_radius_y,
            self.earring_search_radius_x,
            self.earring_search_attach_y_ratio,
            fallback_is_anchor=True,
        )
        right_lobe_roi = _build_anchor_window_roi(
            right_target_lobe_anchor,
            right_fallback_anchor,
            False,
            self.earring_lobe_down_shift,
            self.earring_outer_shift,
            self.earring_search_radius_y,
            self.earring_search_radius_x,
            self.earring_search_attach_y_ratio,
            fallback_is_anchor=True,
        )
        if self.earring_lobe_dilate > 1:
            left_lobe_roi = dilate_mask(left_lobe_roi, self.earring_lobe_dilate)
            right_lobe_roi = dilate_mask(right_lobe_roi, self.earring_lobe_dilate)
        target_hair_soft_gate = (1 - 0.65 * target_hair_context).clamp(0, 1)
        left_earring_search_roi = left_lobe_roi * left_side_visible_map * target_hair_soft_gate
        right_earring_search_roi = right_lobe_roi * right_side_visible_map * target_hair_soft_gate
        earring_search_roi = torch.clamp(left_earring_search_roi + right_earring_search_roi, 0, 1) * (1 - hat_mask)

        visible_ear_roi = torch.clamp(left_visible_roi + right_visible_roi + earring_search_roi, 0, 1)

        earring_protect = dilate_mask(earring_search_roi + source_earring_keep + target_masks["earring"], 9)
        source_hair_block_mask = (
            torch.clamp(left_visible_roi + right_visible_roi, 0, 1)
            * source_hair_context
            * (1 - target_hair_context)
            * (1 - source_earring_keep)
            * (1 - earring_protect).clamp(0, 1)
            * (1 - hat_mask)
        ).clamp(0, 1)
        block_attenuation = (1 - self.source_hair_block_strength * source_hair_block_mask).clamp(0, 1)
        query_seed = torch.clamp(hair_change + 0.25 * ring_hint + self.earring_query_floor * earring_search_roi, 0, 1)
        search_attenuation = (1 - 0.35 * self.source_hair_block_strength * source_hair_block_mask).clamp(0, 1)
        query_mask = torch.clamp(
            visible_ear_roi * query_seed * block_attenuation
            + self.earring_query_floor * earring_search_roi * search_attenuation,
            0,
            1,
        )
        query_mask = query_mask * (1 - hat_mask)

        fallback_query = visible_ear_roi * (1 - hat_mask) * block_attenuation
        empty_query = query_mask.flatten(1).amax(dim=1).view(-1, 1, 1, 1) == 0
        query_mask = torch.where(empty_query, fallback_query, query_mask)

        presence_target = torch.stack(
            [
                source_left_ring.flatten(1).amax(dim=1),
                source_right_ring.flatten(1).amax(dim=1),
                source_masks["earring"].flatten(1).amax(dim=1),
            ],
            dim=1,
        ).float()
        visibility_target = torch.stack(
            [
                left_side_visible,
                right_side_visible,
                torch.maximum(left_side_visible, right_side_visible),
            ],
            dim=1,
        ).float()
        presence_target = presence_target * visibility_target

        source_left_ring = source_left_ring * torch.clamp(left_roi + left_visible_roi + left_earring_search_roi, 0, 1)
        source_right_ring = source_right_ring * torch.clamp(right_roi + right_visible_roi + right_earring_search_roi, 0, 1)
        source_earring_transfer_roi = torch.clamp(
            ear_roi + earring_search_roi + dilate_mask(visible_ear_roi, 5),
            0,
            1,
        ) * (1 - hat_mask)
        source_earring_mask = source_masks["earring"] * source_earring_transfer_roi

        return {
            "left_ear_roi": left_roi,
            "right_ear_roi": right_roi,
            "ear_roi": ear_roi,
            "visible_ear_roi": visible_ear_roi,
            "earring_search_roi": earring_search_roi,
            "query_mask": query_mask,
            "source_hair_block_mask": source_hair_block_mask,
            "source_hair_mask": source_hair_mask,
            "target_hair_mask": target_hair_mask,
            "source_earring_mask": source_earring_mask,
            "target_earring_mask": target_masks["earring"],
            "source_left_earring_mask": source_left_ring,
            "source_right_earring_mask": source_right_ring,
            "presence_target": presence_target,
            "visibility_target": visibility_target,
        }


class ShadowSuppressedHFExtractor(nn.Module):
    def __init__(
        self,
        low_alpha: float = 0.1,
        feature_channels: int = 128,
        blur_kernel: int = 11,
        blur_sigma: float = 3.0,
    ):
        super().__init__()
        self.low_alpha = low_alpha
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma

        self.low_gate = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.prior_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, feature_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(
        self,
        source_01: torch.Tensor,
        query_mask: torch.Tensor,
        prior_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query_mask = ensure_mask_4d(query_mask).float()
        prior_mask = query_mask if prior_mask is None else ensure_mask_4d(prior_mask).float()
        prior_mask = prior_mask * query_mask
        low = low_pass_filter(source_01, kernel_size=self.blur_kernel, sigma=self.blur_sigma)
        high = source_01 - low
        low_gate = self.low_gate(torch.cat([low, prior_mask], dim=1))
        prior = (high + self.low_alpha * low_gate * low) * prior_mask
        high_energy = high.abs().mean(dim=1, keepdim=True) * prior_mask
        prior_feature = self.prior_encoder(torch.cat([prior, prior_mask], dim=1))

        return {
            "low": low,
            "high": high,
            "prior": prior,
            "prior_feature": prior_feature,
            "low_gate": low_gate,
            "high_energy": high_energy,
            "prior_mask": prior_mask,
        }


class DynamicFineMaskRefresher(nn.Module):
    def __init__(self, hidden_channels: int = 32, init_bias: float = -4.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(12, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.mask_head = nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels // 2, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 3),
        )
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, float(init_bias))
        nn.init.zeros_(self.presence_head[-1].weight)
        nn.init.zeros_(self.presence_head[-1].bias)

    def forward(
        self,
        source_01: torch.Tensor,
        target_01: torch.Tensor,
        high_energy: torch.Tensor,
        query_mask: torch.Tensor,
        source_ear_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_mask = ensure_mask_4d(query_mask).float()
        source_ear_mask = ensure_mask_4d(source_ear_mask).float()
        residual = (source_01 - target_01).abs()
        features = self.backbone(torch.cat([source_01, target_01, residual, high_energy, query_mask, source_ear_mask], dim=1))
        fine_mask_logits = self.mask_head(features)
        fine_mask = torch.sigmoid(fine_mask_logits) * query_mask

        pooled_features = features * (query_mask + source_ear_mask).clamp(0, 1)
        presence_logits = self.presence_head(pooled_features)
        return {
            "fine_mask_logits": fine_mask_logits,
            "fine_mask": fine_mask,
            "presence_logits": presence_logits,
        }


class BrightnessReEstimator(nn.Module):
    def __init__(self, feature_channels: int = 128):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_gamma = nn.Sequential(
            nn.Linear(32 + feature_channels, feature_channels),
            nn.Tanh(),
        )
        self.to_beta = nn.Sequential(
            nn.Linear(32 + feature_channels, feature_channels),
            nn.Tanh(),
        )
        nn.init.zeros_(self.to_gamma[0].weight)
        nn.init.zeros_(self.to_gamma[0].bias)
        nn.init.zeros_(self.to_beta[0].weight)
        nn.init.zeros_(self.to_beta[0].bias)

    def forward(
        self,
        target_01: torch.Tensor,
        prior_feature: torch.Tensor,
        fine_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fine_mask = resize_mask(fine_mask, target_01.shape[-2:])
        query_mask = resize_mask(query_mask, target_01.shape[-2:])
        context_input = torch.cat([target_01, fine_mask, query_mask, target_01 * query_mask], dim=1)
        context_embedding = self.context_encoder(context_input).flatten(1)
        prior_embedding = F.adaptive_avg_pool2d(prior_feature, 1).flatten(1)
        embedding = torch.cat([context_embedding, prior_embedding], dim=1)

        gamma = 1 + 0.1 * self.to_gamma(embedding).unsqueeze(-1).unsqueeze(-1)
        beta = 0.1 * self.to_beta(embedding).unsqueeze(-1).unsqueeze(-1)
        adjusted = gamma * prior_feature + beta
        return adjusted, {"gamma": gamma, "beta": beta}


class HFDAGatedInjectionUnit(nn.Module):
    def __init__(self, base_channels: int = 512, prior_channels: int = 128):
        super().__init__()
        self.prior_proj = nn.Sequential(
            nn.Conv2d(prior_channels, base_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
        )
        nn.init.zeros_(self.fuse[2].weight)
        if self.fuse[2].bias is not None:
            nn.init.zeros_(self.fuse[2].bias)

    def forward(
        self,
        base_feature: torch.Tensor,
        prior_feature: torch.Tensor,
        fine_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fine_mask = resize_mask(fine_mask, base_feature.shape[-2:])
        projected_prior = self.prior_proj(prior_feature)
        candidate = self.fuse(torch.cat([base_feature, projected_prior], dim=1))
        fused = base_feature + candidate * fine_mask
        return fused, fine_mask
