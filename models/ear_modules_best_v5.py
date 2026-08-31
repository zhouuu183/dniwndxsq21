from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CtrlHair.external_code.face_parsing.my_parsing_util import FaceParsing_tensor
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample

RAW_LEFT_EAR = 7
RAW_RIGHT_EAR = 8
RAW_EARRING = 9
RAW_FACE_SURFACE_LABELS = (1, 10)
RAW_SKIN_SURFACE_LABELS = (1, 7, 8, 10)
RAW_EAR_SURFACE_LABELS = (7, 8)
RAW_HAIR = 17
RAW_HAT = 18
RAW_DETAIL_LABELS = (2, 3, 4, 5, 6, 11, 12, 13)


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
    shifted = torch.zeros_like(mask)

    h_slice_src = slice(0, mask.size(-2) - max(0, down))
    h_slice_dst = slice(max(0, down), mask.size(-2))
    if down < 0:
        h_slice_src = slice(-down, mask.size(-2))
        h_slice_dst = slice(0, mask.size(-2) + down)

    w_slice_src = slice(0, mask.size(-1) - max(0, right))
    w_slice_dst = slice(max(0, right), mask.size(-1))
    if right < 0:
        w_slice_src = slice(-right, mask.size(-1))
        w_slice_dst = slice(0, mask.size(-1) + right)

    shifted[..., h_slice_dst, w_slice_dst] = mask[..., h_slice_src, w_slice_src]
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


def parsing_label_mask(parsing: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    parsing = ensure_mask_4d(parsing).long()
    mask = torch.zeros_like(parsing, dtype=torch.bool)
    for label in labels:
        mask |= parsing == label
    return mask.float()


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    gray = rgb_to_gray(image) if image.size(1) != 1 else image
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=image.device, dtype=image.dtype)
    kernel_y = kernel_x.t()
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


def _masked_threshold_candidate(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_ratio: float = 0.45,
    std_ratio: float = 1.0,
    floor: float = 0.0,
) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    value = value * mask
    flat_value = value.flatten(2)
    flat_mask = mask.flatten(2)
    denom = flat_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    mean = (flat_value.sum(dim=2, keepdim=True) / denom).view(-1, 1, 1, 1)
    centered = (flat_value - mean.flatten(1).unsqueeze(-1)).pow(2) * flat_mask
    std = torch.sqrt(centered.sum(dim=2, keepdim=True) / denom).view(-1, 1, 1, 1)
    peak = flat_value.amax(dim=2, keepdim=True).view(-1, 1, 1, 1)
    threshold = torch.maximum(max_ratio * peak, mean + std_ratio * std)
    if floor > 0:
        threshold = torch.maximum(threshold, torch.full_like(threshold, floor))
    return (value >= threshold).float() * mask


def build_weak_earring_mask(
    source_01: torch.Tensor,
    query_mask: torch.Tensor,
    parser_mask: torch.Tensor | None = None,
    source_hair_block_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    query_mask = ensure_mask_4d(query_mask).float()
    parser_mask = torch.zeros_like(query_mask) if parser_mask is None else resize_mask(parser_mask, query_mask.shape[-2:])
    source_hair_block_mask = (
        torch.zeros_like(query_mask)
        if source_hair_block_mask is None
        else resize_mask(source_hair_block_mask, query_mask.shape[-2:])
    )

    high_energy = high_pass_filter(source_01).abs().mean(dim=1, keepdim=True) * query_mask
    edge_energy = sobel_magnitude(source_01) * query_mask
    gray = rgb_to_gray(source_01)
    local_gray = low_pass_filter(gray, kernel_size=9, sigma=2.0)
    local_contrast = (gray - local_gray).abs() * query_mask
    bright_spike = F.relu(gray - local_gray) * query_mask
    dark_edge = edge_energy * F.relu(local_gray - gray) * query_mask
    color_spread = (source_01.amax(dim=1, keepdim=True) - source_01.amin(dim=1, keepdim=True)) * query_mask

    high_candidate = _masked_threshold_candidate(high_energy, query_mask, max_ratio=0.40, std_ratio=0.75)
    edge_candidate = _masked_threshold_candidate(edge_energy, query_mask, max_ratio=0.42, std_ratio=0.85)
    contrast_candidate = _masked_threshold_candidate(local_contrast, query_mask, max_ratio=0.40, std_ratio=0.90)
    bright_candidate = _masked_threshold_candidate(bright_spike, query_mask, max_ratio=0.45, std_ratio=1.10)
    dark_candidate = _masked_threshold_candidate(dark_edge, query_mask, max_ratio=0.38, std_ratio=0.75)
    color_candidate = _masked_threshold_candidate(color_spread, query_mask, max_ratio=0.42, std_ratio=0.90)

    weak_mask = (
        high_candidate * torch.clamp(edge_candidate + contrast_candidate, 0, 1)
        + bright_candidate
        + dark_candidate
        + color_candidate * torch.clamp(edge_candidate + contrast_candidate, 0, 1)
        + parser_mask * query_mask
    ).clamp(0, 1)
    weak_mask = weak_mask * (1.0 - 0.20 * source_hair_block_mask).clamp(0, 1)
    weak_mask = dilate_mask(weak_mask, 3) * query_mask

    return {
        "weak_earring_mask": weak_mask.clamp(0, 1),
        "weak_high_energy": high_energy,
        "weak_edge_energy": edge_energy,
        "weak_local_contrast": local_contrast,
        "weak_bright_spike": bright_spike,
        "weak_dark_edge": dark_edge,
        "weak_color_spread": color_spread,
    }


def build_weak_earring_masks(
    source_01: torch.Tensor,
    visible_ear_roi: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    parser_mask: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    source_hair_block_mask: torch.Tensor | None = None,
    source_parsing: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    source_01 = normalized_to_01(source_01)
    image_size = tuple(source_01.shape[-2:])
    if visible_ear_roi is None:
        if query_mask is None:
            raise ValueError("visible_ear_roi or query_mask is required to build weak earring masks.")
        visible_ear_roi = query_mask
    visible_ear_roi = resize_mask(visible_ear_roi, image_size)
    query_mask = visible_ear_roi if query_mask is None else resize_mask(query_mask, image_size)
    parser_mask = torch.zeros_like(query_mask) if parser_mask is None else resize_mask(parser_mask, image_size)

    if source_parsing is not None:
        parsing = ensure_mask_4d(source_parsing).float()
        if tuple(parsing.shape[-2:]) != image_size:
            parsing = F.interpolate(parsing, size=image_size, mode="nearest")
        parser_mask = torch.clamp(parser_mask + (parsing.long() == RAW_EARRING).float(), 0, 1)
        source_ear_mask = parsing_label_mask(parsing, RAW_EAR_SURFACE_LABELS)
        source_lobe_roi = torch.clamp(
            dilate_mask(source_ear_mask, 13)
            + dilate_mask(shift_mask(source_ear_mask, down=8), 15)
            + dilate_mask(shift_mask(source_ear_mask, down=18), 11)
            + dilate_mask(parser_mask, 11),
            0,
            1,
        )
    else:
        source_lobe_roi = visible_ear_roi

    source_hair_mask = torch.zeros_like(query_mask) if source_hair_mask is None else resize_mask(source_hair_mask, image_size)
    source_hair_block_mask = (
        torch.zeros_like(query_mask)
        if source_hair_block_mask is None
        else resize_mask(source_hair_block_mask, image_size)
    )

    source_earring_clean_mask = parser_mask * source_lobe_roi
    search_mask = torch.clamp(
        query_mask
        + 0.35 * visible_ear_roi
        + 0.90 * source_lobe_roi
        + dilate_mask(source_earring_clean_mask, 7),
        0,
        1,
    )
    hair_attenuation = (1.0 - 0.10 * source_hair_mask - 0.25 * source_hair_block_mask).clamp(0, 1)
    search_mask = search_mask * hair_attenuation

    gray = rgb_to_gray(source_01)
    local_gray = low_pass_filter(gray, kernel_size=9, sigma=2.0)
    high_energy = high_pass_filter(source_01).abs().mean(dim=1, keepdim=True) * search_mask
    edge_energy = sobel_magnitude(source_01) * search_mask
    local_contrast = (gray - local_gray).abs() * search_mask
    bright_spike = F.relu(gray - local_gray) * search_mask
    dark_spike = F.relu(local_gray - gray) * search_mask
    dark_edge = edge_energy * dark_spike
    color_spread = (source_01.amax(dim=1, keepdim=True) - source_01.amin(dim=1, keepdim=True)) * search_mask

    high_candidate = _masked_threshold_candidate(high_energy, search_mask, max_ratio=0.38, std_ratio=0.80, floor=0.006)
    edge_candidate = _masked_threshold_candidate(edge_energy, search_mask, max_ratio=0.38, std_ratio=0.75, floor=0.006)
    contrast_candidate = _masked_threshold_candidate(local_contrast, search_mask, max_ratio=0.38, std_ratio=0.80, floor=0.006)
    bright_candidate = _masked_threshold_candidate(bright_spike, search_mask, max_ratio=0.40, std_ratio=0.85, floor=0.006)
    dark_candidate = _masked_threshold_candidate(dark_edge, search_mask, max_ratio=0.36, std_ratio=0.70, floor=0.002)
    color_candidate = _masked_threshold_candidate(color_spread, search_mask, max_ratio=0.42, std_ratio=0.80, floor=0.030)

    structure_gate = torch.clamp(edge_candidate + contrast_candidate, 0, 1)
    texture_core = high_candidate * structure_gate
    highlight_core = bright_candidate * torch.clamp(high_candidate + edge_candidate + color_candidate, 0, 1)
    dark_core = dark_candidate * structure_gate
    chroma_core = color_candidate * torch.clamp(edge_candidate + contrast_candidate + bright_candidate, 0, 1)
    parser_core = dilate_mask(source_earring_clean_mask, 3) * search_mask

    earring_candidate_mask = torch.clamp(
        texture_core + highlight_core + dark_core + chroma_core + parser_core,
        0,
        1,
    )
    earring_confident_mask = dilate_mask(earring_candidate_mask, 5) * search_mask
    earring_highlight_mask = torch.clamp(highlight_core + color_candidate * high_candidate, 0, 1)
    earring_highlight_mask = dilate_mask(earring_highlight_mask, 3) * earring_confident_mask

    return {
        "weak_earring_mask": earring_confident_mask.clamp(0, 1),
        "source_earring_clean_mask": source_earring_clean_mask.clamp(0, 1),
        "earring_confident_mask": earring_confident_mask.clamp(0, 1),
        "earring_highlight_mask": earring_highlight_mask.clamp(0, 1),
        "earring_candidate_mask": earring_candidate_mask.clamp(0, 1),
        "earring_search_mask": search_mask.clamp(0, 1),
        "source_lobe_search_mask": source_lobe_roi.clamp(0, 1),
        "weak_high_energy": high_energy,
        "weak_edge_energy": edge_energy,
        "weak_local_contrast": local_contrast,
        "weak_bright_spike": bright_spike,
        "weak_dark_edge": dark_edge,
        "weak_color_spread": color_spread,
    }


def enhance_query_with_earring_recall(
    query_mask: torch.Tensor,
    source_earring_mask: torch.Tensor | None,
    source_hair_block_mask: torch.Tensor | None,
    weak_masks: dict[str, torch.Tensor],
    *,
    recall_dilate: int = 7,
    downward_shift: int = 18,
    lower_lobe_weight: float = 0.20,
    candidate_boost: float = 0.90,
    block_protect: float = 0.85,
) -> dict[str, torch.Tensor]:
    query_mask = ensure_mask_4d(query_mask).float()
    source_earring_mask = (
        torch.zeros_like(query_mask)
        if source_earring_mask is None
        else resize_mask(source_earring_mask, query_mask.shape[-2:])
    )
    source_hair_block_mask = (
        torch.zeros_like(query_mask)
        if source_hair_block_mask is None
        else resize_mask(source_hair_block_mask, query_mask.shape[-2:])
    )

    def weak_mask(name: str) -> torch.Tensor:
        value = weak_masks.get(name)
        if value is None:
            return torch.zeros_like(query_mask)
        return resize_mask(value, query_mask.shape[-2:])

    candidate_mask = weak_mask("earring_candidate_mask")
    clean_parser_mask = weak_mask("source_earring_clean_mask")
    search_mask = weak_mask("earring_search_mask")
    lobe_search_mask = weak_mask("source_lobe_search_mask")

    recall_seed = torch.clamp(source_earring_mask + clean_parser_mask + candidate_mask, 0, 1)
    object_recall_mask = dilate_mask(recall_seed, max(3, recall_dilate))
    query_recall_mask = object_recall_mask
    if downward_shift > 0:
        query_recall_mask = torch.clamp(query_recall_mask + shift_mask(query_recall_mask, down=downward_shift), 0, 1)

    has_recall = (recall_seed.flatten(1).amax(dim=1) > 0).float().view(-1, 1, 1, 1)
    lobe_support = lobe_search_mask * torch.clamp(search_mask + query_recall_mask, 0, 1) * has_recall

    enhanced_query_mask = torch.clamp(
        query_mask
        + max(0.0, float(candidate_boost)) * query_recall_mask
        + max(0.0, float(lower_lobe_weight)) * lobe_support,
        0,
        1,
    )

    protect_mask = object_recall_mask
    if downward_shift > 1:
        protect_mask = torch.clamp(protect_mask + shift_mask(protect_mask, down=max(1, downward_shift // 2)), 0, 1)
    protected_block_mask = source_hair_block_mask * (1 - max(0.0, min(1.0, float(block_protect))) * protect_mask).clamp(0, 1)

    return {
        "query_mask_before_recall": query_mask,
        "query_mask": enhanced_query_mask,
        "source_hair_block_mask": protected_block_mask,
        "source_earring_mask": torch.clamp(source_earring_mask + object_recall_mask, 0, 1),
        "earring_query_recall_mask": query_recall_mask,
        "earring_object_recall_mask": object_recall_mask,
        "online_earring_candidate_mask": candidate_mask,
        "online_earring_search_mask": search_mask,
        "source_lobe_search_mask": lobe_search_mask,
        "earring_recall_lobe_support": lobe_support,
        "earring_recall_block_protect_mask": protect_mask,
    }


def build_earring_highlight_mask(
    reference_01: torch.Tensor,
    earring_mask: torch.Tensor,
    query_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    earring_mask = ensure_mask_4d(earring_mask).float()
    query_mask = earring_mask if query_mask is None else resize_mask(query_mask, earring_mask.shape[-2:])
    highlight_support = (earring_mask * query_mask).clamp(0, 1)
    gray = rgb_to_gray(reference_01)
    local_gray = low_pass_filter(gray, kernel_size=9, sigma=2.0)
    bright_spike = F.relu(gray - local_gray) * highlight_support
    high_energy = high_pass_filter(reference_01).abs().mean(dim=1, keepdim=True) * highlight_support
    color_spread = (
        reference_01.amax(dim=1, keepdim=True) - reference_01.amin(dim=1, keepdim=True)
    ) * highlight_support

    bright_candidate = _masked_threshold_candidate(bright_spike, highlight_support, max_ratio=0.48, std_ratio=1.15)
    high_candidate = _masked_threshold_candidate(high_energy, highlight_support, max_ratio=0.45, std_ratio=0.90)
    color_candidate = _masked_threshold_candidate(color_spread, highlight_support, max_ratio=0.50, std_ratio=1.00)
    highlight = bright_candidate * torch.clamp(high_candidate + color_candidate + earring_mask, 0, 1)
    return (dilate_mask(highlight, 3) * highlight_support).clamp(0, 1)


def select_reference_earring_mask(
    clean_source_earring_mask: torch.Tensor,
    candidate_source_earring_mask: torch.Tensor,
    left_roi: torch.Tensor,
    right_roi: torch.Tensor,
    *,
    support_dilate: int = 13,
    min_clean_area: float = 8.0,
    min_candidate_area: float = 16.0,
    max_candidate_density: float = 0.28,
    candidate_balance_ratio: float = 0.70,
    missing_side_clean_ratio: float = 0.85,
) -> torch.Tensor:
    clean_source_earring_mask = ensure_mask_4d(clean_source_earring_mask).float()
    candidate_source_earring_mask = ensure_mask_4d(candidate_source_earring_mask).float()
    left_roi = ensure_mask_4d(left_roi).float()
    right_roi = ensure_mask_4d(right_roi).float()

    height, width = clean_source_earring_mask.shape[-2:]
    area_scale = float(height * width) / float(256 * 256)
    min_clean_area = float(min_clean_area) * area_scale
    min_candidate_area = float(min_candidate_area) * area_scale

    x_coords = torch.linspace(
        0,
        1,
        width,
        device=clean_source_earring_mask.device,
        dtype=clean_source_earring_mask.dtype,
    ).view(1, 1, 1, width)
    left_roi = left_roi * (x_coords <= 0.5).float()
    right_roi = right_roi * (x_coords > 0.5).float()

    def side_parts(side_roi: torch.Tensor) -> dict[str, torch.Tensor]:
        clean_side = clean_source_earring_mask * side_roi
        candidate_side = candidate_source_earring_mask * side_roi
        clean_support = dilate_mask(clean_side, support_dilate) * side_roi
        parser_guided = torch.clamp(clean_side + candidate_side * clean_support, 0, 1)
        fallback = torch.clamp(clean_side + candidate_side, 0, 1)
        clean_area = clean_side.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
        candidate_area = candidate_side.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
        roi_area = side_roi.flatten(1).sum(dim=1).view(-1, 1, 1, 1).clamp_min(1.0)
        candidate_density = candidate_area / roi_area
        return {
            "parser_guided": parser_guided,
            "fallback": fallback,
            "clean_area": clean_area,
            "candidate_area": candidate_area,
            "candidate_density": candidate_density,
        }

    left = side_parts(left_roi)
    right = side_parts(right_roi)
    left_clean_present = left["clean_area"] >= min_clean_area
    right_clean_present = right["clean_area"] >= min_clean_area
    any_clean_present = left_clean_present | right_clean_present

    left_candidate_valid = (
        (left["candidate_area"] >= min_candidate_area)
        & (left["candidate_density"] <= max_candidate_density)
    )
    right_candidate_valid = (
        (right["candidate_area"] >= min_candidate_area)
        & (right["candidate_density"] <= max_candidate_density)
    )

    left_valid_area = torch.where(left_candidate_valid, left["candidate_area"], torch.zeros_like(left["candidate_area"]))
    right_valid_area = torch.where(right_candidate_valid, right["candidate_area"], torch.zeros_like(right["candidate_area"]))
    max_candidate_area = torch.maximum(left_valid_area, right_valid_area).clamp_min(1.0)
    left_candidate_balanced = left["candidate_area"] >= candidate_balance_ratio * max_candidate_area
    right_candidate_balanced = right["candidate_area"] >= candidate_balance_ratio * max_candidate_area
    max_clean_area = torch.maximum(left["clean_area"], right["clean_area"]).clamp_min(1.0)
    left_candidate_matches_clean = left["candidate_area"] >= missing_side_clean_ratio * max_clean_area
    right_candidate_matches_clean = right["candidate_area"] >= missing_side_clean_ratio * max_clean_area

    left_keep_candidate = left_candidate_valid & torch.where(
        any_clean_present,
        left_candidate_matches_clean,
        left_candidate_balanced,
    )
    right_keep_candidate = right_candidate_valid & torch.where(
        any_clean_present,
        right_candidate_matches_clean,
        right_candidate_balanced,
    )

    left_selected = torch.where(
        left_clean_present,
        left["parser_guided"],
        torch.where(left_keep_candidate, left["fallback"], torch.zeros_like(left["fallback"])),
    )
    right_selected = torch.where(
        right_clean_present,
        right["parser_guided"],
        torch.where(right_keep_candidate, right["fallback"], torch.zeros_like(right["fallback"])),
    )
    return torch.clamp(left_selected + right_selected, 0, 1)


def _weighted_centroid(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = ensure_mask_4d(mask).float()
    batch, _, height, width = mask.shape
    y_coords = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    x_coords = torch.arange(width, device=mask.device, dtype=mask.dtype).view(1, 1, 1, width)
    denom = mask.sum(dim=(-2, -1), keepdim=True)
    valid = denom.view(batch) > 1.0
    safe_denom = denom.clamp_min(1.0)
    y_center = (mask * y_coords).sum(dim=(-2, -1), keepdim=True) / safe_denom
    x_center = (mask * x_coords).sum(dim=(-2, -1), keepdim=True) / safe_denom
    return y_center.view(batch), x_center.view(batch), valid


def shift_tensor_per_batch(
    tensor: torch.Tensor,
    shift_y: torch.Tensor,
    shift_x: torch.Tensor | None = None,
) -> torch.Tensor:
    shift_x = torch.zeros_like(shift_y) if shift_x is None else shift_x
    shifted = torch.zeros_like(tensor)
    height, width = tensor.shape[-2:]
    for idx in range(tensor.size(0)):
        sy = int(shift_y[idx].item())
        sx = int(shift_x[idx].item())

        src_y0 = max(0, -sy)
        src_y1 = min(height, height - sy)
        dst_y0 = max(0, sy)
        dst_y1 = min(height, height + sy)
        src_x0 = max(0, -sx)
        src_x1 = min(width, width - sx)
        dst_x0 = max(0, sx)
        dst_x1 = min(width, width + sx)
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            continue
        shifted[idx, :, dst_y0:dst_y1, dst_x0:dst_x1] = tensor[idx, :, src_y0:src_y1, src_x0:src_x1]
    return shifted


def align_earring_reference_to_target(
    source_01: torch.Tensor,
    source_earring_mask: torch.Tensor,
    source_left_ear_mask: torch.Tensor,
    source_right_ear_mask: torch.Tensor,
    target_left_ear_mask: torch.Tensor,
    target_right_ear_mask: torch.Tensor,
    left_roi: torch.Tensor,
    right_roi: torch.Tensor,
    *,
    max_vertical_shift: int = 12,
    max_horizontal_shift: int = 6,
    reference_base: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    source_01 = normalized_to_01(source_01)
    source_earring_mask = ensure_mask_4d(source_earring_mask).float()
    left_roi = ensure_mask_4d(left_roi).float()
    right_roi = ensure_mask_4d(right_roi).float()
    source_left_ear_mask = ensure_mask_4d(source_left_ear_mask).float()
    source_right_ear_mask = ensure_mask_4d(source_right_ear_mask).float()
    target_left_ear_mask = ensure_mask_4d(target_left_ear_mask).float()
    target_right_ear_mask = ensure_mask_4d(target_right_ear_mask).float()

    left_ring = source_earring_mask * left_roi
    right_ring = source_earring_mask * right_roi
    left_source_anchor = (source_left_ear_mask + left_ring).clamp(0, 1) * left_roi
    right_source_anchor = (source_right_ear_mask + right_ring).clamp(0, 1) * right_roi
    left_target_anchor = target_left_ear_mask * left_roi
    right_target_anchor = target_right_ear_mask * right_roi

    left_src_y, left_src_x, left_src_valid = _weighted_centroid(left_source_anchor)
    left_tgt_y, left_tgt_x, left_tgt_valid = _weighted_centroid(left_target_anchor)
    right_src_y, right_src_x, right_src_valid = _weighted_centroid(right_source_anchor)
    right_tgt_y, right_tgt_x, right_tgt_valid = _weighted_centroid(right_target_anchor)

    left_valid = left_src_valid & left_tgt_valid & (left_ring.flatten(1).sum(dim=1) > 1.0)
    right_valid = right_src_valid & right_tgt_valid & (right_ring.flatten(1).sum(dim=1) > 1.0)
    left_shift_y = torch.where(left_valid, torch.round(left_tgt_y - left_src_y), torch.zeros_like(left_src_y))
    left_shift_x = torch.where(left_valid, torch.round(left_tgt_x - left_src_x), torch.zeros_like(left_src_x))
    right_shift_y = torch.where(right_valid, torch.round(right_tgt_y - right_src_y), torch.zeros_like(right_src_y))
    right_shift_x = torch.where(right_valid, torch.round(right_tgt_x - right_src_x), torch.zeros_like(right_src_x))

    left_shift_y = left_shift_y.clamp(-max_vertical_shift, max_vertical_shift)
    right_shift_y = right_shift_y.clamp(-max_vertical_shift, max_vertical_shift)
    left_shift_x = left_shift_x.clamp(-max_horizontal_shift, max_horizontal_shift)
    right_shift_x = right_shift_x.clamp(-max_horizontal_shift, max_horizontal_shift)

    shifted_left_mask = shift_tensor_per_batch(left_ring, left_shift_y, left_shift_x)
    shifted_right_mask = shift_tensor_per_batch(right_ring, right_shift_y, right_shift_x)
    shifted_left_rgb = shift_tensor_per_batch(source_01 * left_ring, left_shift_y, left_shift_x)
    shifted_right_rgb = shift_tensor_per_batch(source_01 * right_ring, right_shift_y, right_shift_x)

    original_ring = (left_ring + right_ring).clamp(0, 1)
    aligned_mask = (shifted_left_mask + shifted_right_mask).clamp(0, 1)
    earring_rgb = shifted_left_rgb + shifted_right_rgb
    aligned_support = dilate_mask(aligned_mask, 3)
    if reference_base is None:
        source_cleanup_mask = dilate_mask(original_ring, 3) * (1.0 - aligned_support).clamp(0, 1)
        source_cleanup_fill = low_pass_filter(source_01, kernel_size=9, sigma=2.0)
        clean_reference_base = (
            source_01 * (1.0 - source_cleanup_mask)
            + source_cleanup_fill * source_cleanup_mask
        ).clamp(0, 1)
    else:
        clean_reference_base = normalized_to_01(reference_base)
        if tuple(clean_reference_base.shape[-2:]) != tuple(source_01.shape[-2:]):
            clean_reference_base = F.interpolate(
                clean_reference_base,
                size=source_01.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0, 1)
        source_cleanup_mask = torch.zeros_like(original_ring)
    aligned_source = (clean_reference_base * (1.0 - aligned_mask) + earring_rgb).clamp(0, 1)

    return {
        "earring_reference": aligned_source,
        "earring_confident_mask": aligned_mask,
        "source_earring_original_mask": original_ring,
        "source_earring_cleanup_mask": source_cleanup_mask,
        "source_left_earring_aligned_mask": shifted_left_mask,
        "source_right_earring_aligned_mask": shifted_right_mask,
        "left_earring_shift_y": left_shift_y.view(-1, 1),
        "left_earring_shift_x": left_shift_x.view(-1, 1),
        "right_earring_shift_y": right_shift_y.view(-1, 1),
        "right_earring_shift_x": right_shift_x.view(-1, 1),
    }


def combine_cleanup_masks(cleanup_masks: dict[str, torch.Tensor]) -> torch.Tensor | None:
    masks = [ensure_mask_4d(mask).float() for mask in cleanup_masks.values() if torch.is_tensor(mask)]
    if not masks:
        return None
    return torch.stack(masks, dim=0).amax(dim=0).clamp(0, 1)


def build_revealed_skin_mask(
    cleanup_masks: dict[str, torch.Tensor],
    target_parsing: torch.Tensor,
    source_parsing: torch.Tensor | None = None,
    target_hair_mask: torch.Tensor | None = None,
    source_hair_mask: torch.Tensor | None = None,
    earring_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    cleanup_mask = combine_cleanup_masks(cleanup_masks)
    if cleanup_mask is None:
        base = parsing_label_mask(target_parsing, RAW_FACE_SURFACE_LABELS)
        cleanup_mask = torch.zeros_like(base)

    face_surface = parsing_label_mask(target_parsing, RAW_FACE_SURFACE_LABELS)
    remove_face = cleanup_masks.get("M_remove_face", torch.zeros_like(cleanup_mask))
    remove_halo = cleanup_masks.get("M_remove_halo", torch.zeros_like(cleanup_mask))
    remove_face = resize_mask(remove_face, cleanup_mask.shape[-2:])
    remove_halo = resize_mask(remove_halo, cleanup_mask.shape[-2:])
    cleanup_inner_edge = (cleanup_mask - erode_mask(cleanup_mask, 5)).clamp(0, 1)

    target_hair_mask = torch.zeros_like(cleanup_mask) if target_hair_mask is None else resize_mask(target_hair_mask, cleanup_mask.shape[-2:])
    source_hair_mask = torch.zeros_like(cleanup_mask) if source_hair_mask is None else resize_mask(source_hair_mask, cleanup_mask.shape[-2:])
    earring_mask = torch.zeros_like(cleanup_mask) if earring_mask is None else resize_mask(earring_mask, cleanup_mask.shape[-2:])

    revealed = (
        remove_face
        + 0.45 * cleanup_inner_edge * dilate_mask(remove_face + remove_halo, 5)
        + 0.22 * remove_halo * dilate_mask(remove_face, 5)
    ).clamp(0, 1)
    revealed = (
        revealed
        * face_surface
        * (1.0 - target_hair_mask).clamp(0, 1)
        * (1.0 - 0.80 * earring_mask).clamp(0, 1)
    ).clamp(0, 1)

    source_face_surface = (
        parsing_label_mask(source_parsing, RAW_FACE_SURFACE_LABELS)
        if source_parsing is not None
        else torch.zeros_like(revealed)
    )
    source_skin_valid = (source_face_surface * (1.0 - source_hair_mask)).clamp(0, 1)

    return {
        "cleanup_mask": cleanup_mask,
        "cleanup_inner_edge": cleanup_inner_edge,
        "revealed_skin_mask": revealed,
        "source_skin_valid_mask": source_skin_valid,
        "target_face_surface_mask": face_surface,
    }


def build_raw_face_masks(parsing: torch.Tensor) -> dict[str, torch.Tensor]:
    parsing = ensure_mask_4d(parsing).long()
    return {
        "left_ear": (parsing == RAW_LEFT_EAR).float(),
        "right_ear": (parsing == RAW_RIGHT_EAR).float(),
        "earring": (parsing == RAW_EARRING).float(),
        "face_surface": parsing_label_mask(parsing, RAW_FACE_SURFACE_LABELS),
        "skin_surface": parsing_label_mask(parsing, RAW_SKIN_SURFACE_LABELS),
        "ear_surface": parsing_label_mask(parsing, RAW_EAR_SURFACE_LABELS),
        "detail": torch.stack([(parsing == label).float() for label in RAW_DETAIL_LABELS], dim=0).amax(dim=0),
        "hair": (parsing == RAW_HAIR).float(),
        "hat": (parsing == RAW_HAT).float(),
    }


class FaceParsingHelperV5:
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


class HairMaskExtractorV5(nn.Module):
    def __init__(self, device: str = "cuda", dilate_erosion: int = 5):
        super().__init__()
        from utils.image_utils import DilateErosion

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
        earring_align_max_shift: int = 12,
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
        self.earring_align_max_shift = earring_align_max_shift

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
        visible_ear_roi = torch.clamp(left_visible_roi + right_visible_roi, 0, 1)

        source_hair_block_mask = (
            visible_ear_roi
            * source_hair_context
            * (1 - target_hair_context)
            * (1 - source_earring_keep)
            * (1 - hat_mask)
        ).clamp(0, 1)
        block_attenuation = (1 - self.source_hair_block_strength * source_hair_block_mask).clamp(0, 1)
        query_seed = torch.clamp(hair_change + 0.25 * ring_hint, 0, 1)
        query_mask = visible_ear_roi * query_seed * block_attenuation
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

        source_left_ring = source_left_ring * left_visible_roi
        source_right_ring = source_right_ring * right_visible_roi
        source_earring_mask = source_masks["earring"] * visible_ear_roi

        return {
            "left_ear_roi": left_roi,
            "right_ear_roi": right_roi,
            "ear_roi": ear_roi,
            "visible_ear_roi": visible_ear_roi,
            "query_mask": query_mask,
            "source_hair_block_mask": source_hair_block_mask,
            "source_hair_mask": source_hair_mask,
            "target_hair_mask": target_hair_mask,
            "source_earring_mask": source_earring_mask,
            "target_earring_mask": target_masks["earring"],
            "source_left_earring_mask": source_left_ring,
            "source_right_earring_mask": source_right_ring,
            "source_left_ear_mask": source_masks["left_ear"],
            "source_right_ear_mask": source_masks["right_ear"],
            "target_left_ear_mask": target_masks["left_ear"],
            "target_right_ear_mask": target_masks["right_ear"],
            "source_face_surface_mask": source_masks["face_surface"],
            "target_face_surface_mask": target_masks["face_surface"],
            "source_skin_surface_mask": source_masks["skin_surface"],
            "target_skin_surface_mask": target_masks["skin_surface"],
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
    def __init__(self, hidden_channels: int = 32, init_bias: float = -4.0, fine_support_dilate: int = 3):
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
        nn.init.zeros_(self.prior_proj[0].weight)
        if self.prior_proj[0].bias is not None:
            nn.init.zeros_(self.prior_proj[0].bias)
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
        delta = self.fuse(torch.cat([base_feature, projected_prior], dim=1))
        fused = base_feature + delta * fine_mask
        return fused, fine_mask
