from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gray(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    g = gray(image)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(g, kernel_x, padding=1)
    grad_y = F.conv2d(g, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


def laplacian_edges(image: torch.Tensor) -> torch.Tensor:
    g = gray(image)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    return F.conv2d(g, kernel, padding=1).abs()


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    return (value * mask).sum() / mask.sum().clamp(min=1.0)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target).abs(), mask)


def masked_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target).pow(2), mask)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.float()
    kernel = radius * 2 + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return 1.0 - _dilate(1.0 - mask.float(), radius)


def binary_mask(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (mask.float() >= threshold).float()


def boundary_map(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    mask = binary_mask(mask)
    return (_dilate(mask, radius) - _erode(mask, radius)).clamp(0, 1)


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred_bin = binary_mask(pred, threshold)
    target_bin = binary_mask(target, threshold)
    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = ((pred_bin + target_bin) > 0).float().sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (intersection / union).mean()


def boundary_fscore(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    boundary_radius: int = 1,
    tolerance: int = 2,
) -> torch.Tensor:
    pred_edge = boundary_map(pred, radius=boundary_radius)
    target_edge = boundary_map(target, radius=boundary_radius)
    pred_hit = pred_edge * _dilate(target_edge, tolerance)
    target_hit = target_edge * _dilate(pred_edge, tolerance)
    precision = pred_hit.sum(dim=(1, 2, 3)) / pred_edge.sum(dim=(1, 2, 3)).clamp(min=1.0)
    recall = target_hit.sum(dim=(1, 2, 3)) / target_edge.sum(dim=(1, 2, 3)).clamp(min=1.0)
    fscore = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
    return fscore.mean()


def _sample_points(mask: torch.Tensor, max_points: int = 2048) -> torch.Tensor:
    points = torch.nonzero(mask > 0.5, as_tuple=False)
    if points.numel() == 0:
        return points.float()
    points = points[:, -2:].float()
    if points.shape[0] > max_points:
        step = max(1, math.ceil(points.shape[0] / max_points))
        points = points[::step]
    return points


def _point_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
    max_points: int = 2048,
) -> torch.Tensor:
    batch_scores = []
    for pred_item, target_item in zip(pred, target):
        pred_points = _sample_points(pred_item, max_points=max_points)
        target_points = _sample_points(target_item, max_points=max_points)
        if pred_points.numel() == 0 or target_points.numel() == 0:
            batch_scores.append(pred_item.new_tensor(1.0))
            continue
        dist = torch.cdist(pred_points, target_points, p=2)
        forward = dist.min(dim=1).values
        backward = dist.min(dim=0).values
        if reduction == "chamfer":
            score = 0.5 * (forward.mean() + backward.mean())
        elif reduction == "hausdorff":
            score = torch.maximum(forward.max(), backward.max())
        else:
            raise ValueError(f"Unsupported reduction={reduction!r}")
        h, w = pred_item.shape[-2:]
        diag = math.sqrt(float(h * h + w * w))
        batch_scores.append(score / max(diag, 1.0))
    return torch.stack(batch_scores).mean()


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor, max_points: int = 2048) -> torch.Tensor:
    return _point_distance(boundary_map(pred), boundary_map(target), "chamfer", max_points=max_points)


def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor, max_points: int = 2048) -> torch.Tensor:
    return _point_distance(boundary_map(pred), boundary_map(target), "hausdorff", max_points=max_points)


def high_frequency_spectrum_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    low_freq_radius: int = 8,
) -> torch.Tensor:
    pred_gray = gray(pred)
    target_gray = gray(target)
    if mask is not None:
        pred_gray = pred_gray * mask
        target_gray = target_gray * mask

    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred_gray, norm="ortho"), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target_gray, norm="ortho"), dim=(-2, -1))
    pred_mag = pred_fft.abs()
    target_mag = target_fft.abs()

    h, w = pred_mag.shape[-2:]
    cy, cx = h // 2, w // 2
    band_mask = torch.ones((1, 1, h, w), device=pred.device, dtype=pred.dtype)
    y0, y1 = max(0, cy - low_freq_radius), min(h, cy + low_freq_radius + 1)
    x0, x1 = max(0, cx - low_freq_radius), min(w, cx + low_freq_radius + 1)
    band_mask[..., y0:y1, x0:x1] = 0.0

    pred_vec = (pred_mag * band_mask).flatten(1)
    target_vec = (target_mag * band_mask).flatten(1)
    pred_vec = F.normalize(pred_vec, dim=1, eps=1e-6)
    target_vec = F.normalize(target_vec, dim=1, eps=1e-6)
    return (pred_vec * target_vec).sum(dim=1).mean()


def cosine_feature_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_vec = F.normalize(pred.flatten(1), dim=1, eps=1e-6)
    target_vec = F.normalize(target.flatten(1), dim=1, eps=1e-6)
    return (pred_vec * target_vec).sum(dim=1).mean()
