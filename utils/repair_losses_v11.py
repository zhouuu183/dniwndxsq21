from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.deocclusion_masks_v11 import ensure_mask_4d


def gray_v11(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def masked_l1_v11(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    denom = mask.sum().clamp(min=1.0) * pred.shape[1]
    return ((pred - target).abs() * mask).sum() / denom


def sobel_edges_v11(image: torch.Tensor) -> torch.Tensor:
    g = gray_v11(image)
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


def masked_grad_l1_v11(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_l1_v11(sobel_edges_v11(pred), sobel_edges_v11(target), mask)


def alpha_tv_v11(alpha: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    alpha = ensure_mask_4d(alpha).float()
    dx = (alpha[:, :, :, 1:] - alpha[:, :, :, :-1]).abs()
    dy = (alpha[:, :, 1:, :] - alpha[:, :, :-1, :]).abs()
    if mask is None:
        return dx.mean() + dy.mean()
    mask = ensure_mask_4d(mask).float()
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    loss_x = (dx * mask_x).sum() / mask_x.sum().clamp(min=1.0)
    loss_y = (dy * mask_y).sum() / mask_y.sum().clamp(min=1.0)
    return loss_x + loss_y


def non_dark_v11(base: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    value = F.relu(gray_v11(base) - gray_v11(pred)) * mask
    return value.sum() / mask.sum().clamp(min=1.0)


def chroma_v11(image: torch.Tensor) -> torch.Tensor:
    return image - gray_v11(image)


def chroma_delta_l1_v11(pred: torch.Tensor, base: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    value = (chroma_v11(pred) - chroma_v11(base)).abs() * mask
    return value.sum() / (mask.sum().clamp(min=1.0) * pred.shape[1])


def dark_residual_v11(pred: torch.Tensor, mask: torch.Tensor, threshold: float = 0.16) -> torch.Tensor:
    mask = ensure_mask_4d(mask).float()
    value = F.relu(float(threshold) - gray_v11(pred)) * mask
    return value.sum() / mask.sum().clamp(min=1.0)


__all__ = [
    "alpha_tv_v11",
    "chroma_delta_l1_v11",
    "chroma_v11",
    "dark_residual_v11",
    "gray_v11",
    "masked_grad_l1_v11",
    "masked_l1_v11",
    "non_dark_v11",
    "sobel_edges_v11",
]
