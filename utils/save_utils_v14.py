import os
from pathlib import Path

import torch
import torchvision.transforms as T


to_pil = T.ToPILImage()


def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def _prepare_dir(output_dir: Path | str, folder: str) -> Path:
    save_dir = Path(output_dir) / folder
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _to_display_image(tensor: torch.Tensor, value_range: str) -> torch.Tensor:
    tensor = _ensure_4d(tensor)[0].detach().cpu().float()
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if value_range == "tanh":
        tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    elif value_range == "unit":
        tensor = tensor.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported value_range={value_range!r}")
    return tensor


def save_tensor_image(output_dir: Path | str, folder: str, name: str, tensor: torch.Tensor, value_range: str = "tanh") -> None:
    image = _to_display_image(tensor, value_range=value_range)
    save_dir = _prepare_dir(output_dir, folder)
    to_pil(image).save(save_dir / name)


def save_tensor_mask(output_dir: Path | str, folder: str, name: str, tensor: torch.Tensor) -> None:
    tensor = _ensure_4d(tensor)[0].detach().cpu().float()
    if tensor.shape[0] != 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    tensor = tensor.clamp(0.0, 1.0)
    save_dir = _prepare_dir(output_dir, folder)
    to_pil(tensor).save(save_dir / name)


def save_feature_map(output_dir: Path | str, folder: str, name: str, feature: torch.Tensor) -> None:
    feature = _ensure_4d(feature)[0].detach().cpu().float()
    if feature.shape[0] == 1:
        vis = feature
    else:
        vis = feature.abs().mean(dim=0, keepdim=True)
    vis = vis - vis.amin(dim=(1, 2), keepdim=True)
    vis = vis / vis.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    save_dir = _prepare_dir(output_dir, folder)
    to_pil(vis).save(save_dir / name)
