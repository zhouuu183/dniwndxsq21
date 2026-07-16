from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from utils.train import image_grid


def _to_pil_image(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    image = image.detach().cpu().float()
    if image.size(0) == 1:
        image = image.repeat(3, 1, 1)
    image = image.clamp(0, 1)
    return T.functional.to_pil_image(image)


def _expand_box(box: torch.Tensor, image_hw: tuple[int, int], scale: float = 1.25) -> tuple[int, int, int, int]:
    image_h, image_w = image_hw
    x1, y1, x2, y2 = [float(v) for v in box]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    new_w = width * scale
    new_h = height * scale
    x1 = int(max(0.0, cx - 0.5 * new_w))
    y1 = int(max(0.0, cy - 0.5 * new_h))
    x2 = int(min(float(image_w), cx + 0.5 * new_w))
    y2 = int(min(float(image_h), cy + 0.5 * new_h))
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    return x1, y1, x2, y2


def crop_from_box(image: torch.Tensor, box: torch.Tensor, scale: float = 1.25) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    box = box.detach().cpu().flatten().float()
    x1, y1, x2, y2 = _expand_box(box, tuple(image.shape[-2:]), scale=scale)
    return _to_pil_image(image[:, y1:y2, x1:x2])


def make_fixed_pair_panels(
    *,
    source: torch.Tensor,
    reference: torch.Tensor,
    spsa: torch.Tensor,
    bang_box: torch.Tensor,
    tail_box: torch.Tensor,
) -> dict[str, Image.Image]:
    source_image = _to_pil_image(source)
    reference_image = _to_pil_image(reference)
    spsa_image = _to_pil_image(spsa)

    bang_tiles = [
        crop_from_box(source, bang_box),
        crop_from_box(reference, bang_box),
    ]
    tail_tiles = [
        crop_from_box(source, tail_box),
        crop_from_box(reference, tail_box),
    ]
    bang_tiles.append(crop_from_box(spsa, bang_box))
    tail_tiles.append(crop_from_box(spsa, tail_box))

    panels = {
        "source": source_image,
        "reference": reference_image,
        "spsa": spsa_image,
        "bangs_crop": image_grid(bang_tiles, 1, len(bang_tiles)),
        "tail_crop": image_grid(tail_tiles, 1, len(tail_tiles)),
    }
    return panels


def save_fixed_pair_panels(output_dir: str | Path, panels: dict[str, Image.Image]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in panels.items():
        image.save(output_dir / f"{name}.png")
