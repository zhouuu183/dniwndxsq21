from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from models.CtrlHair.external_code.face_parsing.my_parsing_util import FaceParsing, FaceParsing_tensor
from models.CtrlHair.global_value_utils import PARSING_LABEL_LIST
from models.Net import get_segmentation
from utils.image_utils import list_image_files


HAIR_LABEL = 13
FACE_LABELS = tuple(range(1, 13)) + (15,)
NECK_LABELS = (16, 17)
CLOTH_LABELS = (18,)
NON_HAIR_LABELS = FACE_LABELS + NECK_LABELS + CLOTH_LABELS
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def label_mask(parsing: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    parsing = ensure_4d(parsing)
    mask = torch.zeros_like(parsing, dtype=torch.bool)
    for label in labels:
        mask |= parsing == label
    return mask.float()


def binary_mask(mask: torch.Tensor) -> torch.Tensor:
    return (ensure_4d(mask) > 0.5).float()


def dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    mask = binary_mask(mask)
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width)


def erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    mask = binary_mask(mask)
    if width <= 0:
        return mask
    return 1.0 - dilate(1.0 - mask, width)


def build_reveal_masks(
    source_parsing: torch.Tensor,
    target_hair_mask: torch.Tensor,
    *,
    occ_dilate: int = 7,
    ring_dilate: int = 9,
    ring_erode: int = 1,
    use_ring: bool = True,
) -> dict[str, torch.Tensor]:
    source_parsing = ensure_4d(source_parsing)
    H_source = label_mask(source_parsing, (HAIR_LABEL,))
    H_align = binary_mask(target_hair_mask)

    M_occ = (H_source * (1.0 - H_align)).clamp(0.0, 1.0)
    M_occ_d = (dilate(M_occ, occ_dilate) * (1.0 - H_align)).clamp(0.0, 1.0)

    M_face = label_mask(source_parsing, FACE_LABELS)
    M_neck = label_mask(source_parsing, NECK_LABELS)
    M_cloth = label_mask(source_parsing, CLOTH_LABELS)
    M_nonhair = (M_face + M_neck + M_cloth).clamp(0.0, 1.0)

    if use_ring:
        ring = (dilate(H_source, ring_dilate) - erode(H_align, ring_erode)).clamp(0.0, 1.0)
        M_ring = (ring * M_nonhair * (1.0 - H_align)).clamp(0.0, 1.0)
    else:
        M_ring = torch.zeros_like(M_occ)

    M_safe = ((1.0 - H_source) * (1.0 - H_align) * (1.0 - M_occ_d)).clamp(0.0, 1.0)

    return {
        "H_source": H_source,
        "H_align": H_align,
        "M_occ": M_occ,
        "M_occ_d": M_occ_d,
        "M_ring": M_ring,
        "M_face": M_face,
        "M_neck": M_neck,
        "M_cloth": M_cloth,
        "M_nonhair": M_nonhair,
        "M_safe": M_safe,
    }


def mask_to_box(mask: torch.Tensor) -> torch.Tensor:
    mask = binary_mask(mask)[0, 0]
    coords = torch.nonzero(mask > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return torch.tensor([0.0, 0.0, 1.0, 1.0], device=mask.device)
    y1x1 = coords.min(dim=0).values
    y2x2 = coords.max(dim=0).values + 1
    return torch.tensor([y1x1[1], y1x1[0], y2x2[1], y2x2[0]], dtype=torch.float32, device=mask.device)


def resize_mask(mask: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    mask = ensure_4d(mask).float()
    if mode == "nearest":
        return F.interpolate(mask, size=size, mode=mode)
    return F.interpolate(mask, size=size, mode=mode, align_corners=False)


def normalize_bisenet_input(image_tanh_256: torch.Tensor) -> torch.Tensor:
    image_tanh_256 = ensure_4d(image_tanh_256).float()
    image = ((image_tanh_256 + 1.0) / 2.0).clamp(0.0, 1.0)
    image = F.interpolate(image, size=(512, 512), mode="bilinear", align_corners=False)
    normalizer = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return torch.stack([normalizer(img) for img in image], dim=0)


def get_celeba_parsing_logits(image_tanh_256: torch.Tensor) -> torch.Tensor:
    image_bisenet = normalize_bisenet_input(image_tanh_256)
    FaceParsing_tensor.parsing_img()
    parsing_net = FaceParsing_tensor.bise_net
    if parsing_net is None:
        parsing_net = FaceParsing.bise_net
    if parsing_net is None:
        raise RuntimeError("Face parsing network failed to initialize.")
    logits = parsing_net(image_bisenet)[0]

    source_names = list(FaceParsing_tensor.label_list.values())
    celeb_logits = []
    for label_name in PARSING_LABEL_LIST:
        celeb_logits.append(logits[:, source_names.index(label_name)].unsqueeze(1))
    celeb_logits = torch.cat(celeb_logits, dim=1)
    celeb_logits = F.interpolate(celeb_logits, size=image_tanh_256.shape[-2:], mode="bilinear", align_corners=False)
    return celeb_logits


def read_image_tensor(path: str | Path, *, size: tuple[int, int] = (256, 256), normalize_tanh: bool = False) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(size[::-1], Image.BILINEAR)
        tensor = T.functional.to_tensor(image)
    if normalize_tanh:
        tensor = tensor * 2.0 - 1.0
    return tensor


def resolve_image_paths(
    *,
    image_dir: str | Path,
    list_file: str | Path | None = None,
    limit: int | None = None,
) -> list[Path]:
    image_dir = Path(image_dir)
    if list_file is None:
        paths = [image_dir / name for name in list_image_files(image_dir)]
    else:
        paths = []
        with open(list_file, "r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                path = Path(raw)
                if not path.is_absolute():
                    path = image_dir / path
                paths.append(path)
    if limit is not None:
        paths = paths[: max(0, int(limit))]
    return paths


def write_jsonl(records: list[dict], path: str | Path) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def save_parsing_cache(image_path: str | Path, cache_path: str | Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = read_image_tensor(image_path, size=(512, 512), normalize_tanh=False).unsqueeze(0).to(device)
    image = torch.stack([T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(img) for img in image], dim=0)
    parsing = get_segmentation(image).detach().cpu().numpy().astype(np.int16)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, parsing[0, 0])


def load_parsing_cache(cache_path: str | Path) -> torch.Tensor:
    return torch.from_numpy(np.load(cache_path)).long().unsqueeze(0)
