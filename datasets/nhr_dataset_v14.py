from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from utils.nhr_utils_v14 import IMAGENET_MEAN, IMAGENET_STD
from utils.nhr_utils_v14 import FACE_LABELS, HAIR_LABEL, NECK_LABELS, CLOTH_LABELS, build_reveal_masks, dilate, ensure_4d, label_mask
from models.Net import get_segmentation


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_rgb(path: str | Path, *, size: tuple[int, int] | None = (256, 256), tanh: bool = False) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None:
            image = image.resize(size[::-1], Image.BILINEAR)
        tensor = T.functional.to_tensor(image)
    if tanh:
        tensor = tensor * 2.0 - 1.0
    return tensor


def load_parsing(cache_path: str | Path) -> torch.Tensor:
    cache_path = str(cache_path)
    if cache_path:
        parsing = torch.from_numpy(__import__("numpy").load(cache_path)).long()
        if parsing.dim() == 2:
            parsing = parsing.unsqueeze(0)
        return parsing
    raise ValueError("Empty parsing cache path.")


def load_or_compute_parsing(image_path: str | Path, cache_path: str | Path) -> torch.Tensor:
    cache_path = str(cache_path)
    if cache_path:
        return load_parsing(cache_path)

    image = load_rgb(image_path, size=(512, 512), tanh=False).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = image.to(device)
    image = torch.stack([T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(img) for img in image], dim=0)
    parsing = get_segmentation(image).detach().cpu().long()[0]
    return parsing


def _maybe_flip(item: dict, enabled: bool) -> dict:
    if not enabled or random.random() <= 0.5:
        return item
    flipped = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            if value.dim() == 3:
                flipped[key] = T.functional.hflip(value)
            elif value.dim() == 2:
                flipped[key] = T.functional.hflip(value.unsqueeze(0)).squeeze(0)
            else:
                flipped[key] = value
        else:
            flipped[key] = value
    return flipped


class NHRPairDatasetV14(Dataset):
    def __init__(
        self,
        manifest_path: str | Path | None = None,
        augment_flip: bool = False,
        records: list[dict] | None = None,
        include_full_images: bool = True,
    ):
        if records is None:
            if manifest_path is None:
                raise ValueError("Either manifest_path or records must be provided.")
            records = read_jsonl(manifest_path)
        self.records = list(records)
        self.augment_flip = augment_flip
        self.include_full_images = bool(include_full_images)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        item = {
            "sample_id": record["sample_id"],
            "source_path": record["source_path"],
            "reference_path": record["reference_path"],
            "source_image": load_rgb(record["source_path"], tanh=False),
            "reference_image": load_rgb(record["reference_path"], tanh=False),
            "source_image_tanh": load_rgb(record["source_path"], tanh=True),
            "reference_image_tanh": load_rgb(record["reference_path"], tanh=True),
            "source_parsing": load_or_compute_parsing(record["source_path"], record["source_parsing_path"]),
            "reference_parsing": load_or_compute_parsing(record["reference_path"], record["reference_parsing_path"]),
        }
        if self.include_full_images:
            item["source_image_full"] = load_rgb(record["source_path"], size=None, tanh=False)
            item["reference_image_full"] = load_rgb(record["reference_path"], size=None, tanh=False)
        return _maybe_flip(item, self.augment_flip)


class NHRSourceDatasetV14(Dataset):
    def __init__(
        self,
        manifest_path: str | Path | None = None,
        augment_flip: bool = False,
        records: list[dict] | None = None,
        include_full_images: bool = True,
    ):
        if records is None:
            if manifest_path is None:
                raise ValueError("Either manifest_path or records must be provided.")
            records = read_jsonl(manifest_path)
        unique = {}
        for record in records:
            unique.setdefault(
                record["source_path"],
                {
                    "sample_id": Path(record["source_path"]).stem,
                    "source_path": record["source_path"],
                    "source_parsing_path": record["source_parsing_path"],
                },
            )
        self.records = list(unique.values())
        self.augment_flip = augment_flip
        self.include_full_images = bool(include_full_images)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        item = {
            "sample_id": record["sample_id"],
            "source_path": record["source_path"],
            "source_image": load_rgb(record["source_path"], tanh=False),
            "source_image_tanh": load_rgb(record["source_path"], tanh=True),
            "source_parsing": load_or_compute_parsing(record["source_path"], record["source_parsing_path"]),
        }
        if self.include_full_images:
            item["source_image_full"] = load_rgb(record["source_path"], size=None, tanh=False)
        return _maybe_flip(item, self.augment_flip)


def collate_dict(batch: list[dict]) -> dict:
    output = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        output[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return output


def _shift_tensor(tensor: torch.Tensor, shift_x: int, shift_y: int) -> torch.Tensor:
    tensor = ensure_4d(tensor)
    shifted = torch.roll(tensor, shifts=(shift_y, shift_x), dims=(-2, -1))
    if shift_y > 0:
        shifted[..., :shift_y, :] = 0
    elif shift_y < 0:
        shifted[..., shift_y:, :] = 0
    if shift_x > 0:
        shifted[..., :, :shift_x] = 0
    elif shift_x < 0:
        shifted[..., :, shift_x:] = 0
    return shifted


def build_synthetic_pretrain_batch_v14(
    source_image_tanh: torch.Tensor,
    source_parsing: torch.Tensor,
    *,
    occ_dilate: int = 7,
) -> dict[str, torch.Tensor]:
    source_image_tanh = ensure_4d(source_image_tanh).float()
    source_parsing = ensure_4d(source_parsing).long()

    H_clean = label_mask(source_parsing, (HAIR_LABEL,))
    M_face = label_mask(source_parsing, FACE_LABELS)
    M_neck = label_mask(source_parsing, NECK_LABELS)
    M_cloth = label_mask(source_parsing, CLOTH_LABELS)
    M_nonhair = (M_face + M_neck + M_cloth).clamp(0.0, 1.0)

    ring = (dilate(H_clean, random.randint(5, 9)) - H_clean).clamp(0.0, 1.0)
    candidate = (ring * M_nonhair).clamp(0.0, 1.0)
    if candidate.sum() < 32:
        candidate = M_nonhair

    shift_x = random.randint(-24, 24)
    shift_y = random.randint(6, 28)
    shifted_hair = _shift_tensor(dilate(H_clean, random.randint(3, 7)), shift_x, shift_y)
    occ = (shifted_hair * candidate).clamp(0.0, 1.0)
    if occ.sum() < 32:
        occ = (candidate * (dilate(H_clean, 9) - H_clean).clamp(0.0, 1.0)).clamp(0.0, 1.0)

    shifted_image = _shift_tensor(source_image_tanh, shift_x, shift_y)
    blurred = F.avg_pool2d(source_image_tanh, kernel_size=7, stride=1, padding=3)

    H_source_occ = (H_clean + occ).clamp(0.0, 1.0)
    H_align = H_clean
    M_occ = occ
    M_occ_d = (dilate(M_occ, occ_dilate) * (1.0 - H_align)).clamp(0.0, 1.0)
    M_safe = ((1.0 - H_source_occ) * (1.0 - H_align) * (1.0 - M_occ_d)).clamp(0.0, 1.0)

    I_source_occ = source_image_tanh * (1.0 - occ) + shifted_image * occ
    I_bg0 = source_image_tanh * (1.0 - M_occ_d) + blurred * M_occ_d
    I_hair_coarse = source_image_tanh.clone()
    I_gt = source_image_tanh.clone()

    return {
        "I_source": I_source_occ,
        "I_bg0": I_bg0,
        "I_hair_coarse": I_hair_coarse,
        "I_gt": I_gt,
        "H_source": H_source_occ,
        "H_align": H_align,
        "M_occ": M_occ,
        "M_occ_d": M_occ_d,
        "M_ring": (ring * M_nonhair).clamp(0.0, 1.0),
        "M_face": M_face,
        "M_neck": M_neck,
        "M_cloth": M_cloth,
        "M_safe": M_safe,
        "source_parsing": source_parsing,
    }
