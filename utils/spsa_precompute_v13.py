from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from torchvision import transforms as T

from models.Net import get_segmentation
from utils.image_utils import list_image_files

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HAIR_LABEL = 13
FACE_LABELS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
EYE_BROW_LABELS = {4, 5, 6, 7}


def read_image_tensor(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        return T.functional.to_tensor(image.convert("RGB"))


def path_to_prior_key(path: str | Path, root: str | Path | None = None) -> str:
    path = Path(path)
    if root is not None:
        try:
            path = path.relative_to(Path(root))
        except ValueError:
            path = path
    stem = path.with_suffix("")
    return "__".join(stem.parts)


def resolve_image_paths(
    *,
    image_dir: str | Path,
    list_file: str | Path | None = None,
    limit: int | None = None,
) -> list[Path]:
    image_dir = Path(image_dir)
    if list_file is None:
        image_paths = [image_dir / file_name for file_name in list_image_files(image_dir)]
    else:
        with open(list_file, "r", encoding="utf-8") as file:
            image_paths = []
            for line in file:
                line = line.strip()
                if not line:
                    continue
                path = Path(line)
                if not path.is_absolute():
                    path = image_dir / path
                image_paths.append(path)
    if limit is not None:
        image_paths = image_paths[: max(0, int(limit))]
    return image_paths


def build_celeba_parsing(image_tensor: torch.Tensor) -> np.ndarray:
    image_512 = T.functional.resize(
        image_tensor,
        [512, 512],
        interpolation=T.InterpolationMode.BILINEAR,
    )
    image_norm = T.functional.normalize(image_512, IMAGENET_MEAN, IMAGENET_STD)
    image_batch = image_norm.unsqueeze(0)
    if torch.cuda.is_available():
        image_batch = image_batch.cuda()
    parsing = get_segmentation(image_batch, resize=False)
    return parsing[0, 0].detach().cpu().numpy().astype(np.int64)


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask > 0.5)
    if coords.size == 0:
        return np.array([0, 0, 1, 1], dtype=np.float32)
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0) + 1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def expand_box(box: np.ndarray, *, image_hw: tuple[int, int], scale_x: float, scale_y: float) -> np.ndarray:
    image_h, image_w = image_hw
    x1, y1, x2, y2 = box.astype(np.float32)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    new_w = width * scale_x
    new_h = height * scale_y
    x1 = max(0.0, cx - 0.5 * new_w)
    y1 = max(0.0, cy - 0.5 * new_h)
    x2 = min(float(image_w), cx + 0.5 * new_w)
    y2 = min(float(image_h), cy + 0.5 * new_h)
    if x2 <= x1:
        x2 = min(float(image_w), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_h), y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def boundary_from_mask(hair_mask: np.ndarray, dilate_iter: int = 1) -> np.ndarray:
    mask_bool = hair_mask > 0.5
    dilated = ndi.binary_dilation(mask_bool, iterations=dilate_iter)
    eroded = ndi.binary_erosion(mask_bool, iterations=dilate_iter)
    boundary = np.logical_and(dilated, np.logical_not(eroded))
    return boundary.astype(np.float32)


def distance_map_from_mask(hair_mask: np.ndarray) -> np.ndarray:
    mask_bool = hair_mask > 0.5
    distance = ndi.distance_transform_edt(mask_bool).astype(np.float32)
    max_value = float(distance.max())
    if max_value > 1e-6:
        distance /= max_value
    return distance * hair_mask.astype(np.float32)


def direction_prior_from_image(image_rgb: np.ndarray, hair_mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    image_u8 = np.clip(image_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)

    angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-6)
    dx = np.cos(angle) * hair_mask
    dy = np.sin(angle) * hair_mask
    norm = np.sqrt(dx * dx + dy * dy)
    norm = np.where(norm > 1e-6, norm, 1.0)
    direction = np.stack([dx / norm, dy / norm], axis=0).astype(np.float32)
    return direction


def bang_roi_from_parsing(hair_mask: np.ndarray, parsing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_h, image_w = hair_mask.shape
    hair_box = mask_to_box(hair_mask)
    face_mask = np.isin(parsing, list(FACE_LABELS)).astype(np.float32)
    face_box = mask_to_box(face_mask) if face_mask.any() else hair_box.copy()
    feature_mask = np.isin(parsing, list(EYE_BROW_LABELS))
    if feature_mask.any():
        feature_y = np.argwhere(feature_mask)[:, 0]
        forehead_bottom = min(image_h, int(feature_y.max() + 0.18 * max(1.0, face_box[3] - face_box[1])))
    else:
        forehead_bottom = min(image_h, int(face_box[1] + 0.30 * max(1.0, face_box[3] - face_box[1])))

    x_center = 0.5 * (face_box[0] + face_box[2])
    width = max(1.0, face_box[2] - face_box[0])
    x1 = max(hair_box[0], x_center - 0.62 * width)
    x2 = min(hair_box[2], x_center + 0.62 * width)
    y1 = hair_box[1]
    y2 = max(y1 + 1.0, min(float(hair_box[3]), float(forehead_bottom)))
    bang_box = expand_box(
        np.array([x1, y1, x2, y2], dtype=np.float32),
        image_hw=(image_h, image_w),
        scale_x=1.10,
        scale_y=1.20,
    )
    bang_mask = np.zeros_like(hair_mask, dtype=np.float32)
    bx1, by1, bx2, by2 = bang_box.astype(np.int32)
    bang_mask[by1:by2, bx1:bx2] = hair_mask[by1:by2, bx1:bx2]
    return bang_box, bang_mask


def tail_roi_from_parsing(hair_mask: np.ndarray, parsing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_h, image_w = hair_mask.shape
    hair_box = mask_to_box(hair_mask)
    face_mask = np.isin(parsing, list(FACE_LABELS))
    face_box = mask_to_box(face_mask.astype(np.float32)) if face_mask.any() else hair_box.copy()

    dilated_face = ndi.binary_dilation(face_mask, iterations=18)
    yy, xx = np.indices(hair_mask.shape)
    lower_band = yy >= int(hair_box[1] + 0.52 * max(1.0, hair_box[3] - hair_box[1]))
    side_band = np.logical_or(xx <= face_box[0], xx >= face_box[2])
    tail_candidate = np.logical_and(hair_mask > 0.5, np.logical_and(np.logical_not(dilated_face), np.logical_or(lower_band, side_band)))
    if tail_candidate.sum() < 32:
        tail_candidate = np.logical_and(hair_mask > 0.5, lower_band)
    tail_box = expand_box(
        mask_to_box(tail_candidate.astype(np.float32)),
        image_hw=(image_h, image_w),
        scale_x=1.20,
        scale_y=1.20,
    )
    tail_mask = np.zeros_like(hair_mask, dtype=np.float32)
    tx1, ty1, tx2, ty2 = tail_box.astype(np.int32)
    tail_mask[ty1:ty2, tx1:tx2] = hair_mask[ty1:ty2, tx1:tx2]
    return tail_box, tail_mask


def compute_spsa_prior(
    image_tensor: torch.Tensor,
    *,
    direction_sigma: float = 2.0,
) -> dict[str, np.ndarray]:
    parsing = build_celeba_parsing(image_tensor)
    hair_mask = (parsing == HAIR_LABEL).astype(np.float32)
    boundary = boundary_from_mask(hair_mask)
    distance_map = distance_map_from_mask(hair_mask)

    image_512 = T.functional.resize(
        image_tensor,
        [512, 512],
        interpolation=T.InterpolationMode.BILINEAR,
    ).permute(1, 2, 0).detach().cpu().numpy()
    direction = direction_prior_from_image(image_512, hair_mask, sigma=direction_sigma)
    bang_box, bang_mask = bang_roi_from_parsing(hair_mask, parsing)
    tail_box, tail_mask = tail_roi_from_parsing(hair_mask, parsing)

    return {
        "H_tgt": hair_mask[None].astype(np.float32),
        "E_tgt": boundary[None].astype(np.float32),
        "D_tgt": distance_map[None].astype(np.float32),
        "O_tgt": direction.astype(np.float32),
        "bang_box": bang_box.astype(np.float32),
        "tail_box": tail_box.astype(np.float32),
        "bang_mask": bang_mask[None].astype(np.float32),
        "tail_mask": tail_mask[None].astype(np.float32),
    }


def save_prior_npz(path: str | Path, prior: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **prior)


def load_prior_npz(path: str | Path) -> dict[str, torch.Tensor]:
    data = np.load(path)
    loaded: dict[str, torch.Tensor] = {}
    for key in data.files:
        loaded[key] = torch.from_numpy(data[key]).float()
    return loaded


def make_manifest_record(
    *,
    sample_id: str,
    source_path: str | Path,
    reference_path: str | Path,
    color_path: str | Path,
    prior_path: str | Path,
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "source_path": str(Path(source_path)),
        "reference_path": str(Path(reference_path)),
        "color_path": str(Path(color_path)),
        "prior_path": str(Path(prior_path)),
    }


def write_manifest(records: Iterable[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
