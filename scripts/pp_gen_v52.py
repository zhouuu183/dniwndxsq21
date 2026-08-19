import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hair_swap_v52 import HairFastV52 as HairFast, get_parser
from models.ear_modules_v52 import (
    EarAnchoredQueryBuilder,
    FaceParsingHelperV52,
    HairMaskExtractorV52,
    dilate_mask,
    erode_mask,
    gaussian_blur,
    high_pass_filter,
)
from utils.bicubic import BicubicDownSample
from utils.image_utils import list_image_files
from utils.train import seed_everything

CLEANUP_MASK_KEYS = ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")

# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small_accessory_ffhq"  # "small_accessory_ffhq" or "full_ffhq"

USER_FACE_GALLERY_DIR_SMALL = Path("images/ear")
USER_DONOR_GALLERY_DIR_SMALL = Path("images/FFHQ_short")
USER_OUTPUT_DIR_SMALL = Path("images/pp_dataset_v52_dual_small")
USER_DATASET_SIZE_SMALL = 0  # 0 means use every source image.
USER_CHUNK_SIZE_SMALL = 128
USER_MASK_BATCH_SIZE_SMALL = 8

USER_FACE_GALLERY_DIR_FULL = Path("images/FFHQ")
USER_DONOR_GALLERY_DIR_FULL = Path("images/FFHQ_short")
USER_OUTPUT_DIR_FULL = Path("images/pp_dataset_v52_dual_full")
USER_DATASET_SIZE_FULL = 10_000
USER_CHUNK_SIZE_FULL = 256
USER_MASK_BATCH_SIZE_FULL = 16

USER_RANDOM_SEED = 3407
USER_BLENDING_CHECKPOINT = "./checkpoints/blending_3000best.pth"
USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = "./checkpoints/satd_3000_best.pth"
USER_SATD_BLEND_V8 = 0.28
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0

USER_IO_NUM_WORKERS = 0
USER_PREFETCH_FACTOR = 1
USER_SMOOTH = 5

USER_EAR_PARSE_SIZE = 512
USER_EAR_DILATE = 21
USER_HAIR_CHANGE_DILATE = 25
USER_EARRING_EXPAND = 15
USER_EAR_DOWNWARD_SHIFT = 10
USER_TARGET_HAIR_DILATE = 11
USER_SOURCE_HAIR_BLOCK_DILATE = 5
USER_SOURCE_HAIR_BLOCK_STRENGTH = 0.95
USER_TARGET_VISIBILITY_EXPAND = 5
USER_MAX_TARGET_HAIR_OVERLAP = 0.55

USER_EARRING_RECALL_MIN_HIGH = 0.012  # Minimum high-frequency response for weak earring recall.
USER_EARRING_RECALL_MIN_CHROMA = 0.035  # Color contrast threshold for colored earrings.
USER_EARRING_RECALL_MIN_CONTRAST = 0.018  # Local luminance contrast threshold for metallic earrings.
USER_EARRING_RECALL_FALLBACK_PIXELS = 48  # Top-k fallback pixels when parser/highlight recall is empty.
USER_EARRING_CLEAN_MIN_AREA = 2  # Reject tiny isolated speckles.
USER_EARRING_CLEAN_MAX_AREA = 420  # Reject large hair/background blobs.
USER_EARRING_CLEAN_MAX_ASPECT = 5.5  # Reject long strip-like hair/background components.
USER_EARRING_CLEAN_MAX_HAIR_OVERLAP = 0.35  # Reject components mostly overlapping source hair.
USER_EARRING_CLEAN_MIN_TEXTURE_DENSITY = 0.18  # Keep components with enough local detail.
USER_EARRING_CLEAN_KEEP_PER_SIDE = 2  # Keep at most this many candidate components per ear side.
# ============================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "small_accessory_ffhq": {
            "face_gallery_dir": USER_FACE_GALLERY_DIR_SMALL,
            "donor_gallery_dir": USER_DONOR_GALLERY_DIR_SMALL,
            "output": USER_OUTPUT_DIR_SMALL,
            "size": USER_DATASET_SIZE_SMALL,
            "chunk_size": USER_CHUNK_SIZE_SMALL,
            "mask_batch_size": USER_MASK_BATCH_SIZE_SMALL,
        },
        "full_ffhq": {
            "face_gallery_dir": USER_FACE_GALLERY_DIR_FULL,
            "donor_gallery_dir": USER_DONOR_GALLERY_DIR_FULL,
            "output": USER_OUTPUT_DIR_FULL,
            "size": USER_DATASET_SIZE_FULL,
            "chunk_size": USER_CHUNK_SIZE_FULL,
            "mask_batch_size": USER_MASK_BATCH_SIZE_FULL,
        },
    }
    if USER_DATASET_PROFILE not in profiles:
        raise RuntimeError(
            f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}. "
            f"Choose one of: {', '.join(sorted(profiles))}."
        )
    return profiles[USER_DATASET_PROFILE]


PROFILE = resolve_dataset_profile()
ACTIVE_FACE_GALLERY_DIR = PROFILE["face_gallery_dir"]
ACTIVE_DONOR_GALLERY_DIR = PROFILE["donor_gallery_dir"]
ACTIVE_OUTPUT_DIR = PROFILE["output"]
ACTIVE_DATASET_SIZE = PROFILE["size"]
ACTIVE_CHUNK_SIZE = PROFILE["chunk_size"]
ACTIVE_MASK_BATCH_SIZE = PROFILE["mask_batch_size"]

RESOLVED_USER_CONFIG = {
    "dataset_profile": USER_DATASET_PROFILE,
    "face_gallery_dir": ACTIVE_FACE_GALLERY_DIR,
    "donor_gallery_dir": ACTIVE_DONOR_GALLERY_DIR,
    "seed": USER_RANDOM_SEED,
    "size": ACTIVE_DATASET_SIZE,
    "output": ACTIVE_OUTPUT_DIR,
    "blending_checkpoint": USER_BLENDING_CHECKPOINT,
    "use_satd_v8": USER_USE_SATD_V8,
    "satd_checkpoint_v8": USER_SATD_CHECKPOINT_V8,
    "satd_blend_v8": USER_SATD_BLEND_V8,
    "satd_boundary_v8": USER_SATD_BOUNDARY_V8,
    "eq8_reference_blend_v8": USER_EQ8_REFERENCE_BLEND_V8,
    "chunk_size": ACTIVE_CHUNK_SIZE,
    "mask_batch_size": ACTIVE_MASK_BATCH_SIZE,
    "io_num_workers": USER_IO_NUM_WORKERS,
    "prefetch_factor": USER_PREFETCH_FACTOR,
    "smooth": USER_SMOOTH,
    "ear_parse_size": USER_EAR_PARSE_SIZE,
    "ear_dilate": USER_EAR_DILATE,
    "hair_change_dilate": USER_HAIR_CHANGE_DILATE,
    "earring_expand": USER_EARRING_EXPAND,
    "ear_downward_shift": USER_EAR_DOWNWARD_SHIFT,
    "target_hair_dilate": USER_TARGET_HAIR_DILATE,
    "source_hair_block_dilate": USER_SOURCE_HAIR_BLOCK_DILATE,
    "source_hair_block_strength": USER_SOURCE_HAIR_BLOCK_STRENGTH,
    "target_visibility_expand": USER_TARGET_VISIBILITY_EXPAND,
    "max_target_hair_overlap": USER_MAX_TARGET_HAIR_OVERLAP,
    "earring_recall_min_high": USER_EARRING_RECALL_MIN_HIGH,
    "earring_recall_min_chroma": USER_EARRING_RECALL_MIN_CHROMA,
    "earring_recall_min_contrast": USER_EARRING_RECALL_MIN_CONTRAST,
    "earring_recall_fallback_pixels": USER_EARRING_RECALL_FALLBACK_PIXELS,
    "earring_clean_min_area": USER_EARRING_CLEAN_MIN_AREA,
    "earring_clean_max_area": USER_EARRING_CLEAN_MAX_AREA,
    "earring_clean_max_aspect": USER_EARRING_CLEAN_MAX_ASPECT,
    "earring_clean_max_hair_overlap": USER_EARRING_CLEAN_MAX_HAIR_OVERLAP,
    "earring_clean_min_texture_density": USER_EARRING_CLEAN_MIN_TEXTURE_DENSITY,
    "earring_clean_keep_per_side": USER_EARRING_CLEAN_KEEP_PER_SIDE,
}


class ImageException(Exception):
    def __init__(self, image, message="Return image before PP"):
        self.image = image
        self.message = message
        super().__init__(self.message)


def to_single_mask(mask):
    mask = mask.detach().float().cpu()
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3 and mask.size(0) != 1:
        mask = mask[:1]
    return mask.clamp(0, 1)


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def build_earring_recall_mask(
    source_01: torch.Tensor,
    visible_roi: torch.Tensor,
    parser_mask: torch.Tensor,
    source_hair_block_mask: torch.Tensor | None,
    args,
) -> torch.Tensor:
    visible_roi = visible_roi.float().clamp(0, 1)
    parser_mask = parser_mask.float().clamp(0, 1) * visible_roi

    high_energy = high_pass_filter(source_01).abs().mean(dim=1, keepdim=True) * visible_roi
    chroma = (source_01.amax(dim=1, keepdim=True) - source_01.amin(dim=1, keepdim=True)) * visible_roi
    gray = rgb_to_gray(source_01)
    contrast = (gray - gaussian_blur(gray, kernel_size=15, sigma=4.0)).abs() * visible_roi

    flat_signal = high_energy.flatten(2)
    flat_roi = visible_roi.flatten(2)
    roi_pixels = flat_roi.sum(dim=2, keepdim=True).clamp_min(1.0)
    high_mean = (flat_signal.sum(dim=2, keepdim=True) / roi_pixels).view(-1, 1, 1, 1)
    high_max = flat_signal.amax(dim=2, keepdim=True).view(-1, 1, 1, 1)
    min_high = torch.full_like(high_mean, float(args.earring_recall_min_high))
    high_threshold = torch.maximum(min_high, torch.maximum(0.45 * high_max, 1.8 * high_mean))

    high_seed = (high_energy >= high_threshold).float()
    contrast_seed = (
        (chroma > float(args.earring_recall_min_chroma)).float()
        + (contrast > float(args.earring_recall_min_contrast)).float()
    ).clamp(0, 1)
    weak_seed = high_seed * contrast_seed * visible_roi

    if source_hair_block_mask is not None:
        source_hair_block_mask = source_hair_block_mask.float().clamp(0, 1)
        weak_seed = weak_seed * (1 - 0.35 * source_hair_block_mask).clamp(0, 1)

    recall_mask = torch.clamp(parser_mask + dilate_mask(weak_seed, 3), 0, 1) * visible_roi

    empty = recall_mask.flatten(1).sum(dim=1) < 2
    if empty.any():
        fallback = torch.zeros_like(recall_mask)
        fallback_signal = (high_energy + 0.7 * contrast + 0.3 * chroma) * visible_roi
        topk_pixels = max(1, int(args.earring_recall_fallback_pixels))
        for idx in torch.nonzero(empty, as_tuple=False).flatten().tolist():
            valid = visible_roi[idx].flatten() > 0
            if valid.sum().item() <= 0:
                continue
            signal = fallback_signal[idx].flatten()
            masked_signal = torch.where(valid, signal, torch.full_like(signal, -1.0))
            k = min(topk_pixels, int(valid.sum().item()))
            top_indices = torch.topk(masked_signal, k=k).indices
            fallback.view(fallback.size(0), -1)[idx, top_indices] = 1.0
        recall_mask = torch.where(empty.view(-1, 1, 1, 1), dilate_mask(fallback, 3) * visible_roi, recall_mask)

    return (recall_mask > 0.05).float()


def _component_boxes(mask_2d: torch.Tensor) -> list[tuple[list[tuple[int, int]], int, int, int, int]]:
    mask_np = (mask_2d.detach().cpu().numpy() > 0)
    height, width = mask_np.shape
    visited = np.zeros_like(mask_np, dtype=bool)
    components = []
    for y in range(height):
        for x in range(width):
            if not mask_np[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels = []
            y_min = y_max = y
            x_min = x_max = x
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                y_min = min(y_min, cy)
                y_max = max(y_max, cy)
                x_min = min(x_min, cx)
                x_max = max(x_max, cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask_np[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append((pixels, y_min, y_max, x_min, x_max))
    return components


def build_earring_clean_mask(
    recall_mask: torch.Tensor,
    parser_mask: torch.Tensor,
    source_01: torch.Tensor,
    visible_roi: torch.Tensor,
    source_hair_mask: torch.Tensor | None,
    source_hair_block_mask: torch.Tensor | None,
    args,
) -> torch.Tensor:
    recall_mask = (recall_mask.float() > 0.05).float()
    parser_mask = (parser_mask.float() > 0.05).float()
    visible_roi = visible_roi.float().clamp(0, 1)
    source_hair_mask = torch.zeros_like(recall_mask) if source_hair_mask is None else source_hair_mask.float().clamp(0, 1)
    source_hair_block_mask = (
        torch.zeros_like(recall_mask)
        if source_hair_block_mask is None
        else source_hair_block_mask.float().clamp(0, 1)
    )

    detail_signal = (
        high_pass_filter(source_01).abs().mean(dim=1, keepdim=True)
        + 0.5 * (rgb_to_gray(source_01) - gaussian_blur(rgb_to_gray(source_01), kernel_size=15, sigma=4.0)).abs()
    ) * visible_roi
    strong_detail = (detail_signal > 0.012).float()

    clean = torch.zeros_like(recall_mask)
    keep_per_side = max(1, int(args.earring_clean_keep_per_side))
    min_area = max(1, int(args.earring_clean_min_area))
    max_area = max(min_area, int(args.earring_clean_max_area))
    max_aspect = max(1.0, float(args.earring_clean_max_aspect))
    max_hair_overlap = max(0.0, min(1.0, float(args.earring_clean_max_hair_overlap)))
    min_texture_density = max(0.0, min(1.0, float(args.earring_clean_min_texture_density)))

    batch, _, height, width = recall_mask.shape
    for batch_idx in range(batch):
        candidates: list[tuple[int, float, list[tuple[int, int]]]] = []
        for pixels, y_min, y_max, x_min, x_max in _component_boxes(recall_mask[batch_idx, 0]):
            area = len(pixels)
            if area < min_area or area > max_area:
                continue
            box_h = max(1, y_max - y_min + 1)
            box_w = max(1, x_max - x_min + 1)
            aspect = max(box_h / box_w, box_w / box_h)
            if aspect > max_aspect:
                continue

            ys = torch.tensor([p[0] for p in pixels], device=recall_mask.device)
            xs = torch.tensor([p[1] for p in pixels], device=recall_mask.device)
            hair_overlap = source_hair_mask[batch_idx, 0, ys, xs].mean().item()
            block_overlap = source_hair_block_mask[batch_idx, 0, ys, xs].mean().item()
            parser_overlap = parser_mask[batch_idx, 0, ys, xs].mean().item()
            texture_density = strong_detail[batch_idx, 0, ys, xs].mean().item()

            if parser_overlap < 0.25 and max(hair_overlap, block_overlap) > max_hair_overlap:
                continue
            if parser_overlap < 0.25 and texture_density < min_texture_density:
                continue

            center_x = 0.5 * (x_min + x_max)
            side_id = 0 if center_x < width / 2 else 1
            score = 4.0 * parser_overlap + 2.0 * texture_density - 1.5 * max(hair_overlap, block_overlap) + min(area, 80) / 80
            candidates.append((side_id, score, pixels))

        for side in (0, 1):
            side_candidates = sorted(
                [(score, pixels) for side_id, score, pixels in candidates if side_id == side],
                key=lambda item: item[0],
                reverse=True,
            )[:keep_per_side]
            for _, pixels in side_candidates:
                ys = torch.tensor([p[0] for p in pixels], device=recall_mask.device)
                xs = torch.tensor([p[1] for p in pixels], device=recall_mask.device)
                clean[batch_idx, 0, ys, xs] = 1.0

    clean = erode_mask(dilate_mask(clean, 3), 3) * visible_roi
    fallback_empty = clean.flatten(1).sum(dim=1).view(-1, 1, 1, 1) <= 0
    parser_fallback = parser_mask * visible_roi
    clean = torch.where(fallback_empty, parser_fallback, clean)
    return (clean > 0.05).float()


def extract_cleanup_masks(align_shape):
    if not isinstance(align_shape, dict):
        return {}
    delta_masks = align_shape.get("delta_masks")
    if not isinstance(delta_masks, dict):
        return {}

    fallback = None
    for key in CLEANUP_MASK_KEYS:
        value = delta_masks.get(key)
        if torch.is_tensor(value):
            fallback = torch.zeros_like(value)
            break
    if fallback is None:
        return {}

    return {
        key: to_single_mask(delta_masks.get(key, fallback))
        for key in CLEANUP_MASK_KEYS
    }


def str2path(value):
    return None if value in {None, "", "None"} else Path(value)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def build_parser(defaults):
    parser = argparse.ArgumentParser(description="PP dataset generator v52")
    parser.add_argument("--dataset_profile", type=str, default=defaults["dataset_profile"])
    parser.add_argument("--face_gallery_dir", type=str2path, default=defaults["face_gallery_dir"])
    parser.add_argument("--donor_gallery_dir", type=str2path, default=defaults["donor_gallery_dir"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--size", type=int, default=defaults["size"])
    parser.add_argument("--output", type=Path, default=defaults["output"])
    parser.add_argument("--blending_checkpoint", type=str, default=defaults["blending_checkpoint"])
    parser.add_argument("--use_satd_v8", type=str2bool, default=defaults["use_satd_v8"])
    parser.add_argument("--satd_checkpoint_v8", type=str, default=defaults["satd_checkpoint_v8"])
    parser.add_argument("--satd_blend_v8", type=float, default=defaults["satd_blend_v8"])
    parser.add_argument("--satd_boundary_v8", type=int, default=defaults["satd_boundary_v8"])
    parser.add_argument("--eq8_reference_blend_v8", type=float, default=defaults["eq8_reference_blend_v8"])
    parser.add_argument("--chunk_size", type=int, default=defaults["chunk_size"])
    parser.add_argument("--mask_batch_size", type=int, default=defaults["mask_batch_size"])
    parser.add_argument("--io_num_workers", type=int, default=defaults["io_num_workers"])
    parser.add_argument("--prefetch_factor", type=int, default=defaults["prefetch_factor"])
    parser.add_argument("--smooth", type=int, default=defaults["smooth"])
    parser.add_argument("--ear_parse_size", type=int, default=defaults["ear_parse_size"])
    parser.add_argument("--ear_dilate", type=int, default=defaults["ear_dilate"])
    parser.add_argument("--hair_change_dilate", type=int, default=defaults["hair_change_dilate"])
    parser.add_argument("--earring_expand", type=int, default=defaults["earring_expand"])
    parser.add_argument("--ear_downward_shift", type=int, default=defaults["ear_downward_shift"])
    parser.add_argument("--target_hair_dilate", type=int, default=defaults["target_hair_dilate"])
    parser.add_argument("--source_hair_block_dilate", type=int, default=defaults["source_hair_block_dilate"])
    parser.add_argument("--source_hair_block_strength", type=float, default=defaults["source_hair_block_strength"])
    parser.add_argument("--target_visibility_expand", type=int, default=defaults["target_visibility_expand"])
    parser.add_argument("--max_target_hair_overlap", type=float, default=defaults["max_target_hair_overlap"])
    parser.add_argument("--earring_recall_min_high", type=float, default=defaults["earring_recall_min_high"])
    parser.add_argument("--earring_recall_min_chroma", type=float, default=defaults["earring_recall_min_chroma"])
    parser.add_argument("--earring_recall_min_contrast", type=float, default=defaults["earring_recall_min_contrast"])
    parser.add_argument("--earring_recall_fallback_pixels", type=int, default=defaults["earring_recall_fallback_pixels"])
    parser.add_argument("--earring_clean_min_area", type=int, default=defaults["earring_clean_min_area"])
    parser.add_argument("--earring_clean_max_area", type=int, default=defaults["earring_clean_max_area"])
    parser.add_argument("--earring_clean_max_aspect", type=float, default=defaults["earring_clean_max_aspect"])
    parser.add_argument("--earring_clean_max_hair_overlap", type=float, default=defaults["earring_clean_max_hair_overlap"])
    parser.add_argument("--earring_clean_min_texture_density", type=float, default=defaults["earring_clean_min_texture_density"])
    parser.add_argument("--earring_clean_keep_per_side", type=int, default=defaults["earring_clean_keep_per_side"])
    return parser


def hairfast_wo_pp(hair_fast):
    class RaiseDownsample(nn.Module):
        def forward(self, image):
            image = ((image[0] + 1) / 2).clip(0, 1)
            raise ImageException(image)

    def blend_images(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except ImageException as error:
                align_shape = args[0] if args else {}
                return error.image, extract_cleanup_masks(align_shape)

        return wrapper

    hair_fast.blend.downsample_256 = RaiseDownsample()
    hair_fast.blend.blend_images = blend_images(hair_fast.blend.blend_images)


def load_image(path):
    with Image.open(path) as image:
        return T.functional.to_tensor(image.convert("RGB"))


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def count_dataset_parts(total_items, chunk_size, batch_size):
    total_parts = 0
    for start in range(0, total_items, chunk_size):
        total_parts += ceil_div(min(chunk_size, total_items - start), batch_size)
    return total_parts


class RenderedPairDataset(Dataset):
    def __init__(self, experiments, dataset_path, face_gallery_root):
        self.experiments = experiments
        self.dataset_path = Path(dataset_path)
        self.face_gallery_root = Path(face_gallery_root)

    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, idx):
        item = self.experiments[idx]
        source_path = self.face_gallery_root / item["source_name"]
        target_path = self.dataset_path / item["target_name"]
        return {
            "source_path": str(source_path),
            "source_full": load_image(source_path),
            "target_full": load_image(target_path),
            "cleanup_masks": item.get("cleanup_masks", {}),
        }


class DatasetItemBatchBuilder:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.downsample_256 = BicubicDownSample(factor=4)
        self.hair_mask_extractor = HairMaskExtractorV52(device=self.device, dilate_erosion=args.smooth)
        self.parsing_helper = FaceParsingHelperV52(parse_size=args.ear_parse_size)
        query_builder_kwargs = {
            "ear_dilate": args.ear_dilate,
            "hair_change_dilate": args.hair_change_dilate,
            "earring_expand": args.earring_expand,
            "downward_shift": args.ear_downward_shift,
            "target_hair_dilate": args.target_hair_dilate,
            "source_hair_block_dilate": args.source_hair_block_dilate,
            "source_hair_block_strength": args.source_hair_block_strength,
            "target_visibility_expand": args.target_visibility_expand,
            "max_target_hair_overlap": args.max_target_hair_overlap,
        }
        accepted = inspect.signature(EarAnchoredQueryBuilder.__init__).parameters
        query_builder_kwargs = {
            key: value for key, value in query_builder_kwargs.items() if key in accepted
        }
        self.query_builder = EarAnchoredQueryBuilder(**query_builder_kwargs)

    def _iter_single_process_batches(self, dataset):
        batch_size = self.args.mask_batch_size
        total_batches = ceil_div(len(dataset), batch_size)
        for start in tqdm(range(0, len(dataset), batch_size), total=total_batches, leave=False):
            end = min(len(dataset), start + batch_size)
            batch_items = [dataset[idx] for idx in range(start, end)]
            yield {
                "source_path": [item["source_path"] for item in batch_items],
                "source_full": torch.stack([item["source_full"] for item in batch_items], dim=0),
                "target_full": torch.stack([item["target_full"] for item in batch_items], dim=0),
                "cleanup_masks": {
                    key: torch.stack(
                        [
                            item.get("cleanup_masks", {}).get(
                                key,
                                torch.zeros(1, 256, 256),
                            )
                            for item in batch_items
                        ],
                        dim=0,
                    )
                    for key in CLEANUP_MASK_KEYS
                },
            }

    @torch.no_grad()
    def iter_batches(self, experiments, dataset_path, face_gallery_root):
        dataset = RenderedPairDataset(experiments, dataset_path, face_gallery_root)
        if self.args.io_num_workers <= 0:
            batch_iterator = self._iter_single_process_batches(dataset)
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.args.mask_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.args.io_num_workers,
                pin_memory=False,
                prefetch_factor=self.args.prefetch_factor,
                persistent_workers=False,
            )
            batch_iterator = tqdm(dataloader, leave=False)

        for batch in batch_iterator:
            source_paths = batch["source_path"]
            source_full = batch["source_full"].to(self.device, non_blocking=False)
            target_full = batch["target_full"].to(self.device, non_blocking=False)
            batch_cleanup_masks = batch.get("cleanup_masks", {})

            source_256 = self.downsample_256(source_full).clip(0, 1)
            target_256 = self.downsample_256(target_full).clip(0, 1)
            source_hair_d, _ = self.hair_mask_extractor.generate_mask(source_full)
            target_hair_d, target_hair_e = self.hair_mask_extractor.generate_mask(target_full)
            target_mask = (1 - source_hair_d) * (1 - target_hair_d)

            source_parsing = self.parsing_helper.parse(source_256, out_size=(256, 256))
            target_parsing = self.parsing_helper.parse(target_256, out_size=(256, 256))
            query_info = self.query_builder(source_parsing, target_parsing, source_hair_d, target_hair_d)
            source_hair_block_mask = query_info.get("source_hair_block_mask")
            if source_hair_block_mask is None:
                source_hair_block_mask = torch.zeros_like(query_info["query_mask"])
            visible_roi = query_info.get("visible_ear_roi", query_info["ear_roi"])
            source_earring_recall_mask = build_earring_recall_mask(
                source_256,
                visible_roi,
                query_info["source_earring_mask"],
                source_hair_block_mask,
                self.args,
            )
            source_earring_clean_mask = build_earring_clean_mask(
                source_earring_recall_mask,
                query_info["source_earring_mask"],
                source_256,
                visible_roi,
                query_info.get("source_hair_mask"),
                source_hair_block_mask,
                self.args,
            )
            cleanup_masks = {}
            for key in CLEANUP_MASK_KEYS:
                value = batch_cleanup_masks.get(key) if isinstance(batch_cleanup_masks, dict) else None
                if value is None:
                    value = torch.zeros_like(query_info["query_mask"])
                cleanup_masks[key] = value.to(self.device, non_blocking=False).float().clamp(0, 1)

            batch_size = source_256.size(0)
            dataset_items = []
            for idx in range(batch_size):
                item = {
                    "source_path": source_paths[idx],
                    "target": target_256[idx].cpu(),
                    "target_mask": target_mask[idx].cpu(),
                    "HT_E": target_hair_e[idx].cpu(),
                    "source_parsing": source_parsing[idx].cpu(),
                    "target_parsing": target_parsing[idx].cpu(),
                    "source_hair_mask": query_info["source_hair_mask"][idx].cpu(),
                    "target_hair_mask": query_info["target_hair_mask"][idx].cpu(),
                    "source_hair_block_mask": source_hair_block_mask[idx].cpu(),
                    "source_earring_mask": query_info["source_earring_mask"][idx].cpu(),
                    "source_earring_recall_mask": source_earring_recall_mask[idx].cpu(),
                    "source_earring_clean_mask": source_earring_clean_mask[idx].cpu(),
                    "target_earring_mask": query_info["target_earring_mask"][idx].cpu(),
                    "query_mask": query_info["query_mask"][idx].cpu(),
                    "ear_roi": query_info["ear_roi"][idx].cpu(),
                    "left_ear_roi": query_info["left_ear_roi"][idx].cpu(),
                    "right_ear_roi": query_info["right_ear_roi"][idx].cpu(),
                    "presence_target": query_info["presence_target"][idx].cpu(),
                }
                for key in CLEANUP_MASK_KEYS:
                    item[key] = cleanup_masks[key][idx].cpu()
                dataset_items.append(item)

            yield dataset_items

            del batch
            del dataset_items
            del source_full, target_full, source_256, target_256
            del source_hair_d, target_hair_d, target_hair_e, target_mask
            del source_parsing, target_parsing, query_info, cleanup_masks
            del source_earring_recall_mask, source_earring_clean_mask
            if self.device == "cuda":
                torch.cuda.empty_cache()


def main(args):
    if args.face_gallery_dir is None:
        raise ValueError("Please set USER_FACE_GALLERY_DIR_SMALL/FULL in the user config.")
    if args.donor_gallery_dir is None:
        raise ValueError("Please set USER_DONOR_GALLERY_DIR_SMALL/FULL in the user config.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if args.mask_batch_size <= 0:
        raise ValueError("--mask_batch_size must be positive")
    if args.io_num_workers < 0:
        raise ValueError("--io_num_workers must be non-negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be positive")

    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    model_parser = get_parser()
    model_args = model_parser.parse_args([])
    model_args.smooth = args.smooth
    model_args.blending_checkpoint = args.blending_checkpoint
    model_args.use_satd_v8 = args.use_satd_v8
    model_args.satd_checkpoint_v8 = args.satd_checkpoint_v8
    model_args.satd_blend_v8 = args.satd_blend_v8
    model_args.satd_boundary_v8 = args.satd_boundary_v8
    model_args.eq8_reference_blend_v8 = args.eq8_reference_blend_v8
    hair_fast = HairFast(model_args)
    hairfast_wo_pp(hair_fast)
    item_batch_builder = DatasetItemBatchBuilder(args)

    face_images = list_image_files(args.face_gallery_dir)
    donor_images = list_image_files(args.donor_gallery_dir)
    if len(face_images) == 0:
        raise ValueError(f"No images were found under face_gallery_dir: {args.face_gallery_dir}")
    if len(donor_images) == 0:
        raise ValueError(f"No images were found under donor_gallery_dir: {args.donor_gallery_dir}")

    resolved_size = args.size if args.size > 0 else len(face_images)
    if resolved_size <= 0:
        raise ValueError("Resolved experiment size must be positive")

    print(
        f"Using dataset_profile={args.dataset_profile}, source_dir={args.face_gallery_dir}, "
        f"donor_dir={args.donor_gallery_dir}, size={resolved_size}"
    )

    face_replace = resolved_size > len(face_images)
    donor_replace = (2 * resolved_size) > len(donor_images)
    face = np.random.choice(face_images, size=resolved_size, replace=face_replace)
    shape, color = np.array_split(np.random.choice(donor_images, size=2 * resolved_size, replace=donor_replace), 2)

    experiments = []
    for exp in zip(face, shape, color):
        stem_names = [Path(name).stem for name in exp]
        experiments.append(
            {
                "source_name": exp[0],
                "target_name": f"{'_'.join(stem_names)}.png",
                "triplet": exp,
            }
        )

    total_parts = count_dataset_parts(len(experiments), args.chunk_size, args.mask_batch_size)
    print(
        f"Planned {len(experiments)} experiments, expected to write {total_parts} dataset parts "
        f"(chunk_size={args.chunk_size}, mask_batch_size={args.mask_batch_size}, "
        f"io_num_workers={args.io_num_workers})."
    )

    left = 0
    right = min(len(experiments), args.chunk_size)
    part_idx = 1
    while left < len(experiments):
        batch_experiments = experiments[left:right]
        with tempfile.TemporaryDirectory() as temp_dir:
            for item in tqdm(batch_experiments, desc=f"Render chunk {left}:{right}"):
                im1, im2, im3 = item["triplet"]
                result = hair_fast(
                    args.face_gallery_dir / im1,
                    args.donor_gallery_dir / im2,
                    args.donor_gallery_dir / im3,
                )
                if isinstance(result, tuple):
                    image, cleanup_masks = result
                else:
                    image, cleanup_masks = result, {}
                item["cleanup_masks"] = cleanup_masks
                save_image(image, os.path.join(temp_dir, item["target_name"]))

            for dataset_items in item_batch_builder.iter_batches(batch_experiments, temp_dir, args.face_gallery_dir):
                torch.save(dataset_items, args.output / f"pp_part_{part_idx}.dataset")
                part_idx += 1

        left = right
        right = min(len(experiments), right + args.chunk_size)


if __name__ == "__main__":
    parser = build_parser(RESOLVED_USER_CONFIG)
    main(parser.parse_args())


