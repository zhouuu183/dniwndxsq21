import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from hair_swap_v5 import HairFastV5 as HairFast, get_parser
from models.ear_modules_v5 import (
    EarAnchoredQueryBuilder,
    FaceParsingHelperV5,
    HairMaskExtractorV5,
    align_earring_reference_to_target,
    build_earring_highlight_mask,
    build_revealed_skin_mask,
    build_weak_earring_masks,
    dilate_mask,
    enhance_query_with_earring_recall,
    resize_mask,
    select_reference_earring_mask,
)
from utils.bicubic import BicubicDownSample
from utils.image_utils import list_image_files
from utils.train import seed_everything

CLEANUP_MASK_KEYS = ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")
PP_EXTRA_MASK_KEYS = (
    "cleanup_inner_edge",
    "revealed_skin_mask",
    "source_skin_valid_mask",
    "earring_confident_mask",
    "earring_highlight_mask",
    "earring_candidate_mask",
)

# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "full_ffhq"  # "small_accessory_ffhq" or "full_ffhq"

USER_FACE_GALLERY_DIR_SMALL = Path("images/ear")
USER_DONOR_GALLERY_DIR_SMALL = Path("images/FFHQ")
USER_OUTPUT_DIR_SMALL = Path("images/pp_dataset_v5_dual_small")
USER_DATASET_SIZE_SMALL = 30  # 0 means use every source image.
USER_CHUNK_SIZE_SMALL = 128
USER_MASK_BATCH_SIZE_SMALL = 8

USER_FACE_GALLERY_DIR_FULL = Path("images/FFHQ")
USER_DONOR_GALLERY_DIR_FULL = Path("images/FFHQ")
USER_OUTPUT_DIR_FULL = Path("images/pp_dataset_v5_dual_full2")
USER_DATASET_SIZE_FULL = 10_000
USER_CHUNK_SIZE_FULL = 256
USER_MASK_BATCH_SIZE_FULL = 16

USER_RESUME_EXISTING_PARTS = True
USER_SKIP_BAD_RENDERED_SAMPLES = True

USER_RANDOM_SEED = 3407
USER_BLENDING_CHECKPOINT = "checkpoints/blending_3000best.pth"
USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = "checkpoints/satd_3000_best.pth"
USER_SATD_BLEND_V8 = 0.28
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0

USER_IO_NUM_WORKERS = 0
USER_PREFETCH_FACTOR = 1
USER_SMOOTH = 5

USER_EAR_PARSE_SIZE = 512
USER_EAR_LOW_ALPHA = 0.1
USER_EAR_DILATE = 21
USER_HAIR_CHANGE_DILATE = 25
USER_EARRING_EXPAND = 15
USER_EAR_DOWNWARD_SHIFT = 10
USER_TARGET_HAIR_DILATE = 11
USER_SOURCE_HAIR_BLOCK_DILATE = 5
USER_SOURCE_HAIR_BLOCK_STRENGTH = 0.6
USER_TARGET_VISIBILITY_EXPAND = 5
USER_MAX_TARGET_HAIR_OVERLAP = 0.55
USER_TARGET_EAR_COVER_OVERLAP = 0.55
USER_MIN_TARGET_VISIBLE_OVERLAP = 0.02
USER_ENABLE_EARRING_QUERY_RECALL = True
USER_EARRING_QUERY_RECALL_DILATE = 7
USER_EARRING_QUERY_DOWNWARD_SHIFT = 18
USER_EARRING_QUERY_LOWER_LOBE_WEIGHT = 0.20
USER_EARRING_QUERY_CANDIDATE_BOOST = 0.90
USER_EARRING_QUERY_BLOCK_PROTECT = 0.85
USER_EARRING_ALIGN_MAX_SHIFT = 12
USER_EARRING_FINE_MASK_FLOOR = 0.18
USER_EARRING_FINE_MASK_DILATE = 5
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
    "resume_existing_parts": USER_RESUME_EXISTING_PARTS,
    "skip_bad_rendered_samples": USER_SKIP_BAD_RENDERED_SAMPLES,
    "io_num_workers": USER_IO_NUM_WORKERS,
    "prefetch_factor": USER_PREFETCH_FACTOR,
    "smooth": USER_SMOOTH,
    "ear_parse_size": USER_EAR_PARSE_SIZE,
    "ear_low_alpha": USER_EAR_LOW_ALPHA,
    "ear_dilate": USER_EAR_DILATE,
    "hair_change_dilate": USER_HAIR_CHANGE_DILATE,
    "earring_expand": USER_EARRING_EXPAND,
    "ear_downward_shift": USER_EAR_DOWNWARD_SHIFT,
    "target_hair_dilate": USER_TARGET_HAIR_DILATE,
    "source_hair_block_dilate": USER_SOURCE_HAIR_BLOCK_DILATE,
    "source_hair_block_strength": USER_SOURCE_HAIR_BLOCK_STRENGTH,
    "target_visibility_expand": USER_TARGET_VISIBILITY_EXPAND,
    "max_target_hair_overlap": USER_MAX_TARGET_HAIR_OVERLAP,
    "target_ear_cover_overlap": USER_TARGET_EAR_COVER_OVERLAP,
    "min_target_visible_overlap": USER_MIN_TARGET_VISIBLE_OVERLAP,
    "enable_earring_query_recall": USER_ENABLE_EARRING_QUERY_RECALL,
    "earring_query_recall_dilate": USER_EARRING_QUERY_RECALL_DILATE,
    "earring_query_downward_shift": USER_EARRING_QUERY_DOWNWARD_SHIFT,
    "earring_query_lower_lobe_weight": USER_EARRING_QUERY_LOWER_LOBE_WEIGHT,
    "earring_query_candidate_boost": USER_EARRING_QUERY_CANDIDATE_BOOST,
    "earring_query_block_protect": USER_EARRING_QUERY_BLOCK_PROTECT,
    "earring_align_max_shift": USER_EARRING_ALIGN_MAX_SHIFT,
    "earring_fine_mask_floor": USER_EARRING_FINE_MASK_FLOOR,
    "earring_fine_mask_dilate": USER_EARRING_FINE_MASK_DILATE,
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
    parser = argparse.ArgumentParser(description="PP dataset generator v5")
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
    parser.add_argument("--resume_existing_parts", type=str2bool, default=defaults["resume_existing_parts"])
    parser.add_argument("--skip_bad_rendered_samples", type=str2bool, default=defaults["skip_bad_rendered_samples"])
    parser.add_argument("--io_num_workers", type=int, default=defaults["io_num_workers"])
    parser.add_argument("--prefetch_factor", type=int, default=defaults["prefetch_factor"])
    parser.add_argument("--smooth", type=int, default=defaults["smooth"])
    parser.add_argument("--ear_parse_size", type=int, default=defaults["ear_parse_size"])
    parser.add_argument("--ear_low_alpha", type=float, default=defaults["ear_low_alpha"])
    parser.add_argument("--ear_dilate", type=int, default=defaults["ear_dilate"])
    parser.add_argument("--hair_change_dilate", type=int, default=defaults["hair_change_dilate"])
    parser.add_argument("--earring_expand", type=int, default=defaults["earring_expand"])
    parser.add_argument("--ear_downward_shift", type=int, default=defaults["ear_downward_shift"])
    parser.add_argument("--target_hair_dilate", type=int, default=defaults["target_hair_dilate"])
    parser.add_argument("--source_hair_block_dilate", type=int, default=defaults["source_hair_block_dilate"])
    parser.add_argument("--source_hair_block_strength", type=float, default=defaults["source_hair_block_strength"])
    parser.add_argument("--target_visibility_expand", type=int, default=defaults["target_visibility_expand"])
    parser.add_argument("--max_target_hair_overlap", type=float, default=defaults["max_target_hair_overlap"])
    parser.add_argument("--target_ear_cover_overlap", type=float, default=defaults["target_ear_cover_overlap"])
    parser.add_argument("--min_target_visible_overlap", type=float, default=defaults["min_target_visible_overlap"])
    parser.add_argument("--enable_earring_query_recall", type=str2bool, default=defaults["enable_earring_query_recall"])
    parser.add_argument("--earring_query_recall_dilate", type=int, default=defaults["earring_query_recall_dilate"])
    parser.add_argument("--earring_query_downward_shift", type=int, default=defaults["earring_query_downward_shift"])
    parser.add_argument("--earring_query_lower_lobe_weight", type=float, default=defaults["earring_query_lower_lobe_weight"])
    parser.add_argument("--earring_query_candidate_boost", type=float, default=defaults["earring_query_candidate_boost"])
    parser.add_argument("--earring_query_block_protect", type=float, default=defaults["earring_query_block_protect"])
    parser.add_argument("--earring_align_max_shift", type=int, default=defaults["earring_align_max_shift"])
    parser.add_argument("--earring_fine_mask_floor", type=float, default=defaults["earring_fine_mask_floor"])
    parser.add_argument("--earring_fine_mask_dilate", type=int, default=defaults["earring_fine_mask_dilate"])
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


def dataset_part_path(output_dir: Path, part_idx: int) -> Path:
    return output_dir / f"pp_part_{part_idx}.dataset"


def chunk_done_path(output_dir: Path, left: int, right: int) -> Path:
    return output_dir / f"pp_chunk_{left}_{right}.done"


def read_chunk_done(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return int(payload.get("part_count", 0))
    except Exception:
        return None


def write_chunk_done(path: Path, left: int, right: int, start_part: int, part_count: int, skipped_count: int) -> None:
    payload = {
        "left": int(left),
        "right": int(right),
        "start_part": int(start_part),
        "part_count": int(part_count),
        "skipped_count": int(skipped_count),
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
        file.write("\n")


def log_failed_sample(output_dir: Path, item: dict, stage: str, error: Exception) -> None:
    triplet = item.get("triplet", ("", "", ""))
    payload = {
        "stage": stage,
        "source_name": str(item.get("source_name", triplet[0] if len(triplet) > 0 else "")),
        "shape_name": str(triplet[1] if len(triplet) > 1 else ""),
        "color_name": str(triplet[2] if len(triplet) > 2 else ""),
        "target_name": str(item.get("target_name", "")),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    with open(output_dir / "failed_samples.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


class RenderedPairDataset(Dataset):
    def __init__(self, experiments, dataset_path, face_gallery_root, donor_gallery_root):
        self.experiments = experiments
        self.dataset_path = Path(dataset_path)
        self.face_gallery_root = Path(face_gallery_root)
        self.donor_gallery_root = Path(donor_gallery_root)

    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, idx):
        item = self.experiments[idx]
        triplet = item.get("triplet", ("", "", ""))
        source_path = self.face_gallery_root / item["source_name"]
        target_path = self.dataset_path / item["target_name"]
        shape_path = self.donor_gallery_root / triplet[1] if len(triplet) > 1 else None
        color_path = self.donor_gallery_root / triplet[2] if len(triplet) > 2 else None
        return {
            "source_path": str(source_path),
            "shape_path": str(shape_path) if shape_path is not None else "",
            "color_path": str(color_path) if color_path is not None else "",
            "source_full": load_image(source_path),
            "target_full": load_image(target_path),
            "cleanup_masks": item.get("cleanup_masks", {}),
        }


class DatasetItemBatchBuilder:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.downsample_256 = BicubicDownSample(factor=4)
        self.hair_mask_extractor = HairMaskExtractorV5(device=self.device, dilate_erosion=args.smooth)
        self.parsing_helper = FaceParsingHelperV5(parse_size=args.ear_parse_size)
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
            "target_ear_cover_overlap": args.target_ear_cover_overlap,
            "min_target_visible_overlap": args.min_target_visible_overlap,
            "earring_align_max_shift": args.earring_align_max_shift,
        }
        accepted = inspect.signature(EarAnchoredQueryBuilder.__init__).parameters
        query_builder_kwargs = {
            key: value for key, value in query_builder_kwargs.items() if key in accepted
        }
        self.query_builder = EarAnchoredQueryBuilder(**query_builder_kwargs)

    def _iter_single_process_batches(self, dataset):
        batch_size = self.args.mask_batch_size
        total_batches = ceil_div(len(dataset), batch_size)
        for start in tqdm(range(0, len(dataset), batch_size), total=total_batches, desc="Build dataset masks"):
            end = min(len(dataset), start + batch_size)
            batch_items = [dataset[idx] for idx in range(start, end)]
            yield {
                "source_path": [item["source_path"] for item in batch_items],
                "shape_path": [item["shape_path"] for item in batch_items],
                "color_path": [item["color_path"] for item in batch_items],
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
        dataset = RenderedPairDataset(experiments, dataset_path, face_gallery_root, self.args.donor_gallery_dir)
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
            shape_paths = batch["shape_path"]
            color_paths = batch["color_path"]
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
            cleanup_masks = {}
            for key in CLEANUP_MASK_KEYS:
                value = batch_cleanup_masks.get(key) if isinstance(batch_cleanup_masks, dict) else None
                if value is None:
                    value = torch.zeros_like(query_info["query_mask"])
                cleanup_masks[key] = value.to(self.device, non_blocking=False).float().clamp(0, 1)

            visible_ear_roi = query_info.get("visible_ear_roi")
            if visible_ear_roi is None:
                visible_ear_roi = query_info["ear_roi"]

            weak_earring = build_weak_earring_masks(
                source_256,
                visible_ear_roi,
                query_info["query_mask"],
                query_info["source_earring_mask"],
                query_info.get("source_hair_mask"),
                source_hair_block_mask,
                source_parsing,
            )
            if self.args.enable_earring_query_recall:
                recall_info = enhance_query_with_earring_recall(
                    query_info["query_mask"],
                    query_info["source_earring_mask"],
                    source_hair_block_mask,
                    weak_earring,
                    recall_dilate=self.args.earring_query_recall_dilate,
                    downward_shift=self.args.earring_query_downward_shift,
                    lower_lobe_weight=self.args.earring_query_lower_lobe_weight,
                    candidate_boost=self.args.earring_query_candidate_boost,
                    block_protect=self.args.earring_query_block_protect,
                )
                query_info.update(recall_info)
                source_hair_block_mask = recall_info["source_hair_block_mask"]
            earring_search_mask = weak_earring["earring_search_mask"]
            clean_source_earring_mask = weak_earring.get(
                "source_earring_clean_mask",
                torch.zeros_like(query_info["source_earring_mask"]),
            )
            candidate_source_earring_mask = weak_earring.get(
                "earring_candidate_mask",
                torch.zeros_like(query_info["source_earring_mask"]),
            )
            source_earring_mask = select_reference_earring_mask(
                clean_source_earring_mask,
                candidate_source_earring_mask,
                query_info["left_ear_roi"],
                query_info["right_ear_roi"],
            )
            align_info = align_earring_reference_to_target(
                source_256,
                source_earring_mask,
                query_info["source_left_ear_mask"],
                query_info["source_right_ear_mask"],
                query_info["target_left_ear_mask"],
                query_info["target_right_ear_mask"],
                query_info["left_ear_roi"],
                query_info["right_ear_roi"],
                max_vertical_shift=self.args.earring_align_max_shift,
                max_horizontal_shift=max(1, self.args.earring_align_max_shift // 2),
                reference_base=target_256,
            )
            earring_confident_mask = align_info["earring_confident_mask"]
            earring_valid_roi = None
            if query_info.get("ear_roi") is not None:
                earring_valid_roi = dilate_mask(resize_mask(query_info["ear_roi"], earring_confident_mask.shape[-2:]), 5)
            if query_info.get("visible_ear_roi") is not None:
                visible_roi = dilate_mask(
                    resize_mask(query_info["visible_ear_roi"], earring_confident_mask.shape[-2:]),
                    9,
                )
                earring_valid_roi = (
                    visible_roi
                    if earring_valid_roi is None
                    else torch.clamp(earring_valid_roi + visible_roi, 0, 1)
                )
            if earring_valid_roi is not None:
                earring_confident_mask = earring_confident_mask * earring_valid_roi
                align_info["earring_confident_mask"] = earring_confident_mask
                align_info["earring_reference"] = (
                    target_256 * (1.0 - earring_confident_mask)
                    + align_info["earring_reference"] * earring_confident_mask
                ).clamp(0, 1)
            earring_highlight_mask = build_earring_highlight_mask(
                align_info["earring_reference"],
                earring_confident_mask,
                query_info["query_mask"],
            )
            revealed_info = build_revealed_skin_mask(
                cleanup_masks,
                target_parsing,
                source_parsing,
                target_hair_d,
                source_hair_d,
                earring_confident_mask,
            )

            batch_size = source_256.size(0)
            dataset_items = []
            for idx in range(batch_size):
                item = {
                    "source_path": source_paths[idx],
                    "shape_path": shape_paths[idx],
                    "color_path": color_paths[idx],
                    "target": target_256[idx].cpu(),
                    "target_mask": target_mask[idx].cpu(),
                    "HT_E": target_hair_e[idx].cpu(),
                    "source_parsing": source_parsing[idx].cpu(),
                    "target_parsing": target_parsing[idx].cpu(),
                    "source_hair_mask": query_info["source_hair_mask"][idx].cpu(),
                    "target_hair_mask": query_info["target_hair_mask"][idx].cpu(),
                    "source_hair_block_mask": source_hair_block_mask[idx].cpu(),
                    "source_earring_mask": source_earring_mask[idx].cpu(),
                    "target_earring_mask": query_info["target_earring_mask"][idx].cpu(),
                    "earring_reference": align_info["earring_reference"][idx].cpu(),
                    "query_mask": query_info["query_mask"][idx].cpu(),
                    "earring_search_mask": earring_search_mask[idx].cpu(),
                    "ear_roi": query_info["ear_roi"][idx].cpu(),
                    "left_ear_roi": query_info["left_ear_roi"][idx].cpu(),
                    "right_ear_roi": query_info["right_ear_roi"][idx].cpu(),
                    "presence_target": query_info["presence_target"][idx].cpu(),
                }
                for key in CLEANUP_MASK_KEYS:
                    item[key] = cleanup_masks[key][idx].cpu()
                for key in PP_EXTRA_MASK_KEYS:
                    if key == "earring_confident_mask":
                        item[key] = earring_confident_mask[idx].cpu()
                    elif key == "earring_highlight_mask":
                        item[key] = earring_highlight_mask[idx].cpu()
                    elif key == "earring_candidate_mask":
                        item[key] = weak_earring["earring_candidate_mask"][idx].cpu()
                    else:
                        item[key] = revealed_info[key][idx].cpu()
                dataset_items.append(item)

            yield dataset_items

            del batch
            del dataset_items
            del source_full, target_full, source_256, target_256
            del source_hair_d, target_hair_d, target_hair_e, target_mask
            del source_parsing, target_parsing, query_info, cleanup_masks
            del weak_earring, earring_search_mask, source_earring_mask, align_info, earring_confident_mask, earring_highlight_mask, revealed_info
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
    model_args.ear_low_alpha = args.ear_low_alpha
    model_args.ear_dilate = args.ear_dilate
    model_args.hair_change_dilate = args.hair_change_dilate
    model_args.earring_expand = args.earring_expand
    model_args.ear_downward_shift = args.ear_downward_shift
    model_args.target_hair_dilate = args.target_hair_dilate
    model_args.source_hair_block_dilate = args.source_hair_block_dilate
    model_args.source_hair_block_strength = args.source_hair_block_strength
    model_args.target_visibility_expand = args.target_visibility_expand
    model_args.max_target_hair_overlap = args.max_target_hair_overlap
    model_args.target_ear_cover_overlap = args.target_ear_cover_overlap
    model_args.min_target_visible_overlap = args.min_target_visible_overlap
    model_args.earring_align_max_shift = args.earring_align_max_shift
    model_args.earring_query_dilate = 3
    model_args.earring_query_boost = 1.0
    model_args.enable_earring_query_recall = args.enable_earring_query_recall
    model_args.earring_query_recall_dilate = args.earring_query_recall_dilate
    model_args.earring_query_downward_shift = args.earring_query_downward_shift
    model_args.earring_query_lower_lobe_weight = args.earring_query_lower_lobe_weight
    model_args.earring_query_candidate_boost = args.earring_query_candidate_boost
    model_args.earring_query_block_protect = args.earring_query_block_protect
    model_args.ear_fine_support_dilate = 3
    model_args.earring_fine_mask_floor = args.earring_fine_mask_floor
    model_args.earring_fine_mask_dilate = args.earring_fine_mask_dilate
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
        chunk_start_part = part_idx
        expected_chunk_parts = ceil_div(len(batch_experiments), args.mask_batch_size)
        done_path = chunk_done_path(args.output, left, right)
        done_part_count = read_chunk_done(done_path) if args.resume_existing_parts else None
        if done_part_count is not None:
            print(f"Skipping completed chunk {left}:{right} from {done_path}")
            part_idx += done_part_count
            left = right
            right = min(len(experiments), right + args.chunk_size)
            continue

        expected_part_paths = [
            dataset_part_path(args.output, idx)
            for idx in range(part_idx, part_idx + expected_chunk_parts)
        ]
        if args.resume_existing_parts and expected_part_paths and all(path.exists() for path in expected_part_paths):
            print(f"Skipping existing chunk {left}:{right} with parts {part_idx}-{part_idx + expected_chunk_parts - 1}")
            part_idx += expected_chunk_parts
            left = right
            right = min(len(experiments), right + args.chunk_size)
            continue

        skipped_count = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_experiments = []
            for item in tqdm(batch_experiments, desc=f"Render chunk {left}:{right}"):
                im1, im2, im3 = item["triplet"]
                try:
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
                    target_path = Path(temp_dir) / item["target_name"]
                    save_image(image, target_path)
                    load_image(args.face_gallery_dir / im1)
                    load_image(target_path)
                    valid_experiments.append(item)
                except Exception as error:
                    if not args.skip_bad_rendered_samples:
                        raise RuntimeError(
                            "Failed to render or read generated PP sample: "
                            f"source={args.face_gallery_dir / im1}, "
                            f"shape={args.donor_gallery_dir / im2}, "
                            f"color={args.donor_gallery_dir / im3}, "
                            f"target={Path(temp_dir) / item['target_name']}"
                        ) from error
                    skipped_count += 1
                    log_failed_sample(args.output, item, "render_or_read", error)
                    print(
                        "Skipped bad PP sample: "
                        f"source={args.face_gallery_dir / im1}, "
                        f"shape={args.donor_gallery_dir / im2}, "
                        f"color={args.donor_gallery_dir / im3}, "
                        f"target={Path(temp_dir) / item['target_name']}, "
                        f"error={type(error).__name__}: {error}"
                    )

            chunk_part_count = 0
            for dataset_items in item_batch_builder.iter_batches(valid_experiments, temp_dir, args.face_gallery_dir):
                part_path = dataset_part_path(args.output, part_idx)
                if args.resume_existing_parts and part_path.exists():
                    print(f"Skipping existing {part_path}")
                else:
                    torch.save(dataset_items, part_path)
                    print(f"Saved {part_path}")
                part_idx += 1
                chunk_part_count += 1

            write_chunk_done(done_path, left, right, chunk_start_part, chunk_part_count, skipped_count)

        left = right
        right = min(len(experiments), right + args.chunk_size)


if __name__ == "__main__":
    parser = build_parser(RESOLVED_USER_CONFIG)
    main(parser.parse_args())
