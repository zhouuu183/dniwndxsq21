import argparse
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
from hair_swap_v53 import HairFastV53 as HairFast, get_parser
from models.ear_modules_v53 import (
    EarAnchoredQueryBuilder,
    FaceParsingHelperV53,
    HairMaskExtractorV53,
    build_weak_earring_masks,
    dilate_mask,
    erode_mask,
    ensure_mask_4d,
)
from utils.bicubic import BicubicDownSample
from utils.image_utils import list_image_files
from utils.train import seed_everything

CLEANUP_MASK_KEYS = ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")
DERIVED_MASK_KEYS = ("revealed_skin_mask", "earring_confident_mask", "earring_highlight_mask")
FACE_SURFACE_LABELS = (1, 2)

# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small_accessory_ffhq"  # "small_accessory_ffhq" or "full_ffhq"

USER_FACE_GALLERY_DIR_SMALL = Path("images/ear")
USER_DONOR_GALLERY_DIR_SMALL = Path("images/FFHQ_short")
USER_OUTPUT_DIR_SMALL = Path("images/pp_dataset_v53_dual_small")
USER_DATASET_SIZE_SMALL = 0  # 0 means use every source image.
USER_CHUNK_SIZE_SMALL = 128
USER_MASK_BATCH_SIZE_SMALL = 8

USER_FACE_GALLERY_DIR_FULL = Path("images/FFHQ")
USER_DONOR_GALLERY_DIR_FULL = Path("images/FFHQ")
USER_OUTPUT_DIR_FULL = Path("images/pp_dataset_v53_dual_full")
USER_DATASET_SIZE_FULL = 10_000
USER_CHUNK_SIZE_FULL = 256
USER_MASK_BATCH_SIZE_FULL = 16

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
USER_EAR_DILATE = 21
USER_HAIR_CHANGE_DILATE = 25
USER_EARRING_EXPAND = 15
USER_EAR_DOWNWARD_SHIFT = 10
USER_TARGET_HAIR_DILATE = 11
USER_SOURCE_HAIR_BLOCK_DILATE = 5
USER_SOURCE_HAIR_BLOCK_STRENGTH = 0.95
USER_TARGET_VISIBILITY_EXPAND = 5
USER_MAX_TARGET_HAIR_OVERLAP = 0.55
USER_EARRING_LOBE_DILATE = 17
USER_EARRING_LOBE_DOWN_SHIFT = 18
USER_EARRING_OUTER_SHIFT = 10
USER_EARRING_QUERY_FLOOR = 0.45
USER_EAR_INJECTION_STRENGTH = 2.5
USER_EAR_INJECTION_MASK_DILATE = 3
USER_EAR_INJECTION_MASK_BOOST = 1.35
USER_EARRING_INJECTION_LATERAL_RATIO = 0.34
USER_EARRING_INJECTION_SEARCH_DILATE = 1
USER_EARRING_INJECTION_MIN_HIGH = 0.010
USER_EARRING_INJECTION_MIN_CHROMA = 0.026
USER_EARRING_INJECTION_MIN_CONTRAST = 0.014
USER_EARRING_INJECTION_MIN_BRIGHT = 0.012
USER_EARRING_INJECTION_MIN_DARK = 0.014
USER_EARRING_INJECTION_MIN_VOTES = 2
USER_EARRING_INJECTION_SKIN_MAX_CHROMA = 0.16
USER_EARRING_INJECTION_SKIN_MAX_HIGH = 0.026
USER_EARRING_INJECTION_SKIN_MAX_CONTRAST = 0.026
USER_EARRING_INJECTION_FACE_REJECT_DILATE = 1
USER_EARRING_INJECTION_FACE_REJECT_STRENGTH = 0.65
USER_EARRING_INJECTION_PARSER_MAX_AREA = 260.0
USER_EARRING_INJECTION_SEED_MAX_AREA_FRAC = 0.012
USER_EARRING_INJECTION_STRICT_MIN_HIGH = 0.018
USER_EARRING_INJECTION_STRICT_MIN_CHROMA = 0.045
USER_EARRING_INJECTION_STRICT_MIN_CONTRAST = 0.024
USER_EARRING_INJECTION_STRICT_MIN_BRIGHT = 0.022
USER_EARRING_INJECTION_STRICT_MIN_DARK = 0.022
USER_EARRING_INJECTION_SEED_DILATE = 1
USER_EARRING_STRICT_CORE_ENABLE = True
USER_EARRING_CORE_MIN_HIGH = 0.014
USER_EARRING_CORE_MIN_CHROMA = 0.038
USER_EARRING_CORE_MIN_CONTRAST = 0.018
USER_EARRING_CORE_DILATE = 3
USER_EARRING_RECALL_QUERY_DILATE = 5
USER_EARRING_RAW_QUERY_RECALL_WEIGHT = 0.35
USER_EARRING_WEAK_RECALL_WEIGHT = 0.55
USER_EARRING_FALLBACK_RECALL_WEIGHT = 0.45
USER_EARRING_SAFE_RECALL_WEIGHT = 0.45
USER_EARRING_CORE_FLOOR_WEIGHT = 0.35
USER_EARRING_PARSER_SUPPORT_DILATE = 5
USER_EARRING_PARSER_KEEP_DILATE = 3
USER_EARRING_SAFE_ROI_DILATE = 7
USER_EARRING_DARK_REJECT_MAX_GRAY = 0.22
USER_EARRING_DARK_REJECT_MAX_CHROMA = 0.07
USER_EARRING_DARK_REJECT_MAX_HIGH = 0.018
USER_EARRING_DARK_REJECT_DILATE = 3
USER_EARRING_HIGHLIGHT_SUPPORT_DILATE = 5
USER_EARRING_HIGHLIGHT_KEEP_DILATE = 3
USER_EARRING_RAW_HINT_RESIDUAL = 0.12
USER_EARRING_WEAK_SUPPORT_DILATE = 5
USER_EARRING_WEAK_OBJECT_WEIGHT = 0.35
USER_EARRING_GUARDED_COMPOSITE = True
USER_EARRING_COMPOSITE_EXCLUDE_FACE = True
USER_EARRING_COMPOSITE_FACE_KEEP_DILATE = 3
USER_EARRING_COMPOSITE_MAX_AREA_FRAC = 0.012
USER_EARRING_COMPOSITE_EXCLUDE_TARGET_HAIR = False
USER_EARRING_COMPOSITE_EXCLUDE_HAIR_BLOCK = False
USER_EARRING_COMPOSITE_RESTRICT_VISIBLE_ROI = False
USER_EARRING_COMPOSITE_SEED_DILATE = 2
USER_EARRING_COMPOSITE_SEED_MAX_AREA_FRAC = 0.018
USER_EARRING_COMPOSITE_SEED_PARSER_MAX_AREA = 520.0
USER_EARRING_COMPOSITE_SEED_FACE_REJECT_STRENGTH = 0.35
USER_EARRING_COMPOSITE_SEED_USE_SEARCH_ROI = False
USER_EARRING_PRIOR_REFERENCE_DILATE = 1
USER_EARRING_PRIOR_QUERY_DILATE = 5
USER_EARRING_OUTPUT_GUARD = True
USER_EARRING_OUTPUT_GUARD_DILATE = 21
USER_EARRING_OUTPUT_GUARD_BLUR = 11
USER_EARRING_OUTPUT_GUARD_SIGMA = 3.0
USER_EARRING_OUTPUT_GUARD_MIN_AREA = 4.0
USER_EARRING_OUTPUT_GUARD_MAX_AREA_FRAC = 0.035
USER_EARRING_OUTPUT_GUARD_FALLBACK_MAX_AREA_FRAC = 0.055
USER_FACE_OUTPUT_GUARD = True
USER_FACE_OUTPUT_GUARD_STRENGTH = 0.85
USER_FACE_OUTPUT_GUARD_BLUR = 13
USER_FACE_OUTPUT_GUARD_SIGMA = 4.0
USER_FACE_OUTPUT_GUARD_MIN_AREA = 128.0
USER_FACE_OUTPUT_GUARD_EXCLUDE_EARRING_DILATE = 9
USER_EARRING_COMPOSITE_STRENGTH = 0.85
USER_EARRING_COMPOSITE_FEATHER = 3
USER_EARRING_COMPOSITE_SIGMA = 1.2
USER_EARRING_COMPOSITE_STRICT_MASK = True
USER_EARRING_COMPOSITE_MIN_HIGH = 0.018
USER_EARRING_COMPOSITE_MIN_CHROMA = 0.045
USER_EARRING_COMPOSITE_MIN_CONTRAST = 0.022
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
    "earring_lobe_dilate": USER_EARRING_LOBE_DILATE,
    "earring_lobe_down_shift": USER_EARRING_LOBE_DOWN_SHIFT,
    "earring_outer_shift": USER_EARRING_OUTER_SHIFT,
    "earring_query_floor": USER_EARRING_QUERY_FLOOR,
    "ear_injection_strength": USER_EAR_INJECTION_STRENGTH,
    "ear_injection_mask_dilate": USER_EAR_INJECTION_MASK_DILATE,
    "ear_injection_mask_boost": USER_EAR_INJECTION_MASK_BOOST,
    "earring_injection_lateral_ratio": USER_EARRING_INJECTION_LATERAL_RATIO,
    "earring_injection_search_dilate": USER_EARRING_INJECTION_SEARCH_DILATE,
    "earring_injection_min_high": USER_EARRING_INJECTION_MIN_HIGH,
    "earring_injection_min_chroma": USER_EARRING_INJECTION_MIN_CHROMA,
    "earring_injection_min_contrast": USER_EARRING_INJECTION_MIN_CONTRAST,
    "earring_injection_min_bright": USER_EARRING_INJECTION_MIN_BRIGHT,
    "earring_injection_min_dark": USER_EARRING_INJECTION_MIN_DARK,
    "earring_injection_min_votes": USER_EARRING_INJECTION_MIN_VOTES,
    "earring_injection_skin_max_chroma": USER_EARRING_INJECTION_SKIN_MAX_CHROMA,
    "earring_injection_skin_max_high": USER_EARRING_INJECTION_SKIN_MAX_HIGH,
    "earring_injection_skin_max_contrast": USER_EARRING_INJECTION_SKIN_MAX_CONTRAST,
    "earring_injection_face_reject_dilate": USER_EARRING_INJECTION_FACE_REJECT_DILATE,
    "earring_injection_face_reject_strength": USER_EARRING_INJECTION_FACE_REJECT_STRENGTH,
    "earring_injection_parser_max_area": USER_EARRING_INJECTION_PARSER_MAX_AREA,
    "earring_injection_seed_max_area_frac": USER_EARRING_INJECTION_SEED_MAX_AREA_FRAC,
    "earring_injection_strict_min_high": USER_EARRING_INJECTION_STRICT_MIN_HIGH,
    "earring_injection_strict_min_chroma": USER_EARRING_INJECTION_STRICT_MIN_CHROMA,
    "earring_injection_strict_min_contrast": USER_EARRING_INJECTION_STRICT_MIN_CONTRAST,
    "earring_injection_strict_min_bright": USER_EARRING_INJECTION_STRICT_MIN_BRIGHT,
    "earring_injection_strict_min_dark": USER_EARRING_INJECTION_STRICT_MIN_DARK,
    "earring_injection_seed_dilate": USER_EARRING_INJECTION_SEED_DILATE,
    "earring_strict_core_enable": USER_EARRING_STRICT_CORE_ENABLE,
    "earring_core_min_high": USER_EARRING_CORE_MIN_HIGH,
    "earring_core_min_chroma": USER_EARRING_CORE_MIN_CHROMA,
    "earring_core_min_contrast": USER_EARRING_CORE_MIN_CONTRAST,
    "earring_core_dilate": USER_EARRING_CORE_DILATE,
    "earring_recall_query_dilate": USER_EARRING_RECALL_QUERY_DILATE,
    "earring_raw_query_recall_weight": USER_EARRING_RAW_QUERY_RECALL_WEIGHT,
    "earring_weak_recall_weight": USER_EARRING_WEAK_RECALL_WEIGHT,
    "earring_fallback_recall_weight": USER_EARRING_FALLBACK_RECALL_WEIGHT,
    "earring_safe_recall_weight": USER_EARRING_SAFE_RECALL_WEIGHT,
    "earring_core_floor_weight": USER_EARRING_CORE_FLOOR_WEIGHT,
    "earring_parser_support_dilate": USER_EARRING_PARSER_SUPPORT_DILATE,
    "earring_parser_keep_dilate": USER_EARRING_PARSER_KEEP_DILATE,
    "earring_safe_roi_dilate": USER_EARRING_SAFE_ROI_DILATE,
    "earring_dark_reject_max_gray": USER_EARRING_DARK_REJECT_MAX_GRAY,
    "earring_dark_reject_max_chroma": USER_EARRING_DARK_REJECT_MAX_CHROMA,
    "earring_dark_reject_max_high": USER_EARRING_DARK_REJECT_MAX_HIGH,
    "earring_dark_reject_dilate": USER_EARRING_DARK_REJECT_DILATE,
    "earring_highlight_support_dilate": USER_EARRING_HIGHLIGHT_SUPPORT_DILATE,
    "earring_highlight_keep_dilate": USER_EARRING_HIGHLIGHT_KEEP_DILATE,
    "earring_raw_hint_residual": USER_EARRING_RAW_HINT_RESIDUAL,
    "earring_weak_support_dilate": USER_EARRING_WEAK_SUPPORT_DILATE,
    "earring_weak_object_weight": USER_EARRING_WEAK_OBJECT_WEIGHT,
    "earring_guarded_composite": USER_EARRING_GUARDED_COMPOSITE,
    "earring_composite_exclude_face": USER_EARRING_COMPOSITE_EXCLUDE_FACE,
    "earring_composite_face_keep_dilate": USER_EARRING_COMPOSITE_FACE_KEEP_DILATE,
    "earring_composite_max_area_frac": USER_EARRING_COMPOSITE_MAX_AREA_FRAC,
    "earring_composite_exclude_target_hair": USER_EARRING_COMPOSITE_EXCLUDE_TARGET_HAIR,
    "earring_composite_restrict_visible_roi": USER_EARRING_COMPOSITE_RESTRICT_VISIBLE_ROI,
    "earring_composite_seed_dilate": USER_EARRING_COMPOSITE_SEED_DILATE,
    "earring_composite_seed_max_area_frac": USER_EARRING_COMPOSITE_SEED_MAX_AREA_FRAC,
    "earring_composite_seed_parser_max_area": USER_EARRING_COMPOSITE_SEED_PARSER_MAX_AREA,
    "earring_composite_seed_face_reject_strength": USER_EARRING_COMPOSITE_SEED_FACE_REJECT_STRENGTH,
    "earring_composite_seed_use_search_roi": USER_EARRING_COMPOSITE_SEED_USE_SEARCH_ROI,
    "earring_prior_reference_dilate": USER_EARRING_PRIOR_REFERENCE_DILATE,
    "earring_prior_query_dilate": USER_EARRING_PRIOR_QUERY_DILATE,
    "earring_output_guard": USER_EARRING_OUTPUT_GUARD,
    "earring_output_guard_dilate": USER_EARRING_OUTPUT_GUARD_DILATE,
    "earring_output_guard_blur": USER_EARRING_OUTPUT_GUARD_BLUR,
    "earring_output_guard_sigma": USER_EARRING_OUTPUT_GUARD_SIGMA,
    "earring_output_guard_min_area": USER_EARRING_OUTPUT_GUARD_MIN_AREA,
    "earring_output_guard_max_area_frac": USER_EARRING_OUTPUT_GUARD_MAX_AREA_FRAC,
    "earring_output_guard_fallback_max_area_frac": USER_EARRING_OUTPUT_GUARD_FALLBACK_MAX_AREA_FRAC,
    "face_output_guard": USER_FACE_OUTPUT_GUARD,
    "face_output_guard_strength": USER_FACE_OUTPUT_GUARD_STRENGTH,
    "face_output_guard_blur": USER_FACE_OUTPUT_GUARD_BLUR,
    "face_output_guard_sigma": USER_FACE_OUTPUT_GUARD_SIGMA,
    "face_output_guard_min_area": USER_FACE_OUTPUT_GUARD_MIN_AREA,
    "face_output_guard_exclude_earring_dilate": USER_FACE_OUTPUT_GUARD_EXCLUDE_EARRING_DILATE,
    "earring_composite_strength": USER_EARRING_COMPOSITE_STRENGTH,
    "earring_composite_feather": USER_EARRING_COMPOSITE_FEATHER,
    "earring_composite_sigma": USER_EARRING_COMPOSITE_SIGMA,
    "earring_composite_strict_mask": USER_EARRING_COMPOSITE_STRICT_MASK,
    "earring_composite_exclude_hair_block": USER_EARRING_COMPOSITE_EXCLUDE_HAIR_BLOCK,
    "earring_composite_min_high": USER_EARRING_COMPOSITE_MIN_HIGH,
    "earring_composite_min_chroma": USER_EARRING_COMPOSITE_MIN_CHROMA,
    "earring_composite_min_contrast": USER_EARRING_COMPOSITE_MIN_CONTRAST,
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


def parsing_label_mask(parsing: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    parsing = ensure_mask_4d(parsing).long()
    mask = torch.zeros_like(parsing, dtype=torch.bool)
    for label in labels:
        mask |= parsing == label
    return mask.float()


def combine_cleanup_masks(cleanup_masks: dict[str, torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
    masks = []
    for key in CLEANUP_MASK_KEYS:
        value = cleanup_masks.get(key)
        if value is not None:
            masks.append(ensure_mask_4d(value).float())
    if not masks:
        return torch.zeros_like(fallback)
    return torch.stack(masks, dim=0).amax(dim=0).clamp(0, 1)


def get_cleanup_mask(cleanup_masks: dict[str, torch.Tensor], key: str, fallback: torch.Tensor) -> torch.Tensor:
    value = cleanup_masks.get(key)
    if value is None:
        return torch.zeros_like(fallback)
    return ensure_mask_4d(value).float().clamp(0, 1)


def build_revealed_skin_mask(
    cleanup_masks: dict[str, torch.Tensor],
    target_parsing: torch.Tensor,
    target_hair_mask: torch.Tensor,
    target_earring_mask: torch.Tensor,
) -> torch.Tensor:
    cleanup_mask = combine_cleanup_masks(cleanup_masks, target_hair_mask)
    target_face_surface = parsing_label_mask(target_parsing, FACE_SURFACE_LABELS)
    face_cleanup = get_cleanup_mask(cleanup_masks, "M_remove_face", cleanup_mask)
    halo_cleanup = get_cleanup_mask(cleanup_masks, "M_remove_halo", cleanup_mask)
    cleanup_inner_edge = cleanup_mask * (1 - erode_mask(cleanup_mask, 15)).clamp(0, 1)
    focused_cleanup = torch.clamp(
        face_cleanup + 0.65 * cleanup_inner_edge + 0.35 * halo_cleanup * target_face_surface,
        0,
        1,
    )
    target_earring_mask = dilate_mask(target_earring_mask, 5)
    return (
        focused_cleanup
        * target_face_surface
        * (1 - ensure_mask_4d(target_hair_mask).float()).clamp(0, 1)
        * (1 - target_earring_mask).clamp(0, 1)
    ).clamp(0, 1)


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
    parser = argparse.ArgumentParser(description="PP dataset generator v53")
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
    parser.add_argument("--earring_lobe_dilate", type=int, default=defaults["earring_lobe_dilate"])
    parser.add_argument("--earring_lobe_down_shift", type=int, default=defaults["earring_lobe_down_shift"])
    parser.add_argument("--earring_outer_shift", type=int, default=defaults["earring_outer_shift"])
    parser.add_argument("--earring_query_floor", type=float, default=defaults["earring_query_floor"])
    parser.add_argument("--ear_injection_strength", type=float, default=defaults["ear_injection_strength"])
    parser.add_argument("--ear_injection_mask_dilate", type=int, default=defaults["ear_injection_mask_dilate"])
    parser.add_argument("--ear_injection_mask_boost", type=float, default=defaults["ear_injection_mask_boost"])
    parser.add_argument("--earring_injection_lateral_ratio", type=float, default=defaults["earring_injection_lateral_ratio"])
    parser.add_argument("--earring_injection_search_dilate", type=int, default=defaults["earring_injection_search_dilate"])
    parser.add_argument("--earring_injection_min_high", type=float, default=defaults["earring_injection_min_high"])
    parser.add_argument("--earring_injection_min_chroma", type=float, default=defaults["earring_injection_min_chroma"])
    parser.add_argument("--earring_injection_min_contrast", type=float, default=defaults["earring_injection_min_contrast"])
    parser.add_argument("--earring_injection_min_bright", type=float, default=defaults["earring_injection_min_bright"])
    parser.add_argument("--earring_injection_min_dark", type=float, default=defaults["earring_injection_min_dark"])
    parser.add_argument("--earring_injection_min_votes", type=int, default=defaults["earring_injection_min_votes"])
    parser.add_argument("--earring_injection_skin_max_chroma", type=float, default=defaults["earring_injection_skin_max_chroma"])
    parser.add_argument("--earring_injection_skin_max_high", type=float, default=defaults["earring_injection_skin_max_high"])
    parser.add_argument("--earring_injection_skin_max_contrast", type=float, default=defaults["earring_injection_skin_max_contrast"])
    parser.add_argument("--earring_injection_face_reject_dilate", type=int, default=defaults["earring_injection_face_reject_dilate"])
    parser.add_argument("--earring_injection_face_reject_strength", type=float, default=defaults["earring_injection_face_reject_strength"])
    parser.add_argument("--earring_injection_parser_max_area", type=float, default=defaults["earring_injection_parser_max_area"])
    parser.add_argument("--earring_injection_seed_max_area_frac", type=float, default=defaults["earring_injection_seed_max_area_frac"])
    parser.add_argument("--earring_injection_strict_min_high", type=float, default=defaults["earring_injection_strict_min_high"])
    parser.add_argument("--earring_injection_strict_min_chroma", type=float, default=defaults["earring_injection_strict_min_chroma"])
    parser.add_argument("--earring_injection_strict_min_contrast", type=float, default=defaults["earring_injection_strict_min_contrast"])
    parser.add_argument("--earring_injection_strict_min_bright", type=float, default=defaults["earring_injection_strict_min_bright"])
    parser.add_argument("--earring_injection_strict_min_dark", type=float, default=defaults["earring_injection_strict_min_dark"])
    parser.add_argument("--earring_injection_seed_dilate", type=int, default=defaults["earring_injection_seed_dilate"])
    parser.add_argument("--earring_strict_core_enable", type=str2bool, default=defaults["earring_strict_core_enable"])
    parser.add_argument("--earring_core_min_high", type=float, default=defaults["earring_core_min_high"])
    parser.add_argument("--earring_core_min_chroma", type=float, default=defaults["earring_core_min_chroma"])
    parser.add_argument("--earring_core_min_contrast", type=float, default=defaults["earring_core_min_contrast"])
    parser.add_argument("--earring_core_dilate", type=int, default=defaults["earring_core_dilate"])
    parser.add_argument("--earring_recall_query_dilate", type=int, default=defaults["earring_recall_query_dilate"])
    parser.add_argument("--earring_raw_query_recall_weight", type=float, default=defaults["earring_raw_query_recall_weight"])
    parser.add_argument("--earring_weak_recall_weight", type=float, default=defaults["earring_weak_recall_weight"])
    parser.add_argument("--earring_fallback_recall_weight", type=float, default=defaults["earring_fallback_recall_weight"])
    parser.add_argument("--earring_safe_recall_weight", type=float, default=defaults["earring_safe_recall_weight"])
    parser.add_argument("--earring_core_floor_weight", type=float, default=defaults["earring_core_floor_weight"])
    parser.add_argument("--earring_parser_support_dilate", type=int, default=defaults["earring_parser_support_dilate"])
    parser.add_argument("--earring_parser_keep_dilate", type=int, default=defaults["earring_parser_keep_dilate"])
    parser.add_argument("--earring_safe_roi_dilate", type=int, default=defaults["earring_safe_roi_dilate"])
    parser.add_argument("--earring_dark_reject_max_gray", type=float, default=defaults["earring_dark_reject_max_gray"])
    parser.add_argument("--earring_dark_reject_max_chroma", type=float, default=defaults["earring_dark_reject_max_chroma"])
    parser.add_argument("--earring_dark_reject_max_high", type=float, default=defaults["earring_dark_reject_max_high"])
    parser.add_argument("--earring_dark_reject_dilate", type=int, default=defaults["earring_dark_reject_dilate"])
    parser.add_argument("--earring_highlight_support_dilate", type=int, default=defaults["earring_highlight_support_dilate"])
    parser.add_argument("--earring_highlight_keep_dilate", type=int, default=defaults["earring_highlight_keep_dilate"])
    parser.add_argument("--earring_raw_hint_residual", type=float, default=defaults["earring_raw_hint_residual"])
    parser.add_argument("--earring_weak_support_dilate", type=int, default=defaults["earring_weak_support_dilate"])
    parser.add_argument("--earring_weak_object_weight", type=float, default=defaults["earring_weak_object_weight"])
    parser.add_argument("--earring_guarded_composite", type=str2bool, default=defaults["earring_guarded_composite"])
    parser.add_argument("--earring_composite_exclude_face", type=str2bool, default=defaults["earring_composite_exclude_face"])
    parser.add_argument("--earring_composite_face_keep_dilate", type=int, default=defaults["earring_composite_face_keep_dilate"])
    parser.add_argument("--earring_composite_max_area_frac", type=float, default=defaults["earring_composite_max_area_frac"])
    parser.add_argument("--earring_composite_exclude_target_hair", type=str2bool, default=defaults["earring_composite_exclude_target_hair"])
    parser.add_argument("--earring_composite_restrict_visible_roi", type=str2bool, default=defaults["earring_composite_restrict_visible_roi"])
    parser.add_argument("--earring_composite_seed_dilate", type=int, default=defaults["earring_composite_seed_dilate"])
    parser.add_argument("--earring_composite_seed_max_area_frac", type=float, default=defaults["earring_composite_seed_max_area_frac"])
    parser.add_argument("--earring_composite_seed_parser_max_area", type=float, default=defaults["earring_composite_seed_parser_max_area"])
    parser.add_argument("--earring_composite_seed_face_reject_strength", type=float, default=defaults["earring_composite_seed_face_reject_strength"])
    parser.add_argument("--earring_composite_seed_use_search_roi", type=str2bool, default=defaults["earring_composite_seed_use_search_roi"])
    parser.add_argument("--earring_prior_reference_dilate", type=int, default=defaults["earring_prior_reference_dilate"])
    parser.add_argument("--earring_prior_query_dilate", type=int, default=defaults["earring_prior_query_dilate"])
    parser.add_argument("--earring_output_guard", type=str2bool, default=defaults["earring_output_guard"])
    parser.add_argument("--earring_output_guard_dilate", type=int, default=defaults["earring_output_guard_dilate"])
    parser.add_argument("--earring_output_guard_blur", type=int, default=defaults["earring_output_guard_blur"])
    parser.add_argument("--earring_output_guard_sigma", type=float, default=defaults["earring_output_guard_sigma"])
    parser.add_argument("--earring_output_guard_min_area", type=float, default=defaults["earring_output_guard_min_area"])
    parser.add_argument("--earring_output_guard_max_area_frac", type=float, default=defaults["earring_output_guard_max_area_frac"])
    parser.add_argument("--earring_output_guard_fallback_max_area_frac", type=float, default=defaults["earring_output_guard_fallback_max_area_frac"])
    parser.add_argument("--face_output_guard", type=str2bool, default=defaults["face_output_guard"])
    parser.add_argument("--face_output_guard_strength", type=float, default=defaults["face_output_guard_strength"])
    parser.add_argument("--face_output_guard_blur", type=int, default=defaults["face_output_guard_blur"])
    parser.add_argument("--face_output_guard_sigma", type=float, default=defaults["face_output_guard_sigma"])
    parser.add_argument("--face_output_guard_min_area", type=float, default=defaults["face_output_guard_min_area"])
    parser.add_argument("--face_output_guard_exclude_earring_dilate", type=int, default=defaults["face_output_guard_exclude_earring_dilate"])
    parser.add_argument("--earring_composite_strength", type=float, default=defaults["earring_composite_strength"])
    parser.add_argument("--earring_composite_feather", type=int, default=defaults["earring_composite_feather"])
    parser.add_argument("--earring_composite_sigma", type=float, default=defaults["earring_composite_sigma"])
    parser.add_argument("--earring_composite_strict_mask", type=str2bool, default=defaults["earring_composite_strict_mask"])
    parser.add_argument("--earring_composite_exclude_hair_block", type=str2bool, default=defaults["earring_composite_exclude_hair_block"])
    parser.add_argument("--earring_composite_min_high", type=float, default=defaults["earring_composite_min_high"])
    parser.add_argument("--earring_composite_min_chroma", type=float, default=defaults["earring_composite_min_chroma"])
    parser.add_argument("--earring_composite_min_contrast", type=float, default=defaults["earring_composite_min_contrast"])
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


def build_experiment_plan(face_images, donor_images, resolved_size):
    face_replace = resolved_size > len(face_images)
    donor_replace = (2 * resolved_size) > len(donor_images)
    face = np.random.choice(face_images, size=resolved_size, replace=face_replace)
    shape, color = np.array_split(
        np.random.choice(donor_images, size=2 * resolved_size, replace=donor_replace),
        2,
    )

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
    return experiments


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
        self.hair_mask_extractor = HairMaskExtractorV53(device=self.device, dilate_erosion=args.smooth)
        self.parsing_helper = FaceParsingHelperV53(parse_size=args.ear_parse_size)
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
            "earring_lobe_dilate": args.earring_lobe_dilate,
            "earring_lobe_down_shift": args.earring_lobe_down_shift,
            "earring_outer_shift": args.earring_outer_shift,
            "earring_query_floor": args.earring_query_floor,
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
            cleanup_masks = {}
            for key in CLEANUP_MASK_KEYS:
                value = batch_cleanup_masks.get(key) if isinstance(batch_cleanup_masks, dict) else None
                if value is None:
                    value = torch.zeros_like(query_info["query_mask"])
                cleanup_masks[key] = value.to(self.device, non_blocking=False).float().clamp(0, 1)
            revealed_skin_mask = build_revealed_skin_mask(
                cleanup_masks,
                target_parsing,
                query_info["target_hair_mask"],
                query_info["target_earring_mask"],
            )
            earring_masks = build_weak_earring_masks(
                source_256,
                query_info.get("visible_ear_roi", query_info["ear_roi"]),
                query_info["query_mask"],
                query_info["source_earring_mask"],
                query_info.get("source_hair_mask"),
                source_hair_block_mask,
            )
            source_earring_mask = torch.clamp(
                query_info["source_earring_mask"] + earring_masks["earring_confident_mask"],
                0,
                1,
            )

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
                    "source_earring_mask": source_earring_mask[idx].cpu(),
                    "target_earring_mask": query_info["target_earring_mask"][idx].cpu(),
                    "query_mask": query_info["query_mask"][idx].cpu(),
                    "ear_roi": query_info["ear_roi"][idx].cpu(),
                    "left_ear_roi": query_info["left_ear_roi"][idx].cpu(),
                    "right_ear_roi": query_info["right_ear_roi"][idx].cpu(),
                    "presence_target": query_info["presence_target"][idx].cpu(),
                    "revealed_skin_mask": revealed_skin_mask[idx].cpu(),
                    "earring_confident_mask": earring_masks["earring_confident_mask"][idx].cpu(),
                    "earring_highlight_mask": earring_masks["earring_highlight_mask"][idx].cpu(),
                }
                for key in CLEANUP_MASK_KEYS:
                    item[key] = cleanup_masks[key][idx].cpu()
                dataset_items.append(item)

            yield dataset_items

            del batch
            del dataset_items
            del source_full, target_full, source_256, target_256
            del source_hair_d, target_hair_d, target_hair_e, target_mask
            del source_parsing, target_parsing, query_info, cleanup_masks, earring_masks
            del source_earring_mask
            del revealed_skin_mask
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
    for key in (
        "ear_parse_size",
        "ear_dilate",
        "hair_change_dilate",
        "earring_expand",
        "ear_downward_shift",
        "target_hair_dilate",
        "source_hair_block_dilate",
        "source_hair_block_strength",
        "target_visibility_expand",
        "max_target_hair_overlap",
        "earring_lobe_dilate",
        "earring_lobe_down_shift",
        "earring_outer_shift",
        "earring_query_floor",
        "ear_injection_strength",
        "ear_injection_mask_dilate",
        "ear_injection_mask_boost",
        "earring_injection_lateral_ratio",
        "earring_injection_search_dilate",
        "earring_injection_min_high",
        "earring_injection_min_chroma",
        "earring_injection_min_contrast",
        "earring_injection_min_bright",
        "earring_injection_min_dark",
        "earring_injection_min_votes",
        "earring_injection_skin_max_chroma",
        "earring_injection_skin_max_high",
        "earring_injection_skin_max_contrast",
        "earring_injection_face_reject_dilate",
        "earring_injection_face_reject_strength",
        "earring_injection_parser_max_area",
        "earring_injection_seed_max_area_frac",
        "earring_injection_strict_min_high",
        "earring_injection_strict_min_chroma",
        "earring_injection_strict_min_contrast",
        "earring_injection_strict_min_bright",
        "earring_injection_strict_min_dark",
        "earring_injection_seed_dilate",
        "earring_strict_core_enable",
        "earring_core_min_high",
        "earring_core_min_chroma",
        "earring_core_min_contrast",
        "earring_core_dilate",
        "earring_recall_query_dilate",
        "earring_raw_query_recall_weight",
        "earring_weak_recall_weight",
        "earring_fallback_recall_weight",
        "earring_safe_recall_weight",
        "earring_core_floor_weight",
        "earring_parser_support_dilate",
        "earring_parser_keep_dilate",
        "earring_safe_roi_dilate",
        "earring_dark_reject_max_gray",
        "earring_dark_reject_max_chroma",
        "earring_dark_reject_max_high",
        "earring_dark_reject_dilate",
        "earring_highlight_support_dilate",
        "earring_highlight_keep_dilate",
        "earring_raw_hint_residual",
        "earring_weak_support_dilate",
        "earring_weak_object_weight",
        "earring_guarded_composite",
        "earring_composite_exclude_face",
        "earring_composite_face_keep_dilate",
        "earring_composite_max_area_frac",
        "earring_composite_exclude_target_hair",
        "earring_composite_restrict_visible_roi",
        "earring_composite_seed_dilate",
        "earring_composite_seed_max_area_frac",
        "earring_composite_seed_parser_max_area",
        "earring_composite_seed_face_reject_strength",
        "earring_composite_seed_use_search_roi",
        "earring_prior_reference_dilate",
        "earring_prior_query_dilate",
        "earring_output_guard",
        "earring_output_guard_dilate",
        "earring_output_guard_blur",
        "earring_output_guard_sigma",
        "earring_output_guard_min_area",
        "earring_output_guard_max_area_frac",
        "earring_output_guard_fallback_max_area_frac",
        "face_output_guard",
        "face_output_guard_strength",
        "face_output_guard_blur",
        "face_output_guard_sigma",
        "face_output_guard_min_area",
        "face_output_guard_exclude_earring_dilate",
        "earring_composite_strength",
        "earring_composite_feather",
        "earring_composite_sigma",
        "earring_composite_strict_mask",
        "earring_composite_exclude_hair_block",
        "earring_composite_min_high",
        "earring_composite_min_chroma",
        "earring_composite_min_contrast",
    ):
        setattr(model_args, key, getattr(args, key))
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

    experiments = build_experiment_plan(face_images, donor_images, resolved_size)
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
