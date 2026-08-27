import argparse
import gzip
import hashlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hair_swap_v6 import HairFastV6 as HairFast, get_parser
from models.ear_modules_v5 import (
    EarAnchoredQueryBuilder,
    FaceParsingHelperV5,
    HairMaskExtractorV5,
    RAW_EAR_SURFACE_LABELS,
    RAW_EARRING,
    RAW_HAIR,
    build_earring_highlight_mask,
    build_earring_search_mask,
    build_earring_write_masks,
    build_source_earring_instance_masks_v5,
    build_strong_earring_candidate,
    compute_earring_hole_mask,
    dilate_mask,
    refine_earring_instances_highres,
    refine_earring_hoops_highres,
    expand_valid_roi_by_completion,
    build_revealed_skin_mask,
    build_weak_earring_masks,
    enhance_query_with_earring_recall,
    assign_components_to_ear_sides,
    parsing_label_mask,
    resize_mask,
)
from models.earring_foreground_v6 import (
    EarringCoordinateSpace,
    EarringNativeInstanceV6,
    align_earring_instance_v6,
    enforce_exclusive_earring_sides_v6,
    extract_source_native_earring_v6,
    retain_single_earring_group_v6,
)
from utils.bicubic import BicubicDownSample
from utils.image_utils import list_image_files
from utils.train import seed_everything

CLEANUP_MASK_KEYS = (
    "M_boundary",
    "M_remove",
    "M_remove_halo",
    "M_remove_tail",
    "M_remove_face",
    "M_remove_neck",
    "M_remove_context",
)
# Schema 34 stores the SATD-rendered image as the direct PP target.  PP then
# decodes that image live with the source, and no decoded-image SATD residual
# is applied a second time.  Older parts are intentionally rejected because
# they represent a different PP input contract.
# Schema 34 also fixes ``source_earring_object_mask`` to SOURCE_NATIVE
# coordinates; ``source_earring_mask`` remains TARGET_CANONICAL.
DATASET_CONFIG_SCHEMA_VERSION = 34
DATASET_POLICY_FILES = (
    "scripts/pp_gen_v6.py",
    "hair_swap_v6.py",
    "models/Alignment.py",
    "models/Alignment_v6.py",
    "models/Blending.py",
    "models/Blending_v6.py",
    "models/SATD_v8.py",
    "models/ear_modules_v5.py",
    "models/earring_foreground_v6.py",
    "models/postprocess_v6.py",
)
PP_EXTRA_MASK_KEYS = (
    "cleanup_inner_edge",
    "revealed_skin_mask",
    "revealed_skin_seam_mask",
    "revealed_skin_blend_mask",
    "source_visible_skin_reference_mask",
    "source_skin_valid_mask",
    "earring_confident_mask",
    "earring_highlight_mask",
    "earring_candidate_mask",
)

# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small_accessory_ffhq"  # "small_accessory_ffhq" or "full_ffhq"

USER_FACE_GALLERY_DIR_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/ear/")
USER_DONOR_GALLERY_DIR_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short/")
# This directory is incompatible with all V5 targets and face compositors.
USER_OUTPUT_DIR_SMALL = Path("images/pp_dataset_v6_direct_satd_100_r17")
# Generate 100 distinct source/shape/colour triplets.  The matching trainer
# reserves 50 of these samples for validation and writes all 50 previews.
USER_DATASET_SIZE_SMALL = 100
# Chunk size now controls checkpoint frequency only.  Render/mask work streams
# one mask batch at a time, so this does not retain a whole chunk in memory.
USER_CHUNK_SIZE_SMALL = 256
USER_MASK_BATCH_SIZE_SMALL = 8
# Dataset samples carry two 1024px RGB tensors plus many sparse masks. A raw
# torch container makes 100 samples need roughly 1.7 GB and can fail on a
# quota-limited training volume. gzip is lossless: it changes only the outer
# file container, not a serialized tensor value.
USER_DATASET_COMPRESSION = "gzip"  # "gzip" or "none"
USER_DATASET_GZIP_LEVEL = 1  # lossless and substantially faster than level 6

USER_FACE_GALLERY_DIR_FULL = Path("/root/shared-nvme/HairFastGAN/images/FFHQ")
USER_DONOR_GALLERY_DIR_FULL = Path("/root/shared-nvme/HairFastGAN/images/FFHQ")
USER_OUTPUT_DIR_FULL = Path("images/pp_dataset_v6_direct_satd_full_r17")
USER_DATASET_SIZE_FULL = 10_000
USER_CHUNK_SIZE_FULL = 256
USER_MASK_BATCH_SIZE_FULL = 16

USER_RANDOM_SEED = 3407
USER_BLENDING_CHECKPOINT = "pretrained_models/Blending/checkpoint.pth"
# PP target construction may deliberately use the pre-v8 blending checkpoint
# above.  The resulting PP dataset records this choice in dataset_config.json;
# it is not an inference default for a newly trained blending checkpoint.
USER_ALLOW_LEGACY_BLENDING_CHECKPOINT_V8 = True
USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = "/data/coding/HairFastGAN/HairFastGAN-main/checkpoints/satd_3000_best.pth"
# Match the inference blend used when SATD itself was trained.  The final
# compositor applies this candidate exactly once instead of amplifying its RGB
# difference after decoding.
USER_SATD_BLEND_V8 = 0.75
USER_DIRECT_SATD_PP_INPUT = True
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0
USER_SATD_HAIR_EXCLUDE_STRENGTH = 1.0
# SATD is applied only as the candidate-minus-author residual in parsed
# background.  These values widen M_remove support without authorizing any
# face, hair, ear, neck, or earring pixel.
USER_SATD_BACKGROUND_CLEANUP_DILATE = 110
USER_SATD_BACKGROUND_HAIR_EDGE_DILATE = 16
USER_SATD_BACKGROUND_HAIR_EDGE_SUPPORT_DILATE = 72
USER_SATD_BACKGROUND_HAIR_EDGE_STRENGTH = 1.0
USER_SATD_BACKGROUND_RESIDUAL_STRENGTH = 1.25
USER_SATD_BACKGROUND_ALPHA_FEATHER = 7

# These values define the authoritative v8 target distribution consumed by PP.
# Changing any of them requires a fresh dataset; the resume fingerprint below
# prevents parts from two colour/mask policies being mixed.
USER_TARGET_HAIR_CLOSE_KERNEL = 9
USER_TARGET_HAIR_HOLE_MAX_AREA_RATIO = 0.003
USER_TARGET_HAIR_TOP_FILL_ONLY = True
USER_TARGET_HAIR_EAR_BRIDGE_RADIUS = 3
USER_HAIR_COLOR_REFERENCE_STRENGTH_V8 = 0.90
USER_HAIR_COLOR_LOW_FREQUENCY_RADIUS_V8 = 15
USER_HAIR_COLOR_LOW_FREQUENCY_SIGMA_V8 = None
USER_HAIR_COLOR_FEATHER_RADIUS_V8 = 5
USER_HAIR_COLOR_SPATIAL_REFERENCE_WEIGHT_V8 = 0.75
USER_HAIR_COLOR_DETAIL_CHROMA_GAIN_V8 = 1.0
USER_DISABLE_REFERENCE_DOMINANT_HAIR_COLOR_V8 = False
USER_BLEND_CHROMA_CORRECT_STRENGTH = 0.0

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
USER_EARRING_OCCLUSION_DILATE = 3
USER_SOURCE_HAIR_BLOCK_DILATE = 8
USER_SOURCE_HAIR_BLOCK_STRENGTH = 0.95
USER_TARGET_VISIBILITY_EXPAND = 5
USER_MAX_TARGET_HAIR_OVERLAP = 0.30
USER_MIN_TARGET_VISIBLE_OVERLAP = 0.10
USER_MIN_TARGET_EAR_AREA = 8.0
USER_EARRING_CHANNEL_DOWN = 32
USER_ENABLE_EARRING_QUERY_RECALL = True
USER_EARRING_QUERY_RECALL_DILATE = 7
USER_EARRING_QUERY_DOWNWARD_SHIFT = 10
USER_EARRING_QUERY_LOWER_LOBE_WEIGHT = 0.20
USER_EARRING_QUERY_CANDIDATE_BOOST = 0.90
USER_EARRING_QUERY_BLOCK_PROTECT = 0.95
USER_EARRING_QUERY_BOOST = 1.0
USER_DISABLE_EARRING_PATH_IF_LOW_CONFIDENCE = True
USER_EARRING_SOURCE_PRESENCE_MIN_AREA = 4.0
USER_EARRING_SEARCH_DOWNWARD_SHIFT = 10
USER_EARRING_SEARCH_DILATE = 7
USER_EARRING_WRITE_MAX_TARGET_HAIR_OVERLAP = 0.30
USER_EARRING_WRITE_SOURCE_BLOCK_DILATE = 3
USER_EARRING_WRITE_DILATE = 3
USER_EARRING_WRITE_CONNECTIVITY_ITERS = 32
USER_EARRING_WRITE_CONNECTIVITY_KERNEL = 5
USER_EARRING_WRITE_BRIDGE_DILATE = 17
USER_EARRING_ANCHOR_VISIBLE_DILATE = 3
# Source identity geometry is preserved by HairFast.  Default to no spatial
# movement so the target compositor cannot leave a second shifted earring.
# Align the source earring attachment to the final visible target earlobe.
# 32px is the 256px-reference limit; it permits a PP-reconstructed lobe to
# move while preventing an accessory from jumping across the face.
USER_EARRING_ALIGN_MAX_SHIFT = 32
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
    "allow_legacy_blending_checkpoint_v8": USER_ALLOW_LEGACY_BLENDING_CHECKPOINT_V8,
    "use_satd_v8": USER_USE_SATD_V8,
    "satd_checkpoint_v8": USER_SATD_CHECKPOINT_V8,
    "satd_blend_v8": USER_SATD_BLEND_V8,
    "satd_boundary_v8": USER_SATD_BOUNDARY_V8,
    "direct_satd_pp_input": USER_DIRECT_SATD_PP_INPUT,
    "eq8_reference_blend_v8": USER_EQ8_REFERENCE_BLEND_V8,
    "satd_hair_exclude_strength": USER_SATD_HAIR_EXCLUDE_STRENGTH,
    "satd_background_cleanup_dilate": USER_SATD_BACKGROUND_CLEANUP_DILATE,
    "satd_background_hair_edge_dilate": USER_SATD_BACKGROUND_HAIR_EDGE_DILATE,
    "satd_background_hair_edge_support_dilate": USER_SATD_BACKGROUND_HAIR_EDGE_SUPPORT_DILATE,
    "satd_background_hair_edge_strength": USER_SATD_BACKGROUND_HAIR_EDGE_STRENGTH,
    "satd_background_residual_strength": USER_SATD_BACKGROUND_RESIDUAL_STRENGTH,
    "satd_background_alpha_feather": USER_SATD_BACKGROUND_ALPHA_FEATHER,
    "target_hair_close_kernel": USER_TARGET_HAIR_CLOSE_KERNEL,
    "target_hair_hole_max_area_ratio": USER_TARGET_HAIR_HOLE_MAX_AREA_RATIO,
    "target_hair_top_fill_only": USER_TARGET_HAIR_TOP_FILL_ONLY,
    "target_hair_ear_bridge_radius": USER_TARGET_HAIR_EAR_BRIDGE_RADIUS,
    "hair_color_reference_strength_v8": USER_HAIR_COLOR_REFERENCE_STRENGTH_V8,
    "hair_color_low_frequency_radius_v8": USER_HAIR_COLOR_LOW_FREQUENCY_RADIUS_V8,
    "hair_color_low_frequency_sigma_v8": USER_HAIR_COLOR_LOW_FREQUENCY_SIGMA_V8,
    "hair_color_feather_radius_v8": USER_HAIR_COLOR_FEATHER_RADIUS_V8,
    "hair_color_spatial_reference_weight_v8": USER_HAIR_COLOR_SPATIAL_REFERENCE_WEIGHT_V8,
    "hair_color_detail_chroma_gain_v8": USER_HAIR_COLOR_DETAIL_CHROMA_GAIN_V8,
    "disable_reference_dominant_hair_color_v8": USER_DISABLE_REFERENCE_DOMINANT_HAIR_COLOR_V8,
    "blend_chroma_correct_strength": USER_BLEND_CHROMA_CORRECT_STRENGTH,
    "chunk_size": ACTIVE_CHUNK_SIZE,
    "mask_batch_size": ACTIVE_MASK_BATCH_SIZE,
    "dataset_compression": USER_DATASET_COMPRESSION,
    "dataset_gzip_level": USER_DATASET_GZIP_LEVEL,
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
    "earring_occlusion_dilate": USER_EARRING_OCCLUSION_DILATE,
    "source_hair_block_dilate": USER_SOURCE_HAIR_BLOCK_DILATE,
    "source_hair_block_strength": USER_SOURCE_HAIR_BLOCK_STRENGTH,
    "target_visibility_expand": USER_TARGET_VISIBILITY_EXPAND,
    "max_target_hair_overlap": USER_MAX_TARGET_HAIR_OVERLAP,
    "min_target_visible_overlap": USER_MIN_TARGET_VISIBLE_OVERLAP,
    "min_target_ear_area": USER_MIN_TARGET_EAR_AREA,
    "earring_channel_down": USER_EARRING_CHANNEL_DOWN,
    "enable_earring_query_recall": USER_ENABLE_EARRING_QUERY_RECALL,
    "earring_query_recall_dilate": USER_EARRING_QUERY_RECALL_DILATE,
    "earring_query_downward_shift": USER_EARRING_QUERY_DOWNWARD_SHIFT,
    "earring_query_lower_lobe_weight": USER_EARRING_QUERY_LOWER_LOBE_WEIGHT,
    "earring_query_candidate_boost": USER_EARRING_QUERY_CANDIDATE_BOOST,
    "earring_query_block_protect": USER_EARRING_QUERY_BLOCK_PROTECT,
    "earring_query_boost": USER_EARRING_QUERY_BOOST,
    "disable_earring_path_if_low_confidence": USER_DISABLE_EARRING_PATH_IF_LOW_CONFIDENCE,
    "earring_source_presence_min_area": USER_EARRING_SOURCE_PRESENCE_MIN_AREA,
    "earring_search_downward_shift": USER_EARRING_SEARCH_DOWNWARD_SHIFT,
    "earring_search_dilate": USER_EARRING_SEARCH_DILATE,
    "earring_write_max_target_hair_overlap": USER_EARRING_WRITE_MAX_TARGET_HAIR_OVERLAP,
    "earring_write_source_block_dilate": USER_EARRING_WRITE_SOURCE_BLOCK_DILATE,
    "earring_write_dilate": USER_EARRING_WRITE_DILATE,
    "earring_write_connectivity_iters": USER_EARRING_WRITE_CONNECTIVITY_ITERS,
    "earring_write_connectivity_kernel": USER_EARRING_WRITE_CONNECTIVITY_KERNEL,
    "earring_write_bridge_dilate": USER_EARRING_WRITE_BRIDGE_DILATE,
    "earring_anchor_visible_dilate": USER_EARRING_ANCHOR_VISIBLE_DILATE,
    "earring_align_max_shift": USER_EARRING_ALIGN_MAX_SHIFT,
    "earring_fine_mask_floor": USER_EARRING_FINE_MASK_FLOOR,
    "earring_fine_mask_dilate": USER_EARRING_FINE_MASK_DILATE,
}


def to_single_mask(mask):
    mask = mask.detach().float().cpu()
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3 and mask.size(0) != 1:
        mask = mask[:1]
    return mask.clamp(0, 1)


def unpack_color_before_pp_stage(result):
    """Validate and unpack BlendingV6's formal pre-PP stage result."""

    if not isinstance(result, dict) or result.get("stage") != "color_before_pp":
        raise RuntimeError(
            "HairFastV6 did not return the requested color_before_pp stage. "
            "Use the matching BlendingV6 implementation before generating PP data."
        )
    image = result.get("color_before_pp", result.get("image"))
    if not torch.is_tensor(image):
        raise RuntimeError("color_before_pp stage is missing its image tensor.")
    if image.ndim == 4:
        if image.size(0) != 1:
            raise RuntimeError("PP generation expects one rendered sample per call.")
        image = image[0]
    if image.ndim != 3 or image.size(0) != 3:
        raise RuntimeError(
            f"Expected color_before_pp RGB [3,H,W], got {tuple(image.shape)}."
        )

    raw_cleanup = result.get("cleanup_masks", {})
    if not isinstance(raw_cleanup, dict):
        raise RuntimeError("color_before_pp cleanup_masks must be a dictionary.")
    fallback = next(
        (
            torch.zeros_like(value)
            for value in raw_cleanup.values()
            if torch.is_tensor(value)
        ),
        None,
    )
    cleanup_masks = {}
    if fallback is not None:
        cleanup_masks = {
            key: to_single_mask(raw_cleanup.get(key, fallback))
            for key in CLEANUP_MASK_KEYS
        }

    pre_reference_color = result.get("pre_reference_color")
    if torch.is_tensor(pre_reference_color) and pre_reference_color.ndim == 4:
        if pre_reference_color.size(0) != 1:
            raise RuntimeError("pre_reference_color must contain one sample.")
        pre_reference_color = pre_reference_color[0]
    target_hair_mask = result.get("target_hair_mask")
    if target_hair_mask is not None:
        target_hair_mask = to_single_mask(target_hair_mask)
        if tuple(target_hair_mask.shape[-2:]) != tuple(image.shape[-2:]):
            target_hair_mask = F.interpolate(
                target_hair_mask.unsqueeze(0),
                size=image.shape[-2:],
                mode="nearest",
            )[0]
    completed_hair_highres = result.get("completed_hair_highres")
    if not torch.is_tensor(completed_hair_highres):
        raise RuntimeError(
            "color_before_pp stage is missing completed_hair_highres. "
            "Use the matching V5 Blending implementation before generating PP data."
        )
    if completed_hair_highres.ndim == 4:
        if completed_hair_highres.size(0) != 1:
            raise RuntimeError("completed_hair_highres must contain one sample.")
        completed_hair_highres = completed_hair_highres[0]
    if completed_hair_highres.ndim != 3 or completed_hair_highres.size(0) != 3:
        raise RuntimeError(
            "completed_hair_highres must be RGB [3,H,W], "
            f"got {tuple(completed_hair_highres.shape)}."
        )
    satd_background_highres = result.get("satd_background_highres")
    if not torch.is_tensor(satd_background_highres):
        raise RuntimeError(
            "color_before_pp stage is missing satd_background_highres. "
            "Use the matching V5 Blending implementation before generating PP data."
        )
    if satd_background_highres.ndim == 4:
        if satd_background_highres.size(0) != 1:
            raise RuntimeError("satd_background_highres must contain one sample.")
        satd_background_highres = satd_background_highres[0]
    if satd_background_highres.ndim != 3 or satd_background_highres.size(0) != 3:
        raise RuntimeError(
            "satd_background_highres must be RGB [3,H,W], "
            f"got {tuple(satd_background_highres.shape)}."
        )
    direct_satd_pp_input = bool(result.get("direct_satd_pp_input", False))
    return (
        image.clamp(0, 1),
        cleanup_masks,
        pre_reference_color,
        target_hair_mask,
        completed_hair_highres.clamp(0, 1),
        satd_background_highres.clamp(0, 1),
        direct_satd_pp_input,
    )


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
    parser = argparse.ArgumentParser(description="PP dataset generator v6")
    parser.add_argument("--dataset_profile", type=str, default=defaults["dataset_profile"])
    parser.add_argument("--face_gallery_dir", type=str2path, default=defaults["face_gallery_dir"])
    parser.add_argument("--donor_gallery_dir", type=str2path, default=defaults["donor_gallery_dir"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--size", type=int, default=defaults["size"])
    parser.add_argument("--output", type=Path, default=defaults["output"])
    parser.add_argument("--blending_checkpoint", type=str, default=defaults["blending_checkpoint"])
    parser.add_argument(
        "--allow_legacy_blending_checkpoint_v8",
        type=str2bool,
        default=defaults["allow_legacy_blending_checkpoint_v8"],
        help="Allow a pre-v8 blending checkpoint only while constructing PP data.",
    )
    parser.add_argument("--use_satd_v8", type=str2bool, default=defaults["use_satd_v8"])
    parser.add_argument(
        "--direct_satd_pp_input",
        type=str2bool,
        default=defaults["direct_satd_pp_input"],
        help="Feed I_satd_blend_256 directly to PP and disable post-decode SATD residuals.",
    )
    parser.add_argument("--satd_checkpoint_v8", type=str, default=defaults["satd_checkpoint_v8"])
    parser.add_argument("--satd_blend_v8", type=float, default=defaults["satd_blend_v8"])
    parser.add_argument("--satd_boundary_v8", type=int, default=defaults["satd_boundary_v8"])
    parser.add_argument("--eq8_reference_blend_v8", type=float, default=defaults["eq8_reference_blend_v8"])
    parser.add_argument("--satd_hair_exclude_strength", type=float, default=defaults["satd_hair_exclude_strength"])
    parser.add_argument("--satd_background_cleanup_dilate", type=int, default=defaults["satd_background_cleanup_dilate"])
    parser.add_argument("--satd_background_hair_edge_dilate", type=int, default=defaults["satd_background_hair_edge_dilate"])
    parser.add_argument("--satd_background_hair_edge_support_dilate", type=int, default=defaults["satd_background_hair_edge_support_dilate"])
    parser.add_argument("--satd_background_hair_edge_strength", type=float, default=defaults["satd_background_hair_edge_strength"])
    parser.add_argument("--satd_background_residual_strength", type=float, default=defaults["satd_background_residual_strength"])
    parser.add_argument("--satd_background_alpha_feather", type=int, default=defaults["satd_background_alpha_feather"])
    parser.add_argument("--target_hair_close_kernel", type=int, default=defaults["target_hair_close_kernel"])
    parser.add_argument("--target_hair_hole_max_area_ratio", type=float, default=defaults["target_hair_hole_max_area_ratio"])
    parser.add_argument("--target_hair_top_fill_only", type=str2bool, default=defaults["target_hair_top_fill_only"])
    parser.add_argument("--target_hair_ear_bridge_radius", type=int, default=defaults["target_hair_ear_bridge_radius"])
    parser.add_argument("--hair_color_reference_strength_v8", type=float, default=defaults["hair_color_reference_strength_v8"])
    parser.add_argument("--hair_color_low_frequency_radius_v8", type=int, default=defaults["hair_color_low_frequency_radius_v8"])
    parser.add_argument("--hair_color_low_frequency_sigma_v8", type=float, default=defaults["hair_color_low_frequency_sigma_v8"])
    parser.add_argument("--hair_color_feather_radius_v8", type=int, default=defaults["hair_color_feather_radius_v8"])
    parser.add_argument("--hair_color_spatial_reference_weight_v8", type=float, default=defaults["hair_color_spatial_reference_weight_v8"])
    parser.add_argument("--hair_color_detail_chroma_gain_v8", type=float, default=defaults["hair_color_detail_chroma_gain_v8"])
    parser.add_argument("--disable_reference_dominant_hair_color_v8", type=str2bool, default=defaults["disable_reference_dominant_hair_color_v8"])
    parser.add_argument("--blend_chroma_correct_strength", type=float, default=defaults["blend_chroma_correct_strength"])
    parser.add_argument("--chunk_size", type=int, default=defaults["chunk_size"])
    parser.add_argument("--mask_batch_size", type=int, default=defaults["mask_batch_size"])
    parser.add_argument(
        "--dataset_compression",
        choices=("gzip", "none"),
        default=defaults["dataset_compression"],
        help="Lossless on-disk container for pp_part_*.dataset files.",
    )
    parser.add_argument(
        "--dataset_gzip_level",
        type=int,
        default=defaults["dataset_gzip_level"],
        help="gzip compression level (1..9); ignored for --dataset_compression none.",
    )
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
    parser.add_argument("--earring_occlusion_dilate", type=int, default=defaults["earring_occlusion_dilate"])
    parser.add_argument("--source_hair_block_dilate", type=int, default=defaults["source_hair_block_dilate"])
    parser.add_argument("--source_hair_block_strength", type=float, default=defaults["source_hair_block_strength"])
    parser.add_argument("--target_visibility_expand", type=int, default=defaults["target_visibility_expand"])
    parser.add_argument("--max_target_hair_overlap", type=float, default=defaults["max_target_hair_overlap"])
    parser.add_argument("--min_target_visible_overlap", type=float, default=defaults["min_target_visible_overlap"])
    parser.add_argument("--min_target_ear_area", type=float, default=defaults["min_target_ear_area"])
    parser.add_argument("--earring_channel_down", type=int, default=defaults["earring_channel_down"])
    parser.add_argument("--enable_earring_query_recall", type=str2bool, default=defaults["enable_earring_query_recall"])
    parser.add_argument("--earring_query_recall_dilate", type=int, default=defaults["earring_query_recall_dilate"])
    parser.add_argument("--earring_query_downward_shift", type=int, default=defaults["earring_query_downward_shift"])
    parser.add_argument("--earring_query_lower_lobe_weight", type=float, default=defaults["earring_query_lower_lobe_weight"])
    parser.add_argument("--earring_query_candidate_boost", type=float, default=defaults["earring_query_candidate_boost"])
    parser.add_argument("--earring_query_block_protect", type=float, default=defaults["earring_query_block_protect"])
    parser.add_argument("--earring_query_boost", type=float, default=defaults["earring_query_boost"])
    parser.add_argument(
        "--disable_earring_path_if_low_confidence",
        type=str2bool,
        default=defaults["disable_earring_path_if_low_confidence"],
    )
    parser.add_argument(
        "--earring_source_presence_min_area",
        type=float,
        default=defaults["earring_source_presence_min_area"],
    )
    parser.add_argument(
        "--earring_search_downward_shift",
        type=int,
        default=defaults["earring_search_downward_shift"],
    )
    parser.add_argument("--earring_search_dilate", type=int, default=defaults["earring_search_dilate"])
    parser.add_argument(
        "--earring_write_max_target_hair_overlap",
        type=float,
        default=defaults["earring_write_max_target_hair_overlap"],
    )
    parser.add_argument(
        "--earring_write_source_block_dilate",
        type=int,
        default=defaults["earring_write_source_block_dilate"],
    )
    parser.add_argument("--earring_write_dilate", type=int, default=defaults["earring_write_dilate"])
    parser.add_argument(
        "--earring_write_connectivity_iters",
        type=int,
        default=defaults["earring_write_connectivity_iters"],
    )
    parser.add_argument(
        "--earring_write_connectivity_kernel",
        type=int,
        default=defaults["earring_write_connectivity_kernel"],
    )
    parser.add_argument(
        "--earring_write_bridge_dilate",
        type=int,
        default=defaults["earring_write_bridge_dilate"],
    )
    parser.add_argument(
        "--earring_anchor_visible_dilate",
        type=int,
        default=defaults["earring_anchor_visible_dilate"],
    )
    parser.add_argument("--earring_align_max_shift", type=int, default=defaults["earring_align_max_shift"])
    parser.add_argument("--earring_fine_mask_floor", type=float, default=defaults["earring_fine_mask_floor"])
    parser.add_argument("--earring_fine_mask_dilate", type=int, default=defaults["earring_fine_mask_dilate"])
    return parser


def load_image(path):
    with Image.open(path) as image:
        return T.functional.to_tensor(image.convert("RGB"))


def png_roundtrip_tensor(image):
    """Match the old save-PNG/reload path without an on-disk round trip."""

    if not torch.is_tensor(image):
        raise TypeError("Expected an RGB tensor for the in-memory PNG round trip.")
    if image.ndim != 3 or image.size(0) != 3:
        raise ValueError(f"Expected RGB [3,H,W], got {tuple(image.shape)}.")
    # torchvision.utils.save_image stores 8-bit PNG values with nearest integer
    # rounding.  Preserve that legacy target distribution while avoiding a
    # synchronous encode, write, decode, and second CPU allocation per sample.
    return (
        image.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0, 1)
        .mul(255)
        .add(0.5)
        .clamp(0, 255)
        .to(torch.uint8)
        .to(torch.float32)
        .div(255)
    )


def png_uint8_tensor(image):
    """Store a high-resolution RGB hand-off with PNG-equivalent precision."""

    if not torch.is_tensor(image):
        raise TypeError("Expected an RGB tensor for 8-bit dataset storage.")
    if image.ndim != 3 or image.size(0) != 3:
        raise ValueError(f"Expected RGB [3,H,W], got {tuple(image.shape)}.")
    return (
        image.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0, 1)
        .mul(255)
        .add(0.5)
        .clamp(0, 255)
        .to(torch.uint8)
    )

def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def count_dataset_parts(total_items, chunk_size, batch_size):
    total_parts = 0
    for start in range(0, total_items, chunk_size):
        total_parts += ceil_div(min(chunk_size, total_items - start), batch_size)
    return total_parts


def save_dataset_part(dataset_items, part_path: Path, compression: str, gzip_level: int) -> None:
    """Atomically write a dataset part without changing any tensor values."""

    part_path = Path(part_path)
    temporary_path = part_path.with_name(f"{part_path.name}.tmp")
    try:
        if compression == "gzip":
            with gzip.open(temporary_path, mode="wb", compresslevel=gzip_level) as handle:
                torch.save(dataset_items, handle)
        elif compression == "none":
            torch.save(dataset_items, temporary_path)
        else:
            raise ValueError(f"Unsupported dataset compression: {compression!r}")
        os.replace(temporary_path, part_path)
    except Exception:
        # A .tmp file belongs only to this failed atomic write. Completed parts
        # are deliberately never touched by this recovery path.
        temporary_path.unlink(missing_ok=True)
        raise


def identity_key(name: str) -> str:
    """Return the per-person key used for source/donor exclusion."""

    # FFHQ_3000 has one image per identity, so the filename stem is the stable
    # identity key.  The key is shared across separate gallery roots, which also
    # prevents a same-named source and donor when both roots point to one folder.
    return Path(str(name)).stem.casefold()


def sample_distinct_triplets(face_images, donor_images, size: int):
    """Sample triplets with three different identity keys per experiment."""

    face_images = list(face_images)
    donor_images = list(donor_images)
    if size <= 0:
        raise ValueError("Triplet sample size must be positive")

    donor_by_identity = {}
    for name in donor_images:
        donor_by_identity.setdefault(identity_key(name), []).append(name)
    if len(donor_by_identity) < 3:
        raise ValueError(
            "The donor gallery must contain at least three distinct identity "
            "keys for source/shape/color separation."
        )

    # The normal V5 dataset mode is one render per source image.  Make that
    # explicit instead of relying on ``choice(..., replace=False)`` so the
    # generation log and the produced dataset cannot silently contain repeated
    # targets when ``--size`` is at or below the source-gallery size.
    if size <= len(face_images):
        sources = np.random.permutation(np.asarray(face_images, dtype=object))[:size]
    else:
        initial = np.random.permutation(np.asarray(face_images, dtype=object))
        extra = np.random.choice(face_images, size=size - len(face_images), replace=True)
        sources = np.concatenate((initial, extra))
    identity_keys = np.array(sorted(donor_by_identity), dtype=object)
    experiments = []
    for source_name in sources:
        source_identity = identity_key(source_name)
        allowed_shape_ids = identity_keys[identity_keys != source_identity]
        shape_identity = str(np.random.choice(allowed_shape_ids))
        allowed_color_ids = allowed_shape_ids[allowed_shape_ids != shape_identity]
        color_identity = str(np.random.choice(allowed_color_ids))
        shape_name = donor_by_identity[shape_identity][
            int(np.random.randint(len(donor_by_identity[shape_identity])))
        ]
        color_name = donor_by_identity[color_identity][
            int(np.random.randint(len(donor_by_identity[color_identity])))
        ]
        experiments.append((str(source_name), str(shape_name), str(color_name)))
    return experiments


def build_dataset_earring_policy_masks(query_info, weak_earring, source_parsing, source_01, args):
    """Apply the inference-time broad-search/narrow-write policy to PP data.

    Presence uses parser/explicit evidence or the same strong visual candidate
    as inference.  Weak recall remains search-only, so a no-earring source
    cannot acquire a synthetic lower-ear restoration target.
    """

    reference = query_info["query_mask"].float()
    parsing = source_parsing
    if tuple(parsing.shape[-2:]) != tuple(reference.shape[-2:]):
        parsing = F.interpolate(parsing.float(), size=reference.shape[-2:], mode="nearest")
    parser_earring = (parsing.long() == RAW_EARRING).to(dtype=reference.dtype)

    strong_info = build_strong_earring_candidate(
        source_01,
        weak_earring.get("earring_candidate_mask", torch.zeros_like(reference)),
        # Parser-missed hoops extend beyond the raw ear shell.  The lobe search
        # support remains tied to the actual source ear, not a generic box.
        weak_earring.get("source_lobe_search_mask", query_info.get("ear_roi", reference)),
        query_info.get("left_ear_roi", reference),
        query_info.get("right_ear_roi", reference),
        query_info.get("left_lobe_anchor", query_info.get("target_left_ear_mask", reference)),
        query_info.get("right_lobe_anchor", query_info.get("target_right_ear_mask", reference)),
        source_background_mask=(parsing.long() == 0).to(dtype=reference.dtype),
        source_hair_mask=query_info.get("source_hair_mask"),
        source_ear_mask=parsing_label_mask(parsing, RAW_EAR_SURFACE_LABELS),
        parser_earring_mask=parser_earring,
    )
    strong_candidate = strong_info["strong_candidate_mask"]

    height, width = reference.shape[-2:]
    area_scale = float(height * width) / float(256 * 256)
    min_area = max(0.0, float(args.earring_source_presence_min_area) * area_scale)
    presence_evidence = torch.clamp(parser_earring + strong_candidate, 0, 1)
    evidence_any = presence_evidence.flatten(1).amax(dim=1) > 0
    evidence_reliable = presence_evidence.flatten(1).sum(dim=1) >= min_area
    # V19 source presence is tri-state.  A small stud is uncertain rather
    # than absent, so it continues to the high-resolution foreground
    # extractor.  Only a complete lack of evidence is a hard no-op.
    presence_state = torch.where(
        evidence_reliable,
        torch.full_like(evidence_reliable, 2, dtype=torch.long),
        torch.where(evidence_any, torch.ones_like(evidence_reliable, dtype=torch.long), torch.zeros_like(evidence_reliable, dtype=torch.long)),
    )
    active = (presence_state > 0).to(dtype=reference.dtype).view(-1, 1, 1, 1)
    no_earring = (presence_state == 0).to(dtype=reference.dtype).view(-1, 1, 1, 1).expand_as(reference)
    earring_roi = query_info.get(
        "earring_valid_roi",
        query_info.get("visible_ear_roi", query_info["ear_roi"]),
    )

    # Weak evidence is useful only after parser-confirmed presence.  This hard
    # gate prevents ear edges/background texture from creating a synthetic
    # accessory branch on no-earring sources.
    gated_weak = {
        key: value * active if torch.is_tensor(value) else value
        for key, value in weak_earring.items()
    }
    if args.enable_earring_query_recall:
        recall_info = enhance_query_with_earring_recall(
            query_info["query_mask"],
            parser_earring * active,
            query_info.get("source_hair_block_mask"),
            gated_weak,
            visibility_mask=earring_roi,
            recall_dilate=args.earring_query_recall_dilate,
            downward_shift=args.earring_query_downward_shift,
            lower_lobe_weight=args.earring_query_lower_lobe_weight,
            candidate_boost=args.earring_query_candidate_boost,
            block_protect=args.earring_query_block_protect,
        )
        query_info.update(recall_info)

    candidate_mask = gated_weak.get("earring_candidate_mask", torch.zeros_like(reference))
    # Broad candidate completion is not trusted.  Only the validated visual
    # core may join parser evidence as a write-capable object.
    selected_object = torch.clamp(parser_earring + strong_candidate, 0, 1) * active

    search_mask = build_earring_search_mask(
        parser_earring,
        earring_roi,
        online_candidate_mask=candidate_mask,
        weak_recall_mask=query_info.get("earring_query_recall_mask"),
        lobe_search_mask=gated_weak.get("source_lobe_search_mask"),
        downward_shift=args.earring_search_downward_shift,
        search_dilate=args.earring_search_dilate,
        no_earring=no_earring,
    )

    target_hair_occlusion = query_info.get("target_ear_hair_occlusion_mask")
    if target_hair_occlusion is None:
        target_hair_occlusion = query_info["target_hair_mask"] * query_info["ear_roi"]
    target_hair_occlusion = resize_mask(target_hair_occlusion, reference.shape[-2:])

    def resize_or_zero(value):
        if value is None:
            return torch.zeros_like(reference)
        return resize_mask(value, reference.shape[-2:])

    left_roi = resize_or_zero(query_info.get("left_ear_roi"))
    right_roi = resize_or_zero(query_info.get("right_ear_roi"))
    left_active = resize_or_zero(query_info.get("left_side_active"))
    right_active = resize_or_zero(query_info.get("right_side_active"))
    # The compact ear ROIs decide whether a side is exposed.  They must not
    # geometrically crop a validated parser-missed hoop after its outer arc has
    # been recovered, otherwise generated supervision disagrees with V5
    # inference and teaches the model to keep only lobe-adjacent dots.
    ear_roi_gate = torch.clamp(left_active * left_roi + right_active * right_roi, 0, 1)
    visible_side_gate = torch.clamp(left_active + right_active, 0, 1)
    earring_roi = earring_roi * ear_roi_gate

    earlobe_anchor = (
        resize_or_zero(query_info.get("left_lobe_anchor")) * left_active * left_roi
        + resize_or_zero(query_info.get("right_lobe_anchor")) * right_active * right_roi
    ).clamp(0, 1)
    # The query builder already closes a lobe only when target-hair *interior*
    # covers it.  ``target_hair_occlusion`` is deliberately broader: it also
    # contains hair below the exposed lobe.  Applying it to this connectivity
    # anchor made generated labels zero out a verified long earring exactly
    # when its lower body should be restored in front of that transferred hair.
    earlobe_anchor = earlobe_anchor * active
    trusted_left, trusted_right = assign_components_to_ear_sides(
        selected_object,
        left_roi,
        right_roi,
        resize_or_zero(query_info.get("left_lobe_anchor")),
        resize_or_zero(query_info.get("right_lobe_anchor")),
    )
    completion_left, completion_right = assign_components_to_ear_sides(
        candidate_mask,
        left_roi,
        right_roi,
        resize_or_zero(query_info.get("left_lobe_anchor")),
        resize_or_zero(query_info.get("right_lobe_anchor")),
    )
    selected_object = (
        trusted_left * left_active + trusted_right * right_active
    ).clamp(0, 1) * visible_side_gate
    candidate_mask = (
        completion_left * left_active + completion_right * right_active
    ).clamp(0, 1) * visible_side_gate
    earring_roi = expand_valid_roi_by_completion(
        earring_roi,
        selected_object,
        grow_iters=int(args.earring_write_connectivity_iters),
        seed_dilate=max(1, int(args.earring_write_bridge_dilate)),
    ) * visible_side_gate

    source_background = (parsing.long() == 0).to(dtype=reference.dtype)
    source_non_earring = (parsing.long() != RAW_EARRING).to(dtype=reference.dtype)
    write_masks = build_earring_write_masks(
        selected_object,
        candidate_mask * search_mask * active,
        earring_roi,
        target_hair_occlusion,
        earlobe_anchor,
        no_earring=no_earring,
        source_background_mask=source_background,
        source_hair_mask=query_info.get("source_hair_mask"),
        source_hair_block_mask=query_info.get("source_hair_block_mask"),
        source_semantic_block_mask=source_non_earring,
        max_target_hair_overlap=args.earring_write_max_target_hair_overlap,
        source_block_dilate=args.earring_write_source_block_dilate,
        write_dilate=args.earring_write_dilate,
        connectivity_iters=args.earring_write_connectivity_iters,
        connectivity_kernel=args.earring_write_connectivity_kernel,
        bridge_dilate=args.earring_write_bridge_dilate,
    )
    geometric_hoop_hole = torch.clamp(
        strong_info["left_elliptical_hoop_hole"]
        + strong_info["right_elliptical_hoop_hole"],
        0,
        1,
    ) * active
    hoop_hole = torch.maximum(write_masks["hoop_hole_mask"], geometric_hoop_hole)
    write_mask = write_masks["write_mask"] * (1.0 - hoop_hole).clamp(0, 1)
    visible_segment = write_masks["earring_object_mask"] * active
    query_info["query_mask"] = torch.clamp(
        query_info["query_mask"] + max(0.0, float(args.earring_query_boost)) * search_mask,
        0,
        1,
    )

    return {
        "active": active,
        "presence_state": presence_state,
        "no_earring_case_mask": no_earring,
        "source_parser_earring_mask": parser_earring * active,
        "strong_earring_candidate_core": strong_candidate * active,
        "left_strong_candidate": strong_info["left_strong_candidate"] * active,
        "right_strong_candidate": strong_info["right_strong_candidate"] * active,
        "left_elliptical_hoop": strong_info["left_elliptical_hoop"] * active,
        "right_elliptical_hoop": strong_info["right_elliptical_hoop"] * active,
        "earring_candidate_mask": candidate_mask * search_mask * active,
        "selected_object_mask": selected_object,
        "earring_search_mask": search_mask,
        "earring_write_mask": write_mask,
        "earring_visible_segment_mask": visible_segment * active,
        "earring_core_mask": write_masks["core_mask"],
        "earring_completion_mask": write_masks["completion_mask"],
        "earring_object_mask": write_masks["earring_object_mask"],
        "earring_filled_mask": write_masks["earring_filled_mask"],
        "hoop_hole_mask": hoop_hole,
        "earlobe_anchor_mask": earlobe_anchor,
        "target_hair_ear_bridge_mask": (
            target_hair_occlusion * (1.0 - write_mask).clamp(0, 1)
        ).clamp(0, 1),
        "gated_weak": gated_weak,
    }


class RenderedPairDataset(Dataset):
    def __init__(self, experiments, dataset_path, face_gallery_root, donor_gallery_root):
        self.experiments = experiments
        self.dataset_path = Path(dataset_path)
        self.face_gallery_root = Path(face_gallery_root)
        self.donor_gallery_root = Path(donor_gallery_root)
        self.uses_in_memory_tensors = any(
            any(
                torch.is_tensor(item.get(key))
                for key in (
                    "source_full",
                    "shape_reference_full",
                    "color_reference_full",
                    "target_full",
                    "pre_reference_color_full",
                    "satd_background_highres",
                )
            )
            for item in experiments
        )

    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, idx):
        item = self.experiments[idx]
        source_path = self.face_gallery_root / item["source_name"]
        target_path = self.dataset_path / item["target_name"]
        pre_reference_name = item.get("pre_reference_color_name")
        pre_reference_path = (
            self.dataset_path / pre_reference_name
            if pre_reference_name is not None
            else target_path
        )
        _, shape_name, color_name = item["triplet"]
        source_full = item.get("source_full")
        if not torch.is_tensor(source_full):
            source_full = load_image(source_path)
        shape_reference_full = item.get("shape_reference_full")
        if not torch.is_tensor(shape_reference_full):
            shape_reference_full = load_image(self.donor_gallery_root / shape_name)
        color_reference_full = item.get("color_reference_full")
        if not torch.is_tensor(color_reference_full):
            color_reference_full = load_image(self.donor_gallery_root / color_name)
        target_full = item.get("target_full")
        if not torch.is_tensor(target_full):
            target_full = load_image(target_path)
        pre_reference_color_full = item.get("pre_reference_color_full")
        if not torch.is_tensor(pre_reference_color_full):
            pre_reference_color_full = (
                load_image(pre_reference_path)
                if pre_reference_name is not None
                else target_full
            )
        completed_hair_highres = item.get("completed_hair_highres")
        if not torch.is_tensor(completed_hair_highres):
            raise RuntimeError(
                "Rendered V6 dataset item is missing completed_hair_highres. "
                "Regenerate this schema-34 dataset from the matching V6 generator."
            )
        if completed_hair_highres.ndim != 3 or completed_hair_highres.size(0) != 3:
            raise RuntimeError(
                "completed_hair_highres must be RGB [3,H,W], "
                f"got {tuple(completed_hair_highres.shape)}."
            )
        if completed_hair_highres.dtype == torch.uint8:
            completed_hair_highres = completed_hair_highres.float().div(255)
        else:
            completed_hair_highres = completed_hair_highres.float().clamp(0, 1)
        satd_background_highres = item.get("satd_background_highres")
        if not torch.is_tensor(satd_background_highres):
            raise RuntimeError(
                "Rendered V6 dataset item is missing satd_background_highres. "
                "Regenerate this schema-34 dataset from the matching V6 generator."
            )
        if satd_background_highres.ndim != 3 or satd_background_highres.size(0) != 3:
            raise RuntimeError(
                "satd_background_highres must be RGB [3,H,W], "
                f"got {tuple(satd_background_highres.shape)}."
            )
        if satd_background_highres.dtype == torch.uint8:
            satd_background_highres = satd_background_highres.float().div(255)
        else:
            satd_background_highres = satd_background_highres.float().clamp(0, 1)
        direct_satd_pp_input = item.get("direct_satd_pp_input")
        if torch.is_tensor(direct_satd_pp_input):
            direct_satd_pp_input = bool(direct_satd_pp_input.item())
        else:
            direct_satd_pp_input = bool(direct_satd_pp_input)
        return {
            "source_path": str(source_path),
            "shape_reference_path": str(self.donor_gallery_root / shape_name),
            "color_reference_path": str(self.donor_gallery_root / color_name),
            "source_full": source_full,
            # Dataset parts must be self-contained.  Previewing them on a
            # training machine with a different gallery mount used to silently
            # replace both references with the target image.
            "shape_reference_full": shape_reference_full,
            "color_reference_full": color_reference_full,
            "target_full": target_full,
            "pre_reference_color_full": pre_reference_color_full,
            "completed_hair_highres": completed_hair_highres,
            "satd_background_highres": satd_background_highres,
            "direct_satd_pp_input": torch.tensor(direct_satd_pp_input, dtype=torch.bool),
            "cleanup_masks": item.get("cleanup_masks", {}),
            "target_hair_mask_override": item.get("target_hair_mask"),
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
            "earring_occlusion_dilate": args.earring_occlusion_dilate,
            "source_hair_block_dilate": args.source_hair_block_dilate,
            "source_hair_block_strength": args.source_hair_block_strength,
            "target_visibility_expand": args.target_visibility_expand,
            "max_target_hair_overlap": args.max_target_hair_overlap,
            "min_target_visible_overlap": args.min_target_visible_overlap,
            "min_target_ear_area": args.min_target_ear_area,
            "earring_channel_down": args.earring_channel_down,
            "earring_align_max_shift": args.earring_align_max_shift,
        }
        accepted = inspect.signature(EarAnchoredQueryBuilder.__init__).parameters
        query_builder_kwargs = {
            key: value for key, value in query_builder_kwargs.items() if key in accepted
        }
        self.query_builder = EarAnchoredQueryBuilder(**query_builder_kwargs)

    def _resize_to_256(self, images):
        """Return exact PP-resolution RGB for both old and new dataset parts."""
        if tuple(images.shape[-2:]) == (256, 256):
            return images
        if tuple(images.shape[-2:]) == (1024, 1024):
            return self.downsample_256(images)
        return F.interpolate(
            images,
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    @staticmethod
    def _resize_for_hair_parser(images):
        """HairMaskExtractorV5 expects 1024 input before its fixed 2x shrink."""
        if tuple(images.shape[-2:]) == (1024, 1024):
            return images
        return F.interpolate(
            images,
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    def _iter_single_process_batches(self, dataset):
        batch_size = self.args.mask_batch_size
        total_batches = ceil_div(len(dataset), batch_size)
        for start in tqdm(range(0, len(dataset), batch_size), total=total_batches, desc="Build dataset masks"):
            end = min(len(dataset), start + batch_size)
            batch_items = [dataset[idx] for idx in range(start, end)]
            yield {
                "source_path": [item["source_path"] for item in batch_items],
                "shape_reference_path": [item["shape_reference_path"] for item in batch_items],
                "color_reference_path": [item["color_reference_path"] for item in batch_items],
                "source_full": torch.stack([item["source_full"] for item in batch_items], dim=0),
                "shape_reference_full": torch.stack(
                    [item["shape_reference_full"] for item in batch_items], dim=0
                ),
                "color_reference_full": torch.stack(
                    [item["color_reference_full"] for item in batch_items], dim=0
                ),
                "target_full": torch.stack([item["target_full"] for item in batch_items], dim=0),
                "pre_reference_color_full": torch.stack(
                    [item["pre_reference_color_full"] for item in batch_items],
                    dim=0,
                ),
                "completed_hair_highres": torch.stack(
                    [item["completed_hair_highres"] for item in batch_items],
                    dim=0,
                ),
                "satd_background_highres": torch.stack(
                    [item["satd_background_highres"] for item in batch_items],
                    dim=0,
                ),
                "direct_satd_pp_input": torch.stack(
                    [item["direct_satd_pp_input"] for item in batch_items],
                    dim=0,
                ),
                "target_hair_mask_override": torch.stack(
                    [
                        item.get(
                            # RenderedPairDataset normalizes the stage output
                            # under this explicit name.  Reading the old key
                            # silently replaced every streamed sample with an
                            # all-zero override, so the label builder reparsed
                            # a stale hair topology around exposed earlobes.
                            "target_hair_mask_override",
                            torch.zeros(1, 256, 256),
                        )
                        for item in batch_items
                    ],
                    dim=0,
                ),
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
    def iter_batches(self, experiments, dataset_path, face_gallery_root, donor_gallery_root):
        dataset = RenderedPairDataset(experiments, dataset_path, face_gallery_root, donor_gallery_root)
        # Large CPU tensors are already resident for streamed render groups.
        # Sending them through DataLoader worker processes copies them and is
        # slower than building the one local batch directly.
        if self.args.io_num_workers <= 0 or dataset.uses_in_memory_tensors:
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
            shape_reference_paths = batch["shape_reference_path"]
            color_reference_paths = batch["color_reference_path"]
            source_full = batch["source_full"].to(self.device, non_blocking=False)
            shape_reference_full = batch["shape_reference_full"].to(
                self.device,
                non_blocking=False,
            )
            color_reference_full = batch["color_reference_full"].to(
                self.device,
                non_blocking=False,
            )
            target_full = batch["target_full"].to(self.device, non_blocking=False)
            pre_reference_color_full = batch["pre_reference_color_full"].to(
                self.device,
                non_blocking=False,
            )
            completed_hair_highres = batch["completed_hair_highres"].to(
                self.device,
                non_blocking=False,
            )
            satd_background_highres = batch["satd_background_highres"].to(
                self.device,
                non_blocking=False,
            )
            direct_satd_pp_input = batch["direct_satd_pp_input"].to(
                self.device,
                non_blocking=False,
            )
            batch_cleanup_masks = batch.get("cleanup_masks", {})

            source_256 = self._resize_to_256(source_full).clip(0, 1)
            shape_reference_256 = self._resize_to_256(shape_reference_full).clip(0, 1)
            color_reference_256 = self._resize_to_256(color_reference_full).clip(0, 1)
            target_256 = self._resize_to_256(target_full).clip(0, 1)
            pre_reference_color_256 = self._resize_to_256(pre_reference_color_full).clip(0, 1)
            source_hair_input = self._resize_for_hair_parser(source_full)
            target_hair_input = self._resize_for_hair_parser(target_full)
            source_hair_d, _ = self.hair_mask_extractor.generate_mask(source_hair_input)
            target_hair_d, target_hair_e = self.hair_mask_extractor.generate_mask(target_hair_input)
            target_hair_override = batch.get("target_hair_mask_override")
            if target_hair_override is not None:
                target_hair_override = resize_mask(
                    target_hair_override.to(self.device, non_blocking=False).float(),
                    target_hair_d.shape[-2:],
                )
                # A missing override is represented by an all-zero tensor for
                # compatibility with old parts; only replace the parser mask
                # when the stage supplied a non-empty repaired topology.
                has_override = (
                    target_hair_override.flatten(1).amax(dim=1, keepdim=True) > 0
                ).view(-1, 1, 1, 1)
                target_hair_d = torch.where(
                    has_override,
                    target_hair_override,
                    target_hair_d,
                ).clamp(0, 1)
            target_mask = (1 - source_hair_d) * (1 - target_hair_d)

            # One source parser pass must serve both PP-resolution conditioning
            # and native earring-instance extraction.  Parsing the 256px source
            # and enlarging its labels permanently loses a small stud or the
            # lower half of a pendant before the native verifier starts.
            source_native_size = tuple(source_full.shape[-2:])
            source_parsing_full = self.parsing_helper.parse(
                source_full,
                out_size=source_native_size,
            )
            source_parsing = F.interpolate(
                source_parsing_full.float(),
                size=source_256.shape[-2:],
                mode="nearest",
            ).long()
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
            earring_valid_roi = query_info.get("earring_valid_roi", visible_ear_roi)
            detection_ear_roi = query_info.get("ear_roi", visible_ear_roi)
            source_earring_detection_mask = query_info.get("source_earring_detection_mask")

            weak_earring = build_weak_earring_masks(
                source_256,
                detection_ear_roi,
                query_info["query_mask"],
                source_earring_detection_mask,
                query_info.get("source_hair_mask"),
                source_hair_block_mask,
                source_parsing,
            )
            earring_policy = build_dataset_earring_policy_masks(
                query_info,
                weak_earring,
                source_parsing,
                source_256,
                self.args,
            )
            source_hair_block_mask = query_info.get(
                "source_hair_block_mask",
                source_hair_block_mask,
            )
            earring_search_mask = earring_policy["earring_search_mask"]
            # Dataset supervision must use the same source-instance contract as
            # final V5 inference.  The older ``earring_write_mask`` is a
            # low-resolution *permission* region; using it as a label taught
            # the PP model that ear-side background was an accessory and that
            # a partially parsed hoop was a solid blob.
            reliable_source_seed = torch.maximum(
                earring_policy.get("source_parser_earring_mask", torch.zeros_like(earring_search_mask)),
                earring_policy.get("strong_earring_candidate_core", torch.zeros_like(earring_search_mask)),
            )
            # Final V5 inference extracts ordinary instances and hoop evidence
            # in source-native coordinates.  Doing this first at 256px taught
            # the model that a thin wire, a small stud or the lower part of a
            # long pendant simply did not exist.  Keep the high-resolution
            # contract here, then reduce only the verified object alpha/RGB
            # label that PP is trained to consume.
            reliable_source_seed_full = F.interpolate(
                reliable_source_seed.float(),
                size=source_native_size,
                mode="nearest",
            )
            source_recall_hint = torch.zeros_like(reliable_source_seed)
            for key in (
                "strong_earring_candidate_core",
                "online_earring_candidate_mask",
                "earring_candidate_mask",
                "earring_query_recall_mask",
                "earring_object_recall_mask",
                "earring_object_detection_mask",
            ):
                value = earring_policy.get(key)
                if value is not None:
                    source_recall_hint = torch.maximum(
                        source_recall_hint,
                        F.interpolate(value.float(), size=source_recall_hint.shape[-2:], mode="nearest"),
                    )
            source_recall_hint_full = F.interpolate(
                source_recall_hint,
                size=source_native_size,
                mode="nearest",
            )
            # V5 dataset authority is one source-native foreground
            # instance. The target-aligned supervision below is derived from
            # it once; no target field is ever resized back into this extractor.
            source_foreground_v6 = extract_source_native_earring_v6(
                source_full,
                source_parsing_full,
                source_native_seed=reliable_source_seed_full,
                seed_space=EarringCoordinateSpace.SOURCE_NATIVE,
                source_native_recall_hint=source_recall_hint_full,
                recall_hint_space=EarringCoordinateSpace.SOURCE_NATIVE,
                # The generic graph remains conservative.  A long pendant is
                # completed only by the independent structured source
                # verifier below, not by chaining arbitrary visual fragments.
                max_graph_depth=4,
                max_cumulative_cost=1.85,
                # Match the final compositor's bounded continuation policy so
                # long pendants are not shortened in the training target.
                # Candidate components still pass the source-lobe, hair and
                # background gates; no free-form ROI pixels are added.
                allow_long_continuation=True,
            )
            # The native foreground extractor is conservative by design.  A
            # parser-missed pearl or a low-contrast long pendant can therefore
            # produce zero alpha even when the 256px policy supplied a strong,
            # lobe-associated location hint.  Train V6 on the same bounded
            # structured fallback used by the final compositor: it verifies
            # source-native components and never treats an ear ROI or a source
            # crop as write alpha.  This replaces a shorter/empty native
            # result only; it does not lower the no-earring threshold.
            structured_source_instances = build_source_earring_instance_masks_v5(
                source_full,
                source_parsing_full,
                source_hair_mask=parsing_label_mask(source_parsing_full, (RAW_HAIR,)),
                source_seed_mask=torch.clamp(
                    reliable_source_seed_full + source_recall_hint_full,
                    0,
                    1,
                ),
            )
            native_area = float(source_native_size[0] * source_native_size[1])
            native_scale = max(source_native_size) / 256.0
            minimum_structured_area = max(8.0, 2.0 * native_scale * native_scale)
            maximum_structured_area = 0.10 * native_area
            source_hair_full = parsing_label_mask(source_parsing_full, (RAW_HAIR,)).to(
                device=source_full.device,
                dtype=source_full.dtype,
            )

            def prefer_structured_source_instance(
                direct: torch.Tensor,
                structured: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                structured = structured.to(device=source_full.device, dtype=source_full.dtype)
                # A real source-hair pixel is not recoverable earring RGB.
                structured = structured * (1.0 - source_hair_full).clamp(0, 1)
                direct_area = direct.flatten(1).sum(dim=1, keepdim=True)
                structured_area = structured.flatten(1).sum(dim=1, keepdim=True)
                def vertical_extent(value: torch.Tensor) -> torch.Tensor:
                    rows = (value > 0.01).amax(dim=3).float()
                    height = value.shape[-2]
                    ids = torch.arange(height, device=value.device, dtype=value.dtype).view(1, 1, height)
                    first = torch.where(rows > 0, ids, torch.full_like(ids, float(height))).amin(dim=2)
                    last = (rows * ids).amax(dim=2)
                    return (last - first + 1.0).clamp_min(0).view(-1, 1)

                direct_extent = vertical_extent(direct)
                structured_extent = vertical_extent(structured)
                use_structured_area = (
                    (structured_area >= minimum_structured_area)
                    & (structured_area <= maximum_structured_area)
                )
                # The native GrabCut graph and the source-lobe structured
                # verifier observe different parts of a long pendant.  The
                # old replacement rule discarded whichever observer had the
                # shorter contour, which serialized only a root/rim into
                # ``source_native_earring_alpha``.  A validated structured
                # instance is object-only and source-lobe connected, so merge
                # it only when the direct contour is absent or truncated;
                # no ROI or unverified background pixels are introduced.
                # Pick a complete verifier result
                # only when the direct native contour is absent or truncated.
                prefer_structured = (
                    use_structured_area
                    & (
                        (direct_area < minimum_structured_area)
                        | (structured_area >= direct_area * 1.15)
                        | (
                            (structured_extent >= direct_extent + max(3.0, 0.04 * source_native_size[0]))
                            & (structured_area >= direct_area * 0.70)
                        )
                    )
                ).view(-1, 1, 1, 1)
                selected = torch.where(prefer_structured, structured, direct)
                return selected, prefer_structured

            selected_source_left, use_structured_left = prefer_structured_source_instance(
                source_foreground_v6["source_native_left_alpha"],
                structured_source_instances["left_instance_mask"],
            )
            selected_source_right, use_structured_right = prefer_structured_source_instance(
                source_foreground_v6["source_native_right_alpha"],
                structured_source_instances["right_instance_mask"],
            )
            # Match final V6 inference: raw label-9 is trusted only after it
            # has been assigned to one real source ear and reduced to one
            # lobe-connected vertical object chain.  This preserves a long
            # pendant that the visual extractor kept only at its root, but it
            # does not turn the surrounding source ear/background corridor
            # into training RGB alpha.
            source_left_ear_full = parsing_label_mask(source_parsing_full, (7,))
            source_right_ear_full = parsing_label_mask(source_parsing_full, (8,))
            parser_left_full, parser_right_full = assign_components_to_ear_sides(
                parsing_label_mask(source_parsing_full, (RAW_EARRING,)),
                source_left_ear_full,
                source_right_ear_full,
                source_left_ear_full,
                source_right_ear_full,
            )
            pendant_gap = max(8, int(round(42.0 * max(source_native_size) / 256.0)))
            pendant_extent = max(
                pendant_gap,
                int(round(240.0 * max(source_native_size) / 256.0)),
            )
            parser_left_full = retain_single_earring_group_v6(
                parser_left_full,
                source_left_ear_full,
                max_gap=pendant_gap,
                max_downward_extent=pendant_extent,
            )
            parser_right_full = retain_single_earring_group_v6(
                parser_right_full,
                source_right_ear_full,
                max_gap=pendant_gap,
                max_downward_extent=pendant_extent,
            )

            def prefer_complete_semantic_chain(
                selected: torch.Tensor,
                semantic_chain: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                selected_area = selected.flatten(1).sum(dim=1, keepdim=True)
                chain_area = semantic_chain.flatten(1).sum(dim=1, keepdim=True)
                selected_rows = (selected > 0.01).amax(dim=3).float().sum(dim=2)
                chain_rows = (semantic_chain > 0.01).amax(dim=3).float().sum(dim=2)
                prefer_chain = (
                    (chain_area >= minimum_structured_area)
                    & (
                        (selected_area < minimum_structured_area)
                        | (chain_area >= selected_area * 1.08)
                        | (chain_rows >= selected_rows + max(3.0, 0.04 * source_native_size[0]))
                    )
                ).view(-1, 1, 1, 1)
                return torch.where(prefer_chain, semantic_chain, selected), prefer_chain

            selected_source_left, use_parser_left = prefer_complete_semantic_chain(
                selected_source_left,
                parser_left_full,
            )
            selected_source_right, use_parser_right = prefer_complete_semantic_chain(
                selected_source_right,
                parser_right_full,
            )
            # A single side may still contain several components from a weak
            # detector.  Retain one root plus downward pendant segments only;
            # a second side-by-side object can no longer become a duplicate
            # earring in the serialized target.
            selected_source_left = retain_single_earring_group_v6(
                selected_source_left,
                source_left_ear_full,
                max_gap=pendant_gap,
                max_downward_extent=pendant_extent,
            )
            selected_source_right = retain_single_earring_group_v6(
                selected_source_right,
                source_right_ear_full,
                max_gap=pendant_gap,
                max_downward_extent=pendant_extent,
            )
            selected_source_left_hole = torch.where(
                use_parser_left,
                compute_earring_hole_mask(parser_left_full),
                torch.where(
                    use_structured_left,
                    structured_source_instances["left_hoop_hole_mask"].to(
                        device=source_full.device,
                        dtype=source_full.dtype,
                    ),
                    source_foreground_v6["source_native_left_hole_alpha"],
                ),
            )
            selected_source_right_hole = torch.where(
                use_parser_right,
                compute_earring_hole_mask(parser_right_full),
                torch.where(
                    use_structured_right,
                    structured_source_instances["right_hoop_hole_mask"].to(
                        device=source_full.device,
                        dtype=source_full.dtype,
                    ),
                    source_foreground_v6["source_native_right_hole_alpha"],
                ),
            )
            # Keep source semantic regions out of the serialized alpha even
            # when the structured fallback selected a parser-background
            # continuation.  The native extractor/structured verifier has
            # already made the object-vs-background decision; removing every
            # label-0 tail here would truncate precisely the long pendants
            # this fallback is meant to recover.
            source_labels_full = source_parsing_full.long()
            source_subject_block_full = (
                (source_labels_full != 0) & (source_labels_full != RAW_EARRING)
            ).to(source_full.dtype)
            source_object_gate_full = (1.0 - source_subject_block_full).clamp(0, 1)
            selected_source_left = selected_source_left * source_object_gate_full
            selected_source_right = selected_source_right * source_object_gate_full
            selected_source_left = selected_source_left * (1.0 - selected_source_left_hole).clamp(0, 1)
            selected_source_right = selected_source_right * (1.0 - selected_source_right_hole).clamp(0, 1)
            (
                selected_source_left,
                selected_source_right,
                removed_source_left,
                removed_source_right,
            ) = enforce_exclusive_earring_sides_v6(
                selected_source_left,
                selected_source_right,
                parsing_label_mask(source_parsing_full, (7,)),
                parsing_label_mask(source_parsing_full, (8,)),
            )
            selected_source_left_hole = selected_source_left_hole * (
                1.0 - removed_source_left
            ).clamp(0, 1)
            selected_source_right_hole = selected_source_right_hole * (
                1.0 - removed_source_right
            ).clamp(0, 1)
            source_foreground_v6["source_native_left_alpha"] = selected_source_left
            source_foreground_v6["source_native_right_alpha"] = selected_source_right
            source_foreground_v6["source_native_left_hole_alpha"] = selected_source_left_hole
            source_foreground_v6["source_native_right_hole_alpha"] = selected_source_right_hole
            source_foreground_v6["source_native_earring_alpha"] = torch.clamp(
                selected_source_left + selected_source_right,
                0,
                1,
            )
            source_foreground_v6["source_native_hole_alpha"] = torch.clamp(
                selected_source_left_hole + selected_source_right_hole,
                0,
                1,
            )
            source_foreground_v6["source_native_left_label9_chain"] = parser_left_full
            source_foreground_v6["source_native_right_label9_chain"] = parser_right_full
            structured_source_alpha = torch.clamp(
                structured_source_instances["left_instance_mask"].to(
                    device=source_full.device,
                    dtype=source_full.dtype,
                )
                + structured_source_instances["right_instance_mask"].to(
                    device=source_full.device,
                    dtype=source_full.dtype,
                ),
                0,
                1,
            ) * (1.0 - source_hair_full).clamp(0, 1)
            source_foreground_v6["source_native_presence_state"] = torch.cat(
                (
                    (selected_source_left.flatten(1).sum(dim=1, keepdim=True) >= 1.0),
                    (selected_source_right.flatten(1).sum(dim=1, keepdim=True) >= 1.0),
                ),
                dim=1,
            ).to(dtype=source_full.dtype)
            source_foreground_v6["source_native_presence_score"] = torch.maximum(
                source_foreground_v6["source_native_presence_score"],
                source_foreground_v6["source_native_presence_state"],
            )
            # Keep this diagnostic in the generated item so a missed object
            # can be distinguished from a target-side visibility rejection.
            structured_source_fallback = torch.clamp(
                use_structured_left.to(source_full.dtype)
                + use_structured_right.to(source_full.dtype),
                0,
                1,
            )
            native_zero = torch.zeros_like(source_foreground_v6["source_native_earring_alpha"])
            source_instances_full = {
                "left_instance_mask": source_foreground_v6["source_native_left_alpha"],
                "right_instance_mask": source_foreground_v6["source_native_right_alpha"],
                "instance_mask": source_foreground_v6["source_native_earring_alpha"],
                "left_parser_instance_mask": source_foreground_v6["source_native_left_alpha"],
                "right_parser_instance_mask": source_foreground_v6["source_native_right_alpha"],
                "left_hole_mask": source_foreground_v6["source_native_left_hole_alpha"],
                "right_hole_mask": source_foreground_v6["source_native_right_hole_alpha"],
                "left_context": source_foreground_v6["localization_core"],
                "right_context": source_foreground_v6["localization_core"],
                "locator_presence_seed": source_foreground_v6["parser_earring_seed"],
                "left_visual_recall_hoop_seed_mask": native_zero,
                "right_visual_recall_hoop_seed_mask": native_zero,
            }

            # Ordinary earrings need the same source-native refinement as the
            # final V5 compositor.  Schema 11 only used ``instance_mask`` here;
            # when the parser saw a lobe fragment or a thin contour, PP was
            # trained to reproduce that fragment even though inference could
            # later verify the complete pendant/solid ornament at high
            # resolution.  Refine before downsampling so the training target
            # and final RGB authority describe the same object.
            source_ear_surface_full = parsing_label_mask(
                source_parsing_full,
                RAW_EAR_SURFACE_LABELS,
            )
            accepted_presence_seed_full = source_instances_full.get(
                "locator_presence_seed",
                torch.zeros_like(source_instances_full["instance_mask"]),
            )
            left_bootstrap_seed_full = (
                accepted_presence_seed_full
                * source_instances_full["left_context"]
            )
            right_bootstrap_seed_full = (
                accepted_presence_seed_full
                * source_instances_full["right_context"]
            )
            # Match final inference: a source-lobe-associated low-resolution
            # seed may start high-resolution inspection even if parser labels
            # only the earlobe-side fragment.  It is never stored as object
            # alpha; the high-resolution extractor must verify the actual
            # source pixels before this dataset gets a positive target.
            left_native_active = (
                torch.clamp(
                    source_instances_full["left_instance_mask"]
                    + left_bootstrap_seed_full,
                    0,
                    1,
                ).flatten(1).sum(
                    dim=1,
                    keepdim=True,
                ) >= 1.0
            ).to(source_full.dtype).view(-1, 1, 1, 1)
            right_native_active = (
                torch.clamp(
                    source_instances_full["right_instance_mask"]
                    + right_bootstrap_seed_full,
                    0,
                    1,
                ).flatten(1).sum(
                    dim=1,
                    keepdim=True,
                ) >= 1.0
            ).to(source_full.dtype).view(-1, 1, 1, 1)
            # V19 foreground alpha is already the complete source object.
            # The old high-resolution refinement repeatedly intersected that
            # object with blocker/connectivity masks and is intentionally
            # disabled for dataset authority.
            native_regular_refined_full = {
                "left_instance_mask": torch.zeros_like(source_instances_full["left_instance_mask"]),
                "right_instance_mask": torch.zeros_like(source_instances_full["right_instance_mask"]),
                "left_hoop_hole_mask": torch.zeros_like(source_instances_full["left_instance_mask"]),
                "right_hoop_hole_mask": torch.zeros_like(source_instances_full["right_instance_mask"]),
            }

            def source_instance_to_256(value):
                if value.shape[-2:] == (1, 1):
                    return value.to(dtype=source_256.dtype)
                # A verified source object can be thinner than one 256px
                # sampling interval.  Nearest sampling may select only its
                # background neighbour and turn a real stud/wire into an empty
                # training target.  Occupancy pooling preserves one positive
                # PP pixel for each native cell containing actual jewellery;
                # it never opens unverified surroundings.
                if (
                    value.shape[-2] >= source_256.shape[-2]
                    and value.shape[-1] >= source_256.shape[-1]
                ):
                    return F.adaptive_max_pool2d(
                        value.float(),
                        output_size=source_256.shape[-2:],
                    ).to(dtype=source_256.dtype)
                return F.interpolate(
                    value.float(),
                    size=source_256.shape[-2:],
                    mode="nearest",
                ).to(dtype=source_256.dtype)

            source_instances_256 = {
                key: source_instance_to_256(value)
                for key, value in source_instances_full.items()
            }
            source_earring_mask = source_instances_256["instance_mask"]
            left_source_instance = source_instances_256["left_instance_mask"]
            right_source_instance = source_instances_256["right_instance_mask"]
            left_regular_hole = source_instances_256["left_hole_mask"]
            right_regular_hole = source_instances_256["right_hole_mask"]
            active_earring_case = (
                source_earring_mask.flatten(1).sum(dim=1, keepdim=True) >= 1.0
            ).to(source_earring_mask.dtype).view(-1, 1, 1, 1)

            # Use the same source-native annulus extractor as final V5
            # inference.  A hoop verifier is never a presence detector: each
            # side first needs parser evidence or a native closed-loop visual
            # seed.  The low-resolution strong proposal is a locator hint, not
            # authority for a background contour.
            source_ear_mask = source_ear_surface_full
            native_zero = torch.zeros_like(
                source_instances_full["left_parser_instance_mask"]
            )
            # The visual-recall helper exposes a separate closed-loop seed for
            # the contour verifier.  Ordinary recall alpha/attachment evidence
            # must stay on the ordinary-instance path and cannot authorize a
            # synthetic hoop search.
            left_visual_recall_hoop_seed = source_instances_full.get(
                "left_visual_recall_hoop_seed_mask",
                native_zero,
            )
            right_visual_recall_hoop_seed = source_instances_full.get(
                "right_visual_recall_hoop_seed_mask",
                native_zero,
            )
            left_hoop_evidence = torch.clamp(
                source_instances_full["left_parser_instance_mask"]
                + left_visual_recall_hoop_seed,
                0,
                1,
            )
            right_hoop_evidence = torch.clamp(
                source_instances_full["right_parser_instance_mask"]
                + right_visual_recall_hoop_seed,
                0,
                1,
            )
            # Hoops are ordinary foreground instances in V19.  Their natural
            # alpha hole is preserved by source segmentation; no ellipse,
            # contour or synthetic hoop branch can create RGB authority.
            hoop_zero = torch.zeros_like(source_instances_full["instance_mask"])
            hoop_instances_full = {
                "left_elliptical_hoop": hoop_zero,
                "right_elliptical_hoop": hoop_zero,
                "left_elliptical_hoop_hole": hoop_zero,
                "right_elliptical_hoop_hole": hoop_zero,
                "left_elliptical_hoop_footprint": hoop_zero,
                "right_elliptical_hoop_footprint": hoop_zero,
                "left_elliptical_hoop_connector": hoop_zero,
                "right_elliptical_hoop_connector": hoop_zero,
            }
            hoop_instances = {
                key: resize_mask(value, source_earring_mask.shape[-2:])
                for key, value in hoop_instances_full.items()
            }
            def side_is_open(mask):
                mask = torch.zeros_like(source_earring_mask) if mask is None else mask
                return (
                    mask.flatten(1).amax(dim=1, keepdim=True) > 0.5
                ).to(source_earring_mask.dtype).view(-1, 1, 1, 1)

            # A target side is a visibility decision, not a source-coordinate
            # clip.  Multiplying a full source hoop by a target ear shell cut
            # the outer arc back to a short fragment in the generated labels.
            #
            # The 256px query can miss a narrow but genuinely exposed lobe.
            # The completed transfer's parser supplies the only permissible
            # fallback: an actual target ear label after target-hair removal.
            # A generic face/neck skin mask is deliberately not accepted here.
            target_left_parser_visible = (
                parsing_label_mask(target_parsing, (7,))
                * (1.0 - target_hair_d).clamp(0, 1)
            ).clamp(0, 1)
            target_right_parser_visible = (
                parsing_label_mask(target_parsing, (8,))
                * (1.0 - target_hair_d).clamp(0, 1)
            ).clamp(0, 1)
            target_skin_surface = query_info.get("target_skin_surface_mask")
            if target_skin_surface is None:
                target_skin_surface = torch.zeros_like(target_left_parser_visible)
            else:
                target_skin_surface = resize_mask(
                    target_skin_surface,
                    target_left_parser_visible.shape[-2:],
                ).to(
                    device=target_left_parser_visible.device,
                    dtype=target_left_parser_visible.dtype,
                )

            def exposed_lobe_fallback(lobe_key: str, roi_key: str) -> torch.Tensor:
                lobe = query_info.get(lobe_key, query_info.get(roi_key))
                if lobe is None:
                    return torch.zeros_like(target_left_parser_visible[:, :1, :1, :1])
                lobe = resize_mask(lobe, target_left_parser_visible.shape[-2:]).to(
                    device=target_left_parser_visible.device,
                    dtype=target_left_parser_visible.dtype,
                )
                window = dilate_mask(lobe, 5)
                skin = target_skin_surface * window * (1.0 - target_hair_d).clamp(0, 1)
                hair = target_hair_d * window
                corridor = (skin + hair).flatten(1).sum(dim=1, keepdim=True).clamp_min(1.0)
                return (
                    (skin.flatten(1).sum(dim=1, keepdim=True) >= 1.0)
                    & ((hair.flatten(1).sum(dim=1, keepdim=True) / corridor) <= 0.25)
                ).to(source_earring_mask.dtype).view(-1, 1, 1, 1)

            # Query-time side flags may remain open after transferred hair
            # covers an ear.  Use only final target parser/lobe evidence for
            # both training visibility and the target attachment point.
            parser_min_visible_area = 2.0
            left_open = (
                target_left_parser_visible.flatten(1).sum(dim=1, keepdim=True)
                >= parser_min_visible_area
            ).to(source_earring_mask.dtype).view(-1, 1, 1, 1)
            left_open = torch.maximum(
                left_open,
                exposed_lobe_fallback("left_lobe_anchor", "left_ear_roi"),
            )
            right_open = (
                target_right_parser_visible.flatten(1).sum(dim=1, keepdim=True)
                >= parser_min_visible_area
            ).to(source_earring_mask.dtype).view(-1, 1, 1, 1)
            right_open = torch.maximum(
                right_open,
                exposed_lobe_fallback("right_lobe_anchor", "right_ear_roi"),
            )

            def exposed_lobe_mask(lobe_key: str, roi_key: str) -> torch.Tensor:
                lobe = query_info.get(lobe_key, query_info.get(roi_key))
                if lobe is None:
                    return torch.zeros_like(target_left_parser_visible)
                lobe = resize_mask(lobe, target_left_parser_visible.shape[-2:]).to(
                    device=target_left_parser_visible.device,
                    dtype=target_left_parser_visible.dtype,
                )
                return (
                    target_skin_surface
                    * dilate_mask(lobe, 5)
                    * (1.0 - target_hair_d).clamp(0, 1)
                ).clamp(0, 1)

            target_left_alignment = torch.clamp(
                target_left_parser_visible
                + exposed_lobe_mask("left_lobe_anchor", "left_ear_roi"),
                0,
                1,
            ) * left_open
            target_right_alignment = torch.clamp(
                target_right_parser_visible
                + exposed_lobe_mask("right_lobe_anchor", "right_ear_roi"),
                0,
                1,
            ) * right_open

            def ordinary_body_overrides_hoop(
                ordinary: torch.Tensor,
                footprint: torch.Tensor,
            ) -> torch.Tensor:
                """Do not train a verified solid ornament as a hollow hoop."""

                footprint_area = footprint.flatten(1).sum(dim=1, keepdim=True)
                ordinary_inside = (ordinary * footprint).flatten(1).sum(
                    dim=1,
                    keepdim=True,
                )
                coverage = ordinary_inside / footprint_area.clamp_min(1.0)
                ordinary_hole = compute_earring_hole_mask(dilate_mask(ordinary, 3))
                hole_coverage = (ordinary_hole * footprint).flatten(1).sum(
                    dim=1,
                    keepdim=True,
                ) / footprint_area.clamp_min(1.0)
                return (
                    (footprint_area >= 12.0)
                    & (coverage >= 0.62)
                    & (hole_coverage <= 0.08)
                ).to(ordinary.dtype).view(-1, 1, 1, 1)

            left_regular_is_solid = ordinary_body_overrides_hoop(
                left_source_instance,
                hoop_instances["left_elliptical_hoop_footprint"],
            )
            right_regular_is_solid = ordinary_body_overrides_hoop(
                right_source_instance,
                hoop_instances["right_elliptical_hoop_footprint"],
            )
            left_hoop_instance = (
                hoop_instances["left_elliptical_hoop"]
                * left_open
                * (1.0 - left_regular_is_solid)
            )
            right_hoop_instance = (
                hoop_instances["right_elliptical_hoop"]
                * right_open
                * (1.0 - right_regular_is_solid)
            )
            left_hoop_hole = (
                hoop_instances["left_elliptical_hoop_hole"]
                * left_open
                * (1.0 - left_regular_is_solid)
            )
            right_hoop_hole = (
                hoop_instances["right_elliptical_hoop_hole"]
                * right_open
                * (1.0 - right_regular_is_solid)
            )
            source_hoop_instance = torch.clamp(left_hoop_instance + right_hoop_instance, 0, 1)
            source_hoop_hole = torch.clamp(left_hoop_hole + right_hoop_hole, 0, 1)
            hoop_present = (
                source_hoop_instance.flatten(1).sum(dim=1, keepdim=True) >= 1.0
            ).to(source_earring_mask.dtype).view(-1, 1, 1, 1)
            active_earring_case = torch.maximum(active_earring_case, hoop_present)
            no_earring_case = (1.0 - active_earring_case).clamp(0, 1).expand_as(source_earring_mask)
            earring_search_mask = torch.clamp(
                earring_search_mask
                + dilate_mask(source_earring_mask + source_hoop_instance, 3),
                0,
                1,
            ) * active_earring_case
            earring_valid_roi = earring_valid_roi * active_earring_case

            # Per-side visibility gating: replace pixel-wise multiply (which clips
            # the completed earring back to the thin shell) with a per-side scalar
            # decision.  If a side is visible (valid_roi non-empty), keep the FULL
            # completed earring on that side; if occluded (valid_roi empty), zero
            # that side.  This prevents the shell from clipping the outer arc of a
            # large hoop or the lower portion of a long earring — the completion
            # logic already validated those pixels, so trust it rather than
            # re-clipping with the geometric shell.
            if earring_valid_roi is not None:
                left_earring_valid_roi = query_info.get("left_earring_valid_roi", query_info["left_ear_roi"])
                right_earring_valid_roi = query_info.get("right_earring_valid_roi", query_info["right_ear_roi"])

                height, width = source_earring_mask.shape[-2:]
                area_scale = float(height * width) / float(256 * 256)
                min_visible_area = 8.0 * area_scale  # ~8px² at 256res

                # The final target-only decisions above are stricter than the
                # broad query ROI and prevent hidden ears from becoming
                # positive earring supervision.
                left_visible = left_open
                right_visible = right_open

                # Gate by visibility (scalar decision per side, not pixel-wise multiply)
                left_mask = left_source_instance * left_visible.view(-1, 1, 1, 1).float()
                right_mask = right_source_instance * right_visible.view(-1, 1, 1, 1).float()
                left_regular_hole = left_regular_hole * left_visible.view(-1, 1, 1, 1).float()
                right_regular_hole = right_regular_hole * right_visible.view(-1, 1, 1, 1).float()
                left_hoop_instance = left_hoop_instance * left_visible.view(-1, 1, 1, 1).float()
                right_hoop_instance = right_hoop_instance * right_visible.view(-1, 1, 1, 1).float()
                left_hoop_hole = left_hoop_hole * left_visible.view(-1, 1, 1, 1).float()
                right_hoop_hole = right_hoop_hole * right_visible.view(-1, 1, 1, 1).float()

                source_earring_mask = torch.clamp(
                    left_mask + right_mask + left_hoop_instance + right_hoop_instance,
                    0,
                    1,
                ) * (
                    1.0
                    - torch.clamp(
                        left_regular_hole
                        + right_regular_hole
                        + left_hoop_hole
                        + right_hoop_hole,
                        0,
                        1,
                    )
                ).clamp(0, 1)
            else:
                left_mask, right_mask = left_source_instance, right_source_instance
                source_earring_mask = torch.clamp(
                    left_mask + right_mask + left_hoop_instance + right_hoop_instance,
                    0,
                    1,
                ) * (
                    1.0
                    - torch.clamp(
                        left_regular_hole
                        + right_regular_hole
                        + left_hoop_hole
                        + right_hoop_hole,
                        0,
                        1,
                    )
                ).clamp(0, 1)
            # The contour-verified hoop owns that ear.  Keeping a simultaneous
            # ordinary parser arc on the same side taught a double/thick ring
            # even though final inference correctly selects one instance.
            left_hoop_present = (
                left_hoop_instance.flatten(1).sum(dim=1, keepdim=True) >= 1.0
            ).to(source_earring_mask.dtype).view(-1, 1, 1, 1)
            right_hoop_present = (
                right_hoop_instance.flatten(1).sum(dim=1, keepdim=True) >= 1.0
            ).to(source_earring_mask.dtype).view(-1, 1, 1, 1)
            left_mask = left_mask * (1.0 - left_hoop_present)
            right_mask = right_mask * (1.0 - right_hoop_present)
            left_regular_hole = left_regular_hole * (1.0 - left_hoop_present)
            right_regular_hole = right_regular_hole * (1.0 - right_hoop_present)
            source_earring_mask = torch.clamp(
                left_mask + right_mask + left_hoop_instance + right_hoop_instance,
                0,
                1,
            ) * (
                1.0
                - torch.clamp(
                    left_regular_hole
                    + right_regular_hole
                    + left_hoop_hole
                    + right_hoop_hole,
                    0,
                    1,
                )
            ).clamp(0, 1)
            # A source accessory on a target-covered side is intentionally not
            # a recovery target.  Store this post-visibility truth so a zero
            # output alpha is not supervised as an inconsistent positive.
            active_earring_case = (
                source_earring_mask.flatten(1).sum(dim=1, keepdim=True) >= 1.0
            ).to(source_earring_mask.dtype).view(-1, 1, 1, 1)
            no_earring_case = (1.0 - active_earring_case).clamp(0, 1).expand_as(source_earring_mask)
            earring_valid_roi = earring_valid_roi * active_earring_case
            native_instance_256 = EarringNativeInstanceV6(
                alpha=torch.clamp(left_mask + right_mask, 0, 1),
                rgb=source_256,
                hole_alpha=torch.clamp(left_regular_hole + right_regular_hole, 0, 1),
                left_alpha=left_mask,
                right_alpha=right_mask,
                left_hole_alpha=left_regular_hole,
                right_hole_alpha=right_regular_hole,
                space=EarringCoordinateSpace.SOURCE_CANONICAL,
            )
            aligned_v6 = align_earring_instance_v6(
                native_instance_256,
                query_info["source_left_ear_mask"],
                query_info["source_right_ear_mask"],
                target_left_alignment,
                target_right_alignment,
                max_shift=max(0, int(self.args.earring_align_max_shift)),
            )
            aligned_alpha_v6 = aligned_v6["target_aligned_earring_alpha"]
            align_info = {
                "earring_confident_mask": aligned_alpha_v6,
                "hoop_hole_mask": aligned_v6["target_aligned_hole_alpha"],
                "earring_reference": (
                    target_256 * (1.0 - aligned_alpha_v6)
                    + aligned_v6["target_aligned_earring_rgb"] * aligned_alpha_v6
                ).clamp(0, 1),
                "coordinate_space": "TARGET_CANONICAL",
            }
            # Keep the aligned alpha/hole fields under one foreground contract.
            # The zero hoop compatibility fields above never grant RGB.
            # No hoop-specific final alignment in V19.  Keep zero topology
            # fields only for the legacy dataset schema reader.
            hoop_align_info = {
                "earring_confident_mask": torch.zeros_like(align_info["earring_confident_mask"]),
                "hoop_hole_mask": torch.zeros_like(align_info["hoop_hole_mask"]),
                "earring_reference": target_256,
            }
            # The target-side visibility decision is per side.  No target-hair
            # shell is applied to the complete source foreground alpha.
            normal_earring_mask = align_info["earring_confident_mask"] * active_earring_case
            aligned_regular_hole = align_info["hoop_hole_mask"] * active_earring_case
            aligned_hoop_instance = hoop_align_info["earring_confident_mask"] * active_earring_case
            aligned_hoop_hole = hoop_align_info["hoop_hole_mask"] * active_earring_case
            aligned_hole = torch.clamp(
                aligned_regular_hole + aligned_hoop_hole,
                0,
                1,
            )
            earring_confident_mask = (
                torch.maximum(normal_earring_mask, aligned_hoop_instance)
                * (1.0 - aligned_hole).clamp(0, 1)
                * active_earring_case
            ).clamp(0, 1)
            align_info["earring_confident_mask"] = earring_confident_mask
            aligned_reference = (
                target_256 * (1.0 - normal_earring_mask)
                + align_info["earring_reference"] * normal_earring_mask
            ).clamp(0, 1)
            align_info["earring_reference"] = (
                aligned_reference * (1.0 - aligned_hoop_instance)
                + hoop_align_info["earring_reference"] * aligned_hoop_instance
            ).clamp(0, 1)
            source_earring_mask = earring_confident_mask
            earring_highlight_mask = build_earring_highlight_mask(
                align_info["earring_reference"],
                earring_confident_mask,
                query_info["query_mask"],
            )
            # The learned branch must see the same complete source-native
            # object as the final compositor.  The old low-resolution write
            # permission is intentionally conservative and can end at the
            # lobe/root of a genuine stud or pendant; serialising it as the
            # learning target taught the network to reproduce that truncation.
            # ``align_info`` carries the per-side target-frame instance RGB
            # and the topology hole, so it does not include source hair or
            # background around the accessory.
            earring_learning_mask = earring_confident_mask
            earring_learning_hole = aligned_hole
            earring_learning_reference = align_info["earring_reference"]
            revealed_info = build_revealed_skin_mask(
                cleanup_masks,
                target_parsing,
                source_parsing,
                target_hair_d,
                source_hair_d,
                earring_confident_mask,
                source_256,
            )

            batch_size = source_256.size(0)
            dataset_items = []
            for idx in range(batch_size):
                item = {
                    "source_path": source_paths[idx],
                    "shape_reference_path": shape_reference_paths[idx],
                    "color_reference_path": color_reference_paths[idx],
                    "shape_reference": shape_reference_256[idx].cpu(),
                    "color_reference": color_reference_256[idx].cpu(),
                    "target": target_256[idx].cpu(),
                    # The completed high-resolution transfer is immutable
                    # final-output authority.  It is quantized only for the
                    # on-disk dataset item below.
                    "completed_hair_highres": png_uint8_tensor(
                        completed_hair_highres[idx]
                    ),
                    "satd_background_highres": png_uint8_tensor(
                        satd_background_highres[idx]
                    ),
                    "direct_satd_pp_input": torch.tensor(
                        bool(direct_satd_pp_input[idx].item()),
                        dtype=torch.bool,
                    ),
                    # Keep both stages for reproducible validation previews.  In
                    # schema 34 the PP target is the SATD-rendered image itself;
                    # the explicit flag below prevents any later residual pass.
                    "color_before_pp": target_256[idx].cpu(),
                    "target_mask": target_mask[idx].cpu(),
                    "HT_E": target_hair_e[idx].cpu(),
                    "source_parsing": source_parsing[idx].cpu(),
                    "target_parsing": target_parsing[idx].cpu(),
                    "source_hair_mask": query_info["source_hair_mask"][idx].cpu(),
                    "target_hair_mask": query_info["target_hair_mask"][idx].cpu(),
                    "source_hair_block_mask": source_hair_block_mask[idx].cpu(),
                    "source_earring_mask": source_earring_mask[idx].cpu(),
                    # This field is consumed by the final compositor as a
                    # SOURCE_NATIVE object alpha.  ``source_earring_mask``
                    # above is TARGET_CANONICAL (the 256px aligned training
                    # mask) and must never be reused for source extraction.
                    # The old code serialized the target mask under both
                    # names, which caused a second alignment and brought
                    # source background/duplicate earrings back at inference.
                    "source_earring_object_mask": source_foreground_v6[
                        "source_native_earring_alpha"
                    ][idx].cpu(),
                    # V5 coordinate contract: native extraction evidence
                    # and target-frame learned labels are separate fields.
                    "source_native_earring_alpha": source_foreground_v6["source_native_earring_alpha"][idx].cpu(),
                    "source_native_earring_rgb_reference": {
                        "kind": "source_path",
                        "path": source_paths[idx],
                        "space": "SOURCE_NATIVE",
                    },
                    "source_earring_presence": source_foreground_v6["source_native_presence_state"][idx].cpu(),
                    "source_earring_structured_fallback": structured_source_fallback[idx].cpu(),
                    "source_earring_structured_alpha": structured_source_alpha[idx].cpu(),
                    "target_aligned_earring_alpha": earring_confident_mask[idx].cpu(),
                    "target_aligned_earring_rgb": earring_learning_reference[idx].cpu(),
                    "coordinate_manifest": {
                        "source_earring_object_mask": "SOURCE_NATIVE",
                        "source_native_earring_alpha": "SOURCE_NATIVE",
                        "source_native_earring_rgb_reference": "SOURCE_NATIVE",
                        "target_aligned_earring_alpha": "TARGET_CANONICAL",
                        "target_aligned_earring_rgb": "TARGET_CANONICAL",
                    },
                    "earring_instance_mask": earring_confident_mask[idx].cpu(),
                    "earring_learning_mask": earring_learning_mask[idx].cpu(),
                    "earring_learning_reference": earring_learning_reference[idx].cpu(),
                    "earring_learning_hole_mask": earring_learning_hole[idx].cpu(),
                    "earring_presence_state": source_foreground_v6["source_native_presence_state"][idx].cpu(),
                    "earring_instance_confidence": source_foreground_v6["source_native_presence_score"][idx].cpu(),
                    "hoop_instance_mask": aligned_hoop_instance[idx].cpu(),
                    "source_earring_seed_mask": earring_policy["source_parser_earring_mask"][idx].cpu(),
                    "target_earring_mask": query_info["target_earring_mask"][idx].cpu(),
                    "earring_reference": align_info["earring_reference"][idx].cpu(),
                    "query_mask": query_info["query_mask"][idx].cpu(),
                    "earring_search_mask": earring_search_mask[idx].cpu(),
                    "earring_write_mask": source_earring_mask[idx].cpu(),
                    "earring_visible_segment_mask": earring_policy["earring_visible_segment_mask"][idx].cpu(),
                    # These masks are produced by the dataset policy at 256px.
                    # ``aligned_masks`` was removed when native-resolution
                    # instance extraction was added; keep the serialized
                    # fields sourced from the policy that created them.
                    "earring_core_mask": earring_policy["earring_core_mask"][idx].cpu(),
                    "earring_completion_mask": earring_policy["earring_completion_mask"][idx].cpu(),
                    "earring_object_mask": earring_policy["earring_object_mask"][idx].cpu(),
                    "earring_filled_mask": earring_policy["earring_filled_mask"][idx].cpu(),
                    "hoop_hole_mask": aligned_hole[idx].cpu(),
                    "strong_earring_candidate_core": earring_policy["strong_earring_candidate_core"][idx].cpu(),
                    "left_strong_candidate": earring_policy["left_strong_candidate"][idx].cpu(),
                    "right_strong_candidate": earring_policy["right_strong_candidate"][idx].cpu(),
                    "no_earring_case_mask": no_earring_case[idx].cpu(),
                    "target_hair_ear_bridge_mask": earring_policy["target_hair_ear_bridge_mask"][idx].cpu(),
                    "ear_roi": query_info["ear_roi"][idx].cpu(),
                    "visible_ear_roi": visible_ear_roi[idx].cpu(),
                    "earring_valid_roi": earring_valid_roi[idx].cpu(),
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
                        item[key] = earring_policy["earring_candidate_mask"][idx].cpu()
                    else:
                        item[key] = revealed_info[key][idx].cpu()
                dataset_items.append(item)

            yield dataset_items

            del batch
            del dataset_items
            del source_full, shape_reference_full, color_reference_full
            del target_full, pre_reference_color_full
            del source_256, shape_reference_256, color_reference_256
            del target_256, pre_reference_color_256
            del source_hair_d, target_hair_d, target_hair_e, target_mask
            del source_parsing, target_parsing, query_info, cleanup_masks
            del weak_earring, earring_policy, earring_search_mask, source_earring_mask
            del align_info, hoop_align_info, earring_confident_mask, earring_highlight_mask, revealed_info
            if self.device == "cuda":
                torch.cuda.empty_cache()


def _progress_path(output_dir):
    return Path(output_dir) / "gen_progress.json"


def load_progress(output_dir):
    """Resume state: completed chunk starts, next part index, corrupted stems."""
    path = _progress_path(output_dir)
    if not path.exists():
        return {"completed_left": [], "next_part_idx": 1, "corrupted": []}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {"completed_left": [], "next_part_idx": 1, "corrupted": []}
    data.setdefault("completed_left", [])
    data.setdefault("next_part_idx", 1)
    data.setdefault("corrupted", [])
    return data


def save_progress(output_dir, progress):
    path = _progress_path(output_dir)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=2)
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable_config(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_config(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_dataset_identity(args, model_args) -> dict[str, object]:
    """Fingerprint the exact target distribution used by resumable PP data."""

    repo_root = Path(__file__).resolve().parents[1]
    code_hashes = {}
    for relative in DATASET_POLICY_FILES:
        path = repo_root / relative
        if not path.exists():
            raise FileNotFoundError(f"Cannot fingerprint PP policy file: {path}")
        code_hashes[relative] = _sha256_file(path)

    checkpoint_values = {
        "stylegan": getattr(model_args, "ckpt", None),
        "rotate": getattr(model_args, "rotate_checkpoint", None),
        "blending": getattr(model_args, "blending_checkpoint", None),
        "pp_v6": getattr(model_args, "pp_v6_checkpoint", None),
        "satd_v8": (
            getattr(model_args, "satd_checkpoint_v8", None)
            if bool(getattr(model_args, "use_satd_v8", False))
            else None
        ),
        "e4e": repo_root / "pretrained_models/encoder4editing/e4e_ffhq_encode.pt",
    }
    checkpoints = {}
    for name, value in checkpoint_values.items():
        if value in {None, ""}:
            checkpoints[name] = None
            continue
        path = Path(value)
        if not path.is_absolute():
            path = repo_root / path
        resolved = path.resolve()
        checkpoints[name] = {
            "path": str(resolved),
            "sha256": _sha256_file(resolved) if resolved.exists() else None,
        }

    identity = {
        "schema_version": DATASET_CONFIG_SCHEMA_VERSION,
        "generator_args": _jsonable_config(vars(args)),
        "model_policy": {
            key: _jsonable_config(value)
            for key, value in vars(model_args).items()
            if key not in {
                "save_all",
                "save_all_dir",
                "device",
            }
        },
        "code_sha256": code_hashes,
        "checkpoints": checkpoints,
    }
    encoded = json.dumps(identity, ensure_ascii=True, sort_keys=True).encode("utf-8")
    identity["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return identity


def validate_pp_dataset_resume(args, model_args) -> None:
    """Refuse to mix old PP parts with the new colour/mask distribution."""

    output = Path(args.output)
    config_path = output / "dataset_config.json"
    identity = build_dataset_identity(args, model_args)
    # A progress file alone does not constitute a dataset.  Older generator
    # code could mark a run complete after every sampled triplet was skipped,
    # leaving only metadata/progress and no trainable part on disk.
    part_files_exist = any(output.glob("pp_part_*.dataset"))

    previous = None
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                previous = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            if part_files_exist:
                raise RuntimeError(
                    f"Cannot safely resume {output}: dataset_config.json is unreadable. "
                    "Use a fresh output directory for the new PP distribution."
                ) from error
    elif part_files_exist:
        raise RuntimeError(
            f"Cannot safely resume legacy PP data in {output}: dataset_config.json is missing. "
            "The new color_before_pp distribution must be generated into a fresh directory."
        )

    if previous is not None and part_files_exist:
        if previous.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError(
                f"Cannot mix PP dataset policies in {output}: code, checkpoint, or generation "
                "settings changed. Use a fresh output directory and regenerate all parts."
            )

    temporary = config_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(identity, handle, ensure_ascii=True, indent=2, sort_keys=True)
    temporary.replace(config_path)


def is_corrupt_image_error(error):
    text = str(error).lower()
    return (
        "decode" in text
        or "corrupt" in text
        or "out of bound" in text
        or "truncated" in text
        or isinstance(error, (OSError, ValueError))
    )


def main(args):
    if args.face_gallery_dir is None:
        raise ValueError("Please set USER_FACE_GALLERY_DIR_SMALL/FULL in the user config.")
    if args.donor_gallery_dir is None:
        raise ValueError("Please set USER_DONOR_GALLERY_DIR_SMALL/FULL in the user config.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if args.mask_batch_size <= 0:
        raise ValueError("--mask_batch_size must be positive")
    if args.dataset_compression not in {"gzip", "none"}:
        raise ValueError("--dataset_compression must be either 'gzip' or 'none'")
    if not 1 <= args.dataset_gzip_level <= 9:
        raise ValueError("--dataset_gzip_level must be in [1, 9]")
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
    model_args.allow_legacy_blending_checkpoint_v8 = args.allow_legacy_blending_checkpoint_v8
    model_args.use_satd_v8 = args.use_satd_v8
    model_args.direct_satd_pp_input = args.direct_satd_pp_input
    model_args.satd_checkpoint_v8 = args.satd_checkpoint_v8
    model_args.satd_blend_v8 = args.satd_blend_v8
    model_args.satd_boundary_v8 = args.satd_boundary_v8
    model_args.satd_background_cleanup_dilate = args.satd_background_cleanup_dilate
    model_args.satd_background_hair_edge_dilate = args.satd_background_hair_edge_dilate
    model_args.satd_background_hair_edge_support_dilate = args.satd_background_hair_edge_support_dilate
    model_args.satd_background_hair_edge_strength = args.satd_background_hair_edge_strength
    model_args.satd_background_residual_strength = args.satd_background_residual_strength
    model_args.satd_background_alpha_feather = args.satd_background_alpha_feather
    model_args.eq8_reference_blend_v8 = args.eq8_reference_blend_v8
    model_args.satd_hair_exclude_strength = args.satd_hair_exclude_strength
    model_args.target_hair_close_kernel = args.target_hair_close_kernel
    model_args.target_hair_hole_max_area_ratio = args.target_hair_hole_max_area_ratio
    model_args.target_hair_top_fill_only = args.target_hair_top_fill_only
    model_args.target_hair_ear_bridge_radius = args.target_hair_ear_bridge_radius
    model_args.hair_color_reference_strength_v8 = args.hair_color_reference_strength_v8
    model_args.hair_color_low_frequency_radius_v8 = args.hair_color_low_frequency_radius_v8
    model_args.hair_color_low_frequency_sigma_v8 = args.hair_color_low_frequency_sigma_v8
    model_args.hair_color_feather_radius_v8 = args.hair_color_feather_radius_v8
    model_args.hair_color_spatial_reference_weight_v8 = args.hair_color_spatial_reference_weight_v8
    model_args.hair_color_detail_chroma_gain_v8 = args.hair_color_detail_chroma_gain_v8
    model_args.disable_reference_dominant_hair_color_v8 = args.disable_reference_dominant_hair_color_v8
    model_args.blend_chroma_correct_strength = args.blend_chroma_correct_strength
    model_args.ear_low_alpha = args.ear_low_alpha
    model_args.ear_dilate = args.ear_dilate
    model_args.hair_change_dilate = args.hair_change_dilate
    model_args.earring_expand = args.earring_expand
    model_args.ear_downward_shift = args.ear_downward_shift
    model_args.target_hair_dilate = args.target_hair_dilate
    model_args.earring_occlusion_dilate = args.earring_occlusion_dilate
    model_args.source_hair_block_dilate = args.source_hair_block_dilate
    model_args.source_hair_block_strength = args.source_hair_block_strength
    model_args.target_visibility_expand = args.target_visibility_expand
    model_args.max_target_hair_overlap = args.max_target_hair_overlap
    model_args.min_target_visible_overlap = args.min_target_visible_overlap
    model_args.min_target_ear_area = args.min_target_ear_area
    model_args.earring_channel_down = args.earring_channel_down
    model_args.earring_align_max_shift = args.earring_align_max_shift
    model_args.earring_fine_mask_floor = args.earring_fine_mask_floor
    model_args.earring_fine_mask_dilate = args.earring_fine_mask_dilate
    model_args.earring_query_boost = args.earring_query_boost
    model_args.enable_earring_query_recall = args.enable_earring_query_recall
    model_args.earring_query_recall_dilate = args.earring_query_recall_dilate
    model_args.earring_query_downward_shift = args.earring_query_downward_shift
    model_args.earring_query_lower_lobe_weight = args.earring_query_lower_lobe_weight
    model_args.earring_query_candidate_boost = args.earring_query_candidate_boost
    model_args.earring_query_block_protect = args.earring_query_block_protect
    model_args.disable_earring_path_if_low_confidence = args.disable_earring_path_if_low_confidence
    model_args.earring_source_presence_min_area = args.earring_source_presence_min_area
    model_args.earring_search_downward_shift = args.earring_search_downward_shift
    model_args.earring_search_dilate = args.earring_search_dilate
    model_args.earring_write_max_target_hair_overlap = args.earring_write_max_target_hair_overlap
    model_args.earring_write_source_block_dilate = args.earring_write_source_block_dilate
    model_args.earring_write_dilate = args.earring_write_dilate
    model_args.earring_write_connectivity_iters = args.earring_write_connectivity_iters
    model_args.earring_write_connectivity_kernel = args.earring_write_connectivity_kernel
    model_args.earring_write_bridge_dilate = args.earring_write_bridge_dilate
    model_args.earring_anchor_visible_dilate = args.earring_anchor_visible_dilate
    model_args.ear_fine_support_dilate = 3
    validate_pp_dataset_resume(args, model_args)
    hair_fast = HairFast(model_args)
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
    if resolved_size > len(face_images):
        print(
            f"Source gallery has {len(face_images)} images; sampling {resolved_size} "
            "experiments with replacement, matching the legacy generator."
        )
    else:
        print(
            f"Source coverage: rendering {resolved_size} distinct target image(s) "
            f"from all {len(face_images)} image(s) in {args.face_gallery_dir}."
        )

    print(
        f"Using dataset_profile={args.dataset_profile}, source_dir={args.face_gallery_dir}, "
        f"donor_dir={args.donor_gallery_dir}, size={resolved_size}"
    )

    experiments = []
    for exp in sample_distinct_triplets(face_images, donor_images, resolved_size):
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

    # Resume state.  Each part is saved through an atomic replace, and a chunk is
    # marked complete only after every one of its parts is present.  If a process
    # stops mid-chunk, restart safely re-renders that chunk and overwrites its
    # deterministic part numbers.  Delete this file to regenerate from scratch.
    progress = load_progress(args.output)
    if progress["completed_left"] and not any(args.output.glob("pp_part_*.dataset")):
        # The previous run produced no usable data.  Retrying must not skip all
        # work simply because its stale progress file says the chunk completed.
        print("Resetting stale generation progress: no pp_part_*.dataset files exist.")
        progress = {"completed_left": [], "next_part_idx": 1, "corrupted": []}
    completed_left = set(int(value) for value in progress["completed_left"])
    part_idx = int(progress["next_part_idx"])
    corrupted = set(str(value) for value in progress["corrupted"])
    if completed_left:
        print(f"Resuming: {len(completed_left)} chunks already done, next part index {part_idx}, "
              f"{len(corrupted)} corrupted images recorded.")

    left = 0
    right = min(len(experiments), args.chunk_size)
    while left < len(experiments):
        if left in completed_left:
            left = right
            right = min(len(experiments), right + args.chunk_size)
            continue

        batch_experiments = experiments[left:right]
        skipped_in_chunk = 0
        chunk_started_at = time.perf_counter()
        # Keep only one mask batch of full-resolution source/donor images in
        # memory.  The old workflow rendered the complete chunk, wrote target
        # PNGs to a temporary directory, then re-opened every input and target.
        for group_left in range(0, len(batch_experiments), args.mask_batch_size):
            group_right = min(group_left + args.mask_batch_size, len(batch_experiments))
            rendered_experiments = []
            render_started_at = time.perf_counter()
            for item in tqdm(
                batch_experiments[group_left:group_right],
                desc=f"Render chunk {left + group_left}:{left + group_right}",
                leave=False,
            ):
                im1, im2, im3 = item["triplet"]
                triplet_stems = {Path(im1).stem, Path(im2).stem, Path(im3).stem}
                # Skip triplets referencing a known-corrupted image without retry.
                if triplet_stems & corrupted:
                    skipped_in_chunk += 1
                    continue
                try:
                    # Loading is the only stage where an image decode error is
                    # expected.  Model/label errors below must remain visible;
                    # treating every ValueError as a bad image could silently
                    # produce an empty "completed" dataset.
                    source_full = load_image(args.face_gallery_dir / im1)
                    shape_reference_full = load_image(args.donor_gallery_dir / im2)
                    color_reference_full = load_image(args.donor_gallery_dir / im3)
                except Exception as error:  # noqa: BLE001 - skip bad images, keep going
                    if is_corrupt_image_error(error):
                        corrupted |= {str(stem) for stem in triplet_stems}
                        skipped_in_chunk += 1
                        print(f"[skip] corrupted image in triplet {im1}, {im2}, {im3}: {error}")
                        continue
                    raise
                result = hair_fast(
                    source_full,
                    shape_reference_full,
                    color_reference_full,
                    return_stage="color_before_pp",
                    stop_before_pp=True,
                )
                (
                    image,
                    cleanup_masks,
                    pre_reference_color,
                    target_hair_mask,
                    completed_hair_highres,
                    satd_background_highres,
                    direct_satd_pp_input,
                ) = unpack_color_before_pp_stage(result)
                if bool(args.direct_satd_pp_input) != bool(direct_satd_pp_input):
                    raise RuntimeError(
                        "BlendingV6 returned a direct-SATD flag that does not "
                        "match --direct_satd_pp_input. Regenerate with the "
                        "matching V6 code/configuration."
                    )
                # The target used to be serialized to a temporary 8-bit PNG
                # and read back.  Preserve that numerical distribution in RAM
                # while avoiding the PNG encode/decode and filesystem latency.
                target_full = png_roundtrip_tensor(image)
                if torch.is_tensor(pre_reference_color):
                    pre_reference_color_full = png_roundtrip_tensor(pre_reference_color)
                else:
                    pre_reference_color_full = target_full
                rendered_item = {
                    **item,
                    "source_full": source_full,
                    "shape_reference_full": shape_reference_full,
                    "color_reference_full": color_reference_full,
                    "target_full": target_full,
                    "pre_reference_color_full": pre_reference_color_full,
                    "completed_hair_highres": png_uint8_tensor(completed_hair_highres),
                    "satd_background_highres": png_uint8_tensor(satd_background_highres),
                    "direct_satd_pp_input": torch.tensor(
                        bool(direct_satd_pp_input), dtype=torch.bool
                    ),
                    "cleanup_masks": cleanup_masks,
                }
                if target_hair_mask is not None:
                    rendered_item["target_hair_mask"] = target_hair_mask
                # Only successfully-rendered triplets go to the mask builder.
                rendered_experiments.append(rendered_item)

            render_seconds = time.perf_counter() - render_started_at
            label_started_at = time.perf_counter()
            if rendered_experiments:
                for dataset_items in item_batch_builder.iter_batches(
                    rendered_experiments,
                    args.output,
                    args.face_gallery_dir,
                    args.donor_gallery_dir,
                ):
                    part_path = args.output / f"pp_part_{part_idx}.dataset"
                    save_dataset_part(
                        dataset_items,
                        part_path,
                        args.dataset_compression,
                        args.dataset_gzip_level,
                    )
                    print(f"Saved {part_path} ({args.dataset_compression})")
                    part_idx += 1
            label_seconds = time.perf_counter() - label_started_at
            print(
                f"Chunk {left}:{right}, group {left + group_left}:{left + group_right}: "
                f"rendered {len(rendered_experiments)}, render {render_seconds:.1f}s, "
                f"labels {label_seconds:.1f}s."
            )
            del rendered_experiments

        # Chunk finished (all parts on disk): checkpoint the resume state.
        completed_left.add(left)
        progress["completed_left"] = sorted(completed_left)
        progress["next_part_idx"] = part_idx
        progress["corrupted"] = sorted(corrupted)
        save_progress(args.output, progress)
        if skipped_in_chunk:
            print(f"Chunk {left}:{right} done, skipped {skipped_in_chunk} corrupted triplet(s).")
        else:
            print(f"Chunk {left}:{right} completed in {time.perf_counter() - chunk_started_at:.1f}s.")

        left = right
        right = min(len(experiments), right + args.chunk_size)

    written_parts = list(args.output.glob("pp_part_*.dataset"))
    if not written_parts:
        raise RuntimeError(
            "PP dataset generation finished without writing any pp_part_*.dataset files. "
            f"All {len(experiments)} scheduled triplets were skipped or yielded no labels; "
            "inspect the preceding [skip] messages or the surfaced rendering error."
        )

    print(
        f"Generation complete: scheduled {len(experiments)} experiments, "
        f"wrote {part_idx - 1} dataset parts, "
        f"{len(corrupted)} corrupted images skipped."
    )


if __name__ == "__main__":
    parser = build_parser(RESOLVED_USER_CONFIG)
    main(parser.parse_args())
