import argparse
import faulthandler
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
import shutil
import sys
from argparse import Namespace
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from losses.pp_losses_v53 import EarAwareLossBuilder
from models.ear_modules_v53 import build_aligned_earring_reference, normalized_to_01
from models.Net import Net
from models.postprocess_v53 import PostProcessModelV53
from models.stylegan2 import dnnlib
from utils.bicubic import BicubicDownSample
from utils.train import WandbLogger, _LegacyUnpickler, get_fid_calc, image_grid, seed_everything, toggle_grad

faulthandler.enable(all_threads=True)

CLEANUP_MASK_KEYS = ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")
DERIVED_MASK_KEYS = ("revealed_skin_mask", "earring_confident_mask", "earring_highlight_mask")
VALIDATION_COLUMNS = (
    "source",
    "target",
    "gen_w",
    "gen_f",
    "gen_f_composite",
    "raw_query_src",
    "ear_detail_src",
    "source_ear_src",
    "ref_raw_src",
    "ref_src",
    "core_src",
    "safe_src",
    "floor_src",
    "dark_candidate_src",
    "dark_reject_src",
    "hair_reject_src",
    "object_src",
    "confident_src",
    "highlight_raw_src",
    "highlight_src",
    "learned_fine_src",
    "fine_src",
    "fine_floor_src",
    "face_reject_src",
    "inject_seed_src",
    "inject_src",
    "composite_seed_src",
    "output_guard_src",
    "face_guard_src",
    "hair_block_src",
    "cleanup_tgt",
    "revealed_tgt",
    "cleanup_face_tgt",
    "composite_alpha",
)
VALIDATION_MASK_STAT_KEYS = (
    "raw_query_mask",
    "ear_detail_query_mask",
    "source_earring_mask",
    "earring_reference_raw_mask",
    "earring_reference_mask",
    "earring_core_mask",
    "earring_safe_mask",
    "earring_recall_floor_mask",
    "earring_dark_candidate_mask",
    "earring_dark_reject_mask",
    "earring_hair_reject_mask",
    "earring_object_mask",
    "earring_confident_mask",
    "earring_highlight_raw_mask",
    "earring_highlight_mask",
    "learned_fine_mask",
    "fine_mask",
    "earring_fine_floor_mask",
    "earring_face_reject_mask",
    "earring_injection_seed_mask",
    "earring_composite_seed_mask",
    "earring_output_guard_alpha",
    "face_output_guard_alpha",
    "injection_fine_mask",
    "source_hair_block_mask",
    "cleanup_face_mask",
    "earring_composite_alpha",
)

# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small_accessory_ffhq"  # "small_accessory_ffhq" or "full_ffhq"

USER_DATASET_DIR_SMALL = Path("images/pp_dataset_v53_dual_small")
USER_OUTPUT_DIR_SMALL = Path("output/pp_v53_checkpoints_small")
USER_RUN_NAME_SMALL = "ear_refine_v53_dual_small"

USER_DATASET_DIR_FULL = Path("images/pp_dataset_v53_dual_full")
USER_OUTPUT_DIR_FULL = Path("output/pp_v53_checkpoints_full")
USER_RUN_NAME_FULL = "ear_refine_v53_dual_full"

USER_FID_DATASET = "fid_images"
USER_USE_FID = False
USER_USE_WANDB = False
USER_RESUME_CHECKPOINT = None
USER_BASE_CHECKPOINT = "pretrained_models/PostProcess/pp_model.pth"

USER_BATCH_SIZE = 8
USER_NUM_WORKERS = 0
USER_EPOCHS = 120
USER_VAL_SIZE = 250
USER_VAL_PREVIEW_COUNT = 50
USER_VAL_SHOW_GUARDED_COMPOSITE = True
USER_GRAD_ACCUM_STEPS = 2  # batch_size=8 with 2 accumulation steps gives an effective batch size of 16.

USER_TRAINING_STAGE = "ear_only"  # "ear_only", "cleanup_face", "joint_highres", or "full"
USER_PRETRAIN = False
USER_FINETUNE = False
USER_USE_MOD = True
USER_USE_FULL = True

USER_ITER_BEFORE_ADV = 10_000
USER_D_REG_EVERY = 16
USER_INPAINT = 0.0
USER_USE_ADV = False
USER_ADV_COEF = 0.05

USER_EAR_PARSE_SIZE = 512
USER_EAR_FEATURE_CHANNELS = 128
USER_EAR_LOW_ALPHA = 0.15
USER_EAR_DILATE = 21
USER_HAIR_CHANGE_DILATE = 25
USER_EARRING_EXPAND = 15
USER_EAR_DOWNWARD_SHIFT = 10
USER_TARGET_HAIR_DILATE = 11
USER_SOURCE_HAIR_BLOCK_DILATE = 5
USER_SOURCE_HAIR_BLOCK_STRENGTH = 0.95
USER_SOURCE_HAIR_BLOCK_HIGH_FLOOR = 0.45
USER_TARGET_VISIBILITY_EXPAND = 5
USER_MAX_TARGET_HAIR_OVERLAP = 0.55
USER_EARRING_LOBE_DILATE = 17
USER_EARRING_LOBE_DOWN_SHIFT = 18
USER_EARRING_OUTER_SHIFT = 10
USER_EARRING_QUERY_FLOOR = 0.45

USER_EAR_BLUR_KERNEL = 11
USER_EAR_BLUR_SIGMA = 3.0
USER_EAR_MASK_HIDDEN = 32
USER_EAR_MASK_INIT_BIAS = -4.0
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
USER_EARRING_QUERY_DILATE = 11
USER_EARRING_QUERY_BOOST = 1.75
USER_EARRING_TEXTURE_QUERY_BOOST = 1.0
USER_EARRING_FINE_MASK_FLOOR = 0.22
USER_EARRING_OBJECT_DILATE = 9
USER_EARRING_OBJECT_SUPPORT_DILATE = 13
USER_EARRING_ALIGN_TO_TARGET = True
USER_EARRING_ALIGN_STRENGTH = 1.0
USER_EARRING_ALIGN_MAX_SHIFT = 26
USER_EARRING_ATTACH_Y_RATIO = 0.78
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
USER_EARRING_GUARDED_COMPOSITE = False
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
USER_CLEANUP_FACE_HIDDEN = 128
USER_CLEANUP_FACE_STRENGTH = 1.0
USER_CLEANUP_FACE_DILATE = 5
USER_CLEANUP_FACE_EXCLUDE_EARRING_DILATE = 13

USER_LAMBDA_EAR_MASK = 2.0
USER_LAMBDA_EAR_HIGH = 3.0
USER_LAMBDA_EARRING_CONFIDENT_HIGH = 3.5
USER_LAMBDA_EARRING_HIGHLIGHT = 4.0
USER_LAMBDA_EARRING_COLOR = 1.2
USER_LAMBDA_EARRING_DIRECT = 8.0
USER_LAMBDA_EAR_PRESENCE = 1.0
USER_LAMBDA_EAR_LIGHTING = 0.5
USER_LAMBDA_EAR_HAIR_LEAK = 1.0
USER_LAMBDA_EAR_HAIR_ANCHOR = 2.0
USER_LAMBDA_EAR_CLEANUP_ANCHOR = 2.0
USER_LAMBDA_CLEANUP_ANCHOR = 0.25
USER_LAMBDA_CLEANUP_LOW_ANCHOR = 3.0
USER_LAMBDA_CLEANUP_SOURCE_REJECT = 4.0
USER_LAMBDA_CLEANUP_NON_DARK = 4.0
USER_LAMBDA_CLEANUP_DETAIL_HIGH = 0.0
USER_LAMBDA_CLEANUP_TEXTURE_STATS = 0.0
USER_LAMBDA_REVEALED_SKIN_TEXTURE = 1.2
USER_LAMBDA_REVEALED_SKIN_ENERGY = 0.8
USER_LAMBDA_REVEALED_SKIN_HIGH = 0.4
USER_LAMBDA_REVEALED_SKIN_SEAM = 0.8
USER_LAMBDA_CLEANUP_BACKGROUND_TEXTURE_STATS = 0.0
USER_LAMBDA_CLEANUP_FACE_LOW = 1.2
USER_LAMBDA_CLEANUP_FACE_TEXTURE = 1.0
USER_LAMBDA_CLEANUP_FACE_SEAM = 0.8
USER_LAMBDA_CLEANUP_FACE_SHADOW_REJECT = 1.0
USER_LAMBDA_FACE_TEXTURE_CONSISTENCY = 0.8
USER_LAMBDA_FACE_GUARD_LOW = 4.0
USER_LAMBDA_FACE_GUARD_NON_DARK = 6.0
USER_LAMBDA_FACE_GUARD_HIGH = 1.0
USER_LAMBDA_DETAIL_HIGH = 1.0
USER_LAMBDA_DETAIL_LOW_ANCHOR = 1.2
USER_LAMBDA_DETAIL_STRUCTURE = 0.5
USER_LAMBDA_EAR_QUERY_EXPAND = 0.05
USER_LAMBDA_EAR_MASK_AREA = 0.1
USER_LAMBDA_EAR_EDGE = 1.0
USER_LAMBDA_EAR_BRIGHTNESS_REG = 0.05
USER_LAMBDA_EAR_PRESERVE_DILATE = 5

USER_USE_DATASET_QUERY_MASK = False
USER_POSITIVE_ONLY_WARMUP_EPOCHS = 10
USER_POSITIVE_SAMPLE_WEIGHT = 4.0
USER_POSITIVE_QUERY_AREA_THRESHOLD = 96.0

USER_DATALOADER_START_METHOD = "auto"
USER_DATALOADER_PREFETCH_FACTOR = 1
USER_DATALOADER_PERSISTENT_WORKERS = True
USER_DATALOADER_PIN_MEMORY = False
# ============================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "small_accessory_ffhq": {
            "dataset": USER_DATASET_DIR_SMALL,
            "checkpoint_dir": USER_OUTPUT_DIR_SMALL,
            "name_run": USER_RUN_NAME_SMALL,
        },
        "full_ffhq": {
            "dataset": USER_DATASET_DIR_FULL,
            "checkpoint_dir": USER_OUTPUT_DIR_FULL,
            "name_run": USER_RUN_NAME_FULL,
        },
    }
    if USER_DATASET_PROFILE not in profiles:
        raise RuntimeError(
            f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}. "
            f"Choose one of: {', '.join(sorted(profiles))}."
        )
    return profiles[USER_DATASET_PROFILE]


PROFILE = resolve_dataset_profile()
ACTIVE_DATASET_DIR = PROFILE["dataset"]
ACTIVE_OUTPUT_DIR = PROFILE["checkpoint_dir"]
ACTIVE_RUN_NAME = PROFILE["name_run"]

RESOLVED_USER_CONFIG = {
    "dataset_profile": USER_DATASET_PROFILE,
    "name_run": ACTIVE_RUN_NAME,
    "dataset": ACTIVE_DATASET_DIR,
    "fid_dataset": USER_FID_DATASET,
    "batch_size": USER_BATCH_SIZE,
    "num_workers": USER_NUM_WORKERS,
    "epochs": USER_EPOCHS,
    "test_size": USER_VAL_SIZE,
    "val_preview_count": USER_VAL_PREVIEW_COUNT,
    "val_show_guarded_composite": USER_VAL_SHOW_GUARDED_COMPOSITE,
    "compute_fid": USER_USE_FID,
    "use_wandb": USER_USE_WANDB,
    "checkpoint_dir": ACTIVE_OUTPUT_DIR,
    "iter_before": USER_ITER_BEFORE_ADV,
    "d_reg_every": USER_D_REG_EVERY,
    "inpaint": USER_INPAINT,
    "use_adv": USER_USE_ADV,
    "adv_coef": USER_ADV_COEF,
    "base_checkpoint": USER_BASE_CHECKPOINT,
    "resume_checkpoint": USER_RESUME_CHECKPOINT,
    "use_mod": USER_USE_MOD,
    "use_full": USER_USE_FULL,
    "pretrain": USER_PRETRAIN,
    "finetune": USER_FINETUNE,
    "training_stage": USER_TRAINING_STAGE,
    "ear_parse_size": USER_EAR_PARSE_SIZE,
    "ear_feature_channels": USER_EAR_FEATURE_CHANNELS,
    "ear_low_alpha": USER_EAR_LOW_ALPHA,
    "ear_dilate": USER_EAR_DILATE,
    "hair_change_dilate": USER_HAIR_CHANGE_DILATE,
    "earring_expand": USER_EARRING_EXPAND,
    "ear_downward_shift": USER_EAR_DOWNWARD_SHIFT,
    "target_hair_dilate": USER_TARGET_HAIR_DILATE,
    "source_hair_block_dilate": USER_SOURCE_HAIR_BLOCK_DILATE,
    "source_hair_block_strength": USER_SOURCE_HAIR_BLOCK_STRENGTH,
    "source_hair_block_high_floor": USER_SOURCE_HAIR_BLOCK_HIGH_FLOOR,
    "target_visibility_expand": USER_TARGET_VISIBILITY_EXPAND,
    "max_target_hair_overlap": USER_MAX_TARGET_HAIR_OVERLAP,
    "earring_lobe_dilate": USER_EARRING_LOBE_DILATE,
    "earring_lobe_down_shift": USER_EARRING_LOBE_DOWN_SHIFT,
    "earring_outer_shift": USER_EARRING_OUTER_SHIFT,
    "earring_query_floor": USER_EARRING_QUERY_FLOOR,
    "ear_blur_kernel": USER_EAR_BLUR_KERNEL,
    "ear_blur_sigma": USER_EAR_BLUR_SIGMA,
    "ear_mask_hidden": USER_EAR_MASK_HIDDEN,
    "ear_mask_init_bias": USER_EAR_MASK_INIT_BIAS,
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
    "earring_query_dilate": USER_EARRING_QUERY_DILATE,
    "earring_query_boost": USER_EARRING_QUERY_BOOST,
    "earring_texture_query_boost": USER_EARRING_TEXTURE_QUERY_BOOST,
    "earring_fine_mask_floor": USER_EARRING_FINE_MASK_FLOOR,
    "earring_object_dilate": USER_EARRING_OBJECT_DILATE,
    "earring_object_support_dilate": USER_EARRING_OBJECT_SUPPORT_DILATE,
    "earring_align_to_target": USER_EARRING_ALIGN_TO_TARGET,
    "earring_align_strength": USER_EARRING_ALIGN_STRENGTH,
    "earring_align_max_shift": USER_EARRING_ALIGN_MAX_SHIFT,
    "earring_attach_y_ratio": USER_EARRING_ATTACH_Y_RATIO,
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
    "cleanup_face_hidden": USER_CLEANUP_FACE_HIDDEN,
    "cleanup_face_strength": USER_CLEANUP_FACE_STRENGTH,
    "cleanup_face_dilate": USER_CLEANUP_FACE_DILATE,
    "cleanup_face_exclude_earring_dilate": USER_CLEANUP_FACE_EXCLUDE_EARRING_DILATE,
    "ear_mask": USER_LAMBDA_EAR_MASK,
    "ear_high": USER_LAMBDA_EAR_HIGH,
    "earring_confident_high": USER_LAMBDA_EARRING_CONFIDENT_HIGH,
    "earring_highlight": USER_LAMBDA_EARRING_HIGHLIGHT,
    "earring_color": USER_LAMBDA_EARRING_COLOR,
    "earring_direct": USER_LAMBDA_EARRING_DIRECT,
    "ear_presence": USER_LAMBDA_EAR_PRESENCE,
    "ear_lighting": USER_LAMBDA_EAR_LIGHTING,
    "ear_hair_leak": USER_LAMBDA_EAR_HAIR_LEAK,
    "ear_hair_anchor": USER_LAMBDA_EAR_HAIR_ANCHOR,
    "ear_cleanup_anchor": USER_LAMBDA_EAR_CLEANUP_ANCHOR,
    "cleanup_anchor": USER_LAMBDA_CLEANUP_ANCHOR,
    "cleanup_low_anchor": USER_LAMBDA_CLEANUP_LOW_ANCHOR,
    "cleanup_source_reject": USER_LAMBDA_CLEANUP_SOURCE_REJECT,
    "cleanup_non_dark": USER_LAMBDA_CLEANUP_NON_DARK,
    "cleanup_detail_high": USER_LAMBDA_CLEANUP_DETAIL_HIGH,
    "cleanup_texture_stats": USER_LAMBDA_CLEANUP_TEXTURE_STATS,
    "revealed_skin_texture": USER_LAMBDA_REVEALED_SKIN_TEXTURE,
    "revealed_skin_energy": USER_LAMBDA_REVEALED_SKIN_ENERGY,
    "revealed_skin_high": USER_LAMBDA_REVEALED_SKIN_HIGH,
    "revealed_skin_seam": USER_LAMBDA_REVEALED_SKIN_SEAM,
    "cleanup_background_texture_stats": USER_LAMBDA_CLEANUP_BACKGROUND_TEXTURE_STATS,
    "cleanup_face_low": USER_LAMBDA_CLEANUP_FACE_LOW,
    "cleanup_face_texture": USER_LAMBDA_CLEANUP_FACE_TEXTURE,
    "cleanup_face_seam": USER_LAMBDA_CLEANUP_FACE_SEAM,
    "cleanup_face_shadow_reject": USER_LAMBDA_CLEANUP_FACE_SHADOW_REJECT,
    "face_texture_consistency": USER_LAMBDA_FACE_TEXTURE_CONSISTENCY,
    "face_guard_low": USER_LAMBDA_FACE_GUARD_LOW,
    "face_guard_non_dark": USER_LAMBDA_FACE_GUARD_NON_DARK,
    "face_guard_high": USER_LAMBDA_FACE_GUARD_HIGH,
    "detail_high": USER_LAMBDA_DETAIL_HIGH,
    "detail_low_anchor": USER_LAMBDA_DETAIL_LOW_ANCHOR,
    "detail_structure": USER_LAMBDA_DETAIL_STRUCTURE,
    "ear_query_expand": USER_LAMBDA_EAR_QUERY_EXPAND,
    "ear_mask_area": USER_LAMBDA_EAR_MASK_AREA,
    "ear_edge": USER_LAMBDA_EAR_EDGE,
    "ear_brightness_reg": USER_LAMBDA_EAR_BRIGHTNESS_REG,
    "ear_preserve_dilate": USER_LAMBDA_EAR_PRESERVE_DILATE,
    "use_dataset_query_mask": USER_USE_DATASET_QUERY_MASK,
    "positive_only_warmup_epochs": USER_POSITIVE_ONLY_WARMUP_EPOCHS,
    "positive_sample_weight": USER_POSITIVE_SAMPLE_WEIGHT,
    "positive_query_area_threshold": USER_POSITIVE_QUERY_AREA_THRESHOLD,
    "dataloader_start_method": USER_DATALOADER_START_METHOD,
    "dataloader_prefetch_factor": USER_DATALOADER_PREFETCH_FACTOR,
    "dataloader_persistent_workers": USER_DATALOADER_PERSISTENT_WORKERS,
    "dataloader_pin_memory": USER_DATALOADER_PIN_MEMORY,
    "grad_accum_steps": USER_GRAD_ACCUM_STEPS,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def str2path(value):
    return None if value in {None, "", "None"} else Path(value)


def build_parser(defaults):
    parser = argparse.ArgumentParser(description="Post Process trainer v53")
    parser.add_argument("--dataset_profile", type=str, default=defaults["dataset_profile"])
    parser.add_argument("--name_run", type=str, default=defaults["name_run"])
    parser.add_argument("--dataset", type=Path, default=defaults["dataset"])
    parser.add_argument("--fid_dataset", type=str, default=defaults["fid_dataset"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--test_size", type=int, default=defaults["test_size"])
    parser.add_argument("--val_preview_count", type=int, default=defaults["val_preview_count"])
    parser.add_argument("--val_show_guarded_composite", type=str2bool, default=defaults["val_show_guarded_composite"])
    parser.add_argument("--compute_fid", type=str2bool, default=defaults["compute_fid"])
    parser.add_argument("--use_wandb", type=str2bool, default=defaults["use_wandb"])
    parser.add_argument("--checkpoint_dir", type=Path, default=defaults["checkpoint_dir"])
    parser.add_argument("--iter_before", type=int, default=defaults["iter_before"])
    parser.add_argument("--d_reg_every", type=int, default=defaults["d_reg_every"])
    parser.add_argument("--inpaint", type=float, default=defaults["inpaint"])
    parser.add_argument("--use_adv", type=str2bool, default=defaults["use_adv"])
    parser.add_argument("--adv_coef", type=float, default=defaults["adv_coef"])
    parser.add_argument("--base_checkpoint", type=str, default=defaults["base_checkpoint"])
    parser.add_argument("--resume_checkpoint", type=str2path, default=defaults["resume_checkpoint"])
    parser.add_argument("--use_mod", type=str2bool, default=defaults["use_mod"])
    parser.add_argument("--use_full", type=str2bool, default=defaults["use_full"])
    parser.add_argument("--pretrain", type=str2bool, default=defaults["pretrain"])
    parser.add_argument("--finetune", type=str2bool, default=defaults["finetune"])
    parser.add_argument("--training_stage", type=str, default=defaults["training_stage"])
    parser.add_argument("--ear_parse_size", type=int, default=defaults["ear_parse_size"])
    parser.add_argument("--ear_feature_channels", type=int, default=defaults["ear_feature_channels"])
    parser.add_argument("--ear_low_alpha", type=float, default=defaults["ear_low_alpha"])
    parser.add_argument("--ear_dilate", type=int, default=defaults["ear_dilate"])
    parser.add_argument("--hair_change_dilate", type=int, default=defaults["hair_change_dilate"])
    parser.add_argument("--earring_expand", type=int, default=defaults["earring_expand"])
    parser.add_argument("--ear_downward_shift", type=int, default=defaults["ear_downward_shift"])
    parser.add_argument("--target_hair_dilate", type=int, default=defaults["target_hair_dilate"])
    parser.add_argument("--source_hair_block_dilate", type=int, default=defaults["source_hair_block_dilate"])
    parser.add_argument("--source_hair_block_strength", type=float, default=defaults["source_hair_block_strength"])
    parser.add_argument("--source_hair_block_high_floor", type=float, default=defaults["source_hair_block_high_floor"])
    parser.add_argument("--target_visibility_expand", type=int, default=defaults["target_visibility_expand"])
    parser.add_argument("--max_target_hair_overlap", type=float, default=defaults["max_target_hair_overlap"])
    parser.add_argument("--earring_lobe_dilate", type=int, default=defaults["earring_lobe_dilate"])
    parser.add_argument("--earring_lobe_down_shift", type=int, default=defaults["earring_lobe_down_shift"])
    parser.add_argument("--earring_outer_shift", type=int, default=defaults["earring_outer_shift"])
    parser.add_argument("--earring_query_floor", type=float, default=defaults["earring_query_floor"])
    parser.add_argument("--ear_blur_kernel", type=int, default=defaults["ear_blur_kernel"])
    parser.add_argument("--ear_blur_sigma", type=float, default=defaults["ear_blur_sigma"])
    parser.add_argument("--ear_mask_hidden", type=int, default=defaults["ear_mask_hidden"])
    parser.add_argument("--ear_mask_init_bias", type=float, default=defaults["ear_mask_init_bias"])
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
    parser.add_argument("--earring_query_dilate", type=int, default=defaults["earring_query_dilate"])
    parser.add_argument("--earring_query_boost", type=float, default=defaults["earring_query_boost"])
    parser.add_argument("--earring_texture_query_boost", type=float, default=defaults["earring_texture_query_boost"])
    parser.add_argument("--earring_fine_mask_floor", type=float, default=defaults["earring_fine_mask_floor"])
    parser.add_argument("--earring_object_dilate", type=int, default=defaults["earring_object_dilate"])
    parser.add_argument("--earring_object_support_dilate", type=int, default=defaults["earring_object_support_dilate"])
    parser.add_argument("--earring_align_to_target", type=str2bool, default=defaults["earring_align_to_target"])
    parser.add_argument("--earring_align_strength", type=float, default=defaults["earring_align_strength"])
    parser.add_argument("--earring_align_max_shift", type=int, default=defaults["earring_align_max_shift"])
    parser.add_argument("--earring_attach_y_ratio", type=float, default=defaults["earring_attach_y_ratio"])
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
    parser.add_argument("--cleanup_face_hidden", type=int, default=defaults["cleanup_face_hidden"])
    parser.add_argument("--cleanup_face_strength", type=float, default=defaults["cleanup_face_strength"])
    parser.add_argument("--cleanup_face_dilate", type=int, default=defaults["cleanup_face_dilate"])
    parser.add_argument("--cleanup_face_exclude_earring_dilate", type=int, default=defaults["cleanup_face_exclude_earring_dilate"])
    parser.add_argument("--ear_mask", type=float, default=defaults["ear_mask"])
    parser.add_argument("--ear_high", type=float, default=defaults["ear_high"])
    parser.add_argument("--earring_confident_high", type=float, default=defaults["earring_confident_high"])
    parser.add_argument("--earring_highlight", type=float, default=defaults["earring_highlight"])
    parser.add_argument("--earring_color", type=float, default=defaults["earring_color"])
    parser.add_argument("--earring_direct", type=float, default=defaults["earring_direct"])
    parser.add_argument("--ear_presence", type=float, default=defaults["ear_presence"])
    parser.add_argument("--ear_lighting", type=float, default=defaults["ear_lighting"])
    parser.add_argument("--ear_hair_leak", type=float, default=defaults["ear_hair_leak"])
    parser.add_argument("--ear_hair_anchor", type=float, default=defaults["ear_hair_anchor"])
    parser.add_argument("--ear_cleanup_anchor", type=float, default=defaults["ear_cleanup_anchor"])
    parser.add_argument("--cleanup_anchor", type=float, default=defaults["cleanup_anchor"])
    parser.add_argument("--cleanup_low_anchor", type=float, default=defaults["cleanup_low_anchor"])
    parser.add_argument("--cleanup_source_reject", type=float, default=defaults["cleanup_source_reject"])
    parser.add_argument("--cleanup_non_dark", type=float, default=defaults["cleanup_non_dark"])
    parser.add_argument("--cleanup_detail_high", type=float, default=defaults["cleanup_detail_high"])
    parser.add_argument("--cleanup_texture_stats", type=float, default=defaults["cleanup_texture_stats"])
    parser.add_argument("--revealed_skin_texture", type=float, default=defaults["revealed_skin_texture"])
    parser.add_argument("--revealed_skin_energy", type=float, default=defaults["revealed_skin_energy"])
    parser.add_argument("--revealed_skin_high", type=float, default=defaults["revealed_skin_high"])
    parser.add_argument("--revealed_skin_seam", type=float, default=defaults["revealed_skin_seam"])
    parser.add_argument("--cleanup_background_texture_stats", type=float, default=defaults["cleanup_background_texture_stats"])
    parser.add_argument("--cleanup_face_low", type=float, default=defaults["cleanup_face_low"])
    parser.add_argument("--cleanup_face_texture", type=float, default=defaults["cleanup_face_texture"])
    parser.add_argument("--cleanup_face_seam", type=float, default=defaults["cleanup_face_seam"])
    parser.add_argument("--cleanup_face_shadow_reject", type=float, default=defaults["cleanup_face_shadow_reject"])
    parser.add_argument("--face_texture_consistency", type=float, default=defaults["face_texture_consistency"])
    parser.add_argument("--face_guard_low", type=float, default=defaults["face_guard_low"])
    parser.add_argument("--face_guard_non_dark", type=float, default=defaults["face_guard_non_dark"])
    parser.add_argument("--face_guard_high", type=float, default=defaults["face_guard_high"])
    parser.add_argument("--detail_high", type=float, default=defaults["detail_high"])
    parser.add_argument("--detail_low_anchor", type=float, default=defaults["detail_low_anchor"])
    parser.add_argument("--detail_structure", type=float, default=defaults["detail_structure"])
    parser.add_argument("--ear_query_expand", type=float, default=defaults["ear_query_expand"])
    parser.add_argument("--ear_mask_area", type=float, default=defaults["ear_mask_area"])
    parser.add_argument("--ear_edge", type=float, default=defaults["ear_edge"])
    parser.add_argument("--ear_brightness_reg", type=float, default=defaults["ear_brightness_reg"])
    parser.add_argument("--ear_preserve_dilate", type=int, default=defaults["ear_preserve_dilate"])
    parser.add_argument("--use_dataset_query_mask", type=str2bool, default=defaults["use_dataset_query_mask"])
    parser.add_argument("--positive_only_warmup_epochs", type=int, default=defaults["positive_only_warmup_epochs"])
    parser.add_argument("--positive_sample_weight", type=float, default=defaults["positive_sample_weight"])
    parser.add_argument("--positive_query_area_threshold", type=float, default=defaults["positive_query_area_threshold"])
    parser.add_argument("--dataloader_start_method", type=str, default=defaults["dataloader_start_method"])
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=defaults["dataloader_prefetch_factor"])
    parser.add_argument("--dataloader_persistent_workers", type=str2bool, default=defaults["dataloader_persistent_workers"])
    parser.add_argument("--dataloader_pin_memory", type=str2bool, default=defaults["dataloader_pin_memory"])
    parser.add_argument("--grad_accum_steps", type=int, default=defaults["grad_accum_steps"])
    return parser


class NullLoggerV53:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.train_step = 0

    def start_logging(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def next_step(self):
        self.train_step += 1

    def log_scalars(self, scalars):
        return scalars

    def save(self, file_path, save_online=True):
        target = self.checkpoint_dir / Path(file_path).name
        if Path(file_path).resolve() != target.resolve():
            shutil.copy2(file_path, target)


def move_batch_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def accumulate_metrics(target, update):
    for key, value in update.items():
        target[key] = target.get(key, 0) + (value.detach() if torch.is_tensor(value) else value)
    return target


def configure_training_stage(model: PostProcessModelV53, args):
    for param in model.parameters():
        param.requires_grad = False

    if args.training_stage == "ear_only":
        modules = [
            model.hf_extractor,
            model.mask_refresher,
            model.brightness_reestimator,
            model.ear_injector_64,
            model.ear_injector_128,
        ]
    elif args.training_stage == "cleanup_face":
        modules = [
            model.cleanup_face_refiner_64,
            model.cleanup_face_refiner_128,
        ]
    elif args.training_stage == "joint_highres":
        modules = [
            model.to_feature,
            model.hf_extractor,
            model.mask_refresher,
            model.brightness_reestimator,
            model.ear_injector_64,
            model.ear_injector_128,
        ]
    elif args.training_stage == "full":
        for param in model.parameters():
            param.requires_grad = True
        if not args.finetune:
            toggle_grad(model.encoder_face, False)
        return
    else:
        raise ValueError(f"Unsupported training_stage: {args.training_stage}")

    for module in modules:
        toggle_grad(module, True)


def eval_if_frozen(module):
    if module is not None and not any(param.requires_grad for param in module.parameters()):
        module.eval()


class TrainerV53:
    def __init__(
        self,
        model=None,
        args=None,
        optimizer=None,
        train_dataloader=None,
        train_dataloader_positive=None,
        test_dataloader=None,
        logger=None,
    ):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.train_dataloader_positive = train_dataloader_positive
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.net = Net(Namespace(size=1024, ckpt="pretrained_models/StyleGAN/ffhq.pt", channel_multiplier=2,
                                 latent=512, n_mlp=8, device=self.device))
        with dnnlib.util.open_url("pretrained_models/StyleGAN/ffhq.pkl") as file:
            data = _LegacyUnpickler(file).load()
        self.discriminator = data["D"].to(self.device).eval()
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=3e-4, betas=(0.9, 0.999),
                                           amsgrad=False, weight_decay=0)

        toggle_grad(self.discriminator, False)
        toggle_grad(self.net.generator, False)

        self.downsample_256 = BicubicDownSample(factor=4)
        self.best_loss = float("+inf")
        self.cur_iter = 1
        self.current_epoch = 0

        self.loss_builder = None
        self.fid_calc = None
        if args is not None:
            ear_only_stage = args.training_stage == "ear_only"
            cleanup_face_stage = args.training_stage == "cleanup_face"
            cleanup_general_stage = args.training_stage in {"joint_highres", "full"}
            loss_weights = {
                "lpips_scale": 0.8,
                "id": 0.1,
                "landmark": 0.1 if not args.pretrain else 0.0,
                "feat_rec": 0.01,
                "adv": args.adv_coef,
                "inpaint": args.inpaint,
                "training_stage": args.training_stage,
                "ear_mask": 0.0 if cleanup_face_stage else args.ear_mask,
                "ear_high": 0.0 if cleanup_face_stage else args.ear_high,
                "earring_confident_high": 0.0 if cleanup_face_stage else args.earring_confident_high,
                "earring_highlight": 0.0 if cleanup_face_stage else args.earring_highlight,
                "earring_color": 0.0 if cleanup_face_stage else args.earring_color,
                "earring_direct": 0.0 if cleanup_face_stage else args.earring_direct,
                "ear_presence": 0.0 if cleanup_face_stage else args.ear_presence,
                "ear_lighting": 0.0 if cleanup_face_stage else args.ear_lighting,
                "ear_hair_leak": 0.0 if cleanup_face_stage else args.ear_hair_leak,
                "ear_hair_anchor": 0.0 if cleanup_face_stage else args.ear_hair_anchor,
                "ear_cleanup_anchor": 0.0 if cleanup_face_stage else args.ear_cleanup_anchor,
                "base_cleanup_exclude": 0.0 if ear_only_stage else 1.0,
                "cleanup_anchor": args.cleanup_anchor if cleanup_general_stage else 0.0,
                "cleanup_low_anchor": args.cleanup_low_anchor if cleanup_general_stage else 0.0,
                "cleanup_source_reject": args.cleanup_source_reject if cleanup_general_stage else 0.0,
                "cleanup_non_dark": args.cleanup_non_dark if cleanup_general_stage else 0.0,
                "cleanup_detail_high": args.cleanup_detail_high if cleanup_general_stage else 0.0,
                "cleanup_texture_stats": args.cleanup_texture_stats if cleanup_general_stage else 0.0,
                "revealed_skin_texture": args.revealed_skin_texture if cleanup_general_stage else 0.0,
                "revealed_skin_energy": args.revealed_skin_energy if cleanup_general_stage else 0.0,
                "revealed_skin_high": args.revealed_skin_high if cleanup_general_stage else 0.0,
                "revealed_skin_seam": args.revealed_skin_seam if cleanup_general_stage else 0.0,
                "cleanup_background_texture_stats": (
                    args.cleanup_background_texture_stats if cleanup_general_stage else 0.0
                ),
                "cleanup_face_low": args.cleanup_face_low if cleanup_face_stage else 0.0,
                "cleanup_face_texture": args.cleanup_face_texture if cleanup_face_stage else 0.0,
                "cleanup_face_seam": args.cleanup_face_seam if cleanup_face_stage else 0.0,
                "cleanup_face_shadow_reject": args.cleanup_face_shadow_reject if cleanup_face_stage else 0.0,
                "face_texture_consistency": 0.0 if cleanup_face_stage else args.face_texture_consistency,
                "face_guard_low": 0.0 if cleanup_face_stage else args.face_guard_low,
                "face_guard_non_dark": 0.0 if cleanup_face_stage else args.face_guard_non_dark,
                "face_guard_high": 0.0 if cleanup_face_stage else args.face_guard_high,
                "detail_high": 0.0 if cleanup_face_stage else args.detail_high,
                "detail_low_anchor": 0.0 if cleanup_face_stage else args.detail_low_anchor,
                "detail_structure": 0.0 if cleanup_face_stage else args.detail_structure,
                "ear_query_expand": 0.0 if cleanup_face_stage else args.ear_query_expand,
                "ear_mask_area": 0.0 if cleanup_face_stage else args.ear_mask_area,
                "ear_block_strength": args.source_hair_block_strength,
                "ear_edge": 0.0 if cleanup_face_stage else args.ear_edge,
                "ear_brightness_reg": 0.0 if cleanup_face_stage else args.ear_brightness_reg,
                "ear_preserve_dilate": args.ear_preserve_dilate,
            }
            self.loss_builder = EarAwareLossBuilder(loss_weights, device=self.device)
            if args.compute_fid:
                self.fid_calc = get_fid_calc("input/fid.pkl", args.fid_dataset)

    def save_model(self, name):
        save_path = self.args.checkpoint_dir / f"{name}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "D": self.discriminator.state_dict(),
                "cur_iter": self.cur_iter,
                "args": vars(self.args),
            },
            save_path,
        )
        if self.logger is not None:
            self.logger.save(str(save_path), save_online=False)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "D" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["D"], strict=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.cur_iter = checkpoint.get("cur_iter", self.cur_iter)

    def save_validation_images(self, files, epoch_tag):
        if not files:
            return

        vis_dir = self.args.checkpoint_dir / "val_images" / epoch_tag
        vis_dir.mkdir(parents=True, exist_ok=True)
        with open(vis_dir / "columns.txt", "w", encoding="utf-8") as file:
            file.write(" | ".join(VALIDATION_COLUMNS) + "\n")

        preview_count = max(1, int(getattr(self.args, "val_preview_count", 20)))
        np.random.seed(1927)
        indices = np.random.choice(len(files), size=min(len(files), preview_count), replace=False)
        for order, idx in enumerate(indices):
            image = self.labeled_validation_grid(files[idx])
            image.save(vis_dir / f"val_{order:03d}.png")

    @staticmethod
    def labeled_validation_grid(row_tensors):
        images = list(map(T.functional.to_pil_image, row_tensors))
        if len(images) != len(VALIDATION_COLUMNS):
            raise RuntimeError(
                f"Validation row has {len(images)} images, but {len(VALIDATION_COLUMNS)} columns are declared."
            )

        w, h = images[0].size
        header_h = 28
        labeled_tiles = []
        for label, image in zip(VALIDATION_COLUMNS, images):
            tile = Image.new("RGB", (w, h + header_h), (255, 255, 255))
            tile.paste(image.convert("RGB"), (0, header_h))
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, w, header_h), fill=(20, 20, 20))
            text = str(label)
            while len(text) > 4:
                try:
                    text_w = draw.textbbox((0, 0), text)[2]
                except AttributeError:
                    text_w = draw.textsize(text)[0]
                if text_w <= w - 8:
                    break
                text = text[:-2] + "."
            draw.text((4, 7), text, fill=(255, 255, 255))
            labeled_tiles.append(tile)
        return image_grid(labeled_tiles, 1, len(labeled_tiles))

    @staticmethod
    def update_preview_buffer(buffer, sample, seen_count, max_count):
        if max_count <= 0:
            return seen_count

        seen_count += 1
        if len(buffer) < max_count:
            buffer.append(sample)
            return seen_count

        replace_idx = random.randint(0, seen_count - 1)
        if replace_idx < max_count:
            buffer[replace_idx] = sample
        return seen_count

    @staticmethod
    def mask_to_rgb(mask, size):
        if mask is None:
            return torch.zeros(3, *size, dtype=torch.float32)

        mask = mask.detach().float().cpu()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.size(0) != 1:
            mask = mask[:1]

        if tuple(mask.shape[-2:]) != tuple(size):
            mask = T.functional.resize(mask, list(size), interpolation=T.InterpolationMode.BILINEAR)
        return mask.clamp(0, 1).repeat(3, 1, 1)

    @staticmethod
    def mask_to_cutout(mask, image, size):
        if image is None:
            return TrainerV53.mask_to_rgb(mask, size)

        image = image.detach().float().cpu().clamp(0, 1)
        if image.ndim == 4:
            image = image[0]
        if tuple(image.shape[-2:]) != tuple(size):
            image = T.functional.resize(image, list(size), interpolation=T.InterpolationMode.BILINEAR)

        if mask is None:
            return image * 0.18

        mask = mask.detach().float().cpu()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.size(0) != 1:
            mask = mask[:1]
        if tuple(mask.shape[-2:]) != tuple(size):
            mask = T.functional.resize(mask, list(size), interpolation=T.InterpolationMode.BILINEAR)

        mask = mask.clamp(0, 1)
        cutout = image * (0.18 + 0.82 * mask)
        return cutout.clamp(0, 1)

    @staticmethod
    def combined_cleanup_mask(aux):
        masks = []
        for key in CLEANUP_MASK_KEYS:
            value = aux.get(key)
            if value is not None:
                masks.append(value.detach().float())
        if not masks:
            return None
        return torch.stack(masks, dim=0).amax(dim=0).clamp(0, 1)

    @staticmethod
    def accumulate_mask_pixel_stats(target, aux, batch_size):
        for key in VALIDATION_MASK_STAT_KEYS:
            value = aux.get(key)
            if value is None:
                target[key] = target.get(key, 0.0)
                continue
            value = value.detach().float()
            if value.ndim == 2:
                value = value.unsqueeze(0).unsqueeze(0)
            elif value.ndim == 3:
                value = value.unsqueeze(1)
            per_sample = value.flatten(1).sum(dim=1)
            target[key] = target.get(key, 0.0) + per_sample.sum().item()
        return batch_size

    def build_val_guarded_composite(self, gen_im_F, aux, batch):
        if not bool(getattr(self.args, "val_show_guarded_composite", True)):
            return gen_im_F
        if bool(getattr(self.args, "earring_guarded_composite", False)):
            return gen_im_F
        if not hasattr(self.model, "_apply_guarded_earring_composite"):
            return gen_im_F
        core_mask = aux.get(
            "earring_injection_seed_mask",
            aux.get("earring_safe_mask", aux.get("earring_core_mask", aux.get("earring_reference_mask"))),
        )
        raw_reference_mask = aux.get(
            "source_earring_parser_mask",
            aux.get("source_earring_raw_mask", aux.get("source_earring_mask")),
        )
        if core_mask is None or raw_reference_mask is None or aux.get("earring_reference_mask") is None:
            return gen_im_F
        if core_mask.detach().float().flatten(1).amax(dim=1).sum().item() <= 0:
            return gen_im_F

        source_full_01 = normalized_to_01(batch["source"]).to(self.device)
        reference_full, reference_mask_full = build_aligned_earring_reference(
            source_full_01,
            raw_reference_mask,
            aux.get("target_parsing"),
            aux.get("visible_ear_roi"),
            align_strength=float(getattr(self.args, "earring_align_strength", 1.0)),
            max_shift=int(getattr(self.args, "earring_align_max_shift", 26)) * 4,
            attach_y_ratio=float(getattr(self.args, "earring_attach_y_ratio", 0.78)),
        )
        preview_aux = dict(aux)
        preview_aux["earring_reference_image_full"] = reference_full
        preview_aux["earring_reference_mask_full"] = reference_mask_full
        preview_aux["earring_core_mask_full"] = core_mask
        output = self.model._apply_guarded_earring_composite(gen_im_F, preview_aux)
        if preview_aux.get("earring_composite_alpha") is not None:
            aux["earring_composite_alpha"] = preview_aux["earring_composite_alpha"]
        return output

    def _run_model(self, batch):
        batch = move_batch_to_device(batch, self.device)
        source_full = batch["source"]
        source = self.downsample_256(source_full).clip(0, 1)
        target = batch["target"]
        target_mask = batch["target_mask"]
        HT_E = batch["HT_E"]
        source_ear_mask = batch["source_earring_mask"]
        if "earring_confident_mask" in batch:
            source_ear_mask = torch.clamp(source_ear_mask + batch["earring_confident_mask"], 0, 1)

        latent_s, latent_f, aux = self.model(
            self.normalize(source),
            self.normalize(target),
            target_mask,
            HT_E,
            source_parsing=batch["source_parsing"],
            target_parsing=batch["target_parsing"],
            source_hair_mask=batch["source_hair_mask"],
            target_hair_mask=batch["target_hair_mask"],
            query_mask=batch["query_mask"],
            source_ear_mask=source_ear_mask,
            presence_target=batch["presence_target"],
            cleanup_masks={key: batch[key] for key in CLEANUP_MASK_KEYS if key in batch},
            revealed_skin_mask=batch.get("revealed_skin_mask"),
        )
        for key in CLEANUP_MASK_KEYS:
            aux[key] = batch[key]
        for key in DERIVED_MASK_KEYS:
            aux[f"dataset_{key}"] = batch[key]
            if key not in aux:
                aux[key] = batch[key]

        gen_im_W, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False)
        F_w, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False, start_layer=0, end_layer=4)

        if self.args.pretrain:
            alpha = min(1, self.cur_iter / self.args.iter_before)
            latent_f_gen = alpha * latent_f + (1 - alpha) * F_w
        else:
            latent_f_gen = latent_f

        aux["base_image_full"] = gen_im_W.detach()
        aux["source_full_01"] = normalized_to_01(batch["source"]).to(self.device)
        gen_im_F, aux = self.model.render_refined(self.net.generator, latent_s, latent_f_gen, aux)
        return source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux, batch

    def train_one_epoch(self):
        self.model.to(self.device).train()
        eval_if_frozen(self.model.encoder_face)
        eval_if_frozen(self.model.to_feature)
        eval_if_frozen(getattr(self.model, "to_latent_1", None))
        eval_if_frozen(getattr(self.model, "to_latent_2", None))
        eval_if_frozen(getattr(self.model, "to_latent", None))
        eval_if_frozen(getattr(self.model, "cleanup_face_refiner_64", None))
        eval_if_frozen(getattr(self.model, "cleanup_face_refiner_128", None))
        loader = self.train_dataloader
        if self.train_dataloader_positive is not None and self.current_epoch < self.args.positive_only_warmup_epochs:
            loader = self.train_dataloader_positive

        accum_steps = max(1, int(self.args.grad_accum_steps))
        total_batches = len(loader)
        accum_in_group = 0
        current_group_size = min(accum_steps, total_batches) if total_batches > 0 else accum_steps
        step_metrics = {}

        self.optimizer.zero_grad(set_to_none=True)
        self.disc_optim.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(tqdm(loader)):
            if accum_in_group == 0:
                remaining_batches = total_batches - batch_idx
                current_group_size = min(accum_steps, remaining_batches)
                step_metrics = {}
                self.optimizer.zero_grad(set_to_none=True)
                self.disc_optim.zero_grad(set_to_none=True)

            source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux, batch = self._run_model(batch)

            losses = self.loss_builder(source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux=aux)
            if self.args.use_adv and self.cur_iter >= self.args.iter_before:
                losses.update(self.loss_builder.CalcAdvLoss(self.discriminator, gen_im_F))

            losses["loss"] = sum(losses.values())
            accumulate_metrics(step_metrics, losses)
            (losses["loss"] / current_group_size).backward()

            disc_grad = None
            if self.args.use_adv and self.cur_iter >= self.args.iter_before:
                toggle_grad(self.discriminator, True)
                self.discriminator.train()

                source_1024 = self.normalize(batch["source"].to(self.device))
                disc_loss = self.loss_builder.CalcDisLoss(self.discriminator, source_1024, gen_im_F.detach())
                if self.cur_iter % self.args.d_reg_every:
                    disc_loss.update(self.loss_builder.CalcR1Loss(self.discriminator, source_1024))

                total_disc_loss = sum(disc_loss.values())
                accumulate_metrics(step_metrics, disc_loss)
                (total_disc_loss / current_group_size).backward()
                toggle_grad(self.discriminator, False)
                self.discriminator.eval()

            accum_in_group += 1
            should_step = accum_in_group >= current_group_size
            if not should_step:
                continue

            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            if self.args.use_adv and self.cur_iter >= self.args.iter_before:
                disc_grad = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
                self.disc_optim.step()
                step_metrics["grad disc"] = disc_grad

            step_metrics["train grad"] = total_norm
            self.logger.next_step()
            self.logger.log_scalars(
                {
                    f"train {key}": (
                        value.item() / current_group_size if torch.is_tensor(value) else value / current_group_size
                    )
                    for key, value in step_metrics.items()
                    if key not in {"train grad", "grad disc"}
                }
                | {
                    "train grad": total_norm.item() if torch.is_tensor(total_norm) else total_norm,
                    **(
                        {"train grad disc": disc_grad.item() if torch.is_tensor(disc_grad) else disc_grad}
                        if disc_grad is not None else {}
                    ),
                }
            )
            self.cur_iter += 1
            accum_in_group = 0

    @torch.no_grad()
    def validate(self, epoch_tag="initial"):
        self.model.to(self.device).eval()
        val_losses = {}
        val_mask_pixels = {}
        val_mask_samples = 0
        preview_files = []
        preview_count = max(1, int(getattr(self.args, "val_preview_count", 20)))
        preview_seen = 0
        to_299 = T.Resize((299, 299))
        images_to_fid = []

        def accumulate(target, update):
            for key, value in update.items():
                target[key] = target.get(key, 0) + value
            return target

        for batch in tqdm(self.test_dataloader):
            source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux, batch = self._run_model(batch)
            losses = self.loss_builder(source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux=aux)
            losses["loss"] = sum(losses.values())
            val_losses = accumulate(val_losses, losses)

            gen_w_256 = self.downsample_256((gen_im_W + 1) / 2).clip(0, 1)
            gen_f_256 = self.downsample_256((gen_im_F + 1) / 2).clip(0, 1)
            gen_im_F_composite = self.build_val_guarded_composite(gen_im_F, aux, batch)
            gen_f_composite_256 = self.downsample_256((gen_im_F_composite + 1) / 2).clip(0, 1)
            val_mask_samples += self.accumulate_mask_pixel_stats(val_mask_pixels, aux, source.size(0))
            if self.fid_calc is not None:
                images_to_fid.append(to_299((gen_im_F + 1) / 2).clip(0, 1))

            cleanup_debug = self.combined_cleanup_mask(aux)
            for idx in range(source.size(0)):
                size = tuple(source.shape[-2:])
                preview_row = [
                    source[idx].cpu(),
                    target[idx].cpu(),
                    gen_w_256[idx].cpu(),
                    gen_f_256[idx].cpu(),
                    gen_f_composite_256[idx].cpu(),
                    self.mask_to_cutout(aux.get("raw_query_mask")[idx] if aux.get("raw_query_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("ear_detail_query_mask")[idx] if aux.get("ear_detail_query_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("source_earring_mask")[idx] if aux.get("source_earring_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_reference_raw_mask")[idx] if aux.get("earring_reference_raw_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_reference_mask")[idx] if aux.get("earring_reference_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_core_mask")[idx] if aux.get("earring_core_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_safe_mask")[idx] if aux.get("earring_safe_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_recall_floor_mask")[idx] if aux.get("earring_recall_floor_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_dark_candidate_mask")[idx] if aux.get("earring_dark_candidate_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_dark_reject_mask")[idx] if aux.get("earring_dark_reject_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_hair_reject_mask")[idx] if aux.get("earring_hair_reject_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_object_mask")[idx] if aux.get("earring_object_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_confident_mask")[idx] if aux.get("earring_confident_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_highlight_raw_mask")[idx] if aux.get("earring_highlight_raw_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_highlight_mask")[idx] if aux.get("earring_highlight_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("learned_fine_mask")[idx] if aux.get("learned_fine_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("fine_mask")[idx] if aux.get("fine_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_fine_floor_mask")[idx] if aux.get("earring_fine_floor_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_face_reject_mask")[idx] if aux.get("earring_face_reject_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_injection_seed_mask")[idx] if aux.get("earring_injection_seed_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("injection_fine_mask")[idx] if aux.get("injection_fine_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_composite_seed_mask")[idx] if aux.get("earring_composite_seed_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("earring_output_guard_alpha")[idx] if aux.get("earring_output_guard_alpha") is not None else None, gen_f_256[idx], size),
                    self.mask_to_cutout(aux.get("face_output_guard_alpha")[idx] if aux.get("face_output_guard_alpha") is not None else None, source[idx], size),
                    self.mask_to_cutout(aux.get("source_hair_block_mask")[idx] if aux.get("source_hair_block_mask") is not None else None, source[idx], size),
                    self.mask_to_cutout(cleanup_debug[idx] if cleanup_debug is not None else None, target[idx], size),
                    self.mask_to_cutout(aux.get("revealed_skin_mask")[idx] if aux.get("revealed_skin_mask") is not None else None, target[idx], size),
                    self.mask_to_cutout(aux.get("cleanup_face_mask")[idx] if aux.get("cleanup_face_mask") is not None else None, target[idx], size),
                    self.mask_to_cutout(aux.get("earring_composite_alpha")[idx] if aux.get("earring_composite_alpha") is not None else None, gen_f_composite_256[idx], size),
                ]
                preview_seen = self.update_preview_buffer(
                    preview_files,
                    preview_row,
                    preview_seen,
                    preview_count,
                )

            del source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, gen_im_F_composite, latent_f, aux
            del gen_w_256, gen_f_256, gen_f_composite_256, cleanup_debug
            if self.device == "cuda":
                torch.cuda.empty_cache()

        if self.fid_calc is not None and images_to_fid:
            val_losses["FID CLIP"] = self.fid_calc(torch.cat(images_to_fid))

        for key, value in val_losses.items():
            if key != "FID CLIP":
                value = value.item() / max(1, len(self.test_dataloader))
            self.logger.log_scalars({f"val {key}": value})
        for key, value in val_mask_pixels.items():
            self.logger.log_scalars({f"val mask_pixels/{key}": value / max(1, val_mask_samples)})
        if val_mask_pixels:
            stats_line = ", ".join(
                f"{key}={value / max(1, val_mask_samples):.1f}"
                for key, value in val_mask_pixels.items()
            )
            print(f"[validate:{epoch_tag}] mean mask pixels: {stats_line}")

        if preview_files:
            self.save_validation_images(preview_files, epoch_tag)
            np.random.seed(1927)
            indices = np.random.choice(len(preview_files), size=min(len(preview_files), preview_count), replace=False)
            images_to_log = [self.labeled_validation_grid(preview_files[idx]) for idx in indices]
            self.logger.log_scalars({"val images": [wandb.Image(image) for image in images_to_log]})

        return val_losses["loss"]

    def train_loop(self):
        self.validate("epoch_0000_initial")
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            loss = self.validate(f"epoch_{epoch + 1:04d}")
            self.save_model("last")
            if loss <= self.best_loss:
                self.best_loss = loss
                self.save_model(f"best_{epoch}")


class DatasetPartIndexV53:
    def __init__(self, part_files, part_lengths, positive_hints):
        self.part_files = [str(path) for path in part_files]
        self.part_lengths = list(part_lengths)
        self.positive_hints = positive_hints
        self.cumulative_sizes = []

        total = 0
        for part_length in self.part_lengths:
            total += part_length
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def locate(self, global_idx: int):
        if global_idx < 0 or global_idx >= len(self):
            raise IndexError(f"Dataset index out of range: {global_idx}")

        part_idx = bisect_right(self.cumulative_sizes, global_idx)
        part_start = 0 if part_idx == 0 else self.cumulative_sizes[part_idx - 1]
        return self.part_files[part_idx], global_idx - part_start

    def is_positive(self, global_idx: int) -> bool:
        return bool(self.positive_hints[global_idx])


class PPDatasetV53(Dataset):
    def __init__(self, dataset_index: DatasetPartIndexV53, sample_indices, is_test=False):
        super().__init__()
        self.dataset_index = dataset_index
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.is_test = is_test
        self._cached_part_path = None
        self._cached_part_items = None

    def __len__(self):
        return len(self.sample_indices)

    def load_image(self, path):
        with Image.open(path) as image:
            return T.functional.to_tensor(image.convert("RGB"))

    def load_part_items(self, part_path):
        part_path = str(part_path)
        if self._cached_part_path != part_path:
            self._cached_part_items = None
            self._cached_part_path = None
            gc.collect()
            self._cached_part_items = torch.load(part_path, map_location="cpu")
            self._cached_part_path = part_path
        return self._cached_part_items

    def flip_parsing_lr(self, parsing):
        parsing = T.functional.hflip(parsing)
        left_mask = parsing == 7
        right_mask = parsing == 8
        parsing[left_mask] = 8
        parsing[right_mask] = 7
        return parsing

    def transform(self, sample):
        if self.is_test or random.random() <= 0.5:
            return sample

        keys_to_flip = ["source", "target", "target_mask", "HT_E", "source_hair_mask", "target_hair_mask",
                        "source_earring_mask", "target_earring_mask", "query_mask", "ear_roi",
                        "source_hair_block_mask", *CLEANUP_MASK_KEYS, *DERIVED_MASK_KEYS]
        for key in keys_to_flip:
            sample[key] = T.functional.hflip(sample[key])
        sample["source_parsing"] = self.flip_parsing_lr(sample["source_parsing"])
        sample["target_parsing"] = self.flip_parsing_lr(sample["target_parsing"])

        left_roi = T.functional.hflip(sample["left_ear_roi"])
        right_roi = T.functional.hflip(sample["right_ear_roi"])
        sample["left_ear_roi"] = right_roi
        sample["right_ear_roi"] = left_roi
        sample["presence_target"] = sample["presence_target"][[1, 0, 2]]
        return sample

    def __getitem__(self, idx):
        global_idx = int(self.sample_indices[idx])
        part_path, item_idx = self.dataset_index.locate(global_idx)
        item = self.load_part_items(part_path)[item_idx]
        fallback_mask = item["target_mask"]
        sample = {
            "source": self.load_image(item["source_path"]),
            "target": item["target"].clone(),
            "target_mask": item["target_mask"].clone(),
            "HT_E": item["HT_E"].clone(),
            "source_parsing": item["source_parsing"].clone(),
            "target_parsing": item["target_parsing"].clone(),
            "source_hair_mask": item["source_hair_mask"].clone(),
            "target_hair_mask": item["target_hair_mask"].clone(),
            "source_hair_block_mask": item.get("source_hair_block_mask", torch.zeros_like(fallback_mask)).clone(),
            "source_earring_mask": item["source_earring_mask"].clone(),
            "target_earring_mask": item["target_earring_mask"].clone(),
            "query_mask": item["query_mask"].clone(),
            "ear_roi": item["ear_roi"].clone(),
            "left_ear_roi": item["left_ear_roi"].clone(),
            "right_ear_roi": item["right_ear_roi"].clone(),
            "presence_target": item["presence_target"].clone(),
        }
        for key in CLEANUP_MASK_KEYS:
            sample[key] = item.get(key, torch.zeros_like(fallback_mask)).clone()
        for key in DERIVED_MASK_KEYS:
            sample[key] = item.get(key, torch.zeros_like(fallback_mask)).clone()
        return self.transform(sample)


def build_dataset_index(dataset_dir: Path, query_area_threshold: float):
    files = sorted(dataset_dir.glob("pp_part_*.dataset"))
    if not files:
        raise FileNotFoundError(f"No pp_part_*.dataset files were found under {dataset_dir}")

    part_lengths = []
    positive_hints = bytearray()

    for file in tqdm(files, desc="Index dataset parts"):
        part_items = torch.load(file, map_location="cpu")
        part_lengths.append(len(part_items))
        for item in part_items:
            positive_hints.append(1 if is_positive_hint(item, query_area_threshold) else 0)
        del part_items
        gc.collect()

    return DatasetPartIndexV53(files, part_lengths, positive_hints)


def is_positive_hint(item, query_area_threshold: float) -> bool:
    parser_positive = bool(item["source_earring_mask"].sum().item() > 0 or item["target_earring_mask"].sum().item() > 0)
    derived_positive = bool(item.get("earring_confident_mask", item["source_earring_mask"]).sum().item() > 0)
    query_positive = bool(item["query_mask"].sum().item() > query_area_threshold)
    return parser_positive or derived_positive or query_positive


def split_dataset_indices(dataset_size: int, test_size: int, seed: int = 42):
    if dataset_size <= 0:
        raise ValueError("Dataset is empty.")

    indices = np.arange(dataset_size, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    return indices[test_size:], indices[:test_size]


def resolve_dataloader_context(args):
    if args.num_workers <= 0:
        return None

    start_method = str(args.dataloader_start_method).lower()
    if start_method in {"", "none", "default"}:
        return None
    if start_method == "auto":
        return None if os.name == "nt" else "spawn"
    return start_method


def build_dataloader_kwargs(args, shuffle=False, sampler=None, drop_last=False):
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.dataloader_pin_memory),
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "drop_last": drop_last,
    }

    loader_context = resolve_dataloader_context(args)
    if loader_context is not None:
        loader_kwargs["multiprocessing_context"] = loader_context

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.dataloader_persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(args.dataloader_prefetch_factor))

    return loader_kwargs


def effective_batch_size(args):
    return int(args.batch_size) * max(1, int(args.grad_accum_steps))


def main(args):
    print(
        f"Using dataset_profile={args.dataset_profile}, dataset={args.dataset}, "
        f"checkpoint_dir={args.checkpoint_dir}"
    )
    print(
        f"Using batch_size={args.batch_size}, grad_accum_steps={max(1, int(args.grad_accum_steps))}, "
        f"effective_batch_size={effective_batch_size(args)}."
    )
    seed_everything()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset_index = build_dataset_index(args.dataset, args.positive_query_area_threshold)
    test_size = min(args.test_size, max(1, len(dataset_index) // 10))
    train_indices, test_indices = split_dataset_indices(len(dataset_index), test_size, seed=42)

    train_dataset = PPDatasetV53(dataset_index, train_indices)
    positive_train_indices = [idx for idx in train_indices if dataset_index.is_positive(int(idx))]
    train_dataset_positive = PPDatasetV53(dataset_index, positive_train_indices) if positive_train_indices else None
    test_dataset = PPDatasetV53(dataset_index, test_indices, is_test=True)

    sampler = None
    weights = [args.positive_sample_weight if dataset_index.is_positive(int(idx)) else 1.0 for idx in train_indices]
    if any(weight > 1.0 for weight in weights):
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    loader_context = resolve_dataloader_context(args)
    if loader_context is not None:
        print(
            f"Using DataLoader multiprocessing_context='{loader_context}' "
            f"(num_workers={args.num_workers}, prefetch_factor={max(1, int(args.dataloader_prefetch_factor))}, "
            f"pin_memory={bool(args.dataloader_pin_memory)})."
        )
    elif args.num_workers > 0:
        print(
            f"Using default DataLoader worker start method "
            f"(num_workers={args.num_workers}, pin_memory={bool(args.dataloader_pin_memory)})."
        )
    else:
        print(f"Using single-process DataLoader (num_workers=0, pin_memory={bool(args.dataloader_pin_memory)}).")

    train_dataloader = DataLoader(
        train_dataset,
        **build_dataloader_kwargs(args, shuffle=sampler is None, sampler=sampler, drop_last=True),
    )
    train_dataloader_positive = None
    if train_dataset_positive is not None and len(train_dataset_positive) > 0:
        train_dataloader_positive = DataLoader(
            train_dataset_positive,
            **build_dataloader_kwargs(args, shuffle=True, drop_last=False),
        )
    test_dataloader = DataLoader(
        test_dataset,
        **build_dataloader_kwargs(args, shuffle=False, drop_last=False),
    )

    logger = WandbLogger(name=args.name_run, project="HairFast-PostProcess-V53") if args.use_wandb else NullLoggerV53(
        args.checkpoint_dir
    )
    logger.start_logging()

    model = PostProcessModelV53(args)
    model.load_base_checkpoint(args.base_checkpoint)
    configure_training_stage(model, args)
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=1e-4, weight_decay=0)

    trainer = TrainerV53(model, args, optimizer, train_dataloader, train_dataloader_positive, test_dataloader, logger)
    if args.resume_checkpoint is not None:
        trainer.load_model(args.resume_checkpoint)
    trainer.train_loop()


if __name__ == "__main__":
    parser = build_parser(RESOLVED_USER_CONFIG)
    main(parser.parse_args())
