import argparse
import faulthandler
import gc
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import shutil
import sys
from argparse import Namespace
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from losses.pp_losses_v5 import EarAwareLossBuilder
from models.Net import Net
from models.postprocess_v5 import PostProcessModelV5
from models.stylegan2 import dnnlib
from utils.bicubic import BicubicDownSample
from utils.train import WandbLogger, _LegacyUnpickler, get_fid_calc, image_grid, seed_everything, toggle_grad

faulthandler.enable(all_threads=True)

CLEANUP_MASK_KEYS = ("M_remove", "M_remove_halo", "M_remove_face", "M_remove_tail", "M_remove_neck")
# Must match ``scripts/pp_gen_v5.py``. Schema 22 also stores the completed
# high-resolution hair transfer as an immutable final-output authority.
PP_DATASET_SCHEMA_VERSION = 22
PP_EXTRA_MASK_KEYS = (
    "cleanup_inner_edge",
    "revealed_skin_mask",
    "revealed_skin_seam_mask",
    "revealed_skin_blend_mask",
    "source_visible_skin_reference_mask",
    "source_skin_valid_mask",
    "earring_confident_mask",
    "earring_instance_mask",
    "hoop_instance_mask",
    "hoop_hole_mask",
    "earring_highlight_mask",
    "earring_candidate_mask",
)
DATASET_AUX_MASK_FLAGS = {
    "earring_confident_mask": "has_earring_confident_mask",
    "earring_highlight_mask": "has_earring_highlight_mask",
    "earring_candidate_mask": "has_earring_candidate_mask",
    "earring_search_mask": "has_earring_search_mask",
}
ONLINE_EARRING_AUX_KEYS = {
    "earring_confident_mask",
    "earring_highlight_mask",
    "earring_candidate_mask",
    "earring_search_mask",
}
# Only the six pipeline images the user wants to inspect.  All mask/debug
# columns were removed from the validation preview.
VAL_COLUMNS = (
    "source",                 # 1. source face
    "shape_reference",        # 2. reference hairstyle
    "color_reference",        # 3. reference hair color
    "target_deshadow",        # 4. shape transfer + SATD de-shadow (pre-color)
    "target",                 # 5. de-shadow + color blend (the PP input)
    "gen_f",                  # 6. PP earring/face-detail recovery (final)
)

# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small_accessory_ffhq"  # "small_accessory_ffhq" or "full_ffhq"

USER_DATASET_DIR_SMALL = Path("images/pp_dataset_v5_dual_ear_short_long_hair_locked_v5")
USER_OUTPUT_DIR_SMALL = Path("output/pp_v5_checkpoints_ear_short_long_hair_locked_v5")
USER_RUN_NAME_SMALL = "ear_refine_v5_dual_small_hair_locked_v5"

USER_DATASET_DIR_FULL = Path("images/pp_dataset_v5_dual_full_instance_v21_structural_correction")
USER_OUTPUT_DIR_FULL = Path("output/pp_v5_checkpoints_full_instance_v21_structural_correction")
USER_RUN_NAME_FULL = "ear_refine_v5_dual_full_instance_v21_structural_correction"

USER_FID_DATASET = "fid_images"
USER_USE_FID = False
USER_USE_WANDB = False
USER_RESUME_CHECKPOINT = None
USER_BASE_CHECKPOINT = "pretrained_models/PostProcess/pp_model.pth"

USER_BATCH_SIZE = 8
USER_NUM_WORKERS = 0
USER_EPOCHS = 120
USER_VAL_SIZE = 512
USER_VAL_PREVIEW_COUNT = 50
# Filling a small validation split with training samples runs extra full
# inference passes, but does not contribute to validation loss or training.
USER_VAL_SUPPLEMENT_TRAIN_PREVIEWS = True
USER_GRAD_ACCUM_STEPS = 2

USER_TRAINING_STAGE = "joint_highres"  # "ear_only", "joint_highres", or "full"
USER_PRETRAIN = False
USER_FINETUNE = False
USER_USE_MOD = True
USER_USE_FULL = True

# V5 structural compositor contract. These are intentionally independent from
# legacy write/search-mask knobs, which are ignored by the final source-native
# compositor.
USER_ENABLE_V5_STRUCTURAL_COMPOSITOR = True
USER_FACE_CONTINUITY_WORK_SIZE = 512
USER_FACE_DETAIL_SOFT_EDGE = 6
USER_FACE_HAIR_SOFT_EDGE = 4
USER_EARRING_USE_NATIVE_COORD_CONTRACT = True
USER_EARRING_COMPONENT_MAX_DEPTH = 4
USER_EARRING_COMPONENT_MAX_CUMULATIVE_COST = 1.85

USER_ITER_BEFORE_ADV = 10_000
USER_D_REG_EVERY = 16
USER_INPAINT = 0.0
USER_USE_ADV = False
USER_ADV_COEF = 0.05

USER_EAR_PARSE_SIZE = 512
USER_EAR_FEATURE_CHANNELS = 128
USER_EAR_LOW_ALPHA = 0.1
USER_EAR_DILATE = 21
USER_HAIR_CHANGE_DILATE = 25
USER_EARRING_EXPAND = 15
USER_EAR_DOWNWARD_SHIFT = 10
USER_TARGET_HAIR_DILATE = 11
USER_SOURCE_HAIR_BLOCK_DILATE = 8
USER_SOURCE_HAIR_BLOCK_STRENGTH = 0.95
USER_TARGET_VISIBILITY_EXPAND = 5
USER_MAX_TARGET_HAIR_OVERLAP = 0.30
USER_TARGET_EAR_COVER_OVERLAP = 0.55
USER_MIN_TARGET_VISIBLE_OVERLAP = 0.10
USER_MIN_TARGET_EAR_AREA = 8.0
USER_EARRING_CHANNEL_DOWN = 32
USER_SOURCE_EARRING_OPEN_OVERLAP = 0.35

USER_EAR_BLUR_KERNEL = 11
USER_EAR_BLUR_SIGMA = 3.0
USER_EAR_MASK_HIDDEN = 32
USER_EAR_MASK_INIT_BIAS = -4.0
USER_EARRING_QUERY_DILATE = 3
USER_EARRING_QUERY_BOOST = 1.0
USER_ENABLE_EARRING_QUERY_RECALL = True
USER_EARRING_QUERY_RECALL_DILATE = 7
USER_EARRING_QUERY_DOWNWARD_SHIFT = 10
USER_EARRING_QUERY_LOWER_LOBE_WEIGHT = 0.20
USER_EARRING_QUERY_CANDIDATE_BOOST = 0.90
USER_EARRING_QUERY_BLOCK_PROTECT = 0.95
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
# Training data uses the same fixed source-face coordinate system as inference.
# A non-zero parser-centroid shift teaches duplicate/offset accessories.
USER_EARRING_ALIGN_MAX_SHIFT = 0
USER_EAR_FINE_SUPPORT_DILATE = 3
USER_EARRING_FINE_MASK_FLOOR = 0.18
USER_EARRING_FINE_MASK_DILATE = 5
# When the native source-instance extractor rejects an uncertain edge, let the
# trained PP branch fill only the already verified low-resolution object mask.
# Direct source RGB still wins wherever a native instance is available.
USER_EARRING_LEARNED_FALLBACK_ALPHA = 0.0
USER_EARRING_LOBE_SUPPORT_SOURCE_WEIGHT = 0.0
USER_EARRING_LOBE_SUPPORT_FINE_WEIGHT = 0.0
USER_TARGET_HAIR_EAR_PROTECT_DILATE = 3
USER_TARGET_HAIR_EAR_SCOPE_DILATE = 3
USER_TARGET_HAIR_EAR_PROTECT_STRENGTH = 1.0
USER_ALLOW_EARRING_THROUGH_TARGET_HAIR_EAR_PROTECT = True
USER_SOURCE_HAIR_FACE_SUPPRESS_DILATE = 5
USER_SOURCE_HAIR_FACE_SUPPRESS_STRENGTH = 0.65
USER_SOURCE_HAIR_FACE_SUPPRESS_MAX_Y = 0.45
USER_SOURCE_HAIR_FACE_SUPPRESS_EAR_EXCLUDE_DILATE = 9
# Dilate the source hair mask before excluding it from the base reconstruction
# loss.  This prevents the "three-tone face" artifact when source bangs (forehead)
# or figure-8 bangs (cheeks) cover part of the face: the base loss must exclude
# the entire covered region plus a safety margin so parser under-segmentation
# (fine hair strands, semi-transparent hair edges) cannot leave a "grey zone" of
# pixels that have near-zero supervision.  Raised from 9 to 15 to cover figure-8
# bang edges and stray wisps that the parser misses; slightly over-excludes the
# true hair boundary (a few real-face pixels get zero base supervision) but
# eliminates the half-transparent brush stroke / grey halo / colour mismatch at
# the hair/face seam.
USER_BASE_SOURCE_HAIR_EXCLUDE_DILATE = 15
USER_BASE_SOURCE_HAIR_EXCLUDE_STRENGTH = 1.0
USER_BASE_SOURCE_HAIR_EXCLUDE_MAX_Y = 0.55
USER_BASE_SOURCE_HAIR_EXCLUDE_EAR_DILATE = 9
# Final V5 composition keeps target hair and target ear geometry authoritative.
# A source earring can open this preserve gate only through the connected,
# object-supported write mask; its hoop hole remains target-owned.
# NOTE: retrain after this change — an old paste-trained checkpoint has
# unsupervised raw hair.
USER_ENABLE_OUTPUT_TARGET_PRESERVE = True
USER_OUTPUT_TARGET_HAIR_PRESERVE_DILATE = 5
USER_OUTPUT_FACE_HAIR_SEAM_PRESERVE_DILATE = 7
USER_OUTPUT_REVEALED_SKIN_PRESERVE_DILATE = 3
USER_OUTPUT_EARRING_KEEP_DILATE = 0
USER_OUTPUT_PRESERVE_BLUR = 1
# Preserve the target hairline as a single authority.  Blending it with the
# PP output makes a visible colour ring when their illumination differs.
USER_OUTPUT_HAIRLINE_FEATHER = 0
# Source content gate: prevent source background/neck from bleeding into the
# target when the source has short hair or exposed ears (the v58 ear ROI is
# geometric and can cover source background).  Restricts source-sampling to real
# source content (skin, ear, earring label-9), excluding hair/background.  The
# earring label-9 is explicitly protected from hair subtraction so fine hair
# strands covering the earring edge cannot remove confirmed earring pixels.
USER_ENABLE_SOURCE_CONTENT_GATE = True
USER_SOURCE_CONTENT_GATE_DILATE = 3
# Problem 4A: inference-time revealed-forehead skin harmonization.
# v58 alignment: v58 has NO revealed-skin harmonize and produces a natural
# forehead purely from the generator + general reconstruction losses.  The
# harmonize below is a low-pass + diffusion fill (== skin smoothing / 磨皮) that
# runs at INFERENCE, so it degrades even the current checkpoint and is the source
# of the airbrushed look and the seam against the real skin.  Disabled so the
# revealed forehead keeps the generator's source-like texture like v58.
USER_ENABLE_REVEALED_SKIN_HARMONIZE = False
USER_REVEALED_SKIN_HARMONIZE_STRENGTH = 0.9
USER_REVEALED_SKIN_TONE_KERNEL = 15
USER_REVEALED_SKIN_TONE_SIGMA = 7.0
USER_REVEALED_SKIN_DIFFUSE_ITERS = 24
USER_REVEALED_SKIN_TONE_LIMIT = 0.28
USER_REVEALED_SKIN_DETAIL_GAIN = 1.0
USER_REVEALED_SKIN_SEAM_BAND = 7
USER_REVEALED_SKIN_MIN_REFERENCE_AREA = 96.0
USER_ENABLE_OUTPUT_SOURCE_EARRING_COMPOSITE = False
USER_OUTPUT_SOURCE_EARRING_ALPHA = 1.0
USER_OUTPUT_SOURCE_EARRING_MASK_DILATE = 0
USER_OUTPUT_SOURCE_EARRING_MASK_BLUR = 1
USER_OUTPUT_SOURCE_EARRING_ALIGN_MAX_SHIFT = 48
USER_OUTPUT_SOURCE_EARRING_MIN_AREA = 1.0
USER_ENABLE_OUTPUT_EARRING_REFERENCE_COMPOSITE = False
USER_OUTPUT_EARRING_REFERENCE_ALPHA = 1.0
USER_OUTPUT_EARRING_REFERENCE_BLUR = 1
USER_OUTPUT_EARRING_REFERENCE_OBJECT_DILATE = 1
USER_OUTPUT_EARRING_REFERENCE_DIFF_THRESHOLD = 0.018
USER_OUTPUT_EARRING_REFERENCE_HIGH_THRESHOLD = 0.004
USER_OUTPUT_EARRING_REFERENCE_COLOR_THRESHOLD = 0.020
USER_OUTPUT_EARRING_REFERENCE_STRONG_DIFF_THRESHOLD = 0.045
USER_OUTPUT_EARRING_REFERENCE_MIN_AREA = 0.5
USER_ENABLE_REVEALED_DARK_LINE_REPAIR = False
USER_REVEALED_DARK_LINE_THRESHOLD = 0.025
USER_REVEALED_DARK_LINE_SOURCE_HAIR_DILATE = 9
USER_REVEALED_DARK_LINE_SOURCE_EDGE_DILATE = 5
USER_REVEALED_DARK_LINE_EDGE_THRESHOLD = 0.008
USER_REVEALED_DARK_LINE_LOCAL_KERNEL = 11
USER_REVEALED_DARK_LINE_DILATE = 5
USER_REVEALED_DARK_LINE_BLUR = 3
USER_REVEALED_DARK_LINE_FILL_KERNEL = 25
USER_ENABLE_FACE_DARK_LINE_REPAIR = False
USER_FACE_DARK_LINE_THRESHOLD = 0.020
USER_FACE_DARK_LINE_LOCAL_KERNEL = 13
USER_FACE_DARK_LINE_VERTICAL_KERNEL = 9
USER_FACE_DARK_LINE_DILATE = 3
USER_FACE_DARK_LINE_BLUR = 3
USER_FACE_DARK_LINE_FILL_KERNEL = 21
USER_FACE_DARK_LINE_REPAIR_STRENGTH = 0.85
USER_FACE_DARK_LINE_DETAIL_EXCLUDE_DILATE = 5
USER_FACE_DARK_LINE_EAR_EXCLUDE_DILATE = 9
USER_EARRING_REFERENCE_OBJECT_DILATE = 5
USER_ENABLE_ONLINE_SOURCE_EARRING_OBJECT_MASK = True
USER_ONLINE_SOURCE_EARRING_OBJECT_MIN_AREA = 2
USER_USE_REFINED_EARRING_OBJECT_MASK = False
USER_EARRING_REQUIRE_SOURCE_PARSER_SEED = False
USER_EARRING_MIN_SOURCE_PARSER_AREA = 0.5
USER_EARRING_MIN_REFINED_OBJECT_AREA = 2.0
USER_ALLOW_VISUAL_EARRING_SEED = False
USER_VISUAL_EARRING_SEED_MIN_AREA = 6.0
USER_VISUAL_EARRING_SEED_MAX_DENSITY = 0.12
# Only contour-verified hollow hoops use the precise dataset alpha/hole.  The
# established ordinary-earring PP path remains responsible for studs, solid
# pendants and other non-hollow accessories.
USER_PREFER_DATASET_EARRING_REFERENCE = True
USER_USE_DATASET_EARRING_AUX = True
USER_EXPAND_DATASET_EARRING_FROM_REFERENCE_DELTA = False
USER_EARRING_VISIBLE_ROI_EXCLUDE_TARGET_HAIR = False
USER_REFRESH_EARRING_REFERENCE = False

USER_LAMBDA_EAR_MASK = 6.0
USER_LAMBDA_EAR_HIGH = 8.0
USER_LAMBDA_EAR_PRESENCE = 3.0
USER_LAMBDA_EAR_LIGHTING = 0.35
USER_LAMBDA_EAR_HAIR_LEAK = 1.0
USER_LAMBDA_EAR_HAIR_ANCHOR = 5.0
USER_LAMBDA_EAR_COLOR = 1.5
USER_LAMBDA_EAR_HIGHLIGHT = 3.0
USER_LAMBDA_CLEANUP_ANCHOR = 0.0
USER_LAMBDA_CLEANUP_LOW_ANCHOR = 0.0
USER_LAMBDA_CLEANUP_SOURCE_REJECT = 4.0
USER_LAMBDA_CLEANUP_NON_DARK = 0.0
USER_LAMBDA_CLEANUP_HIGH = 0.0
USER_LAMBDA_CLEANUP_TEXTURE_STAT = 0.0
USER_LAMBDA_DETAIL_HIGH = 0.0
USER_LAMBDA_DETAIL_LOW_ANCHOR = 1.5
# V5 keeps the entire PP face as one image authority, then supervises real
# source-visible skin as a high-frequency/color anchor through loss masks.
USER_LAMBDA_SOURCE_VALID_FACE_HIGH = 0.35
USER_LAMBDA_SOURCE_VALID_FACE_COLOR = 0.15
USER_LAMBDA_FACE_SOURCE_DETAIL_HR = 1.0
USER_LAMBDA_FACE_LOWFREQ_ANCHOR_HR = 0.35
USER_LAMBDA_FACE_LOWFREQ_CONTINUITY_HR = 1.0
USER_LAMBDA_REVEALED_BOUNDARY_SEAM_HR = 1.5
USER_LAMBDA_REVEALED_SOURCE_TEXTURE_STAT_HR = 0.60
USER_LAMBDA_FACE_HAIR_BOUNDARY_SEAM_HR = 1.0
# V5 continuity losses act only across the revealed-skin boundary.  They do
# not blur or overwrite the face during compositing, and they leave the normal
# joint_highres training cadence unchanged.
USER_LAMBDA_FACE_LOWFREQ_CONTINUITY = 0.30
USER_LAMBDA_REVEALED_BOUNDARY_SEAM = 0.35
USER_LAMBDA_REVEALED_TEXTURE_STAT = 0.12
# Revealed skin is constrained only against neighbouring target/PP skin; the
# loss implementation no longer copies source bang-hidden RGB/texture.
# PP owns ordinary face pixels.  Do not pull them back toward smooth SATD
# target pixels or impose a low-frequency forehead tone anchor.
USER_LAMBDA_REVEALED_SKIN_TEXTURE = 0.0
USER_LAMBDA_REVEALED_SKIN_TONE = 0.0
USER_LAMBDA_NORMAL_FACE_PRESERVE = 0.0
USER_LAMBDA_FACE_SOURCE_DARK_REJECT = 2.0
USER_FACE_SOURCE_DARK_REJECT_MARGIN = 0.006
USER_FACE_SOURCE_DARK_REJECT_SOURCE_THRESHOLD = 0.018
USER_FACE_SOURCE_DARK_REJECT_KERNEL = 11
USER_FACE_SOURCE_DARK_SOURCE_HAIR_DILATE = 7
USER_LAMBDA_EAR_QUERY_EXPAND = 0.02
USER_LAMBDA_EAR_MASK_AREA = 0.0
USER_LAMBDA_EAR_EDGE = 3.0
USER_LAMBDA_EAR_BRIGHTNESS_REG = 0.05
USER_LAMBDA_TARGET_HAIR_EAR_ANCHOR = 6.0
USER_LAMBDA_TARGET_HAIR_EAR_SOURCE_REJECT = 3.0
USER_LAMBDA_TARGET_PRESERVE = 4.0
USER_LAMBDA_TARGET_HAIR_PRESERVE = 8.0
USER_TARGET_PRESERVE_EXCLUDE_DILATE = 9
USER_LAMBDA_BASE_CLEANUP_EXCLUDE = 1.0
USER_EARRING_SUPERVISION_DILATE = 5
USER_LAMBDA_TARGET_EARRING_SUPPRESS_LOW = 1.5
USER_LAMBDA_TARGET_EARRING_SUPPRESS_HIGH = 0.8
USER_LAMBDA_TARGET_EAR_GEOMETRY = 5.0
USER_LAMBDA_NO_EARRING_NOOP = 6.0
USER_LAMBDA_HOOP_HOLE_PRESERVE = 8.0
USER_LAMBDA_EARRING_OBJECT_RESTORE = 3.0
USER_LAMBDA_EARRING_FOREGROUND_RESTORE = 6.0
USER_USE_DATASET_QUERY_MASK = False
USER_USE_DATASET_SOURCE_EARRING_MASK = True
USER_POSITIVE_ONLY_WARMUP_EPOCHS = 20
USER_POSITIVE_SAMPLE_WEIGHT = 8.0
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
    "val_supplement_train_previews": USER_VAL_SUPPLEMENT_TRAIN_PREVIEWS,
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
    "enable_v5_structural_compositor": USER_ENABLE_V5_STRUCTURAL_COMPOSITOR,
    "face_continuity_work_size": USER_FACE_CONTINUITY_WORK_SIZE,
    "face_detail_soft_edge": USER_FACE_DETAIL_SOFT_EDGE,
    "face_hair_soft_edge": USER_FACE_HAIR_SOFT_EDGE,
    "earring_use_native_coord_contract": USER_EARRING_USE_NATIVE_COORD_CONTRACT,
    "earring_component_max_depth": USER_EARRING_COMPONENT_MAX_DEPTH,
    "earring_component_max_cumulative_cost": USER_EARRING_COMPONENT_MAX_CUMULATIVE_COST,
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
    "target_visibility_expand": USER_TARGET_VISIBILITY_EXPAND,
    "max_target_hair_overlap": USER_MAX_TARGET_HAIR_OVERLAP,
    "target_ear_cover_overlap": USER_TARGET_EAR_COVER_OVERLAP,
    "min_target_visible_overlap": USER_MIN_TARGET_VISIBLE_OVERLAP,
    "min_target_ear_area": USER_MIN_TARGET_EAR_AREA,
    "earring_channel_down": USER_EARRING_CHANNEL_DOWN,
    "source_earring_open_overlap": USER_SOURCE_EARRING_OPEN_OVERLAP,
    "ear_blur_kernel": USER_EAR_BLUR_KERNEL,
    "ear_blur_sigma": USER_EAR_BLUR_SIGMA,
    "ear_mask_hidden": USER_EAR_MASK_HIDDEN,
    "ear_mask_init_bias": USER_EAR_MASK_INIT_BIAS,
    "earring_query_dilate": USER_EARRING_QUERY_DILATE,
    "earring_query_boost": USER_EARRING_QUERY_BOOST,
    "enable_earring_query_recall": USER_ENABLE_EARRING_QUERY_RECALL,
    "earring_query_recall_dilate": USER_EARRING_QUERY_RECALL_DILATE,
    "earring_query_downward_shift": USER_EARRING_QUERY_DOWNWARD_SHIFT,
    "earring_query_lower_lobe_weight": USER_EARRING_QUERY_LOWER_LOBE_WEIGHT,
    "earring_query_candidate_boost": USER_EARRING_QUERY_CANDIDATE_BOOST,
    "earring_query_block_protect": USER_EARRING_QUERY_BLOCK_PROTECT,
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
    "ear_fine_support_dilate": USER_EAR_FINE_SUPPORT_DILATE,
    "earring_fine_mask_floor": USER_EARRING_FINE_MASK_FLOOR,
    "earring_fine_mask_dilate": USER_EARRING_FINE_MASK_DILATE,
    "earring_learned_fallback_alpha": USER_EARRING_LEARNED_FALLBACK_ALPHA,
    "earring_lobe_support_source_weight": USER_EARRING_LOBE_SUPPORT_SOURCE_WEIGHT,
    "earring_lobe_support_fine_weight": USER_EARRING_LOBE_SUPPORT_FINE_WEIGHT,
    "target_hair_ear_protect_dilate": USER_TARGET_HAIR_EAR_PROTECT_DILATE,
    "target_hair_ear_scope_dilate": USER_TARGET_HAIR_EAR_SCOPE_DILATE,
    "target_hair_ear_protect_strength": USER_TARGET_HAIR_EAR_PROTECT_STRENGTH,
    "allow_earring_through_target_hair_ear_protect": USER_ALLOW_EARRING_THROUGH_TARGET_HAIR_EAR_PROTECT,
    "source_hair_face_suppress_dilate": USER_SOURCE_HAIR_FACE_SUPPRESS_DILATE,
    "source_hair_face_suppress_strength": USER_SOURCE_HAIR_FACE_SUPPRESS_STRENGTH,
    "source_hair_face_suppress_max_y": USER_SOURCE_HAIR_FACE_SUPPRESS_MAX_Y,
    "source_hair_face_suppress_ear_exclude_dilate": USER_SOURCE_HAIR_FACE_SUPPRESS_EAR_EXCLUDE_DILATE,
    "base_source_hair_exclude_dilate": USER_BASE_SOURCE_HAIR_EXCLUDE_DILATE,
    "base_source_hair_exclude_strength": USER_BASE_SOURCE_HAIR_EXCLUDE_STRENGTH,
    "base_source_hair_exclude_max_y": USER_BASE_SOURCE_HAIR_EXCLUDE_MAX_Y,
    "base_source_hair_exclude_ear_dilate": USER_BASE_SOURCE_HAIR_EXCLUDE_EAR_DILATE,
    "enable_source_content_gate": USER_ENABLE_SOURCE_CONTENT_GATE,
    "source_content_gate_dilate": USER_SOURCE_CONTENT_GATE_DILATE,
    "enable_output_target_preserve": USER_ENABLE_OUTPUT_TARGET_PRESERVE,
    "output_target_hair_preserve_dilate": USER_OUTPUT_TARGET_HAIR_PRESERVE_DILATE,
    "output_face_hair_seam_preserve_dilate": USER_OUTPUT_FACE_HAIR_SEAM_PRESERVE_DILATE,
    "output_revealed_skin_preserve_dilate": USER_OUTPUT_REVEALED_SKIN_PRESERVE_DILATE,
    "output_earring_keep_dilate": USER_OUTPUT_EARRING_KEEP_DILATE,
    "output_preserve_blur": USER_OUTPUT_PRESERVE_BLUR,
    "output_hairline_feather": USER_OUTPUT_HAIRLINE_FEATHER,
    "enable_revealed_skin_harmonize": USER_ENABLE_REVEALED_SKIN_HARMONIZE,
    "revealed_skin_harmonize_strength": USER_REVEALED_SKIN_HARMONIZE_STRENGTH,
    "revealed_skin_tone_kernel": USER_REVEALED_SKIN_TONE_KERNEL,
    "revealed_skin_tone_sigma": USER_REVEALED_SKIN_TONE_SIGMA,
    "revealed_skin_diffuse_iters": USER_REVEALED_SKIN_DIFFUSE_ITERS,
    "revealed_skin_tone_limit": USER_REVEALED_SKIN_TONE_LIMIT,
    "revealed_skin_detail_gain": USER_REVEALED_SKIN_DETAIL_GAIN,
    "revealed_skin_seam_band": USER_REVEALED_SKIN_SEAM_BAND,
    "revealed_skin_min_reference_area": USER_REVEALED_SKIN_MIN_REFERENCE_AREA,
    "enable_output_source_earring_composite": USER_ENABLE_OUTPUT_SOURCE_EARRING_COMPOSITE,
    "output_source_earring_alpha": USER_OUTPUT_SOURCE_EARRING_ALPHA,
    "output_source_earring_mask_dilate": USER_OUTPUT_SOURCE_EARRING_MASK_DILATE,
    "output_source_earring_mask_blur": USER_OUTPUT_SOURCE_EARRING_MASK_BLUR,
    "output_source_earring_align_max_shift": USER_OUTPUT_SOURCE_EARRING_ALIGN_MAX_SHIFT,
    "output_source_earring_min_area": USER_OUTPUT_SOURCE_EARRING_MIN_AREA,
    "enable_output_earring_reference_composite": USER_ENABLE_OUTPUT_EARRING_REFERENCE_COMPOSITE,
    "output_earring_reference_alpha": USER_OUTPUT_EARRING_REFERENCE_ALPHA,
    "output_earring_reference_blur": USER_OUTPUT_EARRING_REFERENCE_BLUR,
    "output_earring_reference_object_dilate": USER_OUTPUT_EARRING_REFERENCE_OBJECT_DILATE,
    "output_earring_reference_diff_threshold": USER_OUTPUT_EARRING_REFERENCE_DIFF_THRESHOLD,
    "output_earring_reference_high_threshold": USER_OUTPUT_EARRING_REFERENCE_HIGH_THRESHOLD,
    "output_earring_reference_color_threshold": USER_OUTPUT_EARRING_REFERENCE_COLOR_THRESHOLD,
    "output_earring_reference_strong_diff_threshold": USER_OUTPUT_EARRING_REFERENCE_STRONG_DIFF_THRESHOLD,
    "output_earring_reference_min_area": USER_OUTPUT_EARRING_REFERENCE_MIN_AREA,
    "enable_revealed_dark_line_repair": USER_ENABLE_REVEALED_DARK_LINE_REPAIR,
    "revealed_dark_line_threshold": USER_REVEALED_DARK_LINE_THRESHOLD,
    "revealed_dark_line_source_hair_dilate": USER_REVEALED_DARK_LINE_SOURCE_HAIR_DILATE,
    "revealed_dark_line_source_edge_dilate": USER_REVEALED_DARK_LINE_SOURCE_EDGE_DILATE,
    "revealed_dark_line_edge_threshold": USER_REVEALED_DARK_LINE_EDGE_THRESHOLD,
    "revealed_dark_line_local_kernel": USER_REVEALED_DARK_LINE_LOCAL_KERNEL,
    "revealed_dark_line_dilate": USER_REVEALED_DARK_LINE_DILATE,
    "revealed_dark_line_blur": USER_REVEALED_DARK_LINE_BLUR,
    "revealed_dark_line_fill_kernel": USER_REVEALED_DARK_LINE_FILL_KERNEL,
    "enable_face_dark_line_repair": USER_ENABLE_FACE_DARK_LINE_REPAIR,
    "face_dark_line_threshold": USER_FACE_DARK_LINE_THRESHOLD,
    "face_dark_line_local_kernel": USER_FACE_DARK_LINE_LOCAL_KERNEL,
    "face_dark_line_vertical_kernel": USER_FACE_DARK_LINE_VERTICAL_KERNEL,
    "face_dark_line_dilate": USER_FACE_DARK_LINE_DILATE,
    "face_dark_line_blur": USER_FACE_DARK_LINE_BLUR,
    "face_dark_line_fill_kernel": USER_FACE_DARK_LINE_FILL_KERNEL,
    "face_dark_line_repair_strength": USER_FACE_DARK_LINE_REPAIR_STRENGTH,
    "face_dark_line_detail_exclude_dilate": USER_FACE_DARK_LINE_DETAIL_EXCLUDE_DILATE,
    "face_dark_line_ear_exclude_dilate": USER_FACE_DARK_LINE_EAR_EXCLUDE_DILATE,
    "earring_reference_object_dilate": USER_EARRING_REFERENCE_OBJECT_DILATE,
    "enable_online_source_earring_object_mask": USER_ENABLE_ONLINE_SOURCE_EARRING_OBJECT_MASK,
    "online_source_earring_object_min_area": USER_ONLINE_SOURCE_EARRING_OBJECT_MIN_AREA,
    "use_refined_earring_object_mask": USER_USE_REFINED_EARRING_OBJECT_MASK,
    "earring_require_source_parser_seed": USER_EARRING_REQUIRE_SOURCE_PARSER_SEED,
    "earring_min_source_parser_area": USER_EARRING_MIN_SOURCE_PARSER_AREA,
    "earring_min_refined_object_area": USER_EARRING_MIN_REFINED_OBJECT_AREA,
    "allow_visual_earring_seed": USER_ALLOW_VISUAL_EARRING_SEED,
    "visual_earring_seed_min_area": USER_VISUAL_EARRING_SEED_MIN_AREA,
    "visual_earring_seed_max_density": USER_VISUAL_EARRING_SEED_MAX_DENSITY,
    "prefer_dataset_earring_reference": USER_PREFER_DATASET_EARRING_REFERENCE,
    "use_dataset_earring_aux": USER_USE_DATASET_EARRING_AUX,
    "expand_dataset_earring_from_reference_delta": USER_EXPAND_DATASET_EARRING_FROM_REFERENCE_DELTA,
    "earring_visible_roi_exclude_target_hair": USER_EARRING_VISIBLE_ROI_EXCLUDE_TARGET_HAIR,
    "refresh_earring_reference": USER_REFRESH_EARRING_REFERENCE,
    "ear_mask": USER_LAMBDA_EAR_MASK,
    "ear_high": USER_LAMBDA_EAR_HIGH,
    "ear_presence": USER_LAMBDA_EAR_PRESENCE,
    "ear_lighting": USER_LAMBDA_EAR_LIGHTING,
    "ear_hair_leak": USER_LAMBDA_EAR_HAIR_LEAK,
    "ear_hair_anchor": USER_LAMBDA_EAR_HAIR_ANCHOR,
    "ear_color": USER_LAMBDA_EAR_COLOR,
    "ear_highlight": USER_LAMBDA_EAR_HIGHLIGHT,
    "cleanup_anchor": USER_LAMBDA_CLEANUP_ANCHOR,
    "cleanup_low_anchor": USER_LAMBDA_CLEANUP_LOW_ANCHOR,
    "cleanup_source_reject": USER_LAMBDA_CLEANUP_SOURCE_REJECT,
    "cleanup_non_dark": USER_LAMBDA_CLEANUP_NON_DARK,
    "cleanup_high": USER_LAMBDA_CLEANUP_HIGH,
    "cleanup_texture_stat": USER_LAMBDA_CLEANUP_TEXTURE_STAT,
    "detail_high": USER_LAMBDA_DETAIL_HIGH,
    "detail_low_anchor": USER_LAMBDA_DETAIL_LOW_ANCHOR,
    "source_valid_face_high": USER_LAMBDA_SOURCE_VALID_FACE_HIGH,
    "source_valid_face_color": USER_LAMBDA_SOURCE_VALID_FACE_COLOR,
    "face_source_detail_hr": USER_LAMBDA_FACE_SOURCE_DETAIL_HR,
    "face_lowfreq_anchor_hr": USER_LAMBDA_FACE_LOWFREQ_ANCHOR_HR,
    "face_lowfreq_continuity_hr": USER_LAMBDA_FACE_LOWFREQ_CONTINUITY_HR,
    "revealed_boundary_seam_hr": USER_LAMBDA_REVEALED_BOUNDARY_SEAM_HR,
    "revealed_source_texture_stat_hr": USER_LAMBDA_REVEALED_SOURCE_TEXTURE_STAT_HR,
    "face_hair_boundary_seam_hr": USER_LAMBDA_FACE_HAIR_BOUNDARY_SEAM_HR,
    "face_lowfreq_continuity": USER_LAMBDA_FACE_LOWFREQ_CONTINUITY,
    "revealed_boundary_seam": USER_LAMBDA_REVEALED_BOUNDARY_SEAM,
    "revealed_texture_stat": USER_LAMBDA_REVEALED_TEXTURE_STAT,
    "revealed_skin_texture": USER_LAMBDA_REVEALED_SKIN_TEXTURE,
    "revealed_skin_tone": USER_LAMBDA_REVEALED_SKIN_TONE,
    "normal_face_preserve": USER_LAMBDA_NORMAL_FACE_PRESERVE,
    "face_source_dark_reject": USER_LAMBDA_FACE_SOURCE_DARK_REJECT,
    "face_source_dark_reject_margin": USER_FACE_SOURCE_DARK_REJECT_MARGIN,
    "face_source_dark_reject_source_threshold": USER_FACE_SOURCE_DARK_REJECT_SOURCE_THRESHOLD,
    "face_source_dark_reject_kernel": USER_FACE_SOURCE_DARK_REJECT_KERNEL,
    "face_source_dark_source_hair_dilate": USER_FACE_SOURCE_DARK_SOURCE_HAIR_DILATE,
    "ear_query_expand": USER_LAMBDA_EAR_QUERY_EXPAND,
    "ear_mask_area": USER_LAMBDA_EAR_MASK_AREA,
    "ear_edge": USER_LAMBDA_EAR_EDGE,
    "ear_brightness_reg": USER_LAMBDA_EAR_BRIGHTNESS_REG,
    "target_hair_ear_anchor": USER_LAMBDA_TARGET_HAIR_EAR_ANCHOR,
    "target_hair_ear_source_reject": USER_LAMBDA_TARGET_HAIR_EAR_SOURCE_REJECT,
    "target_preserve": USER_LAMBDA_TARGET_PRESERVE,
    "target_hair_preserve": USER_LAMBDA_TARGET_HAIR_PRESERVE,
    "target_preserve_exclude_dilate": USER_TARGET_PRESERVE_EXCLUDE_DILATE,
    "base_cleanup_exclude": USER_LAMBDA_BASE_CLEANUP_EXCLUDE,
    "earring_supervision_dilate": USER_EARRING_SUPERVISION_DILATE,
    "target_earring_suppress_low": USER_LAMBDA_TARGET_EARRING_SUPPRESS_LOW,
    "target_earring_suppress_high": USER_LAMBDA_TARGET_EARRING_SUPPRESS_HIGH,
    "target_ear_geometry": USER_LAMBDA_TARGET_EAR_GEOMETRY,
    "no_earring_noop": USER_LAMBDA_NO_EARRING_NOOP,
    "hoop_hole_preserve": USER_LAMBDA_HOOP_HOLE_PRESERVE,
    "earring_object_restore": USER_LAMBDA_EARRING_OBJECT_RESTORE,
    "earring_foreground_restore": USER_LAMBDA_EARRING_FOREGROUND_RESTORE,
    "use_dataset_query_mask": USER_USE_DATASET_QUERY_MASK,
    "use_dataset_source_earring_mask": USER_USE_DATASET_SOURCE_EARRING_MASK,
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
    parser = argparse.ArgumentParser(description="Post Process trainer v5")
    parser.add_argument("--dataset_profile", type=str, default=defaults["dataset_profile"])
    parser.add_argument("--name_run", type=str, default=defaults["name_run"])
    parser.add_argument("--dataset", type=Path, default=defaults["dataset"])
    parser.add_argument("--fid_dataset", type=str, default=defaults["fid_dataset"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--test_size", type=int, default=defaults["test_size"])
    parser.add_argument("--val_preview_count", type=int, default=defaults["val_preview_count"])
    parser.add_argument(
        "--val_supplement_train_previews",
        type=str2bool,
        default=defaults["val_supplement_train_previews"],
    )
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
    parser.add_argument(
        "--enable_v5_structural_compositor",
        type=str2bool,
        default=defaults["enable_v5_structural_compositor"],
    )
    parser.add_argument("--face_continuity_work_size", type=int, default=defaults["face_continuity_work_size"])
    parser.add_argument("--face_detail_soft_edge", type=int, default=defaults["face_detail_soft_edge"])
    parser.add_argument("--face_hair_soft_edge", type=int, default=defaults["face_hair_soft_edge"])
    parser.add_argument("--earring_use_native_coord_contract", type=str2bool, default=defaults["earring_use_native_coord_contract"])
    parser.add_argument("--earring_component_max_depth", type=int, default=defaults["earring_component_max_depth"])
    parser.add_argument("--earring_component_max_cumulative_cost", type=float, default=defaults["earring_component_max_cumulative_cost"])
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
    parser.add_argument("--target_visibility_expand", type=int, default=defaults["target_visibility_expand"])
    parser.add_argument("--max_target_hair_overlap", type=float, default=defaults["max_target_hair_overlap"])
    parser.add_argument("--target_ear_cover_overlap", type=float, default=defaults["target_ear_cover_overlap"])
    parser.add_argument("--min_target_visible_overlap", type=float, default=defaults["min_target_visible_overlap"])
    parser.add_argument("--min_target_ear_area", type=float, default=defaults["min_target_ear_area"])
    parser.add_argument("--earring_channel_down", type=int, default=defaults["earring_channel_down"])
    parser.add_argument("--source_earring_open_overlap", type=float, default=defaults["source_earring_open_overlap"])
    parser.add_argument("--ear_blur_kernel", type=int, default=defaults["ear_blur_kernel"])
    parser.add_argument("--ear_blur_sigma", type=float, default=defaults["ear_blur_sigma"])
    parser.add_argument("--ear_mask_hidden", type=int, default=defaults["ear_mask_hidden"])
    parser.add_argument("--ear_mask_init_bias", type=float, default=defaults["ear_mask_init_bias"])
    parser.add_argument("--earring_query_dilate", type=int, default=defaults["earring_query_dilate"])
    parser.add_argument("--earring_query_boost", type=float, default=defaults["earring_query_boost"])
    parser.add_argument("--enable_earring_query_recall", type=str2bool, default=defaults["enable_earring_query_recall"])
    parser.add_argument("--earring_query_recall_dilate", type=int, default=defaults["earring_query_recall_dilate"])
    parser.add_argument("--earring_query_downward_shift", type=int, default=defaults["earring_query_downward_shift"])
    parser.add_argument("--earring_query_lower_lobe_weight", type=float, default=defaults["earring_query_lower_lobe_weight"])
    parser.add_argument("--earring_query_candidate_boost", type=float, default=defaults["earring_query_candidate_boost"])
    parser.add_argument("--earring_query_block_protect", type=float, default=defaults["earring_query_block_protect"])
    parser.add_argument("--disable_earring_path_if_low_confidence", type=str2bool, default=defaults["disable_earring_path_if_low_confidence"])
    parser.add_argument("--earring_source_presence_min_area", type=float, default=defaults["earring_source_presence_min_area"])
    parser.add_argument("--earring_search_downward_shift", type=int, default=defaults["earring_search_downward_shift"])
    parser.add_argument("--earring_search_dilate", type=int, default=defaults["earring_search_dilate"])
    parser.add_argument("--earring_write_max_target_hair_overlap", type=float, default=defaults["earring_write_max_target_hair_overlap"])
    parser.add_argument("--earring_write_source_block_dilate", type=int, default=defaults["earring_write_source_block_dilate"])
    parser.add_argument("--earring_write_dilate", type=int, default=defaults["earring_write_dilate"])
    parser.add_argument("--earring_write_connectivity_iters", type=int, default=defaults["earring_write_connectivity_iters"])
    parser.add_argument("--earring_write_connectivity_kernel", type=int, default=defaults["earring_write_connectivity_kernel"])
    parser.add_argument("--earring_write_bridge_dilate", type=int, default=defaults["earring_write_bridge_dilate"])
    parser.add_argument("--earring_anchor_visible_dilate", type=int, default=defaults["earring_anchor_visible_dilate"])
    parser.add_argument("--earring_align_max_shift", type=int, default=defaults["earring_align_max_shift"])
    parser.add_argument("--ear_fine_support_dilate", type=int, default=defaults["ear_fine_support_dilate"])
    parser.add_argument("--earring_fine_mask_floor", type=float, default=defaults["earring_fine_mask_floor"])
    parser.add_argument("--earring_fine_mask_dilate", type=int, default=defaults["earring_fine_mask_dilate"])
    parser.add_argument("--earring_learned_fallback_alpha", type=float, default=defaults["earring_learned_fallback_alpha"])
    parser.add_argument("--earring_lobe_support_source_weight", type=float, default=defaults["earring_lobe_support_source_weight"])
    parser.add_argument("--earring_lobe_support_fine_weight", type=float, default=defaults["earring_lobe_support_fine_weight"])
    parser.add_argument("--target_hair_ear_protect_dilate", type=int, default=defaults["target_hair_ear_protect_dilate"])
    parser.add_argument("--target_hair_ear_scope_dilate", type=int, default=defaults["target_hair_ear_scope_dilate"])
    parser.add_argument("--target_hair_ear_protect_strength", type=float, default=defaults["target_hair_ear_protect_strength"])
    parser.add_argument(
        "--allow_earring_through_target_hair_ear_protect",
        type=str2bool,
        default=defaults["allow_earring_through_target_hair_ear_protect"],
    )
    parser.add_argument("--source_hair_face_suppress_dilate", type=int, default=defaults["source_hair_face_suppress_dilate"])
    parser.add_argument("--source_hair_face_suppress_strength", type=float, default=defaults["source_hair_face_suppress_strength"])
    parser.add_argument("--source_hair_face_suppress_max_y", type=float, default=defaults["source_hair_face_suppress_max_y"])
    parser.add_argument("--source_hair_face_suppress_ear_exclude_dilate", type=int, default=defaults["source_hair_face_suppress_ear_exclude_dilate"])
    parser.add_argument("--base_source_hair_exclude_dilate", type=int, default=defaults["base_source_hair_exclude_dilate"])
    parser.add_argument("--base_source_hair_exclude_strength", type=float, default=defaults["base_source_hair_exclude_strength"])
    parser.add_argument("--base_source_hair_exclude_max_y", type=float, default=defaults["base_source_hair_exclude_max_y"])
    parser.add_argument("--base_source_hair_exclude_ear_dilate", type=int, default=defaults["base_source_hair_exclude_ear_dilate"])
    parser.add_argument("--enable_source_content_gate", type=str2bool, default=defaults["enable_source_content_gate"])
    parser.add_argument("--source_content_gate_dilate", type=int, default=defaults["source_content_gate_dilate"])
    parser.add_argument("--enable_output_target_preserve", type=str2bool, default=defaults["enable_output_target_preserve"])
    parser.add_argument("--output_target_hair_preserve_dilate", type=int, default=defaults["output_target_hair_preserve_dilate"])
    parser.add_argument("--output_face_hair_seam_preserve_dilate", type=int, default=defaults["output_face_hair_seam_preserve_dilate"])
    parser.add_argument("--output_revealed_skin_preserve_dilate", type=int, default=defaults["output_revealed_skin_preserve_dilate"])
    parser.add_argument("--output_earring_keep_dilate", type=int, default=defaults["output_earring_keep_dilate"])
    parser.add_argument("--output_preserve_blur", type=int, default=defaults["output_preserve_blur"])
    parser.add_argument("--output_hairline_feather", type=int, default=defaults["output_hairline_feather"])
    parser.add_argument("--enable_revealed_skin_harmonize", type=str2bool, default=defaults["enable_revealed_skin_harmonize"])
    parser.add_argument("--revealed_skin_harmonize_strength", type=float, default=defaults["revealed_skin_harmonize_strength"])
    parser.add_argument("--revealed_skin_tone_kernel", type=int, default=defaults["revealed_skin_tone_kernel"])
    parser.add_argument("--revealed_skin_tone_sigma", type=float, default=defaults["revealed_skin_tone_sigma"])
    parser.add_argument("--revealed_skin_diffuse_iters", type=int, default=defaults["revealed_skin_diffuse_iters"])
    parser.add_argument("--revealed_skin_tone_limit", type=float, default=defaults["revealed_skin_tone_limit"])
    parser.add_argument("--revealed_skin_detail_gain", type=float, default=defaults["revealed_skin_detail_gain"])
    parser.add_argument("--revealed_skin_seam_band", type=int, default=defaults["revealed_skin_seam_band"])
    parser.add_argument("--revealed_skin_min_reference_area", type=float, default=defaults["revealed_skin_min_reference_area"])
    parser.add_argument("--enable_output_source_earring_composite", type=str2bool, default=defaults["enable_output_source_earring_composite"])
    parser.add_argument("--output_source_earring_alpha", type=float, default=defaults["output_source_earring_alpha"])
    parser.add_argument("--output_source_earring_mask_dilate", type=int, default=defaults["output_source_earring_mask_dilate"])
    parser.add_argument("--output_source_earring_mask_blur", type=int, default=defaults["output_source_earring_mask_blur"])
    parser.add_argument("--output_source_earring_align_max_shift", type=int, default=defaults["output_source_earring_align_max_shift"])
    parser.add_argument("--output_source_earring_min_area", type=float, default=defaults["output_source_earring_min_area"])
    parser.add_argument("--enable_output_earring_reference_composite", type=str2bool, default=defaults["enable_output_earring_reference_composite"])
    parser.add_argument("--output_earring_reference_alpha", type=float, default=defaults["output_earring_reference_alpha"])
    parser.add_argument("--output_earring_reference_blur", type=int, default=defaults["output_earring_reference_blur"])
    parser.add_argument("--output_earring_reference_object_dilate", type=int, default=defaults["output_earring_reference_object_dilate"])
    parser.add_argument("--output_earring_reference_diff_threshold", type=float, default=defaults["output_earring_reference_diff_threshold"])
    parser.add_argument("--output_earring_reference_high_threshold", type=float, default=defaults["output_earring_reference_high_threshold"])
    parser.add_argument("--output_earring_reference_color_threshold", type=float, default=defaults["output_earring_reference_color_threshold"])
    parser.add_argument("--output_earring_reference_strong_diff_threshold", type=float, default=defaults["output_earring_reference_strong_diff_threshold"])
    parser.add_argument("--output_earring_reference_min_area", type=float, default=defaults["output_earring_reference_min_area"])
    parser.add_argument("--enable_revealed_dark_line_repair", type=str2bool, default=defaults["enable_revealed_dark_line_repair"])
    parser.add_argument("--revealed_dark_line_threshold", type=float, default=defaults["revealed_dark_line_threshold"])
    parser.add_argument("--revealed_dark_line_source_hair_dilate", type=int, default=defaults["revealed_dark_line_source_hair_dilate"])
    parser.add_argument("--revealed_dark_line_source_edge_dilate", type=int, default=defaults["revealed_dark_line_source_edge_dilate"])
    parser.add_argument("--revealed_dark_line_edge_threshold", type=float, default=defaults["revealed_dark_line_edge_threshold"])
    parser.add_argument("--revealed_dark_line_local_kernel", type=int, default=defaults["revealed_dark_line_local_kernel"])
    parser.add_argument("--revealed_dark_line_dilate", type=int, default=defaults["revealed_dark_line_dilate"])
    parser.add_argument("--revealed_dark_line_blur", type=int, default=defaults["revealed_dark_line_blur"])
    parser.add_argument("--revealed_dark_line_fill_kernel", type=int, default=defaults["revealed_dark_line_fill_kernel"])
    parser.add_argument("--enable_face_dark_line_repair", type=str2bool, default=defaults["enable_face_dark_line_repair"])
    parser.add_argument("--face_dark_line_threshold", type=float, default=defaults["face_dark_line_threshold"])
    parser.add_argument("--face_dark_line_local_kernel", type=int, default=defaults["face_dark_line_local_kernel"])
    parser.add_argument("--face_dark_line_vertical_kernel", type=int, default=defaults["face_dark_line_vertical_kernel"])
    parser.add_argument("--face_dark_line_dilate", type=int, default=defaults["face_dark_line_dilate"])
    parser.add_argument("--face_dark_line_blur", type=int, default=defaults["face_dark_line_blur"])
    parser.add_argument("--face_dark_line_fill_kernel", type=int, default=defaults["face_dark_line_fill_kernel"])
    parser.add_argument("--face_dark_line_repair_strength", type=float, default=defaults["face_dark_line_repair_strength"])
    parser.add_argument("--face_dark_line_detail_exclude_dilate", type=int, default=defaults["face_dark_line_detail_exclude_dilate"])
    parser.add_argument("--face_dark_line_ear_exclude_dilate", type=int, default=defaults["face_dark_line_ear_exclude_dilate"])
    parser.add_argument("--earring_reference_object_dilate", type=int, default=defaults["earring_reference_object_dilate"])
    parser.add_argument("--enable_online_source_earring_object_mask", type=str2bool, default=defaults["enable_online_source_earring_object_mask"])
    parser.add_argument("--online_source_earring_object_min_area", type=int, default=defaults["online_source_earring_object_min_area"])
    parser.add_argument("--use_refined_earring_object_mask", type=str2bool, default=defaults["use_refined_earring_object_mask"])
    parser.add_argument("--earring_require_source_parser_seed", type=str2bool, default=defaults["earring_require_source_parser_seed"])
    parser.add_argument("--earring_min_source_parser_area", type=float, default=defaults["earring_min_source_parser_area"])
    parser.add_argument("--earring_min_refined_object_area", type=float, default=defaults["earring_min_refined_object_area"])
    parser.add_argument("--allow_visual_earring_seed", type=str2bool, default=defaults["allow_visual_earring_seed"])
    parser.add_argument("--visual_earring_seed_min_area", type=float, default=defaults["visual_earring_seed_min_area"])
    parser.add_argument("--visual_earring_seed_max_density", type=float, default=defaults["visual_earring_seed_max_density"])
    parser.add_argument("--prefer_dataset_earring_reference", type=str2bool, default=defaults["prefer_dataset_earring_reference"])
    parser.add_argument("--use_dataset_earring_aux", type=str2bool, default=defaults["use_dataset_earring_aux"])
    parser.add_argument(
        "--expand_dataset_earring_from_reference_delta",
        type=str2bool,
        default=defaults["expand_dataset_earring_from_reference_delta"],
    )
    parser.add_argument("--earring_visible_roi_exclude_target_hair", type=str2bool, default=defaults["earring_visible_roi_exclude_target_hair"])
    parser.add_argument("--refresh_earring_reference", type=str2bool, default=defaults["refresh_earring_reference"])
    parser.add_argument("--ear_mask", type=float, default=defaults["ear_mask"])
    parser.add_argument("--ear_high", type=float, default=defaults["ear_high"])
    parser.add_argument("--ear_presence", type=float, default=defaults["ear_presence"])
    parser.add_argument("--ear_lighting", type=float, default=defaults["ear_lighting"])
    parser.add_argument("--ear_hair_leak", type=float, default=defaults["ear_hair_leak"])
    parser.add_argument("--ear_hair_anchor", type=float, default=defaults["ear_hair_anchor"])
    parser.add_argument("--ear_color", type=float, default=defaults["ear_color"])
    parser.add_argument("--ear_highlight", type=float, default=defaults["ear_highlight"])
    parser.add_argument("--cleanup_anchor", type=float, default=defaults["cleanup_anchor"])
    parser.add_argument("--cleanup_low_anchor", type=float, default=defaults["cleanup_low_anchor"])
    parser.add_argument("--cleanup_source_reject", type=float, default=defaults["cleanup_source_reject"])
    parser.add_argument("--cleanup_non_dark", type=float, default=defaults["cleanup_non_dark"])
    parser.add_argument("--cleanup_high", type=float, default=defaults["cleanup_high"])
    parser.add_argument("--cleanup_texture_stat", type=float, default=defaults["cleanup_texture_stat"])
    parser.add_argument("--detail_high", type=float, default=defaults["detail_high"])
    parser.add_argument("--detail_low_anchor", type=float, default=defaults["detail_low_anchor"])
    parser.add_argument("--source_valid_face_high", type=float, default=defaults["source_valid_face_high"])
    parser.add_argument("--source_valid_face_color", type=float, default=defaults["source_valid_face_color"])
    parser.add_argument("--face_source_detail_hr", type=float, default=defaults["face_source_detail_hr"])
    parser.add_argument("--face_lowfreq_anchor_hr", type=float, default=defaults["face_lowfreq_anchor_hr"])
    parser.add_argument("--face_lowfreq_continuity_hr", type=float, default=defaults["face_lowfreq_continuity_hr"])
    parser.add_argument("--revealed_boundary_seam_hr", type=float, default=defaults["revealed_boundary_seam_hr"])
    parser.add_argument("--revealed_source_texture_stat_hr", type=float, default=defaults["revealed_source_texture_stat_hr"])
    parser.add_argument("--face_hair_boundary_seam_hr", type=float, default=defaults["face_hair_boundary_seam_hr"])
    parser.add_argument("--face_lowfreq_continuity", type=float, default=defaults["face_lowfreq_continuity"])
    parser.add_argument("--revealed_boundary_seam", type=float, default=defaults["revealed_boundary_seam"])
    parser.add_argument("--revealed_texture_stat", type=float, default=defaults["revealed_texture_stat"])
    parser.add_argument("--revealed_skin_texture", type=float, default=defaults["revealed_skin_texture"])
    parser.add_argument("--revealed_skin_tone", type=float, default=defaults["revealed_skin_tone"])
    parser.add_argument("--normal_face_preserve", type=float, default=defaults["normal_face_preserve"])
    parser.add_argument("--face_source_dark_reject", type=float, default=defaults["face_source_dark_reject"])
    parser.add_argument("--face_source_dark_reject_margin", type=float, default=defaults["face_source_dark_reject_margin"])
    parser.add_argument("--face_source_dark_reject_source_threshold", type=float, default=defaults["face_source_dark_reject_source_threshold"])
    parser.add_argument("--face_source_dark_reject_kernel", type=int, default=defaults["face_source_dark_reject_kernel"])
    parser.add_argument("--face_source_dark_source_hair_dilate", type=int, default=defaults["face_source_dark_source_hair_dilate"])
    parser.add_argument("--ear_query_expand", type=float, default=defaults["ear_query_expand"])
    parser.add_argument("--ear_mask_area", type=float, default=defaults["ear_mask_area"])
    parser.add_argument("--ear_edge", type=float, default=defaults["ear_edge"])
    parser.add_argument("--ear_brightness_reg", type=float, default=defaults["ear_brightness_reg"])
    parser.add_argument("--target_hair_ear_anchor", type=float, default=defaults["target_hair_ear_anchor"])
    parser.add_argument("--target_hair_ear_source_reject", type=float, default=defaults["target_hair_ear_source_reject"])
    parser.add_argument("--target_preserve", type=float, default=defaults["target_preserve"])
    parser.add_argument("--target_hair_preserve", type=float, default=defaults["target_hair_preserve"])
    parser.add_argument("--target_preserve_exclude_dilate", type=int, default=defaults["target_preserve_exclude_dilate"])
    parser.add_argument("--base_cleanup_exclude", type=float, default=defaults["base_cleanup_exclude"])
    parser.add_argument("--earring_supervision_dilate", type=int, default=defaults["earring_supervision_dilate"])
    parser.add_argument("--target_earring_suppress_low", type=float, default=defaults["target_earring_suppress_low"])
    parser.add_argument("--target_earring_suppress_high", type=float, default=defaults["target_earring_suppress_high"])
    parser.add_argument("--target_ear_geometry", type=float, default=defaults["target_ear_geometry"])
    parser.add_argument("--no_earring_noop", type=float, default=defaults["no_earring_noop"])
    parser.add_argument("--hoop_hole_preserve", type=float, default=defaults["hoop_hole_preserve"])
    parser.add_argument("--earring_object_restore", type=float, default=defaults["earring_object_restore"])
    parser.add_argument(
        "--earring_foreground_restore",
        type=float,
        default=defaults["earring_foreground_restore"],
    )
    parser.add_argument("--use_dataset_query_mask", type=str2bool, default=defaults["use_dataset_query_mask"])
    parser.add_argument(
        "--use_dataset_source_earring_mask",
        type=str2bool,
        default=defaults["use_dataset_source_earring_mask"],
    )
    parser.add_argument("--positive_only_warmup_epochs", type=int, default=defaults["positive_only_warmup_epochs"])
    parser.add_argument("--positive_sample_weight", type=float, default=defaults["positive_sample_weight"])
    parser.add_argument("--positive_query_area_threshold", type=float, default=defaults["positive_query_area_threshold"])
    parser.add_argument("--dataloader_start_method", type=str, default=defaults["dataloader_start_method"])
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=defaults["dataloader_prefetch_factor"])
    parser.add_argument("--dataloader_persistent_workers", type=str2bool, default=defaults["dataloader_persistent_workers"])
    parser.add_argument("--dataloader_pin_memory", type=str2bool, default=defaults["dataloader_pin_memory"])
    parser.add_argument("--grad_accum_steps", type=int, default=defaults["grad_accum_steps"])
    return parser


class NullLoggerV5:
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


def move_batch_to_device(batch, device, keep_on_cpu=()):
    keep_on_cpu = set(keep_on_cpu)
    return {
        key: value.to(device) if torch.is_tensor(value) and key not in keep_on_cpu else value
        for key, value in batch.items()
    }


def accumulate_metrics(target, update):
    for key, value in update.items():
        target[key] = target.get(key, 0) + (value.detach() if torch.is_tensor(value) else value)
    return target


def configure_training_stage(model: PostProcessModelV5, args):
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


class TrainerV5:
    def __init__(
        self,
        model=None,
        args=None,
        optimizer=None,
        train_dataloader=None,
        train_dataloader_positive=None,
        test_dataloader=None,
        logger=None,
        train_preview_dataloader=None,
    ):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.train_dataloader_positive = train_dataloader_positive
        self.test_dataloader = test_dataloader
        self.train_preview_dataloader = train_preview_dataloader
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
            loss_weights = {
                "lpips_scale": 0.8,
                "id": 0.1,
                "landmark": 0.1 if not args.pretrain else 0.0,
                "feat_rec": 0.01,
                "adv": args.adv_coef,
                "inpaint": args.inpaint,
                "ear_mask": args.ear_mask,
                "ear_high": args.ear_high,
                "ear_presence": args.ear_presence,
                "ear_lighting": args.ear_lighting,
                "ear_hair_leak": args.ear_hair_leak,
                "ear_hair_anchor": args.ear_hair_anchor,
                "ear_color": args.ear_color,
                "ear_highlight": args.ear_highlight,
                "cleanup_anchor": args.cleanup_anchor,
                "cleanup_low_anchor": args.cleanup_low_anchor,
                "cleanup_source_reject": args.cleanup_source_reject,
                "cleanup_non_dark": args.cleanup_non_dark,
                "cleanup_high": args.cleanup_high,
                "cleanup_texture_stat": args.cleanup_texture_stat,
                "detail_high": args.detail_high,
                "detail_low_anchor": args.detail_low_anchor,
                "source_valid_face_high": args.source_valid_face_high,
                "source_valid_face_color": args.source_valid_face_color,
                "enable_v5_structural_compositor": args.enable_v5_structural_compositor,
                "face_source_detail_hr": args.face_source_detail_hr,
                "face_lowfreq_anchor_hr": args.face_lowfreq_anchor_hr,
                "face_lowfreq_continuity_hr": args.face_lowfreq_continuity_hr,
                "revealed_boundary_seam_hr": args.revealed_boundary_seam_hr,
                "revealed_source_texture_stat_hr": args.revealed_source_texture_stat_hr,
                "face_hair_boundary_seam_hr": args.face_hair_boundary_seam_hr,
                "face_lowfreq_continuity": args.face_lowfreq_continuity,
                "revealed_boundary_seam": args.revealed_boundary_seam,
                "revealed_texture_stat": args.revealed_texture_stat,
                "revealed_skin_texture": args.revealed_skin_texture,
                "revealed_skin_tone": args.revealed_skin_tone,
                "normal_face_preserve": args.normal_face_preserve,
                "face_source_dark_reject": args.face_source_dark_reject,
                "face_source_dark_reject_margin": args.face_source_dark_reject_margin,
                "face_source_dark_reject_source_threshold": args.face_source_dark_reject_source_threshold,
                "face_source_dark_reject_kernel": args.face_source_dark_reject_kernel,
                "face_source_dark_source_hair_dilate": args.face_source_dark_source_hair_dilate,
                "face_source_dark_detail_exclude_dilate": args.face_dark_line_detail_exclude_dilate,
                "face_source_dark_ear_exclude_dilate": args.face_dark_line_ear_exclude_dilate,
                "ear_query_expand": args.ear_query_expand,
                "ear_mask_area": args.ear_mask_area,
                "ear_block_strength": args.source_hair_block_strength,
                "ear_edge": args.ear_edge,
                "ear_brightness_reg": args.ear_brightness_reg,
                "target_hair_ear_anchor": args.target_hair_ear_anchor,
                "target_hair_ear_source_reject": args.target_hair_ear_source_reject,
                "occluded_hair_anchor": args.target_hair_ear_anchor,
                "target_preserve": args.target_preserve,
                "target_hair_preserve": args.target_hair_preserve,
                "target_preserve_exclude_dilate": args.target_preserve_exclude_dilate,
                "base_cleanup_exclude": args.base_cleanup_exclude,
                "base_source_hair_exclude_dilate": args.base_source_hair_exclude_dilate,
                "base_source_hair_exclude_strength": args.base_source_hair_exclude_strength,
                "base_source_hair_exclude_max_y": args.base_source_hair_exclude_max_y,
                "base_source_hair_exclude_ear_dilate": args.base_source_hair_exclude_ear_dilate,
                "earring_supervision_dilate": args.earring_supervision_dilate,
                "target_earring_suppress_low": args.target_earring_suppress_low,
                "target_earring_suppress_high": args.target_earring_suppress_high,
                "target_ear_geometry": args.target_ear_geometry,
                "no_earring_noop": args.no_earring_noop,
                "hoop_hole_preserve": args.hoop_hole_preserve,
                "earring_object_restore": args.earring_object_restore,
                "earring_foreground_restore": args.earring_foreground_restore,
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

    def save_validation_images(self, files, epoch_tag, highres_files=None):
        if not files:
            return

        vis_dir = self.args.checkpoint_dir / "val_images" / epoch_tag
        vis_dir.mkdir(parents=True, exist_ok=True)
        for old_preview in vis_dir.glob("val_*.png"):
            old_preview.unlink()
        for old_preview in vis_dir.glob("final_highres_*.png"):
            old_preview.unlink()
        with open(vis_dir / "columns.txt", "w", encoding="utf-8") as file:
            file.write(" | ".join(VAL_COLUMNS) + "\n")

        for order, preview_row in enumerate(files):
            image = image_grid(list(map(T.functional.to_pil_image, preview_row)), 1, len(preview_row))
            image.save(vis_dir / f"val_{order:03d}.png")
        # The six-column validation sheet is intentionally 256px per sample.
        # Keep the final native-resolution output beside it so a thin stud or
        # wire is never judged from a downsampled thumbnail.
        for order, image in enumerate(highres_files or []):
            T.functional.to_pil_image(image).save(
                vis_dir / f"final_highres_{order:03d}.png"
            )

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
    def image_to_rgb(image_batch, idx, size):
        if image_batch is None:
            return torch.zeros(3, *size, dtype=torch.float32)

        image = image_batch[idx].detach().float().cpu()
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        elif image.size(0) > 3:
            image = image[:3]

        if tuple(image.shape[-2:]) != tuple(size):
            image = T.functional.resize(image, list(size), interpolation=T.InterpolationMode.BILINEAR)
        return image.clamp(0, 1)

    def build_preview_row(self, source, target, gen_w_256, gen_f_256, aux, batch, idx):
        image_size = tuple(source.shape[-2:])
        # Column 4 is the de-shadowed, pre-color-blend image.  The current
        # pipeline bakes de-shadow and color into one generator pass, so it is
        # only present when the dataset stored a separate "target_deshadow";
        # otherwise fall back to the color-blended target so the row stays valid.
        deshadow = batch.get("target_deshadow")
        if deshadow is not None:
            col4 = self.image_to_rgb(deshadow, idx, image_size)
        else:
            col4 = target[idx].detach().cpu()
        return [
            source[idx].detach().cpu(),
            self.image_to_rgb(batch.get("shape_reference"), idx, image_size),
            self.image_to_rgb(batch.get("color_reference"), idx, image_size),
            col4,
            target[idx].detach().cpu(),
            gen_f_256[idx].detach().cpu(),
        ]

    def _run_model(self, batch):
        batch = move_batch_to_device(
            batch,
            self.device,
            keep_on_cpu=("shape_reference", "color_reference"),
        )
        source_full = batch["source"]
        source = self.downsample_256(source_full).clip(0, 1)
        target = batch["target"]
        completed_hair_highres = batch["completed_hair_highres"].clamp(0, 1)
        target_mask = batch["target_mask"]
        HT_E = batch["HT_E"]
        use_dataset_earring_aux = bool(getattr(self.args, "use_dataset_earring_aux", False))
        prefer_dataset_earring_reference = bool(getattr(self.args, "prefer_dataset_earring_reference", False))
        # ``earring_instance_mask`` is the complete ordinary-or-hoop accessory
        # label.  ``hoop_instance_mask`` is topology metadata only.  Reading
        # the latter here trained every ordinary earring as an all-zero
        # no-earring sample because schema-v5 stores a zero hoop tensor for
        # every item.
        dataset_instance_mask = batch.get("earring_learning_mask") if use_dataset_earring_aux else None
        dataset_hole_mask = batch.get("earring_learning_hole_mask") if use_dataset_earring_aux else None
        dataset_instance_gate = batch.get("has_earring_learning_mask") if use_dataset_earring_aux else None
        earring_confident_mask = (
            dataset_instance_mask
            if torch.is_tensor(dataset_instance_mask)
            else (batch.get("earring_confident_mask") if use_dataset_earring_aux else None)
        )
        earring_highlight_mask = batch.get("earring_highlight_mask") if use_dataset_earring_aux else None
        if use_dataset_earring_aux:
            highlight_gate = batch.get("has_earring_highlight_mask")
            if torch.is_tensor(highlight_gate):
                use_dataset_highlight = bool((highlight_gate.float() > 0.5).all().item())
                if not use_dataset_highlight:
                    earring_highlight_mask = None
        earring_reference = batch.get("earring_learning_reference") if prefer_dataset_earring_reference else None
        use_dataset_source_mask = bool(getattr(self.args, "use_dataset_source_earring_mask", False))
        source_ear_mask = batch["source_earring_mask"] if use_dataset_source_mask else None
        source_earring_object_mask = batch.get("source_earring_object_mask") if use_dataset_source_mask else None
        if torch.is_tensor(dataset_instance_mask):
            # The instance is narrower than the historical source_earring_mask:
            # it contains only source accessory pixels, never its surrounding
            # hair/background or the inner area of a hoop.
            source_ear_mask = dataset_instance_mask
            source_earring_object_mask = dataset_instance_mask
        cleanup_masks = {key: batch[key] for key in CLEANUP_MASK_KEYS}

        latent_s, latent_f, aux = self.model(
            self.normalize(source),
            self.normalize(target),
            target_mask,
            HT_E,
            source_parsing=batch["source_parsing"],
            target_parsing=batch["target_parsing"],
            source_hair_mask=batch["source_hair_mask"],
            target_hair_mask=batch["target_hair_mask"],
            authoritative_hair_highres=self.normalize(completed_hair_highres),
            authoritative_target_highres=self.normalize(completed_hair_highres),
            query_mask=batch["query_mask"],
            source_ear_mask=source_ear_mask,
            source_earring_object_mask=source_earring_object_mask,
            presence_target=batch["presence_target"],
            earring_confident_mask=earring_confident_mask,
            earring_supervision_mask=earring_confident_mask,
            earring_highlight_mask=earring_highlight_mask,
            earring_reference=earring_reference,
            earring_mask_is_dataset=dataset_instance_gate,
            earring_reference_is_dataset=batch.get("has_earring_learning_reference") if prefer_dataset_earring_reference else None,
            hoop_hole_mask=dataset_hole_mask,
            cleanup_masks=cleanup_masks,
            revealed_skin_mask=batch.get("revealed_skin_mask"),
            revealed_skin_seam_mask=batch.get("revealed_skin_seam_mask"),
            source_visible_skin_reference_mask=batch.get("source_visible_skin_reference_mask"),
            source_skin_valid_mask=batch.get("source_skin_valid_mask"),
        )
        # The V5 final compositor consumes these runtime-only controls from
        # its args; keep them explicit rather than falling back to legacy
        # rail/write-mask defaults.
        aux["v5_structural_compositor_enabled"] = torch.full_like(
            source[:, :1],
            float(self.args.enable_v5_structural_compositor),
        )
        for key in CLEANUP_MASK_KEYS:
            aux[key] = batch[key]
        if torch.is_tensor(dataset_instance_mask) and torch.is_tensor(dataset_instance_gate):
            instance_gate = dataset_instance_gate.float().view(-1, 1, 1, 1) > 0.5
            aux["earring_instance_is_dataset"] = instance_gate.float()
            exact_instance = dataset_instance_mask.clamp(0, 1)
            exact_hole = (
                torch.zeros_like(exact_instance)
                if dataset_hole_mask is None
                else dataset_hole_mask.clamp(0, 1)
            )
            exact_instance = exact_instance * (1.0 - exact_hole).clamp(0, 1)
            exact_no_earring = (
                exact_instance.flatten(1).sum(dim=1, keepdim=True) < 1.0
            ).to(exact_instance.dtype).view(-1, 1, 1, 1).expand_as(exact_instance)

            def prefer_exact(name, value):
                current = aux.get(name)
                aux[name] = value if current is None else torch.where(instance_gate, value, current)

            # Losses must see precisely the same object topology saved by
            # pp_gen_v5.  In particular, a zero instance is an intentional
            # no-earring example, not a signal to fall back to online recall.
            prefer_exact("source_earring_mask", exact_instance)
            prefer_exact("source_earring_object_mask", exact_instance)
            prefer_exact("earring_confident_mask", exact_instance)
            prefer_exact("earring_write_mask", exact_instance)
            prefer_exact("hoop_hole_mask", exact_hole)
            prefer_exact("no_earring_case_mask", exact_no_earring)
            prefer_exact("no_earring_case", exact_no_earring)
        dataset_mask_gate = batch.get("has_earring_confident_mask")
        if torch.is_tensor(dataset_mask_gate):
            dataset_mask_gate = dataset_mask_gate.float().view(-1, 1, 1, 1) > 0.5

        def resolve_dataset_gate(flag_key):
            gate = batch.get(flag_key)
            if torch.is_tensor(gate):
                return gate.float().view(-1, 1, 1, 1) > 0.5
            return dataset_mask_gate

        def merge_dataset_aux_mask(key, gate):
            if gate is None:
                aux[key] = batch[key]
            elif aux.get(key) is None:
                aux[key] = batch[key] * gate.float()
            else:
                aux[key] = torch.where(gate, batch[key], aux[key])

        for key in PP_EXTRA_MASK_KEYS:
            if key in batch:
                if key in ONLINE_EARRING_AUX_KEYS:
                    if aux.get(key) is None:
                        merge_dataset_aux_mask(key, resolve_dataset_gate(DATASET_AUX_MASK_FLAGS[key]))
                else:
                    aux[key] = batch[key]
        for key in ("earring_presence_state", "earring_instance_confidence"):
            if key in batch:
                aux[key] = batch[key]
        if "earring_search_mask" in batch and aux.get("earring_search_mask") is None:
            merge_dataset_aux_mask(
                "earring_search_mask",
                resolve_dataset_gate(DATASET_AUX_MASK_FLAGS["earring_search_mask"]),
            )

        gen_im_W, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False)
        F_w, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False, start_layer=0, end_layer=4)

        if self.args.pretrain:
            alpha = min(1, self.cur_iter / self.args.iter_before)
            latent_f_gen = alpha * latent_f + (1 - alpha) * F_w
        else:
            latent_f_gen = latent_f

        aux["source_full_01"] = source_full.clamp(0, 1)
        gen_im_F, aux = self.model.render_refined(self.net.generator, latent_s, latent_f_gen, aux)
        return source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux, batch

    def train_one_epoch(self):
        self.model.to(self.device).train()
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
        # Losses and test-set previews always use the validation loader.  The
        # optional train loader only provides extra images for inspection.
        self.model.to(self.device).eval()
        val_losses = {}
        preview_files = []
        highres_preview_files = []
        preview_count = max(0, int(getattr(self.args, "val_preview_count", 20)))
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
            if self.fid_calc is not None:
                images_to_fid.append(to_299((gen_im_F + 1) / 2).clip(0, 1))

            for idx in range(source.size(0)):
                preview_row = self.build_preview_row(
                    source,
                    target,
                    gen_w_256,
                    gen_f_256,
                    aux,
                    batch,
                    idx,
                )
                preview_seen = self.update_preview_buffer(
                    preview_files,
                    preview_row,
                    preview_seen,
                    preview_count,
                )
                if len(highres_preview_files) < preview_count:
                    highres_preview_files.append(
                        ((gen_im_F[idx] + 1) / 2).detach().cpu().clamp(0, 1)
                    )

            del source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux
            del gen_w_256, gen_f_256
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Validation metrics above are computed exclusively from test_dataloader.
        # Training samples below only supplement the user-facing preview images.
        missing_preview_count = preview_count - len(preview_files)
        train_preview_files = []
        if (
            bool(getattr(self.args, "val_supplement_train_previews", False))
            and missing_preview_count > 0
            and self.train_preview_dataloader is not None
        ):
            for batch in tqdm(self.train_preview_dataloader, desc="Supplement train previews", leave=False):
                source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux, batch = self._run_model(batch)
                gen_w_256 = self.downsample_256((gen_im_W + 1) / 2).clip(0, 1)
                gen_f_256 = self.downsample_256((gen_im_F + 1) / 2).clip(0, 1)

                take_count = min(source.size(0), missing_preview_count - len(train_preview_files))
                for idx in range(take_count):
                    train_preview_files.append(
                        self.build_preview_row(
                            source,
                            target,
                            gen_w_256,
                            gen_f_256,
                            aux,
                            batch,
                            idx,
                        )
                    )
                    if len(highres_preview_files) < preview_count:
                        highres_preview_files.append(
                            ((gen_im_F[idx] + 1) / 2).detach().cpu().clamp(0, 1)
                        )

                del source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux
                del gen_w_256, gen_f_256
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if len(train_preview_files) >= missing_preview_count:
                    break

            preview_files.extend(train_preview_files)
            preview_dataset = self.train_preview_dataloader.dataset
            missing_reference_paths = getattr(
                preview_dataset,
                "_missing_preview_reference_paths",
                set(),
            )
            if missing_reference_paths and not getattr(
                preview_dataset,
                "_reported_missing_preview_references",
                False,
            ):
                print(
                    f"Preview references unavailable for {len(missing_reference_paths)} path(s); "
                    "used the dataset target image as the preview fallback."
                )
                preview_dataset._reported_missing_preview_references = True

        if preview_files and len(preview_files) < preview_count:
            print(
                f"Validation preview request ({preview_count}) exceeds the number of available distinct samples; "
                f"saved {len(preview_files)} distinct preview(s) without reuse."
            )

        if self.fid_calc is not None and images_to_fid:
            val_losses["FID CLIP"] = self.fid_calc(torch.cat(images_to_fid))

        for key, value in val_losses.items():
            if key != "FID CLIP":
                value = value.item() / max(1, len(self.test_dataloader))
            self.logger.log_scalars({f"val {key}": value})

        if preview_files:
            self.save_validation_images(
                preview_files,
                epoch_tag,
                highres_files=highres_preview_files,
            )
            images_to_log = [
                image_grid(list(map(T.functional.to_pil_image, preview_row)), 1, len(preview_row))
                for preview_row in preview_files
            ]
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


class DatasetPartIndexV5:
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


class PPDatasetV5(Dataset):
    def __init__(
        self,
        dataset_index: DatasetPartIndexV5,
        sample_indices,
        is_test=False,
        include_preview_references=False,
    ):
        super().__init__()
        self.dataset_index = dataset_index
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.is_test = is_test
        self.include_preview_references = include_preview_references
        self._cached_part_path = None
        self._cached_part_items = None
        self._missing_preview_reference_paths = set()
        self._reported_missing_preview_references = False

    def __len__(self):
        return len(self.sample_indices)

    def load_image(self, path):
        with Image.open(path) as image:
            return T.functional.to_tensor(image.convert("RGB"))

    def load_preview_reference(self, path, fallback):
        if not path:
            return fallback.clone()
        path = Path(path)
        if not path.is_file():
            self._missing_preview_reference_paths.add(str(path))
            return fallback.clone()
        try:
            image = self.load_image(path)
        except (OSError, ValueError):
            self._missing_preview_reference_paths.add(str(path))
            return fallback.clone()
        if tuple(image.shape[-2:]) != tuple(fallback.shape[-2:]):
            image = T.functional.resize(
                image,
                list(fallback.shape[-2:]),
                interpolation=T.InterpolationMode.BILINEAR,
            )
        return image

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

        keys_to_flip = ["source", "shape_reference", "color_reference", "target", "completed_hair_highres", "earring_reference", "earring_learning_reference", "target_mask", "HT_E", "source_hair_mask", "target_hair_mask",
                        "source_earring_mask", "target_earring_mask", "query_mask", "ear_roi",
                        "visible_ear_roi", "earring_valid_roi", "target_covered_ear_block_mask",
                        "source_hair_block_mask", "source_earring_object_mask", "source_earring_seed_mask",
                        "earring_search_mask", "earring_learning_mask", "earring_learning_hole_mask",
                        *CLEANUP_MASK_KEYS, *PP_EXTRA_MASK_KEYS]
        for key in keys_to_flip:
            if key in sample:
                sample[key] = T.functional.hflip(sample[key])
        sample["source_parsing"] = self.flip_parsing_lr(sample["source_parsing"])
        sample["target_parsing"] = self.flip_parsing_lr(sample["target_parsing"])

        left_roi = T.functional.hflip(sample["left_ear_roi"])
        right_roi = T.functional.hflip(sample["right_ear_roi"])
        sample["left_ear_roi"] = right_roi
        sample["right_ear_roi"] = left_roi
        sample["presence_target"] = sample["presence_target"][[1, 0, 2]]
        for key in ("earring_presence_state", "earring_instance_confidence"):
            if key in sample:
                sample[key] = sample[key][[1, 0]]
        return sample

    def __getitem__(self, idx):
        global_idx = int(self.sample_indices[idx])
        part_path, item_idx = self.dataset_index.locate(global_idx)
        item = self.load_part_items(part_path)[item_idx]
        fallback_mask = item["target_mask"]
        completed_hair_highres = item.get("completed_hair_highres")
        if not torch.is_tensor(completed_hair_highres):
            raise RuntimeError(
                "V5 dataset item is missing completed_hair_highres. "
                "Regenerate the schema-22 V5 dataset before training."
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
        sample = {
            "source": self.load_image(item["source_path"]),
            "target": item["target"].clone(),
            "completed_hair_highres": completed_hair_highres,
            "target_mask": item["target_mask"].clone(),
            "HT_E": item["HT_E"].clone(),
            "source_parsing": item["source_parsing"].clone(),
            "target_parsing": item["target_parsing"].clone(),
            "source_hair_mask": item["source_hair_mask"].clone(),
            "target_hair_mask": item["target_hair_mask"].clone(),
            "source_hair_block_mask": item.get("source_hair_block_mask", torch.zeros_like(fallback_mask)).clone(),
            "earring_reference": item.get("earring_reference", item["target"]).clone(),
            "has_earring_reference": torch.tensor(
                1.0 if torch.is_tensor(item.get("earring_reference")) else 0.0,
                dtype=torch.float32,
            ),
            "source_earring_mask": item["source_earring_mask"].clone(),
            "source_earring_object_mask": item.get("source_earring_object_mask", item["source_earring_mask"]).clone(),
            "earring_instance_mask": item.get(
                "earring_instance_mask",
                item.get("earring_confident_mask", item["source_earring_mask"]),
            ).clone(),
            "earring_learning_mask": item.get(
                "target_aligned_earring_alpha",
                item.get("earring_learning_mask", item.get("earring_write_mask", item["source_earring_mask"])),
            ).clone(),
            "earring_learning_hole_mask": item.get(
                "earring_learning_hole_mask",
                item.get("hoop_hole_mask", torch.zeros_like(fallback_mask)),
            ).clone(),
            "earring_learning_reference": item.get(
                "target_aligned_earring_rgb",
                item.get("earring_learning_reference", item.get("earring_reference", item["target"])),
            ).clone(),
            "earring_presence_state": item.get(
                "earring_presence_state", torch.zeros(2, dtype=torch.float32)
            ).clone(),
            "earring_instance_confidence": item.get(
                "earring_instance_confidence", torch.zeros(2, dtype=torch.float32)
            ).clone(),
            "hoop_instance_mask": item.get("hoop_instance_mask", torch.zeros_like(fallback_mask)).clone(),
            "hoop_hole_mask": item.get("hoop_hole_mask", torch.zeros_like(fallback_mask)).clone(),
            "source_earring_seed_mask": item.get("source_earring_seed_mask", item["source_earring_mask"]).clone(),
            "target_earring_mask": item["target_earring_mask"].clone(),
            "query_mask": item["query_mask"].clone(),
            "earring_search_mask": item.get("earring_search_mask", torch.zeros_like(fallback_mask)).clone(),
            "ear_roi": item["ear_roi"].clone(),
            "visible_ear_roi": item.get("visible_ear_roi", item["ear_roi"]).clone(),
            "earring_valid_roi": item.get(
                "earring_valid_roi",
                item.get("visible_ear_roi", item["ear_roi"]),
            ).clone(),
            "target_covered_ear_block_mask": item.get(
                "target_covered_ear_block_mask",
                torch.zeros_like(fallback_mask),
            ).clone(),
            "left_ear_roi": item["left_ear_roi"].clone(),
            "right_ear_roi": item["right_ear_roi"].clone(),
            "presence_target": item["presence_target"].clone(),
        }
        for key in CLEANUP_MASK_KEYS:
            sample[key] = item.get(key, torch.zeros_like(fallback_mask)).clone()
        for key in PP_EXTRA_MASK_KEYS:
            sample[key] = item.get(key, torch.zeros_like(fallback_mask)).clone()
        for mask_key, flag_key in DATASET_AUX_MASK_FLAGS.items():
            sample[flag_key] = torch.tensor(
                1.0 if item_has_nonempty_mask(item, mask_key) else 0.0,
                dtype=torch.float32,
            )
        # Unlike the older confidence flag, these flags mean that the field is
        # present in the dataset schema, even when the alpha is deliberately
        # all-zero for a no-earring sample.  That distinction prevents online
        # recall from inventing an accessory during supervision.
        sample["has_earring_instance_mask"] = torch.tensor(
            1.0 if torch.is_tensor(item.get("earring_instance_mask")) else 0.0,
            dtype=torch.float32,
        )
        sample["has_earring_learning_mask"] = torch.tensor(
            1.0 if torch.is_tensor(item.get("earring_learning_mask")) else 0.0,
            dtype=torch.float32,
        )
        sample["has_earring_learning_reference"] = torch.tensor(
            1.0 if torch.is_tensor(item.get("earring_learning_reference")) else 0.0,
            dtype=torch.float32,
        )
        # Unlike the all-earring schema flag above, a hoop flag means that this
        # sample actually contains a hollow topology.  Every sample stores a
        # zero tensor for collation, so field presence would incorrectly turn
        # ordinary earrings into hoop examples.
        sample["has_hoop_instance_mask"] = torch.tensor(
            1.0 if item_has_nonempty_mask(item, "hoop_instance_mask") else 0.0,
            dtype=torch.float32,
        )
        sample["has_hoop_hole_mask"] = torch.tensor(
            1.0 if item_has_nonempty_mask(item, "hoop_hole_mask") else 0.0,
            dtype=torch.float32,
        )
        if self.include_preview_references:
            embedded_shape = item.get("shape_reference")
            embedded_color = item.get("color_reference")
            # New V5 dataset parts embed the references so previews remain
            # meaningful when generation and training use different mounts.
            # Path loading is retained only for already-generated old parts.
            sample["shape_reference"] = (
                embedded_shape.clone()
                if torch.is_tensor(embedded_shape)
                else self.load_preview_reference(
                    item.get("shape_reference_path"),
                    sample["target"],
                )
            )
            sample["color_reference"] = (
                embedded_color.clone()
                if torch.is_tensor(embedded_color)
                else self.load_preview_reference(
                    item.get("color_reference_path"),
                    sample["target"],
                )
            )
        return self.transform(sample)


def build_dataset_index(dataset_dir: Path, query_area_threshold: float):
    config_path = dataset_dir / "dataset_config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"PP dataset metadata is missing: {config_path}. Regenerate the complete V5 dataset."
        )
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            schema_version = json.load(handle).get("schema_version")
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read PP dataset metadata: {config_path}. Regenerate the complete V5 dataset."
        ) from error
    if schema_version != PP_DATASET_SCHEMA_VERSION:
        raise RuntimeError(
            f"PP dataset schema {schema_version!r} is incompatible with V5 instance supervision "
            f"schema {PP_DATASET_SCHEMA_VERSION}. Regenerate the complete dataset in a fresh directory."
        )
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

    return DatasetPartIndexV5(files, part_lengths, positive_hints)


def item_has_nonempty_mask(item, key: str, min_area: float = 1.0) -> bool:
    value = item.get(key)
    return torch.is_tensor(value) and value.float().sum().item() > min_area


def item_mask_area(item, key: str) -> float | None:
    value = item.get(key)
    if not torch.is_tensor(value):
        return None
    return float(value.float().sum().item())


def item_visible_earring_area(item) -> float:
    visible = item.get(
        "earring_valid_roi",
        item.get("visible_ear_roi", item.get("ear_roi")),
    )
    if not torch.is_tensor(visible):
        return 0.0
    visible = visible.float()
    area = 0.0
    for key in (
        "source_earring_object_mask",
        "source_earring_mask",
        "earring_confident_mask",
        "earring_learning_mask",
    ):
        value = item.get(key)
        if torch.is_tensor(value):
            area = max(area, float((value.float() * visible).sum().item()))
    return area


def is_positive_hint(item, query_area_threshold: float) -> bool:
    has_earring_seed = bool(
        item["source_earring_mask"].sum().item() > 0
        or item_has_nonempty_mask(item, "source_earring_object_mask")
        or item_has_nonempty_mask(item, "earring_learning_mask")
        or item["target_earring_mask"].sum().item() > 0
        or item_has_nonempty_mask(item, "earring_confident_mask")
        or item_has_nonempty_mask(item, "earring_highlight_mask")
    )
    if not has_earring_seed:
        return False

    # Mixed long/short-hair datasets contain two different cases:
    # visible ears should learn earring recovery, covered ears should learn hair preservation.
    # Prefer the visible earring/object overlap; query_mask alone can be too narrow for
    # partially visible earrings and would under-sample exactly the hard recovery cases.
    if item_visible_earring_area(item) > 1.0:
        return True
    query_area = item_mask_area(item, "query_mask")
    if query_area is None:
        return False
    return query_area > max(0.0, float(query_area_threshold))


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


def main(args):
    print(
        f"Using dataset_profile={args.dataset_profile}, dataset={args.dataset}, "
        f"checkpoint_dir={args.checkpoint_dir}"
    )
    seed_everything()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset_index = build_dataset_index(args.dataset, args.positive_query_area_threshold)
    recovery_positive_count = int(sum(dataset_index.positive_hints))
    print(
        f"Dataset items={len(dataset_index)}, "
        f"earring_recovery_positive={recovery_positive_count}, "
        f"covered_or_nonpositive={len(dataset_index) - recovery_positive_count}"
    )
    test_size = min(args.test_size, max(1, len(dataset_index) // 10))
    train_indices, test_indices = split_dataset_indices(len(dataset_index), test_size, seed=42)

    train_dataset = PPDatasetV5(dataset_index, train_indices)
    positive_train_indices = [idx for idx in train_indices if dataset_index.is_positive(int(idx))]
    train_dataset_positive = PPDatasetV5(dataset_index, positive_train_indices) if positive_train_indices else None
    test_dataset = PPDatasetV5(
        dataset_index,
        test_indices,
        is_test=True,
        include_preview_references=True,
    )

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
    train_preview_dataloader = None
    if (
        bool(args.val_supplement_train_previews)
        and max(0, int(args.val_preview_count)) > len(test_dataset)
        and len(train_indices) > 0
    ):
        train_preview_dataset = PPDatasetV5(
            dataset_index,
            train_indices,
            is_test=True,
            include_preview_references=True,
        )
        train_preview_dataloader = DataLoader(
            train_preview_dataset,
            **build_dataloader_kwargs(args, shuffle=False, drop_last=False),
        )

    logger = WandbLogger(name=args.name_run, project="HairFast-PostProcess-V5") if args.use_wandb else NullLoggerV5(
        args.checkpoint_dir
    )
    logger.start_logging()

    model = PostProcessModelV5(args)
    model.load_base_checkpoint(args.base_checkpoint)
    configure_training_stage(model, args)
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=1e-4, weight_decay=0)

    trainer = TrainerV5(
        model=model,
        args=args,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        train_dataloader_positive=train_dataloader_positive,
        test_dataloader=test_dataloader,
        logger=logger,
        train_preview_dataloader=train_preview_dataloader,
    )
    if args.resume_checkpoint is not None:
        trainer.load_model(args.resume_checkpoint)
    trainer.train_loop()


if __name__ == "__main__":
    parser = build_parser(RESOLVED_USER_CONFIG)
    main(parser.parse_args())
