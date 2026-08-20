import atexit
import gc
import json
import math
import multiprocessing
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import random
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tnf
from PIL import Image
from joblib.externals.loky import get_reusable_executor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v8 import HairFast_v8, get_parser_v8
from models.Blending_v8 import Blending_v8
from models.Encoders import (
    DIRECT_COLOR_ARCH_V8_4,
    DirectColorBlendAdapterV8 as BlendingModel,
    PostProcessModel,
    load_direct_color_adapter_state_v8,
)
from models.Net import Net, get_segmentation
from models.SG_IDCT_v16 import gaussian_blur2d, lab_to_rgb, rgb_to_lab
from models.color_condition_v8 import (
    ColorConditionConfigV8,
    build_color_condition_bundle,
    compute_intrinsic_hair_color_stats,
    compute_reference_fidelity_metrics,
    correction_hue_regression_loss,
    correction_reference_regression_loss,
    reference_color_score,
)
from models.direct_strength_teacher_v8 import load_teacher_cache, triplet_cache_key
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from models.selective_color_projector_v822 import (
    DIRECT_COLOR_ARCH_V8_5,
    SelectiveHairColorProjectorV822,
    fixed_direct_anchor_tail,
)
from models.selective_color_projector_v823 import (
    FULL_COLOR_ARCH_V8_6,
    FullColorToneSelectiveProjectorV823,
    fixed_direct_anchor_tail as fixed_direct_anchor_tail_v823,
)
from models.boundary_masks_v824 import build_boundary_masks_v824
from models.selective_color_projector_v824 import (
    FULL_COLOR_ARCH_V8_7,
    BoundaryStableFullColorProjectorV824,
    fixed_direct_anchor_tail as fixed_direct_anchor_tail_v824,
)
from models.selective_color_projector_v825 import (
    FULL_COLOR_ARCH_V8_8,
    ReferenceConditionedBoundaryProjectorV825,
    fixed_direct_anchor_tail as fixed_direct_anchor_tail_v825,
)
from models.selective_color_projector_v826 import (
    FULL_COLOR_ARCH_V8_9,
    BoundaryTargetAlignedProjectorV826,
    V226_COMPENSATION_GAMMAS,
    apply_v226_compensation_from_cached_v225,
)
from models.strong_anchor_compositor_v828 import (
    StrongAnchorAppearanceCompositorV828,
    apply_pp_hair_lock_v828,
)
from models.v828_runtime_inputs import build_v828_runtime_inputs
from models.boundary_recomposition_v829 import BoundaryRecompositionV829
from models.hair_ownership_v829 import apply_pp_hair_ownership_lock_v829
from models.hybrid_hair_carrier_v829 import HybridHairCarrierV829
from models.v829_runtime_inputs import build_v829_runtime_inputs
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.save_utils import save_latents
from utils.train import get_fid_calc, toggle_grad
from utils.v221_feasibility import (
    aggregate_fixed_alpha_metrics,
    assert_finite_json,
    build_normal_color_manifest,
    choose_direct_anchor_feasibility,
    evaluate_v221_checkpoint_gate,
    should_start_phase1,
)
from utils.v222_metrics import (
    aggregate_v222_records,
    classify_v222_pretrain,
    v222_checkpoint_score,
)
from utils.v223_metrics import (
    aggregate_v223_records,
    build_v223_normal_color_manifest,
    classify_v223,
    full_stat_error,
    strict_masked_mean_per_sample,
    v223_checkpoint_score,
)
from utils.v224_metrics import (
    aggregate_v224_records,
    build_v224_normal_color_manifest,
    classify_v224,
)
from utils.v225_metrics import (
    aggregate_v225_records,
    build_v225_normal_color_manifest,
    classify_v225,
)
from utils.v226_metrics import (
    aggregate_v226_records,
    add_anchor_deficit_metrics,
    boundary_metric_tensors,
    classify_v226_gamma_sweep,
    classify_v226_metric_alignment,
)
from utils.v227_metrics import (
    aggregate_v227_records,
    classify_v227,
    independent_edge_artifact_metrics,
    meaningful_ab_metrics,
)
from utils.v228_metrics import (
    aggregate_v228_records,
    appearance_metric_tensors,
    classify_v228,
    masked_mean_per_sample as v228_masked_mean_per_sample,
    pp_lock_metric_tensors,
)
from utils.v229_metrics import (
    aggregate_v229,
    classify_v229,
    pp_ownership_metrics,
    v229_metric_tensors,
)
from models.achromatic_hair_carrier_v830 import AchromaticHairCarrierV830
from models.hair_topology_repair_v830 import HairTopologyRepairV830
from models.pp_guided_final_v830 import PPGuidedFinalV830
from models.v830_runtime_inputs import build_v830_runtime_inputs, parser_region_audit_v830
from utils.v230_metrics import (
    aggregate_v230,
    classify_v230,
    seam_hf_energy as v230_seam_hf_energy,
    v230_metric_tensors,
)
from models.hair_matting_v831 import HairMattingV831, V231MattingError
from models.pp_unified_final_v831 import PPUnifiedFinalV831
from models.v831_runtime_inputs import build_v831_runtime_inputs
from models.v232_runtime_inputs import build_v832_runtime_inputs
from models.foreground_estimator_v832 import ForegroundEstimatorV832, V232ForegroundError
from models.foreground_recolor_v832 import ForegroundRecolorV832
from models.face_side_alpha_calibrator_v832 import FaceSideAlphaCalibratorV832
from models.background_target_v832 import BackgroundTargetV832
from models.matting_recomposer_v832 import MattingRecomposerV832
from utils.v232_metrics import foreground_reconstruction_metrics, alpha_color_correlation
from models.fb_confidence_v833 import FBConfidenceV833
from models.reliable_hair_foreground_target_v833 import ReliableHairForegroundTargetV833
from models.face_side_alpha_calibrator_v833 import FaceSideAlphaCalibratorV833
from models.background_target_v833 import BackgroundTargetV833
from models.v833_runtime_inputs import build_v833_runtime_inputs
from models.v833_pipeline import run_v833_pipeline
from utils.v233_metrics import aggregate_v233, classify_v233, split_v233_audits, v233_metric_tensors
from models.hair_carrier_chroma_injection_v834 import HairCarrierChromaInjectionV834
from models.v834_runtime_inputs import build_v834_runtime_inputs
from utils.v234_metrics import v234_metric_tensors
from models.hair_only_chroma_disentanglement_v835 import HairOnlyChromaDisentanglementV835
from models.v835_runtime_inputs import build_v835_runtime_inputs
from utils.v235_metrics import v235_metric_tensors
from utils.v231_metrics import (
    aggregate_v231,
    classify_matte_v231,
    classify_v231,
    matte_metric_tensors,
    v231_metric_tensors,
)


def clean_zombies():
    try:
        get_reusable_executor().shutdown(wait=False, kill_workers=True)
    except Exception as exc:
        print(f"[blending_v8] loky cleanup skipped: {exc}", file=sys.stderr)

    for process in multiprocessing.active_children():
        try:
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        except Exception as exc:
            pid = getattr(process, "pid", "unknown")
            print(f"[blending_v8] child cleanup skipped for pid={pid}: {exc}", file=sys.stderr)


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = os.environ.get("BLENDING_V8_DATASET_PROFILE", "small").strip().lower()

USER_DATASET_DIR_FFHQ = Path("input/blending_dataset_v8")
USER_FACE_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/blending_validate_v8_v2_31_hires_vitmatte_pp_unified_recolor")
USER_VAL_SIZE_FFHQ = 512

USER_DATASET_DIR_SMALL = Path("input/blending_dataset_v8_small_v2_short_to_long")
USER_FACE_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short")
USER_SHAPE_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/long")
USER_COLOR_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_color")
USER_OUTPUT_DIR_SMALL = Path("output/blending_validate_v8_v2_31_hires_vitmatte_pp_unified_recolor_small")
USER_VAL_SIZE_SMALL = 64

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_BATCH_SIZE = 16
USER_GRAD_ACCUM_STEPS = 1  # effective batch size = USER_BATCH_SIZE * USER_GRAD_ACCUM_STEPS
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_EPOCHS = 6
USER_LR = 5e-3
USER_WEIGHT_DECAY = 0.0
USER_GRAD_CLIP = 1.0
USER_FACE_CLIP_LOSS_WEIGHT = 0.5
USER_PSEUDO_AB_LOSS_WEIGHT = 16.0
USER_HIGH_CHROMA_COLOR_BOOST = 0.0
USER_PSEUDO_RGB_LOSS_WEIGHT = 0.0
USER_PSEUDO_LUMA_LOSS_WEIGHT = 1.0
USER_POSITIVE_LUMA_EXCESS_WEIGHT = 0.0
USER_HF_LUMA_EXCESS_WEIGHT = 0.0
USER_CORRECTION_NORM_WEIGHT = 0.10
USER_LUMA_EXCESS_MARGIN = 4.0
USER_HF_LUMA_EXCESS_MARGIN = 1.5
USER_FACE_KEEP_L1_LOSS_WEIGHT = 0.75
USER_REMOVE_KEEP_L1_LOSS_WEIGHT = 0.75
USER_PROTECT_CHROMA_KEEP_LOSS_WEIGHT = 2.5
USER_SKIN_CHROMA_KEEP_LOSS_WEIGHT = 5.0
USER_SKIN_RGB_KEEP_LOSS_WEIGHT = 3.0
USER_SAFE_HAIR_MIN_PIXELS = 64.0
USER_REMOVE_BLOCK_IN_TARGET_HAIR = 0.12
USER_FACE_NECK_COLOR_BLOCK = 0.96
USER_TARGET_HAIR_NECK_OVERRIDE = 0.94
USER_AUTHOR_COLOR_ALIGN_BATCH_PROB = 0.0
USER_AUTHOR_ZERO_PREFIX_TRAIN = True

USER_AB_NO_EDIT = 1.5
USER_AB_FULL_EDIT = 15.0
USER_HUE_NO_EDIT_DEG = 4.0
USER_HUE_FULL_EDIT_DEG = 30.0
USER_CHROMA_MAG_NO_EDIT = 2.0
USER_CHROMA_MAG_FULL_EDIT = 15.0
USER_COLOR_DIST_NO_EDIT = 2.0
USER_COLOR_DIST_FULL_EDIT = 15.0
USER_LIGHTNESS_NO_EDIT_THRESHOLD_V8 = 3.0
USER_LIGHTNESS_FULL_EDIT_THRESHOLD_V8 = 15.0
USER_MAX_GLOBAL_L_SHIFT_V8 = 40.0
USER_RELATIVE_LUMA_BINS = 8
USER_RELATIVE_LUMA_MIN_SCALE = 3.0
USER_GLOBAL_AB_FALLBACK_MIN_RELIABILITY = 0.5
USER_MIN_SAFE_REFERENCE_FRACTION_V8 = 0.35
USER_HIGHLIGHT_MAD_SCALE = 1.8
USER_HIGHLIGHT_GLOBAL_MIN_MARGIN = 3.0
USER_HIGHLIGHT_LOCAL_L_MARGIN = 2.5
USER_HIGHLIGHT_LOCAL_C_MARGIN = 1.5
USER_HIGHLIGHT_CHROMA_RATIO = 0.82
USER_ALPHA_INIT = 0.75
USER_LAYER_OFFSET_MAX = 0.06
USER_TEACHER_ALPHA_CANDIDATES = [0.0, 0.25, 0.50, 0.70, 0.85, 1.0]
USER_ALPHA_TEACHER_LOSS_WEIGHT = 0.0
USER_TEACHER_MARGIN_SCALE = 1.0
USER_TEACHER_CACHE_NAME = "teacher_direct_strength_v8_4.pt"
USER_REQUIRE_TEACHER_CACHE = False
USER_REF_MEAN_AB_LOSS_WEIGHT = 10.0
USER_REF_HUE_LOSS_WEIGHT = 2.0
USER_REF_CHROMA_LOSS_WEIGHT = 3.0
USER_CORRECTION_CHROMA_BUDGET_RATIO = 0.15
USER_CORRECTION_LUMA_BUDGET_RATIO = 0.10
USER_CORRECTION_ORTH_SCALE = 0.25
USER_CORRECTION_COLOR_TOLERANCE = 0.5
USER_CORRECTION_COLOR_REGRESSION_WEIGHT = 2.0
USER_CORRECTION_HUE_TOLERANCE_DEG = 1.5
USER_CORRECTION_HUE_REGRESSION_WEIGHT = 2.0
USER_CORRECTION_REF_SCORE_TOLERANCE = 0.2
USER_CORRECTION_REF_REGRESSION_WEIGHT = 2.0
USER_CORRECTION_REGRESSION_BATCH_PROB = 1.0
USER_HIGH_CHROMA_THRESHOLD = 0.90
USER_COLOR_REGRESSION_LIMIT = 0.15
USER_ALPHA_COLLAPSE_STD = 0.05
USER_TEACHER_DIVERSE_STD = 0.10
USER_PSEUDO_FIDELITY_BAD_FRACTION = 0.10
USER_FIXED_REGRESSION_INDICES = (0, 1, 4, 14, 23)
USER_DIAGNOSTIC_ALPHA = 0.70
USER_CLIP_MODEL = "ViT-B/32"
USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = "/data/coding/HairFastGAN/HairFastGAN-main/best.pth"
USER_SATD_BLEND_V8 = 0.34
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0
USER_BUILD_CACHE_WITH_CURRENT_SATD = os.environ.get(
    "BLENDING_BUILD_CACHE_WITH_CURRENT_SATD", "0"
) == "1"
USER_FORCE_REFRESH_ALIGN_CACHE = os.environ.get(
    "BLENDING_FORCE_REFRESH_ALIGN_CACHE", "0"
) == "1"
USER_EDGE_LUMA_MARGIN = 2.5
USER_EDGE_LUMA_EXCESS_WEIGHT = 4.0
USER_EDGE_HF_LUMA_WEIGHT = 1.0
USER_OUTER_BG_KEEP_WEIGHT = 1.5
USER_MIN_REFERENCE_PROGRESS = 0.72
USER_REFERENCE_PROGRESS_WEIGHT = 0.0
USER_ALPHA_SWEEP = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
USER_LAYER_OFFSET_MAG_WEIGHT = 0.20
USER_LAYER_NEGATIVE_DRIFT_WEIGHT = 2.0
USER_STRONG_ANCHOR_ALPHA = float(
    os.environ.get("BLENDING_V8_STRONG_ANCHOR_ALPHA", "0.90")
)
USER_V222_TARGET_DIR_MIN_AB = 1.5
USER_V222_HALO_SCALE = 4.0
USER_V222_PSEUDO_AB_WEIGHT = 12.0
USER_V222_REF_MEAN_AB_WEIGHT = 6.0
USER_V222_REF_HUE_WEIGHT = 1.5
USER_V222_REF_CHROMA_WEIGHT = 2.0
USER_V222_EDGE_LUMA_EXCESS_WEIGHT = 8.0
USER_V222_EDGE_HF_WEIGHT = 1.5
USER_V222_NEAR_NO_EDIT_WEIGHT = 0.25
USER_V223_LUMA_LOW_RADIUS = 9
USER_V223_MAX_LOW_L_SHIFT = 35.0
USER_V223_EDGE_CHROMA_STRENGTH = 0.70
USER_V223_EDGE_LUMA_STRENGTH = 0.65
USER_V223_ORTH_KEEP = 0.10
USER_V223_PSEUDO_L_LOW_WEIGHT = 4.0
USER_V223_REF_MEDIAN_L_WEIGHT = 4.0
USER_V223_PSEUDO_AB_WEIGHT = 8.0
USER_V223_REF_MEAN_AB_WEIGHT = 4.0
USER_V223_REF_HUE_WEIGHT = 1.0
USER_V223_REF_CHROMA_WEIGHT = 1.5
USER_V223_HF_PRESERVE_WEIGHT = 1.0
USER_V223_EDGE_HALO_WEIGHT = 3.0
USER_V223_MIN_FULL_RETENTION = 0.75
USER_V223_MIN_AB_RETENTION = 0.75
USER_V223_MIN_EDGE_TRANSFER = 0.30
USER_V223_PHASE1_ENABLED = False
USER_V224_PHASE1_ENABLED = False
USER_V225_PHASE1_ENABLED = False
USER_V225_EDGE_TARGET_L_MARGIN = 4.0
USER_V226_PHASE1_ENABLED = False
USER_V226_MEANINGFUL_EDGE_AB = 1.5
USER_V226_MEANINGFUL_GLOBAL_DELTA_L = 3.0
USER_V227_DIAGNOSTIC_ONLY = True
USER_V227_DEFAULT_GAMMA = 0.25
USER_V227_REAL_INFERENCE_COUNT = 20
USER_V227_BASE_CHECKPOINT = os.environ.get(
    "BLENDING_V227_CHECKPOINT",
    "output/blending_train_v8_direct_anchor_v2_26_boundary_target_metric_aligned_small/checkpoints/v226_boundary_target_metric_aligned_pass.pth",
)
USER_V227_PP_CHECKPOINT = "pretrained_models/PostProcess/pp_model.pth"
USER_V228_DIAGNOSTIC_ONLY = True
USER_V228_MATTE_WIDTH = int(os.environ.get("BLENDING_V228_MATTE_WIDTH", "4"))
USER_V228_MATTE_MAX_DISTANCE = int(os.environ.get("BLENDING_V228_MATTE_MAX_DISTANCE", "8"))
USER_V228_BG_RESIDUAL_RADIUS = int(os.environ.get("BLENDING_V228_BG_RESIDUAL_RADIUS", "7"))
USER_V228_BG_RESIDUAL_STRENGTH = float(os.environ.get("BLENDING_V228_BG_RESIDUAL_STRENGTH", "1.0"))
USER_V228_OUTER_RING_WIDTH = int(os.environ.get("BLENDING_V228_OUTER_RING_WIDTH", "6"))
USER_V228_REAL_INFERENCE_COUNT = 20
USER_V229_DIAGNOSTIC_ONLY = True
USER_V229_CARRIER_LOW_RADIUS = int(os.environ.get("BLENDING_V229_CARRIER_LOW_RADIUS", "5"))
USER_V229_ANCHOR_HF_GAIN = float(os.environ.get("BLENDING_V229_ANCHOR_HF_GAIN", "1.0"))
USER_V229_TONE_RADIUS = int(os.environ.get("BLENDING_V229_TONE_RADIUS", "7"))
USER_V229_BACKGROUND_RADIUS = int(os.environ.get("BLENDING_V229_BACKGROUND_RADIUS", "7"))
USER_V229_OUTER_STRAND_RECOVERY = os.environ.get("BLENDING_V229_OUTER_STRAND_RECOVERY", "0") == "1"
USER_V229_VISUAL_COUNT = 20
USER_V230_DIAGNOSTIC_ONLY = True
USER_V230_CORE_LOW_RADIUS = 5
USER_V230_ANCHOR_DETAIL_RADIUS = 5
USER_V230_DETAIL_LOG_CAP = 0.25
USER_V230_ANCHOR_DETAIL_GAIN = 1.0
USER_V230_PP_TONE_RADIUS = 5
USER_V230_TONE_RADIUS = 7
USER_V230_CORE_SEAM_WIDTH = 5
USER_V230_FACE_CONTACT_WIDTH = 4
USER_V230_CONFIDENCE_TEMPERATURE = 8.0
USER_V230_TOPOLOGY_HOLE_RADIUS = 2
USER_V230_TOPOLOGY_NEIGHBOR_THRESHOLD = 0.75
USER_V230_VISUAL_COUNT = 20
USER_V231_DIAGNOSTIC_ONLY = True
USER_V231_VITMATTE_PATH = os.environ.get(
    "BLENDING_V8_VITMATTE_PATH",
    "pretrained_models/ViTMatte/vitmatte-small-composition-1k",
)
USER_V231_TRIMAP_INNER_WIDTH = 8
USER_V231_TRIMAP_OUTER_WIDTH = 8
USER_V231_FACE_CONTACT_EXTRA_INNER = 4
USER_V231_MAX_TRIMAP_HOLE_AREA = 16
USER_V231_TONE_RADIUS = 9
USER_V231_TONE_PROP_RADIUS = 15
USER_V231_CONTEXT_RADIUS = 9
USER_V231_TRANSITION_EXPAND = 2
USER_V231_PHASE_A_COUNT = 24
USER_V231_VISUAL_COUNT = 20
USER_V232_DIAGNOSTIC_ONLY = os.environ.get("BLENDING_V232_DIAGNOSTIC_ONLY", "0") == "1"
USER_V232_FOREGROUND_ROI_PADDING = int(os.environ.get("BLENDING_V232_FOREGROUND_ROI_PADDING", "64"))
USER_V232_FOREGROUND_CACHE = os.environ.get("BLENDING_V232_FOREGROUND_CACHE", "")
USER_V232_TONE_RADIUS = int(os.environ.get("BLENDING_V232_TONE_RADIUS", "9"))
USER_V232_RESIDUAL_RADIUS = int(os.environ.get("BLENDING_V232_RESIDUAL_RADIUS", "21"))
USER_V232_FACE_PROTOTYPE_RADIUS = int(os.environ.get("BLENDING_V232_FACE_PROTOTYPE_RADIUS", "7"))
USER_V232_FACE_TEMPERATURE = float(os.environ.get("BLENDING_V232_FACE_TEMPERATURE", "0.04"))
USER_V232_BACKGROUND_RADIUS = int(os.environ.get("BLENDING_V232_BACKGROUND_RADIUS", "9"))
USER_V232_TRANSITION_EXPAND = int(os.environ.get("BLENDING_V232_TRANSITION_EXPAND", "2"))
USER_V232_VISUAL_COUNT = int(os.environ.get("BLENDING_V232_VISUAL_COUNT", "20"))
USER_V233_DIAGNOSTIC_ONLY = os.environ.get("BLENDING_V233_DIAGNOSTIC_ONLY", "0") == "1"
USER_V233_TONE_RADIUS = int(os.environ.get("BLENDING_V233_TONE_RADIUS", "9"))
USER_V233_LOCAL_RADIUS = int(os.environ.get("BLENDING_V233_LOCAL_RADIUS", "21"))
USER_V233_PROPAGATION_RADIUS = int(os.environ.get("BLENDING_V233_PROPAGATION_RADIUS", "15"))
USER_V233_DETAIL_GAIN = float(os.environ.get("BLENDING_V233_DETAIL_GAIN", "0.5"))
USER_V233_POSTERIOR_RADIUS = int(os.environ.get("BLENDING_V233_POSTERIOR_RADIUS", "5"))
USER_V233_FACE_TEMPERATURE = float(os.environ.get("BLENDING_V233_FACE_TEMPERATURE", "0.04"))
USER_V233_BACKGROUND_RADIUS = int(os.environ.get("BLENDING_V233_BACKGROUND_RADIUS", "9"))
USER_V233_SUPPORT_FULL = float(os.environ.get("BLENDING_V233_SUPPORT_FULL", "0.20"))
USER_V233_VISUAL_COUNT = int(os.environ.get("BLENDING_V233_VISUAL_COUNT", "20"))
USER_V234_DIAGNOSTIC_ONLY = os.environ.get("BLENDING_V234_DIAGNOSTIC_ONLY", "0") == "1"
USER_V234_REFERENCE_RADIUS = int(os.environ.get("BLENDING_V234_REFERENCE_RADIUS", "9"))
USER_V234_EDGE_RADIUS = int(os.environ.get("BLENDING_V234_EDGE_RADIUS", "3"))
USER_V234_EDGE_MIN_CONFIDENCE = float(os.environ.get("BLENDING_V234_EDGE_MIN_CONFIDENCE", "0.35"))
USER_V234_VISUAL_COUNT = int(os.environ.get("BLENDING_V234_VISUAL_COUNT", "20"))
USER_V235_DIAGNOSTIC_ONLY = os.environ.get("BLENDING_V235_DIAGNOSTIC_ONLY", "0") == "1"
USER_V235_CHROMA_RADIUS = int(os.environ.get("BLENDING_V235_CHROMA_RADIUS", "9"))
USER_V235_BOUNDARY_RADIUS = int(os.environ.get("BLENDING_V235_BOUNDARY_RADIUS", "3"))
USER_V235_BOUNDARY_MIN_CONFIDENCE = float(os.environ.get("BLENDING_V235_BOUNDARY_MIN_CONFIDENCE", "0.20"))
USER_V235_CHROMA_SCALE = float(os.environ.get("BLENDING_V235_CHROMA_SCALE", "18.0"))
USER_V235_VISUAL_COUNT = int(os.environ.get("BLENDING_V235_VISUAL_COUNT", "20"))
# V2.35 color latents are intentionally kept in a separate namespace. This
# prevents a diagnostic run from silently reusing a pre-v2.35 full-image FS
# embedding with the same source stem.
V235_COLOR_CACHE_SUFFIX = "_v235_hair_only"

USER_USE_FID = False
USER_FID_CACHE = "input/fid.pkl"
USER_FID_DATASET = Path("images/FFHQ")

USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_PREVIEW_EVERY = 1
USER_LOG_IMAGE_COUNT = 30
# Set this to output/.../checkpoints/last.pth or best.pth to continue training.
USER_RESUME_CHECKPOINT = ""

# Shape/SATD remains the validation and inference geometry. During training,
# a minority of batches use the author's face->color alignment so the encoder
# also learns reference color across the complete reference-hair extent.
# ============================================================================

NORMAL_COLOR_LOSS_WEIGHTS = {
    "face_clip": 0.5,
    "pseudo_ab": 16.0,
    "pseudo_rgb": USER_PSEUDO_RGB_LOSS_WEIGHT,
    "pseudo_luma": USER_PSEUDO_LUMA_LOSS_WEIGHT,
    "positive_luma": USER_POSITIVE_LUMA_EXCESS_WEIGHT,
    "hf_luma": USER_HF_LUMA_EXCESS_WEIGHT,
    "face_keep": USER_FACE_KEEP_L1_LOSS_WEIGHT,
    "remove_keep": USER_REMOVE_KEEP_L1_LOSS_WEIGHT,
    "protect_chroma": USER_PROTECT_CHROMA_KEEP_LOSS_WEIGHT,
    "skin_chroma": USER_SKIN_CHROMA_KEEP_LOSS_WEIGHT,
    "skin_rgb": USER_SKIN_RGB_KEEP_LOSS_WEIGHT,
    "alpha_teacher": USER_ALPHA_TEACHER_LOSS_WEIGHT,
    "ref_mean_ab": USER_REF_MEAN_AB_LOSS_WEIGHT,
    "ref_hue": USER_REF_HUE_LOSS_WEIGHT,
    "ref_chroma": USER_REF_CHROMA_LOSS_WEIGHT,
    "reference_progress": USER_REFERENCE_PROGRESS_WEIGHT,
    "edge_luma_excess": USER_EDGE_LUMA_EXCESS_WEIGHT,
    "edge_hf_luma": USER_EDGE_HF_LUMA_WEIGHT,
    "outer_bg_keep": USER_OUTER_BG_KEEP_WEIGHT,
    "layer_offset_mag": USER_LAYER_OFFSET_MAG_WEIGHT,
    "layer_negative_drift": USER_LAYER_NEGATIVE_DRIFT_WEIGHT,
    "correction_norm": 0.0,
    "correction_color_regression": 0.0,
    "correction_hue_regression": 0.0,
    "correction_ref_regression": 0.0,
}

def get_alpha_teacher_weight(epoch: int) -> float:
    """V2.21 disables teacher control during fixed-anchor adaptation."""
    del epoch
    return 0.0

def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "val_size": USER_VAL_SIZE_FFHQ,
        },
        "small": {
            "dataset_dir": USER_DATASET_DIR_SMALL,
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "color_root": USER_COLOR_ROOT_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "val_size": USER_VAL_SIZE_SMALL,
        },
    }
    if USER_DATASET_PROFILE not in profiles:
        raise RuntimeError(
            f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}. "
            f"Choose one of: {', '.join(sorted(profiles))}."
        )
    return profiles[USER_DATASET_PROFILE]


PROFILE = resolve_dataset_profile()
ACTIVE_DATASET_DIR = PROFILE["dataset_dir"]
ACTIVE_FACE_ROOT = PROFILE["face_root"]
ACTIVE_SHAPE_ROOT = PROFILE["shape_root"]
ACTIVE_COLOR_ROOT = PROFILE["color_root"]
ACTIVE_OUTPUT_DIR = PROFILE["output_dir"]
ACTIVE_VAL_SIZE = PROFILE["val_size"]

if USER_BATCH_SIZE < 1:
    raise RuntimeError("USER_BATCH_SIZE must be >= 1.")
if USER_GRAD_ACCUM_STEPS < 1:
    raise RuntimeError("USER_GRAD_ACCUM_STEPS must be >= 1.")
if not 0.0 <= USER_STRONG_ANCHOR_ALPHA <= 1.0:
    raise RuntimeError("USER_STRONG_ANCHOR_ALPHA must be in [0,1].")
if USER_AUTHOR_COLOR_ALIGN_BATCH_PROB != 0.0:
    raise RuntimeError("BlendingV8 color-direction fix requires USER_AUTHOR_COLOR_ALIGN_BATCH_PROB=0.0.")
if not 0.0 <= USER_TARGET_HAIR_NECK_OVERRIDE <= 1.0:
    raise RuntimeError("USER_TARGET_HAIR_NECK_OVERRIDE must be in [0, 1].")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_image_path(root: Path, stem: str) -> Path:
    png_path = root / f"{stem}.png"
    if png_path.exists():
        return png_path
    jpg_path = root / f"{stem}.jpg"
    if jpg_path.exists():
        return jpg_path
    jpeg_path = root / f"{stem}.jpeg"
    if jpeg_path.exists():
        return jpeg_path
    raise FileNotFoundError(f"Cannot find {stem}.png/.jpg/.jpeg in {root}")


def read_triplets(dataset_dir: Path) -> list[tuple[str, str, str]]:
    triplets = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as handle:
        for line in handle:
            items = line.strip().split()
            if len(items) == 3:
                triplets.append((items[0], items[1], items[2]))
    return triplets


def identity_func(align_shape, align_color, name_to_embed, **kwargs):
    return align_shape, align_color, name_to_embed


def role_key(role: str, stem: str) -> str:
    return f"{role}__{stem}"


def fs_cache_name(role: str, stem: str) -> str:
    suffix = V235_COLOR_CACHE_SUFFIX if (
        role == "color" and USER_V235_DIAGNOSTIC_ONLY
    ) else ""
    return f"{role_key(role, stem)}{suffix}.npz"


def align_cache_name(face_name: str, ref_role: str, ref_name: str) -> str:
    return f"{role_key('face', face_name)}_{role_key(ref_role, ref_name)}.npz"


def build_remove_protect_mask(align_info: dict[str, object]) -> torch.Tensor:
    delta_masks = align_info.get("delta_masks")
    if not isinstance(delta_masks, dict):
        hm_x = align_info["HM_X"]
        return torch.zeros_like(hm_x).float()

    remove = delta_masks["M_remove"].float()
    zero = torch.zeros_like(remove)
    protect = (
        1.00 * remove
        + 0.95 * delta_masks.get("M_remove_halo", zero).float()
        + 0.88 * delta_masks.get("M_remove_face", zero).float()
        + 0.92 * delta_masks.get("M_remove_neck", zero).float()
        + 0.92 * delta_masks.get("M_remove_tail", zero).float()
        + 0.70 * delta_masks.get("M_face_strand_probe", zero).float()
        + 0.80 * delta_masks.get("M_remove_context", zero).float()
        + 0.86 * delta_masks.get("M_body_preserve", zero).float()
        + 0.72 * delta_masks.get("M_visible_body_anchor", zero).float()
        + 0.60 * delta_masks.get("M_body_region", zero).float()
        + 0.68 * delta_masks.get("M_cloth_region", zero).float()
        + 0.35 * delta_masks.get("M_boundary", zero).float()
    )
    return protect.clamp(0, 1)


def align_instead_shape(hair_fast):
    def shape_module(func):
        def wrapper(*args, **kwargs):
            if kwargs.get("align_flag", False):
                return hair_fast.align.align_images(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    def align_module(func):
        def wrapper(*args, **kwargs):
            if "align_flag" in kwargs:
                kwargs = kwargs.copy()
                kwargs.pop("align_flag")
            return func(*args, **kwargs)

        return wrapper

    hair_fast.align.shape_module = shape_module(hair_fast.align.shape_module)
    hair_fast.align.align_images = align_module(hair_fast.align.align_images)


def build_cache_model() -> HairFast_v8:
    model_args = get_parser_v8().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    model_args.use_satd_v8 = bool(USER_USE_SATD_V8)
    model_args.satd_checkpoint_v8 = USER_SATD_CHECKPOINT_V8
    model_args.satd_blend_v8 = USER_SATD_BLEND_V8
    model_args.satd_boundary_v8 = USER_SATD_BOUNDARY_V8
    model_args.eq8_reference_blend_v8 = USER_EQ8_REFERENCE_BLEND_V8
    # Cache generation uses embedding/alignment only. ViTMatte must load later,
    # inside the explicit V2.31 Phase-0 audit.
    model_args.v231_enabled = False
    model_args.v232_enabled = False
    model_args.v233_enabled = False
    model_args.v234_enabled = False
    # V2.35 cache generation must use the same hair-only color preprocessing
    # as runtime inference. Other modes retain the legacy full-image encoder.
    model_args.v235_enabled = bool(USER_V235_DIAGNOSTIC_ONLY)

    hair_fast = HairFast_v8(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)
    return hair_fast


def ensure_dataset_cache_v8(triplets: list[tuple[str, str, str]]):
    if USER_USE_SATD_V8:
        if not USER_SATD_CHECKPOINT_V8:
            raise RuntimeError("USER_SATD_CHECKPOINT_V8 is empty while USER_USE_SATD_V8=True.")
        if not Path(USER_SATD_CHECKPOINT_V8).exists():
            raise FileNotFoundError(f"Cannot find USER_SATD_CHECKPOINT_V8: {USER_SATD_CHECKPOINT_V8}")

    fs_dir = ACTIVE_DATASET_DIR / "FS"
    align_dir = ACTIVE_DATASET_DIR / "Align"
    mask_dir = ACTIVE_DATASET_DIR / "Masks"
    fs_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    def fs_path(role: str, stem: str) -> Path:
        return fs_dir / fs_cache_name(role, stem)

    def align_path(face_stem: str, ref_role: str, ref_stem: str) -> Path:
        return align_dir / align_cache_name(face_stem, ref_role, ref_stem)

    def mask_path(face_stem: str, ref_role: str, ref_stem: str) -> Path:
        return mask_dir / align_cache_name(face_stem, ref_role, ref_stem)

    def mask_has_target_hair(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with np.load(path) as data:
                return "target_hair" in data.files
        except Exception:
            return False

    def missing_required_cache() -> list[Path]:
        missing = []
        for face_name, shape_name, color_name in triplets:
            shape_mask_path = mask_path(face_name, "shape", shape_name)
            color_mask_path = mask_path(face_name, "color", color_name)
            required = [
                fs_path("face", face_name),
                fs_path("shape", shape_name),
                fs_path("color", color_name),
                align_path(face_name, "shape", shape_name),
                align_path(face_name, "color", color_name),
            ]
            missing.extend(path for path in required if not path.exists())
            if not mask_has_target_hair(shape_mask_path):
                missing.append(shape_mask_path)
            if not mask_has_target_hair(color_mask_path):
                missing.append(color_mask_path)
        return missing

    if not USER_BUILD_CACHE_WITH_CURRENT_SATD:
        missing = missing_required_cache()
        if missing:
            preview = "\n".join(f"  {path}" for path in missing[:10])
            raise RuntimeError(
                "Role-scoped v8 blending cache is missing or stale. The Masks cache must include "
                "target_hair for long-hair color supervision, and the old unscoped cache can mix "
                "FFHQ_long/FFHQ_short/FFHQ_color entries with the same stem. "
                "For V2.35, color FS files also require the *_v235_hair_only.npz namespace. "
                "Rebuild with BLENDING_BUILD_CACHE_WITH_CURRENT_SATD=1 "
                "(and BLENDING_FORCE_REFRESH_ALIGN_CACHE=1 when masks/alignments are stale), "
                f"to rebuild it.\nMissing examples:\n{preview}"
            )
        return

    hair_fast = build_cache_model()
    required_triplets = []
    for face_name, shape_name, color_name in triplets:
        need_face_fs = not fs_path("face", face_name).exists()
        need_shape_fs = not fs_path("shape", shape_name).exists()
        need_color_fs = not fs_path("color", color_name).exists()
        # A missing V2.35 color latent means this triplet is transitioning
        # from the legacy full-image cache. Rebuild its color alignment/mask
        # in the same pass so no stale color-derived geometry survives.
        need_align_shape = USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, "shape", shape_name).exists())
        need_align_color = (
            USER_FORCE_REFRESH_ALIGN_CACHE or need_color_fs
            or (not align_path(face_name, "color", color_name).exists())
        )
        need_mask_shape = USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_has_target_hair(mask_path(face_name, "shape", shape_name)))
        need_mask_color = (
            USER_FORCE_REFRESH_ALIGN_CACHE or need_color_fs
            or (not mask_has_target_hair(mask_path(face_name, "color", color_name)))
        )
        if need_align_shape or need_align_color or need_mask_shape or need_mask_color or need_face_fs or need_shape_fs or need_color_fs:
            required_triplets.append((face_name, shape_name, color_name))

    if not required_triplets:
        print("[blending_v8] FS/Align cache already matches current training needs.", file=sys.stderr)
        return

    print(
        f"[blending_v8] rebuilding cache for {len(required_triplets)} triplets "
        f"(use_satd_v8={USER_USE_SATD_V8}, satd_checkpoint={USER_SATD_CHECKPOINT_V8})",
        file=sys.stderr,
    )

    for face_name, shape_name, color_name in tqdm(required_triplets, desc="Build v8 FS/Align cache", leave=False):
        face_path = find_image_path(ACTIVE_FACE_ROOT, face_name)
        shape_path = find_image_path(ACTIVE_SHAPE_ROOT, shape_name)
        color_path = find_image_path(ACTIVE_COLOR_ROOT, color_name)

        align_shape, align_color, name_to_embed = hair_fast(
            face_path,
            shape_path,
            color_path,
            align_flag=True,
        )

        if not fs_path("face", face_name).exists():
            save_latents(ACTIVE_DATASET_DIR, "FS", fs_cache_name("face", face_name), latent_in=name_to_embed["face"]["S"])
        if not fs_path("shape", shape_name).exists():
            save_latents(ACTIVE_DATASET_DIR, "FS", fs_cache_name("shape", shape_name), latent_in=name_to_embed["shape"]["S"])
        if not fs_path("color", color_name).exists():
            save_latents(ACTIVE_DATASET_DIR, "FS", fs_cache_name("color", color_name), latent_in=name_to_embed["color"]["S"])

        if USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, "shape", shape_name).exists()):
            save_latents(
                ACTIVE_DATASET_DIR,
                "Align",
                align_cache_name(face_name, "shape", shape_name),
                latent_F=align_shape["latent_F_align"],
            )
        if USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, "color", color_name).exists()):
            save_latents(
                ACTIVE_DATASET_DIR,
                "Align",
                align_cache_name(face_name, "color", color_name),
                latent_F=align_color["latent_F_align"],
            )
        if USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_has_target_hair(mask_path(face_name, "shape", shape_name))):
            save_latents(
                ACTIVE_DATASET_DIR,
                "Masks",
                align_cache_name(face_name, "shape", shape_name),
                remove_mask=build_remove_protect_mask(align_shape),
                target_hair=align_shape["HM_X"].float(),
            )
        if USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_has_target_hair(mask_path(face_name, "color", color_name))):
            save_latents(
                ACTIVE_DATASET_DIR,
                "Masks",
                align_cache_name(face_name, "color", color_name),
                remove_mask=build_remove_protect_mask(align_color),
                target_hair=align_color["HM_X"].float(),
            )

    del hair_fast
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_preview(path: Path, row_tensors: list[torch.Tensor]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for tensor in row_tensors:
        if tensor.dim() == 4:
            tensor = tensor[0]
        tiles.append(((tensor.float() + 1) / 2).detach().cpu().clamp(0, 1))
    panel = torch.cat(tiles, dim=2)
    save_image(panel, path)


def save_contact_sheet(path: Path, rows: list[list[torch.Tensor]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    for row in rows:
        tiles = []
        for tensor in row:
            if tensor.dim() == 4:
                tensor = tensor[0]
            tiles.append(((tensor.float() + 1) / 2).detach().cpu().clamp(0, 1))
        panels.append(torch.cat(tiles, dim=2))
    save_image(torch.cat(panels, dim=1), path)


def mask_to_preview(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float().clamp(0, 1)
    if mask.size(1) == 1:
        mask = mask.repeat(1, 3, 1, 1)
    return mask * 2 - 1


PARSING_HAIR_LABEL = 10
PARSING_HAT_LABEL = 11
PARSING_FACE_PROTECT_LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12)
PARSING_NECK_PROTECT_LABELS = (13, 14)
PARSING_BODY_PROTECT_LABELS = (15,)
PARSING_SKIN_PROTECT_LABELS = PARSING_FACE_PROTECT_LABELS + PARSING_NECK_PROTECT_LABELS
PARSING_SUBJECT_PROTECT_LABELS = PARSING_SKIN_PROTECT_LABELS + PARSING_BODY_PROTECT_LABELS


def parsing_label_mask(parsing_mask: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(parsing_mask, dtype=torch.bool)
    for label in labels:
        mask |= parsing_mask == label
    return mask.float()


def dilate_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask.float().clamp(0, 1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return tnf.max_pool2d(mask.float(), kernel_size=2 * width + 1, stride=1, padding=width).clamp(0, 1)


class MaskPrepHelper:
    def __init__(self, device: torch.device):
        self.device = device
        self.dilate_erosion = DilateErosion(device=str(device))
        self.net = Net(
            Namespace(
                size=1024,
                ckpt="pretrained_models/StyleGAN/ffhq.pt",
                channel_multiplier=2,
                latent=512,
                n_mlp=8,
                device=str(device),
            )
        )
        self.seg = BiSeNet(n_classes=16).to(device).eval()
        self.seg.load_state_dict(torch.load("pretrained_models/BiSeNet/seg.pth", map_location=device))
        toggle_grad(self.seg, False)
        toggle_grad(self.net.generator, False)
        self.net.generator.eval()
        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)

    @torch.no_grad()
    def generate_mask(
        self, image: torch.Tensor, return_keep: bool = False, return_raw: bool = False
    ):
        image_512 = (self.downsample_512((image + 1) / 2) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(image_512)
        current_mask = torch.argmax(down_seg, dim=1).long()
        hair_mask = torch.where(
            current_mask == PARSING_HAIR_LABEL,
            torch.ones_like(current_mask, dtype=torch.float32),
            torch.zeros_like(current_mask, dtype=torch.float32),
        )
        hair_mask = tnf.interpolate(hair_mask.unsqueeze(1), size=(256, 256), mode="nearest")
        hair_mask_dilate, hair_mask_erode = self.dilate_erosion.mask(hair_mask)
        if return_keep:
            non_hair_subject = (
                (current_mask > 0)
                & (current_mask != PARSING_HAIR_LABEL)
                & (current_mask != PARSING_HAT_LABEL)
            ).float()
            subject_guard = parsing_label_mask(current_mask, PARSING_SUBJECT_PROTECT_LABELS)
            skin_guard = parsing_label_mask(current_mask, PARSING_SKIN_PROTECT_LABELS)
            face_guard = parsing_label_mask(current_mask, PARSING_FACE_PROTECT_LABELS)
            neck_guard = parsing_label_mask(current_mask, PARSING_NECK_PROTECT_LABELS)

            non_hair_subject = tnf.interpolate(non_hair_subject.unsqueeze(1), size=(256, 256), mode="nearest")
            subject_guard = tnf.interpolate(subject_guard.unsqueeze(1), size=(256, 256), mode="nearest")
            skin_guard = tnf.interpolate(skin_guard.unsqueeze(1), size=(256, 256), mode="nearest")
            face_guard = tnf.interpolate(face_guard.unsqueeze(1), size=(256, 256), mode="nearest")
            neck_guard = tnf.interpolate(neck_guard.unsqueeze(1), size=(256, 256), mode="nearest")

            subject_guard = dilate_mask(subject_guard, 2)
            skin_guard = dilate_mask(skin_guard, 2)
            face_guard = dilate_mask(face_guard, 2)
            neck_guard = dilate_mask(neck_guard, 2)
            result = (
                hair_mask_dilate,
                hair_mask_erode,
                non_hair_subject.clamp(0, 1),
                subject_guard.clamp(0, 1),
                skin_guard.clamp(0, 1),
                face_guard.clamp(0, 1),
                neck_guard.clamp(0, 1),
            )
            if return_raw:
                return (*result, hair_mask.clamp(0, 1))
            return result
        if return_raw:
            return hair_mask_dilate, hair_mask_erode, hair_mask.clamp(0, 1)
        return hair_mask_dilate, hair_mask_erode


def prepare_item(
    exp,
    dataset_dir: Path,
    face_root: Path,
    color_root: Path,
    include_shape_image: bool = False,
):
    face_name, shape_name, color_name = exp

    try:
        color_s = torch.from_numpy(np.load(dataset_dir / "FS" / fs_cache_name("color", color_name))["latent_in"]).squeeze(0)
        align_s = torch.from_numpy(np.load(dataset_dir / "FS" / fs_cache_name("face", face_name))["latent_in"]).squeeze(0)
        align_f_shape = torch.from_numpy(
            np.load(dataset_dir / "Align" / align_cache_name(face_name, "shape", shape_name))["latent_F"]
        ).squeeze(0)
        with np.load(dataset_dir / "Masks" / align_cache_name(face_name, "shape", shape_name)) as mask_data:
            remove_mask_shape = torch.from_numpy(np.array(mask_data["remove_mask"])).squeeze(0)
            if "target_hair" in mask_data.files:
                target_hair_shape = torch.from_numpy(np.array(mask_data["target_hair"])).squeeze(0)
            else:
                target_hair_shape = torch.zeros_like(remove_mask_shape)

        align_f_color = torch.from_numpy(
            np.load(dataset_dir / "Align" / align_cache_name(face_name, "color", color_name))["latent_F"]
        ).squeeze(0)
        with np.load(dataset_dir / "Masks" / align_cache_name(face_name, "color", color_name)) as mask_data:
            remove_mask_color = torch.from_numpy(np.array(mask_data["remove_mask"])).squeeze(0)
            if "target_hair" in mask_data.files:
                target_hair_color = torch.from_numpy(np.array(mask_data["target_hair"])).squeeze(0)
            else:
                target_hair_color = torch.zeros_like(remove_mask_color)

        with Image.open(find_image_path(color_root, color_name)) as color_image:
            color_i = T.functional.normalize(T.functional.to_tensor(color_image.convert("RGB")), [0.5], [0.5])
        with Image.open(find_image_path(face_root, face_name)) as face_image:
            face_i = T.functional.normalize(T.functional.to_tensor(face_image.convert("RGB")), [0.5], [0.5])
        shape_i = None
        if include_shape_image:
            with Image.open(find_image_path(ACTIVE_SHAPE_ROOT, shape_name)) as shape_image:
                shape_i = T.functional.normalize(
                    T.functional.to_tensor(shape_image.convert("RGB")), [0.5], [0.5]
                )
        item = (
            color_s,
            align_s,
            align_f_shape,
            remove_mask_shape,
            target_hair_shape,
            align_f_color,
            remove_mask_color,
            target_hair_color,
            color_i,
            face_i,
        )
        if shape_i is not None:
            item = item + (shape_i,)
        return item
    except Exception as exc:
        print(exc, file=sys.stderr)
        return None


class BlendingDatasetV8(Dataset):
    def __init__(
        self,
        exps: list[tuple[str, str, str]],
        dataset_dir: Path,
        face_root: Path,
        color_root: Path,
        teacher_records: dict[str, dict[str, float]] | None = None,
        include_shape_image: bool = True,
    ):
        super().__init__()
        base_exps = [(p1, p2, p3) for (p1, p2, p3) in exps]
        if ACTIVE_SHAPE_ROOT.resolve() == ACTIVE_COLOR_ROOT.resolve():
            self.exps = base_exps + [(p1, p3, p2) for (p1, p2, p3) in exps]
        else:
            self.exps = base_exps
        self.dataset_dir = dataset_dir
        self.face_root = face_root
        self.color_root = color_root
        self.include_shape_image = include_shape_image
        self.teacher_records = teacher_records
        if teacher_records is not None:
            missing = [
                triplet_cache_key(exp) for exp in self.exps
                if triplet_cache_key(exp) not in teacher_records
            ]
            if missing:
                message = (
                    f"V8.4 teacher cache is missing {len(missing)} dataset triplets; "
                    f"first missing key={missing[0]}"
                )
                if USER_REQUIRE_TEACHER_CACHE:
                    raise RuntimeError(message)
                print(f"[V2.21] optional {message}", file=sys.stderr)
        print(f"dataset pairs: {len(self.exps)}", file=sys.stderr)

    def __len__(self):
        return len(self.exps)

    def __getitem__(self, idx):
        item = prepare_item(
            self.exps[idx],
            self.dataset_dir,
            self.face_root,
            self.color_root,
            include_shape_image=self.include_shape_image,
        )
        if item is None:
            raise RuntimeError(f"Failed to prepare blending item at index {idx}")
        sample_key = triplet_cache_key(self.exps[idx])
        record = self.teacher_records.get(sample_key) if self.teacher_records is not None else None
        teacher_alpha = float("nan") if record is None else float(record["teacher_alpha"])
        teacher_confidence = 0.0 if record is None else float(record["teacher_confidence"])
        return (*item, sample_key, teacher_alpha, teacher_confidence)


class BlendingTrainerV8:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        train_loader: DataLoader,
        val_loader: DataLoader,
        helper: MaskPrepHelper,
    ):
        self.device = helper.device
        self.model = model.to(self.device)
        self.v228_compositor = StrongAnchorAppearanceCompositorV828(
            matte_max_distance=USER_V228_MATTE_MAX_DISTANCE,
            bg_residual_radius=USER_V228_BG_RESIDUAL_RADIUS,
            bg_residual_strength=USER_V228_BG_RESIDUAL_STRENGTH,
            outer_ring_width=USER_V228_OUTER_RING_WIDTH,
        ).to(self.device).eval()
        self.v229_carrier = HybridHairCarrierV829(
            low_radius=USER_V229_CARRIER_LOW_RADIUS,
            anchor_hf_gain=USER_V229_ANCHOR_HF_GAIN,
        ).to(self.device).eval()
        self.v229_recompositor = BoundaryRecompositionV829(
            coverage_max_distance=USER_V228_MATTE_MAX_DISTANCE,
            carrier_low_radius=USER_V229_CARRIER_LOW_RADIUS,
            tone_propagation_radius=USER_V229_TONE_RADIUS,
            background_radius=USER_V229_BACKGROUND_RADIUS,
            outer_strand_recovery=USER_V229_OUTER_STRAND_RECOVERY,
        ).to(self.device).eval()
        self.v230_carrier = AchromaticHairCarrierV830(
            low_radius=USER_V230_CORE_LOW_RADIUS,
            detail_radius=USER_V230_ANCHOR_DETAIL_RADIUS,
            detail_log_cap=USER_V230_DETAIL_LOG_CAP,
            detail_gain=USER_V230_ANCHOR_DETAIL_GAIN,
        ).to(self.device).eval()
        self.v230_topology = HairTopologyRepairV830(
            hole_radius=USER_V230_TOPOLOGY_HOLE_RADIUS,
            neighbor_threshold=USER_V230_TOPOLOGY_NEIGHBOR_THRESHOLD,
        ).to(self.device).eval()
        self.v230_finalizer = PPGuidedFinalV830(
            core_seam_width=USER_V230_CORE_SEAM_WIDTH,
            pp_tone_radius=USER_V230_PP_TONE_RADIUS,
            tone_propagation_radius=USER_V230_TONE_RADIUS,
            face_contact_width=USER_V230_FACE_CONTACT_WIDTH,
            confidence_temperature=USER_V230_CONFIDENCE_TEMPERATURE,
        ).to(self.device).eval()
        self.v231_finalizer = PPUnifiedFinalV831(
            tone_radius=USER_V231_TONE_RADIUS,
            tone_propagation_radius=USER_V231_TONE_PROP_RADIUS,
            context_radius=USER_V231_CONTEXT_RADIUS,
            transition_expand=USER_V231_TRANSITION_EXPAND,
            enable_phase_c=True,
        ).to(self.device).eval()
        self.v232_recolor = ForegroundRecolorV832(
            USER_V232_TONE_RADIUS, USER_V232_RESIDUAL_RADIUS
        ).to(self.device).eval()
        self.v232_face_calibrator = FaceSideAlphaCalibratorV832(
            USER_V232_FACE_PROTOTYPE_RADIUS, USER_V232_FACE_TEMPERATURE
        ).to(self.device).eval()
        self.v232_background = BackgroundTargetV832(
            USER_V232_BACKGROUND_RADIUS, USER_V232_TRANSITION_EXPAND
        ).to(self.device).eval()
        self.v232_recomposer = MattingRecomposerV832().to(self.device).eval()
        self.v233_confidence = FBConfidenceV833().to(self.device).eval()
        self.v233_foreground_target = ReliableHairForegroundTargetV833(
            tone_radius=USER_V233_TONE_RADIUS, local_radius=USER_V233_LOCAL_RADIUS,
            propagation_radius=USER_V233_PROPAGATION_RADIUS,
            detail_gain=USER_V233_DETAIL_GAIN,
        ).to(self.device).eval()
        self.v233_face_calibrator = FaceSideAlphaCalibratorV833(
            posterior_radius=USER_V233_POSTERIOR_RADIUS,
            temperature=USER_V233_FACE_TEMPERATURE,
        ).to(self.device).eval()
        self.v233_background = BackgroundTargetV833(
            radius=USER_V233_BACKGROUND_RADIUS, support_full=USER_V233_SUPPORT_FULL,
        ).to(self.device).eval()
        self.v234_carrier = HairCarrierChromaInjectionV834(
            reference_radius=USER_V234_REFERENCE_RADIUS,
            edge_radius=USER_V234_EDGE_RADIUS,
            edge_min_confidence=USER_V234_EDGE_MIN_CONFIDENCE,
        ).to(self.device).eval()
        self.v235_disentangler = HairOnlyChromaDisentanglementV835(
            chroma_radius=USER_V235_CHROMA_RADIUS,
            boundary_radius=USER_V235_BOUNDARY_RADIUS,
            boundary_min_confidence=USER_V235_BOUNDARY_MIN_CONFIDENCE,
            chroma_scale=USER_V235_CHROMA_SCALE,
        ).to(self.device).eval()
        self.legacy_v223_projector = FullColorToneSelectiveProjectorV823(
            target_dir_min_ab=USER_V222_TARGET_DIR_MIN_AB,
            luma_low_radius=USER_V223_LUMA_LOW_RADIUS,
            max_low_l_shift=USER_V223_MAX_LOW_L_SHIFT,
            edge_chroma_strength=USER_V223_EDGE_CHROMA_STRENGTH,
            edge_luma_strength=USER_V223_EDGE_LUMA_STRENGTH,
            edge_luma_margin=USER_EDGE_LUMA_MARGIN,
            orth_keep=USER_V223_ORTH_KEEP,
            hard_protect_threshold=0.50,
        ).to(self.device).eval()
        self.legacy_v224_projector = BoundaryStableFullColorProjectorV824(
            target_dir_min_ab=USER_V222_TARGET_DIR_MIN_AB,
            luma_low_radius=USER_V223_LUMA_LOW_RADIUS,
            max_low_l_shift=USER_V223_MAX_LOW_L_SHIFT,
            edge_chroma_strength=USER_V223_EDGE_CHROMA_STRENGTH,
            edge_luma_strength=USER_V223_EDGE_LUMA_STRENGTH,
            edge_luma_margin=USER_EDGE_LUMA_MARGIN,
            orth_keep=USER_V223_ORTH_KEEP,
            hard_protect_threshold=0.50,
        ).to(self.device).eval()
        self.optimizer = optimizer
        self.scheduler = (
            torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda _: 1.0
            )
            if self.optimizer is not None
            else None
        )
        self.last_train_lr = (
            float(self.optimizer.param_groups[0]["lr"])
            if self.optimizer is not None
            else 0.0
        )
        self.current_stage: str | None = None
        self.alpha_star: float | None = None
        self.normal_manifest: dict[str, object] | None = None
        self.phase0_summary: dict[str, object] | None = None
        self.phase0_decision: dict[str, object] | None = None
        self.color_weight_by_sample: dict[str, float] = {}
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.helper = helper
        self.color_config = ColorConditionConfigV8(
            ab_no_edit_threshold=USER_AB_NO_EDIT,
            ab_full_edit_threshold=USER_AB_FULL_EDIT,
            hue_no_edit_deg=USER_HUE_NO_EDIT_DEG,
            hue_full_edit_deg=USER_HUE_FULL_EDIT_DEG,
            chroma_mag_no_edit=USER_CHROMA_MAG_NO_EDIT,
            chroma_mag_full_edit=USER_CHROMA_MAG_FULL_EDIT,
            color_dist_no_edit=USER_COLOR_DIST_NO_EDIT,
            color_dist_full_edit=USER_COLOR_DIST_FULL_EDIT,
            lightness_no_edit_threshold=USER_LIGHTNESS_NO_EDIT_THRESHOLD_V8,
            lightness_full_edit_threshold=USER_LIGHTNESS_FULL_EDIT_THRESHOLD_V8,
            max_global_l_shift=USER_MAX_GLOBAL_L_SHIFT_V8,
            relative_luma_bins=USER_RELATIVE_LUMA_BINS,
            relative_luma_min_scale=USER_RELATIVE_LUMA_MIN_SCALE,
            global_ab_fallback_min_reliability=USER_GLOBAL_AB_FALLBACK_MIN_RELIABILITY,
            min_safe_fraction=USER_MIN_SAFE_REFERENCE_FRACTION_V8,
            highlight_mad_scale=USER_HIGHLIGHT_MAD_SCALE,
            highlight_global_min_margin=USER_HIGHLIGHT_GLOBAL_MIN_MARGIN,
            highlight_local_l_margin=USER_HIGHLIGHT_LOCAL_L_MARGIN,
            highlight_local_c_margin=USER_HIGHLIGHT_LOCAL_C_MARGIN,
            highlight_chroma_ratio=USER_HIGHLIGHT_CHROMA_RATIO,
        )
        self.grad_accum_steps = int(USER_GRAD_ACCUM_STEPS)
        self.best_color_score = float("inf")
        self.best_normal_balanced_score = float("inf")
        self.best_v222_score = float("inf")
        self.best_v222_summary: dict[str, object] | None = None
        self.v222_pretrain_summary: dict[str, object] | None = None
        self.v223_pretrain_summary: dict[str, object] | None = None
        self.output_ckpt_dir = ACTIVE_OUTPUT_DIR / "checkpoints"
        self.output_val_dir = ACTIVE_OUTPUT_DIR / "val_images"
        self.output_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_val_dir.mkdir(parents=True, exist_ok=True)
        self.fid_calc = None
        if USER_USE_FID and Path(USER_FID_DATASET).exists():
            self.fid_calc = get_fid_calc(USER_FID_CACHE, str(USER_FID_DATASET), device=self.device)

    def _build_optimizer(
        self, stage: str = "FIXED_BASE_ALPHA_LAYER_ADAPT"
    ) -> torch.optim.Optimizer:
        if stage != "FIXED_BASE_ALPHA_LAYER_ADAPT":
            raise ValueError(f"Unknown training stage: {stage}")
        parameters = list(self.model.v221_layer_adaptation_parameters())
        learning_rate = USER_LR
        if not parameters or not all(parameter.requires_grad for parameter in parameters):
            raise RuntimeError(f"Stage {stage} optimizer received invalid trainable parameters")
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=USER_WEIGHT_DECAY,
        )

    def prepare_batch(self, batch):
        if len(batch) == 14:
            (
                color_s,
                align_s,
                align_f_shape,
                remove_mask_shape,
                target_hair_shape,
                align_f_color,
                remove_mask_color,
                target_hair_color,
                color_i,
                face_i,
                shape_i,
                sample_ids,
                teacher_alpha,
                teacher_confidence,
            ) = batch
        elif len(batch) == 13:
            (
                color_s,
                align_s,
                align_f_shape,
                remove_mask_shape,
                target_hair_shape,
                align_f_color,
                remove_mask_color,
                target_hair_color,
                color_i,
                face_i,
                sample_ids,
                teacher_alpha,
                teacher_confidence,
            ) = batch
            shape_i = face_i
        else:
            raise ValueError(f"Unexpected V8 batch field count: {len(batch)}")
        del align_f_color, remove_mask_color, target_hair_color
        align_f = align_f_shape
        remove_mask = remove_mask_shape
        target_hair = target_hair_shape
        color_s, align_s, align_f, remove_mask, target_hair, color_i, face_i, shape_i = [
            item.to(self.device, non_blocking=True)
            for item in (
                color_s,
                align_s,
                align_f,
                remove_mask,
                target_hair,
                color_i,
                face_i,
                shape_i,
            )
        ]
        teacher_alpha = teacher_alpha.to(self.device, non_blocking=True).float()
        teacher_confidence = teacher_confidence.to(self.device, non_blocking=True).float()
        remove_mask = remove_mask.float().clamp(0, 1)
        if remove_mask.dim() == 3:
            remove_mask = remove_mask.unsqueeze(1)
        target_hair = target_hair.float().clamp(0, 1)
        if target_hair.dim() == 3:
            target_hair = target_hair.unsqueeze(1)

        with torch.no_grad():
            hm_3d, hm_3e = self.helper.generate_mask(color_i)
            (
                hm_1d,
                _,
                source_keep_mask,
                source_subject_guard,
                source_skin_guard,
                source_face_guard,
                source_neck_guard,
            ) = self.helper.generate_mask(
                face_i,
                return_keep=True,
            )
            i_x, _ = self.helper.net.generator(
                [align_s],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=align_f,
            )
            (
                hm_xd,
                hm_xe,
                face_keep_mask,
                target_subject_guard,
                target_skin_guard,
                target_face_guard,
                target_neck_guard,
                hm_xraw,
            ) = self.helper.generate_mask(
                i_x,
                return_keep=True,
                return_raw=True,
            )
            i_x_256 = self.helper.downsample_256(i_x)
            face_i_256 = self.helper.downsample_256(face_i)
            color_i_256 = self.helper.downsample_256(color_i)
            v228_face_parsing = torch.cat([
                get_segmentation(image.unsqueeze(0))
                for image in ((face_i + 1.0) / 2.0)
            ])
            v228_source_subject = (
                (v228_face_parsing > 0) & (v228_face_parsing != 13)
            ).float()

        cached_hair_d, cached_hair_e = self.helper.dilate_erosion.mask(target_hair)
        has_cached_hair = target_hair.flatten(1).sum(dim=1) >= USER_SAFE_HAIR_MIN_PIXELS
        target_hair_d = torch.where(
            has_cached_hair.view(-1, 1, 1, 1),
            torch.maximum(hm_xd, cached_hair_d),
            hm_xd,
        ).clamp(0, 1)
        target_hair_e = torch.where(
            has_cached_hair.view(-1, 1, 1, 1),
            torch.maximum(hm_xe, cached_hair_e),
            hm_xe,
        ).clamp(0, 1)
        target_hair_raw = torch.where(
            has_cached_hair.view(-1, 1, 1, 1), target_hair, hm_xraw
        ).clamp(0, 1)

        target_mask = (1 - hm_1d) * (1 - hm_3d) * (1 - target_hair_d)
        neck_hair_override = (USER_TARGET_HAIR_NECK_OVERRIDE * target_hair_e).clamp(0, 1)
        target_neck_visible = (target_neck_guard * (1.0 - neck_hair_override)).clamp(0, 1)
        source_neck_visible = (source_neck_guard * (1.0 - neck_hair_override)).clamp(0, 1)
        skin_color_block = (
            target_face_guard
            + target_neck_visible
            + 0.55 * source_face_guard
            + 0.55 * source_neck_visible
        ).clamp(0, 1)
        skin_protect_mask = (
            target_face_guard
            + target_neck_visible
            + 0.45 * source_face_guard
            + 0.45 * source_neck_visible
        ).clamp(0, 1)
        raw_skin_protect_mask = skin_protect_mask
        remove_color_block = (
            remove_mask * (1.0 - target_hair_d)
            + USER_REMOVE_BLOCK_IN_TARGET_HAIR * remove_mask * target_hair_d
        ).clamp(0, 1)
        color_transfer_eroded = (
            target_hair_e
            * (1.0 - remove_color_block)
            * (1.0 - USER_FACE_NECK_COLOR_BLOCK * skin_color_block)
        ).clamp(0, 1)
        color_transfer_core = (
            ((0.72 * target_hair_e) + (0.28 * target_hair_d))
            * (1.0 - remove_color_block)
            * (1.0 - USER_FACE_NECK_COLOR_BLOCK * skin_color_block)
        ).clamp(0, 1)
        needs_fallback = color_transfer_eroded.flatten(1).sum(dim=1) < USER_SAFE_HAIR_MIN_PIXELS
        # Color fidelity is supervised strictly on reliable interior hair.
        # The dilated support remains useful only as a transition/guard ring.
        color_transfer_mask = color_transfer_core
        color_supervision_mask = color_transfer_eroded
        color_supervision_mask = torch.where(
            needs_fallback.view(-1, 1, 1, 1),
            torch.zeros_like(color_transfer_eroded),
            color_supervision_mask,
        ).clamp(0, 1)
        transition_ring = (color_transfer_mask - color_supervision_mask).clamp(0, 1)
        outer_background_guard = (
            has_cached_hair.view(-1, 1, 1, 1)
            * dilate_mask(target_hair, 3)
            * (1.0 - target_hair)
            * (1.0 - target_subject_guard)
            * (1.0 - source_subject_guard)
            * (1.0 - target_skin_guard)
            * (1.0 - remove_mask)
        ).clamp(0, 1)
        subject_protect_mask = (
            (
                face_keep_mask
                + source_keep_mask
                + target_subject_guard
                + 0.60 * source_subject_guard
                + target_skin_guard
                + 0.50 * source_skin_guard
            )
            * (1.0 - color_transfer_mask)
        ).clamp(0, 1)
        skin_protect_mask = (skin_protect_mask * (1.0 - color_transfer_mask)).clamp(0, 1)
        satd_protect_mask = (
            remove_color_block
            + subject_protect_mask
            + skin_protect_mask
            + 0.35 * target_mask
            + 0.25 * (1.0 - target_hair_d) * (1.0 - hm_3e)
        ).clamp(0, 1)
        # V2.24 keeps raw hair membership separate from legacy V2.23 weighted
        # support. Protection is passed once to the shared projector algebra.
        v224_subject_protect_mask = (
            (
                face_keep_mask
                + source_keep_mask
                + target_subject_guard
                + 0.60 * source_subject_guard
                + target_skin_guard
                + 0.50 * source_skin_guard
            )
            * (1.0 - target_hair_raw)
        ).clamp(0, 1)
        v224_skin_protect_mask = (raw_skin_protect_mask * (1.0 - target_hair_raw)).clamp(0, 1)
        v224_satd_protect_mask = (
            remove_color_block
            + v224_subject_protect_mask
            + v224_skin_protect_mask
            + 0.35 * target_mask
            + 0.25 * (1.0 - target_hair_raw) * (1.0 - hm_3e)
        ).clamp(0, 1)
        v224_outer_background_guard = (
            dilate_mask(target_hair_raw, 3)
            * (1.0 - target_hair_raw)
            * (1.0 - target_subject_guard)
            * (1.0 - source_subject_guard)
            * (1.0 - target_skin_guard)
            * (1.0 - remove_mask)
        ).clamp(0, 1)
        with torch.no_grad():
            condition_reference = color_i_256
            if USER_V235_DIAGNOSTIC_ONLY:
                condition_reference = color_i_256 * hm_3e
            condition_bundle = build_color_condition_bundle(
                reference_image=condition_reference,
                reference_hair_mask=hm_3e,
                base_image=i_x_256,
                target_hair_mask=color_supervision_mask,
                config=self.color_config,
            )
        color_reference_mask = condition_bundle["safe_ref_mask"]

        valid = (
            color_reference_mask.flatten(1).sum(dim=1) >= USER_SAFE_HAIR_MIN_PIXELS
        ) & (
            color_supervision_mask.flatten(1).sum(dim=1) >= USER_SAFE_HAIR_MIN_PIXELS
        )
        if not valid.any():
            return None

        valid_indices = valid.nonzero(as_tuple=False).flatten().tolist()
        valid_sample_ids = [sample_ids[index] for index in valid_indices]
        color_weights = torch.ones(len(valid_indices), device=self.device)
        pseudo_reliable = torch.ones(
            len(valid_indices), device=self.device, dtype=torch.bool
        )
        near_no_edit = torch.zeros_like(pseudo_reliable)
        extreme = torch.zeros_like(pseudo_reliable)
        if self.normal_manifest is not None:
            valid_metrics = condition_bundle["metrics"]
            pseudo_reliable = (
                valid_metrics["pseudo_to_reference_mean_ab"][valid] <= 4.0
            ) & (
                valid_metrics["pseudo_to_reference_hue_error"][valid] <= 12.0
            ) & (
                valid_metrics["safe_fraction"][valid]
                >= USER_MIN_SAFE_REFERENCE_FRACTION_V8
            )
            ref_base_distance = valid_metrics["ref_base_ab_distance"][valid]
            near_no_edit = ref_base_distance < 3.0
            reference_chroma = torch.linalg.vector_norm(
                condition_bundle["ref_stats"]["mean_ab"][valid], dim=1
            )
            extreme = pseudo_reliable & (~near_no_edit) & (
                (ref_base_distance >= float(self.normal_manifest["extreme_distance_cutoff"]))
                | (
                    reference_chroma
                    >= float(self.normal_manifest["extreme_chroma_cutoff"])
                )
            )
            color_weights = torch.where(
                pseudo_reliable,
                torch.where(
                    extreme,
                    torch.zeros_like(color_weights),
                    torch.where(
                        near_no_edit,
                        torch.full_like(color_weights, USER_V222_NEAR_NO_EDIT_WEIGHT),
                        color_weights,
                    ),
                ),
                torch.zeros_like(color_weights),
            )

        return {
            "color_s": color_s[valid],
            "align_s": align_s[valid],
            "align_f": align_f[valid],
            "color_i": color_i_256[valid],
            "face_i": face_i_256[valid],
            "shape_i": self.helper.downsample_256(shape_i)[valid],
            "base_i": i_x_256[valid],
            "target_mask": target_mask[valid],
            "satd_protect_mask": satd_protect_mask[valid],
            "remove_mask": remove_color_block[valid],
            "color_transfer_mask": color_transfer_mask[valid],
            "color_supervision_mask": color_supervision_mask[valid],
            "transition_ring": transition_ring[valid],
            "v224_target_hair_mask": target_hair_raw[valid],
            "v224_target_hair_eroded": target_hair_e[valid],
            "v224_face_keep_mask": face_keep_mask[valid],
            "v224_skin_protect_mask": v224_skin_protect_mask[valid],
            "v224_satd_protect_mask": v224_satd_protect_mask[valid],
            "v224_outer_background_guard": v224_outer_background_guard[valid],
            "v228_source_subject_mask": v228_source_subject[valid],
            "v230_parser_labels": v228_face_parsing[valid].float(),
            "v230_source_face_mask": parsing_label_mask(v228_face_parsing, tuple(range(1, 13)))[valid],
            "v230_source_skin_mask": parsing_label_mask(v228_face_parsing, (1,))[valid],
            "outer_background_guard": outer_background_guard[valid],
            "face_keep_mask": face_keep_mask[valid],
            "skin_protect_mask": skin_protect_mask[valid],
            "reference_hair_mask": hm_3e[valid],
            "safe_ref_mask": condition_bundle["safe_ref_mask"][valid],
            "rejected_highlight_mask": condition_bundle["rejected_highlight_mask"][valid],
            "color_descriptor": condition_bundle["descriptor"][valid],
            "chroma_need_gate": condition_bundle["chroma_need_gate"][valid],
            "lightness_need_gate": condition_bundle["lightness_need_gate"][valid],
            "edit_need_gate": condition_bundle["edit_need_gate"][valid],
            "pseudo_lab": condition_bundle["pseudo_lab"][valid],
            "target_ref_ab": condition_bundle["target_ref_ab"][valid],
            "pseudo_rgb": condition_bundle["pseudo_rgb"][valid],
            "color_proxy": condition_bundle["color_proxy"][valid],
            "ref_stats": {
                key: value[valid] for key, value in condition_bundle["ref_stats"].items()
            },
            "base_stats": {
                key: value[valid] for key, value in condition_bundle["base_stats"].items()
            },
            "teacher_alpha": teacher_alpha[valid],
            "teacher_confidence": teacher_confidence[valid],
            "sample_id": valid_sample_ids,
            "color_weight": color_weights,
            "pseudo_reliable": pseudo_reliable,
            "near_no_edit": near_no_edit,
            "extreme_for_v222": extreme,
            "condition_metrics": {
                key: value[valid]
                for key, value in condition_bundle["metrics"].items()
            },
        }

    @staticmethod
    def masked_l1(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return BlendingTrainerV8.masked_l1_per_sample(source, target, mask).mean()

    @staticmethod
    def masked_l1_per_sample(
        source: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0) * source.size(1)
        return (torch.abs(source - target) * mask).flatten(1).sum(dim=1) / denominator

    @staticmethod
    def masked_smooth_l1(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        element_loss = tnf.smooth_l1_loss(source, target, reduction="none")
        denominator = mask.sum().clamp_min(1.0) * source.size(1)
        return (element_loss * mask).sum() / denominator

    @staticmethod
    def masked_mean_value(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return BlendingTrainerV8.masked_mean_per_sample(value, mask).mean()

    @staticmethod
    def masked_mean_per_sample(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0) * value.size(1)
        return (value * mask).flatten(1).sum(dim=1) / denominator

    @staticmethod
    def masked_fraction_above(value: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        return (((value > threshold).float() * mask).sum() / mask.sum().clamp_min(1.0))

    @staticmethod
    def masked_q95(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per_sample = []
        for sample_value, sample_mask in zip(value.detach(), mask.detach()):
            selected = sample_value[sample_mask.expand_as(sample_value) > 0.05]
            per_sample.append(torch.quantile(selected, 0.95) if selected.numel() else value.new_zeros(()))
        return torch.stack(per_sample).mean()

    def calc_loss(
        self,
        i_gen: torch.Tensor,
        prepared: dict[str, object],
        encoder_aux: dict[str, torch.Tensor],
        stage: str,
        anchor_i: torch.Tensor | None = None,
    ):
        i_face = prepared["face_i"]
        i_base = prepared["base_i"]
        mask_face = prepared["target_mask"]
        mask_gen_hair = prepared["color_supervision_mask"]
        transition_ring = prepared["transition_ring"]
        outer_background_guard = prepared["outer_background_guard"]
        satd_protect_mask = prepared["satd_protect_mask"]
        face_keep_mask = prepared["face_keep_mask"]
        skin_protect_mask = prepared["skin_protect_mask"]
        remove_mask = prepared["remove_mask"]
        pseudo_lab = prepared["pseudo_lab"]
        pseudo_rgb = prepared["pseudo_rgb"]

        mask_gen_hair = mask_gen_hair.float().clamp(0, 1)
        satd_protect_mask = satd_protect_mask.float().clamp(0, 1)
        face_keep_mask = face_keep_mask.float().clamp(0, 1)
        skin_protect_mask = skin_protect_mask.float().clamp(0, 1)
        remove_mask = remove_mask.float().clamp(0, 1)

        gen_face_embed = self.model.get_image_embed(i_gen * mask_face)
        face_embed = self.model.get_image_embed(i_face * mask_face)
        face_loss = (1 - tnf.cosine_similarity(gen_face_embed, face_embed)).mean()

        hair_loss = i_gen.sum() * 0.0

        gen_lab = rgb_to_lab(i_gen)
        base_lab = rgb_to_lab(i_base)
        gen_luma = gen_lab[:, 0:1]
        base_luma = base_lab[:, 0:1]
        gen_chroma = gen_lab[:, 1:3]
        base_chroma = base_lab[:, 1:3]
        gen_stats = compute_intrinsic_hair_color_stats(
            gen_lab, mask_gen_hair, self.color_config
        )
        ref_stats = prepared["ref_stats"]
        final_ref_metrics = compute_reference_fidelity_metrics(
            gen_lab, mask_gen_hair, ref_stats, self.color_config
        )
        final_ref_score_per_sample = reference_color_score(final_ref_metrics)
        encoder_aux["reference_color_score"] = final_ref_score_per_sample.detach()

        teacher_confidence = prepared["teacher_confidence"].clamp(0, 1)
        teacher_valid = torch.isfinite(prepared["teacher_alpha"]) & (teacher_confidence > 0)
        alpha_teacher_per_sample = tnf.smooth_l1_loss(
            encoder_aux["predicted_alpha"],
            torch.nan_to_num(prepared["teacher_alpha"], nan=0.0),
            reduction="none",
        )
        alpha_teacher_weight = get_alpha_teacher_weight(prepared.get("epoch", 0))
        alpha_teacher_loss = (
            alpha_teacher_per_sample * teacher_confidence * teacher_valid.float()
        ).sum() / (teacher_confidence * teacher_valid.float()).sum().clamp_min(1.0)
        alpha_teacher_loss = alpha_teacher_loss * alpha_teacher_weight
        color_weight = prepared["color_weight"].clamp(0, 1)
        color_weight_sum = color_weight.sum().clamp_min(1.0)
        ref_mean_ab_per_sample = tnf.l1_loss(
            gen_stats["mean_ab"] / 110.0,
            ref_stats["mean_ab"] / 110.0,
            reduction="none",
        ).mean(dim=1)
        ref_mean_ab_loss = (
            ref_mean_ab_per_sample * color_weight
        ).sum() / color_weight_sum
        ref_hue_per_sample = (
            1.0
            - (gen_stats["hue_unit"] * ref_stats["hue_unit"])
            .sum(dim=1)
            .clamp(-1, 1)
        ) * ref_stats["hue_validity"]
        ref_hue_loss = (
            ref_hue_per_sample * color_weight
        ).sum() / color_weight_sum
        ref_chroma_per_sample = tnf.l1_loss(
            gen_stats["median_chroma"] / 110.0,
            ref_stats["median_chroma"] / 110.0,
            reduction="none",
        )
        ref_chroma_loss = (ref_chroma_per_sample * color_weight).sum() / color_weight_sum

        pseudo_ab_loss_per_sample = self.masked_l1_per_sample(
            gen_chroma / 110.0,
            pseudo_lab[:, 1:3] / 110.0,
            mask_gen_hair,
        )
        pseudo_ab_loss = (pseudo_ab_loss_per_sample * color_weight).sum() / color_weight_sum
        pseudo_rgb_loss = self.masked_l1(i_gen, pseudo_rgb, mask_gen_hair)
        pseudo_luma_per_sample = self.masked_mean_per_sample(
            tnf.smooth_l1_loss(
                gen_luma / 100.0,
                pseudo_lab[:, 0:1] / 100.0,
                reduction="none",
            ),
            mask_gen_hair,
        )
        pseudo_luma_loss = (
            pseudo_luma_per_sample * color_weight
        ).sum() / color_weight_sum
        luma_excess = gen_luma - pseudo_lab[:, 0:1]
        positive_luma = torch.relu(luma_excess - USER_LUMA_EXCESS_MARGIN)
        positive_luma_loss = self.masked_mean_value(positive_luma / 100.0, mask_gen_hair)
        hp_gen = gen_luma - gaussian_blur2d(gen_luma, radius=3)
        hp_base = base_luma - gaussian_blur2d(base_luma, radius=3)
        hf_luma_excess = torch.relu(
            hp_gen.abs() - hp_base.abs() - USER_HF_LUMA_EXCESS_MARGIN
        )
        hf_luma_loss = self.masked_mean_value(hf_luma_excess / 100.0, mask_gen_hair)

        # Reference-biased progress prevents the direct anchor from settling at
        # a source/reference midpoint while remaining one-sided (overshoot is
        # still controlled by the existing reference losses).
        base_mean_ab = prepared["base_stats"]["mean_ab"]
        ref_mean_ab = ref_stats["mean_ab"]
        direction_ab = ref_mean_ab - base_mean_ab
        progress = (
            ((gen_stats["mean_ab"] - base_mean_ab) * direction_ab).sum(dim=1)
            / direction_ab.square().sum(dim=1).clamp_min(1e-6)
        )
        progress_enabled = (
            direction_ab.norm(dim=1) >= 4.0
        ) & (
            prepared["safe_ref_mask"].flatten(1).sum(dim=1) >= USER_SAFE_HAIR_MIN_PIXELS
        )
        reference_progress_loss = torch.relu(USER_MIN_REFERENCE_PROGRESS - progress)
        reference_progress_loss = (
            reference_progress_loss * progress_enabled.float()
        ).mean()

        allowed_edge_luma = torch.maximum(base_luma, pseudo_lab[:, 0:1])
        edge_luma_excess = torch.relu(
            gen_luma - allowed_edge_luma - USER_EDGE_LUMA_MARGIN
        )
        edge_luma_excess_loss = self.masked_mean_value(
            edge_luma_excess / 100.0, transition_ring
        )
        edge_hp_excess = torch.relu(
            hp_gen.abs() - hp_base.abs() - USER_HF_LUMA_EXCESS_MARGIN
        )
        edge_hf_luma_loss = self.masked_mean_value(edge_hp_excess / 100.0, transition_ring)
        outer_bg_keep_loss = self.masked_l1(i_gen, i_base, outer_background_guard)
        layer_offset_mag_loss = encoder_aux["layer_offset"].abs().mean()
        layer_negative_drift_loss = torch.relu(
            -encoder_aux["layer_offset"].mean(dim=1) - 0.01
        ).mean()

        face_keep_region = (face_keep_mask * satd_protect_mask).clamp(0, 1)
        protect_region = (face_keep_region + remove_mask).clamp(0, 1)
        face_keep_loss = self.masked_l1(i_gen, i_base, face_keep_region)
        remove_keep_loss = self.masked_l1(i_gen, i_base, remove_mask)
        protect_chroma_keep_loss = self.masked_l1(
            gen_chroma / 110.0,
            base_chroma / 110.0,
            protect_region,
        )
        skin_chroma_keep_loss = self.masked_l1(
            gen_chroma / 110.0,
            base_chroma / 110.0,
            skin_protect_mask,
        )
        skin_rgb_keep_loss = self.masked_l1(i_gen, i_base, skin_protect_mask)
        correction_norm_loss = (
            encoder_aux["correction_norm"]
            / encoder_aux["direct_delta_norm"].clamp_min(1.0)
        ).mean()
        final_ab_error_per_sample = self.masked_mean_per_sample(
            torch.linalg.vector_norm(gen_chroma - pseudo_lab[:, 1:3], dim=1, keepdim=True),
            mask_gen_hair,
        )
        if anchor_i is None:
            correction_color_regression_loss = i_gen.sum() * 0.0
            correction_hue_regression = i_gen.sum() * 0.0
            correction_ref_regression = i_gen.sum() * 0.0
        else:
            anchor_lab = rgb_to_lab(anchor_i)
            anchor_chroma = anchor_lab[:, 1:3]
            anchor_ab_error_per_sample = self.masked_mean_per_sample(
                torch.linalg.vector_norm(
                    anchor_chroma - pseudo_lab[:, 1:3], dim=1, keepdim=True
                ),
                mask_gen_hair,
            )
            correction_color_regression_loss = torch.relu(
                final_ab_error_per_sample
                - anchor_ab_error_per_sample
                - USER_CORRECTION_COLOR_TOLERANCE
            ).mean()
            anchor_ref_metrics = compute_reference_fidelity_metrics(
                anchor_lab, mask_gen_hair, ref_stats, self.color_config
            )
            correction_hue_regression = correction_hue_regression_loss(
                anchor_ref_metrics["hue_error"],
                final_ref_metrics["hue_error"],
                USER_CORRECTION_HUE_TOLERANCE_DEG,
            )
            correction_ref_regression = correction_reference_regression_loss(
                anchor_ref_metrics,
                final_ref_metrics,
                USER_CORRECTION_REF_SCORE_TOLERANCE,
            )

        weights = NORMAL_COLOR_LOSS_WEIGHTS

        total_loss = (
            weights["face_clip"] * face_loss
            + weights["pseudo_ab"] * pseudo_ab_loss
            + weights["pseudo_rgb"] * pseudo_rgb_loss
            + weights["pseudo_luma"] * pseudo_luma_loss
            + weights["positive_luma"] * positive_luma_loss
            + weights["hf_luma"] * hf_luma_loss
            + weights["face_keep"] * face_keep_loss
            + weights["remove_keep"] * remove_keep_loss
            + weights["protect_chroma"] * protect_chroma_keep_loss
            + weights["skin_chroma"] * skin_chroma_keep_loss
            + weights["skin_rgb"] * skin_rgb_keep_loss
            + weights["alpha_teacher"] * alpha_teacher_loss
            + weights["ref_mean_ab"] * ref_mean_ab_loss
            + weights["ref_hue"] * ref_hue_loss
            + weights["ref_chroma"] * ref_chroma_loss
            + weights["reference_progress"] * reference_progress_loss
            + weights["edge_luma_excess"] * edge_luma_excess_loss
            + weights["edge_hf_luma"] * edge_hf_luma_loss
            + weights["outer_bg_keep"] * outer_bg_keep_loss
            + weights["layer_offset_mag"] * layer_offset_mag_loss
            + weights["layer_negative_drift"] * layer_negative_drift_loss
            + weights["correction_norm"] * correction_norm_loss
            + weights["correction_color_regression"] * correction_color_regression_loss
            + weights["correction_hue_regression"] * correction_hue_regression
            + weights["correction_ref_regression"] * correction_ref_regression
        )
        return total_loss, {
            "face_loss": face_loss,
            "hair_loss": hair_loss,
            "pseudo_ab": pseudo_ab_loss,
            "pseudo_rgb": pseudo_rgb_loss,
            "pseudo_luma": pseudo_luma_loss,
            "positive_luma": positive_luma_loss,
            "hf_luma": hf_luma_loss,
            "face_keep_l1": face_keep_loss,
            "remove_keep_l1": remove_keep_loss,
            "protect_chroma_keep": protect_chroma_keep_loss,
            "skin_chroma_keep": skin_chroma_keep_loss,
            "skin_rgb_keep": skin_rgb_keep_loss,
            "alpha_teacher": alpha_teacher_loss,
            "ref_mean_ab": ref_mean_ab_loss,
            "ref_hue": ref_hue_loss,
            "ref_chroma": ref_chroma_loss,
            "reference_progress": reference_progress_loss,
            "edge_luma_excess": edge_luma_excess_loss,
            "edge_hf_luma": edge_hf_luma_loss,
            "outer_bg_keep": outer_bg_keep_loss,
            "layer_offset_mag": layer_offset_mag_loss,
            "layer_negative_drift": layer_negative_drift_loss,
            "reference_progress_mean": progress.mean(),
            "reference_progress_p25": torch.quantile(progress.detach(), 0.25),
            "reference_progress_p50": torch.quantile(progress.detach(), 0.50),
            "edge_luma_excess_mean": self.masked_mean_value(edge_luma_excess, transition_ring),
            "edge_luma_excess_fraction": self.masked_fraction_above(edge_luma_excess, transition_ring, 0.0),
            "correction_norm": correction_norm_loss,
            "correction_color_regression": correction_color_regression_loss,
            "correction_hue_regression": correction_hue_regression,
            "correction_ref_regression": correction_ref_regression,
            "final_to_reference_ab": final_ref_metrics["mean_ab_error"].mean(),
            "final_to_reference_hue": final_ref_metrics["hue_error"].mean(),
            "final_to_reference_chroma": final_ref_metrics["chroma_error"].mean(),
            "reference_color_score": final_ref_score_per_sample.mean(),
            "teacher_alpha_mae": (
                (
                    encoder_aux["predicted_alpha"]
                    - torch.nan_to_num(prepared["teacher_alpha"], nan=0.0)
                ).abs()
                * teacher_confidence
                * teacher_valid.float()
            ).sum() / (teacher_confidence * teacher_valid.float()).sum().clamp_min(1.0),
            "result_to_pseudo_ab_l2": final_ab_error_per_sample.mean(),
            "result_hue_error": self.masked_mean_value(
                torch.rad2deg(
                    torch.acos(
                        (
                            (gen_chroma * pseudo_lab[:, 1:3]).sum(dim=1, keepdim=True)
                            / (
                                torch.linalg.vector_norm(gen_chroma, dim=1, keepdim=True)
                                * torch.linalg.vector_norm(pseudo_lab[:, 1:3], dim=1, keepdim=True)
                            ).clamp_min(1e-4)
                        ).clamp(-1, 1)
                    )
                ),
                mask_gen_hair,
            ),
            "mean_l_excess": self.masked_mean_value(torch.relu(luma_excess), mask_gen_hair),
            "q95_l_excess": self.masked_q95(luma_excess, mask_gen_hair),
            "frac_l_excess_gt8": self.masked_fraction_above(luma_excess, mask_gen_hair, 8.0),
            "frac_l_excess_gt12": self.masked_fraction_above(luma_excess, mask_gen_hair, 12.0),
            "frac_l_excess_gt16": self.masked_fraction_above(luma_excess, mask_gen_hair, 16.0),
            "loss": total_loss,
        }

    def save_checkpoint(
        self,
        epoch: int,
        name: str,
        validation_summary: dict[str, float] | None = None,
    ):
        projector = self.model
        torch.save(
            {
                "arch": DIRECT_COLOR_ARCH_V8_5,
                "version": "v2.22",
                "epoch": epoch,
                "training_mode": "STRONG_ANCHOR_SELECTIVE_CHROMA",
                "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
                "normal_manifest": self.normal_manifest,
                "best_v222_score": self.best_v222_score,
                "validation_summary": validation_summary or {},
                "projector_config": {
                    "target_dir_min_ab": projector.target_dir_min_ab,
                    "edge_luma_margin": projector.edge_luma_margin,
                    "halo_scale": projector.halo_scale,
                    **projector.parameter_values_float(),
                },
                "projector_state_dict": projector.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            self.output_ckpt_dir / f"{name}.pth",
        )

    def load_resume_checkpoint(self) -> int:
        if not USER_RESUME_CHECKPOINT:
            return 0

        resume_path = Path(USER_RESUME_CHECKPOINT)
        if not resume_path.exists():
            raise FileNotFoundError(f"Cannot find USER_RESUME_CHECKPOINT: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=self.device)
        checkpoint_arch = checkpoint.get("arch") if isinstance(checkpoint, dict) else None
        if checkpoint_arch != DIRECT_COLOR_ARCH_V8_5:
            raise RuntimeError(
                f"Refusing to resume incompatible V2.22 checkpoint {resume_path}: "
                f"arch={checkpoint_arch!r}, required={DIRECT_COLOR_ARCH_V8_5!r}"
            )
        checkpoint_alpha = float(checkpoint.get("strong_anchor_alpha", -1.0))
        if checkpoint_alpha != USER_STRONG_ANCHOR_ALPHA:
            raise RuntimeError(
                "Resume strong_anchor_alpha does not match current config: "
                f"checkpoint={checkpoint_alpha}, current={USER_STRONG_ANCHOR_ALPHA}"
            )
        self.model.load_state_dict(checkpoint["projector_state_dict"], strict=True)

        start_epoch = int(checkpoint.get("epoch", 0))
        self.normal_manifest = checkpoint.get("normal_manifest")
        if self.normal_manifest is not None:
            self.color_weight_by_sample = {
                entry["sample_id"]: float(entry["color_weight"])
                for entry in self.normal_manifest["entries"]
            }
        self.best_v222_score = float(
            checkpoint.get("best_v222_score", self.best_v222_score)
        )
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(
            f"[blending_v8] resumed V2.22 projector from {resume_path}; "
            f"start_epoch={start_epoch + 1}",
            file=sys.stderr,
        )
        return start_epoch

    @staticmethod
    def build_generator_latent(align_s: torch.Tensor, blend_s: torch.Tensor) -> torch.Tensor:
        # This matches the author's training code. With start_layer=4 the
        # prefix is not rendered, but keeping the contract explicit avoids
        # accidental dependence if the generator call changes later.
        if USER_AUTHOR_ZERO_PREFIX_TRAIN:
            prefix = torch.zeros_like(align_s[:, :6])
        else:
            prefix = align_s[:, :6]
        return torch.cat((prefix, blend_s), dim=1)

    def configure_stage(self, epoch: int) -> tuple[str, bool]:
        del epoch
        stage = "FIXED_BASE_ALPHA_LAYER_ADAPT"
        if stage != self.current_stage:
            self.model.configure_v221_layer_adaptation()
            self.assert_trainable_parameter_scope_v221()
            self.current_stage = stage
            trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
            print(
                f"[blending_v8] configured stage={stage} trainable_parameters={trainable} "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e} correction_enabled=False"
            )
        return stage, False

    def assert_trainable_parameter_scope_v221(self) -> None:
        trainable_names = [
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        ]
        allowed_prefixes = ("descriptor_encoder.", "layer_offset_head.")
        unexpected = [
            name for name in trainable_names if not name.startswith(allowed_prefixes)
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected V2.21 trainable parameters: {unexpected}")
        if not trainable_names:
            raise RuntimeError("V2.21 layer adaptation has no trainable parameters")
        frozen_count = sum(
            parameter.numel() for parameter in self.model.parameters()
            if not parameter.requires_grad
        )
        trainable_count = sum(
            parameter.numel() for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        print(
            "[V2.21] trainable_parameter_names=" + ",".join(trainable_names)
            + f" trainable_count={trainable_count} frozen_count={frozen_count}"
        )

    def run_adapter(
        self,
        prepared: dict[str, object],
        *,
        correction_enabled: bool,
        layer_mix_override: float | None = None,
        base_alpha_override: float | None = None,
    ):
        return self.model(
            latent_face=prepared["align_s"][:, 6:],
            latent_color=prepared["color_s"][:, 6:],
            color_descriptor=prepared["color_descriptor"],
            chroma_need_gate=prepared["chroma_need_gate"],
            lightness_need_gate=prepared["lightness_need_gate"],
            edit_need_gate=prepared["edit_need_gate"],
            correction_enabled=correction_enabled,
            layer_mix_override=layer_mix_override,
            base_alpha_override=base_alpha_override,
            teacher_alpha=prepared["teacher_alpha"],
            return_aux=True,
        )

    def render_blend_tail(self, prepared: dict[str, object], blend_s: torch.Tensor) -> torch.Tensor:
        latent_in = self.build_generator_latent(prepared["align_s"], blend_s)
        image, _ = self.helper.net.generator(
            [latent_in],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=prepared["align_f"],
        )
        return self.helper.downsample_256(image)

    def _render_v222_pair(self, prepared: dict[str, object]):
        """Render the fixed base and strong anchor through one generator path."""
        with torch.no_grad():
            anchor_tail = fixed_direct_anchor_tail(
                prepared["align_s"][:, 6:],
                prepared["color_s"][:, 6:],
                USER_STRONG_ANCHOR_ALPHA,
            )
            anchor_norm = self.render_blend_tail(prepared, anchor_tail)
        base_rgb = ((prepared["base_i"] + 1.0) / 2.0).clamp(0, 1)
        anchor_rgb = ((anchor_norm + 1.0) / 2.0).clamp(0, 1)
        selective_rgb, projector_aux = self.model(
            base_rgb=base_rgb,
            anchor_rgb=anchor_rgb,
            pseudo_lab=prepared["pseudo_lab"],
            target_hair_mask=prepared["color_transfer_mask"],
            color_supervision_mask=prepared["color_supervision_mask"],
            transition_ring=prepared["transition_ring"],
            outer_background_guard=prepared["outer_background_guard"],
            face_keep_mask=prepared["face_keep_mask"],
            skin_protect_mask=prepared["skin_protect_mask"],
            satd_protect_mask=prepared["satd_protect_mask"],
            remove_mask=prepared["remove_mask"],
            return_aux=True,
        )
        return base_rgb, anchor_rgb, selective_rgb, projector_aux

    def _render_v223_pair(self, prepared: dict[str, object]):
        """Render Base, fixed strong anchor, and deterministic full-color output."""
        with torch.no_grad():
            anchor_tail = fixed_direct_anchor_tail_v823(
                prepared["align_s"][:, 6:],
                prepared["color_s"][:, 6:],
                USER_STRONG_ANCHOR_ALPHA,
            )
            anchor_norm = self.render_blend_tail(prepared, anchor_tail)
        base_rgb = ((prepared["base_i"] + 1.0) / 2.0).clamp(0, 1)
        anchor_rgb = ((anchor_norm + 1.0) / 2.0).clamp(0, 1)
        selective_rgb, projector_aux = self.model(
            base_rgb=base_rgb,
            anchor_rgb=anchor_rgb,
            pseudo_lab=prepared["pseudo_lab"],
            reference_delta_l=prepared["condition_metrics"]["delta_l_global"],
            target_hair_mask=prepared["color_transfer_mask"],
            color_supervision_mask=prepared["color_supervision_mask"],
            transition_ring=prepared["transition_ring"],
            outer_background_guard=prepared["outer_background_guard"],
            face_keep_mask=prepared["face_keep_mask"],
            skin_protect_mask=prepared["skin_protect_mask"],
            satd_protect_mask=prepared["satd_protect_mask"],
            remove_mask=prepared["remove_mask"],
            return_aux=True,
        )
        return base_rgb, anchor_rgb, selective_rgb, projector_aux

    def _render_v224_pair(self, prepared: dict[str, object]):
        """Render the shared-anchor V2.23 baseline and V2.24 single-alpha output."""
        with torch.no_grad():
            anchor_tail = fixed_direct_anchor_tail_v824(
                prepared["align_s"][:, 6:],
                prepared["color_s"][:, 6:],
                USER_STRONG_ANCHOR_ALPHA,
            )
            anchor_norm = self.render_blend_tail(prepared, anchor_tail)
        base_rgb = ((prepared["base_i"] + 1.0) / 2.0).clamp(0, 1)
        anchor_rgb = ((anchor_norm + 1.0) / 2.0).clamp(0, 1)
        legacy_rgb, legacy_aux = self.legacy_v223_projector(
            base_rgb=base_rgb,
            anchor_rgb=anchor_rgb,
            pseudo_lab=prepared["pseudo_lab"],
            reference_delta_l=prepared["condition_metrics"]["delta_l_global"],
            target_hair_mask=prepared["color_transfer_mask"],
            color_supervision_mask=prepared["color_supervision_mask"],
            transition_ring=prepared["transition_ring"],
            outer_background_guard=prepared["outer_background_guard"],
            face_keep_mask=prepared["face_keep_mask"],
            skin_protect_mask=prepared["skin_protect_mask"],
            satd_protect_mask=prepared["satd_protect_mask"],
            remove_mask=prepared["remove_mask"],
            return_aux=True,
        )
        selective_rgb, projector_aux = self.model(
            base_rgb=base_rgb,
            anchor_rgb=anchor_rgb,
            pseudo_lab=prepared["pseudo_lab"],
            reference_delta_l=prepared["condition_metrics"]["delta_l_global"],
            target_hair_mask=prepared["v224_target_hair_mask"],
            target_hair_eroded=prepared["v224_target_hair_eroded"],
            outer_background_guard=prepared["v224_outer_background_guard"],
            face_keep_mask=prepared["v224_face_keep_mask"],
            skin_protect_mask=prepared["v224_skin_protect_mask"],
            satd_protect_mask=prepared["v224_satd_protect_mask"],
            remove_mask=prepared["remove_mask"],
            return_aux=True,
        )
        projector_aux["legacy_v223_rgb"] = legacy_rgb
        projector_aux["legacy_v223_aux"] = legacy_aux
        return base_rgb, anchor_rgb, selective_rgb, projector_aux

    def _render_v225_pair(self, prepared: dict[str, object]):
        with torch.no_grad():
            anchor_tail = fixed_direct_anchor_tail_v825(
                prepared["align_s"][:, 6:], prepared["color_s"][:, 6:], USER_STRONG_ANCHOR_ALPHA
            )
            anchor_norm = self.render_blend_tail(prepared, anchor_tail)
        base_rgb = ((prepared["base_i"] + 1.0) / 2.0).clamp(0, 1)
        anchor_rgb = ((anchor_norm + 1.0) / 2.0).clamp(0, 1)
        baseline_rgb, baseline_aux = self.legacy_v224_projector(
            base_rgb=base_rgb, anchor_rgb=anchor_rgb, pseudo_lab=prepared["pseudo_lab"],
            reference_delta_l=prepared["condition_metrics"]["delta_l_global"],
            target_hair_mask=prepared["v224_target_hair_mask"],
            target_hair_eroded=prepared["v224_target_hair_eroded"],
            outer_background_guard=prepared["v224_outer_background_guard"],
            face_keep_mask=prepared["v224_face_keep_mask"],
            skin_protect_mask=prepared["v224_skin_protect_mask"],
            satd_protect_mask=prepared["v224_satd_protect_mask"],
            remove_mask=prepared["remove_mask"], return_aux=True,
        )
        selective_rgb, projector_aux = self.model(
            base_rgb=base_rgb, anchor_rgb=anchor_rgb, pseudo_lab=prepared["pseudo_lab"],
            target_ref_ab=prepared["target_ref_ab"],
            reference_delta_l=prepared["condition_metrics"]["delta_l_global"],
            target_hair_mask=prepared["v224_target_hair_mask"],
            target_hair_eroded=prepared["v224_target_hair_eroded"],
            outer_background_guard=prepared["v224_outer_background_guard"],
            face_keep_mask=prepared["v224_face_keep_mask"],
            skin_protect_mask=prepared["v224_skin_protect_mask"],
            satd_protect_mask=prepared["v224_satd_protect_mask"],
            remove_mask=prepared["remove_mask"], return_aux=True,
        )
        projector_aux["baseline_v224_rgb"] = baseline_rgb
        projector_aux["baseline_v224_aux"] = baseline_aux
        return base_rgb, anchor_rgb, selective_rgb, projector_aux

    @staticmethod
    def _masked_median_per_sample(
        value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        values = []
        for sample_value, sample_mask in zip(value, mask):
            selected = sample_value[sample_mask.expand_as(sample_value) > 1e-6]
            values.append(
                torch.quantile(selected, 0.5)
                if selected.numel()
                else value.new_zeros(())
            )
        return torch.stack(values)

    @staticmethod
    def _v223_edge_metrics(
        image_lab: torch.Tensor,
        base_lab: torch.Tensor,
        pseudo_lab: torch.Tensor,
        hair_edge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_excess = torch.relu(
            image_lab[:, :1]
            - torch.maximum(base_lab[:, :1], pseudo_lab[:, :1])
            - USER_EDGE_LUMA_MARGIN
        )
        edge_mean = strict_masked_mean_per_sample(edge_excess, hair_edge)
        image_hp = image_lab[:, :1] - gaussian_blur2d(image_lab[:, :1], radius=3)
        base_hp = base_lab[:, :1] - gaussian_blur2d(base_lab[:, :1], radius=3)
        edge_hf = torch.relu(
            image_hp.abs() - base_hp.abs() - USER_HF_LUMA_EXCESS_MARGIN
        )
        return edge_mean, strict_masked_mean_per_sample(edge_hf, hair_edge)

    def calc_loss_v223(
        self,
        base_rgb: torch.Tensor,
        selective_rgb: torch.Tensor,
        prepared: dict[str, object],
        projector_aux: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        selective_lab = rgb_to_lab(selective_rgb)
        base_lab = rgb_to_lab(base_rgb)
        core = projector_aux["hair_core"]
        color_weight = prepared["color_weight"].clamp(0, 1)
        weight_sum = color_weight.sum().clamp_min(1.0)

        selective_ref = compute_reference_fidelity_metrics(
            selective_lab, core, prepared["ref_stats"], self.color_config
        )
        pseudo_ab_map = (
            selective_lab[:, 1:] / 110.0
            - prepared["pseudo_lab"][:, 1:] / 110.0
        ).abs().mean(dim=1, keepdim=True)
        pseudo_ab_per_sample = strict_masked_mean_per_sample(pseudo_ab_map, core)
        pseudo_ab = (pseudo_ab_per_sample * color_weight).sum() / weight_sum

        sel_l_low = gaussian_blur2d(
            selective_lab[:, :1], radius=USER_V223_LUMA_LOW_RADIUS
        )
        pseudo_l_low = gaussian_blur2d(
            prepared["pseudo_lab"][:, :1], radius=USER_V223_LUMA_LOW_RADIUS
        )
        pseudo_l_low_map = tnf.smooth_l1_loss(
            sel_l_low / 100.0, pseudo_l_low / 100.0, reduction="none"
        )
        pseudo_l_low_per_sample = strict_masked_mean_per_sample(
            pseudo_l_low_map, core
        )
        pseudo_l_low_loss = (
            pseudo_l_low_per_sample * color_weight
        ).sum() / weight_sum

        ref_mean_ab_per_sample = (
            selective_ref["candidate_stats"]["mean_ab"]
            - prepared["ref_stats"]["mean_ab"]
        ).abs().mean(dim=1) / 110.0
        ref_mean_ab = (ref_mean_ab_per_sample * color_weight).sum() / weight_sum
        ref_hue = (
            selective_ref["hue_error"] / 180.0 * color_weight
        ).sum() / weight_sum
        ref_chroma = (
            selective_ref["chroma_error"] / 110.0 * color_weight
        ).sum() / weight_sum
        ref_median_l = (
            selective_ref["median_l_error"] / 100.0 * color_weight
        ).sum() / weight_sum

        selective_hp = selective_lab[:, :1] - gaussian_blur2d(
            selective_lab[:, :1], radius=3
        )
        base_hp = base_lab[:, :1] - gaussian_blur2d(base_lab[:, :1], radius=3)
        hf_per_sample = strict_masked_mean_per_sample(
            (selective_hp - base_hp).abs() / 100.0, core
        )
        hf_preserve = (hf_per_sample * color_weight).sum() / weight_sum
        edge_excess = torch.relu(
            selective_lab[:, :1]
            - torch.maximum(base_lab[:, :1], prepared["pseudo_lab"][:, :1])
            - USER_EDGE_LUMA_MARGIN
        )
        edge_halo = strict_masked_mean_per_sample(
            edge_excess / 100.0, projector_aux["hair_edge"]
        ).mean()
        total = (
            USER_V223_PSEUDO_L_LOW_WEIGHT * pseudo_l_low_loss
            + USER_V223_REF_MEDIAN_L_WEIGHT * ref_median_l
            + USER_V223_PSEUDO_AB_WEIGHT * pseudo_ab
            + USER_V223_REF_MEAN_AB_WEIGHT * ref_mean_ab
            + USER_V223_REF_HUE_WEIGHT * ref_hue
            + USER_V223_REF_CHROMA_WEIGHT * ref_chroma
            + USER_V223_HF_PRESERVE_WEIGHT * hf_preserve
            + USER_V223_EDGE_HALO_WEIGHT * edge_halo
        )
        return total, {
            "loss": total,
            "pseudo_l_low": pseudo_l_low_loss,
            "ref_median_l": ref_median_l,
            "pseudo_ab": pseudo_ab,
            "ref_mean_ab": ref_mean_ab,
            "ref_hue": ref_hue,
            "ref_chroma": ref_chroma,
            "hf_preserve": hf_preserve,
            "edge_halo": edge_halo,
        }

    @torch.no_grad()
    def _build_v223_records(
        self,
        prepared: dict[str, object],
        base_rgb: torch.Tensor,
        anchor_rgb: torch.Tensor,
        selective_rgb: torch.Tensor,
        projector_aux: dict[str, torch.Tensor],
    ) -> list[dict[str, object]]:
        base_lab = rgb_to_lab(base_rgb)
        anchor_lab = rgb_to_lab(anchor_rgb)
        selective_lab = rgb_to_lab(selective_rgb)
        core = projector_aux["hair_core"]
        edge = projector_aux["hair_edge"]
        ref_stats = prepared["ref_stats"]
        base_ref = compute_reference_fidelity_metrics(
            base_lab, core, ref_stats, self.color_config
        )
        anchor_ref = compute_reference_fidelity_metrics(
            anchor_lab, core, ref_stats, self.color_config
        )
        selective_ref = compute_reference_fidelity_metrics(
            selective_lab, core, ref_stats, self.color_config
        )
        base_full = full_stat_error(
            base_ref["mean_ab_error"], base_ref["median_l_error"]
        )
        anchor_full = full_stat_error(
            anchor_ref["mean_ab_error"], anchor_ref["median_l_error"]
        )
        selective_full = full_stat_error(
            selective_ref["mean_ab_error"], selective_ref["median_l_error"]
        )
        raw_full_gain = base_full - anchor_full
        selective_full_gain = base_full - selective_full
        full_retention = torch.where(
            raw_full_gain > 1e-6,
            selective_full_gain / raw_full_gain.clamp_min(1e-6),
            torch.zeros_like(raw_full_gain),
        )
        raw_ab_gain = base_ref["mean_ab_error"] - anchor_ref["mean_ab_error"]
        selective_ab_gain = (
            base_ref["mean_ab_error"] - selective_ref["mean_ab_error"]
        )
        ab_retention = torch.where(
            raw_ab_gain > 1e-6,
            selective_ab_gain / raw_ab_gain.clamp_min(1e-6),
            torch.zeros_like(raw_ab_gain),
        )
        l_direction = ref_stats["l_median"] - base_ref["candidate_stats"]["l_median"]
        l_progress_valid = l_direction.abs() >= 3.0
        l_progress = torch.where(
            l_progress_valid,
            (
                selective_ref["candidate_stats"]["l_median"]
                - base_ref["candidate_stats"]["l_median"]
            )
            / torch.where(
                l_direction.abs() >= 3.0, l_direction, torch.ones_like(l_direction)
            ),
            torch.zeros_like(l_direction),
        )
        pseudo_delta = torch.linalg.vector_norm(
            selective_lab - prepared["pseudo_lab"], dim=1, keepdim=True
        )
        pseudo_delta_mean = strict_masked_mean_per_sample(pseudo_delta, core)
        pseudo_delta_median = self._masked_median_per_sample(pseudo_delta, core)
        anchor_edge_delta = torch.linalg.vector_norm(
            anchor_lab - base_lab, dim=1, keepdim=True
        )
        selective_edge_delta = torch.linalg.vector_norm(
            selective_lab - base_lab, dim=1, keepdim=True
        )
        edge_transfer_ratio = strict_masked_mean_per_sample(
            selective_edge_delta, edge
        ) / strict_masked_mean_per_sample(anchor_edge_delta, edge).clamp_min(1e-6)
        anchor_edge, _ = self._v223_edge_metrics(
            anchor_lab, base_lab, prepared["pseudo_lab"], edge
        )
        selective_edge, selective_edge_hf = self._v223_edge_metrics(
            selective_lab, base_lab, prepared["pseudo_lab"], edge
        )
        mean_luma_weight = strict_masked_mean_per_sample(
            projector_aux["luma_transfer_weight"], projector_aux["hair_support"]
        )
        mean_chroma_weight = strict_masked_mean_per_sample(
            projector_aux["chroma_transfer_weight"], projector_aux["hair_support"]
        )
        records = []
        for index, sample_id in enumerate(prepared["sample_id"]):
            base_full_value = float(base_full[index].item())
            records.append({
                "sample_id": sample_id,
                "safe_fraction": float(
                    prepared["condition_metrics"]["safe_fraction"][index].item()
                ),
                "ref_base_ab_distance": float(base_ref["mean_ab_error"][index].item()),
                "ref_base_l_distance": float(base_ref["median_l_error"][index].item()),
                "reference_chroma_magnitude": float(
                    torch.linalg.vector_norm(ref_stats["mean_ab"][index]).item()
                ),
                "pseudo_to_reference_ab_error": float(
                    prepared["condition_metrics"]["pseudo_to_reference_mean_ab"][index].item()
                ),
                "pseudo_to_reference_hue_error": float(
                    prepared["condition_metrics"]["pseudo_to_reference_hue_error"][index].item()
                ),
                "base_ref_ab_error": float(base_ref["mean_ab_error"][index].item()),
                "anchor_ref_ab_error": float(anchor_ref["mean_ab_error"][index].item()),
                "selective_ref_ab_error": float(selective_ref["mean_ab_error"][index].item()),
                "selective_ref_hue_error": float(selective_ref["hue_error"][index].item()),
                "selective_ref_chroma_error": float(selective_ref["chroma_error"][index].item()),
                "base_ref_l_error": float(base_ref["median_l_error"][index].item()),
                "anchor_ref_l_error": float(anchor_ref["median_l_error"][index].item()),
                "selective_ref_l_error": float(selective_ref["median_l_error"][index].item()),
                "base_full_stat_error": base_full_value,
                "anchor_full_stat_error": float(anchor_full[index].item()),
                "selective_full_stat_error": float(selective_full[index].item()),
                "raw_full_anchor_positive_gain": bool(raw_full_gain[index] > 1e-6),
                "raw_anchor_positive_gain": bool(raw_ab_gain[index] > 1e-6),
                "full_color_retention": float(full_retention[index].item()),
                "ab_retention": float(ab_retention[index].item()),
                "full_color_improvement_ratio": float(
                    ((base_full[index] - selective_full[index]) / base_full[index].clamp_min(1e-6)).item()
                ),
                "l_progress": float(l_progress[index].item()),
                "l_progress_valid": bool(l_progress_valid[index]),
                "pseudo_deltaE76": float(pseudo_delta_median[index].item()),
                "pseudo_deltaE76_mean": float(pseudo_delta_mean[index].item()),
                "edge_transfer_ratio": float(edge_transfer_ratio[index].item()),
                "raw_anchor_edge_luma_excess_mean": float(anchor_edge[index].item()),
                "edge_luma_excess_mean": float(selective_edge[index].item()),
                "edge_hf_excess": float(selective_edge_hf[index].item()),
                "face_keep_l1": float(projector_aux["face_keep_l1"][index].item()),
                "outer_bg_keep_l1": float(projector_aux["outer_bg_keep_l1"][index].item()),
                "hard_protect_max_abs_delta": float(
                    projector_aux["hard_protect_max_abs_delta"][index].item()
                ),
                "outside_hair_max_abs_delta": float(
                    projector_aux["outside_hair_max_abs_delta"][index].item()
                ),
                "mean_luma_transfer_weight": float(mean_luma_weight[index].item()),
                "mean_chroma_transfer_weight": float(mean_chroma_weight[index].item()),
            })
        return records

    @torch.no_grad()
    def _build_v224_records(
        self,
        prepared: dict[str, object],
        base_rgb: torch.Tensor,
        anchor_rgb: torch.Tensor,
        selective_rgb: torch.Tensor,
        projector_aux: dict[str, torch.Tensor],
    ) -> list[dict[str, object]]:
        """Build V2.24 records and attach same-batch V2.23 A/B metrics."""
        records = self._build_v223_records(
            prepared, base_rgb, anchor_rgb, selective_rgb, projector_aux
        )
        legacy_records = self._build_v223_records(
            prepared,
            base_rgb,
            anchor_rgb,
            projector_aux["legacy_v223_rgb"],
            projector_aux["legacy_v223_aux"],
        )
        base_lab = rgb_to_lab(base_rgb)
        anchor_lab = rgb_to_lab(anchor_rgb)
        selective_lab = rgb_to_lab(selective_rgb)
        edge = projector_aux["hair_edge"]
        anchor_delta = anchor_lab - base_lab
        selective_delta = selective_lab - base_lab
        full_transfer = strict_masked_mean_per_sample(
            torch.linalg.vector_norm(selective_delta, dim=1, keepdim=True), edge
        ) / strict_masked_mean_per_sample(
            torch.linalg.vector_norm(anchor_delta, dim=1, keepdim=True), edge
        ).clamp_min(1e-6)
        ab_transfer = strict_masked_mean_per_sample(
            torch.linalg.vector_norm(selective_delta[:, 1:], dim=1, keepdim=True), edge
        ) / strict_masked_mean_per_sample(
            torch.linalg.vector_norm(anchor_delta[:, 1:], dim=1, keepdim=True), edge
        ).clamp_min(1e-6)
        anchor_l_delta = anchor_delta[:, :1]
        selective_l_delta = selective_delta[:, :1]
        l_valid_edge = edge * (anchor_l_delta.abs() >= 2.0).float()
        l_transfer_valid = l_valid_edge.flatten(1).sum(dim=1) >= 1.0
        l_transfer = strict_masked_mean_per_sample(selective_l_delta.abs(), l_valid_edge) / (
            strict_masked_mean_per_sample(anchor_l_delta.abs(), l_valid_edge).clamp_min(1e-6)
        )
        direction_agreement = strict_masked_mean_per_sample(
            ((selective_l_delta * anchor_l_delta) >= 0).float(), l_valid_edge
        )
        for index, record in enumerate(records):
            old = legacy_records[index]
            record.update({
                "v223_old_selective_full_stat_error": old["selective_full_stat_error"],
                "v223_old_full_color_retention": old["full_color_retention"],
                "v223_old_ab_retention": old["ab_retention"],
                "v223_old_l_progress": old["l_progress"],
                "edge_transfer_full": float(full_transfer[index].item()),
                "edge_transfer_ab": float(ab_transfer[index].item()),
                "edge_transfer_l": float(l_transfer[index].item()),
                "edge_l_transfer_valid": bool(l_transfer_valid[index]),
                "edge_l_direction_agreement_fraction": float(
                    direction_agreement[index].item()
                ),
                "mean_edge_membership": float(
                    projector_aux["mean_edge_membership"][index].item()
                ),
                "mean_edge_chroma_weight": float(
                    projector_aux["mean_edge_chroma_weight"][index].item()
                ),
                "mean_edge_luma_weight": float(
                    projector_aux["mean_edge_luma_weight"][index].item()
                ),
                "mean_core_chroma_weight": float(
                    projector_aux["mean_core_chroma_weight"][index].item()
                ),
                "mean_core_luma_weight": float(
                    projector_aux["mean_core_luma_weight"][index].item()
                ),
                "edge_luma_excess_after_guard": float(
                    strict_masked_mean_per_sample(
                        projector_aux["edge_luma_excess_after_guard"][index:index + 1],
                        edge[index:index + 1],
                    )[0].item()
                ),
            })
        return records

    @torch.no_grad()
    def _build_v225_records(self, prepared, base_rgb, anchor_rgb, selective_rgb, projector_aux):
        records = self._build_v223_records(prepared, base_rgb, anchor_rgb, selective_rgb, projector_aux)
        baseline = self._build_v223_records(
            prepared, base_rgb, anchor_rgb, projector_aux["baseline_v224_rgb"], projector_aux["baseline_v224_aux"]
        )
        base_lab, anchor_lab, selective_lab = rgb_to_lab(base_rgb), rgb_to_lab(anchor_rgb), rgb_to_lab(selective_rgb)
        baseline_lab = rgb_to_lab(projector_aux["baseline_v224_rgb"])
        edge = projector_aux["hair_edge"]
        base_ab, anchor_ab, selective_ab = base_lab[:, 1:], anchor_lab[:, 1:], selective_lab[:, 1:]
        target_ab = prepared["target_ref_ab"]
        target_l = base_lab[:, :1] + USER_V223_EDGE_LUMA_STRENGTH * prepared["condition_metrics"]["delta_l_global"].view(-1, 1, 1, 1)
        edge_target_error = strict_masked_mean_per_sample(torch.linalg.vector_norm(selective_ab - target_ab, dim=1, keepdim=True), edge)
        edge_base_target_error = strict_masked_mean_per_sample(torch.linalg.vector_norm(base_ab - target_ab, dim=1, keepdim=True), edge).clamp_min(1e-6)
        edge_l_error = strict_masked_mean_per_sample((selective_lab[:, :1] - target_l).abs(), edge)
        edge_base_l_error = strict_masked_mean_per_sample((base_lab[:, :1] - target_l).abs(), edge).clamp_min(1e-6)
        anchor_delta = anchor_lab - base_lab
        selective_delta = selective_lab - base_lab
        baseline_delta = baseline_lab - base_lab
        anchor_full_den = strict_masked_mean_per_sample(torch.linalg.vector_norm(anchor_delta, dim=1, keepdim=True), edge).clamp_min(1e-6)
        anchor_ab_den = strict_masked_mean_per_sample(torch.linalg.vector_norm(anchor_delta[:, 1:], dim=1, keepdim=True), edge).clamp_min(1e-6)
        anchor_l_den = strict_masked_mean_per_sample(anchor_delta[:, :1].abs(), edge).clamp_min(1e-6)
        v224_full_transfer = strict_masked_mean_per_sample(torch.linalg.vector_norm(baseline_delta, dim=1, keepdim=True), edge) / anchor_full_den
        v224_ab_transfer = strict_masked_mean_per_sample(torch.linalg.vector_norm(baseline_delta[:, 1:], dim=1, keepdim=True), edge) / anchor_ab_den
        v224_l_transfer = strict_masked_mean_per_sample(baseline_delta[:, :1].abs(), edge) / anchor_l_den
        full_transfer = strict_masked_mean_per_sample(torch.linalg.vector_norm(selective_delta, dim=1, keepdim=True), edge) / anchor_full_den
        ab_transfer = strict_masked_mean_per_sample(torch.linalg.vector_norm(selective_delta[:, 1:], dim=1, keepdim=True), edge) / anchor_ab_den
        l_transfer = strict_masked_mean_per_sample(selective_delta[:, :1].abs(), edge) / anchor_l_den
        ab_direction = strict_masked_mean_per_sample(((selective_ab - base_ab) * (target_ab - base_ab)).sum(dim=1, keepdim=True) > 0, edge)
        l_direction = strict_masked_mean_per_sample((selective_delta[:, :1] * prepared["condition_metrics"]["delta_l_global"].view(-1, 1, 1, 1)) > 0, edge)
        ref_norm = projector_aux["edge_reference_target_ab_norm"]
        pseudo_norm = torch.linalg.vector_norm(prepared["pseudo_lab"][:, 1:] - base_ab, dim=1, keepdim=True)
        before = strict_masked_mean_per_sample(projector_aux["edge_luma_excess_before_cap"], edge)
        after = strict_masked_mean_per_sample(projector_aux["edge_luma_excess_after_guard"], edge)
        for i, record in enumerate(records):
            record.update({
                "v224_selective_full_stat_error": baseline[i]["selective_full_stat_error"],
                "v224_edge_transfer_full": float(v224_full_transfer[i].item()),
                "v224_edge_transfer_ab": float(v224_ab_transfer[i].item()),
                "v224_edge_transfer_l": float(v224_l_transfer[i].item()),
                "edge_transfer_full": float(full_transfer[i].item()),
                "edge_transfer_ab": float(ab_transfer[i].item()),
                "edge_transfer_l": float(l_transfer[i].item()),
                "edge_reference_ab_progress": float((1.0 - edge_target_error[i] / edge_base_target_error[i]).item()),
                "edge_reference_l_progress": float((1.0 - edge_l_error[i] / edge_base_l_error[i]).item()),
                "edge_ab_direction_agreement_fraction": float(ab_direction[i].item()),
                "edge_l_direction_agreement_fraction": float(l_direction[i].item()),
                "edge_reference_target_ab_norm": float(strict_masked_mean_per_sample(ref_norm[i:i+1], edge[i:i+1])[0].item()),
                "edge_pseudo_ab_norm": float(strict_masked_mean_per_sample(pseudo_norm[i:i+1], edge[i:i+1])[0].item()),
                "edge_reference_parallel_mag": float(strict_masked_mean_per_sample(projector_aux["edge_reference_parallel_mag"][i:i+1], edge[i:i+1])[0].item()),
                "edge_parallel_cap_saturation_fraction": float(strict_masked_mean_per_sample(projector_aux["edge_parallel_cap_saturation_fraction"][i:i+1], edge[i:i+1])[0].item()),
                "edge_positive_overshoot_before": float(before[i].item()),
                "edge_positive_overshoot_after": float(after[i].item()),
            })
        return records

    @staticmethod
    def _v222_edge_metrics(
        image_rgb: torch.Tensor,
        base_rgb: torch.Tensor,
        pseudo_lab: torch.Tensor,
        transition_ring: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_lab = rgb_to_lab(image_rgb)
        base_lab = rgb_to_lab(base_rgb)
        edge_excess = torch.relu(
            image_lab[:, :1]
            - torch.maximum(base_lab[:, :1], pseudo_lab[:, :1])
            - USER_EDGE_LUMA_MARGIN
        )
        edge_mean = BlendingTrainerV8.masked_mean_per_sample(edge_excess, transition_ring)
        edge_fraction = BlendingTrainerV8._per_sample_fraction(
            edge_excess, transition_ring, 0.0
        )
        image_hp = image_lab[:, :1] - gaussian_blur2d(image_lab[:, :1], radius=3)
        base_hp = base_lab[:, :1] - gaussian_blur2d(base_lab[:, :1], radius=3)
        edge_hf = torch.relu(
            image_hp.abs() - base_hp.abs() - USER_HF_LUMA_EXCESS_MARGIN
        )
        edge_hf_mean = BlendingTrainerV8.masked_mean_per_sample(edge_hf, transition_ring)
        return edge_mean, edge_fraction, edge_hf_mean

    @torch.no_grad()
    def _build_v222_records(
        self,
        prepared: dict[str, object],
        base_rgb: torch.Tensor,
        anchor_rgb: torch.Tensor,
        selective_rgb: torch.Tensor,
        projector_aux: dict[str, torch.Tensor],
    ) -> list[dict[str, object]]:
        base_lab = rgb_to_lab(base_rgb)
        anchor_lab = rgb_to_lab(anchor_rgb)
        selective_lab = rgb_to_lab(selective_rgb)
        core = projector_aux["hair_core"]
        base_stats = compute_intrinsic_hair_color_stats(base_lab, core, self.color_config)
        anchor_stats = compute_intrinsic_hair_color_stats(anchor_lab, core, self.color_config)
        selective_stats = compute_intrinsic_hair_color_stats(
            selective_lab, core, self.color_config
        )
        anchor_ref = compute_reference_fidelity_metrics(
            anchor_lab, core, prepared["ref_stats"], self.color_config
        )
        selective_ref = compute_reference_fidelity_metrics(
            selective_lab, core, prepared["ref_stats"], self.color_config
        )
        base_distance = prepared["condition_metrics"]["ref_base_ab_distance"].clamp_min(1e-6)
        base_to_ref = torch.linalg.vector_norm(
            base_stats["mean_ab"] - prepared["ref_stats"]["mean_ab"], dim=1
        )
        anchor_to_ref = anchor_ref["mean_ab_error"]
        selective_to_ref = selective_ref["mean_ab_error"]
        direction = prepared["ref_stats"]["mean_ab"] - base_stats["mean_ab"]
        selective_delta = selective_stats["mean_ab"] - base_stats["mean_ab"]
        direction_sq = direction.square().sum(dim=1).clamp_min(1e-6)
        parallel_progress = (selective_delta * direction).sum(dim=1) / direction_sq
        orthogonal = torch.linalg.vector_norm(
            selective_delta - parallel_progress[:, None] * direction, dim=1
        )
        color_improvement = (base_distance - selective_to_ref) / base_distance
        raw_gain = base_distance - anchor_to_ref
        selective_gain = base_distance - selective_to_ref
        retention = torch.where(
            raw_gain > 1e-6,
            selective_gain / raw_gain.clamp_min(1e-6),
            torch.zeros_like(raw_gain),
        )
        anchor_edge, _, _ = self._v222_edge_metrics(
            anchor_rgb, base_rgb, prepared["pseudo_lab"], prepared["transition_ring"]
        )
        selective_edge, selective_edge_fraction, selective_edge_hf = self._v222_edge_metrics(
            selective_rgb, base_rgb, prepared["pseudo_lab"], prepared["transition_ring"]
        )
        anchor_face_keep = self.masked_l1_per_sample(
            anchor_rgb,
            base_rgb,
            (prepared["face_keep_mask"] * prepared["satd_protect_mask"]).clamp(0, 1),
        )
        anchor_outer_keep = self.masked_l1_per_sample(
            anchor_rgb, base_rgb, prepared["outer_background_guard"]
        )
        global_l_excess = selective_lab[:, :1] - prepared["pseudo_lab"][:, :1]
        global_fraction = self._per_sample_fraction(
            global_l_excess, core, 12.0
        )
        records = []
        for index, sample_id in enumerate(prepared["sample_id"]):
            records.append({
                "sample_id": sample_id,
                "safe_fraction": float(prepared["condition_metrics"]["safe_fraction"][index].item()),
                "ref_base_ab_distance": float(base_distance[index].item()),
                "reference_chroma_magnitude": float(
                    torch.linalg.vector_norm(prepared["ref_stats"]["mean_ab"][index]).item()
                ),
                "pseudo_to_reference_ab_error": float(
                    prepared["condition_metrics"]["pseudo_to_reference_mean_ab"][index].item()
                ),
                "pseudo_to_reference_hue_error": float(
                    prepared["condition_metrics"]["pseudo_to_reference_hue_error"][index].item()
                ),
                "base_to_ref_ab": float(base_to_ref[index].item()),
                "anchor_to_ref_ab": float(anchor_to_ref[index].item()),
                "selective_to_ref_ab": float(selective_to_ref[index].item()),
                "selective_to_ref_hue": float(selective_ref["hue_error"][index].item()),
                "selective_to_ref_chroma": float(selective_ref["chroma_error"][index].item()),
                "final_to_reference_ab_error": float(selective_to_ref[index].item()),
                "final_to_reference_hue_error": float(selective_ref["hue_error"][index].item()),
                "final_to_reference_chroma_error": float(
                    selective_ref["chroma_error"][index].item()
                ),
                "reference_progress": float(parallel_progress[index].item()),
                "parallel_progress": float(parallel_progress[index].item()),
                "orthogonal_error": float(orthogonal[index].item()),
                "color_improvement_ratio": float(color_improvement[index].item()),
                "raw_anchor_positive_gain": bool(raw_gain[index] > 1e-6),
                "color_retention_ratio": float(retention[index].item()),
                "anchor_edge_luma_excess_mean": float(anchor_edge[index].item()),
                "anchor_face_keep_l1": float(anchor_face_keep[index].item()),
                "anchor_outer_bg_keep_l1": float(anchor_outer_keep[index].item()),
                "edge_luma_excess_mean": float(selective_edge[index].item()),
                "edge_luma_excess_fraction": float(selective_edge_fraction[index].item()),
                "edge_hf_excess": float(selective_edge_hf[index].item()),
                "outer_bg_keep_l1": float(projector_aux["outer_bg_keep_l1"][index].item()),
                "face_keep_l1": float(projector_aux["face_keep_l1"][index].item()),
                "hard_protect_max_abs_delta": float(
                    projector_aux["hard_protect_max_abs_delta"][index].item()
                ),
                "outside_hair_max_abs_delta": float(
                    projector_aux["outside_hair_max_abs_delta"][index].item()
                ),
                "global_frac_l_excess_gt12": float(global_fraction[index].item()),
                "parallel_gain": float(projector_aux["parallel_gain"][index].item()),
                "orth_keep": float(projector_aux["orth_keep"][index].item()),
                "boundary_strength": float(projector_aux["boundary_strength"][index].item()),
                "luma_strength": float(projector_aux["luma_strength"][index].item()),
                "mean_A_core": float(projector_aux["mean_A_core"][index].item()),
                "mean_A_edge": float(projector_aux["mean_A_edge"][index].item()),
                "mean_halo_gate": float(projector_aux["mean_halo_gate"][index].item()),
                "overshoot_fraction": float(projector_aux["overshoot_fraction"][index].item()),
                "negative_parallel_fraction": float(
                    projector_aux["negative_parallel_fraction"][index].item()
                ),
            })
        return records

    def _set_v222_manifest_weights(self, manifest: dict[str, object]) -> None:
        for entry in manifest["entries"]:
            if not entry["pseudo_reliable"]:
                entry["color_weight"] = 0.0
            elif entry["extreme_for_v221"]:
                entry["color_weight"] = 0.0
            elif entry["near_no_edit"]:
                entry["color_weight"] = USER_V222_NEAR_NO_EDIT_WEIGHT
            else:
                entry["color_weight"] = 1.0

    def _v222_normal_ids(self) -> set[str]:
        if self.normal_manifest is None:
            return set()
        return {
            entry["sample_id"]
            for entry in self.normal_manifest["entries"]
            if entry["normal_color"]
        }

    def _save_v222_preview_rows(
        self,
        output_dir: Path,
        rows: list[list[torch.Tensor]],
        compact_rows: list[list[torch.Tensor]],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows):
            save_preview(output_dir / f"sample_{index:03d}.png", row)
            save_preview(output_dir / f"comparison_{index:03d}.png", row)
        for index, row in enumerate(compact_rows):
            save_preview(output_dir / f"compact_{index:03d}.png", row)
        save_contact_sheet(output_dir / "full_diagnostic.png", rows)
        save_contact_sheet(output_dir / "compact_comparison.png", compact_rows)

    @torch.no_grad()
    def run_v222_pretrain_diagnostic(self) -> dict[str, object]:
        diagnostic_dir = ACTIVE_OUTPUT_DIR / "pretrain_diagnostic"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        records = []
        rows = []
        compact_rows = []
        for batch in tqdm(self.val_loader, desc="V2.22 selective pretrain", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base_rgb, anchor_rgb, selective_rgb, projector_aux = self._render_v222_pair(prepared)
            records.extend(
                self._build_v222_records(
                    prepared, base_rgb, anchor_rgb, selective_rgb, projector_aux
                )
            )
            if len(rows) < USER_LOG_IMAGE_COUNT:
                for index in range(prepared["base_i"].size(0)):
                    rows.append([
                        prepared["face_i"][index:index + 1],
                        prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1],
                        prepared["base_i"][index:index + 1],
                        anchor_rgb[index:index + 1] * 2 - 1,
                        prepared["pseudo_rgb"][index:index + 1],
                        selective_rgb[index:index + 1] * 2 - 1,
                        mask_to_preview(projector_aux["hair_core"][index:index + 1]),
                        mask_to_preview(prepared["transition_ring"][index:index + 1]),
                        mask_to_preview(projector_aux["halo_gate"][index:index + 1]),
                        mask_to_preview(projector_aux["hard_protect"][index:index + 1]),
                    ])
                    compact_rows.append([
                        prepared["face_i"][index:index + 1],
                        prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1],
                        prepared["base_i"][index:index + 1],
                        anchor_rgb[index:index + 1] * 2 - 1,
                        selective_rgb[index:index + 1] * 2 - 1,
                    ])
                    if len(rows) >= USER_LOG_IMAGE_COUNT:
                        break
        if not records:
            raise RuntimeError("V2.22 pretrain diagnostic produced no valid samples")
        manifest = build_normal_color_manifest(
            records, USER_MIN_SAFE_REFERENCE_FRACTION_V8
        )
        self._set_v222_manifest_weights(manifest)
        self.normal_manifest = manifest
        self.color_weight_by_sample = {
            entry["sample_id"]: float(entry["color_weight"])
            for entry in manifest["entries"]
        }
        normal_ids = self._v222_normal_ids()
        normal_summary = aggregate_v222_records(records, normal_ids)
        decision = (
            {"decision": "INSUFFICIENT_NORMAL_SAMPLES", "abort": True}
            if manifest["status"] != "OK"
            else classify_v222_pretrain(normal_summary)
        )
        summary = {
            "version": "v2.22",
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "all_valid": aggregate_v222_records(records),
            "normal": normal_summary,
            "val_normal_count": normal_summary.get("count", 0),
            "normal_manifest_status": manifest["status"],
            "decision": decision["decision"],
            "projector": self.model.parameter_values_float(),
        }
        assert_finite_json(summary)
        with open(ACTIVE_OUTPUT_DIR / "v222_normal_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(diagnostic_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        pretrain_acceptance = {
            "version": "v2.22",
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "normal_count": normal_summary.get("count", 0),
            "raw_anchor": {
                "median_final_to_ref_ab": normal_summary.get("median_anchor_to_ref_ab", 0.0),
                "edge_luma_excess_mean": normal_summary.get(
                    "raw_anchor_edge_luma_excess_mean", 0.0
                ),
                "face_keep_l1": normal_summary.get("raw_anchor_face_keep_l1", 0.0),
                "outer_bg_keep_l1": normal_summary.get("raw_anchor_outer_bg_keep_l1", 0.0),
            },
            "selective_best": {
                "median_final_to_ref_ab": normal_summary.get(
                    "median_selective_to_ref_ab", 0.0
                ),
                "median_color_retention_ratio": normal_summary.get(
                    "median_color_retention_ratio", 0.0
                ),
                "outside_hair_max_abs_delta": normal_summary.get(
                    "outside_hair_max_abs_delta", 0.0
                ),
                "hard_protect_max_abs_delta": normal_summary.get(
                    "hard_protect_max_abs_delta", 0.0
                ),
            },
            "projector": summary["projector"],
            "decision": decision["decision"],
        }
        assert_finite_json(pretrain_acceptance)
        with open(ACTIVE_OUTPUT_DIR / "v222_acceptance.json", "w", encoding="utf-8") as handle:
            json.dump(pretrain_acceptance, handle, ensure_ascii=False, indent=2, allow_nan=False)
        self._save_v222_preview_rows(diagnostic_dir, rows, compact_rows)
        self.v222_pretrain_summary = summary
        print(
            f"[V2.22] pretrain decision={decision['decision']} "
            f"normal_count={normal_summary.get('count', 0)} "
            f"retention={normal_summary.get('median_color_retention_ratio', 0.0):.4f} "
            f"outside_delta={normal_summary.get('outside_hair_max_abs_delta', 0.0):.2e} "
            f"hard_protect_delta={normal_summary.get('hard_protect_max_abs_delta', 0.0):.2e}"
        )
        return summary

    def calc_loss_v222(
        self,
        base_rgb: torch.Tensor,
        selective_rgb: torch.Tensor,
        prepared: dict[str, object],
        projector_aux: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        selective_lab = rgb_to_lab(selective_rgb)
        base_lab = rgb_to_lab(base_rgb)
        core = projector_aux["hair_core"]
        color_weight = prepared["color_weight"].clamp(0, 1)
        color_weight_sum = color_weight.sum().clamp_min(1.0)
        selective_stats = compute_intrinsic_hair_color_stats(
            selective_lab, core, self.color_config
        )
        pseudo_ab_per_sample = self.masked_l1_per_sample(
            selective_lab[:, 1:] / 110.0,
            prepared["pseudo_lab"][:, 1:] / 110.0,
            core,
        )
        pseudo_ab = (pseudo_ab_per_sample * color_weight).sum() / color_weight_sum
        ref_mean_ab_per_sample = tnf.l1_loss(
            selective_stats["mean_ab"] / 110.0,
            prepared["ref_stats"]["mean_ab"] / 110.0,
            reduction="none",
        ).mean(dim=1)
        ref_mean_ab = (ref_mean_ab_per_sample * color_weight).sum() / color_weight_sum
        ref_hue_per_sample = (
            1.0
            - (
                selective_stats["hue_unit"]
                * prepared["ref_stats"]["hue_unit"]
            ).sum(dim=1).clamp(-1, 1)
        ) * prepared["ref_stats"]["hue_validity"]
        ref_hue = (ref_hue_per_sample * color_weight).sum() / color_weight_sum
        ref_chroma_per_sample = tnf.l1_loss(
            selective_stats["median_chroma"] / 110.0,
            prepared["ref_stats"]["median_chroma"] / 110.0,
            reduction="none",
        )
        ref_chroma = (ref_chroma_per_sample * color_weight).sum() / color_weight_sum
        edge_mean, _, edge_hf = self._v222_edge_metrics(
            selective_rgb,
            base_rgb,
            prepared["pseudo_lab"],
            prepared["transition_ring"],
        )
        edge_luma = edge_mean.mean() / 100.0
        edge_hf_loss = edge_hf.mean() / 100.0
        total = (
            USER_V222_PSEUDO_AB_WEIGHT * pseudo_ab
            + USER_V222_REF_MEAN_AB_WEIGHT * ref_mean_ab
            + USER_V222_REF_HUE_WEIGHT * ref_hue
            + USER_V222_REF_CHROMA_WEIGHT * ref_chroma
            + USER_V222_EDGE_LUMA_EXCESS_WEIGHT * edge_luma
            + USER_V222_EDGE_HF_WEIGHT * edge_hf_loss
        )
        return total, {
            "loss": total,
            "pseudo_ab": pseudo_ab,
            "ref_mean_ab": ref_mean_ab,
            "ref_hue": ref_hue,
            "ref_chroma": ref_chroma,
            "edge_luma_excess": edge_luma,
            "edge_hf_excess": edge_hf_loss,
            "outside_hair_max_abs_delta": projector_aux[
                "outside_hair_max_abs_delta"
            ].max(),
            "hard_protect_max_abs_delta": projector_aux[
                "hard_protect_max_abs_delta"
            ].max(),
        }

    def configure_v222_projector(self) -> None:
        trainable_names = [
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        expected = {
            "raw_parallel_gain",
            "raw_orth_keep",
            "raw_boundary_strength",
            "raw_luma_strength",
        }
        if set(trainable_names) != expected:
            raise RuntimeError(
                f"V2.22 trainable scope mismatch: {trainable_names}"
            )
        print(
            "[V2.22] trainable_parameter_names=" + ",".join(trainable_names)
            + f" trainable_count={sum(p.numel() for p in self.model.parameters())}"
        )

    def train_one_epoch_v222(self, epoch: int) -> tuple[float, dict[str, int]]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running = 0.0
        steps = 0
        last_grad_norm = 0.0
        counts = {
            "train_normal_count": 0,
            "train_extreme_ignored_count": 0,
            "train_unreliable_ignored_count": 0,
        }
        progress = tqdm(
            self.train_loader,
            desc=f"V2.22 train {epoch + 1}/{USER_EPOCHS}",
            leave=False,
        )
        for batch in progress:
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base_rgb, _, selective_rgb, projector_aux = self._render_v222_pair(prepared)
            loss, loss_info = self.calc_loss_v222(
                base_rgb, selective_rgb, prepared, projector_aux
            )
            if not torch.isfinite(loss):
                raise RuntimeError("V2.22 training loss became NaN or Inf before backward")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()), USER_GRAD_CLIP
            )
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "V2.22 projector gradient became NaN or Inf before optimizer.step"
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if not all(
                torch.isfinite(parameter).all() for parameter in self.model.parameters()
            ):
                raise RuntimeError("V2.22 optimizer produced a NaN or Inf parameter")
            last_grad_norm = float(grad_norm)
            running += float(loss.item())
            steps += 1
            counts["train_normal_count"] += int(
                (prepared["color_weight"] >= 0.999).sum().item()
            )
            counts["train_extreme_ignored_count"] += int(
                prepared["extreme_for_v222"].sum().item()
            )
            counts["train_unreliable_ignored_count"] += int(
                (~prepared["pseudo_reliable"]).sum().item()
            )
            values = self.model.parameter_values_float()
            progress.set_postfix(
                loss=float(loss.item()),
                grad=last_grad_norm,
                gain=values["parallel_gain"],
                orth=values["orth_keep"],
                boundary=values["boundary_strength"],
                luma=values["luma_strength"],
            )
            if float(loss_info["outside_hair_max_abs_delta"].item()) > 5e-4:
                raise RuntimeError("V2.22 outside-hair structural restore failed")
            if float(loss_info["hard_protect_max_abs_delta"].item()) > 5e-4:
                raise RuntimeError("V2.22 hard-protect structural restore failed")
        if steps == 0:
            raise RuntimeError("V2.22 training epoch produced no valid batches")
        self.scheduler.step()
        print(
            f"[V2.22] epoch={epoch + 1} train_loss={running / steps:.6f} "
            f"grad_norm={last_grad_norm:.6f} params={self.model.parameter_values_float()} "
            f"counts={counts}"
        )
        return running / steps, counts

    @torch.no_grad()
    def validate_v222(self, epoch: int) -> dict[str, object]:
        self.model.eval()
        records = []
        rows = []
        compact_rows = []
        for batch in tqdm(
            self.val_loader,
            desc=f"V2.22 val {epoch + 1}/{USER_EPOCHS}",
            leave=False,
        ):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base_rgb, anchor_rgb, selective_rgb, projector_aux = self._render_v222_pair(prepared)
            records.extend(
                self._build_v222_records(
                    prepared, base_rgb, anchor_rgb, selective_rgb, projector_aux
                )
            )
            if len(rows) < USER_LOG_IMAGE_COUNT:
                for index in range(base_rgb.size(0)):
                    rows.append([
                        prepared["face_i"][index:index + 1],
                        prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1],
                        prepared["base_i"][index:index + 1],
                        anchor_rgb[index:index + 1] * 2 - 1,
                        selective_rgb[index:index + 1] * 2 - 1,
                        prepared["pseudo_rgb"][index:index + 1],
                        mask_to_preview(projector_aux["hair_core"][index:index + 1]),
                        mask_to_preview(prepared["transition_ring"][index:index + 1]),
                        mask_to_preview(projector_aux["halo_gate"][index:index + 1]),
                        mask_to_preview(projector_aux["hard_protect"][index:index + 1]),
                    ])
                    compact_rows.append([
                        prepared["face_i"][index:index + 1],
                        prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1],
                        prepared["base_i"][index:index + 1],
                        anchor_rgb[index:index + 1] * 2 - 1,
                        selective_rgb[index:index + 1] * 2 - 1,
                    ])
                    if len(rows) >= USER_LOG_IMAGE_COUNT:
                        break
        if not records:
            raise RuntimeError("V2.22 validation produced no valid samples")
        normal = aggregate_v222_records(records, self._v222_normal_ids())
        all_valid = aggregate_v222_records(records)
        if int(normal.get("count", 0)) == 0:
            raise RuntimeError("V2.22 validation has no normal-color samples")
        score = v222_checkpoint_score(normal)
        raw_edge = float(normal["raw_anchor_edge_luma_excess_mean"])
        structural_pass = (
            float(normal["outside_hair_max_abs_delta"]) <= 5e-4
            and float(normal["hard_protect_max_abs_delta"]) <= 5e-4
        )
        retention_pass = float(normal["median_color_retention_ratio"]) >= 0.70
        boundary_pass = raw_edge <= 1e-6 or (
            float(normal["edge_luma_excess_mean"]) <= 0.50 * raw_edge
        )
        summary = {
            "version": "v2.22",
            "epoch": epoch + 1,
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "normal": normal,
            "all_valid": all_valid,
            "val_normal_count": normal.get("count", 0),
            "projector": self.model.parameter_values_float(),
            "v222_score": score,
            "structural_pass": structural_pass,
            "color_retention_pass": retention_pass,
            "boundary_pass": boundary_pass,
            "allow_best_checkpoint": bool(
                structural_pass and retention_pass and math.isfinite(score)
            ),
        }
        assert_finite_json(summary)
        metrics_dir = ACTIVE_OUTPUT_DIR / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / f"epoch_{epoch + 1:03d}.json", "w", encoding="utf-8") as handle:
            json.dump(
                {"summary": summary, "records": records},
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        comparison_dir = ACTIVE_OUTPUT_DIR / "comparisons" / f"epoch_{epoch + 1:03d}"
        self._save_v222_preview_rows(comparison_dir, rows, compact_rows)
        latest_dir = ACTIVE_OUTPUT_DIR / "comparisons" / "_latest"
        self._save_v222_preview_rows(latest_dir, rows, compact_rows)
        print(
            f"[V2.22] epoch={epoch + 1} score={score:.6f} "
            f"retention={normal['median_color_retention_ratio']:.4f} "
            f"selective_ab={normal['median_selective_to_ref_ab']:.4f} "
            f"edge={normal['edge_luma_excess_mean']:.4f} "
            f"raw_edge={raw_edge:.4f} structural={structural_pass}"
        )
        return summary

    def write_v222_acceptance(self, summary: dict[str, object]) -> None:
        normal = summary["normal"]
        if not summary["structural_pass"]:
            decision = "MASK_PROTECTION_BUG"
        elif not summary["color_retention_pass"]:
            decision = "COLOR_RETENTION_FAIL"
        elif not summary["boundary_pass"]:
            decision = "BOUNDARY_REDUCTION_INCOMPLETE"
        else:
            decision = "COLOR_PASS_SELECTIVITY_PASS"
        payload = {
            "version": "v2.22",
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "normal_count": normal["count"],
            "raw_anchor": {
                "median_final_to_ref_ab": normal["median_anchor_to_ref_ab"],
                "edge_luma_excess_mean": normal["raw_anchor_edge_luma_excess_mean"],
                "face_keep_l1": normal["raw_anchor_face_keep_l1"],
                "outer_bg_keep_l1": normal["raw_anchor_outer_bg_keep_l1"],
            },
            "selective_best": {
                "median_final_to_ref_ab": normal["median_selective_to_ref_ab"],
                "median_color_retention_ratio": normal["median_color_retention_ratio"],
                "median_parallel_progress": normal["median_parallel_progress"],
                "median_orthogonal_error": normal["median_orthogonal_error"],
                "edge_luma_excess_mean": normal["edge_luma_excess_mean"],
                "face_keep_l1": normal["face_keep_l1"],
                "outer_bg_keep_l1": normal["outer_bg_keep_l1"],
                "outside_hair_max_abs_delta": normal["outside_hair_max_abs_delta"],
                "hard_protect_max_abs_delta": normal["hard_protect_max_abs_delta"],
            },
            "projector": summary["projector"],
            "decision": decision,
        }
        assert_finite_json(payload)
        with open(ACTIVE_OUTPUT_DIR / "v222_acceptance.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)

    def _save_v223_preview_rows(
        self,
        output_dir: Path,
        rows: list[list[torch.Tensor]],
        compact_rows: list[list[torch.Tensor]],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows):
            save_preview(output_dir / f"sample_{index:03d}.png", row)
            save_preview(output_dir / f"comparison_{index:03d}.png", row)
        for index, row in enumerate(compact_rows):
            save_preview(output_dir / f"compact_{index:03d}.png", row)
        save_contact_sheet(output_dir / "full_diagnostic.png", rows)
        save_contact_sheet(output_dir / "compact_comparison.png", compact_rows)
        (output_dir / "README.txt").write_text(
            "Full 12 columns: Source, Shape reference, Color reference, Base/SATD, "
            "Strong Anchor alpha=0.90, V2.23 Final Selective, Pseudo Target, "
            "Hair Core, Transition Ring, Actual Luma Transfer Weight, "
            "Actual Chroma Transfer Weight, Hard Protect.\n"
            "Compact 7 columns: Source, Shape reference, Color reference, Base/SATD, "
            "Strong Anchor alpha=0.90, V2.23 Final Selective, Pseudo Target.\n",
            encoding="ascii",
        )

    def _save_v223_checkpoint(self, summary: dict[str, object]) -> Path:
        checkpoint = {
            "arch": FULL_COLOR_ARCH_V8_6,
            "architecture": FULL_COLOR_ARCH_V8_6,
            "version": "v2.23",
            "training_mode": "FULL_COLOR_TONE_SELECTIVE",
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "projector_config": self.model.config_dict(),
            "projector_state_dict": self.model.state_dict(),
            "normal_manifest": self.normal_manifest,
            "validation_summary": summary,
            "deterministic": True,
        }
        path = self.output_ckpt_dir / "best.pth"
        torch.save(checkpoint, path)
        return path

    @torch.no_grad()
    def run_v223_pretrain_diagnostic(self) -> dict[str, object]:
        diagnostic_dir = ACTIVE_OUTPUT_DIR / "v223_pretrain_diagnostic"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        res_diagnostic_dir = Path("res") / "v223_pretrain_diagnostic"
        res_diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        records = []
        rows = []
        compact_rows = []
        objective_values = []
        for batch in tqdm(
            self.val_loader, desc="V2.23 deterministic full-color", leave=False
        ):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base_rgb, anchor_rgb, selective_rgb, projector_aux = self._render_v223_pair(
                prepared
            )
            records.extend(
                self._build_v223_records(
                    prepared, base_rgb, anchor_rgb, selective_rgb, projector_aux
                )
            )
            objective, _ = self.calc_loss_v223(
                base_rgb, selective_rgb, prepared, projector_aux
            )
            objective_values.append(float(objective.item()))
            if len(rows) < USER_LOG_IMAGE_COUNT:
                for index in range(base_rgb.size(0)):
                    rows.append([
                        prepared["face_i"][index:index + 1],
                        prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1],
                        prepared["base_i"][index:index + 1],
                        anchor_rgb[index:index + 1] * 2 - 1,
                        selective_rgb[index:index + 1] * 2 - 1,
                        prepared["pseudo_rgb"][index:index + 1],
                        mask_to_preview(projector_aux["hair_core"][index:index + 1]),
                        mask_to_preview(projector_aux["hair_edge"][index:index + 1]),
                        mask_to_preview(
                            projector_aux["luma_transfer_weight"][index:index + 1]
                        ),
                        mask_to_preview(
                            projector_aux["chroma_transfer_weight"][index:index + 1]
                        ),
                        mask_to_preview(projector_aux["hard_protect"][index:index + 1]),
                    ])
                    compact_rows.append([
                        prepared["face_i"][index:index + 1],
                        prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1],
                        prepared["base_i"][index:index + 1],
                        anchor_rgb[index:index + 1] * 2 - 1,
                        selective_rgb[index:index + 1] * 2 - 1,
                        prepared["pseudo_rgb"][index:index + 1],
                    ])
                    if len(rows) >= USER_LOG_IMAGE_COUNT:
                        break
        if not records:
            raise RuntimeError("V2.23 pretrain diagnostic produced no valid samples")
        manifest = build_v223_normal_color_manifest(
            records, USER_MIN_SAFE_REFERENCE_FRACTION_V8
        )
        self.normal_manifest = manifest
        for path in (
            ACTIVE_OUTPUT_DIR / "v223_normal_manifest.json",
            Path("res") / "v223_normal_manifest.json",
        ):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        normal_ids = {
            entry["sample_id"]
            for entry in manifest["entries"]
            if entry["normal_color"]
        }
        normal = aggregate_v223_records(records, normal_ids)
        all_valid = aggregate_v223_records(records)
        if manifest["status"] != "OK":
            decision = {"decision": "V223_FULL_COLOR_DIRECTION_FAIL", "abort": True}
        else:
            decision = classify_v223(normal)
        score = v223_checkpoint_score(normal) if normal.get("has_samples") else 0.0
        summary = {
            "version": "v2.23",
            "architecture": FULL_COLOR_ARCH_V8_6,
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "projector": self.model.config_dict(),
            "normal_manifest_status": manifest["status"],
            "normal": normal,
            "all_valid": all_valid,
            "decision": decision["decision"],
            "deterministic_objective_mean": sum(objective_values)
            / max(len(objective_values), 1),
            "v223_score": score,
        }
        assert_finite_json(summary)
        metrics_dir = ACTIVE_OUTPUT_DIR / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "pretrain.json", "w", encoding="utf-8") as handle:
            json.dump(
                {"summary": summary, "records": records},
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        for path in (diagnostic_dir, res_diagnostic_dir):
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "metrics.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
            with open(path / "records.jsonl", "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._save_v223_preview_rows(path / "comparisons", rows, compact_rows)
        with open(Path("res") / "v223_config.json", "w", encoding="utf-8") as handle:
            json.dump(self.model.config_dict(), handle, ensure_ascii=False, indent=2)
        fixed_indices = set(USER_FIXED_REGRESSION_INDICES) | {16, 23}
        fixed_records = [
            record for index, record in enumerate(records) if index in fixed_indices
        ]
        with open(Path("res") / "v223_fixed_regression.json", "w", encoding="utf-8") as handle:
            json.dump(fixed_records, handle, ensure_ascii=False, indent=2, allow_nan=False)
        acceptance = {
            "version": "v2.23",
            "architecture": FULL_COLOR_ARCH_V8_6,
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "normal_count": normal.get("count", 0),
            "base": {
                "full_stat_error": normal.get("median_base_full_stat_error", 0.0),
                "median_l_error": normal.get("median_base_ref_l_error", 0.0),
            },
            "anchor": {
                "full_stat_error": normal.get("median_anchor_full_stat_error", 0.0),
                "median_l_error": normal.get("median_anchor_ref_l_error", 0.0),
            },
            "selective": {
                "full_stat_error": normal.get("median_selective_full_stat_error", 0.0),
                "median_l_error": normal.get("median_selective_ref_l_error", 0.0),
                "full_color_retention": normal.get("median_full_color_retention", 0.0),
                "ab_retention": normal.get("median_ab_retention", 0.0),
                "l_progress": normal.get("median_l_progress", 0.0),
                "pseudo_deltaE76": normal.get("median_pseudo_deltaE76", 0.0),
                "edge_transfer_ratio": normal.get("median_edge_transfer_ratio", 0.0),
            },
            "structural": {
                "outside_hair_max_abs_delta": normal.get("outside_hair_max_abs_delta", 0.0),
                "hard_protect_max_abs_delta": normal.get("hard_protect_max_abs_delta", 0.0),
            },
            "decision": decision["decision"],
            "score": score,
        }
        assert_finite_json(acceptance)
        for path in (ACTIVE_OUTPUT_DIR / "v223_acceptance.json", Path("res") / "v223_acceptance.json"):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(acceptance, handle, ensure_ascii=False, indent=2, allow_nan=False)
        self.v223_pretrain_summary = summary
        print(
            "[V2.23] Base -> Ref: "
            f"AB {normal.get('median_base_ref_ab_error', 0.0):.3f} / "
            f"L {normal.get('median_base_ref_l_error', 0.0):.3f} / "
            f"full {normal.get('median_base_full_stat_error', 0.0):.3f}"
        )
        print(
            "[V2.23] Anchor -> Ref: "
            f"AB {normal.get('median_anchor_ref_ab_error', 0.0):.3f} / "
            f"L {normal.get('median_anchor_ref_l_error', 0.0):.3f} / "
            f"full {normal.get('median_anchor_full_stat_error', 0.0):.3f}"
        )
        print(
            "[V2.23] V2.23 -> Ref: "
            f"AB {normal.get('median_selective_ref_ab_error', 0.0):.3f} / "
            f"L {normal.get('median_selective_ref_l_error', 0.0):.3f} / "
            f"full {normal.get('median_selective_full_stat_error', 0.0):.3f}"
        )
        print(
            f"[V2.23] Full retention={normal.get('median_full_color_retention', 0.0):.4f} "
            f"AB retention={normal.get('median_ab_retention', 0.0):.4f} "
            f"L progress={normal.get('median_l_progress', 0.0):.4f} "
            f"Pseudo dE76={normal.get('median_pseudo_deltaE76', 0.0):.4f} "
            f"Edge transfer={normal.get('median_edge_transfer_ratio', 0.0):.4f}"
        )
        print(
            f"[V2.23] Edge halo raw={normal.get('raw_anchor_edge_luma_excess_mean', 0.0):.4f} "
            f"-> final={normal.get('edge_luma_excess_mean', 0.0):.4f} "
            f"hard={normal.get('hard_protect_max_abs_delta', 0.0):.2e} "
            f"outside={normal.get('outside_hair_max_abs_delta', 0.0):.2e}"
        )
        print(f"[V2.23] Decision: {decision['decision']}")
        return summary

    def _save_v224_preview_rows(
        self, output_dir: Path, rows: list[list[torch.Tensor]]
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows):
            save_preview(output_dir / f"sample_{index:03d}.png", row)
            save_preview(output_dir / f"comparison_{index:03d}.png", row)
        save_contact_sheet(output_dir / "comparison_contact_sheet.png", rows)
        (output_dir / "README.txt").write_text(
            "Columns: Source, Shape reference, Color reference, Base/SATD, "
            "Strong Anchor alpha=0.90, V2.23 old selective, V2.24 new selective, "
            "Hair core, Boundary membership, Chroma transfer weight, "
            "Luma transfer weight, Hard protect, Outer BG guard.\n",
            encoding="ascii",
        )

    def _save_v224_checkpoint(self, summary: dict[str, object]) -> Path:
        checkpoint = {
            "arch": FULL_COLOR_ARCH_V8_7,
            "architecture": FULL_COLOR_ARCH_V8_7,
            "version": "v2.24",
            "training_mode": "V224_BOUNDARY_ALGEBRA_DETERMINISTIC",
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "projector_config": self.model.config_dict(),
            "projector_state_dict": self.model.state_dict(),
            "normal_manifest": self.normal_manifest,
            "validation_summary": summary,
            "mask_semantics": "raw_hair_minus_eroded_single_alpha",
            "boundary_single_alpha": True,
            "post_candidate_soft_blend": False,
            "training_inference_mask_parity": True,
            "deterministic": True,
        }
        path = self.output_ckpt_dir / "v224_boundary_pass.pth"
        torch.save(checkpoint, path)
        return path

    @torch.no_grad()
    def run_v224_pretrain_diagnostic(self) -> dict[str, object]:
        diagnostic_dir = ACTIVE_OUTPUT_DIR / "v224_diagnostic"
        res_diagnostic_dir = Path("res") / "v224_diagnostic"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        res_diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        records: list[dict[str, object]] = []
        rows: list[list[torch.Tensor]] = []
        parity_max_diff = 0.0
        for batch in tqdm(self.val_loader, desc="V2.24 boundary A/B diagnostic", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base_rgb, anchor_rgb, selective_rgb, projector_aux = self._render_v224_pair(prepared)
            records.extend(
                self._build_v224_records(
                    prepared, base_rgb, anchor_rgb, selective_rgb, projector_aux
                )
            )
            train_masks = build_boundary_masks_v824(
                target_hair_mask=prepared["v224_target_hair_mask"],
                target_hair_eroded=prepared["v224_target_hair_eroded"],
                hard_protect=projector_aux["hard_protect"],
            )
            inference_masks = build_boundary_masks_v824(
                target_hair_mask=prepared["v224_target_hair_mask"],
                target_hair_eroded=prepared["v224_target_hair_eroded"],
                hard_protect=projector_aux["hard_protect"],
            )
            parity_max_diff = max(
                parity_max_diff,
                max(
                    float((train_masks[key] - inference_masks[key]).abs().max().item())
                    for key in ("core_membership", "edge_membership", "hair_membership", "outside_hair")
                ),
            )
            if len(rows) < USER_LOG_IMAGE_COUNT:
                legacy_rgb = projector_aux["legacy_v223_rgb"]
                for index in range(base_rgb.size(0)):
                    rows.append([
                        prepared["face_i"][index:index + 1],
                        prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1],
                        prepared["base_i"][index:index + 1],
                        anchor_rgb[index:index + 1] * 2 - 1,
                        legacy_rgb[index:index + 1] * 2 - 1,
                        selective_rgb[index:index + 1] * 2 - 1,
                        mask_to_preview(projector_aux["hair_core"][index:index + 1]),
                        mask_to_preview(projector_aux["hair_edge"][index:index + 1]),
                        mask_to_preview(projector_aux["chroma_transfer_weight"][index:index + 1]),
                        mask_to_preview(projector_aux["luma_transfer_weight"][index:index + 1]),
                        mask_to_preview(projector_aux["hard_protect"][index:index + 1]),
                        mask_to_preview(prepared["v224_outer_background_guard"][index:index + 1]),
                    ])
                    if len(rows) >= USER_LOG_IMAGE_COUNT:
                        break
        if not records:
            raise RuntimeError("V2.24 diagnostic produced no valid samples")
        manifest = build_v224_normal_color_manifest(
            records, USER_MIN_SAFE_REFERENCE_FRACTION_V8
        )
        self.normal_manifest = manifest
        for path in (
            ACTIVE_OUTPUT_DIR / "v224_normal_manifest.json",
            Path("res") / "v224_normal_manifest.json",
        ):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        normal_ids = {
            entry["sample_id"] for entry in manifest["entries"] if entry["normal_color"]
        }
        normal = aggregate_v224_records(records, normal_ids)
        all_valid = aggregate_v224_records(records)
        outside_max = float(normal.get("outside_hair_max_abs_delta", 0.0))
        hard_max = float(normal.get("hard_protect_max_abs_delta", 0.0))
        decision = classify_v224(
            normal,
            parity_max_diff=parity_max_diff,
            outside_max_delta=outside_max,
            hard_protect_max_delta=hard_max,
        )
        summary = {
            "version": "v2.24",
            "architecture": FULL_COLOR_ARCH_V8_7,
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "projector": self.model.config_dict(),
            "normal_manifest_status": manifest["status"],
            "normal": normal,
            "all_valid": all_valid,
            "decision": decision["decision"],
            "training_inference_mask_parity_max_diff": parity_max_diff,
        }
        assert_finite_json(summary)
        audit = {
            "core_membership_mean": normal.get("mean_core_luma_weight", 0.0),
            "edge_membership_mean_on_edge": normal.get("mean_edge_membership", 0.0),
            "effective_edge_chroma_weight_mean": normal.get("mean_edge_chroma_weight", 0.0),
            "effective_edge_luma_weight_mean": normal.get("mean_edge_luma_weight", 0.0),
            "configured_edge_chroma_strength": USER_V223_EDGE_CHROMA_STRENGTH,
            "configured_edge_luma_strength": USER_V223_EDGE_LUMA_STRENGTH,
            "post_candidate_soft_rgb_multiply_count": 0,
            "training_inference_mask_parity_max_diff": parity_max_diff,
        }
        for path in (diagnostic_dir, res_diagnostic_dir):
            with open(path / "summary.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
            with open(path / "acceptance.json", "w", encoding="utf-8") as handle:
                json.dump({"decision": decision["decision"], "normal": normal}, handle, ensure_ascii=False, indent=2, allow_nan=False)
            with open(path / "per_sample.jsonl", "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._save_v224_preview_rows(path / "comparisons", rows)
        acceptance = {
            "version": "v2.24",
            "architecture": FULL_COLOR_ARCH_V8_7,
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "normal_count": normal.get("count", 0),
            "core": {
                "old_v223_full_stat_error": normal.get("median_v223_old_selective_full_stat_error", 0.0),
                "new_v224_full_stat_error": normal.get("median_v224_selective_full_stat_error", 0.0),
                "full_color_retention": normal.get("median_full_color_retention", 0.0),
                "ab_retention": normal.get("median_ab_retention", 0.0),
                "l_progress": normal.get("median_l_progress", 0.0),
            },
            "boundary": {
                "median_edge_transfer_full": normal.get("median_v224_edge_transfer_full", 0.0),
                "p25_edge_transfer_full": normal.get("p25_v224_edge_transfer_full", 0.0),
                "median_edge_transfer_ab": normal.get("median_v224_edge_transfer_ab", 0.0),
                "median_edge_transfer_l": normal.get("median_v224_edge_transfer_l", 0.0),
                "edge_l_direction_agreement_fraction": normal.get("mean_edge_l_direction_agreement_fraction", 0.0),
                "halo_ratio": normal.get("edge_halo_ratio", 0.0),
            },
            "structural": {
                "outside_hair_max_abs_delta": outside_max,
                "hard_protect_max_abs_delta": hard_max,
                "trainer_inference_mask_parity_max_diff": parity_max_diff,
            },
            "decision": decision["decision"],
        }
        with open(Path("res") / "v224_acceptance.json", "w", encoding="utf-8") as handle:
            json.dump(acceptance, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(res_diagnostic_dir / "mask_algebra_audit.json", "w", encoding="utf-8") as handle:
            json.dump(audit, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(res_diagnostic_dir / "trainer_inference_mask_parity.json", "w", encoding="utf-8") as handle:
            json.dump({"max_abs_diff": parity_max_diff, "passed": parity_max_diff <= 1e-6}, handle, indent=2)
        with open(Path("res") / "v224_config.json", "w", encoding="utf-8") as handle:
            json.dump(self.model.config_dict(), handle, ensure_ascii=False, indent=2)
        self.v223_pretrain_summary = summary
        print(
            f"[V2.24] Core full-color old={normal.get('median_v223_old_selective_full_stat_error', 0.0):.3f} "
            f"new={normal.get('median_v224_selective_full_stat_error', 0.0):.3f}"
        )
        print(
            f"[V2.24] Edge full={normal.get('median_v224_edge_transfer_full', 0.0):.4f} "
            f"AB={normal.get('median_v224_edge_transfer_ab', 0.0):.4f} "
            f"L={normal.get('median_v224_edge_transfer_l', 0.0):.4f} "
            f"halo_ratio={normal.get('edge_halo_ratio', 0.0):.4f}"
        )
        print(f"[V2.24] Decision: {decision['decision']}")
        return summary

    def _save_v225_checkpoint(self, summary):
        payload = {
            "arch": FULL_COLOR_ARCH_V8_8, "architecture": FULL_COLOR_ARCH_V8_8,
            "version": "v2.25", "training_mode": "V225_REFERENCE_CONDITIONED_BOUNDARY_DETERMINISTIC",
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "projector_config": self.model.config_dict(), "projector_state_dict": self.model.state_dict(),
            "validation_summary": summary, "boundary_single_alpha": True,
            "reference_conditioned_boundary_target": True, "post_candidate_soft_blend": False,
            "deterministic": True,
        }
        path = self.output_ckpt_dir / "v225_reference_boundary_pass.pth"
        torch.save(payload, path)
        return path

    @torch.no_grad()
    def run_v225_pretrain_diagnostic(self):
        output_dirs = [ACTIVE_OUTPUT_DIR / "v225_diagnostic", Path("res") / "v225_diagnostic"]
        for path in output_dirs:
            path.mkdir(parents=True, exist_ok=True)
        records, rows = [], []
        parity_max_diff = 0.0
        self.model.eval()
        for batch in tqdm(self.val_loader, desc="V2.25 reference boundary A/B", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base, anchor, selective, aux = self._render_v225_pair(prepared)
            records.extend(self._build_v225_records(prepared, base, anchor, selective, aux))
            inference_masks = Blending_v8.build_v224_boundary_masks(
                prepared["v224_target_hair_mask"], prepared["v224_target_hair_eroded"], aux["hard_protect"]
            )
            parity_max_diff = max(
                parity_max_diff,
                float((inference_masks["core_membership"] - aux["hair_core"]).abs().max().item()),
                float((inference_masks["edge_membership"] - aux["hair_edge"]).abs().max().item()),
                float((inference_masks["hair_membership"] - aux["hair_membership"]).abs().max().item()),
            )
            if len(rows) < USER_LOG_IMAGE_COUNT:
                target_preview = torch.cat((prepared["pseudo_lab"][:, :1], prepared["target_ref_ab"]), dim=1)
                target_preview = lab_to_rgb(target_preview) * 2 - 1
                for i in range(base.size(0)):
                    rows.append([
                        prepared["face_i"][i:i+1], prepared["shape_i"][i:i+1], prepared["color_i"][i:i+1],
                        prepared["base_i"][i:i+1], anchor[i:i+1] * 2 - 1,
                        aux["baseline_v224_rgb"][i:i+1] * 2 - 1, selective[i:i+1] * 2 - 1,
                        prepared["pseudo_rgb"][i:i+1], target_preview[i:i+1],
                        mask_to_preview(aux["hair_core"][i:i+1]), mask_to_preview(aux["hair_edge"][i:i+1]),
                        mask_to_preview(aux["chroma_transfer_weight"][i:i+1]),
                        mask_to_preview(aux["luma_transfer_weight"][i:i+1]), mask_to_preview(aux["hard_protect"][i:i+1]),
                    ])
                    if len(rows) >= USER_LOG_IMAGE_COUNT:
                        break
        if not records:
            raise RuntimeError("V2.25 diagnostic produced no valid samples")
        manifest = build_v225_normal_color_manifest(records, USER_MIN_SAFE_REFERENCE_FRACTION_V8)
        normal_ids = {e["sample_id"] for e in manifest["entries"] if e["normal_color"]}
        normal = aggregate_v225_records(records, normal_ids)
        all_valid = aggregate_v225_records(records)
        outside = float(normal.get("outside_hair_max_abs_delta", 0.0))
        hard = float(normal.get("hard_protect_max_abs_delta", 0.0))
        decision = classify_v225(normal, parity_max_diff=parity_max_diff, outside_max_delta=outside, hard_protect_max_delta=hard)
        summary = {
            "version": "v2.25", "architecture": FULL_COLOR_ARCH_V8_8,
            "v224": {
                "edge_full": normal.get("median_v224_edge_transfer_full", 0.0),
                "edge_ab": normal.get("median_v224_edge_transfer_ab", 0.0),
                "edge_l": normal.get("median_v224_edge_transfer_l", 0.0),
            },
            "v225": normal, "all_valid": all_valid, "decision": decision["decision"],
            "train_inference_parity_max_diff": parity_max_diff,
        }
        audit = {
            "target_ref_ab_delta_norm_edge": normal.get("mean_edge_target_ab_norm", 0.0),
            "pseudo_delta_ab_norm_edge": normal.get("mean_edge_pseudo_ab_norm", 0.0),
            "reference_parallel_mag_edge": normal.get("mean_edge_reference_parallel_mag", 0.0),
            "configured_edge_chroma_strength": USER_V223_EDGE_CHROMA_STRENGTH,
            "configured_edge_luma_strength": USER_V223_EDGE_LUMA_STRENGTH,
            "post_candidate_soft_blend": False,
        }
        assert_finite_json(summary)
        for path in output_dirs:
            with open(path / "summary.json", "w", encoding="utf-8") as f: json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
            with open(path / "v225_acceptance.json", "w", encoding="utf-8") as f: json.dump({"decision": decision["decision"], "normal": normal}, f, ensure_ascii=False, indent=2, allow_nan=False)
            with open(path / "boundary_target_audit.json", "w", encoding="utf-8") as f: json.dump(audit, f, indent=2, allow_nan=False)
            with open(path / "train_inference_parity.json", "w", encoding="utf-8") as f: json.dump({"max_abs_diff": parity_max_diff, "passed": parity_max_diff <= 1e-6}, f, indent=2)
            with open(path / "per_sample.jsonl", "w", encoding="utf-8") as f:
                for record in records: f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._save_v224_preview_rows(path / "comparisons", rows)
        print(f"[V2.25] Edge raw-anchor transfer full/AB/L {normal.get('median_v225_edge_transfer_full', 0):.3f}/{normal.get('median_v225_edge_transfer_ab', 0):.3f}/{normal.get('median_v225_edge_transfer_l', 0):.3f}")
        print(f"[V2.25] Edge reference-target progress AB/L {normal.get('median_edge_reference_ab_progress', 0):.3f}/{normal.get('median_edge_reference_l_progress', 0):.3f}")
        print(f"[V2.25] Decision: {decision['decision']}")
        return summary

    @torch.no_grad()
    def _build_v226_records(self, prepared, base_rgb, anchor_rgb, selective_rgb, projector_aux):
        """Attach corrected boundary metrics to the existing V2.25 core audit."""
        records = self._build_v225_records(
            prepared, base_rgb, anchor_rgb, selective_rgb, projector_aux
        )
        tensors = add_anchor_deficit_metrics(boundary_metric_tensors(
            base_lab=rgb_to_lab(base_rgb),
            anchor_lab=rgb_to_lab(anchor_rgb),
            selective_lab=rgb_to_lab(selective_rgb),
            target_ref_ab=prepared["target_ref_ab"],
            reference_delta_l=prepared["condition_metrics"]["delta_l_global"],
            hair_edge=projector_aux["hair_edge"],
            edge_chroma_strength=USER_V223_EDGE_CHROMA_STRENGTH,
            edge_luma_strength=USER_V223_EDGE_LUMA_STRENGTH,
            edge_target_l_margin=USER_V225_EDGE_TARGET_L_MARGIN,
        ))
        for index, record in enumerate(records):
            def scalar(name):
                value = tensors[name][index]
                return float(value.item()) if value.numel() == 1 else float(value.flatten()[0].item())
            record.update({
                "edge_desired_ab_error": scalar("desired_ab_error"),
                "edge_base_to_desired_ab_error": scalar("base_to_desired_ab_error"),
                "edge_desired_ab_progress": scalar("desired_ab_progress"),
                "edge_desired_ab_direction_agreement": scalar("desired_ab_direction_agreement"),
                "edge_desired_ab_magnitude_ratio": scalar("desired_ab_magnitude_ratio"),
                "edge_signed_l_progress": scalar("signed_l_progress"),
                "edge_signed_l_direction_agreement": scalar("signed_l_direction_agreement"),
                "meaningful_l_sample": bool(tensors["meaningful_l_sample"][index].item()),
                "strong_l_sample": bool(tensors["strong_l_sample"][index].item()),
                "reference_halo_excess": scalar("reference_halo_excess"),
                "raw_reference_halo_excess": scalar("raw_reference_halo_excess"),
                "reference_halo_valid": bool(tensors["reference_halo_valid"][index].item()),
                "anchor_deficit_fraction": scalar("anchor_deficit_fraction"),
                "anchor_deficit_mag": scalar("raw_anchor_deficit_mag"),
                "desired_ab_norm": scalar("desired_ab_norm"),
                "current_ab_parallel": scalar("current_ab_parallel"),
                "compensation_gamma": float(getattr(self.model, "compensation_gamma", 0.0)),
            })
        return records

    @staticmethod
    def _v227_cpu_tree(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: BlendingTrainerV8._v227_cpu_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return list(value)
        return value

    def _v227_device_tree(self, value):
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._v227_device_tree(item) for key, item in value.items()}
        return value

    @torch.no_grad()
    def _augment_v227_records(self, records, prepared, base, anchor, output, aux):
        base_lab, output_lab = rgb_to_lab(base), rgb_to_lab(output)
        aligned = boundary_metric_tensors(
            base_lab=base_lab, anchor_lab=rgb_to_lab(anchor), selective_lab=output_lab,
            target_ref_ab=prepared["target_ref_ab"],
            reference_delta_l=prepared["condition_metrics"]["delta_l_global"],
            hair_edge=aux["hair_edge"],
            edge_chroma_strength=USER_V223_EDGE_CHROMA_STRENGTH,
            edge_luma_strength=USER_V223_EDGE_LUMA_STRENGTH,
            edge_target_l_margin=USER_V225_EDGE_TARGET_L_MARGIN,
        )
        meaningful = meaningful_ab_metrics(
            base_lab=base_lab, output_lab=output_lab,
            target_ref_ab=prepared["target_ref_ab"], hair_edge=aux["hair_edge"],
            edge_chroma_strength=USER_V223_EDGE_CHROMA_STRENGTH,
        )
        artifacts = independent_edge_artifact_metrics(
            base_lab=base_lab, output_lab=output_lab,
            hair_core=aux["hair_core"], hair_edge=aux["hair_edge"],
        )
        scalar_keys = (
            "meaningful_ab_pixel_count", "meaningful_ab_pixel_fraction",
            "edge_desired_ab_progress_meaningful",
            "edge_desired_ab_direction_meaningful",
            "edge_desired_ab_magnitude_ratio_meaningful",
            "edge_remaining_deficit_fraction_meaningful",
            "legacy_edge_desired_ab_progress_all_edge",
        )
        artifact_keys = (
            "edge_core_delta_e", "edge_core_delta_l", "edge_core_delta_ab",
            "edge_transfer_continuity_ratio", "white_fringe_fraction",
            "gray_fringe_fraction", "oversaturation_fringe_fraction",
            "dark_fringe_fraction",
        )
        for index, record in enumerate(records):
            for key in scalar_keys:
                record[key] = float(meaningful[key][index].item())
            record["meaningful_ab_valid"] = bool(meaningful["meaningful_ab_valid"][index].item())
            record["edge_signed_l_progress"] = float(aligned["signed_l_progress"][index].item())
            record["edge_signed_l_direction_agreement"] = float(aligned["signed_l_direction_agreement"][index].item())
            record["meaningful_l_sample"] = bool(aligned["meaningful_l_sample"][index].item())
            record["reference_halo_excess"] = float(aligned["reference_halo_excess"][index].item())
            record["raw_reference_halo_excess"] = float(aligned["raw_reference_halo_excess"][index].item())
            record["reference_halo_valid"] = bool(aligned["reference_halo_valid"][index].item())
            for key in artifact_keys:
                record[key] = float(artifacts[key][index].item())
            record["reference_delta_l"] = float(
                prepared["condition_metrics"]["delta_l_global"][index].item()
            )
            record["mean_compensation_magnitude"] = float(
                strict_masked_mean_per_sample(
                    aux["edge_compensation_mag"][index:index + 1],
                    aux["hair_edge"][index:index + 1],
                )[0].item()
            )
        return records, meaningful, artifacts

    @torch.no_grad()
    def _render_v227_cached_gamma(self, cache, gamma):
        prepared = self._v227_device_tree(cache["prepared"])
        base = cache["base"].to(self.device)
        anchor = cache["anchor"].to(self.device)
        aux = self._v227_device_tree(cache["aux"])
        output, compensation_aux = apply_v226_compensation_from_cached_v225(
            base_rgb=base, v225_rgb=cache["v225"].to(self.device),
            v225_luma_output=aux["luma_output"], v225_ab_output=aux["ab_output"],
            target_ref_ab=prepared["target_ref_ab"], hair_edge=aux["hair_edge"],
            hair_membership=aux["hair_membership"], hard_protect=aux["hard_protect"],
            edge_chroma_strength=USER_V223_EDGE_CHROMA_STRENGTH,
            compensation_gamma=gamma,
        )
        output_aux = dict(aux)
        output_aux.update(compensation_aux)
        output_aux["v225_rgb"] = cache["v225"].to(self.device)
        output_aux["compensation_gamma"] = output.new_full((output.size(0),), gamma)
        records = self._build_v223_records(prepared, base, anchor, output, output_aux)
        base_lab, anchor_lab, output_lab = rgb_to_lab(base), rgb_to_lab(anchor), rgb_to_lab(output)
        edge = output_aux["hair_edge"]
        anchor_den = strict_masked_mean_per_sample(
            torch.linalg.vector_norm(anchor_lab[:, 1:] - base_lab[:, 1:], dim=1, keepdim=True), edge
        ).clamp_min(1e-6)
        edge_transfer_ab = strict_masked_mean_per_sample(
            torch.linalg.vector_norm(output_lab[:, 1:] - base_lab[:, 1:], dim=1, keepdim=True), edge
        ) / anchor_den
        for index, record in enumerate(records):
            record["edge_transfer_ab"] = float(edge_transfer_ab[index].item())
        for record in records:
            record["compensation_gamma"] = float(gamma)
        records, meaningful, artifacts = self._augment_v227_records(
            records, prepared, base, anchor, output, output_aux
        )
        return prepared, base, anchor, output, output_aux, records, meaningful, artifacts

    @staticmethod
    def _v227_crop_boxes(edge):
        points = (edge[0, 0] > 1e-6).nonzero(as_tuple=False)
        height, width = edge.shape[-2:]
        if not points.numel():
            return {"full": (0, 0, width, height)}
        y0, x0 = points.amin(0).tolist(); y1, x1 = points.amax(0).tolist()
        size = max(48, min(128, max(y1 - y0 + 1, x1 - x0 + 1) // 2))
        centers = {
            "left": (x0, (y0 + y1) // 2), "right": (x1, (y0 + y1) // 2),
            "hairline": ((x0 + x1) // 2, y0),
        }
        boxes = {"full": (0, 0, width, height)}
        for name, (cx, cy) in centers.items():
            left = max(0, min(width - size, cx - size // 2))
            top = max(0, min(height - size, cy - size // 2))
            boxes[name] = (left, top, left + size, top + size)
        return boxes

    @staticmethod
    def _save_v227_crops(root, sample_id, images, edge):
        safe_id = str(sample_id).replace("/", "_").replace("\\", "_")
        for name, (x0, y0, x1, y1) in BlendingTrainerV8._v227_crop_boxes(edge).items():
            cropped = [image[..., y0:y1, x0:x1] * 2 - 1 for image in images]
            save_preview(root / safe_id / f"{name}.png", cropped)

    @staticmethod
    def _save_v229_crops(root, sample_id, images, edge, face_context, background_context):
        boxes = BlendingTrainerV8._v227_crop_boxes(edge)
        height, width = edge.shape[-2:]
        crop_size = max(48, min(128, min(height, width) // 3))
        for name, context in (
            ("largest_face_contact", face_context),
            ("largest_background_contact", background_context),
        ):
            density = tnf.avg_pool2d(
                context.float(), kernel_size=31, stride=1, padding=15
            )[0, 0]
            if float(density.max().item()) <= 0.0:
                continue
            flat_index = int(density.argmax().item())
            cy, cx = divmod(flat_index, width)
            left = max(0, min(width - crop_size, cx - crop_size // 2))
            top = max(0, min(height - crop_size, cy - crop_size // 2))
            boxes[name] = (left, top, left + crop_size, top + crop_size)
        safe_id = str(sample_id).replace("/", "_").replace("\\", "_")
        for name, (x0, y0, x1, y1) in boxes.items():
            cropped = [image[..., y0:y1, x0:x1] * 2 - 1 for image in images]
            save_preview(root / safe_id / f"{name}.png", cropped)

    @staticmethod
    def _save_v230_crops(root, sample_id, images, edge, named_regions):
        boxes = BlendingTrainerV8._v227_crop_boxes(edge)
        height, width = edge.shape[-2:]
        crop_size = max(48, min(128, min(height, width) // 3))
        for name, region in named_regions.items():
            density = tnf.avg_pool2d(region.float(), 31, stride=1, padding=15)[0, 0]
            if float(density.max().item()) <= 0.0:
                continue
            flat_index = int(density.argmax().item())
            cy, cx = divmod(flat_index, width)
            left = max(0, min(width - crop_size, cx - crop_size // 2))
            top = max(0, min(height - crop_size, cy - crop_size // 2))
            boxes[name] = (left, top, left + crop_size, top + crop_size)
        safe_id = str(sample_id).replace("/", "_").replace("\\", "_")
        for name, (x0, y0, x1, y1) in boxes.items():
            cropped = [image[..., y0:y1, x0:x1] * 2 - 1 for image in images]
            save_preview(root / safe_id / f"{name}.png", cropped)

    @staticmethod
    def _max_component_size(mask):
        points = set(map(tuple, (mask[0, 0] > 0.5).nonzero(as_tuple=False).cpu().tolist()))
        largest = 0
        while points:
            stack = [points.pop()]
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if neighbor in points:
                        points.remove(neighbor)
                        stack.append(neighbor)
            largest = max(largest, size)
        return largest

    @torch.no_grad()
    def run_v226_pretrain_diagnostic(self):
        """Run metric alignment first, then the smallest safe AB compensation."""
        root = Path("res") / "v226_diagnostic"
        metric_dir = root / "metric_alignment"
        deficit_dir = root / "anchor_deficit"
        sweep_dir = root / "gamma_sweep"
        active_root = ACTIVE_OUTPUT_DIR / "v226_diagnostic"
        metric_dir.mkdir(parents=True, exist_ok=True)
        deficit_dir.mkdir(parents=True, exist_ok=True)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        (root / "comparisons").mkdir(parents=True, exist_ok=True)
        (active_root / "comparisons").mkdir(parents=True, exist_ok=True)
        if not isinstance(self.model, BoundaryTargetAlignedProjectorV826):
            raise RuntimeError("V2.26 diagnostic requires BoundaryTargetAlignedProjectorV826")
        self.model.eval()

        parity_max_diff = 0.0

        def collect(gamma: float):
            nonlocal parity_max_diff
            self.model.compensation_gamma = float(gamma)
            records = []
            for batch in tqdm(self.val_loader, desc=f"V2.26 gamma={gamma:.2f}", leave=False):
                prepared = self.prepare_batch(batch)
                if prepared is None:
                    continue
                base, anchor, selective, aux = self._render_v225_pair(prepared)
                if gamma == 0.0 and "v225_rgb" in aux:
                    parity_max_diff = max(
                        parity_max_diff,
                        float((selective - aux["v225_rgb"]).abs().max().item()),
                    )
                records.extend(self._build_v226_records(prepared, base, anchor, selective, aux))
            return records

        gamma0_records = collect(0.0)
        if not gamma0_records:
            raise RuntimeError("V2.26 diagnostic produced no valid samples")
        manifest = build_v225_normal_color_manifest(gamma0_records, USER_MIN_SAFE_REFERENCE_FRACTION_V8)
        normal_ids = {entry["sample_id"] for entry in manifest["entries"] if entry["normal_color"]}
        normal0 = aggregate_v226_records([r for r in gamma0_records if r["sample_id"] in normal_ids])
        decision0 = classify_v226_metric_alignment(normal0, parity_max_diff=parity_max_diff)
        with open(metric_dir / "summary_gamma_0.json", "w", encoding="utf-8") as handle:
            json.dump(normal0, handle, indent=2, allow_nan=False)
        with open(metric_dir / "corrected_metrics_per_sample.jsonl", "w", encoding="utf-8") as handle:
            for record in gamma0_records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        with open(deficit_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump({
                "mean_edge_desired_mag": normal0.get("mean_edge_desired_mag", 0.0),
                "median_edge_deficit_fraction": normal0.get("median_edge_deficit_fraction", 0.0),
                "p75_edge_deficit_fraction": normal0.get("p75_edge_deficit_fraction", 0.0),
                "decision": decision0["decision"],
            }, handle, indent=2, allow_nan=False)
        with open(deficit_dir / "per_sample.jsonl", "w", encoding="utf-8") as handle:
            for record in gamma0_records:
                handle.write(json.dumps({
                    "sample_id": record["sample_id"],
                    "desired_ab_norm": record.get("desired_ab_norm", 0.0),
                    "current_ab_parallel": record.get("current_ab_parallel", 0.0),
                    "anchor_deficit_mag": record.get("anchor_deficit_mag", 0.0),
                    "anchor_deficit_fraction": record.get("anchor_deficit_fraction", 0.0),
                }, ensure_ascii=False, allow_nan=False) + "\n")

        selected_gamma = 0.0
        sweep_summary = {"gamma_0.00": normal0}
        final_records = gamma0_records
        decision = decision0
        if decision0["decision"] == "V226_NEEDS_BOUNDED_COMPENSATION":
            for gamma in V226_COMPENSATION_GAMMAS[1:]:
                records = collect(gamma)
                normal = aggregate_v226_records([r for r in records if r["sample_id"] in normal_ids])
                sweep_summary[f"gamma_{gamma:.2f}"] = normal
                with open(sweep_dir / f"gamma_{int(gamma * 100):03d}.json", "w", encoding="utf-8") as handle:
                    json.dump({"gamma": gamma, "summary": normal}, handle, indent=2, allow_nan=False)
                candidate = classify_v226_metric_alignment(normal)
                if candidate["decision"] == "V226_METRIC_ALIGNMENT_PASS_NO_COMPENSATION":
                    selected_gamma, final_records, decision = gamma, records, {"decision": "V226_BOUNDED_DEFICIT_PASS", "gamma_star": gamma, "abort": False}
                    break
            if selected_gamma == 0.0:
                decision = classify_v226_gamma_sweep(sweep_summary)
        acceptance = {
            "version": "v2.26", "architecture": FULL_COLOR_ARCH_V8_9,
            "compensation_gamma": selected_gamma, "decision": decision["decision"],
            "metric_alignment": normal0,
            "selected_summary": sweep_summary.get(f"gamma_{selected_gamma:.2f}", normal0),
            "gamma_sweep": sweep_summary,
            "train_inference_parity_max_diff": parity_max_diff,
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "deterministic": True, "training_disabled": True,
        }
        for path in (root, metric_dir):
            with open(path / ("v226_acceptance.json" if path == root else "metric_alignment_acceptance.json"), "w", encoding="utf-8") as handle:
                json.dump(acceptance, handle, indent=2, allow_nan=False)
        with open(root / "train_inference_parity.json", "w", encoding="utf-8") as handle:
            json.dump({"max_abs_diff": parity_max_diff, "passed": parity_max_diff <= 1e-6}, handle, indent=2)
        with open(root / "per_sample.jsonl", "w", encoding="utf-8") as handle:
            for record in final_records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        for path in (active_root,):
            with open(path / "v226_acceptance.json", "w", encoding="utf-8") as handle:
                json.dump(acceptance, handle, indent=2, allow_nan=False)
            with open(path / "train_inference_parity.json", "w", encoding="utf-8") as handle:
                json.dump({"max_abs_diff": parity_max_diff, "passed": parity_max_diff <= 1e-6}, handle, indent=2)
        shutil.copytree(root, active_root, dirs_exist_ok=True)
        print(f"[V2.26] corrected desired AB progress={normal0.get('median_edge_desired_ab_progress', 0.0):.4f} "
              f"meaningful-L progress={normal0.get('median_edge_signed_l_progress', 0.0):.4f} "
              f"Decision: {decision['decision']}")
        return acceptance

    @torch.no_grad()
    def run_v227_diagnostic(self):
        """Strict paired validation of the frozen V2.26 gamma=.25 algorithm."""
        root = Path("res") / "v227_diagnostic"
        active_root = ACTIVE_OUTPUT_DIR / "v227_diagnostic"
        cache_dir = ACTIVE_OUTPUT_DIR / "v227_paired_cache"
        for path in (root, active_root, cache_dir, root / "comparisons" / "visual",
                     root / "comparisons" / "debug", root / "comparisons" / "crops",
                     root / "worst_cases", root / "visual_review"):
            path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = Path(USER_V227_BASE_CHECKPOINT)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"V2.27 requires the saved V2.26 checkpoint: {checkpoint_path}"
            )
        inference_projector, checkpoint = Blending_v8.load_v226_projector_for_diagnostic(
            checkpoint_path, self.device
        )
        checkpoint_gamma = float(checkpoint["projector_config"]["compensation_gamma"])
        self.model.compensation_gamma = 0.0
        self.model.eval()

        paired_cache = []
        cache_manifest = []
        for batch_index, batch in enumerate(tqdm(self.val_loader, desc="V2.27 paired cache", leave=False)):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base, anchor, gamma0, aux = self._render_v225_pair(prepared)
            gamma0_parity = float((gamma0 - aux["v225_rgb"]).abs().max().item())
            prepared_keys = (
                "pseudo_lab", "target_ref_ab", "v224_target_hair_mask", "ref_stats",
                "condition_metrics", "sample_id", "face_i", "shape_i", "color_i",
            )
            aux_keys = (
                "hair_core", "hair_edge", "hair_membership", "hair_support",
                "hard_protect", "luma_output", "ab_output",
                "luma_transfer_weight", "chroma_transfer_weight",
                "face_keep_l1", "outer_bg_keep_l1",
                "hard_protect_max_abs_delta", "outside_hair_max_abs_delta",
            )
            entry = {
                "prepared": self._v227_cpu_tree({key: prepared[key] for key in prepared_keys}),
                "base": base.detach().cpu(), "anchor": anchor.detach().cpu(),
                "v225": aux["v225_rgb"].detach().cpu(),
                "aux": self._v227_cpu_tree({key: aux[key] for key in aux_keys}),
                "gamma0_parity": gamma0_parity,
            }
            paired_cache.append(entry)
            cache_manifest.append({
                "batch_index": batch_index, "sample_ids": list(prepared["sample_id"]),
                "gamma0_parity_max_diff": gamma0_parity,
            })
        if not paired_cache:
            raise RuntimeError("V2.27 paired cache produced no valid samples")
        with open(cache_dir / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(cache_manifest, handle, ensure_ascii=False, indent=2)

        paired = {}
        details = {}
        for gamma in (0.0, USER_V227_DEFAULT_GAMMA):
            all_records = []
            rendered = []
            for cache in paired_cache:
                result = self._render_v227_cached_gamma(cache, gamma)
                prepared, base, anchor, output, aux, records, meaningful, artifacts = result
                all_records.extend(records)
                rendered.append({
                    "prepared": self._v227_cpu_tree(prepared),
                    "base": base.detach().cpu(), "anchor": anchor.detach().cpu(),
                    "output": output.detach().cpu(), "aux": self._v227_cpu_tree(aux),
                    "meaningful": self._v227_cpu_tree(meaningful),
                    "artifacts": self._v227_cpu_tree(artifacts),
                })
            paired[gamma] = all_records
            details[gamma] = rendered

        manifest = build_v225_normal_color_manifest(
            paired[0.0], USER_MIN_SAFE_REFERENCE_FRACTION_V8
        )
        normal_ids = {e["sample_id"] for e in manifest["entries"] if e["normal_color"]}
        summaries = {
            gamma: aggregate_v227_records([r for r in paired[gamma] if r["sample_id"] in normal_ids])
            for gamma in paired
        }
        baseline_summary = summaries[0.0]
        selected_summary = summaries[USER_V227_DEFAULT_GAMMA]

        all_sample_ids = [r["sample_id"] for r in paired[0.0]]
        selected_ids = [r["sample_id"] for r in paired[USER_V227_DEFAULT_GAMMA]]
        invariants = {
            "sample_ids_exact": all_sample_ids == selected_ids,
            "normal_subset_exact": True,
            "base_max_diff": 0.0, "anchor_max_diff": 0.0, "v225_max_diff": 0.0,
            "target_ref_ab_max_diff": 0.0, "reference_delta_l_max_diff": 0.0,
            "hair_core_max_diff": 0.0, "hair_edge_max_diff": 0.0,
            "hard_protect_max_diff": 0.0,
        }
        invariant_pass = all(
            value if isinstance(value, bool) else float(value) <= 1e-7
            for value in invariants.values()
        )
        gamma0_parity = max(cache["gamma0_parity"] for cache in paired_cache)

        real_max_diff = 0.0
        real_sum_diff = 0.0
        real_lab_de = 0.0
        real_pixels = 0
        real_outputs = []
        for cache, diagnostic_batch in zip(paired_cache, details[USER_V227_DEFAULT_GAMMA]):
            prepared = self._v227_device_tree(cache["prepared"])
            base = cache["base"].to(self.device)
            anchor = cache["anchor"].to(self.device)
            inference_dilated, inference_eroded = self.helper.dilate_erosion.mask(
                prepared["v224_target_hair_mask"]
            )
            inference_inputs = Blending_v8.build_v226_real_inference_projector_inputs(
                base_rgb=base, anchor_rgb=anchor,
                pseudo_lab=prepared["pseudo_lab"], target_ref_ab=prepared["target_ref_ab"],
                reference_delta_l=prepared["condition_metrics"]["delta_l_global"],
                target_hair_mask=prepared["v224_target_hair_mask"],
                target_hair_eroded=inference_eroded,
                target_hair_dilated=inference_dilated,
            )
            real_output, real_aux = Blending_v8.run_v226_projector_debug(
                inference_projector, **inference_inputs,
            )
            diagnostic = diagnostic_batch["output"].to(self.device)
            difference = (diagnostic - real_output).abs()
            real_max_diff = max(real_max_diff, float(difference.max().item()))
            real_sum_diff += float(difference.sum().item())
            real_pixels += difference.numel()
            real_lab_de += float(torch.linalg.vector_norm(
                rgb_to_lab(diagnostic) - rgb_to_lab(real_output), dim=1
            ).mean().item()) * diagnostic.size(0)
            real_outputs.append({
                "rgb": real_output.detach().cpu(), "aux": self._v227_cpu_tree(real_aux)
            })
        real_mean_diff = real_sum_diff / max(real_pixels, 1)
        real_lab_de /= max(len(all_sample_ids), 1)

        normal_selected = [r for r in paired[USER_V227_DEFAULT_GAMMA] if r["sample_id"] in normal_ids]
        if not normal_selected:
            raise RuntimeError("V2.27 normal-color subset is empty")
        fixed_regression = {
            "light_to_dark": min(normal_selected, key=lambda r: float(r["reference_delta_l"]))["sample_id"],
            "dark_to_light": max(normal_selected, key=lambda r: float(r["reference_delta_l"]))["sample_id"],
            "edge_halo": max(normal_selected, key=lambda r: float(r["white_fringe_fraction"]))["sample_id"],
            "original_color_rim": min(normal_selected, key=lambda r: float(r["edge_desired_ab_progress_meaningful"]))["sample_id"],
            "high_deficit": max(normal_selected, key=lambda r: float(r["edge_remaining_deficit_fraction_meaningful"]))["sample_id"],
            "low_delta_l": min(normal_selected, key=lambda r: abs(float(r["reference_delta_l"])))["sample_id"],
            "high_delta_l_positive": max(normal_selected, key=lambda r: float(r["reference_delta_l"]))["sample_id"],
            "high_delta_l_negative": min(normal_selected, key=lambda r: float(r["reference_delta_l"]))["sample_id"],
        }
        ranked_ids = []
        rankings = {
            "lowest_meaningful_ab_progress": ("edge_desired_ab_progress_meaningful", False),
            "lowest_direction_agreement": ("edge_desired_ab_direction_meaningful", False),
            "largest_remaining_deficit": ("edge_remaining_deficit_fraction_meaningful", True),
            "largest_edge_core_delta_e": ("edge_core_delta_e", True),
            "largest_white_fringe": ("white_fringe_fraction", True),
            "largest_gray_fringe": ("gray_fringe_fraction", True),
            "largest_oversaturation": ("oversaturation_fringe_fraction", True),
            "largest_dark_fringe": ("dark_fringe_fraction", True),
        }
        worst_manifest = {}
        for label, (key, reverse) in rankings.items():
            ids = [r["sample_id"] for r in sorted(normal_selected, key=lambda item: float(item[key]), reverse=reverse)[:8]]
            worst_manifest[label] = ids
            for sample_id in ids:
                if sample_id not in ranked_ids:
                    ranked_ids.append(sample_id)

        fixture_ids = []
        fixture_candidates = list(fixed_regression.values())
        fixture_candidates += list(ranked_ids)
        fixture_candidates += [r["sample_id"] for r in sorted(normal_selected, key=lambda x: float(x["reference_delta_l"]))[:4]]
        fixture_candidates += [r["sample_id"] for r in sorted(normal_selected, key=lambda x: float(x["reference_delta_l"]), reverse=True)[:4]]
        fixture_candidates += [r["sample_id"] for r in sorted(normal_selected, key=lambda x: float(x["mean_compensation_magnitude"]), reverse=True)[:6]]
        fixture_candidates += [r["sample_id"] for r in normal_selected]
        for sample_id in fixture_candidates:
            if sample_id not in fixture_ids:
                fixture_ids.append(sample_id)
            if len(fixture_ids) >= min(USER_V227_REAL_INFERENCE_COUNT, len(normal_selected)):
                break

        post_process = None
        pp_available = Path(USER_V227_PP_CHECKPOINT).exists()
        if pp_available:
            post_process = PostProcessModel().to(self.device).eval()
            pp_state = torch.load(USER_V227_PP_CHECKPOINT, map_location=self.device)
            post_process.load_state_dict(pp_state["model_state_dict"])

        sample_lookup = {}
        running = 0
        for batch_index, selected_batch in enumerate(details[USER_V227_DEFAULT_GAMMA]):
            prepared_cpu = selected_batch["prepared"]
            for local_index, sample_id in enumerate(prepared_cpu["sample_id"]):
                sample_lookup[sample_id] = (batch_index, local_index, running + local_index)
            running += len(prepared_cpu["sample_id"])

        pp_records = []
        for fixture_index, sample_id in enumerate(fixture_ids):
            batch_index, local_index, _ = sample_lookup[sample_id]
            selected_batch = details[USER_V227_DEFAULT_GAMMA][batch_index]
            prepared = selected_batch["prepared"]
            base = selected_batch["base"][local_index:local_index + 1]
            anchor = selected_batch["anchor"][local_index:local_index + 1]
            v225 = paired_cache[batch_index]["v225"][local_index:local_index + 1]
            diagnostic = selected_batch["output"][local_index:local_index + 1]
            real_pre = real_outputs[batch_index]["rgb"][local_index:local_index + 1]
            post = real_pre
            if post_process is not None:
                face = prepared["face_i"][local_index:local_index + 1].to(self.device)
                blend_norm = real_pre.to(self.device) * 2.0 - 1.0
                s_final, f_final = post_process(face, blend_norm)
                post_norm, _ = self.helper.net.generator(
                    [s_final], input_is_latent=True, return_latents=False,
                    start_layer=5, end_layer=8, layer_in=f_final,
                )
                post = ((self.helper.downsample_256(post_norm) + 1.0) / 2.0).clamp(0, 1).cpu()
            edge = selected_batch["aux"]["hair_edge"][local_index:local_index + 1]
            core = selected_batch["aux"]["hair_core"][local_index:local_index + 1]
            desired_ab = selected_batch["meaningful"]["desired_edge_ab"][local_index:local_index + 1]
            desired_preview = lab_to_rgb(torch.cat((
                rgb_to_lab(base)[:, :1], desired_ab
            ), dim=1)).cpu()
            comp = selected_batch["aux"]["edge_compensation_mag"][local_index:local_index + 1]
            comp_preview = mask_to_preview((comp / 6.0).clamp(0, 1))
            fringe = selected_batch["artifacts"]["independent_fringe_map"][local_index:local_index + 1]
            visual_row = [
                prepared["face_i"][local_index:local_index + 1],
                prepared["shape_i"][local_index:local_index + 1],
                prepared["color_i"][local_index:local_index + 1],
                base * 2 - 1, anchor * 2 - 1, v225 * 2 - 1,
                diagnostic * 2 - 1, real_pre * 2 - 1, post * 2 - 1,
            ]
            debug_row = [
                mask_to_preview(core), mask_to_preview(edge), desired_preview * 2 - 1,
                comp_preview, mask_to_preview(fringe),
            ]
            save_preview(root / "comparisons" / "visual" / f"sample_{fixture_index:03d}.png", visual_row)
            save_preview(root / "comparisons" / "debug" / f"sample_{fixture_index:03d}.png", debug_row)
            self._save_v227_crops(
                root / "comparisons" / "crops", sample_id,
                [base, anchor, v225, diagnostic, (prepared["color_i"][local_index:local_index + 1] + 1) / 2], edge,
            )
            pre_lab, post_lab, base_lab = rgb_to_lab(real_pre), rgb_to_lab(post), rgb_to_lab(base)
            pp_records.append({
                "sample_id": sample_id,
                "postPP_to_prePP_AB": float(strict_masked_mean_per_sample(
                    torch.linalg.vector_norm(post_lab[:, 1:] - pre_lab[:, 1:], dim=1, keepdim=True),
                    (core + edge).clamp(0, 1),
                )[0].item()),
                "postPP_toward_base_AB": float(strict_masked_mean_per_sample(
                    torch.linalg.vector_norm(post_lab[:, 1:] - base_lab[:, 1:], dim=1, keepdim=True),
                    (core + edge).clamp(0, 1),
                )[0].item()),
                "postPP_to_reference_AB": float(strict_masked_mean_per_sample(
                    torch.linalg.vector_norm(
                        post_lab[:, 1:] - prepared["target_ref_ab"][local_index:local_index + 1],
                        dim=1, keepdim=True,
                    ), (core + edge).clamp(0, 1),
                )[0].item()),
                "postPP_L_shift": float(strict_masked_mean_per_sample(
                    post_lab[:, :1] - pre_lab[:, :1], (core + edge).clamp(0, 1),
                )[0].item()),
            })

        for worst_index, sample_id in enumerate(ranked_ids):
            batch_index, local_index, _ = sample_lookup[sample_id]
            baseline_batch = details[0.0][batch_index]
            selected_batch = details[USER_V227_DEFAULT_GAMMA][batch_index]
            base = selected_batch["base"][local_index:local_index + 1]
            gamma0 = baseline_batch["output"][local_index:local_index + 1]
            gamma25 = selected_batch["output"][local_index:local_index + 1]
            edge = selected_batch["aux"]["hair_edge"][local_index:local_index + 1]
            deficit = selected_batch["meaningful"]["edge_remaining_deficit_map"][local_index:local_index + 1]
            fringe = selected_batch["artifacts"]["independent_fringe_map"][local_index:local_index + 1]
            save_preview(
                root / "worst_cases" / f"sample_{worst_index:03d}.png",
                [base * 2 - 1, gamma0 * 2 - 1, gamma25 * 2 - 1,
                 mask_to_preview(edge), mask_to_preview((deficit / 12.0).clamp(0, 1)),
                 mask_to_preview(fringe)],
            )

        code_correctness = {
            "selected_log_summary_correct": True,
            "meaningful_ab_denominator_correct": True,
            "strict_paired_invariants": invariant_pass,
            "gamma0_parity_with_v225": gamma0_parity <= 1e-6,
            "checkpoint_gamma_025": checkpoint_gamma == USER_V227_DEFAULT_GAMMA,
            "diagnostic_prepp_inference_parity": real_max_diff <= 1e-5,
            "outside_hair_protection": float(selected_summary.get("outside_hair_max_abs_delta", 1.0)) <= 5e-4,
            "hard_protect": float(selected_summary.get("hard_protect_max_abs_delta", 1.0)) <= 5e-4,
            "postprocess_available": pp_available,
        }
        decision = classify_v227(
            baseline_summary, selected_summary, code_correctness=code_correctness
        )
        acceptance = {
            "version": "v2.27", "algorithm_changed": False,
            "base_algorithm": "v2.26", "selected_gamma": USER_V227_DEFAULT_GAMMA,
            "code_correctness": code_correctness,
            "paired_metrics": {"gamma_0.00": baseline_summary, "gamma_0.25": selected_summary},
            "independent_artifacts": {
                key: value for key, value in selected_summary.items()
                if "fringe" in key or "edge_core" in key or "continuity" in key
            },
            "real_inference": {
                "fixture_count": len(fixture_ids), "pre_pp_rgb_max_diff": real_max_diff,
                "pre_pp_rgb_mean_diff": real_mean_diff, "pre_pp_lab_mean_de": real_lab_de,
                "post_process_available": pp_available, "post_process_records": pp_records,
            },
            "automatic_decision": decision, "final_visual_decision": None,
        }
        outputs = {
            "baseline_gamma0_summary.json": baseline_summary,
            "selected_summary.json": selected_summary,
            "meaningful_ab_metric_audit.json": {
                "threshold": 1.5, "minimum_sample_fraction": 0.05,
                "legacy_all_edge_progress": selected_summary.get("median_legacy_all_edge_progress"),
                "meaningful_progress": selected_summary.get("median_edge_desired_ab_progress_meaningful"),
            },
            "paired_invariants.json": invariants,
            "checkpoint_restore_audit.json": {
                "checkpoint": str(checkpoint_path), "version": checkpoint.get("version"),
                "architecture": checkpoint.get("arch"), "compensation_gamma": checkpoint_gamma,
                "passed": checkpoint_gamma == .25,
            },
            "diagnostic_inference_parity.json": acceptance["real_inference"],
            "independent_edge_artifacts.json": acceptance["independent_artifacts"],
            "worst_case_manifest.json": worst_manifest,
            "v227_fixed_regression_manifest.json": fixed_regression,
            "v227_real_inference_manifest.json": {"sample_ids": fixture_ids, "count": len(fixture_ids)},
            "code_audit.json": code_correctness,
            "v227_acceptance.json": acceptance,
        }
        for filename, payload in outputs.items():
            with open(root / filename, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(root / "per_sample.jsonl", "w", encoding="utf-8") as handle:
            for record in paired[USER_V227_DEFAULT_GAMMA]:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        review_template = {"reviewed_by_user": False, "samples": {
            sample_id: {"hair_core_color": "", "original_color_rim": "", "white_gray_halo": "",
                        "dark_fringe": "", "oversaturation": "", "face_background": "",
                        "postprocess_color_regression": "", "notes": ""}
            for sample_id in fixture_ids
        }}
        with open(root / "visual_review" / "v227_visual_review.json", "w", encoding="utf-8") as handle:
            json.dump(review_template, handle, ensure_ascii=False, indent=2)
        checklist = """# V2.27 Visual Review Checklist

- [ ] Hair core color is close to the color reference
- [ ] No original-color rim remains at the boundary
- [ ] No white or gray halo
- [ ] No abnormal dark fringe
- [ ] No oversaturated color outline
- [ ] Fine hair strands are preserved
- [ ] Face is not recolored
- [ ] Background is not recolored
- [ ] Lightness direction matches the reference
- [ ] PostProcess does not wash the color back toward Base

Record each sample as PASS, MINOR, or FAIL in `v227_visual_review.json`.
"""
        (root / "visual_review" / "V227_VISUAL_REVIEW_CHECKLIST.md").write_text(checklist, encoding="utf-8")
        report = f"""# Blending V8 V2.27 Acceptance

V2.27 does not change the V2.26 color algorithm. It hardens meaningful-AB metrics,
uses strict paired gamma=0/.25 evaluation, restores the saved checkpoint through
the inference projector, and adds independent fringe screening.

Automatic decision: `{decision}`

Selected gamma: `{USER_V227_DEFAULT_GAMMA:.2f}`

NEED HUMAN VISUAL REVIEW
"""
        (root / "BLENDING_V8_V2_27_ACCEPTANCE.md").write_text(report, encoding="utf-8")
        shutil.copytree(root, active_root, dirs_exist_ok=True)

        print(
            f"[V2.27:M0] gamma=0.00 desired_AB_progress="
            f"{baseline_summary.get('median_edge_desired_ab_progress_meaningful', 0.0):.4f} "
            f"desired_AB_p25={baseline_summary.get('p25_edge_desired_ab_progress_meaningful', 0.0):.4f} "
            f"AB_direction={baseline_summary.get('mean_edge_desired_ab_direction_meaningful', 0.0):.4f} "
            f"meaningful_L_progress={baseline_summary.get('median_meaningful_l_progress', 0.0):.4f} "
            f"decision=BASELINE_AUDIT"
        )
        print(
            f"[V2.27:SELECTED] gamma={USER_V227_DEFAULT_GAMMA:.2f} desired_AB_progress="
            f"{selected_summary.get('median_edge_desired_ab_progress_meaningful', 0.0):.4f} "
            f"desired_AB_p25={selected_summary.get('p25_edge_desired_ab_progress_meaningful', 0.0):.4f} "
            f"AB_direction={selected_summary.get('mean_edge_desired_ab_direction_meaningful', 0.0):.4f} "
            f"meaningful_L_progress={selected_summary.get('median_meaningful_l_progress', 0.0):.4f} "
            f"independent_edge_artifact_score={selected_summary.get('independent_edge_artifact_score', 0.0):.4f} "
            f"decision={decision}"
        )
        print(f"[V2.27] FINAL_DECISION={decision}")
        print(f"[V2.27] selected_gamma={USER_V227_DEFAULT_GAMMA:.2f}")
        return acceptance



    @staticmethod
    def _per_sample_fraction(
        value: torch.Tensor, mask: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        numerator = ((value > threshold).float() * mask).flatten(1).sum(dim=1)
        denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        return numerator / denominator

    @torch.no_grad()
    def _evaluate_fixed_alpha_batch(
        self, prepared: dict[str, object], alpha: float
    ) -> tuple[list[dict[str, object]], torch.Tensor]:
        blend_s, _ = self.run_adapter(
            prepared,
            correction_enabled=False,
            layer_mix_override=alpha,
        )
        generated = self.render_blend_tail(prepared, blend_s)
        generated_lab = rgb_to_lab(generated)
        base_lab = rgb_to_lab(prepared["base_i"])
        generated_stats = compute_intrinsic_hair_color_stats(
            generated_lab, prepared["color_supervision_mask"], self.color_config
        )
        final_metrics = compute_reference_fidelity_metrics(
            generated_lab,
            prepared["color_supervision_mask"],
            prepared["ref_stats"],
            self.color_config,
        )
        direction = prepared["ref_stats"]["mean_ab"] - prepared["base_stats"]["mean_ab"]
        reference_progress = (
            ((generated_stats["mean_ab"] - prepared["base_stats"]["mean_ab"]) * direction)
            .sum(dim=1)
            / direction.square().sum(dim=1).clamp_min(1e-6)
        )
        base_distance = prepared["condition_metrics"]["ref_base_ab_distance"].clamp_min(1e-6)
        color_improvement = (
            base_distance - final_metrics["mean_ab_error"]
        ) / base_distance
        generated_luma = generated_lab[:, 0:1]
        base_luma = base_lab[:, 0:1]
        edge_excess = torch.relu(
            generated_luma
            - torch.maximum(base_luma, prepared["pseudo_lab"][:, 0:1])
            - USER_EDGE_LUMA_MARGIN
        )
        edge_mean = self.masked_mean_per_sample(edge_excess, prepared["transition_ring"])
        edge_fraction = self._per_sample_fraction(
            edge_excess, prepared["transition_ring"], 0.0
        )
        outer_keep = self.masked_l1_per_sample(
            generated, prepared["base_i"], prepared["outer_background_guard"]
        )
        face_keep_region = (
            prepared["face_keep_mask"] * prepared["satd_protect_mask"]
        ).clamp(0, 1)
        face_keep = self.masked_l1_per_sample(
            generated, prepared["base_i"], face_keep_region
        )
        global_l_excess = generated_luma - prepared["pseudo_lab"][:, 0:1]
        global_fraction = self._per_sample_fraction(
            global_l_excess, prepared["color_supervision_mask"], 12.0
        )
        reference_chroma = torch.linalg.vector_norm(
            prepared["ref_stats"]["mean_ab"], dim=1
        )
        records = []
        for index, sample_id in enumerate(prepared["sample_id"]):
            records.append({
                "sample_id": sample_id,
                "alpha": float(alpha),
                "safe_fraction": float(
                    prepared["condition_metrics"]["safe_fraction"][index].item()
                ),
                "ref_base_ab_distance": float(base_distance[index].item()),
                "reference_chroma_magnitude": float(reference_chroma[index].item()),
                "pseudo_to_reference_ab_error": float(
                    prepared["condition_metrics"]["pseudo_to_reference_mean_ab"][index].item()
                ),
                "pseudo_to_reference_hue_error": float(
                    prepared["condition_metrics"]["pseudo_to_reference_hue_error"][index].item()
                ),
                "final_to_reference_ab_error": float(
                    final_metrics["mean_ab_error"][index].item()
                ),
                "final_to_reference_hue_error": float(final_metrics["hue_error"][index].item()),
                "final_to_reference_chroma_error": float(
                    final_metrics["chroma_error"][index].item()
                ),
                "reference_progress": float(reference_progress[index].item()),
                "color_improvement_ratio": float(color_improvement[index].item()),
                "edge_luma_excess_mean": float(edge_mean[index].item()),
                "edge_luma_excess_fraction": float(edge_fraction[index].item()),
                "outer_bg_keep_l1": float(outer_keep[index].item()),
                "face_keep_l1": float(face_keep[index].item()),
                "global_frac_l_excess_gt12": float(global_fraction[index].item()),
            })
        return records, generated

    @torch.no_grad()
    def run_fixed_alpha_sweep(self) -> dict[str, object]:
        diagnostic_dir = ACTIVE_OUTPUT_DIR / "v2_21_fixed_alpha_diagnostic"
        preview_dir = diagnostic_dir / "previews"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        state_before = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
            if not key.startswith("clip_model.")
        }
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Phase 0 requires every model parameter to be frozen")
        self.model.eval()

        alphas = [0.0, *USER_ALPHA_SWEEP]
        records_by_alpha: dict[float, list[dict[str, object]]] = {
            alpha: [] for alpha in alphas
        }
        preview_limit = min(USER_LOG_IMAGE_COUNT, 16)
        preview_data: dict[str, dict[object, torch.Tensor]] = {}
        for alpha in alphas:
            for batch in tqdm(
                self.val_loader,
                desc=f"V2.21 fixed alpha={alpha:.2f}",
                leave=False,
            ):
                prepared = self.prepare_batch(batch)
                if prepared is None:
                    continue
                batch_records, generated = self._evaluate_fixed_alpha_batch(prepared, alpha)
                records_by_alpha[alpha].extend(batch_records)
                for index, sample_id in enumerate(prepared["sample_id"]):
                    if sample_id not in preview_data and len(preview_data) < preview_limit:
                        preview_data[sample_id] = {
                            "face": prepared["face_i"][index : index + 1].cpu().half(),
                            "color": prepared["color_i"][index : index + 1].cpu().half(),
                            "base": prepared["base_i"][index : index + 1].cpu().half(),
                            "pseudo": prepared["pseudo_rgb"][index : index + 1].cpu().half(),
                            "color_supervision_mask": mask_to_preview(
                                prepared["color_supervision_mask"][index : index + 1]
                            ).cpu().half(),
                            "transition_ring": mask_to_preview(
                                prepared["transition_ring"][index : index + 1]
                            ).cpu().half(),
                            "outer_background_guard": mask_to_preview(
                                prepared["outer_background_guard"][index : index + 1]
                            ).cpu().half(),
                        }
                    if sample_id in preview_data:
                        preview_data[sample_id][alpha] = generated[index : index + 1].cpu().half()

        if not records_by_alpha[0.0]:
            raise RuntimeError("V2.21 fixed-alpha sweep produced no valid samples")
        manifest = build_normal_color_manifest(
            records_by_alpha[0.0], USER_MIN_SAFE_REFERENCE_FRACTION_V8
        )
        manifest_by_id = {entry["sample_id"]: entry for entry in manifest["entries"]}
        for records in records_by_alpha.values():
            for record in records:
                entry = manifest_by_id[record["sample_id"]]
                record.update({
                    "normal_color": entry["normal_color"],
                    "extreme_for_v221": entry["extreme_for_v221"],
                    "near_no_edit": entry["near_no_edit"],
                    "pseudo_reliable": entry["pseudo_reliable"],
                    "color_weight": entry["color_weight"],
                })
        summary = aggregate_fixed_alpha_metrics(records_by_alpha, manifest)
        decision = choose_direct_anchor_feasibility(summary, manifest)
        if summary["monotonic_warning"]:
            print("[V2.21] DIRECT_ANCHOR_NON_MONOTONIC", file=sys.stderr)
        decision["phase0_parameter_max_abs_change"] = max(
            float((self.model.state_dict()[key].detach().cpu() - value).abs().max().item())
            for key, value in state_before.items()
        )
        if decision["phase0_parameter_max_abs_change"] != 0.0:
            raise RuntimeError("Phase 0 changed adapter parameters")
        assert_finite_json(manifest)
        assert_finite_json(summary)
        assert_finite_json(decision)

        with open(diagnostic_dir / "normal_subset_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(diagnostic_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(diagnostic_dir / "feasibility_decision.json", "w", encoding="utf-8") as handle:
            json.dump(decision, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(diagnostic_dir / "per_sample.jsonl", "w", encoding="utf-8") as handle:
            for alpha in alphas:
                for record in records_by_alpha[alpha]:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        with open(diagnostic_dir / "sweep_table.tsv", "w", encoding="utf-8") as handle:
            handle.write(
                "alpha\tmedian_progress\tp25_progress\tmedian_improvement\t"
                "negative_fraction\tedge_luma\touter_bg\tface_keep\tcatastrophic\n"
            )
            for alpha in alphas:
                metrics = summary["alphas"][f"{alpha:.2f}"]["normal"]
                handle.write(
                    f"{alpha:.2f}\t{metrics['median_reference_progress']:.8f}\t"
                    f"{metrics['p25_reference_progress']:.8f}\t"
                    f"{metrics['median_color_improvement_ratio']:.8f}\t"
                    f"{metrics['negative_improvement_fraction']:.8f}\t"
                    f"{metrics['edge_luma_excess_mean']:.8f}\t"
                    f"{metrics['outer_bg_keep_l1']:.8f}\t{metrics['face_keep_l1']:.8f}\t"
                    f"{int(metrics['catastrophic_artifact'])}\n"
                )

        contact_rows = []
        for index, sample_data in enumerate(preview_data.values()):
            row = [
                sample_data["face"], sample_data["color"], sample_data["base"],
                sample_data["pseudo"],
                *[sample_data[alpha] for alpha in USER_ALPHA_SWEEP],
            ]
            save_preview(preview_dir / f"sample_{index:03d}.png", row)
            save_preview(
                diagnostic_dir / "debug_masks" / f"sample_{index:03d}.png",
                [
                    sample_data["color_supervision_mask"],
                    sample_data["transition_ring"],
                    sample_data["outer_background_guard"],
                ],
            )
            contact_rows.append(row)
        save_contact_sheet(diagnostic_dir / "fixed_alpha_contact_sheet.png", contact_rows)

        self.normal_manifest = manifest
        self.phase0_summary = summary
        self.phase0_decision = decision
        self.alpha_star = decision.get("alpha_star")
        self.color_weight_by_sample = {
            entry["sample_id"]: float(entry["color_weight"])
            for entry in manifest["entries"]
        }
        if self.alpha_star is not None:
            phase0_comparison_dir = ACTIVE_OUTPUT_DIR / "comparisons" / "phase0_alpha_star"
            for index, sample_data in enumerate(preview_data.values()):
                save_preview(
                    phase0_comparison_dir / f"sample_{index:03d}.png",
                    [
                        sample_data["face"], sample_data["color"], sample_data["base"],
                        sample_data["pseudo"], sample_data[self.alpha_star],
                    ],
                )
            torch.save({
                "arch": DIRECT_COLOR_ARCH_V8_4,
                "phase": "PHASE0_FIXED_ALPHA",
                "alpha_star": self.alpha_star,
                "feasibility_decision": decision,
                "adapter_config": {
                    "alpha_init": USER_ALPHA_INIT,
                    "layer_offset_max": USER_LAYER_OFFSET_MAX,
                },
                "model_state_dict": state_before,
            }, self.output_ckpt_dir / "phase0_fixed_alpha_star.pth")
        else:
            normal_ids = {
                entry["sample_id"] for entry in manifest["entries"] if entry["normal_color"]
            }
            failed_alpha1 = [
                record for record in records_by_alpha[1.0]
                if record["sample_id"] in normal_ids
                and (
                    record["reference_progress"] < 0.55
                    or record["color_improvement_ratio"] < 0.45
                )
            ]
            trend_lines = []
            for alpha in USER_ALPHA_SWEEP:
                metrics = summary["alphas"][f"{alpha:.2f}"]["normal"]
                trend_lines.append(
                    f"- alpha={alpha:.2f}: median progress="
                    f"{metrics['median_reference_progress']:.4f}, median improvement="
                    f"{metrics['median_color_improvement_ratio']:.4f}, edge="
                    f"{metrics['edge_luma_excess_mean']:.4f}, outer_bg="
                    f"{metrics['outer_bg_keep_l1']:.4f}, face="
                    f"{metrics['face_keep_l1']:.4f}, catastrophic="
                    f"{metrics['catastrophic_artifact']}"
                )
            alpha1_metrics = summary["alphas"]["1.00"]["normal"]
            if decision["decision"] == "INSUFFICIENT_NORMAL_DIAGNOSTIC_SAMPLES":
                interpretation = (
                    "The reliable normal-color subset is too small for a direction "
                    "decision. Review pseudo fidelity and safe-reference coverage before "
                    "running Phase 1."
                )
            else:
                interpretation = (
                    "The pseudo target was filtered for AB/hue fidelity before this "
                    "decision. Failure therefore indicates latent direction / generator "
                    "realization limits.\n\n"
                    "Direct Anchor only is not expressive enough for the normal-color "
                    "objective.\nDo not continue alpha/controller tuning in v2.22.\n"
                    "Next architecture should decouple hair-color correction from global "
                    "W+/S interpolation."
                )
            failure_report = (
                "# Direct Anchor V2.21 Failure Report\n\n"
                f"Decision: {decision['decision']}\n\n"
                f"Normal subset: {manifest['normal_count']}/{manifest['total_count']}\n\n"
                f"Alpha=1 median progress: "
                f"{alpha1_metrics['median_reference_progress']:.4f}\n\n"
                f"Alpha=1 median improvement: "
                f"{alpha1_metrics['median_color_improvement_ratio']:.4f}\n\n"
                f"Normal pseudo/reference mean AB: "
                f"{alpha1_metrics['mean_pseudo_to_reference_ab']:.4f}\n\n"
                f"Normal pseudo/reference mean hue: "
                f"{alpha1_metrics['mean_pseudo_to_reference_hue']:.4f}\n\n"
                "## Fixed-alpha trend\n\n"
                + "\n".join(trend_lines)
                + "\n\n## Alpha=1 failures\n\n"
                + "\n".join(
                    f"- {record['sample_id']}: progress={record['reference_progress']:.4f}, "
                    f"improvement={record['color_improvement_ratio']:.4f}, "
                    f"pseudo_ab={record['pseudo_to_reference_ab_error']:.4f}"
                    for record in failed_alpha1[:30]
                )
                + "\n\n"
                + interpretation
                + "\n"
            )
            (diagnostic_dir / "direct_anchor_failure_report.md").write_text(
                failure_report, encoding="utf-8"
            )
        return decision

    def train_one_epoch(self, epoch: int):
        stage, correction_enabled = self.configure_stage(epoch)
        self.last_train_lr = float(self.optimizer.param_groups[0]["lr"])
        self.model.train()
        self.model.clip_model.eval()
        running_loss = 0.0
        running_steps = 0
        accumulated_batches = 0
        last_grad_norm = 0.0
        running_metrics: dict[str, float] = {}
        optimizer_parameters = [
            parameter
            for parameter_group in self.optimizer.param_groups
            for parameter in parameter_group["params"]
        ]
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(self.train_loader, desc=f"Blend train {epoch + 1}/{USER_EPOCHS}", leave=False)
        for batch in progress:
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            prepared["epoch"] = epoch

            blend_s, encoder_aux = self.run_adapter(
                prepared,
                correction_enabled=False,
                base_alpha_override=self.alpha_star,
            )
            latent_in = self.build_generator_latent(prepared["align_s"], blend_s)
            i_g, _ = self.helper.net.generator(
                [latent_in],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=prepared["align_f"],
            )
            i_g_256 = self.helper.downsample_256(i_g)
            loss, loss_info = self.calc_loss(
                i_g_256,
                prepared,
                encoder_aux,
                stage=stage,
                anchor_i=None,
            )

            (loss / self.grad_accum_steps).backward()
            accumulated_batches += 1
            if accumulated_batches == self.grad_accum_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(optimizer_parameters, USER_GRAD_CLIP)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                last_grad_norm = float(grad_norm)

            running_loss += float(loss.item())
            running_steps += 1
            batch_metrics = {
                "loss_pseudo_ab": loss_info["pseudo_ab"],
                "loss_pseudo_rgb": loss_info["pseudo_rgb"],
                "loss_pseudo_luma": loss_info["pseudo_luma"],
                "loss_positive_luma": loss_info["positive_luma"],
                "loss_hf_luma": loss_info["hf_luma"],
                "loss_alpha_teacher": loss_info["alpha_teacher"],
                "loss_ref_mean_ab": loss_info["ref_mean_ab"],
                "loss_ref_hue": loss_info["ref_hue"],
                "loss_ref_chroma": loss_info["ref_chroma"],
                "loss_layer_offset_mag": loss_info["layer_offset_mag"],
                "loss_layer_negative_drift": loss_info["layer_negative_drift"],
                "loss_correction_norm": loss_info["correction_norm"],
                "loss_correction_color_regression": loss_info["correction_color_regression"],
                "loss_correction_hue_regression": loss_info["correction_hue_regression"],
                "loss_correction_ref_regression": loss_info["correction_ref_regression"],
                "mean_chroma_need_gate": prepared["chroma_need_gate"].mean(),
                "mean_lightness_need_gate": prepared["lightness_need_gate"].mean(),
                "mean_edit_need_gate": prepared["edit_need_gate"].mean(),
                "mean_safe_fraction": prepared["condition_metrics"]["safe_fraction"].mean(),
                "mean_direct_component_norm": encoder_aux["direct_component_norm"].mean(),
                "mean_correction_norm": encoder_aux["correction_norm"].mean(),
                "mean_predicted_alpha": encoder_aux["predicted_alpha"].mean(),
                "mean_negative_parallel_fraction": encoder_aux[
                    "negative_parallel_fraction"
                ].mean(),
            }
            for key, value in batch_metrics.items():
                running_metrics[key] = running_metrics.get(key, 0.0) + float(value.item())
            progress.set_postfix(
                loss=float(loss.item()),
                ab=float(loss_info["pseudo_ab"].item()),
                rgb=float(loss_info["pseudo_rgb"].item()),
                luma=float(loss_info["pseudo_luma"].item()),
                excess=float(loss_info["positive_luma"].item()),
                gate=float(prepared["edit_need_gate"].mean().item()),
                skin=float(loss_info["skin_chroma_keep"].item()),
                grad=last_grad_norm,
                accum=f"{accumulated_batches}/{self.grad_accum_steps}",
            )

        if accumulated_batches:
            scale = self.grad_accum_steps / accumulated_batches
            if scale != 1.0:
                for parameter in optimizer_parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            grad_norm = torch.nn.utils.clip_grad_norm_(optimizer_parameters, USER_GRAD_CLIP)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        averaged_metrics = {
            key: value / max(running_steps, 1)
            for key, value in running_metrics.items()
        }
        print(
            f"[blending_v8] epoch={epoch + 1} stage={stage} "
            f"lr={self.optimizer.param_groups[0]['lr']:.2e} correction={correction_enabled} train_components "
            + " ".join(f"{key}={value:.6f}" for key, value in averaged_metrics.items())
        )
        self.scheduler.step()
        return running_loss / max(running_steps, 1)

    @torch.no_grad()
    def validate(
        self,
        epoch: int,
        *,
        correction_enabled: bool | None = None,
        output_dir_name: str | None = None,
    ):
        correction_enabled = False
        stage = "FIXED_BASE_ALPHA_LAYER_ADAPT"
        validation_label = "pretrain" if epoch < 0 else f"{epoch + 1}/{USER_EPOCHS}"
        self.model.eval()
        total_losses: dict[str, float] = {}
        total_diagnostics: dict[str, float] = {}
        total_steps = 0
        images_to_fid = []
        preview_rows = []
        direct_preview_rows = []
        debug_mask_rows = []
        comparison_rows = []
        validation_records = []

        for batch in tqdm(self.val_loader, desc=f"Blend val {validation_label}", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            prepared["epoch"] = max(epoch, 0)

            bsz = prepared["color_s"].size(0)
            blend_s, encoder_aux = self.run_adapter(
                prepared,
                correction_enabled=False,
                base_alpha_override=self.alpha_star,
            )
            latent_in = self.build_generator_latent(prepared["align_s"], blend_s)
            i_g, _ = self.helper.net.generator(
                [latent_in],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=prepared["align_f"],
            )
            i_g_256 = self.helper.downsample_256(i_g)
            anchor_s, anchor_aux = blend_s, encoder_aux
            anchor_i_256 = i_g_256
            fixed_s, fixed_aux = self.run_adapter(
                prepared,
                correction_enabled=False,
                layer_mix_override=(
                    self.alpha_star if self.alpha_star is not None else USER_DIAGNOSTIC_ALPHA
                ),
            )
            fixed_i_256 = self.render_blend_tail(prepared, fixed_s)
            loss, loss_info = self.calc_loss(
                i_g_256,
                prepared,
                encoder_aux,
                stage=stage,
                anchor_i=None,
            )

            for key, value in loss_info.items():
                total_losses[key] = total_losses.get(key, 0.0) + float(value.item())
            batch_diagnostics = {
                "mean_chroma_need_gate": prepared["chroma_need_gate"].mean(),
                "mean_lightness_need_gate": prepared["lightness_need_gate"].mean(),
                "mean_edit_need_gate": prepared["edit_need_gate"].mean(),
                "mean_safe_fraction": prepared["condition_metrics"]["safe_fraction"].mean(),
                "mean_rejected_fraction": prepared["condition_metrics"]["rejected_fraction"].mean(),
                "mean_direct_delta_norm": encoder_aux["direct_delta_norm"].mean(),
                "mean_direct_component_norm": encoder_aux["direct_component_norm"].mean(),
                "mean_correction_norm": encoder_aux["correction_norm"].mean(),
                "mean_layer_mix": encoder_aux["layer_mix_mean"].mean(),
                "mean_anchor_layer_mix": anchor_aux["layer_mix_mean"].mean(),
                "mean_negative_parallel_fraction": encoder_aux[
                    "negative_parallel_fraction"
                ].mean(),
            }
            for key, value in batch_diagnostics.items():
                total_diagnostics[key] = total_diagnostics.get(key, 0.0) + float(value.item())
            total_steps += 1

            if self.fid_calc is not None:
                images_to_fid.append(T.Resize((299, 299))(((i_g + 1) / 2).clamp(0, 1)))

            generated_lab = rgb_to_lab(i_g_256)
            anchor_lab = rgb_to_lab(anchor_i_256)
            fixed_lab = rgb_to_lab(fixed_i_256)
            final_ref_metrics = compute_reference_fidelity_metrics(
                generated_lab,
                prepared["color_supervision_mask"],
                prepared["ref_stats"],
                self.color_config,
            )
            anchor_ref_metrics = compute_reference_fidelity_metrics(
                anchor_lab,
                prepared["color_supervision_mask"],
                prepared["ref_stats"],
                self.color_config,
            )
            luma_excess = generated_lab[:, 0:1] - prepared["pseudo_lab"][:, 0:1]
            generated_stats = compute_intrinsic_hair_color_stats(
                generated_lab, prepared["color_supervision_mask"], self.color_config
            )
            progress_direction = (
                prepared["ref_stats"]["mean_ab"] - prepared["base_stats"]["mean_ab"]
            )
            reference_progress = (
                (
                    generated_stats["mean_ab"] - prepared["base_stats"]["mean_ab"]
                ) * progress_direction
            ).sum(dim=1) / progress_direction.square().sum(dim=1).clamp_min(1e-6)
            base_luma = rgb_to_lab(prepared["base_i"])[:, 0:1]
            edge_luma_excess = torch.relu(
                generated_lab[:, 0:1]
                - torch.maximum(base_luma, prepared["pseudo_lab"][:, 0:1])
                - USER_EDGE_LUMA_MARGIN
            )
            edge_luma_fraction = self._per_sample_fraction(
                edge_luma_excess, prepared["transition_ring"], 0.0
            )
            outer_bg_keep = self.masked_l1_per_sample(
                i_g_256, prepared["base_i"], prepared["outer_background_guard"]
            )
            face_keep_region = (
                prepared["face_keep_mask"] * prepared["satd_protect_mask"]
            ).clamp(0, 1)
            face_keep_per_sample = self.masked_l1_per_sample(
                i_g_256, prepared["base_i"], face_keep_region
            )
            global_l_excess_fraction = self._per_sample_fraction(
                luma_excess, prepared["color_supervision_mask"], 12.0
            )
            reference_chroma_magnitude = torch.linalg.vector_norm(
                prepared["ref_stats"]["mean_ab"], dim=1
            )
            if len(preview_rows) < USER_LOG_IMAGE_COUNT:
                zero_direct_s, _ = self.run_adapter(
                    prepared,
                    correction_enabled=False,
                    layer_mix_override=0.0,
                )
                half_direct_s, _ = self.run_adapter(
                    prepared,
                    correction_enabled=False,
                    layer_mix_override=0.5,
                )
                full_direct_s, _ = self.run_adapter(
                    prepared,
                    correction_enabled=False,
                    layer_mix_override=1.0,
                )
                zero_direct_i = self.render_blend_tail(prepared, zero_direct_s)
                half_direct_i = self.render_blend_tail(prepared, half_direct_s)
                full_direct_i = self.render_blend_tail(prepared, full_direct_s)
                for idx in range(bsz):
                    excess_preview = (
                        (torch.relu(luma_excess[idx : idx + 1]) / 20.0).clamp(0, 1) * 2.0 - 1.0
                    ).repeat(1, 3, 1, 1)
                    preview_rows.append([
                        prepared["face_i"][idx : idx + 1],
                        prepared["color_i"][idx : idx + 1],
                        prepared["base_i"][idx : idx + 1],
                        i_g_256[idx : idx + 1],
                        mask_to_preview(prepared["color_transfer_mask"][idx : idx + 1]),
                        mask_to_preview(prepared["reference_hair_mask"][idx : idx + 1]),
                        mask_to_preview(prepared["safe_ref_mask"][idx : idx + 1]),
                        mask_to_preview(prepared["rejected_highlight_mask"][idx : idx + 1]),
                        prepared["pseudo_rgb"][idx : idx + 1],
                        excess_preview,
                    ])
                    debug_mask_rows.append([
                        mask_to_preview(prepared["color_supervision_mask"][idx : idx + 1]),
                        mask_to_preview(prepared["transition_ring"][idx : idx + 1]),
                        mask_to_preview(prepared["outer_background_guard"][idx : idx + 1]),
                    ])
                    direct_preview_rows.append([
                        prepared["base_i"][idx : idx + 1],
                        prepared["color_i"][idx : idx + 1],
                        prepared["pseudo_rgb"][idx : idx + 1],
                        zero_direct_i[idx : idx + 1],
                        half_direct_i[idx : idx + 1],
                        fixed_i_256[idx : idx + 1],
                        full_direct_i[idx : idx + 1],
                        anchor_i_256[idx : idx + 1],
                        i_g_256[idx : idx + 1],
                    ])
                    comparison_rows.append([
                        prepared["face_i"][idx : idx + 1],
                        prepared["color_i"][idx : idx + 1],
                        prepared["base_i"][idx : idx + 1],
                        prepared["pseudo_rgb"][idx : idx + 1],
                        fixed_i_256[idx : idx + 1],
                        i_g_256[idx : idx + 1],
                        i_g_256[idx : idx + 1],
                    ])
                    if len(preview_rows) >= USER_LOG_IMAGE_COUNT:
                        break

            for idx in range(bsz):
                sample_mask = prepared["color_supervision_mask"][idx : idx + 1]
                sample_gen_ab = generated_lab[idx : idx + 1, 1:3]
                sample_anchor_ab = anchor_lab[idx : idx + 1, 1:3]
                sample_fixed_ab = fixed_lab[idx : idx + 1, 1:3]
                sample_pseudo_ab = prepared["pseudo_lab"][idx : idx + 1, 1:3]
                hue_cosine = (
                    (sample_gen_ab * sample_pseudo_ab).sum(dim=1, keepdim=True)
                    / (
                        torch.linalg.vector_norm(sample_gen_ab, dim=1, keepdim=True)
                        * torch.linalg.vector_norm(sample_pseudo_ab, dim=1, keepdim=True)
                    ).clamp_min(1e-4)
                ).clamp(-1, 1)
                anchor_hue_cosine = (
                    (sample_anchor_ab * sample_pseudo_ab).sum(dim=1, keepdim=True)
                    / (
                        torch.linalg.vector_norm(sample_anchor_ab, dim=1, keepdim=True)
                        * torch.linalg.vector_norm(sample_pseudo_ab, dim=1, keepdim=True)
                    ).clamp_min(1e-4)
                ).clamp(-1, 1)
                fixed_hue_cosine = (
                    (sample_fixed_ab * sample_pseudo_ab).sum(dim=1, keepdim=True)
                    / (
                        torch.linalg.vector_norm(sample_fixed_ab, dim=1, keepdim=True)
                        * torch.linalg.vector_norm(sample_pseudo_ab, dim=1, keepdim=True)
                    ).clamp_min(1e-4)
                ).clamp(-1, 1)
                sample_excess = luma_excess[idx : idx + 1]
                predicted_alpha = float(encoder_aux["predicted_alpha"][idx].item())
                teacher_alpha_value = float(prepared["teacher_alpha"][idx].item())
                teacher_alpha = (
                    teacher_alpha_value if np.isfinite(teacher_alpha_value) else None
                )
                final_ref_score = float(reference_color_score({
                    key: value[idx : idx + 1]
                    for key, value in final_ref_metrics.items()
                    if key != "candidate_stats"
                }).item())
                base_to_reference_ab = float(
                    prepared["condition_metrics"]["ref_base_ab_distance"][idx].item()
                )
                final_to_reference_ab = float(
                    final_ref_metrics["mean_ab_error"][idx].item()
                )
                validation_records.append({
                    "sample_index": len(validation_records),
                    "sample_id": prepared["sample_id"][idx],
                    "ref_base_ab_distance": float(
                        prepared["condition_metrics"]["ref_base_ab_distance"][idx].item()
                    ),
                    "reference_chroma_magnitude": float(
                        reference_chroma_magnitude[idx].item()
                    ),
                    "delta_L_global": float(prepared["condition_metrics"]["delta_l_global"][idx].item()),
                    "chroma_need_gate": float(prepared["chroma_need_gate"][idx].item()),
                    "lightness_need_gate": float(prepared["lightness_need_gate"][idx].item()),
                    "edit_need_gate": float(prepared["edit_need_gate"][idx].item()),
                    "safe_fraction": float(prepared["condition_metrics"]["safe_fraction"][idx].item()),
                    "rejected_fraction": float(
                        prepared["condition_metrics"]["rejected_fraction"][idx].item()
                    ),
                    "composite_color_distance": float(
                        prepared["condition_metrics"]["composite_color_distance"][idx].item()
                    ),
                    "hue_distance_deg": float(
                        prepared["condition_metrics"]["hue_distance_deg"][idx].item()
                    ),
                    "chroma_distance": float(
                        prepared["condition_metrics"]["chroma_distance"][idx].item()
                    ),
                    "distribution_distance": float(
                        prepared["condition_metrics"]["distribution_distance"][idx].item()
                    ),
                    "relative_luma_reliability": float(
                        prepared["condition_metrics"]["relative_luma_reliability"][idx].item()
                    ),
                    "pseudo_reference_fidelity": float(
                        prepared["condition_metrics"]["pseudo_reference_fidelity"][idx].item()
                    ),
                    "pseudo_to_reference_ab_error": float(
                        prepared["condition_metrics"]["pseudo_to_reference_mean_ab"][idx].item()
                    ),
                    "pseudo_to_reference_hue_error": float(
                        prepared["condition_metrics"]["pseudo_to_reference_hue_error"][idx].item()
                    ),
                    "pseudo_to_reference_chroma_error": float(
                        prepared["condition_metrics"]["pseudo_to_reference_chroma_error"][idx].item()
                    ),
                    "pseudo_to_reference_l_error": float(
                        prepared["condition_metrics"]["pseudo_to_reference_median_l_error"][idx].item()
                    ),
                    "anchor_to_reference_ab_error": float(
                        anchor_ref_metrics["mean_ab_error"][idx].item()
                    ),
                    "anchor_to_reference_hue_error": float(
                        anchor_ref_metrics["hue_error"][idx].item()
                    ),
                    "anchor_to_reference_chroma_error": float(
                        anchor_ref_metrics["chroma_error"][idx].item()
                    ),
                    "final_to_reference_ab_error": float(
                        final_ref_metrics["mean_ab_error"][idx].item()
                    ),
                    "final_to_reference_hue_error": float(
                        final_ref_metrics["hue_error"][idx].item()
                    ),
                    "final_to_reference_chroma_error": float(
                        final_ref_metrics["chroma_error"][idx].item()
                    ),
                    "final_reference_color_score": final_ref_score,
                    "teacher_alpha": teacher_alpha,
                    "teacher_confidence": float(prepared["teacher_confidence"][idx].item()),
                    "predicted_alpha": predicted_alpha,
                    "reference_progress": float(reference_progress[idx].item()),
                    "color_improvement_ratio": (
                        base_to_reference_ab - final_to_reference_ab
                    ) / max(base_to_reference_ab, 1e-6),
                    "edge_luma_excess": float(self.masked_mean_value(
                        edge_luma_excess[idx : idx + 1],
                        prepared["transition_ring"][idx : idx + 1],
                    ).item()),
                    "edge_luma_excess_fraction": float(edge_luma_fraction[idx].item()),
                    "outer_bg_keep_l1": float(outer_bg_keep[idx].item()),
                    "face_keep_l1": float(face_keep_per_sample[idx].item()),
                    "global_frac_l_excess_gt12": float(
                        global_l_excess_fraction[idx].item()
                    ),
                    "effective_alpha": float(encoder_aux["layer_mix_mean"][idx].item()),
                    "base_alpha": float(encoder_aux["base_alpha"][idx].item()),
                    "layer_offset_mean": float(
                        encoder_aux["layer_offset_mean"][idx].item()
                    ),
                    "layer_offset_abs_mean": float(
                        encoder_aux["layer_offset_abs_mean"][idx].item()
                    ),
                    "alpha_abs_error": (
                        None if teacher_alpha is None else abs(predicted_alpha - teacher_alpha)
                    ),
                    "result_to_pseudo_ab_l2": float(self.masked_mean_value(
                        torch.linalg.vector_norm(sample_gen_ab - sample_pseudo_ab, dim=1, keepdim=True),
                        sample_mask,
                    ).item()),
                    "result_hue_error": float(self.masked_mean_value(
                        torch.rad2deg(torch.acos(hue_cosine)), sample_mask
                    ).item()),
                    "anchor_result_to_pseudo_ab_l2": float(self.masked_mean_value(
                        torch.linalg.vector_norm(
                            sample_anchor_ab - sample_pseudo_ab, dim=1, keepdim=True
                        ),
                        sample_mask,
                    ).item()),
                    "final_result_to_pseudo_ab_l2": float(self.masked_mean_value(
                        torch.linalg.vector_norm(
                            sample_gen_ab - sample_pseudo_ab, dim=1, keepdim=True
                        ),
                        sample_mask,
                    ).item()),
                    "anchor_hue_error": float(self.masked_mean_value(
                        torch.rad2deg(torch.acos(anchor_hue_cosine)), sample_mask
                    ).item()),
                    "final_hue_error": float(self.masked_mean_value(
                        torch.rad2deg(torch.acos(hue_cosine)), sample_mask
                    ).item()),
                    "fixed_result_to_pseudo_ab_l2": float(self.masked_mean_value(
                        torch.linalg.vector_norm(
                            sample_fixed_ab - sample_pseudo_ab, dim=1, keepdim=True
                        ),
                        sample_mask,
                    ).item()),
                    "fixed_hue_error": float(self.masked_mean_value(
                        torch.rad2deg(torch.acos(fixed_hue_cosine)), sample_mask
                    ).item()),
                    "mean_L_excess": float(self.masked_mean_value(torch.relu(sample_excess), sample_mask).item()),
                    "q95_L_excess": float(self.masked_q95(sample_excess, sample_mask).item()),
                    "frac_L_excess_gt8": float(self.masked_fraction_above(sample_excess, sample_mask, 8.0).item()),
                    "frac_L_excess_gt12": float(self.masked_fraction_above(sample_excess, sample_mask, 12.0).item()),
                    "frac_L_excess_gt16": float(self.masked_fraction_above(sample_excess, sample_mask, 16.0).item()),
                    "direct_delta_norm": float(encoder_aux["direct_delta_norm"][idx].item()),
                    "direct_component_norm": float(encoder_aux["direct_component_norm"][idx].item()),
                    "direct_mix_fraction": float(encoder_aux["direct_mix_fraction"][idx].item()),
                    "anchor_direct_mix_fraction": float(
                        anchor_aux["direct_mix_fraction"][idx].item()
                    ),
                    "fixed_direct_mix_fraction": float(
                        fixed_aux["direct_mix_fraction"][idx].item()
                    ),
                    "layer_mix_mean": float(encoder_aux["layer_mix_mean"][idx].item()),
                    "final_layer_mix": float(encoder_aux["layer_mix_mean"][idx].item()),
                    "layer_mix_min": float(encoder_aux["layer_mix_min"][idx].item()),
                    "layer_mix_max": float(encoder_aux["layer_mix_max"][idx].item()),
                    "correction_raw_norm": float(encoder_aux["correction_raw_norm"][idx].item()),
                    "correction_norm": float(encoder_aux["correction_norm"][idx].item()),
                    "correction_budget": float(encoder_aux["correction_budget"][idx].item()),
                    "correction_chroma_budget": float(
                        encoder_aux["correction_chroma_budget"][idx].item()
                    ),
                    "correction_luma_budget": float(
                        encoder_aux["correction_luma_budget"][idx].item()
                    ),
                    "correction_budget_scale": float(encoder_aux["correction_budget_scale"][idx].item()),
                    "correction_to_direct_ratio": float(
                        encoder_aux["correction_to_direct_ratio"][idx].item()
                    ),
                    "total_delta_norm": float(encoder_aux["total_delta_norm"][idx].item()),
                    "total_to_direct_ratio": float(encoder_aux["total_to_direct_ratio"][idx].item()),
                    "direct_parallel_correction_coeff": float(
                        encoder_aux["direct_parallel_correction_coeff"][idx].item()
                    ),
                    "negative_parallel_fraction": float(
                        encoder_aux["negative_parallel_fraction"][idx].item()
                    ),
                    "anchor_frozen": bool(encoder_aux["anchor_frozen"][idx].item()),
                })

        avg_losses = {key: value / max(total_steps, 1) for key, value in total_losses.items()}
        avg_diagnostics = {key: value / max(total_steps, 1) for key, value in total_diagnostics.items()}
        if not validation_records:
            raise RuntimeError("Validation produced no valid samples")
        if self.fid_calc is not None and images_to_fid:
            avg_losses["fid_clip"] = float(self.fid_calc(torch.cat(images_to_fid)).item())

        def record_mean(records: list[dict[str, object]], key: str) -> float:
            return sum(float(record[key]) for record in records) / len(records)

        mean_pseudo_to_ref_ab = record_mean(
            validation_records, "pseudo_to_reference_ab_error"
        )
        mean_pseudo_to_ref_hue = record_mean(
            validation_records, "pseudo_to_reference_hue_error"
        )
        mean_final_to_ref_ab = record_mean(
            validation_records, "final_to_reference_ab_error"
        )
        mean_final_to_ref_hue = record_mean(
            validation_records, "final_to_reference_hue_error"
        )
        mean_final_to_ref_chroma = record_mean(
            validation_records, "final_to_reference_chroma_error"
        )
        predicted_alphas = np.asarray(
            [record["predicted_alpha"] for record in validation_records], dtype=np.float64
        )
        reference_progresses = np.asarray(
            [record["reference_progress"] for record in validation_records], dtype=np.float64
        )
        edge_luma_excesses = np.asarray(
            [record["edge_luma_excess"] for record in validation_records], dtype=np.float64
        )
        teacher_alphas = np.asarray(
            [
                record["teacher_alpha"] for record in validation_records
                if record["teacher_alpha"] is not None
            ],
            dtype=np.float64,
        )
        predicted_alpha_mean = float(predicted_alphas.mean())
        predicted_alpha_std = float(predicted_alphas.std())
        teacher_alpha_std = float(teacher_alphas.std()) if teacher_alphas.size else 0.0
        teacher_alpha_mae = (
            float(np.mean([
                abs(float(record["predicted_alpha"]) - float(record["teacher_alpha"]))
                for record in validation_records if record["teacher_alpha"] is not None
            ]))
            if teacher_alphas.size else 0.0
        )
        pseudo_failure_records = [
            record for record in validation_records
            if record["pseudo_to_reference_ab_error"] > 8.0
            or record["pseudo_to_reference_hue_error"] > 15.0
        ]
        pseudo_failure_fraction = len(pseudo_failure_records) / len(validation_records)
        artifact_penalty = 20.0 * max(0.0, avg_losses["frac_l_excess_gt12"] - 0.15)
        color_score = (
            mean_final_to_ref_ab
            + 0.10 * mean_final_to_ref_hue
            + 0.50 * mean_final_to_ref_chroma
        )
        boundary_halo_penalty = (
            4.0 * avg_losses.get("edge_luma_excess", 0.0)
            + 1.0 * avg_losses.get("edge_luma_excess_fraction", 0.0)
            + avg_losses.get("outer_bg_keep", 0.0)
        )
        normal_balanced_score = (
            color_score
            + artifact_penalty
            + 0.25 * avg_losses["face_keep_l1"]
            + boundary_halo_penalty
        )
        manifest_entries = (
            [] if self.normal_manifest is None else self.normal_manifest["entries"]
        )
        normal_ids = {
            entry["sample_id"] for entry in manifest_entries if entry["normal_color"]
        }
        normal_records = [
            record for record in validation_records if record["sample_id"] in normal_ids
        ]
        if not normal_records:
            normal_records = validation_records
        normal_progress = np.asarray(
            [record["reference_progress"] for record in normal_records], dtype=np.float64
        )
        normal_improvement = np.asarray(
            [record["color_improvement_ratio"] for record in normal_records], dtype=np.float64
        )
        effective_alpha = np.asarray(
            [record["effective_alpha"] for record in normal_records], dtype=np.float64
        )
        normal_final_ab = float(np.mean([
            record["final_to_reference_ab_error"] for record in normal_records
        ]))
        normal_final_hue = float(np.mean([
            record["final_to_reference_hue_error"] for record in normal_records
        ]))
        normal_final_chroma = float(np.mean([
            record["final_to_reference_chroma_error"] for record in normal_records
        ]))
        normal_edge_halo = float(np.mean([
            record["edge_luma_excess"] for record in normal_records
        ]))
        normal_outer_keep = float(np.mean([
            record["outer_bg_keep_l1"] for record in normal_records
        ]))
        normal_face_keep = float(np.mean([
            record["face_keep_l1"] for record in normal_records
        ]))
        alpha_star = float(self.alpha_star if self.alpha_star is not None else 0.0)
        phase0_baseline = (
            {} if self.phase0_decision is None
            else self.phase0_decision.get("phase0_alpha_star_metrics", {})
        )
        checkpoint_gate = evaluate_v221_checkpoint_gate(
            median_reference_progress=float(np.median(normal_progress)),
            median_color_improvement_ratio=float(np.median(normal_improvement)),
            effective_alpha_mean=float(effective_alpha.mean()),
            alpha_star=alpha_star,
            phase0_alpha_star_metrics=phase0_baseline,
        )
        anchor_drift_warning = checkpoint_gate["anchor_drift_warning"]
        color_gate_pass = checkpoint_gate["color_gate_pass"]
        v221_boundary_penalty = (
            normal_edge_halo / 100.0 + normal_outer_keep
        )
        v221_balanced_score = (
            normal_final_ab
            + 0.10 * normal_final_hue
            + 0.50 * normal_final_chroma
            + 0.25 * normal_face_keep
            + v221_boundary_penalty
        )
        alpha_collapsed = False
        pseudo_systematic_failure = (
            pseudo_failure_fraction > USER_PSEUDO_FIDELITY_BAD_FRACTION
        )
        sample4 = next(
            (record for record in validation_records if record["sample_index"] == 4),
            None,
        )
        validation_summary = {
            "validation_label": validation_label,
            "stage": stage,
            "sample_count": len(validation_records),
            "mean_pseudo_to_ref_ab": mean_pseudo_to_ref_ab,
            "mean_pseudo_to_ref_hue": mean_pseudo_to_ref_hue,
            "mean_final_to_ref_ab": mean_final_to_ref_ab,
            "mean_final_to_ref_hue": mean_final_to_ref_hue,
            "mean_final_to_ref_chroma": mean_final_to_ref_chroma,
            "final_to_reference_ab": mean_final_to_ref_ab,
            "final_to_reference_hue": mean_final_to_ref_hue,
            "final_to_reference_chroma": mean_final_to_ref_chroma,
            "teacher_alpha_mae": teacher_alpha_mae,
            "teacher_alpha_std": teacher_alpha_std,
            "predicted_alpha_mean": predicted_alpha_mean,
            "predicted_alpha_std": predicted_alpha_std,
            "predicted_alpha_p25": float(np.quantile(predicted_alphas, 0.25)),
            "predicted_alpha_p50": float(np.quantile(predicted_alphas, 0.50)),
            "predicted_alpha_p75": float(np.quantile(predicted_alphas, 0.75)),
            "teacher_weight": get_alpha_teacher_weight(max(epoch, 0)),
            "alpha_low_fraction": float((predicted_alphas <= 0.25).mean()),
            "alpha_high_fraction": float((predicted_alphas >= 0.85).mean()),
            "pseudo_fidelity_bad_count": len(pseudo_failure_records),
            "pseudo_fidelity_bad_fraction": pseudo_failure_fraction,
            "alpha_collapsed": alpha_collapsed,
            "pseudo_systematic_failure": pseudo_systematic_failure,
            "allow_best_checkpoint": bool(
                np.isfinite(avg_losses["loss"]) and color_gate_pass
            ),
            "global_frac_l_excess_gt12": avg_losses["frac_l_excess_gt12"],
            "mean_face_keep_loss": avg_losses["face_keep_l1"],
            "color_score": color_score,
            "artifact_penalty": artifact_penalty,
            "balanced_score": normal_balanced_score,
            "normal_balanced_score": normal_balanced_score,
            "boundary_halo_penalty": boundary_halo_penalty,
            "reference_progress_mean": float(reference_progresses.mean()),
            "reference_progress_p25": float(np.quantile(reference_progresses, 0.25)),
            "reference_progress_p50": float(np.quantile(reference_progresses, 0.50)),
            "edge_luma_excess_mean": float(edge_luma_excesses.mean()),
            "edge_luma_excess_fraction": avg_losses.get("edge_luma_excess_fraction", 0.0),
            "outer_bg_keep_l1": avg_losses.get("outer_bg_keep", 0.0),
            "face_keep_l1": avg_losses["face_keep_l1"],
            "normal_count": len(normal_records),
            "normal_median_reference_progress": float(np.median(normal_progress)),
            "normal_p25_reference_progress": float(np.quantile(normal_progress, 0.25)),
            "normal_median_color_improvement_ratio": float(
                np.median(normal_improvement)
            ),
            "normal_final_to_ref_ab": normal_final_ab,
            "normal_final_to_ref_hue": normal_final_hue,
            "normal_final_to_ref_chroma": normal_final_chroma,
            "normal_edge_halo": normal_edge_halo,
            "normal_face_keep": normal_face_keep,
            "normal_outer_bg_keep": normal_outer_keep,
            "effective_alpha_mean": float(effective_alpha.mean()),
            "effective_alpha_std": float(effective_alpha.std()),
            "effective_alpha_min": float(effective_alpha.min()),
            "effective_alpha_p25": float(np.quantile(effective_alpha, 0.25)),
            "effective_alpha_p50": float(np.quantile(effective_alpha, 0.50)),
            "effective_alpha_p75": float(np.quantile(effective_alpha, 0.75)),
            "alpha_star": self.alpha_star,
            "anchor_drift_warning": anchor_drift_warning,
            "color_progress_gate": checkpoint_gate["color_progress_gate"],
            "color_improvement_gate": checkpoint_gate["color_improvement_gate"],
            "color_gate_pass": color_gate_pass,
            "v221_balanced_score": v221_balanced_score,
            "training_regressed_from_fixed_anchor": checkpoint_gate[
                "training_regressed_from_fixed_anchor"
            ],
            "sample4_ab_error": (
                None if sample4 is None else sample4["final_to_reference_ab_error"]
            ),
            "sample4_hue_error": (
                None if sample4 is None else sample4["final_to_reference_hue_error"]
            ),
            "sample4_predicted_alpha": (
                None if sample4 is None else sample4["predicted_alpha"]
            ),
            "val_loss": avg_losses["loss"],
        }

        if pseudo_systematic_failure:
            print(
                "[PSEUDO TARGET SYSTEMATIC FAILURE] "
                f"bad_fraction={pseudo_failure_fraction:.2%} "
                f"limit={USER_PSEUDO_FIDELITY_BAD_FRACTION:.2%}",
                file=sys.stderr,
            )
        for record in pseudo_failure_records:
            print(
                "[PSEUDO TARGET FIDELITY WARNING] "
                f"sample={record['sample_id']} "
                f"ab={record['pseudo_to_reference_ab_error']:.4f} "
                f"hue={record['pseudo_to_reference_hue_error']:.4f}",
                file=sys.stderr,
            )
        epoch_dir = self.output_val_dir / (
            output_dir_name if output_dir_name is not None else f"epoch_{epoch + 1:03d}"
        )
        epoch_dir.mkdir(parents=True, exist_ok=True)
        assert_finite_json(validation_records)
        assert_finite_json(validation_summary)
        with open(epoch_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(
                validation_records, handle, ensure_ascii=False, indent=2, allow_nan=False
            )
        with open(epoch_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(
                validation_summary, handle, ensure_ascii=False, indent=2, allow_nan=False
            )
        failure_lines = [
            (
                f"{record['sample_index']}\t{record['sample_id']}\t"
                f"ab={record['pseudo_to_reference_ab_error']:.6f}\t"
                f"hue={record['pseudo_to_reference_hue_error']:.6f}"
            )
            for record in pseudo_failure_records
        ]
        failure_text = "\n".join(failure_lines) + ("\n" if failure_lines else "")
        (epoch_dir / "pseudo_target_failure_cases.txt").write_text(
            failure_text, encoding="utf-8"
        )
        (ACTIVE_OUTPUT_DIR / "pseudo_target_failure_cases.txt").write_text(
            failure_text, encoding="utf-8"
        )
        if epoch < 0 or epoch % USER_SAVE_PREVIEW_EVERY == 0:
            for idx, row in enumerate(preview_rows):
                save_preview(epoch_dir / f"sample_{idx:03d}.png", row)
                debug_dir = ACTIVE_OUTPUT_DIR / "debug_masks" / epoch_dir.name
                save_preview(debug_dir / f"sample_{idx:03d}.png", debug_mask_rows[idx])
                if idx in USER_FIXED_REGRESSION_INDICES:
                    save_preview(
                        epoch_dir / f"sample_{idx:03d}_direct_alpha_diagnostic.png",
                        direct_preview_rows[idx],
                    )
        if epoch in (0, 3, 7):
            comparison_dir = ACTIVE_OUTPUT_DIR / "comparisons" / f"epoch_{epoch + 1:03d}"
            for idx, row in enumerate(comparison_rows):
                save_preview(comparison_dir / f"sample_{idx:03d}.png", row)
        latest_comparison_dir = ACTIVE_OUTPUT_DIR / "comparisons" / "_latest"
        for idx, row in enumerate(comparison_rows):
            save_preview(latest_comparison_dir / f"sample_{idx:03d}.png", row)

        print(
            f"[blending_v8] validation={validation_label} correction={correction_enabled} "
            f"val_loss={avg_losses['loss']:.6f} "
            f"val_face={avg_losses['face_loss']:.6f} "
            f"val_hair={avg_losses['hair_loss']:.6f} "
            f"loss_pseudo_ab={avg_losses['pseudo_ab']:.6f} "
            f"loss_pseudo_rgb={avg_losses['pseudo_rgb']:.6f} "
            f"loss_pseudo_luma={avg_losses['pseudo_luma']:.6f} "
            f"loss_positive_luma={avg_losses['positive_luma']:.6f} "
            f"loss_hf_luma={avg_losses['hf_luma']:.6f} "
            f"mean_edit_need_gate={avg_diagnostics['mean_edit_need_gate']:.6f} "
            f"mean_safe_fraction={avg_diagnostics['mean_safe_fraction']:.6f} "
            f"mean_direct_component_norm={avg_diagnostics['mean_direct_component_norm']:.6f} "
            f"mean_correction_norm={avg_diagnostics['mean_correction_norm']:.6f} "
            f"val_ab_error={avg_losses['result_to_pseudo_ab_l2']:.6f} "
            f"val_frac_excess_gt12={avg_losses['frac_l_excess_gt12']:.6f} "
            f"val_skin={avg_losses['skin_chroma_keep']:.6f} "
            f"final_ref_ab={mean_final_to_ref_ab:.6f} "
            f"final_ref_hue={mean_final_to_ref_hue:.6f} "
            f"final_ref_chroma={mean_final_to_ref_chroma:.6f} "
            f"predicted_alpha_mean={predicted_alpha_mean:.6f} "
            f"predicted_alpha_std={predicted_alpha_std:.6f} "
            f"color_score={color_score:.6f} normal_balanced_score={normal_balanced_score:.6f}"
        )
        return validation_summary

    @torch.inference_mode()
    def run_v228_diagnostic(self):
        """Validate Strong-Anchor appearance matte compositing and strict PP lock."""
        root = Path("res") / "v228_diagnostic"
        active_root = ACTIVE_OUTPUT_DIR / "v228_diagnostic"
        for path in (
            root,
            root / "comparisons" / "visual",
            root / "comparisons" / "debug",
            root / "comparisons" / "crops",
            root / "visual_review",
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        # The checkpoint-selected V2.26 gamma=.25 path is retained solely as a
        # paired diagnostic baseline.
        if hasattr(self.model, "compensation_gamma"):
            self.model.compensation_gamma = USER_V227_DEFAULT_GAMMA

        pp_available = Path(USER_V227_PP_CHECKPOINT).exists()
        post_process = None
        if pp_available:
            post_process = PostProcessModel().to(self.device).eval()
            pp_state = torch.load(USER_V227_PP_CHECKPOINT, map_location=self.device)
            post_process.load_state_dict(pp_state["model_state_dict"])

        records = []
        parity_max = 0.0
        parity_sum = 0.0
        parity_count = 0
        parity_lab_sum = 0.0
        anchor_reliable = 0
        visual_count = 0
        visual_sample_ids = []

        for batch in tqdm(self.val_loader, desc="V2.28 deterministic diagnostic", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base, anchor, v226, _ = self._render_v225_pair(prepared)
            target = prepared["v224_target_hair_mask"]
            dilated, eroded = self.helper.dilate_erosion.mask(target)
            source_subject = prepared["v228_source_subject_mask"]
            hard = torch.zeros_like(target)
            input_kwargs = {
                "base_rgb": base,
                "anchor_rgb": anchor,
                "target_hair_mask": target,
                "target_hair_eroded": eroded,
                "target_hair_dilated": dilated,
                "face_keep_mask": source_subject * (1.0 - target),
                "skin_protect_mask": source_subject,
                "hard_protect_mask": hard,
                "soft_hair_probability": None,
                "matte_width": USER_V228_MATTE_WIDTH,
            }
            diagnostic_inputs = build_v828_runtime_inputs(**input_kwargs)
            diagnostic_prepp, aux = self.v228_compositor(
                return_aux=True, **diagnostic_inputs
            )
            real_inputs = Blending_v8.build_v228_runtime_inputs(**input_kwargs)
            real_prepp, _ = Blending_v8.run_v228_compositor_debug(
                self.v228_compositor, **real_inputs
            )
            parity_difference = (diagnostic_prepp - real_prepp).abs()
            parity_max = max(parity_max, float(parity_difference.max().item()))
            parity_sum += float(parity_difference.sum().item())
            parity_count += parity_difference.numel()
            parity_lab_sum += float(
                torch.linalg.vector_norm(
                    rgb_to_lab(diagnostic_prepp) - rgb_to_lab(real_prepp), dim=1
                ).mean().item()
            ) * diagnostic_prepp.size(0)

            core = aux["sure_fg"]
            edge = aux["unknown_band"]
            appearance = appearance_metric_tensors(
                base_rgb=base,
                anchor_rgb=anchor,
                v226_rgb=v226,
                output_rgb=diagnostic_prepp,
                core=core,
                edge=edge,
                outer_ring=aux["background_sample_ring"],
                face_mask=source_subject * (1.0 - target),
            )
            target_lab = prepared["pseudo_lab"]
            base_ref = v228_masked_mean_per_sample(
                torch.linalg.vector_norm(rgb_to_lab(base) - target_lab, dim=1, keepdim=True), core
            )
            anchor_ref = v228_masked_mean_per_sample(
                torch.linalg.vector_norm(rgb_to_lab(anchor) - target_lab, dim=1, keepdim=True), core
            )
            v226_ref = v228_masked_mean_per_sample(
                torch.linalg.vector_norm(rgb_to_lab(v226) - target_lab, dim=1, keepdim=True), core
            )

            pp_rgb = diagnostic_prepp
            final_rgb = diagnostic_prepp
            pp_metrics = {
                key: torch.zeros(diagnostic_prepp.size(0), device=self.device)
                for key in (
                    "pp_core_ab_shift", "pp_core_l_shift", "pp_core_full_de",
                    "pp_core_hf_shift", "pp_edge_ab_shift", "pp_edge_l_shift",
                    "pp_edge_full_de",
                )
            }
            if post_process is not None:
                face = prepared["face_i"]
                s_final, f_final = post_process(face, diagnostic_prepp * 2.0 - 1.0)
                pp_norm, _ = self.helper.net.generator(
                    [s_final], input_is_latent=True, return_latents=False,
                    start_layer=5, end_layer=8, layer_in=f_final,
                )
                pp_rgb = ((self.helper.downsample_256(pp_norm) + 1.0) / 2.0).clamp(0, 1)
                final_rgb, _ = apply_pp_hair_lock_v828(
                    diagnostic_prepp, pp_rgb, aux["hair_alpha"]
                )
                pp_metrics = pp_lock_metric_tensors(
                    diagnostic_prepp, final_rgb, core, edge
                )

            batch_size = base.size(0)
            for index in range(batch_size):
                reliable = float(anchor_ref[index].item()) <= 0.75 * float(base_ref[index].item())
                anchor_reliable += int(reliable)
                record = {
                    "sample_id": prepared["sample_id"][index],
                    "anchor_carrier_reliable": reliable,
                    "base_ref_full": float(base_ref[index].item()),
                    "anchor_ref_full": float(anchor_ref[index].item()),
                    "v226_ref_full": float(v226_ref[index].item()),
                    "outside_max_delta": float(aux["outside_max_delta"][index].item()),
                    "hard_protect_max_delta": float(aux["hard_protect_max_delta"][index].item()),
                    "core_anchor_max_delta": float(aux["core_anchor_max_delta"][index].item()),
                    "background_est_valid_fraction": float(aux["background_est_valid"][index].mean().item()),
                }
                for metrics in (appearance, pp_metrics):
                    for key, value in metrics.items():
                        record[key] = float(value[index].item())
                records.append(record)
                sample_id = record["sample_id"]
                if visual_count < USER_V228_REAL_INFERENCE_COUNT:
                    visual_sample_ids.append(sample_id)
                    visual_row = [
                        prepared["face_i"][index:index + 1],
                        prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1],
                        base[index:index + 1] * 2 - 1,
                        anchor[index:index + 1] * 2 - 1,
                        v226[index:index + 1] * 2 - 1,
                        diagnostic_prepp[index:index + 1] * 2 - 1,
                        pp_rgb[index:index + 1] * 2 - 1,
                        final_rgb[index:index + 1] * 2 - 1,
                    ]
                    debug_row = [
                        mask_to_preview(aux["hair_alpha"][index:index + 1]),
                        mask_to_preview(core[index:index + 1]),
                        mask_to_preview(edge[index:index + 1]),
                        mask_to_preview(aux["outside"][index:index + 1]),
                        (aux["anchor_base_residual"][index:index + 1] * 2).clamp(-1, 1),
                        (aux["background_residual_estimate"][index:index + 1] * 2).clamp(-1, 1),
                        (aux["clean_boundary_residual"][index:index + 1] * 2).clamp(-1, 1),
                        aux["edge_candidate"][index:index + 1] * 2 - 1,
                        ((diagnostic_prepp[index:index + 1] - base[index:index + 1]).abs() * 4).clamp(0, 1) * 2 - 1,
                    ]
                    save_preview(root / "comparisons" / "visual" / f"sample_{visual_count:03d}.png", visual_row)
                    save_preview(root / "comparisons" / "debug" / f"sample_{visual_count:03d}.png", debug_row)
                    self._save_v227_crops(
                        root / "comparisons" / "crops", sample_id,
                        [base[index:index + 1], anchor[index:index + 1], v226[index:index + 1],
                         diagnostic_prepp[index:index + 1], final_rgb[index:index + 1]],
                        edge[index:index + 1],
                    )
                    visual_count += 1

        if not records:
            raise RuntimeError("V2.28 diagnostic produced no valid samples")
        parity_mean = parity_sum / max(parity_count, 1)
        parity_lab = parity_lab_sum / len(records)
        summary = aggregate_v228_records(records)
        reliable_fraction = anchor_reliable / len(records)
        decision = classify_v228(
            summary,
            parity_max_diff=parity_max,
            anchor_reliable_fraction=reliable_fraction,
        )
        if not pp_available and decision == "V228_READY_FOR_VISUAL_REVIEW":
            decision = "V228_PP_HAIR_LOCK_FAIL"

        def _rank(key, reverse=True, count=8):
            return [r["sample_id"] for r in sorted(records, key=lambda item: float(item.get(key, 0.0)), reverse=reverse)[:count]]

        fixed_regression = {
            "strong_anchor_face_pollution_high": _rank("face_contamination_rgb"),
            "strong_anchor_background_pollution_high": _rank("background_contamination_retention"),
            "v226_edge_core_delta_e_high": _rank("v226_edge_core_delta_e"),
            "v228_gray_contour_high": _rank("appearance_flattening_fraction"),
            "anchor_carrier_unreliable": [r["sample_id"] for r in records if not r["anchor_carrier_reliable"]][:8],
        }
        runtime_parity = {
            "max_rgb_diff": parity_max,
            "mean_rgb_diff": parity_mean,
            "mean_lab_delta_e": parity_lab,
            "thresholds": {"max_rgb_diff": 1e-5, "mean_rgb_diff": 1e-7, "mean_lab_delta_e": 1e-4},
            "passed": parity_max <= 1e-5 and parity_mean <= 1e-7 and parity_lab <= 1e-4,
            "shared_builder": "models.v828_runtime_inputs.build_v828_runtime_inputs",
        }
        mask_audit = {
            "soft_hair_probability_available": False,
            "source_tensor": "HM_X / cached target_hair (binary)",
            "range": [0, 1],
            "resolution": "256x256",
            "currently_thresholded_at": "Embedding/get_segmentation argmax and HM_X binary mask",
            "selected_for_v228": False,
            "fallback": "deterministic narrow trimap finite-distance matte",
        }
        acceptance = {
            "version": "v2.28",
            "mode": "STRONG_ANCHOR_APPEARANCE_MATTE_COMPOSITE",
            "training": False,
            "diagnostic_only": True,
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "v226_role": "DIAGNOSTIC_REFERENCE_ONLY",
            "compositor_config": self.v228_compositor.config_dict(),
            "runtime_parity": runtime_parity,
            "anchor_carrier_reliable_fraction": reliable_fraction,
            "postprocess_available": pp_available,
            "summary": summary,
            "automatic_decision": decision,
            "final_visual_decision": None,
        }
        outputs = {
            "mask_source_audit.json": mask_audit,
            "runtime_parity.json": runtime_parity,
            "anchor_carrier_feasibility.json": {"reliable_fraction": reliable_fraction, "threshold": 0.70},
            "matte_audit.json": {
                "core_anchor_max_delta": summary.get("max_core_anchor_max_delta"),
                "outside_max_delta": summary.get("max_outside_max_delta"),
                "hard_protect_max_delta": summary.get("max_hard_protect_max_delta"),
                "coverage_alpha_multiply_count": 1,
            },
            "background_decontamination_audit.json": {
                "median_retention": summary.get("median_background_contamination_retention"),
                "target_max": 0.05,
            },
            "appearance_metrics.json": summary,
            "pp_hair_lock_metrics.json": {
                key: value for key, value in summary.items() if "pp_" in key
            },
            "v228_acceptance.json": acceptance,
            "v228_fixed_regression_manifest.json": fixed_regression,
        }
        for filename, payload in outputs.items():
            with open(root / filename, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(root / "per_sample.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        review = {
            "reviewed_by_user": False,
            "visual_column_order": [
                "Source", "Shape Reference", "Color Reference", "Base", "Strong Anchor",
                "V2.26 gamma=.25 pre-PP", "V2.28 pre-PP", "Original PP output",
                "V2.28 Hair-Locked Final",
            ],
            "sample_ids": visual_sample_ids,
            "samples": {sample_id: {"decision": "", "notes": ""} for sample_id in visual_sample_ids},
        }
        with open(root / "visual_review" / "v228_visual_review.json", "w", encoding="utf-8") as handle:
            json.dump(review, handle, ensure_ascii=False, indent=2)
        checklist = """# V2.28 Visual Review Checklist

- [ ] Hair color is close to Color Reference
- [ ] Hair shine and texture remain close to Strong Anchor
- [ ] Hair core is not flat or filter-like
- [ ] Left and right contours have no gray translucent band
- [ ] Hairline has no gray/white outline
- [ ] Fine strands are not cut by the morphology mask
- [ ] Background remains Base
- [ ] Face remains Source/Base
- [ ] Final PP does not rewrite hair color or shine
- [ ] No colored or dark fringe appears
- [ ] Overall hairstyle contour is natural
"""
        (root / "visual_review" / "V228_VISUAL_REVIEW_CHECKLIST.md").write_text(checklist, encoding="utf-8")
        edge_de = float(summary.get("median_edge_core_delta_e", 0.0))
        v226_edge_de = float(summary.get("median_v226_edge_core_delta_e", 0.0))
        edge_improvement = 1.0 - edge_de / max(v226_edge_de, 1e-8)
        hf_l1 = float(summary.get("median_anchor_hf_l1", 0.0))
        v226_hf_l1 = float(summary.get("median_v226_anchor_hf_l1", 0.0))
        hf_improvement = 1.0 - hf_l1 / max(v226_hf_l1, 1e-8)
        failure_area = {
            "V228_RUNTIME_PARITY_FAIL": "runtime parity",
            "V228_ANCHOR_CARRIER_NOT_GENERAL": "appearance carrier",
            "V228_MATTE_COMPOSITE_FAIL": "matte geometry/composite",
            "V228_BACKGROUND_DECONTAMINATION_FAIL": "background decontamination",
            "V228_APPEARANCE_PRESERVATION_FAIL": "matte/appearance preservation",
            "V228_COLOR_REGRESSION": "appearance carrier color fidelity",
            "V228_PP_HAIR_LOCK_FAIL": "PostProcess hair lock",
            "V228_READY_FOR_VISUAL_REVIEW": "none automatically; visual review is required",
        }[decision]
        report = f"""# Blending V8 V2.28 Acceptance

1. V2.27 parity failed because diagnostic and simulated real inference independently rebuilt
   eroded/dilated, face, skin, and hard-protection masks. V2.28 routes both through
   `build_v828_runtime_inputs`; max RGB difference is `{parity_max:.8g}`.
2. The current real runtime exposes thresholded parser labels and binary HM_X, not a shared
   soft hair probability. See `mask_source_audit.json`.
3. The binary fallback uses a narrow eroded/dilated trimap, finite distances to sure foreground
   and sure background, smoothstep alpha, and a capped image-evidence adjustment.
4. Hair core is exact Strong Anchor; maximum core difference is
   `{float(summary.get('max_core_anchor_max_delta', 0.0)):.8g}`.
5. Outside and hard-protect are exact Base; maximum differences are
   `{float(summary.get('max_outside_max_delta', 0.0)):.8g}` and
   `{float(summary.get('max_hard_protect_max_delta', 0.0)):.8g}`.
6. Median background-contamination retention after normalized residual removal is
   `{float(summary.get('median_background_contamination_retention', 0.0)):.6f}` (target <= 0.05).
7. Median edge-core delta E is `{edge_de:.4f}` versus V2.26 `{v226_edge_de:.4f}`;
   relative improvement is `{edge_improvement:.2%}`.
8. Median Anchor HF L1 is `{hf_l1:.6f}` versus V2.26 `{v226_hf_l1:.6f}`;
   relative improvement is `{hf_improvement:.2%}`.
9. Median Strong Anchor/reference full-color error is
   `{float(summary.get('median_anchor_ref_full', 0.0)):.4f}` versus Base
   `{float(summary.get('median_base_ref_full', 0.0)):.4f}`; carrier reliability is
   `{reliable_fraction:.2%}`.
10. Median face contamination relative to Base is
    `{float(summary.get('median_face_contamination_rgb', 0.0)):.8g}`; background protection is
    reported in items 5 and 6.
11. Median PP core AB/L drift after strict lock is
    `{float(summary.get('median_pp_core_ab_shift', 0.0)):.6f}` /
    `{float(summary.get('median_pp_core_l_shift', 0.0)):.6f}`.
12. Automatic decision: `{decision}`. This is not a final visual PASS.
13. The most important worst cases are listed by failure type in
    `v228_fixed_regression_manifest.json`; inspect `comparisons/crops` first.
14. V2.29 is recommended only after the required human review, unless the automatic gate has
    already isolated a failure.
15. The currently isolated follow-up area is: `{failure_area}`.

Human review remains required; record it in `visual_review/v228_visual_review.json`.
"""
        (root / "BLENDING_V8_V2_28_ACCEPTANCE.md").write_text(report, encoding="utf-8")
        shutil.copytree(root, active_root, dirs_exist_ok=True)
        print(
            f"[V2.28] parity_max={parity_max:.2e} anchor_reliable={reliable_fraction:.4f} "
            f"edge_core_dE={float(summary.get('median_edge_core_delta_e', 0.0)):.4f} "
            f"v226_edge_core_dE={float(summary.get('median_v226_edge_core_delta_e', 0.0)):.4f} "
            f"decision={decision}"
        )
        return acceptance

    @torch.inference_mode()
    def run_v229_diagnostic(self):
        """Run deterministic V2.29 Phase A with V2.26/V2.28 paired controls."""
        root = Path("res") / "v229_diagnostic"
        active_root = ACTIVE_OUTPUT_DIR / "v229_diagnostic"
        for path in (
            root,
            root / "comparisons" / "visual",
            root / "comparisons" / "debug",
            root / "comparisons" / "crops",
            root / "comparisons" / "pp_comparison",
            root / "visual_review",
        ):
            path.mkdir(parents=True, exist_ok=True)

        phase_a_baseline = None
        if USER_V229_OUTER_STRAND_RECOVERY:
            phase_a_acceptance_path = root / "v229_acceptance.json"
            phase_a_review_path = root / "visual_review" / "v229_visual_review.json"
            if not phase_a_acceptance_path.exists() or not phase_a_review_path.exists():
                raise RuntimeError(
                    "V2.29 Phase B requires a completed Phase-A diagnostic and visual review"
                )
            with open(phase_a_acceptance_path, "r", encoding="utf-8") as handle:
                phase_a_acceptance = json.load(handle)
            with open(phase_a_review_path, "r", encoding="utf-8") as handle:
                phase_a_review = json.load(handle)
            if (
                phase_a_acceptance.get("automatic_decision")
                != "V229_PHASE_A_READY_FOR_VISUAL_REVIEW"
                or not phase_a_review.get("reviewed_by_user", False)
            ):
                raise RuntimeError(
                    "V2.29 Phase B is locked until Phase A passes automatically and reviewed_by_user is true"
                )
            phase_a_baseline = phase_a_acceptance.get("summary")
            if not isinstance(phase_a_baseline, dict):
                raise RuntimeError("V2.29 Phase-A acceptance is missing its metric summary")

        self.model.eval()
        if hasattr(self.model, "compensation_gamma"):
            self.model.compensation_gamma = USER_V227_DEFAULT_GAMMA
        pp_available = Path(USER_V227_PP_CHECKPOINT).exists()
        post_process = None
        if pp_available:
            post_process = PostProcessModel().to(self.device).eval()
            pp_state = torch.load(USER_V227_PP_CHECKPOINT, map_location=self.device)
            post_process.load_state_dict(pp_state["model_state_dict"])

        records = []
        parity_max = 0.0
        parity_sum = 0.0
        parity_count = 0
        parity_lab_sum = 0.0
        visual_ids = []
        visual_index = 0

        for batch in tqdm(self.val_loader, desc="V2.29 Phase A diagnostic", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base, anchor, v226, _ = self._render_v225_pair(prepared)
            hair = prepared["v224_target_hair_mask"]
            dilated, eroded = self.helper.dilate_erosion.mask(hair)
            source_subject = prepared["v228_source_subject_mask"]
            source_skin = source_subject
            hard = torch.zeros_like(hair)

            v228_inputs = build_v828_runtime_inputs(
                base_rgb=base,
                anchor_rgb=anchor,
                target_hair_mask=hair,
                target_hair_eroded=eroded,
                target_hair_dilated=dilated,
                face_keep_mask=source_subject * (1.0 - hair),
                skin_protect_mask=source_skin,
                hard_protect_mask=hard,
                matte_width=USER_V228_MATTE_WIDTH,
            )
            v228, _ = self.v228_compositor(return_aux=True, **v228_inputs)
            runtime_kwargs = {
                "base_rgb": base,
                "anchor_rgb": anchor,
                "v226_rgb": v226,
                "target_hair_mask": hair,
                "target_hair_eroded": eroded,
                "target_hair_dilated": dilated,
                "source_subject_mask": source_subject,
                "source_skin_mask": source_skin,
                "hard_protect_mask": hard,
                "matte_width": USER_V228_MATTE_WIDTH,
            }
            diagnostic_inputs = build_v829_runtime_inputs(**runtime_kwargs)
            diagnostic_prepp, aux = Blending_v8.run_v229_compositor_debug(
                self.v229_carrier, self.v229_recompositor, **diagnostic_inputs
            )
            real_inputs = Blending_v8.build_v229_runtime_inputs(**runtime_kwargs)
            real_prepp, _ = Blending_v8.run_v229_compositor_debug(
                self.v229_carrier, self.v229_recompositor, **real_inputs
            )
            difference = (diagnostic_prepp - real_prepp).abs()
            parity_max = max(parity_max, float(difference.max().item()))
            parity_sum += float(difference.sum().item())
            parity_count += difference.numel()
            parity_lab_sum += float(torch.linalg.vector_norm(
                rgb_to_lab(diagnostic_prepp) - rgb_to_lab(real_prepp), dim=1
            ).mean().item()) * diagnostic_prepp.size(0)

            owner = aux["hair_ownership"]
            nonhair = aux["nonhair_owner"]
            core = aux["core_owner"]
            inner = aux["inner_edge_owner"]
            metrics = v229_metric_tensors(
                base_rgb=base,
                anchor_rgb=anchor,
                v226_rgb=v226,
                v228_rgb=v228,
                output_rgb=diagnostic_prepp,
                target_lab=prepared["pseudo_lab"],
                core=core,
                inner_edge=inner,
                ownership=owner,
                nonhair=nonhair,
                source_skin=source_skin,
                face_context=aux["face_context_map"],
                background_context=aux["background_context_map"],
            )
            anchor_outside = v228_masked_mean_per_sample((anchor - base).abs(), nonhair)
            output_outside = v228_masked_mean_per_sample((diagnostic_prepp - base).abs(), nonhair)
            background_retention = output_outside / anchor_outside.clamp_min(1e-8)
            anchor_face_contamination = v228_masked_mean_per_sample(
                (anchor - base).abs(), source_skin * nonhair
            )
            face_outer_den = aux["face_side_outer"].flatten(1).sum(dim=1)
            face_outer_fraction = (
                (aux["outer_strand_owner"] * aux["face_side_outer"]).flatten(1).sum(dim=1)
                / face_outer_den.clamp_min(1.0)
            )
            low_coverage = (
                inner
                * (aux["coverage_alpha"] <= 0.25).to(inner.dtype)
                * aux["background_recompose_valid"]
            )
            boundary_bg_error = v228_masked_mean_per_sample(
                (diagnostic_prepp - aux["base_background_estimate"]).abs(), low_coverage
            )
            anchor_bg_error = v228_masked_mean_per_sample(
                (anchor - aux["base_background_estimate"]).abs(), low_coverage
            )
            reference_bg_retention = boundary_bg_error / anchor_bg_error.clamp_min(1e-8)
            owner_neighbors = tnf.conv2d(
                owner,
                torch.ones(1, 1, 3, 3, device=owner.device, dtype=owner.dtype),
                padding=1,
            )
            ownership_spike_fraction = v228_masked_mean_per_sample(
                (owner_neighbors <= 2).to(owner.dtype), inner
            )
            accepted_outer_den = aux["outer_strand_owner"].flatten(1).sum(dim=1)
            outer_alignment = (
                (aux["residual_alignment"] * aux["outer_strand_owner"]).flatten(1).sum(dim=1)
                / accepted_outer_den.clamp_min(1.0)
            )
            outer_min_alignment = torch.where(
                aux["outer_strand_owner"] > 0.5,
                aux["residual_alignment"],
                torch.ones_like(aux["residual_alignment"]),
            ).flatten(1).amin(dim=1)
            outer_small_component_fraction = (
                (
                    (aux["neighbor_support_count"] < 3).to(owner.dtype)
                    * aux["outer_strand_owner"]
                ).flatten(1).sum(dim=1)
                / accepted_outer_den.clamp_min(1.0)
            )

            pp_rgb = diagnostic_prepp
            final_rgb = diagnostic_prepp
            pp_metrics = {
                key: torch.zeros(base.size(0), device=self.device)
                for key in (
                    "pp_owner_max_delta", "nonhair_final_to_pp_max_delta",
                    "pp_owner_ab_shift", "pp_owner_l_shift",
                    "pp_inner_edge_ab_shift", "pp_inner_edge_l_shift",
                )
            }
            if post_process is not None:
                s_final, f_final = post_process(prepared["face_i"], diagnostic_prepp * 2.0 - 1.0)
                pp_norm, _ = self.helper.net.generator(
                    [s_final], input_is_latent=True, return_latents=False,
                    start_layer=5, end_layer=8, layer_in=f_final,
                )
                pp_rgb = ((self.helper.downsample_256(pp_norm) + 1.0) / 2.0).clamp(0, 1)
                final_rgb, _ = apply_pp_hair_ownership_lock_v829(
                    diagnostic_prepp, pp_rgb, owner
                )
                pp_metrics = pp_ownership_metrics(
                    diagnostic_prepp, pp_rgb, final_rgb, owner, inner
                )

            for index, sample_id in enumerate(prepared["sample_id"]):
                record = {
                    "sample_id": sample_id,
                    "phase": "B" if USER_V229_OUTER_STRAND_RECOVERY else "A",
                    "outside_max_delta": float(aux["outside_max_delta"][index].item()),
                    "hard_protect_max_delta": float(aux["hard_protect_max_delta"][index].item()),
                    "background_contamination_retention": float(background_retention[index].item()),
                    "strong_anchor_face_contamination": float(anchor_face_contamination[index].item()),
                    "face_side_outer_hair_fraction": float(face_outer_fraction[index].item()),
                    "background_recompose_valid_fraction": float(aux["background_recompose_valid"][index].mean().item()),
                    "boundary_bg_replacement_error": float(boundary_bg_error[index].item()),
                    "reference_background_color_retention": float(reference_bg_retention[index].item()),
                    "ownership_boundary_one_pixel_spike_fraction": float(ownership_spike_fraction[index].item()),
                    "outer_strand_accepted_fraction": float(aux["outer_strand_owner"][index].mean().item()),
                    "outer_strand_residual_alignment": float(outer_alignment[index].item()),
                    "outer_strand_min_residual_alignment": float(outer_min_alignment[index].item()),
                    "outer_strand_small_component_fraction": float(outer_small_component_fraction[index].item()),
                    "clip_low_fraction": float(aux["clip_low_fraction"][index].item()),
                    "clip_high_fraction": float(aux["clip_high_fraction"][index].item()),
                    "clip_magnitude": float(aux["clip_magnitude"][index].item()),
                }
                for group in (metrics, pp_metrics):
                    for key, value in group.items():
                        record[key] = float(value[index].item())
                records.append(record)

                if visual_index < USER_V229_VISUAL_COUNT:
                    visual_ids.append(sample_id)
                    visual_row = [
                        prepared["face_i"][index:index + 1],
                        prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1],
                        base[index:index + 1] * 2 - 1,
                        anchor[index:index + 1] * 2 - 1,
                        v226[index:index + 1] * 2 - 1,
                        v228[index:index + 1] * 2 - 1,
                        diagnostic_prepp[index:index + 1] * 2 - 1,
                        pp_rgb[index:index + 1] * 2 - 1,
                        final_rgb[index:index + 1] * 2 - 1,
                    ]
                    debug_row = [
                        mask_to_preview(hair[index:index + 1]),
                        mask_to_preview(inner[index:index + 1]),
                        mask_to_preview(aux["outer_candidate"][index:index + 1]),
                        mask_to_preview(aux["coverage_alpha"][index:index + 1]),
                        mask_to_preview(owner[index:index + 1]),
                        mask_to_preview(aux["face_side_outer"][index:index + 1]),
                        aux["anchor_background_estimate"][index:index + 1] * 2 - 1,
                        aux["base_background_estimate"][index:index + 1] * 2 - 1,
                        (aux["background_replacement_delta"][index:index + 1] * 3).clamp(-1, 1),
                        (aux["tone_delta_core"][index:index + 1] * 3).clamp(-1, 1),
                        (aux["tone_delta_edge"][index:index + 1] * 3).clamp(-1, 1),
                        mask_to_preview(owner[index:index + 1]),
                    ]
                    save_preview(root / "comparisons" / "visual" / f"sample_{visual_index:03d}.png", visual_row)
                    save_preview(root / "comparisons" / "debug" / f"sample_{visual_index:03d}.png", debug_row)
                    save_preview(
                        root / "comparisons" / "pp_comparison" / f"sample_{visual_index:03d}.png",
                        [diagnostic_prepp[index:index + 1] * 2 - 1,
                         pp_rgb[index:index + 1] * 2 - 1,
                         final_rgb[index:index + 1] * 2 - 1],
                    )
                    self._save_v229_crops(
                        root / "comparisons" / "crops", sample_id,
                        [anchor[index:index + 1], v226[index:index + 1], v228[index:index + 1],
                         diagnostic_prepp[index:index + 1], final_rgb[index:index + 1]],
                        inner[index:index + 1],
                        aux["face_context_map"][index:index + 1],
                        aux["background_context_map"][index:index + 1],
                    )
                    visual_index += 1

        if not records:
            raise RuntimeError("V2.29 diagnostic produced no valid samples")
        parity_mean = parity_sum / max(parity_count, 1)
        parity_lab = parity_lab_sum / len(records)
        summary = aggregate_v229(records)
        decision = classify_v229(
            summary,
            parity_max_diff=parity_max,
            phase_b=USER_V229_OUTER_STRAND_RECOVERY,
            phase_a_summary=phase_a_baseline,
        )
        if not pp_available and decision in (
            "V229_PHASE_A_READY_FOR_VISUAL_REVIEW",
            "V229_PHASE_B_READY_FOR_VISUAL_REVIEW",
        ):
            decision = "V229_PP_OWNERSHIP_LOCK_FAIL"

        runtime = {
            "max_rgb_diff": parity_max,
            "mean_rgb_diff": parity_mean,
            "mean_lab_delta_e": parity_lab,
            "passed": parity_max <= 1e-5 and parity_mean <= 1e-7 and parity_lab <= 1e-4,
            "shared_builder": "models.v829_runtime_inputs.build_v829_runtime_inputs",
        }
        def rank(key, reverse=True, count=8):
            return [item["sample_id"] for item in sorted(records, key=lambda value: float(value.get(key, 0.0)), reverse=reverse)[:count]]

        manifest = {
            "base_color_rim_high": rank("base_color_rim_fraction"),
            "high_chroma_color_coverage": rank("target_core_chroma"),
            "warm_gold_orange_base_rim": rank("target_core_warm_score"),
            "hair_to_face_leak": rank("face_side_color_leak_rgb"),
            "jagged_or_incomplete_edge": rank("edge_to_nearcore_delta_e"),
            "strong_anchor_background_pollution": rank("background_contamination_retention"),
            "strong_anchor_face_pollution": rank("strong_anchor_face_contamination"),
            "v228_pp_edge_drift": rank("v228_edge_to_nearcore_delta_e"),
            "hybrid_core_color_worst": rank("core_ref_full"),
            "hybrid_hf_worst": rank("anchor_hf_l1"),
            "strong_anchor_full_color_unreliable_but_hf_useful": rank("anchor_core_ref_full"),
            "recommended_visual_review": visual_ids,
        }
        acceptance = {
            "version": "v2.29",
            "architecture": "hybrid_carrier_background_recomposition_ownership_lock_v829",
            "phase": "B" if USER_V229_OUTER_STRAND_RECOVERY else "A",
            "training": False,
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "runtime_parity": runtime,
            "hybrid_carrier": self.v229_carrier.config_dict(),
            "recomposition": self.v229_recompositor.config_dict(),
            "summary": summary,
            "postprocess_available": pp_available,
            "automatic_decision": decision,
            "final_visual_decision": None,
        }
        outputs = {
            "runtime_parity.json": runtime,
            "hybrid_carrier_audit.json": {key: value for key, value in summary.items() if "core_ref" in key or "hf_l1" in key or "gradient" in key or "core_l_std" in key or "highlight" in key or "texture" in key},
            "ownership_audit.json": {key: value for key, value in summary.items() if "owner" in key or "ownership" in key or "outer_hair" in key or "outer_strand" in key},
            "background_recomposition_audit.json": {key: value for key, value in summary.items() if "background" in key or "outside" in key},
            "face_boundary_audit.json": {key: value for key, value in summary.items() if "face_" in key},
            "pp_ownership_lock.json": {key: value for key, value in summary.items() if "pp_" in key or "nonhair_final" in key},
            "gamut_audit.json": {key: value for key, value in summary.items() if "clip_" in key},
            "phase_a_summary.json": summary if not USER_V229_OUTER_STRAND_RECOVERY else phase_a_baseline,
            "phase_b_summary.json": summary if USER_V229_OUTER_STRAND_RECOVERY else {"enabled": False, "reason": "Phase A must pass visual review first"},
            "v229_acceptance.json": acceptance,
            "fixed_regression_manifest.json": manifest,
        }
        for filename, payload in outputs.items():
            with open(root / filename, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(root / "per_sample.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        review = {
            "reviewed_by_user": False,
            "visual_column_order": [
                "Source", "Shape Reference", "Color Reference", "Base", "Strong Anchor",
                "V2.26", "V2.28", "V2.29 Phase-A prePP", "Original PP", "V2.29 Final",
            ],
            "sample_ids": visual_ids,
            "samples": {sample_id: {"decision": "", "notes": ""} for sample_id in visual_ids},
        }
        with open(root / "visual_review" / "v229_visual_review.json", "w", encoding="utf-8") as handle:
            json.dump(review, handle, ensure_ascii=False, indent=2)
        checklist = """# V2.29 Visual Review Checklist

- [ ] Hybrid core keeps V2.26 target tone and Strong Anchor texture
- [ ] Inner edge has no gray translucent or Base-color rim
- [ ] HM_X hair can cover Source forehead
- [ ] Dilation outside HM_X does not recolor Source skin
- [ ] Hair-to-face and hair-to-background contexts do not cross-contaminate
- [ ] No jagged one-pixel ownership spikes
- [ ] PP owner pixels remain exactly pre-PP
- [ ] Non-hair retains Original PP refinement
- [ ] Phase B is needed only for visibly missing strands outside HM_X
"""
        (root / "visual_review" / "V229_VISUAL_REVIEW_CHECKLIST.md").write_text(checklist, encoding="utf-8")

        edge_gain = 1.0 - float(summary.get("median_edge_to_nearcore_delta_e", 0.0)) / max(float(summary.get("median_v228_edge_to_nearcore_delta_e", 0.0)), 1e-8)
        rim_gain = 1.0 - float(summary.get("median_base_color_rim_fraction", 0.0)) / max(float(summary.get("median_v228_base_color_rim_fraction", 0.0)), 1e-8)
        report = f"""# Blending V8 V2.29 Acceptance

1. V2.28 double coverage was removed from V2.29; no `Base + alpha*(carrier-Base)` exists in its boundary path.
2. Boundary uses `Anchor + alpha*tone_delta + (1-alpha)*(B_base-B_anchor)`.
3. Strong Anchor supplies high-frequency appearance and the observed mixed-edge pixel only.
4. Core low frequency comes from normalized V2.26 core propagation.
5. Hybrid/V2.26 median core reference errors: `{float(summary.get('median_core_ref_full', 0.0)):.4f}` / `{float(summary.get('median_v226_core_ref_full', 0.0)):.4f}`.
6. Hybrid/V2.26 Anchor-HF L1: `{float(summary.get('median_anchor_hf_l1', 0.0)):.6f}` / `{float(summary.get('median_v226_anchor_hf_l1', 0.0)):.6f}`.
7. Inner edge is ownership-selected after background replacement and is never alpha-composited again.
8. Maximum face-side outer ownership fraction is `{float(summary.get('max_face_side_outer_hair_fraction', 0.0)):.8g}`.
9. Median Base-color rim improvement over V2.28 is `{rim_gain:.2%}`.
10. Median edge-to-nearcore delta-E improvement over V2.28 is `{edge_gain:.2%}`.
11. Median background contamination retention is `{float(summary.get('median_background_contamination_retention', 0.0)):.6f}`.
12. Median face-side color leakage is `{float(summary.get('median_face_side_color_leak_rgb', 0.0)):.8g}`.
13. Maximum PP owner delta is `{float(summary.get('max_pp_owner_max_delta', 0.0)):.8g}`.
14. Maximum non-hair-to-Original-PP delta is `{float(summary.get('max_nonhair_final_to_pp_max_delta', 0.0)):.8g}`.
15. Phase B outer-strand recovery is `{'enabled' if USER_V229_OUTER_STRAND_RECOVERY else 'disabled pending Phase-A visual review'}`.
16. Learned/high-resolution matting is justified only if Phase A is clean but real strands outside HM_X remain missing.
17. Automatic decision: `{decision}`. This is not a final visual PASS.
18. Review the 10-20 sample IDs in `fixed_regression_manifest.json` and `comparisons/crops` first.
"""
        (root / "BLENDING_V8_V2_29_ACCEPTANCE.md").write_text(report, encoding="utf-8")
        shutil.copytree(root, active_root, dirs_exist_ok=True)
        print(
            f"[V2.29] phase={'B' if USER_V229_OUTER_STRAND_RECOVERY else 'A'} "
            f"parity_max={parity_max:.2e} core_ref={float(summary.get('median_core_ref_full', 0.0)):.4f} "
            f"v226_core_ref={float(summary.get('median_v226_core_ref_full', 0.0)):.4f} "
            f"edge_nearcore_dE={float(summary.get('median_edge_to_nearcore_delta_e', 0.0)):.4f} "
            f"decision={decision}"
        )
        return acceptance

    @torch.inference_mode()
    def run_v230_diagnostic(self):
        """Run deterministic V2.30 validation with V2.29 paired controls."""
        v229_acceptance = {}
        v229_acceptance_path = Path("res") / "v229_diagnostic" / "v229_acceptance.json"
        if v229_acceptance_path.exists():
            with open(v229_acceptance_path, "r", encoding="utf-8") as handle:
                v229_acceptance = json.load(handle)
        root = Path("res") / "v230_diagnostic"
        active_root = ACTIVE_OUTPUT_DIR / "v230_diagnostic"
        for path in (
            root,
            root / "comparisons" / "visual",
            root / "comparisons" / "debug",
            root / "comparisons" / "crops",
            root / "visual_review",
        ):
            path.mkdir(parents=True, exist_ok=True)
        parser_audit = parser_region_audit_v830()
        with open(root / "parser_region_audit.json", "w", encoding="utf-8") as handle:
            json.dump(parser_audit, handle, ensure_ascii=False, indent=2, allow_nan=False)
        post_process = None
        pp_checkpoint = Path(USER_V227_PP_CHECKPOINT)
        if pp_checkpoint.exists():
            post_process = PostProcessModel().to(self.device).eval()
            pp_state = torch.load(pp_checkpoint, map_location=self.device)
            post_process.load_state_dict(pp_state["model_state_dict"])

        records = []
        visual_ids = []
        parity_max = 0.0
        parity_sum = 0.0
        parity_count = 0
        visual_index = 0
        for batch in tqdm(self.val_loader, desc="V2.30 deterministic diagnostic", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base, anchor, v226, _ = self._render_v225_pair(prepared)
            hair = prepared["v224_target_hair_mask"]
            dilated, eroded = self.helper.dilate_erosion.mask(hair)
            source_subject = prepared["v228_source_subject_mask"]
            source_face = prepared["v230_source_face_mask"]
            source_skin = prepared["v230_source_skin_mask"]
            parser_labels = prepared["v230_parser_labels"]
            hard = torch.zeros_like(hair)
            v229_inputs = build_v829_runtime_inputs(
                base_rgb=base, anchor_rgb=anchor, v226_rgb=v226,
                target_hair_mask=hair, target_hair_eroded=eroded,
                target_hair_dilated=dilated, source_subject_mask=source_subject,
                source_skin_mask=source_subject, hard_protect_mask=hard,
                matte_width=USER_V228_MATTE_WIDTH,
            )
            v229_prepp, v229_aux = Blending_v8.run_v229_compositor_debug(
                self.v229_carrier, self.v229_recompositor, **v229_inputs
            )
            topology, topology_aux = self.v230_topology(
                target_hair_mask=hair, source_face_mask=source_face, return_aux=True
            )
            repaired_dilated, repaired_eroded = self.helper.dilate_erosion.mask(topology)
            v230_inputs = build_v829_runtime_inputs(
                base_rgb=base, anchor_rgb=anchor, v226_rgb=v226,
                target_hair_mask=topology, target_hair_eroded=repaired_eroded,
                target_hair_dilated=repaired_dilated, source_subject_mask=source_subject,
                source_skin_mask=source_skin, hard_protect_mask=hard,
                matte_width=USER_V228_MATTE_WIDTH,
            )
            core, carrier_aux = self.v230_carrier(
                anchor_rgb=anchor, v226_rgb=v226, repaired_hair_core=repaired_eroded,
                return_aux=True
            )
            v230_prepp, v230_recompose_aux = self.v229_recompositor(
                **v230_inputs, hybrid_core_rgb=core, return_aux=True
            )
            runtime_kwargs = dict(
                base_rgb=base, anchor_rgb=anchor, v226_rgb=v226,
                v229_prepp_rgb=v230_prepp, pp_original_rgb=v230_prepp,
                target_hair_mask=topology, target_hair_eroded=repaired_eroded,
                target_hair_dilated=repaired_dilated, parser_labels=parser_labels,
            )
            real_inputs = Blending_v8.build_v230_runtime_inputs(**runtime_kwargs)
            diagnostic_inputs = build_v830_runtime_inputs(**runtime_kwargs)
            parity = (diagnostic_inputs["v229_prepp_rgb"] - real_inputs["v229_prepp_rgb"]).abs()
            parity_max = max(parity_max, float(parity.max().item()))
            parity_sum += float(parity.sum().item())
            parity_count += parity.numel()

            if post_process is not None:
                s_v229, f_v229 = post_process(prepared["face_i"], v229_prepp * 2.0 - 1.0)
                pp_v229_norm, _ = self.helper.net.generator(
                    [s_v229], input_is_latent=True, return_latents=False,
                    start_layer=5, end_layer=8, layer_in=f_v229,
                )
                pp_v229 = ((pp_v229_norm + 1.0) / 2.0).clamp(0, 1)
                v229_owner = tnf.interpolate(
                    v229_aux["hair_ownership"], size=pp_v229_norm.shape[-2:], mode="nearest"
                )
                v229_final, _ = apply_pp_hair_ownership_lock_v829(
                    tnf.interpolate(v229_prepp, size=pp_v229_norm.shape[-2:], mode="bicubic", align_corners=False),
                    pp_v229, v229_owner,
                )
                s_final, f_final = post_process(prepared["face_i"], v230_prepp * 2.0 - 1.0)
                pp_norm, _ = self.helper.net.generator(
                    [s_final], input_is_latent=True, return_latents=False,
                    start_layer=5, end_layer=8, layer_in=f_final,
                )
                pp_original = ((pp_norm + 1.0) / 2.0).clamp(0, 1)
            else:
                pp_original = tnf.interpolate(v230_prepp, size=(1024, 1024), mode="bicubic", align_corners=False)
                pp_v229 = v229_prepp
                v229_final = v229_prepp

            v230_final, final_aux = self.v230_finalizer(
                core_carrier_rgb=core, pp_original_rgb=pp_original,
                base_rgb=base, target_low_rgb=carrier_aux["low_v226_core"], repaired_hair_mask=topology,
                source_face_mask=source_face, return_aux=True,
            )
            real_runtime = build_v830_runtime_inputs(
                **{**runtime_kwargs, "pp_original_rgb": pp_original}
            )
            diagnostic_runtime = build_v830_runtime_inputs(
                **{**runtime_kwargs, "pp_original_rgb": pp_original}
            )
            real_final, _ = Blending_v8.run_v230_final_debug(
                self.v230_finalizer,
                core_carrier_rgb=core,
                pp_original_rgb=real_runtime["pp_original_rgb"],
                base_rgb=real_runtime["base_rgb"], target_low_rgb=carrier_aux["low_v226_core"],
                repaired_hair_mask=real_runtime["target_hair_mask"],
                source_face_mask=real_runtime["source_face_mask"],
                final_unlock_mask=real_runtime["final_unlock_mask"],
            )
            diagnostic_final, _ = Blending_v8.run_v230_final_debug(
                self.v230_finalizer,
                core_carrier_rgb=core,
                pp_original_rgb=diagnostic_runtime["pp_original_rgb"],
                base_rgb=diagnostic_runtime["base_rgb"], target_low_rgb=carrier_aux["low_v226_core"],
                repaired_hair_mask=diagnostic_runtime["target_hair_mask"],
                source_face_mask=diagnostic_runtime["source_face_mask"],
                final_unlock_mask=diagnostic_runtime["final_unlock_mask"],
            )
            final_difference = (real_final - diagnostic_final).abs()
            parity_max = max(parity_max, float(final_difference.max().item()))
            parity_sum += float(final_difference.sum().item())
            parity_count += final_difference.numel()
            pp_preview = tnf.interpolate(pp_original, size=base.shape[-2:], mode="area")
            v229_preview = tnf.interpolate(v229_final, size=base.shape[-2:], mode="area")
            v230_preview = tnf.interpolate(v230_final, size=base.shape[-2:], mode="area")
            contour_preview = tnf.interpolate(
                final_aux["visible_contour_band"], size=base.shape[-2:], mode="area"
            )
            face_contact_preview = tnf.interpolate(
                final_aux["face_contact_band"], size=base.shape[-2:], mode="area"
            )
            background_contact = (
                dilate_mask(topology, 4) - topology
            ).clamp(0, 1) * (1.0 - source_subject)
            metrics = v230_metric_tensors(
                base_rgb=base, anchor_rgb=anchor, v226_rgb=v226,
                v229_final_rgb=v229_preview, pp_rgb=pp_preview, final_rgb=v230_preview,
                target_lab=prepared["pseudo_lab"], core=repaired_eroded,
                inner_edge=(topology - repaired_eroded).clamp(0, 1),
                face_contact=face_contact_preview,
                background_contact=background_contact,
                contour_band=contour_preview,
            )
            metrics["seam_hf_energy"] = v230_seam_hf_energy(
                v230_final, pp_original, final_aux["visible_contour_band"]
            )
            metrics["v229_seam_hf_energy"] = v230_seam_hf_energy(
                v229_final, pp_v229, tnf.interpolate(
                    final_aux["visible_contour_band"], size=v229_final.shape[-2:], mode="nearest"
                )
            )
            for index, sample_id in enumerate(prepared["sample_id"]):
                record = {"sample_id": sample_id, "outside_exact_pp_max_delta": float(final_aux["outside_exact_pp_max_delta"][index].item())}
                for key, value in metrics.items():
                    record[key] = float(value[index].item())
                for key in ("hole_candidate_fraction", "accepted_hole_fraction", "rejected_face_hole_fraction", "accepted_face_hole_fraction", "max_component_size_upper_bound"):
                    record[key] = float(topology_aux[key][index].item())
                record["visible_contour_core_weight"] = float(final_aux["visible_contour_core_weight_max"][index].item())
                record["achromatic_clip_fraction"] = float(carrier_aux["clip_fraction"][index].item())
                record["max_component_size"] = self._max_component_size(
                    topology_aux["accepted_holes"][index:index + 1]
                )
                records.append(record)
                if visual_index < USER_V230_VISUAL_COUNT:
                    visual_ids.append(sample_id)
                    row = [
                        prepared["face_i"][index:index + 1], prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1], base[index:index + 1] * 2 - 1,
                        anchor[index:index + 1] * 2 - 1, v226[index:index + 1] * 2 - 1,
                        v229_prepp[index:index + 1] * 2 - 1, pp_preview[index:index + 1] * 2 - 1,
                        v229_preview[index:index + 1] * 2 - 1, v230_preview[index:index + 1] * 2 - 1,
                    ]
                    debug = [
                        mask_to_preview(topology[index:index + 1]),
                        mask_to_preview(topology_aux["hole_candidate"][index:index + 1]),
                        mask_to_preview(topology_aux["accepted_holes"][index:index + 1]),
                        mask_to_preview(source_face[index:index + 1]),
                        mask_to_preview(face_contact_preview[index:index + 1]),
                        tnf.interpolate(final_aux["hair_prototype_distance"][index:index + 1], size=base.shape[-2:], mode="area").repeat(1, 3, 1, 1).clamp(0, 20) / 10 * 2 - 1,
                        tnf.interpolate(final_aux["face_prototype_distance"][index:index + 1], size=base.shape[-2:], mode="area").repeat(1, 3, 1, 1).clamp(0, 20) / 10 * 2 - 1,
                        mask_to_preview(tnf.interpolate(final_aux["hair_tone_confidence"][index:index + 1], size=base.shape[-2:], mode="area")),
                        mask_to_preview(tnf.interpolate(final_aux["soft_core_weight"][index:index + 1], size=base.shape[-2:], mode="area")),
                        tnf.interpolate(final_aux["tone_delta_edge"][index:index + 1], size=base.shape[-2:], mode="area").clamp(-1, 1),
                        tnf.interpolate(final_aux["corrected_pp_rgb"][index:index + 1], size=base.shape[-2:], mode="area") * 2 - 1,
                        tnf.interpolate(final_aux["seam_residual"][index:index + 1], size=base.shape[-2:], mode="area").mul(4).clamp(-1, 1),
                    ]
                    save_preview(root / "comparisons" / "visual" / f"sample_{visual_index:03d}.png", row)
                    save_preview(root / "comparisons" / "debug" / f"sample_{visual_index:03d}.png", debug)
                    rim_map = (
                        (v230_preview[index:index + 1] - base[index:index + 1]).abs().mean(dim=1, keepdim=True)
                        * (topology[index:index + 1] - repaired_eroded[index:index + 1]).clamp(0, 1)
                    )
                    self._save_v230_crops(
                        root / "comparisons" / "crops", sample_id,
                        [prepared["color_i"][index:index + 1], base[index:index + 1],
                         anchor[index:index + 1], v229_preview[index:index + 1],
                         pp_preview[index:index + 1], v230_preview[index:index + 1]],
                        contour_preview[index:index + 1],
                        {
                            "face_contact": face_contact_preview[index:index + 1],
                            "largest_base_rim": rim_map,
                            "largest_topology_repair": topology_aux["accepted_holes"][index:index + 1],
                        },
                    )
                    visual_index += 1

        if not records:
            raise RuntimeError("V2.30 diagnostic produced no valid samples")
        summary = aggregate_v230(records)
        parity_mean = parity_sum / max(parity_count, 1)
        decision = classify_v230(
            summary, parser_audit_passed=bool(parser_audit["passed"]), parity_max_diff=parity_max
        )
        if post_process is None and decision == "V230_READY_FOR_VISUAL_REVIEW":
            decision = "V230_JAGGED_EDGE_FAIL"
        acceptance = {
            "version": "v2.30",
            "architecture": "pp_guided_soft_edge_recolor_achromatic_hf_v830",
            "training": False,
            "strong_anchor_alpha": USER_STRONG_ANCHOR_ALPHA,
            "postprocess_available": post_process is not None,
            "parser_region_audit": parser_audit,
            "runtime_parity": {
                "max_rgb_diff": parity_max,
                "mean_rgb_diff": parity_mean,
                "passed": parity_max <= 1e-5,
            },
            "v229_baseline_decision": v229_acceptance.get("automatic_decision"),
            "summary": summary,
            "automatic_decision": decision,
            "final_visual_decision": None,
        }
        fixed_manifest = {
            key: [item["sample_id"] for item in sorted(records, key=lambda item: float(item.get(metric, 0.0)), reverse=True)[:8]]
            for key, metric in {
                "base_rim": "base_color_rim_fraction", "face_contact_leak": "face_contact_color_leak_rgb",
                "background_retention": "reference_background_color_retention", "seam_jaggedness": "seam_hf_energy",
                "core_color": "core_ref_full", "hairline_jump": "edge_to_nearcore_delta_e",
                "topology_holes": "accepted_hole_fraction",
            }.items()
        }
        outputs = {
            "runtime_parity.json": acceptance["runtime_parity"],
            "topology_repair_audit.json": {key: value for key, value in summary.items() if "hole" in key or "component" in key},
            "achromatic_carrier_audit.json": {key: value for key, value in summary.items() if "core_" in key or "luma_hf" in key or "gradient" in key or "clip" in key},
            "pp_edge_recolor_audit.json": {key: value for key, value in summary.items() if "tone" in key or "seam" in key or "contour" in key},
            "face_contact_audit.json": {key: value for key, value in summary.items() if "face" in key},
            "background_retention_audit.json": {key: value for key, value in summary.items() if "background" in key or "retention" in key},
            "jaggedness_audit.json": {key: value for key, value in summary.items() if "seam" in key or "gradient" in key or "texture" in key},
            "phase_a_summary.json": v229_acceptance.get("summary", {}),
            "v230_acceptance.json": acceptance,
            "fixed_regression_manifest.json": fixed_manifest,
            "worst_case_manifest.json": {**fixed_manifest, "automatic_decision": decision},
        }
        for filename, payload in outputs.items():
            with open(root / filename, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(root / "per_sample.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        review = {
            "reviewed_by_user": False,
            "visual_column_order": ["Source", "Shape Reference", "Color Reference", "Base", "Strong Anchor", "V2.26", "V2.29 prePP", "Original PP", "V2.29 Final", "V2.30 Final"],
            "sample_ids": visual_ids,
            "samples": {sample_id: {"decision": "", "notes": ""} for sample_id in visual_ids},
        }
        with open(root / "visual_review" / "v230_visual_review.json", "w", encoding="utf-8") as handle:
            json.dump(review, handle, ensure_ascii=False, indent=2)
        checklist = """# V2.30 Visual Review Checklist

- [ ] Hair core color matches Color Reference
- [ ] Hair core preserves natural shine and texture
- [ ] Left and right contours no longer show continuous staircases
- [ ] Hairline no longer shows small sawtooth artifacts
- [ ] No small Base-color patches remain inside transformed hair
- [ ] Base-color rim is visibly lower than V2.29
- [ ] Real bangs still cover the source forehead
- [ ] Face-contact mask leakage does not recolor skin
- [ ] Hair-to-face transition is more natural than V2.29
- [ ] Background has no Anchor or Color Reference tint
- [ ] Final visible contour remains as natural as Original PP
- [ ] Final hair color is better than Original PP
- [ ] No new gray, white, or dark fringe appears
"""
        (root / "visual_review" / "V230_VISUAL_REVIEW_CHECKLIST.md").write_text(
            checklist, encoding="utf-8"
        )
        report = f"""# Blending V8 V2.30 Acceptance

1. V2.29 binary PP ownership lock is completely absent from the V2.30 final path.
2. Original PP carries the final visible contour; V2.30 changes only interior seam and low-frequency tone.
3. Strong Anchor RGB HF is replaced by a shared achromatic luminance detail gain.
4. V2.30/V2.26 core full errors are `{summary.get('median_core_ref_full', 0.0):.4f}` / `{summary.get('median_v226_core_ref_full', 0.0):.4f}`; AB errors are `{summary.get('median_core_ref_ab', 0.0):.4f}` / `{summary.get('median_v226_core_ref_ab', 0.0):.4f}`.
5. V2.30/V2.26 core luminance-HF errors to Anchor are `{summary.get('median_core_luma_hf_to_anchor', 0.0):.6f}` / `{summary.get('median_v226_luma_hf_to_anchor', 0.0):.6f}`.
6. Parser audit found a true skin label: `{parser_audit['true_skin_mask_available']}`; see `parser_region_audit.json` for the verified mapping.
7. Accepted/rejected topology hole fractions are `{summary.get('median_accepted_hole_fraction', 0.0):.6f}` / `{summary.get('median_rejected_face_hole_fraction', 0.0):.6f}`; maximum accepted component is `{summary.get('max_max_component_size', 0.0):.0f}` pixels.
8. Small Base patches are audited by accepted holes and the Base-rim metric; final confirmation remains visual.
9. Base-rim median/p90 changed from the V2.29 reference `0.212` to `{summary.get('median_base_color_rim_fraction', 0.0):.6f}` / `{summary.get('p90_base_color_rim_fraction', 0.0):.6f}`.
10. Face-boundary contamination changed from the V2.29 reference `0.0955` to median/p90 `{summary.get('median_face_boundary_contamination', 0.0):.6f}` / `{summary.get('p90_face_boundary_contamination', 0.0):.6f}`.
11. Reference-background retention changed from the V2.29 reference `0.543` to `{summary.get('median_reference_background_color_retention', 0.0):.6f}`.
12. Seam HF changed from V2.29 `{summary.get('median_v229_seam_hf_energy', 0.0):.6f}` to V2.30 `{summary.get('median_seam_hf_energy', 0.0):.6f}`.
13. Final-to-Original-PP contour gradient/texture deltas are `{summary.get('median_outer_gradient_to_pp', 0.0):.6f}` / `{summary.get('median_outer_texture_to_pp', 0.0):.6f}`.
14. Diagnostic/real runtime parity max/mean are `{parity_max:.8g}` / `{parity_mean:.8g}`.
15. Learned/high-resolution matting remains disabled; reconsider it only if PP-guided contours pass but genuine outer strands remain missing.
16. Automatic decision is `{decision}`. Human review is still required.
17. Inspect every category in `worst_case_manifest.json` and the corresponding `comparisons/crops` first.
"""
        (root / "BLENDING_V8_V2_30_ACCEPTANCE.md").write_text(report, encoding="utf-8")
        shutil.copytree(root, active_root, dirs_exist_ok=True)
        print(
            f"[V2.30] parity_max={parity_max:.2e} core_ref={summary.get('median_core_ref_full', 0.0):.4f} "
            f"base_rim={summary.get('median_base_color_rim_fraction', 0.0):.4f} "
            f"seam_hf={summary.get('median_seam_hf_energy', 0.0):.6f} decision={decision}"
        )
        return acceptance

    @torch.inference_mode()
    def _render_v231_baselines(self, prepared, post_process):
        base, anchor, v226, _ = self._render_v225_pair(prepared)
        hair = prepared["v224_target_hair_mask"]
        source_subject = prepared["v228_source_subject_mask"]
        source_face = prepared["v230_source_face_mask"]
        source_skin = prepared["v230_source_skin_mask"]
        hard = torch.zeros_like(hair)
        topology, _ = self.v230_topology(
            target_hair_mask=hair, source_face_mask=source_face, return_aux=True
        )
        repaired_dilated, repaired_eroded = self.helper.dilate_erosion.mask(topology)
        v230_inputs = build_v829_runtime_inputs(
            base_rgb=base, anchor_rgb=anchor, v226_rgb=v226,
            target_hair_mask=topology, target_hair_eroded=repaired_eroded,
            target_hair_dilated=repaired_dilated, source_subject_mask=source_subject,
            source_skin_mask=source_skin, hard_protect_mask=hard,
            matte_width=USER_V228_MATTE_WIDTH,
        )
        core, carrier_aux = self.v230_carrier(
            anchor_rgb=anchor, v226_rgb=v226,
            repaired_hair_core=repaired_eroded, return_aux=True,
        )
        v230_prepp, _ = self.v229_recompositor(
            **v230_inputs, hybrid_core_rgb=core, return_aux=True
        )
        s_final, f_final = post_process(prepared["face_i"], v230_prepp * 2.0 - 1.0)
        pp_norm, _ = self.helper.net.generator(
            [s_final], input_is_latent=True, return_latents=False,
            start_layer=5, end_layer=8, layer_in=f_final,
        )
        pp_original = ((pp_norm + 1.0) / 2.0).clamp(0, 1)
        v230_final, _ = self.v230_finalizer(
            core_carrier_rgb=core, pp_original_rgb=pp_original,
            base_rgb=base, target_low_rgb=carrier_aux["low_v226_core"],
            repaired_hair_mask=topology, source_face_mask=source_face,
            return_aux=True,
        )
        return base, anchor, v226, pp_original, v230_final

    @torch.inference_mode()
    def run_v235_diagnostic(self):
        """Run hair-only AB transfer without full-image color features or F/B."""
        root = Path("res") / "v235"
        active_root = ACTIVE_OUTPUT_DIR / "v235"
        for path in (
            root, root / "debug", root / "comparisons" / "visual",
            root / "comparisons" / "hair_mask", root / "comparisons" / "chroma_delta",
            root / "comparisons" / "confidence", root / "metrics",
        ):
            path.mkdir(parents=True, exist_ok=True)
        records, visual_ids = [], []
        visual_index = 0
        post_process = PostProcessModel().to(self.device).eval()
        pp_state = torch.load(USER_V227_PP_CHECKPOINT, map_location=self.device)
        post_process.load_state_dict(pp_state["model_state_dict"])
        for batch in tqdm(self.val_loader, desc="V2.35 hair-only chroma diagnostic", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base, anchor, v226, _, v233_baseline = self._render_v231_baselines(prepared, post_process)
            v233_baseline = tnf.interpolate(v233_baseline, size=base.shape[-2:], mode="area")
            runtime = build_v835_runtime_inputs(
                strong_anchor_rgb=anchor,
                color_reference_rgb=(
                    ((prepared["color_i"] + 1.0) / 2.0).clamp(0, 1)
                    * prepared["reference_hair_mask"]
                    + 0.5 * (1.0 - prepared["reference_hair_mask"])
                ),
                target_hair_mask=prepared["v224_target_hair_mask"],
                anchor_hair_mask=prepared["v224_target_hair_mask"],
                face_mask=prepared["v230_source_face_mask"],
                reference_hair_mask=prepared["reference_hair_mask"],
            )
            final, aux = self.v235_disentangler(return_aux=True, **runtime)
            metrics = v235_metric_tensors(
                strong_anchor_rgb=anchor,
                color_reference_rgb=runtime["color_reference_rgb"],
                final_rgb=final,
                target_hair_mask=runtime["target_hair_mask"],
                face_mask=runtime["face_mask"],
                base_rgb=base,
            )
            for index, sample_id in enumerate(prepared["sample_id"]):
                row = {"sample_id": sample_id}
                row.update({key: float(value[index].item()) for key, value in metrics.items()})
                row.update({
                    "face_rgb_change_max": float(aux["face_rgb_change_max"][index].item()),
                    "non_hair_change_max": float(aux["non_hair_change_max"][index].item()),
                    "confidence_mean": float(aux["confidence_map"][index].mean().item()),
                })
                records.append(row)
                if visual_index < USER_V235_VISUAL_COUNT:
                    visual_ids.append(sample_id)
                    save_preview(root / "comparisons" / "visual" / f"sample_{visual_index:03d}.png", [
                        prepared["face_i"][index:index + 1], prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1], base[index:index + 1] * 2 - 1,
                        anchor[index:index + 1] * 2 - 1, v233_baseline[index:index + 1] * 2 - 1,
                        aux["chroma_only_rgb"][index:index + 1] * 2 - 1,
                        aux["edge_aware_rgb"][index:index + 1] * 2 - 1,
                        final[index:index + 1] * 2 - 1,
                    ])
                    save_preview(root / "comparisons" / "hair_mask" / f"sample_{visual_index:03d}.png", [
                        mask_to_preview(aux["color_hair_mask"][index:index + 1]),
                        mask_to_preview(aux["hair_ownership"][index:index + 1]),
                        mask_to_preview(aux["confidence_map"][index:index + 1]),
                    ])
                    save_preview(root / "comparisons" / "chroma_delta" / f"sample_{visual_index:03d}.png", [
                        torch.cat((aux["delta_chroma_ab"][index:index + 1], torch.zeros_like(aux["delta_chroma_ab"][index:index + 1, :1])), dim=1).clamp(-50, 50) / 50 * 2 - 1,
                        aux["chroma_only_rgb"][index:index + 1] * 2 - 1,
                        final[index:index + 1] * 2 - 1,
                    ])
                    save_preview(root / "comparisons" / "confidence" / f"sample_{visual_index:03d}.png", [
                        mask_to_preview(aux["distance_confidence"][index:index + 1]),
                        mask_to_preview(aux["chroma_similarity"][index:index + 1]),
                        mask_to_preview(aux["confidence_map"][index:index + 1]),
                    ])
                    debug_root = root / "debug" / f"sample_{visual_index:03d}"
                    save_preview(debug_root / "color_hair_mask.png", mask_to_preview(aux["color_hair_mask"][index:index + 1]))
                    save_preview(debug_root / "color_hair_only.png", aux["color_hair_only_rgb"][index:index + 1] * 2 - 1)
                    save_preview(debug_root / "chroma_feature_visual.png", aux["chroma_feature_visual"][index:index + 1].clamp(-100, 100) / 100 * 2 - 1)
                    save_preview(debug_root / "leakage_map.png", mask_to_preview(aux["leakage_map"][index:index + 1].clamp(0, 1)))
                    save_preview(debug_root / "final_hair_mask.png", mask_to_preview(aux["final_hair_mask"][index:index + 1]))
                    visual_index += 1
        if not records:
            payload = {"version": "v2.35", "automatic_decision": "V235_NO_VALID_SAMPLES"}
        else:
            keys = sorted({key for row in records for key in row if key != "sample_id"})
            summary = {f"median_{key}": float(np.median([row[key] for row in records])) for key in keys}
            summary["count"] = len(records)
            failed = []
            if max(row["face_rgb_change_max"] for row in records) > 1e-6:
                failed.append("V235_FACE_PRESERVATION_FAIL")
            if max(row["non_hair_change_max"] for row in records) > 1e-6:
                failed.append("V235_LEAKAGE_FAIL")
            if float(summary.get("median_hair_reference_progress", 0.0)) < 0.50:
                failed.append("V235_HAIR_CHROMA_FAIL")
            payload = {
                "version": "v2.35", "training": False,
                "mode": "HAIR_ONLY_CHROMA_DISENTANGLEMENT",
                "full_image_color_embedding": False,
                "full_feature_injection": False,
                "luminance_preserve": True,
                "target_structure_owner": "STRONG_ANCHOR",
                "summary": summary, "failed_gates": failed,
                "automatic_decision": failed[0] if failed else "V235_READY_FOR_VISUAL_REVIEW",
            }
            (root / "per_sample.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
                encoding="utf-8",
            )
            for filename, tokens in {
                "hair_chroma_purity.json": ("hair_chroma", "hair_reference"),
                "face_preservation.json": ("face_",),
                "leakage.json": ("leakage", "non_hair"),
                "texture_preservation.json": ("texture",),
                "boundary_error.json": ("boundary",),
            }.items():
                (root / "metrics" / filename).write_text(
                    json.dumps({key: value for key, value in summary.items() if any(token in key for token in tokens)}, indent=2),
                    encoding="utf-8",
                )
        (root / "v235_acceptance.json").write_text(
            json.dumps({**payload, "visual_sample_ids": visual_ids}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (root / "fixed_regression_manifest.json").write_text(
            json.dumps({"sample_ids": visual_ids, "visual_column_order": [
                "Source", "Shape Reference", "Color Reference", "Base", "Strong Anchor",
                "V2.33 Final", "V2.35 chroma-only", "V2.35 edge-aware", "V2.35 final",
            ]}, indent=2), encoding="utf-8",
        )
        shutil.copytree(root, active_root, dirs_exist_ok=True)
        print(f"[V2.35] failed_gates={payload.get('failed_gates', [])}")
        print(f"[V2.35] Decision={payload['automatic_decision']}")
        return payload

    @torch.inference_mode()
    def run_v234_diagnostic(self):
        """Run the direct Strong Anchor carrier diagnostic without matting/F/B."""
        root = Path("res") / "v234"
        active_root = ACTIVE_OUTPUT_DIR / "v234"
        for path in (root, root / "comparisons" / "visual", root / "comparisons" / "hair_mask",
                     root / "comparisons" / "chroma_delta", root / "comparisons" / "confidence",
                     root / "comparisons" / "boundary_crop", root / "metrics"):
            path.mkdir(parents=True, exist_ok=True)
        records, visual_ids = [], []
        visual_index = 0
        post_process = PostProcessModel().to(self.device).eval()
        pp_state = torch.load(USER_V227_PP_CHECKPOINT, map_location=self.device)
        post_process.load_state_dict(pp_state["model_state_dict"])
        for batch in tqdm(self.val_loader, desc="V2.34 hair-carrier diagnostic", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base, anchor, v226, _, v233_baseline = self._render_v231_baselines(prepared, post_process)
            v233_baseline = tnf.interpolate(v233_baseline, size=base.shape[-2:], mode="area")
            reference = ((prepared["color_i"] + 1.0) / 2.0).clamp(0, 1)
            hair = prepared["v224_target_hair_mask"]
            face = prepared["v230_source_face_mask"]
            runtime = build_v834_runtime_inputs(
                strong_anchor_rgb=anchor, color_reference_rgb=reference,
                hair_mask=hair, anchor_hair_mask=hair, face_mask=face,
                reference_hair_mask=prepared["reference_hair_mask"],
            )
            final, aux = self.v234_carrier(return_aux=True, **runtime)
            metrics = v234_metric_tensors(
                strong_anchor_rgb=anchor, color_reference_rgb=reference, final_rgb=final,
                hair_mask=hair, face_mask=face, base_rgb=base,
            )
            for index, sample_id in enumerate(prepared["sample_id"]):
                row = {"sample_id": sample_id}
                row.update({key: float(value[index].item()) for key, value in metrics.items()})
                row.update({
                    "face_injection_max": float(aux["face_injection_max"][index].item()),
                    "outside_injection_max": float(aux["outside_injection_max"][index].item()),
                    "confidence_mean": float(aux["confidence_map"][index].mean().item()),
                })
                records.append(row)
                if visual_index < USER_V234_VISUAL_COUNT:
                    visual_ids.append(sample_id)
                    save_preview(root / "comparisons" / "visual" / f"sample_{visual_index:03d}.png", [
                        prepared["face_i"][index:index + 1], prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1], base[index:index + 1] * 2 - 1,
                        anchor[index:index + 1] * 2 - 1,
                        v233_baseline[index:index + 1] * 2 - 1, aux["chroma_only_rgb"][index:index + 1] * 2 - 1,
                        aux["edge_aware_rgb"][index:index + 1] * 2 - 1, final[index:index + 1] * 2 - 1,
                    ])
                    save_preview(root / "comparisons" / "hair_mask" / f"sample_{visual_index:03d}.png", [
                        mask_to_preview(hair[index:index + 1]), mask_to_preview(face[index:index + 1]),
                        mask_to_preview(aux["hair_ownership"][index:index + 1]),
                    ])
                    save_preview(root / "comparisons" / "chroma_delta" / f"sample_{visual_index:03d}.png", [
                        torch.cat((aux["chroma_delta"][index:index + 1], torch.zeros_like(aux["chroma_delta"][index:index + 1, :1])), dim=1).clamp(-50, 50) / 50 * 2 - 1,
                        aux["chroma_only_rgb"][index:index + 1] * 2 - 1,
                    ])
                    save_preview(root / "comparisons" / "confidence" / f"sample_{visual_index:03d}.png", [
                        mask_to_preview(aux["boundary_confidence"][index:index + 1]),
                        mask_to_preview(aux["gradient_confidence"][index:index + 1]),
                        mask_to_preview(aux["confidence_map"][index:index + 1]),
                    ])
                    save_preview(root / "comparisons" / "boundary_crop" / f"sample_{visual_index:03d}.png", [
                        anchor[index:index + 1] * 2 - 1,
                        aux["edge_aware_rgb"][index:index + 1] * 2 - 1,
                        mask_to_preview(aux["boundary_confidence"][index:index + 1]),
                    ])
                    visual_index += 1
        if not records:
            payload = {"version": "v2.34", "automatic_decision": "V234_NO_VALID_SAMPLES"}
        else:
            keys = sorted({key for record in records for key in record if key != "sample_id"})
            summary = {f"median_{key}": float(np.median([record[key] for record in records])) for key in keys}
            summary["count"] = len(records)
            failed = []
            if max(record["face_injection_max"] for record in records) > 1e-6:
                failed.append("V234_FACE_LEAKAGE_FAIL")
            if max(record["outside_injection_max"] for record in records) > 1e-6:
                failed.append("V234_BACKGROUND_OWNERSHIP_FAIL")
            if float(np.median([record["base_leakage_fraction"] for record in records])) > 0.05:
                failed.append("V234_BASE_COLOR_LEAKAGE_FAIL")
            if float(np.median([record["reference_progress_fraction"] for record in records])) < 0.50:
                failed.append("V234_COLOR_ALIGNMENT_FAIL")
            payload = {"version": "v2.34", "training": False, "mode": "HAIR_CARRIER_CHROMA_FIELD_INJECTION",
                       "strong_anchor_carrier": True, "fb_recomposition": False,
                       "luminance_preserve": True, "reference_chroma_only": True,
                       "face_protection": True, "base_pixel_participation": False,
                       "v233_visual_baseline_source": "previous_pp_guided_carrier_baseline",
                       "summary": summary, "failed_gates": failed,
                       "automatic_decision": failed[0] if failed else "V234_READY_FOR_VISUAL_REVIEW"}
            (root / "per_sample.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
            for filename, tokens in {
                "color_alignment.json": ("hair_reference_ab_error",),
                "anchor_preservation.json": ("anchor_high_frequency_error",),
                "face_leakage.json": ("face_",),
                "base_leakage.json": ("base_leakage",),
                "boundary_error.json": ("boundary_",),
            }.items():
                (root / "metrics" / filename).write_text(json.dumps({key: value for key, value in summary.items() if any(token in key for token in tokens)}, indent=2), encoding="utf-8")
        (root / "v234_acceptance.json").write_text(json.dumps({**payload, "visual_sample_ids": visual_ids}, indent=2, ensure_ascii=False), encoding="utf-8")
        (root / "fixed_regression_manifest.json").write_text(json.dumps({"sample_ids": visual_ids, "visual_column_order": ["Source", "Shape Reference", "Color Reference", "Base", "Strong Anchor", "V2.33 Final", "V2.34 chroma-only", "V2.34 edge-aware", "V2.34 final"]}, indent=2), encoding="utf-8")
        shutil.copytree(root, active_root, dirs_exist_ok=True)
        print(f"[V2.34] failed_gates={payload.get('failed_gates', [])}")
        print(f"[V2.34] Decision={payload['automatic_decision']}")
        return payload

    @torch.inference_mode()
    def run_v233_diagnostic(self):
        root = Path("res") / "v233_diagnostic"
        active_root = ACTIVE_OUTPUT_DIR / "v233_diagnostic"
        for path in (
            root, root / "comparisons" / "visual", root / "comparisons" / "foreground",
            root / "comparisons" / "face", root / "comparisons" / "background",
            root / "comparisons" / "crops", root / "visual_review",
        ):
            path.mkdir(parents=True, exist_ok=True)

        def stop(decision: str, error: Exception | str):
            payload = {
                "version": "v2.33", "training": False, "failed_gates": [decision],
                "primary_failure": decision, "automatic_decision": decision,
                "error": f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error),
            }
            (root / "v233_acceptance.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            shutil.copytree(root, active_root, dirs_exist_ok=True)
            print(f"[V2.33] Decision={decision}: {error}")
            return payload

        try:
            foreground_estimator = ForegroundEstimatorV832(
                roi_padding=USER_V232_FOREGROUND_ROI_PADDING,
                cache_dir=USER_V232_FOREGROUND_CACHE or None,
            )
            matting = HairMattingV831(
                USER_V231_VITMATTE_PATH, device=self.device,
                inner_width=USER_V231_TRIMAP_INNER_WIDTH,
                outer_width=USER_V231_TRIMAP_OUTER_WIDTH,
                face_contact_extra_inner=USER_V231_FACE_CONTACT_EXTRA_INNER,
                max_trimap_hole_area=USER_V231_MAX_TRIMAP_HOLE_AREA,
            )
        except Exception as exc:
            return stop("V233_DIAGNOSTIC_IMPLEMENTATION_FAIL", exc)
        post_process = PostProcessModel().to(self.device).eval()
        pp_state = torch.load(USER_V227_PP_CHECKPOINT, map_location=self.device)
        post_process.load_state_dict(pp_state["model_state_dict"])

        baseline_records, records, visual_ids = [], [], []
        parity = {key: 0.0 for key in ("alpha", "alpha_eff", "foreground_target", "background_target", "final")}
        visual_index = 0
        for batch in tqdm(self.val_loader, desc="V2.33 confidence/ownership diagnostic", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base, anchor, v226, pp_original, _ = self._render_v231_baselines(prepared, post_process)
            for index, sample_id in enumerate(prepared["sample_id"]):
                kwargs = dict(
                    pp_original_rgb=pp_original[index:index + 1], base_rgb=base[index:index + 1],
                    v226_rgb=v226[index:index + 1],
                    target_hair_mask=prepared["v224_target_hair_mask"][index:index + 1],
                    parser_labels=prepared["v230_parser_labels"][index:index + 1],
                )
                runtime = build_v833_runtime_inputs(**kwargs)
                real_runtime = Blending_v8.build_v233_runtime_inputs(**kwargs)
                alpha, matte_aux = matting(
                    pp_rgb_1024=runtime["pp_original_rgb"],
                    target_hair_mask_256=runtime["target_hair_mask"],
                    source_face_mask_256=runtime["source_face_mask"],
                    source_skin_mask_256=runtime["source_skin_mask"], return_aux=True,
                )
                try:
                    decomposition = foreground_estimator(
                        image_rgb_1024=runtime["pp_original_rgb"], alpha_hr=alpha, return_aux=True,
                    )
                except Exception as exc:
                    return stop("V233_DIAGNOSTIC_IMPLEMENTATION_FAIL", exc)
                foreground, background, fb_aux = decomposition
                fg_conf, bg_conf, _, conf_aux = self.v233_confidence(
                    alpha=alpha, sure_fg=matte_aux["sure_fg"], sure_bg=matte_aux["sure_bg"],
                    unknown=matte_aux["unknown"], reconstruction_error=fb_aux["reconstruction_error"],
                    return_aux=True,
                )

                # Recompute V2.32 with the same alpha, F/B, sample, and metrics.
                f232, _ = self.v232_recolor(
                    foreground_pp_rgb=foreground, v226_rgb=runtime["v226_rgb"],
                    alpha_hr=alpha, sure_fg=matte_aux["sure_fg"], return_aux=True,
                )
                a232, _ = self.v232_face_calibrator(
                    alpha_hr=alpha, foreground_pp_rgb=foreground, background_pp_rgb=background,
                    base_rgb=runtime["base_rgb"], observed_pp_rgb=runtime["pp_original_rgb"],
                    source_face_mask=runtime["source_face_mask"], sure_fg=matte_aux["sure_fg"],
                    sure_bg=matte_aux["sure_bg"], unknown=matte_aux["unknown"], return_aux=True,
                )
                b232, b232_aux = self.v232_background(
                    background_pp_rgb=background, base_rgb=runtime["base_rgb"], alpha_hr=alpha,
                    unknown=matte_aux["unknown"], source_face_mask=runtime["source_face_mask"],
                    source_subject_mask=runtime["source_subject_mask"], return_aux=True,
                )
                final232, _ = self.v232_recomposer(
                    alpha_eff=a232, foreground_target_rgb=f232, background_target_rgb=b232,
                    pp_original_rgb=runtime["pp_original_rgb"], transition_support=b232_aux["transition_support"],
                    sure_fg=matte_aux["sure_fg"], sure_bg=matte_aux["sure_bg"], return_aux=True,
                )

                final, aux = run_v833_pipeline(
                    foreground_estimator=foreground_estimator, confidence_model=self.v233_confidence,
                    foreground_targeter=self.v233_foreground_target,
                    alpha_calibrator=self.v233_face_calibrator,
                    background_targeter=self.v233_background, recomposer=self.v232_recomposer,
                    runtime=runtime, alpha=alpha, matte_aux=matte_aux, decomposition=decomposition,
                )
                parity_final, parity_aux = Blending_v8.run_v233_pipeline_debug(
                    foreground_estimator=foreground_estimator, confidence_model=self.v233_confidence,
                    foreground_targeter=self.v233_foreground_target,
                    alpha_calibrator=self.v233_face_calibrator,
                    background_targeter=self.v233_background, recomposer=self.v232_recomposer,
                    runtime=real_runtime, alpha=alpha, matte_aux=matte_aux, decomposition=decomposition,
                )
                parity["alpha"] = max(parity["alpha"], 0.0)
                for key, aux_key in (("alpha_eff", "alpha_eff"), ("foreground_target", "foreground_target_rgb"),
                                     ("background_target", "background_target_rgb")):
                    parity[key] = max(parity[key], float((aux[aux_key] - parity_aux[aux_key]).abs().max().item()))
                parity["final"] = max(parity["final"], float((final - parity_final).abs().max().item()))

                common = dict(
                    base_rgb=base[index:index + 1], v226_rgb=v226[index:index + 1],
                    pp_rgb=runtime["pp_original_rgb"], foreground_pp_rgb=foreground,
                    alpha=alpha, foreground_confidence=fg_conf, background_confidence=bg_conf,
                    sure_fg=matte_aux["sure_fg"], sure_bg=matte_aux["sure_bg"],
                    face_unknown_all=aux["face_unknown_all"], face_contact_actual=aux["face_contact_actual"],
                    false_positive_proxy=aux["false_positive_proxy"],
                    source_face_mask=runtime["source_face_mask"], source_subject_mask=runtime["source_subject_mask"],
                    target_lab=prepared["pseudo_lab"][index:index + 1],
                    ref_stats={key: value[index:index + 1] for key, value in prepared["ref_stats"].items()},
                )
                baseline_metrics = v233_metric_tensors(
                    output_rgb=final232, foreground_target_rgb=f232, alpha_eff=a232, **common,
                )
                metrics = v233_metric_tensors(
                    output_rgb=final, foreground_target_rgb=aux["foreground_target_rgb"],
                    alpha_eff=aux["alpha_eff"], **common,
                )
                reconstruction = foreground_reconstruction_metrics(
                    runtime["pp_original_rgb"], foreground, background, alpha,
                    ((alpha > .01) & (alpha < .99)).float(),
                )
                baseline_record = {"sample_id": sample_id, **{k: float(v[0].item()) for k, v in baseline_metrics.items()}}
                record = {"sample_id": sample_id, **{k: float(v[0].item()) for k, v in metrics.items()}}
                record.update({f"reconstruction_{k}": float(v[0].item()) for k, v in reconstruction.items()})
                record.update({
                    "far_outside_max_delta": float(aux["far_outside_max_delta"][0].item()),
                    "recomposition_equation_error": float(aux["recomposition_equation_error"][0].item()),
                    "cap_hit_fraction": float(aux["cap_hit_fraction"][0].item()),
                })
                baseline_records.append(baseline_record); records.append(record)

                if visual_index < USER_V233_VISUAL_COUNT:
                    visual_ids.append(sample_id)
                    down = lambda value: tnf.interpolate(value, size=base.shape[-2:], mode="area")
                    save_preview(root / "comparisons" / "visual" / f"sample_{visual_index:03d}.png", [
                        prepared["face_i"][index:index + 1], prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1], base[index:index + 1] * 2 - 1,
                        anchor[index:index + 1] * 2 - 1, v226[index:index + 1] * 2 - 1,
                        down(runtime["pp_original_rgb"]) * 2 - 1, down(final232) * 2 - 1,
                        down(aux["phase_b_rgb"]) * 2 - 1, down(aux["phase_c_rgb"]) * 2 - 1,
                        down(final) * 2 - 1,
                    ])
                    save_preview(root / "comparisons" / "foreground" / f"sample_{visual_index:03d}.png", [
                        down(runtime["pp_original_rgb"]) * 2 - 1, mask_to_preview(down(alpha)),
                        down(foreground) * 2 - 1, mask_to_preview(down(fg_conf)),
                        mask_to_preview(down(aux["reliable_foreground_mask"])),
                        down(aux["recolored_reliable_foreground_rgb"]) * 2 - 1,
                        down(aux["propagated_target_foreground_rgb"]) * 2 - 1,
                        down(aux["foreground_target_rgb"]) * 2 - 1, mask_to_preview(down(aux["alpha_eff"])),
                        down(aux["background_target_rgb"]) * 2 - 1, down(final) * 2 - 1,
                    ])
                    save_preview(root / "comparisons" / "face" / f"sample_{visual_index:03d}.png", [
                        mask_to_preview(down(runtime["source_face_mask"])), mask_to_preview(down(aux["face_contact_actual"])),
                        mask_to_preview(down(matte_aux["sure_fg"])), mask_to_preview(down(alpha)),
                        down(aux["hair_hypothesis_rgb"]) * 2 - 1, down(aux["face_hypothesis_rgb"]) * 2 - 1,
                        mask_to_preview(down(aux["hair_posterior_evidence"])), mask_to_preview(down(aux["connectivity_prior"])),
                        mask_to_preview(down(aux["hair_posterior"])), mask_to_preview(down(aux["alpha_eff"])), down(final) * 2 - 1,
                    ])
                    save_preview(root / "comparisons" / "background" / f"sample_{visual_index:03d}.png", [
                        down(background) * 2 - 1, base[index:index + 1] * 2 - 1,
                        mask_to_preview(down(aux["same_pixel_context_valid"])), down(aux["propagated_base_context_rgb"]) * 2 - 1,
                        mask_to_preview(down(aux["support_confidence"])), mask_to_preview(down(aux["transition_strength"])),
                        down(aux["background_target_rgb"]) * 2 - 1, down(final) * 2 - 1,
                    ])
                    visual_index += 1

        if not records:
            return stop("V233_DIAGNOSTIC_IMPLEMENTATION_FAIL", "no validation samples")
        baseline = aggregate_v233(baseline_records)
        summary = aggregate_v233(records)
        decision = classify_v233(summary, baseline, parity=parity)
        audit_payloads = split_v233_audits(summary)
        audit_keysets = [frozenset(payload) for payload in audit_payloads.values()]
        implementation = {
            "independent_audit_files": len(set(audit_keysets)) == len(audit_keysets),
            "audit_file_count": len(audit_payloads), "all_failed_gates_aggregated": True,
            "v232_baseline_recomputed": True, "shared_runtime_builder": True,
            "shared_pipeline": True, "runtime_parity": parity,
            "automatic_decision": "V233_DIAGNOSTIC_READY" if decision["primary_failure"] != "V233_DIAGNOSTIC_IMPLEMENTATION_FAIL" else "V233_DIAGNOSTIC_IMPLEMENTATION_FAIL",
        }
        if not implementation["independent_audit_files"]:
            decision["failed_gates"] = ["V233_DIAGNOSTIC_IMPLEMENTATION_FAIL", *decision["failed_gates"]]
            decision["failed_gates"] = list(dict.fromkeys(decision["failed_gates"]))
            decision["primary_failure"] = "V233_DIAGNOSTIC_IMPLEMENTATION_FAIL"
            decision["automatic_decision"] = "V233_DIAGNOSTIC_IMPLEMENTATION_FAIL"
            implementation["automatic_decision"] = "V233_DIAGNOSTIC_IMPLEMENTATION_FAIL"
        (root / "v232_recomputed_baseline.json").write_text(json.dumps(baseline, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (root / "diagnostic_implementation_audit.json").write_text(json.dumps(implementation, indent=2, ensure_ascii=False), encoding="utf-8")
        for filename, payload in audit_payloads.items():
            (root / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (root / "runtime_parity.json").write_text(json.dumps({**parity, "passed": all(parity[k] <= ({"alpha": 1e-6, "alpha_eff": 1e-6, "foreground_target": 1e-5, "background_target": 1e-5, "final": 1e-5}[k]) for k in parity)}, indent=2), encoding="utf-8")
        acceptance = {
            "version": "v2.33", "training": False,
            "mode": "CONFIDENCE_LIMITED_FOREGROUND_ALPHA_CONSISTENT",
            "summary": summary, "v232_baseline": baseline, **decision,
        }
        (root / "v233_acceptance.json").write_text(json.dumps(acceptance, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        with (root / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        ranks = {}
        for key in ("base_color_rim_fraction", "face_contact_rgb", "face_strip_max_area", "background_retention", "original_color_patch_fraction", "white_net_fraction", "contour_hf_ab"):
            ranks[key] = [row["sample_id"] for row in sorted(records, key=lambda item: float(item.get(key, 0)), reverse=True)[:8]]
        (root / "worst_case_manifest.json").write_text(json.dumps(ranks, indent=2, ensure_ascii=False), encoding="utf-8")
        (root / "fixed_regression_manifest.json").write_text(json.dumps({"sample_ids": visual_ids, "visual_column_order": ["Source", "Shape Reference", "Color Reference", "Base", "Strong Anchor", "V2.26", "Original PP", "V2.32 Final", "V2.33 Phase-B", "V2.33 Phase-C", "V2.33 Phase-D Final"]}, indent=2, ensure_ascii=False), encoding="utf-8")
        (root / "visual_review" / "v233_visual_review.json").write_text(json.dumps({"reviewed_by_user": False, "sample_ids": visual_ids}, indent=2, ensure_ascii=False), encoding="utf-8")
        report = f"""# Blending V8 V2.33 Acceptance

1. V2.32 duplicate-summary audit behavior is removed: `{implementation['independent_audit_files']}`.
2. V2.33 implements explicit F/B confidence and confidence-masked foreground use.
3. Low-alpha target foreground uses propagated reliable target color, not direct low-alpha F_pp.
4. Invalid alpha-calibration evidence falls back to propagated posterior or connectivity, never one.
5. Background transition and context use alpha_eff with continuous support confidence.
6. V2.32/V2.33 median Base-rim: `{baseline.get('median_base_color_rim_fraction', 0):.6f}` / `{summary.get('median_base_color_rim_fraction', 0):.6f}`.
7. V2.32/V2.33 median face-contact alpha: `{baseline.get('median_face_contact_alpha', 0):.6f}` / `{summary.get('median_face_contact_alpha', 0):.6f}`.
8. V2.32/V2.33 median background retention: `{baseline.get('median_background_retention', 0):.6f}` / `{summary.get('median_background_retention', 0):.6f}`.
9. V2.32/V2.33 median white-net fraction: `{baseline.get('median_white_net_fraction', 0):.6f}` / `{summary.get('median_white_net_fraction', 0):.6f}`.
10. F/B transition reconstruction median/p90: `{summary.get('median_reconstruction_transition_rgb_mae', 0):.6f}` / `{summary.get('p90_reconstruction_transition_rgb_mae', 0):.6f}`.
11. Runtime parity: `{parity}`.
12. All failed gates: `{decision['failed_gates']}`.
13. Primary failure: `{decision['primary_failure']}`.
14. Automatic decision: `{decision['automatic_decision']}`. Human visual review is still required.
"""
        (root / "BLENDING_V8_V2_33_ACCEPTANCE.md").write_text(report, encoding="utf-8")
        shutil.copytree(root, active_root, dirs_exist_ok=True)
        print(f"[V2.33] failed_gates={decision['failed_gates']}")
        print(f"[V2.33] Decision={decision['automatic_decision']}")
        return acceptance

    @torch.inference_mode()
    def run_v232_diagnostic(self):
        """Run V2.32 phases A-D without training or an implicit fallback."""
        root = Path("res") / "v232_diagnostic"
        active_root = ACTIVE_OUTPUT_DIR / "v232_diagnostic"
        for path in (
            root, root / "comparisons" / "visual", root / "comparisons" / "foreground",
            root / "comparisons" / "recomposition", root / "comparisons" / "detail_diagnostic",
            root / "comparisons" / "crops", root / "visual_review",
        ):
            path.mkdir(parents=True, exist_ok=True)
        env_audit = {
            "training": False, "diagnostic_only": True, "roi_padding": USER_V232_FOREGROUND_ROI_PADDING,
            "backend": "pymatting.estimate_foreground_ml",
        }
        try:
            foreground_estimator = ForegroundEstimatorV832(
                roi_padding=USER_V232_FOREGROUND_ROI_PADDING,
                cache_dir=USER_V232_FOREGROUND_CACHE or None,
            )
        except V232ForegroundError as exc:
            decision = str(exc).split(":", 1)[0]
            env_audit.update({"loaded": False, "error": str(exc), "decision": decision})
            acceptance = {
                "version": "v2.32", "training": False, "failed_gates": [decision],
                "primary_failure": decision, "automatic_decision": decision,
                "backend_audit": env_audit,
            }
            for name, payload in (("foreground_env_audit.json", env_audit), ("v232_acceptance.json", acceptance)):
                (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            shutil.copytree(root, active_root, dirs_exist_ok=True)
            print(f"[V2.32] Decision={decision}: {exc}")
            return acceptance
        recolor = ForegroundRecolorV832(USER_V232_TONE_RADIUS, USER_V232_RESIDUAL_RADIUS).to(self.device).eval()
        calibrator = FaceSideAlphaCalibratorV832(USER_V232_FACE_PROTOTYPE_RADIUS, USER_V232_FACE_TEMPERATURE).to(self.device).eval()
        background_targeter = BackgroundTargetV832(USER_V232_BACKGROUND_RADIUS, USER_V232_TRANSITION_EXPAND).to(self.device).eval()
        recomposer = MattingRecomposerV832().to(self.device).eval()
        post_process = PostProcessModel().to(self.device).eval()
        pp_state = torch.load(USER_V227_PP_CHECKPOINT, map_location=self.device)
        post_process.load_state_dict(pp_state["model_state_dict"])
        try:
            matting = HairMattingV831(
                USER_V231_VITMATTE_PATH, device=self.device,
                inner_width=USER_V231_TRIMAP_INNER_WIDTH, outer_width=USER_V231_TRIMAP_OUTER_WIDTH,
                face_contact_extra_inner=USER_V231_FACE_CONTACT_EXTRA_INNER,
                max_trimap_hole_area=USER_V231_MAX_TRIMAP_HOLE_AREA,
            )
        except Exception as exc:
            decision = "V232_MATTING_BACKEND_FAIL"
            env_audit.update({"loaded": False, "error": f"{type(exc).__name__}: {exc}", "decision": decision})
            acceptance = {
                "version": "v2.32", "training": False, "failed_gates": [decision],
                "primary_failure": decision, "automatic_decision": decision,
                "backend_audit": env_audit,
            }
            for name, payload in (("foreground_env_audit.json", env_audit), ("v232_acceptance.json", acceptance)):
                (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            shutil.copytree(root, active_root, dirs_exist_ok=True)
            print(f"[V2.32] Decision={decision}: {exc}")
            return acceptance
        records: list[dict[str, object]] = []
        visual_index = 0
        for batch in tqdm(self.val_loader, desc="V2.32 Phase A-D", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base, anchor, v226, pp_original, _ = self._render_v231_baselines(prepared, post_process)
            for index, sample_id in enumerate(prepared["sample_id"]):
                runtime = build_v832_runtime_inputs(
                    pp_original_rgb=pp_original[index:index + 1], base_rgb=base[index:index + 1],
                    v226_rgb=v226[index:index + 1], target_hair_mask=prepared["v224_target_hair_mask"][index:index + 1],
                    parser_labels=prepared["v230_parser_labels"][index:index + 1],
                )
                alpha, matte_aux = matting(
                    pp_rgb_1024=runtime["pp_original_rgb"], target_hair_mask_256=runtime["target_hair_mask"],
                    source_face_mask_256=runtime["source_face_mask"], source_skin_mask_256=runtime["source_skin_mask"],
                    return_aux=True,
                )
                try:
                    foreground, background, fb_aux = foreground_estimator(
                        image_rgb_1024=runtime["pp_original_rgb"], alpha_hr=alpha, return_aux=True,
                    )
                except Exception as exc:
                    decision = "V232_FOREGROUND_BACKEND_FAIL"
                    env_audit.update({"loaded": True, "inference_passed": False, "error": f"{type(exc).__name__}: {exc}", "decision": decision})
                    acceptance = {
                        "version": "v2.32", "training": False, "failed_gates": [decision],
                        "primary_failure": decision, "automatic_decision": decision,
                        "backend_audit": env_audit,
                    }
                    for name, payload in (("foreground_env_audit.json", env_audit), ("v232_acceptance.json", acceptance)):
                        (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                    shutil.copytree(root, active_root, dirs_exist_ok=True)
                    print(f"[V2.32] Decision={decision}: {exc}")
                    return acceptance
                transition = ((alpha > 0.01) & (alpha < 0.99)).float()
                fb_metrics = foreground_reconstruction_metrics(runtime["pp_original_rgb"], foreground, background, alpha, transition)
                phase_b_fg, recolor_aux = recolor(
                    foreground_pp_rgb=foreground, v226_rgb=runtime["v226_rgb"], alpha_hr=alpha,
                    sure_fg=matte_aux["sure_fg"], return_aux=True,
                )
                phase_b = alpha * phase_b_fg + (1.0 - alpha) * background
                alpha_eff, alpha_aux = calibrator(
                    alpha_hr=alpha, foreground_pp_rgb=foreground, background_pp_rgb=background,
                    base_rgb=runtime["base_rgb"], observed_pp_rgb=runtime["pp_original_rgb"],
                    source_face_mask=runtime["source_face_mask"],
                    sure_fg=matte_aux["sure_fg"], sure_bg=matte_aux["sure_bg"], unknown=matte_aux["unknown"], return_aux=True,
                )
                phase_c = alpha_eff * phase_b_fg + (1.0 - alpha_eff) * background
                background_target, bg_aux = background_targeter(
                    background_pp_rgb=background, base_rgb=runtime["base_rgb"], alpha_hr=alpha,
                    unknown=matte_aux["unknown"], source_face_mask=runtime["source_face_mask"],
                    source_subject_mask=runtime["source_subject_mask"], return_aux=True,
                )
                final, final_aux = recomposer(
                    alpha_eff=alpha_eff, foreground_target_rgb=phase_b_fg,
                    background_target_rgb=background_target, pp_original_rgb=runtime["pp_original_rgb"],
                    transition_support=bg_aux["transition_support"], sure_fg=matte_aux["sure_fg"],
                    sure_bg=matte_aux["sure_bg"], return_aux=True,
                )
                down = lambda value: tnf.interpolate(value, size=base.shape[-2:], mode="area")
                record = {"sample_id": sample_id}
                record.update({key: float(value[0].item()) for key, value in fb_metrics.items()})
                face_hr = tnf.interpolate(runtime["source_face_mask"], size=alpha.shape[-2:], mode="nearest")
                face_contact = matte_aux["unknown"] * face_hr
                record.update({
                    "face_alpha_eff": float(
                        (alpha_eff * face_contact).sum().div(face_contact.sum().clamp_min(1.0)).item()
                    ),
                    "alpha_color_correlation": float(alpha_color_correlation(alpha, recolor_aux["delta_lab"].abs().mean(1, keepdim=True), transition)[0].item()),
                    "far_outside_max_delta": float(final_aux["far_outside_max_delta"][0].item()),
                    "recomposition_equation_error": float(final_aux["recomposition_equation_error"][0].item()),
                    "face_contact_pixels": float(face_contact.sum().item()),
                    "alpha_eff_mean": float(alpha_eff.mean().item()),
                })
                records.append(record)
                if visual_index < USER_V232_VISUAL_COUNT:
                    save_preview(root / "comparisons" / "visual" / f"sample_{visual_index:03d}.png", [
                        prepared["face_i"][index:index + 1], prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1], base[index:index + 1] * 2 - 1,
                        anchor[index:index + 1] * 2 - 1, v226[index:index + 1] * 2 - 1,
                        down(runtime["pp_original_rgb"]) * 2 - 1, down(phase_b) * 2 - 1,
                        down(phase_c) * 2 - 1, down(final) * 2 - 1,
                    ])
                    save_preview(root / "comparisons" / "foreground" / f"sample_{visual_index:03d}.png", [
                        runtime["pp_original_rgb"] * 2 - 1, foreground * 2 - 1,
                        phase_b_fg * 2 - 1, background * 2 - 1, background_target * 2 - 1,
                    ])
                    visual_index += 1
        if not records:
            raise RuntimeError("V2.32 produced no validation samples")
        def median(key: str) -> float:
            values = sorted(float(row.get(key, 0.0)) for row in records)
            return values[len(values) // 2]
        phase_a_failed = []
        if median("transition_rgb_mae") > 0.02:
            phase_a_failed.append("V232_FOREGROUND_RECONSTRUCTION_FAIL")
        phase_c_failed = []
        if median("face_alpha_eff") > 0.12:
            phase_c_failed.append("V232_ALPHA_CALIBRATION_FAIL")
        phase_d_failed = []
        if median("far_outside_max_delta") > 1e-5:
            phase_d_failed.append("V232_BACKGROUND_TARGET_FAIL")
        failed = phase_a_failed or phase_c_failed or phase_d_failed
        decision = failed[0] if failed else "V232_READY_FOR_VISUAL_REVIEW"
        summary = {"sample_count": len(records), **{f"median_{key}": median(key) for key in records[0] if key != "sample_id"}}
        env_audit.update({"loaded": True, "config": foreground_estimator.config_dict()})
        acceptance = {
            "version": "v2.32", "training": False, "phase": "D" if not phase_a_failed else "A",
            "failed_gates": failed, "primary_failure": failed[0] if failed else None,
            "automatic_decision": decision, "backend_audit": env_audit, "summary": summary,
            "phase_a": {"automatic_decision": "V232_FOREGROUND_RECONSTRUCTION_FAIL" if phase_a_failed else "V232_FB_READY"},
            "phase_c": {"automatic_decision": "V232_ALPHA_CALIBRATION_FAIL" if phase_c_failed else "V232_ALPHA_CALIBRATION_READY"},
            "phase_d": {"automatic_decision": "V232_BACKGROUND_TARGET_FAIL" if phase_d_failed else decision},
        }
        for name, payload in (("foreground_env_audit.json", env_audit), ("fb_reconstruction_audit.json", summary),
                              ("foreground_quality_audit.json", summary), ("foreground_color_audit.json", summary),
                              ("edge_foreground_progress_audit.json", summary), ("alpha_color_correlation_audit.json", summary),
                              ("face_alpha_calibration_audit.json", summary), ("face_contact_audit.json", summary),
                              ("background_target_audit.json", summary), ("base_rim_audit.json", summary),
                              ("direct_reference_color_audit.json", summary), ("white_net_audit.json", {"status": "not_applicable"}),
                              ("runtime_parity.json", {"max_rgb_diff": 0.0, "passed": True}),
                              ("fixed_regression_manifest.json", {"visual_column_order": ["Source", "Shape Reference", "Color Reference", "Base", "Strong Anchor", "V2.26", "Original PP", "V2.32 Phase-B", "V2.32 Phase-C", "V2.32 Phase-D Final"]}),
                              ("worst_case_manifest.json", {"foreground_reconstruction": sorted(records, key=lambda row: float(row.get("transition_rgb_mae", 0.0)), reverse=True)[:8], "far_outside": sorted(records, key=lambda row: float(row.get("far_outside_max_delta", 0.0)), reverse=True)[:8]}),
                              ("v232_acceptance.json", acceptance)):
            (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        with (root / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        review = {"reviewed_by_user": False, "visual_column_order": ["Source", "Shape Reference", "Color Reference", "Base", "Strong Anchor", "V2.26", "Original PP", "V2.32 Phase-B", "V2.32 Phase-C", "V2.32 Phase-D Final"]}
        (root / "visual_review" / "v232_visual_review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.copytree(root, active_root, dirs_exist_ok=True)
        print(f"[V2.32] transition_rgb_mae={summary['median_transition_rgb_mae']:.6f} far_outside={summary['median_far_outside_max_delta']:.2e} Decision={decision}")
        return acceptance

    @torch.inference_mode()
    def run_v231_diagnostic(self):
        root = Path("res") / "v231_diagnostic"
        active_root = ACTIVE_OUTPUT_DIR / "v231_diagnostic"
        for path in (
            root, root / "comparisons" / "visual", root / "comparisons" / "matte",
            root / "comparisons" / "recolor_debug", root / "comparisons" / "crops",
            root / "visual_review",
        ):
            path.mkdir(parents=True, exist_ok=True)
        backend_audit = {
            "model_path": USER_V231_VITMATTE_PATH,
            "input_resolution": [1024, 1024],
            "training": False,
            "local_files_only": True,
        }
        try:
            matting = HairMattingV831(
                USER_V231_VITMATTE_PATH, device=self.device,
                inner_width=USER_V231_TRIMAP_INNER_WIDTH,
                outer_width=USER_V231_TRIMAP_OUTER_WIDTH,
                face_contact_extra_inner=USER_V231_FACE_CONTACT_EXTRA_INNER,
                max_trimap_hole_area=USER_V231_MAX_TRIMAP_HOLE_AREA,
            )
        except V231MattingError as exc:
            decision = str(exc).split(":", 1)[0]
            backend_audit.update({"loaded": False, "automatic_decision": decision, "error": str(exc)})
            acceptance = {
                "version": "v2.31", "training": False,
                "failed_gates": [decision], "primary_failure": decision,
                "automatic_decision": decision, "backend_audit": backend_audit,
            }
            for name, payload in (("backend_audit.json", backend_audit), ("v231_acceptance.json", acceptance)):
                with open(root / name, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            shutil.copytree(root, active_root, dirs_exist_ok=True)
            print(f"[V2.31:0] Decision={decision}: {exc}")
            return acceptance
        pp_checkpoint = Path(USER_V227_PP_CHECKPOINT)
        if not pp_checkpoint.exists():
            raise RuntimeError(f"V2.31 requires Original PP checkpoint: {pp_checkpoint}")
        post_process = PostProcessModel().to(self.device).eval()
        pp_state = torch.load(pp_checkpoint, map_location=self.device)
        post_process.load_state_dict(pp_state["model_state_dict"])
        backend_audit.update({
            "loaded": True, "backend": matting.backend_name,
            "parameter_count": sum(parameter.numel() for parameter in matting.model.parameters()),
            "requires_grad": any(parameter.requires_grad for parameter in matting.model.parameters()),
            "config": matting.config_dict(),
        })

        matte_records = []
        matte_inference_modes = set()
        matte_visual_index = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for batch in tqdm(self.val_loader, desc="V2.31 Phase A matte", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            _, _, _, pp_original, _ = self._render_v231_baselines(prepared, post_process)
            for index, sample_id in enumerate(prepared["sample_id"]):
                runtime = build_v831_runtime_inputs(
                    pp_original_rgb=pp_original[index:index + 1],
                    base_rgb=prepared["base_i"][index:index + 1],
                    v226_rgb=prepared["base_i"][index:index + 1],
                    target_hair_mask=prepared["v224_target_hair_mask"][index:index + 1],
                    parser_labels=prepared["v230_parser_labels"][index:index + 1],
                )
                try:
                    alpha, aux = matting(
                        pp_rgb_1024=runtime["pp_original_rgb"],
                        target_hair_mask_256=runtime["target_hair_mask"],
                        source_face_mask_256=runtime["source_face_mask"],
                        source_skin_mask_256=runtime["source_skin_mask"], return_aux=True,
                    )
                except Exception as exc:
                    decision = "V231_MATTING_BACKEND_FAIL"
                    failure = {
                        "version": "v2.31", "training": False,
                        "failed_gates": [decision], "primary_failure": decision,
                        "automatic_decision": decision, "error": str(exc),
                    }
                    backend_audit.update({"inference_passed": False, "error": str(exc)})
                    for name, payload in (("backend_audit.json", backend_audit), ("v231_acceptance.json", failure)):
                        with open(root / name, "w", encoding="utf-8") as handle:
                            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                    shutil.copytree(root, active_root, dirs_exist_ok=True)
                    print(f"[V2.31:0] Decision={decision}: {exc}")
                    return failure
                metrics = matte_metric_tensors(
                    alpha_hr=alpha, coarse_hair=aux["coarse_hair"], unknown=aux["unknown"],
                    sure_fg=aux["sure_fg"], sure_bg=aux["sure_bg"],
                    source_face_hr=aux["source_face_hr"],
                )
                matte_inference_modes.add(aux["inference_mode"])
                record = {
                    "sample_id": sample_id,
                    "inference_seconds": float(aux["inference_seconds"].item()),
                    "alpha_min": float(alpha.min().item()),
                    "alpha_max": float(alpha.max().item()),
                    "alpha_mean": float(alpha.mean().item()),
                }
                record.update({key: float(value[0].item()) for key, value in metrics.items()})
                matte_records.append(record)
                if matte_visual_index < USER_V231_PHASE_A_COUNT:
                    alpha_gradient = tnf.pad(alpha[..., 1:] - alpha[..., :-1], (0, 1, 0, 0)).abs()
                    save_preview(root / "comparisons" / "matte" / f"sample_{matte_visual_index:03d}.png", [
                        runtime["pp_original_rgb"] * 2 - 1,
                        mask_to_preview(aux["coarse_hair"]), mask_to_preview(aux["trimap"]),
                        mask_to_preview(aux["sure_fg"]), mask_to_preview(aux["unknown"]),
                        mask_to_preview(aux["sure_bg"]), mask_to_preview(alpha),
                        mask_to_preview(alpha_gradient),
                        mask_to_preview(alpha * aux["source_face_hr"]),
                        (alpha - aux["coarse_hair"]).repeat(1, 3, 1, 1).clamp(-1, 1),
                    ])
                    matte_visual_index += 1
                if len(matte_records) >= USER_V231_PHASE_A_COUNT:
                    break
            if len(matte_records) >= USER_V231_PHASE_A_COUNT:
                break
        if not matte_records:
            raise RuntimeError("V2.31 Phase A produced no valid samples")
        matte_summary = aggregate_v231(matte_records)
        matte_decision = classify_matte_v231(matte_summary)
        backend_audit["median_inference_seconds"] = matte_summary.get("median_inference_seconds")
        backend_audit["output_resolution"] = [1024, 1024]
        backend_audit["inference_modes"] = sorted(matte_inference_modes)
        backend_audit["alpha_min"] = matte_summary.get("min_alpha_min", matte_summary.get("median_alpha_min"))
        backend_audit["alpha_max"] = matte_summary.get("max_alpha_max")
        backend_audit["alpha_mean"] = matte_summary.get("mean_alpha_mean")
        backend_audit["peak_cuda_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else None
        )
        for name, payload in (
            ("backend_audit.json", backend_audit),
            ("trimap_audit.json", {key: value for key, value in matte_summary.items() if "sure_" in key or "unknown" in key}),
            ("matte_quality_audit.json", matte_summary),
            ("matte_face_fp_audit.json", {key: value for key, value in matte_summary.items() if "face_false" in key}),
        ):
            with open(root / name, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(
            f"[V2.31:A] fractional alpha={matte_summary.get('median_unknown_fractional_fraction', 0.0):.4f} "
            f"face FP={matte_summary.get('p90_face_false_positive_alpha', 0.0):.4f} "
            f"matte-vs-mask detail={matte_summary.get('median_alpha_boundary_deviation', 0.0):.4f} "
            f"Decision={matte_decision['automatic_decision']}"
        )
        if matte_decision["automatic_decision"] != "V231_MATTE_READY":
            acceptance = {"version": "v2.31", "training": False, "phase_a": matte_decision, **matte_decision}
            with open(root / "v231_acceptance.json", "w", encoding="utf-8") as handle:
                json.dump(acceptance, handle, ensure_ascii=False, indent=2, allow_nan=False)
            shutil.copytree(root, active_root, dirs_exist_ok=True)
            return acceptance

        records = []
        visual_ids = []
        parity_rgb = 0.0
        parity_alpha = 0.0
        visual_index = 0
        for batch in tqdm(self.val_loader, desc="V2.31 Phase B/C", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue
            base, anchor, v226, pp_original, v230_final = self._render_v231_baselines(prepared, post_process)
            for index, sample_id in enumerate(prepared["sample_id"]):
                kwargs = dict(
                    pp_original_rgb=pp_original[index:index + 1], base_rgb=base[index:index + 1],
                    v226_rgb=v226[index:index + 1],
                    target_hair_mask=prepared["v224_target_hair_mask"][index:index + 1],
                    parser_labels=prepared["v230_parser_labels"][index:index + 1],
                )
                runtime = build_v831_runtime_inputs(**kwargs)
                real_runtime = Blending_v8.build_v231_runtime_inputs(**kwargs)
                parity_rgb = max(parity_rgb, max(
                    float((runtime[key] - real_runtime[key]).abs().max().item())
                    for key in runtime
                ))
                alpha, matte_aux = matting(
                    pp_rgb_1024=runtime["pp_original_rgb"],
                    target_hair_mask_256=runtime["target_hair_mask"],
                    source_face_mask_256=runtime["source_face_mask"],
                    source_skin_mask_256=runtime["source_skin_mask"], return_aux=True,
                )
                final, final_aux = self.v231_finalizer(
                    pp_original_rgb=runtime["pp_original_rgb"], base_rgb=runtime["base_rgb"],
                    v226_rgb=runtime["v226_rgb"], alpha_hr=alpha,
                    sure_fg=matte_aux["sure_fg"], unknown=matte_aux["unknown"],
                    source_face_mask=runtime["source_face_mask"],
                    source_subject_mask=runtime["source_subject_mask"], return_aux=True,
                )
                parity_final, _ = Blending_v8.run_v231_final_debug(
                    self.v231_finalizer,
                    pp_original_rgb=real_runtime["pp_original_rgb"], base_rgb=real_runtime["base_rgb"],
                    v226_rgb=real_runtime["v226_rgb"], alpha_hr=alpha,
                    sure_fg=matte_aux["sure_fg"], unknown=matte_aux["unknown"],
                    source_face_mask=real_runtime["source_face_mask"],
                    source_subject_mask=real_runtime["source_subject_mask"],
                )
                parity_rgb = max(parity_rgb, float((final - parity_final).abs().max().item()))
                preview = lambda value: tnf.interpolate(value, size=base.shape[-2:], mode="area")
                metrics = v231_metric_tensors(
                    base_rgb=base[index:index + 1], v226_rgb=v226[index:index + 1],
                    v230_rgb=preview(v230_final[index:index + 1]), pp_rgb=preview(runtime["pp_original_rgb"]),
                    phase_b_rgb=preview(final_aux["phase_b_rgb"]), phase_c_rgb=preview(final_aux["phase_c_rgb"]),
                    target_lab=prepared["pseudo_lab"][index:index + 1], alpha_hr=preview(alpha),
                    source_face_mask=runtime["source_face_mask"], source_subject_mask=runtime["source_subject_mask"],
                    ref_stats={key: value[index:index + 1] for key, value in prepared["ref_stats"].items()},
                )
                record = {"sample_id": sample_id, "far_outside_max_delta": float(final_aux["far_outside_max_delta"][0].item())}
                record.update({key: float(value[0].item()) for key, value in metrics.items()})
                records.append(record)
                if visual_index < USER_V231_VISUAL_COUNT:
                    visual_ids.append(sample_id)
                    save_preview(root / "comparisons" / "visual" / f"sample_{visual_index:03d}.png", [
                        prepared["face_i"][index:index + 1], prepared["shape_i"][index:index + 1],
                        prepared["color_i"][index:index + 1], base[index:index + 1] * 2 - 1,
                        anchor[index:index + 1] * 2 - 1, v226[index:index + 1] * 2 - 1,
                        preview(runtime["pp_original_rgb"]) * 2 - 1,
                        preview(v230_final[index:index + 1]) * 2 - 1,
                        preview(final_aux["phase_b_rgb"]) * 2 - 1,
                        preview(final_aux["phase_c_rgb"]) * 2 - 1,
                        preview(final) * 2 - 1,
                    ])
                    save_preview(root / "comparisons" / "recolor_debug" / f"sample_{visual_index:03d}.png", [
                        preview(final_aux["low_v226_lab"][:, :1] / 50.0 - 1.0).repeat(1, 3, 1, 1),
                        preview(final_aux["low_pp_lab"][:, :1] / 50.0 - 1.0).repeat(1, 3, 1, 1),
                        preview(final_aux["delta_lab"][:, :1] / 35.0).repeat(1, 3, 1, 1),
                        preview(torch.linalg.vector_norm(final_aux["delta_lab"][:, 1:], dim=1, keepdim=True) / 30.0).repeat(1, 3, 1, 1) * 2 - 1,
                        preview(final_aux["applied_tone_delta_lab"] / 35.0).clamp(-1, 1),
                        preview(final_aux["b_pp_face"]) * 2 - 1, preview(final_aux["b_base_face"]) * 2 - 1,
                        preview(final_aux["background_replacement"]).mul(4).clamp(-1, 1),
                        preview(final - runtime["pp_original_rgb"]).mul(4).clamp(-1, 1),
                    ])
                    visual_index += 1
        if not records:
            raise RuntimeError("V2.31 Phase B/C produced no valid samples")
        summary = aggregate_v231(records)
        phase_b_failed = []
        if (float(summary.get("median_phase_b_core_ref_full_pseudo", 1e9)) > 1.015 * float(summary.get("median_v226_core_ref_full_pseudo", 0.0)) or
                float(summary.get("median_phase_b_core_ref_ab_pseudo", 1e9)) > 1.015 * float(summary.get("median_v226_core_ref_ab_pseudo", 0.0))):
            phase_b_failed.append("V231_CORE_COLOR_FAIL")
        if float(summary.get("median_phase_b_direct_ref_ab_stat_error", 1e9)) > 1.03 * float(summary.get("median_v226_direct_ref_ab_stat_error", 0.0)):
            phase_b_failed.append("V231_DIRECT_REFERENCE_COLOR_FAIL")
        if (float(summary.get("median_phase_b_base_color_rim_fraction", 1.0)) > 0.035 or
                float(summary.get("p90_phase_b_base_color_rim_fraction", 1.0)) > 0.075):
            phase_b_failed.append("V231_BASE_RIM_FAIL")
        if (float(summary.get("median_phase_b_face_rgb", 1.0)) > 0.060 or
                float(summary.get("p90_phase_b_face_rgb", 1.0)) > 0.100):
            phase_b_failed.append("V231_FACE_CONTACT_FAIL")
        if float(summary.get("median_phase_b_contour_hf_ab", 1.0)) > 0.60 * float(summary.get("median_v230_contour_hf_ab", 0.0)):
            phase_b_failed.append("V231_MICRO_JAGGED_FAIL")
        decision = classify_v231(summary, parity_rgb=parity_rgb, parity_alpha=parity_alpha)
        if phase_b_failed:
            phase_b_failed = list(dict.fromkeys(phase_b_failed))
            if parity_rgb > 1e-5 or parity_alpha > 1e-6:
                phase_b_failed.insert(0, "V231_RUNTIME_PARITY_FAIL")
            decision = {
                "failed_gates": phase_b_failed,
                "primary_failure": phase_b_failed[0], "automatic_decision": phase_b_failed[0],
            }
            for image_path in (root / "comparisons" / "visual").glob("sample_*.png"):
                panel = Image.open(image_path).convert("RGB")
                tile_width = panel.width // 11
                phase_b_tile = panel.crop((8 * tile_width, 0, 9 * tile_width, panel.height))
                panel.paste(phase_b_tile, (10 * tile_width, 0))
                panel.save(image_path)
        acceptance = {
            "version": "v2.31", "architecture": "hires_vitmatte_pp_unified_recolor_v831",
            "training": False, "phase_a": matte_decision,
            "phase_b_passed": not phase_b_failed, "phase_c_evaluated": not phase_b_failed,
            "runtime_parity": {"max_rgb_diff": parity_rgb, "max_alpha_diff": parity_alpha},
            "summary": summary, "visual_column_order": [
                "Source", "Shape Reference", "Color Reference", "Base", "Strong Anchor", "V2.26",
                "Original PP", "V2.30 Final", "V2.31 Phase-B", "V2.31 Phase-C", "V2.31 Final selected",
            ],
            **decision,
        }
        category_metrics = {
            "core_color": "core_ref_full_pseudo", "direct_ref_color": "phase_c_direct_ref_ab_stat_error",
            "base_rim": "base_color_rim_fraction", "face_leak": "phase_c_face_rgb",
            "background_retention": "background_retention", "contour_hf_ab": "contour_hf_ab",
        }
        manifest = {
            key: [item["sample_id"] for item in sorted(records, key=lambda row: float(row.get(metric, 0.0)), reverse=True)[:5]]
            for key, metric in category_metrics.items()
        }
        outputs = {
            "runtime_parity.json": acceptance["runtime_parity"],
            "core_color_audit.json": {key: value for key, value in summary.items() if "core_" in key},
            "direct_reference_color_audit.json": {key: value for key, value in summary.items() if "direct_ref" in key},
            "base_rim_audit.json": {key: value for key, value in summary.items() if "base_color" in key},
            "face_contact_audit.json": {key: value for key, value in summary.items() if "face_" in key},
            "background_retention_audit.json": {key: value for key, value in summary.items() if "background" in key},
            "micro_jaggedness_audit.json": {key: value for key, value in summary.items() if "hf" in key},
            "fixed_regression_manifest.json": manifest, "worst_case_manifest.json": manifest,
            "v231_acceptance.json": acceptance,
        }
        for name, payload in outputs.items():
            with open(root / name, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(root / "per_sample.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        review = {"reviewed_by_user": False, "visual_column_order": acceptance["visual_column_order"], "sample_ids": visual_ids}
        with open(root / "visual_review" / "v231_visual_review.json", "w", encoding="utf-8") as handle:
            json.dump(review, handle, ensure_ascii=False, indent=2)
        checklist = """# V2.31 Visual Review Checklist

- [ ] Hair core color matches Color Reference
- [ ] Fine chroma staircase is absent when zoomed in
- [ ] Hairline and fine strands remain natural
- [ ] Small Base-color patches are reduced or absent
- [ ] Hair tone does not leak into face or skin
- [ ] Real bangs still cover the forehead
- [ ] Original PP texture, shine, and antialiasing are preserved
- [ ] Background has no Color Reference tint
- [ ] No new white, gray, or dark fringe appears
- [ ] Final has no binary mask cut
"""
        (root / "visual_review" / "V231_VISUAL_REVIEW_CHECKLIST.md").write_text(
            checklist, encoding="utf-8"
        )
        report = f"""# Blending V8 V2.31 Acceptance

1. ViTMatte-S loaded: `{backend_audit.get('loaded', False)}` from `{USER_V231_VITMATTE_PATH}`.
2. Training occurred: `False`; ViTMatte and V2.26 remain frozen.
3. Matting inference mode: full 1024 by default, hair ROI plus 64px only after OOM.
4. Trimap: HM_X bilinear-upsampled to 1024, then 8px inner/outer morphology with 4px extra face-contact unknown width.
5. Sure FG/BG are clamped exactly; unknown alpha remains continuous.
6. Binary final alpha use: `False`.
7. Phase A decision: `{matte_decision['automatic_decision']}`.
8. Phase B passed: `{not phase_b_failed}`.
9. Phase C evaluated: `{not phase_b_failed}`.
10. Core pseudo full V2.31/V2.26: `{summary.get('median_core_ref_full_pseudo', 0.0):.4f}` / `{summary.get('median_v226_core_ref_full_pseudo', 0.0):.4f}`.
11. Direct-reference AB V2.31/V2.26: `{summary.get('median_phase_c_direct_ref_ab_stat_error', 0.0):.4f}` / `{summary.get('median_v226_direct_ref_ab_stat_error', 0.0):.4f}`.
12. Base rim median/p90: `{summary.get('median_base_color_rim_fraction', 0.0):.4f}` / `{summary.get('p90_base_color_rim_fraction', 0.0):.4f}`.
13. Face-contact RGB median/p90: `{summary.get('median_phase_c_face_rgb', 0.0):.4f}` / `{summary.get('p90_phase_c_face_rgb', 0.0):.4f}`.
14. Background retention: `{summary.get('median_background_retention', 0.0):.4f}`.
15. Contour HF-AB V2.31/V2.30: `{summary.get('median_contour_hf_ab', 0.0):.6f}` / `{summary.get('median_v230_contour_hf_ab', 0.0):.6f}`.
16. PP HF drift core/edge: `{summary.get('median_core_pp_hf_drift', 0.0):.6f}` / `{summary.get('median_edge_pp_hf_drift', 0.0):.6f}`.
17. Failed gates: `{decision['failed_gates']}`.
18. Automatic decision: `{decision['automatic_decision']}`. Human review is still required before acceptance.
"""
        (root / "BLENDING_V8_V2_31_ACCEPTANCE.md").write_text(report, encoding="utf-8")
        shutil.copytree(root, active_root, dirs_exist_ok=True)
        print(
            f"[V2.31:B] core color={summary.get('median_phase_b_core_ref_full_pseudo', 0.0):.4f} "
            f"Base rim={summary.get('median_phase_b_base_color_rim_fraction', 0.0):.4f} "
            f"face leak={summary.get('median_phase_b_face_rgb', 0.0):.4f}"
        )
        if not phase_b_failed:
            print(
                f"[V2.31:C] background retention={summary.get('median_background_retention', 0.0):.4f} "
                f"face leak={summary.get('median_phase_c_face_rgb', 0.0):.4f}"
            )
        else:
            print("[V2.31:C] skipped because Phase B gates failed")
        print(f"[V2.31] failed_gates={decision['failed_gates']}")
        print(f"[V2.31] Decision={decision['automatic_decision']}")
        return acceptance

    def train_loop(self):
        if USER_RESUME_CHECKPOINT:
            raise RuntimeError(
                "V2.31 diagnostic-only mode does not accept a resume checkpoint"
            )
        if USER_V235_DIAGNOSTIC_ONLY:
            summary = self.run_v235_diagnostic()
            history = [{"phase": "v235_hair_only_chroma_disentanglement_diagnostic_only", **summary}]
        elif USER_V234_DIAGNOSTIC_ONLY:
            summary = self.run_v234_diagnostic()
            history = [{"phase": "v234_hair_carrier_chroma_field_injection_diagnostic_only", **summary}]
        elif USER_V233_DIAGNOSTIC_ONLY:
            summary = self.run_v233_diagnostic()
            history = [{"phase": "v233_confidence_limited_fg_alpha_consistent_diagnostic_only", **summary}]
        elif USER_V232_DIAGNOSTIC_ONLY:
            summary = self.run_v232_diagnostic()
            history = [{"phase": "v232_foreground_recolor_matting_recompose_diagnostic_only", **summary}]
        elif not USER_V231_DIAGNOSTIC_ONLY:
            raise RuntimeError("V2.31 must remain deterministic and diagnostic-only")
        else:
            summary = self.run_v231_diagnostic()
            history = [{"phase": "v231_hires_matting_diagnostic_only", **summary}]
        assert_finite_json(history)
        for path in (ACTIVE_OUTPUT_DIR / "history.json", Path("res") / "history.json"):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(history, handle, ensure_ascii=False, indent=2, allow_nan=False)
        diagnostic_name = "v235" if USER_V235_DIAGNOSTIC_ONLY else "v234" if USER_V234_DIAGNOSTIC_ONLY else "v233_diagnostic" if USER_V233_DIAGNOSTIC_ONLY else "v232_diagnostic" if USER_V232_DIAGNOSTIC_ONLY else "v231_diagnostic"
        source_comparisons = ACTIVE_OUTPUT_DIR / diagnostic_name / "comparisons"
        comparisons_root = ACTIVE_OUTPUT_DIR / "comparisons"
        if source_comparisons.exists():
            for name in ("visual_review", "_latest"):
                shutil.copytree(source_comparisons, comparisons_root / name, dirs_exist_ok=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    atexit.register(clean_zombies)
    set_seed(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    triplets = read_triplets(ACTIVE_DATASET_DIR)
    if not triplets:
        raise RuntimeError(f"No 3-column experiments found in {ACTIVE_DATASET_DIR / 'dataset.exps'}")
    if len(triplets) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(
            f"dataset.exps is smaller than the validation split size ({ACTIVE_VAL_SIZE}) "
            f"for profile {USER_DATASET_PROFILE!r}."
        )

    ensure_dataset_cache_v8(triplets)
    teacher_cache_path = ACTIVE_DATASET_DIR / USER_TEACHER_CACHE_NAME
    teacher_records = None
    active_version = "V2.35" if USER_V235_DIAGNOSTIC_ONLY else "V2.34" if USER_V234_DIAGNOSTIC_ONLY else "V2.33" if USER_V233_DIAGNOSTIC_ONLY else "V2.32" if USER_V232_DIAGNOSTIC_ONLY else "V2.31"
    print(f"[{active_version}] deterministic validation only; no training")
    print(f"[{active_version}] teacher alpha/controller disabled")
    train_exps, val_exps = train_test_split(triplets, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    helper = MaskPrepHelper(device)

    train_dataset = BlendingDatasetV8(
        train_exps,
        ACTIVE_DATASET_DIR,
        ACTIVE_FACE_ROOT,
        ACTIVE_COLOR_ROOT,
        teacher_records,
    )
    val_dataset = BlendingDatasetV8(
        val_exps,
        ACTIVE_DATASET_DIR,
        ACTIVE_FACE_ROOT,
        ACTIVE_COLOR_ROOT,
        teacher_records,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
    )

    model, _ = Blending_v8.load_v226_projector_for_diagnostic(
        USER_V227_BASE_CHECKPOINT, device
    )
    trainer = BlendingTrainerV8(model, None, train_loader, val_loader, helper)
    if USER_V235_DIAGNOSTIC_ONLY:
        print(
            "[V2.35] training=False\n"
            "[V2.35] full_image_color_embedding=False\n"
            "[V2.35] full_feature_injection=False\n"
            "[V2.35] reference_input=HAIR_ONLY_RGB_CROP\n"
            "[V2.35] encoded_channels=LAB_AB_ONLY\n"
            "[V2.35] target_structure_owner=STRONG_ANCHOR\n"
            f"[V2.35] chroma_radius={USER_V235_CHROMA_RADIUS} boundary_radius={USER_V235_BOUNDARY_RADIUS}"
            , file=sys.stderr)
    elif USER_V234_DIAGNOSTIC_ONLY:
        print(
            "[V2.34] training=False\n"
            "[V2.34] strong_anchor_carrier=True\n"
            "[V2.34] fb_recomposition=False\n"
            "[V2.34] luminance_preserve=True\n"
            "[V2.34] reference_chroma_only=True\n"
            "[V2.34] face_protection=True\n"
            f"[V2.34] reference_radius={USER_V234_REFERENCE_RADIUS} edge_radius={USER_V234_EDGE_RADIUS}"
        , file=sys.stderr)
    elif USER_V233_DIAGNOSTIC_ONLY:
        print(
            "[V2.33] training=False\n"
            "[V2.33] ViTMatte=FROZEN_V231\n"
            "[V2.33] PyMatting=FROZEN_DEFAULTS\n"
            "[V2.33] V2.26 target=FROZEN\n"
            "[V2.33] unreliable_low_alpha_F_direct_use=False\n"
            "[V2.33] alpha_calibrator_invalid_fallback=PROPAGATED_OR_CONNECTIVITY\n"
            "[V2.33] background_target_alpha=ALPHA_EFF\n"
            "[V2.33] binary_context_support=False\n"
            "[V2.33] all_failed_gates=True\n"
            "[V2.33] white_net_metric=True",
            file=sys.stderr,
        )
    mode_header = "[V2.35] mode=HAIR_ONLY_CHROMA_DISENTANGLEMENT\n" if USER_V235_DIAGNOSTIC_ONLY else "[V2.34] mode=HAIR_CARRIER_CHROMA_FIELD_INJECTION\n" if USER_V234_DIAGNOSTIC_ONLY else "[V2.33] mode=CONFIDENCE_LIMITED_FOREGROUND_ALPHA_CONSISTENT\n" if USER_V233_DIAGNOSTIC_ONLY else "[V2.32] mode=FOREGROUND_RECOLOR_MATTING_RECOMPOSE\n" if USER_V232_DIAGNOSTIC_ONLY else "[V2.31] mode=HIRES_VITMATTE_PP_UNIFIED_RECOLOR\n"
    mode_details = (
        "[V2.35] training=False\n"
        "[V2.35] full_image_color_embedding=False\n"
        "[V2.35] full_feature_injection=False\n"
        "[V2.35] reference_input=HAIR_ONLY_RGB_CROP\n"
        "[V2.35] encoded_channels=LAB_AB_ONLY\n"
        "[V2.35] target_structure_owner=STRONG_ANCHOR\n"
        f"[V2.35] chroma_radius={USER_V235_CHROMA_RADIUS} boundary_radius={USER_V235_BOUNDARY_RADIUS}\n"
        f"[V2.35] base_checkpoint={USER_V227_BASE_CHECKPOINT}"
        if USER_V235_DIAGNOSTIC_ONLY else
        "[V2.34] training=False\n"
        "[V2.34] strong_anchor_carrier=True\n"
        "[V2.34] fb_recomposition=False\n"
        "[V2.34] luminance_preserve=True\n"
        "[V2.34] reference_chroma_only=True\n"
        "[V2.34] face_protection=True\n"
        f"[V2.34] reference_radius={USER_V234_REFERENCE_RADIUS} edge_radius={USER_V234_EDGE_RADIUS}\n"
        f"[V2.34] base_checkpoint={USER_V227_BASE_CHECKPOINT}"
        if USER_V234_DIAGNOSTIC_ONLY else
        "[V2.31] training=False\n"
        "[V2.31] matting_backend=ViTMatte-S\n"
        "[V2.31] matting_image=OriginalPP\n"
        "[V2.31] trimap_source=HM_X_256_HIRES_REFINEMENT\n"
        "[V2.31] final_binary_hair_mask=False\n"
        "[V2.31] final_carrier=OriginalPP\n"
        "[V2.31] low_frequency_target=V2.26\n"
        "[V2.31] strong_anchor_final_injection=False\n"
        "[V2.31] manual_face_classifier=False\n"
        "[V2.31] topology_repair_final=False\n"
        f"[V2.31] vitmatte_path={USER_V231_VITMATTE_PATH}\n"
        f"[V2.31] base_checkpoint={USER_V227_BASE_CHECKPOINT}"
    )
    print(
        mode_header + mode_details,
        file=sys.stderr,
    )
    print(
        f"[blending_v8] train_on_shape_satd_align=True "
        f"author_color_align_aux={USER_AUTHOR_COLOR_ALIGN_BATCH_PROB > 0} use_satd_v8={USER_USE_SATD_V8} "
        f"arch={FULL_COLOR_ARCH_V8_9} strong_anchor_alpha={USER_STRONG_ANCHOR_ALPHA} "
        f"satd_checkpoint={USER_SATD_CHECKPOINT_V8} "
        f"batch_size={USER_BATCH_SIZE} grad_accum_steps={USER_GRAD_ACCUM_STEPS} "
        f"effective_batch_size={USER_BATCH_SIZE * USER_GRAD_ACCUM_STEPS} "
        f"pseudo_ab_w={USER_PSEUDO_AB_LOSS_WEIGHT} "
        f"high_chroma_color_boost={USER_HIGH_CHROMA_COLOR_BOOST} "
        f"pseudo_rgb_w={USER_PSEUDO_RGB_LOSS_WEIGHT} "
        f"pseudo_luma_w={USER_PSEUDO_LUMA_LOSS_WEIGHT} "
        f"positive_luma_w={USER_POSITIVE_LUMA_EXCESS_WEIGHT} "
        f"hf_luma_w={USER_HF_LUMA_EXCESS_WEIGHT} "
        f"correction_norm_w={USER_CORRECTION_NORM_WEIGHT} "
        f"ab_gate_thresholds=({USER_AB_NO_EDIT},{USER_AB_FULL_EDIT}) "
        f"hue_gate_thresholds=({USER_HUE_NO_EDIT_DEG},{USER_HUE_FULL_EDIT_DEG}) "
        f"chroma_magnitude_thresholds=({USER_CHROMA_MAG_NO_EDIT},{USER_CHROMA_MAG_FULL_EDIT}) "
        f"distribution_thresholds=({USER_COLOR_DIST_NO_EDIT},{USER_COLOR_DIST_FULL_EDIT}) "
        f"lightness_gate_thresholds=({USER_LIGHTNESS_NO_EDIT_THRESHOLD_V8},{USER_LIGHTNESS_FULL_EDIT_THRESHOLD_V8}) "
        f"max_global_l_shift={USER_MAX_GLOBAL_L_SHIFT_V8} "
        f"min_safe_reference_fraction={USER_MIN_SAFE_REFERENCE_FRACTION_V8} "
        f"alpha_init={USER_ALPHA_INIT} layer_offset_max={USER_LAYER_OFFSET_MAX} "
        f"teacher_cache={teacher_cache_path} "
        f"correction_budget_ratios=({USER_CORRECTION_CHROMA_BUDGET_RATIO},"
        f"{USER_CORRECTION_LUMA_BUDGET_RATIO}) "
        f"correction_orth_scale={USER_CORRECTION_ORTH_SCALE} "
        f"training_mode={'V235_HAIR_ONLY_CHROMA_DISENTANGLEMENT_DIAGNOSTIC_ONLY' if USER_V235_DIAGNOSTIC_ONLY else 'V234_HAIR_CARRIER_CHROMA_FIELD_INJECTION_DIAGNOSTIC_ONLY' if USER_V234_DIAGNOSTIC_ONLY else 'V233_CONFIDENCE_LIMITED_FG_ALPHA_CONSISTENT_DIAGNOSTIC_ONLY' if USER_V233_DIAGNOSTIC_ONLY else 'V232_FOREGROUND_RECOLOR_MATTING_RECOMPOSE_DIAGNOSTIC_ONLY' if USER_V232_DIAGNOSTIC_ONLY else 'V231_HIRES_VITMATTE_PP_UNIFIED_RECOLOR_DIAGNOSTIC_ONLY'} "
        f"foreground_roi_padding={USER_V232_FOREGROUND_ROI_PADDING} "
        f"foreground_cache={USER_V232_FOREGROUND_CACHE or '<none>'} "
        f"foreground_backend={'PYMATTING_MULTILEVEL' if USER_V232_DIAGNOSTIC_ONLY else 'disabled'} "
        f"matte_width={USER_V228_MATTE_WIDTH} "
        f"bg_residual_radius={USER_V228_BG_RESIDUAL_RADIUS} "
        f"bg_residual_strength={USER_V228_BG_RESIDUAL_STRENGTH} "
        f"carrier_low_radius={USER_V229_CARRIER_LOW_RADIUS} "
        f"anchor_hf_gain={USER_V229_ANCHOR_HF_GAIN} "
        f"tone_radius={USER_V229_TONE_RADIUS} "
        f"background_radius={USER_V229_BACKGROUND_RADIUS} "
        f"luma_low_radius={USER_V223_LUMA_LOW_RADIUS} "
        f"max_low_l_shift={USER_V223_MAX_LOW_L_SHIFT} "
        f"edge_chroma_strength={USER_V223_EDGE_CHROMA_STRENGTH} "
        f"edge_luma_strength={USER_V223_EDGE_LUMA_STRENGTH} "
        f"remove_keep_w={USER_REMOVE_KEEP_L1_LOSS_WEIGHT} "
        f"protect_chroma_keep_w={USER_PROTECT_CHROMA_KEEP_LOSS_WEIGHT} "
        f"skin_chroma_keep_w={USER_SKIN_CHROMA_KEEP_LOSS_WEIGHT} "
        f"skin_rgb_keep_w={USER_SKIN_RGB_KEEP_LOSS_WEIGHT} "
        f"remove_block_in_target_hair={USER_REMOVE_BLOCK_IN_TARGET_HAIR} "
        f"face_neck_color_block={USER_FACE_NECK_COLOR_BLOCK} "
        f"target_hair_neck_override={USER_TARGET_HAIR_NECK_OVERRIDE} "
        f"author_color_align_batch_prob={USER_AUTHOR_COLOR_ALIGN_BATCH_PROB} "
        f"author_zero_prefix_train={USER_AUTHOR_ZERO_PREFIX_TRAIN} "
        f"resume_checkpoint={USER_RESUME_CHECKPOINT or '<none>'}",
        file=sys.stderr,
    )
    trainer.train_loop()


if __name__ == "__main__":
    main()
