import argparse
import faulthandler
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
PP_EXTRA_MASK_KEYS = (
    "cleanup_inner_edge",
    "revealed_skin_mask",
    "source_skin_valid_mask",
    "earring_confident_mask",
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
VAL_COLUMNS = (
    "source",
    "target",
    "gen_w",
    "gen_f",
    "query_mask",
    "fine_mask",
    "source_hair_block_mask",
    "prior_mask",
    "source_earring_mask",
    "query_before_recall",
    "query_recall_mask",
    "online_earring_candidate_mask",
    "online_earring_search_mask",
    "source_lobe_search_mask",
    "earring_confident_mask",
    "fine_mask_before_floor",
    "earring_fine_floor_support",
)

# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small_accessory_ffhq"  # "small_accessory_ffhq" or "full_ffhq"

USER_DATASET_DIR_SMALL = Path("images/pp_dataset_v5_dual_small")
USER_OUTPUT_DIR_SMALL = Path("output/pp_v5_checkpoints_small")
USER_RUN_NAME_SMALL = "ear_refine_v5_dual_small"

USER_DATASET_DIR_FULL = Path("/data/coding/HairFastGAN_to_xcodecopy/images/pp_dataset_v5_dual_full")
USER_OUTPUT_DIR_FULL = Path("output/pp_v5_checkpoints_full")
USER_RUN_NAME_FULL = "ear_refine_v5_dual_full"

USER_FID_DATASET = "fid_images"
USER_USE_FID = False
USER_USE_WANDB = False
USER_RESUME_CHECKPOINT = None
USER_BASE_CHECKPOINT = "pretrained_models/PostProcess/pp_model.pth"

USER_BATCH_SIZE = 4
USER_NUM_WORKERS = 0
USER_EPOCHS = 120
USER_VAL_SIZE = 512
USER_VAL_PREVIEW_COUNT = 20
USER_GRAD_ACCUM_STEPS = 4

USER_TRAINING_STAGE = "ear_only"  # "ear_only", "joint_highres", or "full"
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

USER_EAR_BLUR_KERNEL = 11
USER_EAR_BLUR_SIGMA = 3.0
USER_EAR_MASK_HIDDEN = 32
USER_EAR_MASK_INIT_BIAS = -4.0
USER_EARRING_QUERY_DILATE = 3
USER_EARRING_QUERY_BOOST = 1.0
USER_ENABLE_EARRING_QUERY_RECALL = True
USER_EARRING_QUERY_RECALL_DILATE = 7
USER_EARRING_QUERY_DOWNWARD_SHIFT = 18
USER_EARRING_QUERY_LOWER_LOBE_WEIGHT = 0.20
USER_EARRING_QUERY_CANDIDATE_BOOST = 0.90
USER_EARRING_QUERY_BLOCK_PROTECT = 0.85
USER_EARRING_ALIGN_MAX_SHIFT = 12
USER_EAR_FINE_SUPPORT_DILATE = 3
USER_EARRING_FINE_MASK_FLOOR = 0.18
USER_EARRING_FINE_MASK_DILATE = 5

USER_LAMBDA_EAR_MASK = 6.0
USER_LAMBDA_EAR_HIGH = 8.0
USER_LAMBDA_EAR_PRESENCE = 3.0
USER_LAMBDA_EAR_LIGHTING = 0.35
USER_LAMBDA_EAR_HAIR_LEAK = 1.0
USER_LAMBDA_EAR_HAIR_ANCHOR = 5.0
USER_LAMBDA_EAR_COLOR = 1.5
USER_LAMBDA_EAR_HIGHLIGHT = 3.0
USER_LAMBDA_CLEANUP_ANCHOR = 0.0
USER_LAMBDA_CLEANUP_LOW_ANCHOR = 3.5
USER_LAMBDA_CLEANUP_SOURCE_REJECT = 4.0
USER_LAMBDA_CLEANUP_NON_DARK = 8.0
USER_LAMBDA_CLEANUP_HIGH = 0.4
USER_LAMBDA_CLEANUP_TEXTURE_STAT = 0.8
USER_LAMBDA_DETAIL_HIGH = 0.0
USER_LAMBDA_DETAIL_LOW_ANCHOR = 1.5
USER_LAMBDA_EAR_QUERY_EXPAND = 0.02
USER_LAMBDA_EAR_MASK_AREA = 0.0
USER_LAMBDA_EAR_EDGE = 3.0
USER_LAMBDA_EAR_BRIGHTNESS_REG = 0.05
USER_LAMBDA_BASE_CLEANUP_EXCLUDE = 1.0
USER_EARRING_SUPERVISION_DILATE = 5
USER_LAMBDA_TARGET_EARRING_SUPPRESS_LOW = 1.5
USER_LAMBDA_TARGET_EARRING_SUPPRESS_HIGH = 0.8
USER_LAMBDA_REVEALED_SKIN_LOW_ANCHOR = 2.5
USER_LAMBDA_REVEALED_SKIN_TEXTURE = 1.0
USER_LAMBDA_REVEALED_SKIN_HIGH = 1.2
USER_LAMBDA_REVEALED_SKIN_ENERGY = 0.6
USER_LAMBDA_REVEALED_SKIN_SEAM = 0.6

USER_USE_DATASET_QUERY_MASK = False
USER_USE_DATASET_SOURCE_EARRING_MASK = False
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
    "target_visibility_expand": USER_TARGET_VISIBILITY_EXPAND,
    "max_target_hair_overlap": USER_MAX_TARGET_HAIR_OVERLAP,
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
    "earring_align_max_shift": USER_EARRING_ALIGN_MAX_SHIFT,
    "ear_fine_support_dilate": USER_EAR_FINE_SUPPORT_DILATE,
    "earring_fine_mask_floor": USER_EARRING_FINE_MASK_FLOOR,
    "earring_fine_mask_dilate": USER_EARRING_FINE_MASK_DILATE,
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
    "ear_query_expand": USER_LAMBDA_EAR_QUERY_EXPAND,
    "ear_mask_area": USER_LAMBDA_EAR_MASK_AREA,
    "ear_edge": USER_LAMBDA_EAR_EDGE,
    "ear_brightness_reg": USER_LAMBDA_EAR_BRIGHTNESS_REG,
    "base_cleanup_exclude": USER_LAMBDA_BASE_CLEANUP_EXCLUDE,
    "earring_supervision_dilate": USER_EARRING_SUPERVISION_DILATE,
    "target_earring_suppress_low": USER_LAMBDA_TARGET_EARRING_SUPPRESS_LOW,
    "target_earring_suppress_high": USER_LAMBDA_TARGET_EARRING_SUPPRESS_HIGH,
    "revealed_skin_low_anchor": USER_LAMBDA_REVEALED_SKIN_LOW_ANCHOR,
    "revealed_skin_texture": USER_LAMBDA_REVEALED_SKIN_TEXTURE,
    "revealed_skin_high": USER_LAMBDA_REVEALED_SKIN_HIGH,
    "revealed_skin_energy": USER_LAMBDA_REVEALED_SKIN_ENERGY,
    "revealed_skin_seam": USER_LAMBDA_REVEALED_SKIN_SEAM,
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
    parser.add_argument("--target_visibility_expand", type=int, default=defaults["target_visibility_expand"])
    parser.add_argument("--max_target_hair_overlap", type=float, default=defaults["max_target_hair_overlap"])
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
    parser.add_argument("--earring_align_max_shift", type=int, default=defaults["earring_align_max_shift"])
    parser.add_argument("--ear_fine_support_dilate", type=int, default=defaults["ear_fine_support_dilate"])
    parser.add_argument("--earring_fine_mask_floor", type=float, default=defaults["earring_fine_mask_floor"])
    parser.add_argument("--earring_fine_mask_dilate", type=int, default=defaults["earring_fine_mask_dilate"])
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
    parser.add_argument("--ear_query_expand", type=float, default=defaults["ear_query_expand"])
    parser.add_argument("--ear_mask_area", type=float, default=defaults["ear_mask_area"])
    parser.add_argument("--ear_edge", type=float, default=defaults["ear_edge"])
    parser.add_argument("--ear_brightness_reg", type=float, default=defaults["ear_brightness_reg"])
    parser.add_argument("--base_cleanup_exclude", type=float, default=defaults["base_cleanup_exclude"])
    parser.add_argument("--earring_supervision_dilate", type=int, default=defaults["earring_supervision_dilate"])
    parser.add_argument("--target_earring_suppress_low", type=float, default=defaults["target_earring_suppress_low"])
    parser.add_argument("--target_earring_suppress_high", type=float, default=defaults["target_earring_suppress_high"])
    parser.add_argument("--revealed_skin_low_anchor", type=float, default=defaults["revealed_skin_low_anchor"])
    parser.add_argument("--revealed_skin_texture", type=float, default=defaults["revealed_skin_texture"])
    parser.add_argument("--revealed_skin_high", type=float, default=defaults["revealed_skin_high"])
    parser.add_argument("--revealed_skin_energy", type=float, default=defaults["revealed_skin_energy"])
    parser.add_argument("--revealed_skin_seam", type=float, default=defaults["revealed_skin_seam"])
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


def move_batch_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


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
                "ear_query_expand": args.ear_query_expand,
                "ear_mask_area": args.ear_mask_area,
                "ear_block_strength": args.source_hair_block_strength,
                "ear_edge": args.ear_edge,
                "ear_brightness_reg": args.ear_brightness_reg,
                "base_cleanup_exclude": args.base_cleanup_exclude,
                "earring_supervision_dilate": args.earring_supervision_dilate,
                "target_earring_suppress_low": args.target_earring_suppress_low,
                "target_earring_suppress_high": args.target_earring_suppress_high,
                "revealed_skin_low_anchor": args.revealed_skin_low_anchor,
                "revealed_skin_texture": args.revealed_skin_texture,
                "revealed_skin_high": args.revealed_skin_high,
                "revealed_skin_energy": args.revealed_skin_energy,
                "revealed_skin_seam": args.revealed_skin_seam,
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
        final_dir = self.args.checkpoint_dir / "val_final_images" / epoch_tag
        final_dir.mkdir(parents=True, exist_ok=True)
        with open(vis_dir / "columns.txt", "w", encoding="utf-8") as file:
            file.write(" | ".join(VAL_COLUMNS) + "\n")

        preview_count = max(1, int(getattr(self.args, "val_preview_count", 20)))
        np.random.seed(1927)
        indices = np.random.choice(len(files), size=min(len(files), preview_count), replace=False)
        for order, idx in enumerate(indices):
            preview_row, final_image = files[idx]
            image = image_grid(
                list(map(T.functional.to_pil_image, preview_row)),
                1,
                len(preview_row),
            )
            image.save(vis_dir / f"val_{order:03d}.png")
            T.functional.to_pil_image(final_image).save(
                final_dir / f"final_{order:03d}.png"
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

    def _run_model(self, batch):
        batch = move_batch_to_device(batch, self.device)
        source_full = batch["source"]
        source = self.downsample_256(source_full).clip(0, 1)
        target = batch["target"]
        target_mask = batch["target_mask"]
        HT_E = batch["HT_E"]
        earring_highlight_mask = batch.get("earring_highlight_mask")
        highlight_gate = batch.get("has_earring_highlight_mask")
        if torch.is_tensor(highlight_gate):
            use_dataset_highlight = bool((highlight_gate.float() > 0.5).all().item())
            if not use_dataset_highlight:
                earring_highlight_mask = None
        source_ear_mask = (
            batch["source_earring_mask"]
            if bool(getattr(self.args, "use_dataset_source_earring_mask", False))
            else None
        )

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
            earring_confident_mask=batch.get("earring_confident_mask"),
            earring_highlight_mask=earring_highlight_mask,
            earring_mask_is_dataset=batch.get("has_earring_confident_mask"),
        )
        for key in CLEANUP_MASK_KEYS:
            aux[key] = batch[key]
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
        self.model.to(self.device).eval()
        val_losses = {}
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
            source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux, _ = self._run_model(batch)
            losses = self.loss_builder(source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux=aux)
            losses["loss"] = sum(losses.values())
            val_losses = accumulate(val_losses, losses)

            gen_w_256 = self.downsample_256((gen_im_W + 1) / 2).clip(0, 1)
            gen_f_256 = self.downsample_256((gen_im_F + 1) / 2).clip(0, 1)
            if self.fid_calc is not None:
                images_to_fid.append(to_299((gen_im_F + 1) / 2).clip(0, 1))

            for idx in range(source.size(0)):
                image_size = tuple(source.shape[-2:])
                preview_row = [
                    source[idx].cpu(),
                    target[idx].cpu(),
                    gen_w_256[idx].cpu(),
                    gen_f_256[idx].cpu(),
                    self.mask_to_rgb(aux.get("raw_query_mask", aux.get("query_mask"))[idx] if aux.get("raw_query_mask", aux.get("query_mask")) is not None else None, image_size),
                    self.mask_to_rgb(aux.get("fine_mask")[idx] if aux.get("fine_mask") is not None else None, image_size),
                    self.mask_to_rgb(aux.get("source_hair_block_mask")[idx] if aux.get("source_hair_block_mask") is not None else None, image_size),
                    self.mask_to_rgb(aux.get("prior_mask")[idx] if aux.get("prior_mask") is not None else None, image_size),
                    self.mask_to_rgb(aux.get("source_earring_mask")[idx] if aux.get("source_earring_mask") is not None else None, image_size),
                    self.mask_to_rgb(aux.get("query_mask_before_recall")[idx] if aux.get("query_mask_before_recall") is not None else None, image_size),
                    self.mask_to_rgb(aux.get("earring_query_recall_mask")[idx] if aux.get("earring_query_recall_mask") is not None else None, image_size),
                    self.mask_to_rgb(aux.get("online_earring_candidate_mask")[idx] if aux.get("online_earring_candidate_mask") is not None else None, image_size),
                    self.mask_to_rgb(aux.get("online_earring_search_mask")[idx] if aux.get("online_earring_search_mask") is not None else None, image_size),
                    self.mask_to_rgb(aux.get("source_lobe_search_mask")[idx] if aux.get("source_lobe_search_mask") is not None else None, image_size),
                    self.mask_to_rgb(aux.get("earring_confident_mask")[idx] if aux.get("earring_confident_mask") is not None else None, image_size),
                    self.mask_to_rgb(aux.get("fine_mask_before_floor")[idx] if aux.get("fine_mask_before_floor") is not None else None, image_size),
                    self.mask_to_rgb(aux.get("earring_fine_floor_support")[idx] if aux.get("earring_fine_floor_support") is not None else None, image_size),
                ]
                preview_seen = self.update_preview_buffer(
                    preview_files,
                    (
                        preview_row,
                        ((gen_im_F[idx] + 1) / 2).detach().cpu().clamp(0, 1),
                    ),
                    preview_seen,
                    preview_count,
                )

            del source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f, aux
            del gen_w_256, gen_f_256
            if self.device == "cuda":
                torch.cuda.empty_cache()

        if self.fid_calc is not None and images_to_fid:
            val_losses["FID CLIP"] = self.fid_calc(torch.cat(images_to_fid))

        for key, value in val_losses.items():
            if key != "FID CLIP":
                value = value.item() / max(1, len(self.test_dataloader))
            self.logger.log_scalars({f"val {key}": value})

        if preview_files:
            self.save_validation_images(preview_files, epoch_tag)
            np.random.seed(1927)
            indices = np.random.choice(len(preview_files), size=min(len(preview_files), preview_count), replace=False)
            images_to_log = [image_grid(list(map(T.functional.to_pil_image, preview_files[idx][0])), 1, len(preview_files[idx][0]))
                             for idx in indices]
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
    def __init__(self, dataset_index: DatasetPartIndexV5, sample_indices, is_test=False):
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
                        "source_hair_block_mask", "earring_search_mask", *CLEANUP_MASK_KEYS, *PP_EXTRA_MASK_KEYS]
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
            "earring_search_mask": item.get("earring_search_mask", torch.zeros_like(fallback_mask)).clone(),
            "ear_roi": item["ear_roi"].clone(),
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

    return DatasetPartIndexV5(files, part_lengths, positive_hints)


def item_has_nonempty_mask(item, key: str, min_area: float = 1.0) -> bool:
    value = item.get(key)
    return torch.is_tensor(value) and value.float().sum().item() > min_area


def is_positive_hint(item, query_area_threshold: float) -> bool:
    return bool(
        item["source_earring_mask"].sum().item() > 0
        or item["target_earring_mask"].sum().item() > 0
        or item_has_nonempty_mask(item, "earring_confident_mask")
        or item_has_nonempty_mask(item, "earring_highlight_mask")
    )


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
    test_size = min(args.test_size, max(1, len(dataset_index) // 10))
    train_indices, test_indices = split_dataset_indices(len(dataset_index), test_size, seed=42)

    train_dataset = PPDatasetV5(dataset_index, train_indices)
    positive_train_indices = [idx for idx in train_indices if dataset_index.is_positive(int(idx))]
    train_dataset_positive = PPDatasetV5(dataset_index, positive_train_indices) if positive_train_indices else None
    test_dataset = PPDatasetV5(dataset_index, test_indices, is_test=True)

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

    logger = WandbLogger(name=args.name_run, project="HairFast-PostProcess-V5") if args.use_wandb else NullLoggerV5(
        args.checkpoint_dir
    )
    logger.start_logging()

    model = PostProcessModelV5(args)
    model.load_base_checkpoint(args.base_checkpoint)
    configure_training_stage(model, args)
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=1e-4, weight_decay=0)

    trainer = TrainerV5(model, args, optimizer, train_dataloader, train_dataloader_positive, test_dataloader, logger)
    if args.resume_checkpoint is not None:
        trainer.load_model(args.resume_checkpoint)
    trainer.train_loop()


if __name__ == "__main__":
    parser = build_parser(RESOLVED_USER_CONFIG)
    main(parser.parse_args())
