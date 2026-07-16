import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Alignment_v18 import Alignment_v18
from models.Embedding import Embedding
from models.Encoders import ClipBlendingModel as BlendingModel
from models.Net import Net
from models.SATD_v18 import SATD_v18
from utils.bicubic import BicubicDownSample
from utils.image_utils import equal_replacer
from utils.train import WandbLogger, toggle_grad


# ========================= User Config: edit here only =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("images/satd_dataset_v18_3000")
USER_FFHQ_ROOT = Path("images/FFHQ")
USER_FFHQ_SHAPE_ROOT = Path("images/FFHQ")
USER_FFHQ_COLOR_ROOT = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/satd_train_v18_3000")

USER_SMALL_DATASET_DIR = Path("images/satd_dataset_v18_small")
USER_SMALL_FACE_ROOT = Path("images/FFHQ_long")
USER_SMALL_SHAPE_ROOT = Path("images/FFHQ_short")
USER_SMALL_COLOR_ROOT = Path("images/FFHQ_color")
USER_SMALL_OUTPUT_DIR = Path("output/satd_train_v18_small")

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_SATD_INIT_CKPT = ""
USER_BLENDING_INIT_CKPT_V18 = "pretrained_models/Blending/checkpoint.pth"
USER_BLENDING_TRAINED_CKPT_V18 = "pretrained_models/Blending/checkpoint.pth"
USER_CLIP_MODEL_V18 = "ViT-B/32"

USER_BATCH_SIZE = 1
USER_EFFECTIVE_BATCH_SIZE = 2
USER_ENCODER_BATCH_SIZE = 1
USER_CPU_THREADS = 1
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_DISABLE_CUDNN_BENCHMARK = True

USER_EPOCHS = 30
USER_LR = 1e-4
USER_WEIGHT_DECAY = 1e-6
USER_VAL_SIZE_FFHQ = 30
USER_SMALL_VAL_SIZE = 30
USER_RANDOM_SEED = 3407

USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_VAL_IMAGES_EVERY = 1
USER_LOG_IMAGE_COUNT = 30

USER_USE_WANDB = False
USER_WANDB_RUN_NAME = "satd_train_v18"
USER_WANDB_PROJECT = "HairFast-SATD-v18"

USER_POST_SATD_BLEND_V18 = 0.28
USER_SATD_BOUNDARY_V18 = 8
USER_EQ8_REFERENCE_BLEND_V18 = 0.0

USER_STAGE1_EPOCHS_V18 = 4
USER_STAGE2_EPOCHS_V18 = 8

USER_LAMBDA_PRESERVE_V18 = 1.35
USER_LAMBDA_MAIN_HAIR_KEEP_V18 = 5.00
USER_LAMBDA_BODY_PRESERVE_V18 = 1.50
USER_LAMBDA_CLEANUP_TARGET_V18 = 1.10
USER_LAMBDA_HALO_V18 = 1.40
USER_LAMBDA_TAIL_V18 = 0.85
USER_LAMBDA_FACE_V18 = 0.70
USER_LAMBDA_NECK_V18 = 1.00
USER_LAMBDA_NON_DARK_V18 = 1.20
USER_LAMBDA_LATENT_KEEP_V18 = 0.04

USER_RESUME_CHECKPOINT = ""
# =============================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "face_root": USER_FFHQ_ROOT,
            "shape_root": USER_FFHQ_SHAPE_ROOT,
            "color_root": USER_FFHQ_COLOR_ROOT,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "val_size": USER_VAL_SIZE_FFHQ,
        },
        "small": {
            "dataset_dir": USER_SMALL_DATASET_DIR,
            "face_root": USER_SMALL_FACE_ROOT,
            "shape_root": USER_SMALL_SHAPE_ROOT,
            "color_root": USER_SMALL_COLOR_ROOT,
            "output_dir": USER_SMALL_OUTPUT_DIR,
            "val_size": USER_SMALL_VAL_SIZE,
        },
    }
    if USER_DATASET_PROFILE not in profiles:
        raise RuntimeError(
            f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}. "
            f"Choose one of: {', '.join(sorted(profiles))}."
        )
    return profiles[USER_DATASET_PROFILE]


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_DATASET_DIR = _DATASET_CFG["dataset_dir"]
ACTIVE_FACE_IMAGE_ROOT = _DATASET_CFG["face_root"]
ACTIVE_SHAPE_IMAGE_ROOT = _DATASET_CFG["shape_root"]
ACTIVE_COLOR_IMAGE_ROOT = _DATASET_CFG["color_root"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_VAL_SIZE = _DATASET_CFG["val_size"]


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

if USER_DISABLE_CUDNN_BENCHMARK and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False

torch.set_num_threads(max(1, USER_CPU_THREADS))
try:
    torch.set_num_interop_threads(max(1, USER_CPU_THREADS))
except RuntimeError:
    pass

if USER_EFFECTIVE_BATCH_SIZE < USER_BATCH_SIZE or USER_EFFECTIVE_BATCH_SIZE % USER_BATCH_SIZE != 0:
    raise RuntimeError("USER_EFFECTIVE_BATCH_SIZE must be >= USER_BATCH_SIZE and divisible by it.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gray(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def gaussian_blur(image: torch.Tensor, kernel_size: int = 11, sigma: float = 3.0) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return image

    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords.pow(2)) / max(2 * sigma**2, 1e-6))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(image.shape[1], 1, -1, -1)
    return F.conv2d(image, kernel_2d, padding=kernel_size // 2, groups=image.shape[1])


def gray_low(image: torch.Tensor) -> torch.Tensor:
    return gaussian_blur(gray(image), kernel_size=11, sigma=3.0)


def erode_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=width)


def dilate_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target).abs() * mask).sum() / denom


def masked_non_darker(pred: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    return (torch.relu(baseline - pred) * mask).sum() / denom


def materialize_tensor_tree(obj):
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {key: materialize_tensor_tree(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [materialize_tensor_tree(val) for val in obj]
    if isinstance(obj, tuple):
        return tuple(materialize_tensor_tree(val) for val in obj)
    return obj


def resolve_checkpoint_state_dict(ckpt: dict[str, object]) -> dict[str, torch.Tensor]:
    if "satd_v18_state_dict" in ckpt:
        return ckpt["satd_v18_state_dict"]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    for key, value in ckpt.items():
        if key.endswith("_state_dict") and isinstance(value, dict):
            return value
    return ckpt


def load_compatible_state_dict(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    model_state = module.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model_state.update(compatible)
    module.load_state_dict(model_state, strict=False)
    return sorted(compatible.keys())


def resolve_blending_checkpoint_v18() -> Path:
    ckpt_path = Path(USER_BLENDING_TRAINED_CKPT_V18) if USER_BLENDING_TRAINED_CKPT_V18 else Path(USER_BLENDING_INIT_CKPT_V18)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Cannot find blending checkpoint for SATD v18 pre-blend stage: {ckpt_path}. "
            "Train blending_train_v18 first, then point USER_BLENDING_TRAINED_CKPT_V18 to its best/last checkpoint."
        )
    return ckpt_path


class NullLogger:
    def __init__(self):
        self.train_step = 0

    def start_logging(self):
        return None

    def log(self, *args, **kwargs):
        return None

    def save(self, *args, **kwargs):
        return None

    def next_step(self):
        self.train_step += 1

    @property
    def wandb(self):
        class _Dummy:
            @staticmethod
            def finish():
                return None

        return _Dummy()


class SATDTripletDataset_v18(Dataset):
    def __init__(self, entries: list[tuple[str, str, str]], face_root: Path, shape_root: Path, color_root: Path):
        self.entries = entries
        self.face_root = face_root
        self.shape_root = shape_root
        self.color_root = color_root
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.entries)

    def _read_image(self, root: Path, name: str) -> torch.Tensor:
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            path = root / f"{name}{suffix}"
            if path.exists():
                with Image.open(path) as image:
                    return self.to_tensor(image.convert("RGB"))
        raise FileNotFoundError(f"Cannot find {name}.png/.jpg/.jpeg/.webp in {root}")

    def __getitem__(self, idx):
        face, shape, color = self.entries[idx]
        return {
            "face_name": face,
            "shape_name": shape,
            "color_name": color,
            "face": self._read_image(self.face_root, face),
            "shape": self._read_image(self.shape_root, shape),
            "color": self._read_image(self.color_root, color),
        }


def collate_triplets(batch):
    return {
        "face_name": [item["face_name"] for item in batch],
        "shape_name": [item["shape_name"] for item in batch],
        "color_name": [item["color_name"] for item in batch],
        "face": torch.stack([item["face"] for item in batch]),
        "shape": torch.stack([item["shape"] for item in batch]),
        "color": torch.stack([item["color"] for item in batch]),
    }


def load_triplets(dataset_dir: Path) -> list[tuple[str, str, str]]:
    triplets = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split()
            if len(items) == 3:
                triplets.append((items[0], items[1], items[2]))
            elif len(items) == 2:
                triplets.append((items[0], items[1], items[1]))
    return triplets


class TrainerSATD_v18:
    def __init__(self, train_loader, val_loader):
        self.device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
        self.logger = self._build_logger()
        self.logger.start_logging()

        self.opts = Namespace(
            size=1024,
            ckpt=USER_STYLEGAN_CKPT,
            channel_multiplier=2,
            latent=512,
            n_mlp=8,
            device=str(self.device),
            batch_size=USER_ENCODER_BATCH_SIZE,
            save_all=False,
            save_all_dir=Path("output"),
            mixing=0.95,
            smooth=5,
            rotate_checkpoint=USER_ROTATE_CKPT,
            satd_boundary_v18=USER_SATD_BOUNDARY_V18,
            eq8_reference_blend_v18=USER_EQ8_REFERENCE_BLEND_V18,
        )

        self.net = Net(self.opts)
        self.embed = Embedding(self.opts, net=self.net).eval()
        self.align = Alignment_v18(self.opts, latent_encoder=self.embed.get_e4e_embed, net=self.net).eval()
        self.blending = BlendingModel(USER_CLIP_MODEL_V18).to(self.device).eval()
        blend_ckpt = torch.load(resolve_blending_checkpoint_v18(), map_location=self.device)
        load_compatible_state_dict(self.blending, resolve_checkpoint_state_dict(blend_ckpt))
        self.satd = SATD_v18().to(self.device)
        if USER_SATD_INIT_CKPT:
            ckpt = torch.load(USER_SATD_INIT_CKPT, map_location=self.device)
            load_compatible_state_dict(self.satd, resolve_checkpoint_state_dict(ckpt))

        self.downsample_256 = BicubicDownSample(factor=4)
        toggle_grad(self.net.generator, False)
        toggle_grad(self.blending, False)
        toggle_grad(self.satd, True)

        self.optimizer = torch.optim.Adam(self.satd.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
        self.grad_accum_steps = max(1, USER_EFFECTIVE_BATCH_SIZE // USER_BATCH_SIZE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_loss = float("inf")
        self.cur_iter = 0
        self.output_ckpt_dir = ACTIVE_OUTPUT_DIR / "checkpoints"
        self.output_val_dir = ACTIVE_OUTPUT_DIR / "val_images"
        self.output_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_val_dir.mkdir(parents=True, exist_ok=True)

    def _build_logger(self):
        if USER_USE_WANDB:
            return WandbLogger(name=USER_WANDB_RUN_NAME, project=USER_WANDB_PROJECT)
        return NullLogger()

    @staticmethod
    def stage_schedule(epoch: int) -> dict[str, float]:
        if epoch < USER_STAGE1_EPOCHS_V18:
            return {"blend": 0.18, "cleanup_scale": 0.35, "non_dark_scale": 1.00}
        if epoch < USER_STAGE2_EPOCHS_V18:
            return {"blend": 0.24, "cleanup_scale": 0.70, "non_dark_scale": 1.00}
        return {"blend": USER_POST_SATD_BLEND_V18, "cleanup_scale": 1.00, "non_dark_scale": 1.00}

    def build_name_to_embed(self, batch):
        images_to_name = defaultdict(list)
        sample_names = []
        bsz = batch["face"].shape[0]
        for idx in range(bsz):
            face, shape, color = equal_replacer([batch["face"][idx], batch["shape"][idx], batch["color"][idx]])
            names = {"face": f"face_{idx}", "shape": f"shape_{idx}", "color": f"color_{idx}"}
            images_to_name[face].append(names["face"])
            images_to_name[shape].append(names["shape"])
            images_to_name[color].append(names["color"])
            sample_names.append(names)
        with torch.no_grad():
            name_to_embed = self.embed.embedding_images(images_to_name)
        return name_to_embed, sample_names

    @staticmethod
    def _slice_batch(batch, start: int, end: int):
        return {key: value[start:end] for key, value in batch.items()}

    @torch.no_grad()
    def _reembed_generated(self, image_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        image_01 = ((image_norm + 1.0) * 0.5).clamp(0, 1)
        image_key = image_01[0].detach().cpu()
        return self.embed.embedding_images({image_key: ["post_blend"]})["post_blend"]

    @torch.no_grad()
    def _build_post_blend_cleanup(self, names: dict[str, str], name_to_embed: dict[str, dict[str, torch.Tensor]]):
        align_shape = self.align.align_images(
            names["face"],
            names["shape"],
            name_to_embed,
            satd_boundary_v18=USER_SATD_BOUNDARY_V18,
            eq8_reference_blend_v18=USER_EQ8_REFERENCE_BLEND_V18,
        )
        align_color = self.align.shape_module(names["face"], names["color"], name_to_embed)

        mask_de = self.align.dilate_erosion.hair_from_mask(
            torch.cat([name_to_embed[names["face"]]["mask"], name_to_embed[names["color"]]["mask"]], dim=0)
        )
        hm_face_d = mask_de[0][0].unsqueeze(0)
        hm_color_d = mask_de[0][1].unsqueeze(0)
        hm_color_e = mask_de[1][1].unsqueeze(0)

        latent_s_face = name_to_embed[names["face"]]["S"]
        latent_s_color = name_to_embed[names["color"]]["S"]
        latent_f_author = align_shape["latent_F_align"]
        target_hair_mask = align_color["HM_X"]

        target_hair_mask_d, _ = self.align.dilate_erosion.mask(target_hair_mask)
        target_mask = (1 - hm_face_d) * (1 - hm_color_d) * (1 - target_hair_mask_d)

        s_blend_6_18 = self.blending(
            latent_s_face[:, 6:],
            latent_s_color[:, 6:],
            name_to_embed[names["face"]]["image_norm_256"] * target_mask,
            name_to_embed[names["color"]]["image_norm_256"] * hm_color_e,
        )
        s_blend = torch.cat((latent_s_face[:, :6], s_blend_6_18), dim=1)
        i_color_blend, _ = self.net.generator(
            [s_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_f_author,
        )
        i_color_blend_256 = self.downsample_256(i_color_blend)
        post_blend_embed = self._reembed_generated(i_color_blend)

        return {
            "align_shape": materialize_tensor_tree(align_shape),
            "color_blend_256": i_color_blend_256.detach().clone(),
            "cleanup_context_256": post_blend_embed["image_norm_256"].detach().clone(),
            "latent_s_cleanup": post_blend_embed["S"].detach().clone(),
            "latent_f_cleanup_base": post_blend_embed["F"].detach().clone(),
            "color_image_256": name_to_embed[names["color"]]["image_norm_256"].detach().clone(),
        }

    def forward_batch(self, batch, blend_strength: float):
        pred_images = []
        blended_images = []
        latent_list = []
        info_list = []
        total_samples = batch["face"].shape[0]
        chunk_size = max(1, min(USER_ENCODER_BATCH_SIZE, total_samples))

        for start in range(0, total_samples, chunk_size):
            end = min(total_samples, start + chunk_size)
            batch_chunk = self._slice_batch(batch, start, end)
            name_to_embed, sample_names = self.build_name_to_embed(batch_chunk)

            for names in sample_names:
                with torch.no_grad():
                    post_blend = self._build_post_blend_cleanup(names, name_to_embed)
                info = post_blend["align_shape"]
                info["latent_F_cleanup_base"] = post_blend["latent_f_cleanup_base"]
                info["color_image_256"] = post_blend["color_image_256"]

                blended_images.append(post_blend["color_blend_256"])

                satd_out, satd_aux = self.satd(
                    F_base=post_blend["latent_f_cleanup_base"],
                    F_src=info["latent_F_src"],
                    F_src_inpaint=info["latent_F_src_inpaint"],
                    cleanup_masks_256=info["cleanup_masks_256"],
                    source_rgb_256=post_blend["cleanup_context_256"],
                )
                cleanup_support = Alignment_v18._protected_cleanup_support_from_delta_masks(
                    info["delta_masks"],
                    target_hair_mask=info["HM_X"],
                    out_hw=post_blend["latent_f_cleanup_base"].shape[-2:],
                )
                latent_F = post_blend["latent_f_cleanup_base"] + blend_strength * cleanup_support * (
                    satd_out - post_blend["latent_f_cleanup_base"]
                )

                pred_image, _ = self.net.generator(
                    [post_blend["latent_s_cleanup"]],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=4,
                    end_layer=8,
                    layer_in=latent_F,
                )
                pred_images.append(self.downsample_256(pred_image))
                latent_list.append(latent_F)
                info["post_satd_support"] = cleanup_support
                info_list.append((info, satd_aux))

            del name_to_embed, sample_names, batch_chunk
            gc.collect()

        return torch.cat(pred_images, dim=0), torch.cat(blended_images, dim=0), latent_list, info_list

    def calc_losses(self, pred_images, blended_images, latent_list, info_list, schedule: dict[str, float]):
        losses = defaultdict(float)
        for idx, (latent_F, (info, _satd_aux)) in enumerate(zip(latent_list, info_list)):
            pred = pred_images[idx : idx + 1]
            blended = blended_images[idx : idx + 1]
            delta_masks = info["delta_masks"]
            remove = delta_masks["M_remove"]
            zero = torch.zeros_like(remove)
            boundary = delta_masks.get("M_boundary", zero)
            remove_halo = delta_masks.get("M_remove_halo", zero)
            remove_tail = delta_masks.get("M_remove_tail", zero)
            remove_face = delta_masks.get("M_remove_face", zero)
            remove_neck = delta_masks.get("M_remove_neck", zero)
            body_preserve = delta_masks.get("M_body_preserve", zero)
            detail_protect = delta_masks.get("M_detail_protect", zero)
            ref_overlap = delta_masks.get("M_ref_overlap", zero)
            keep = delta_masks.get("M_keep", zero)
            add = delta_masks.get("M_add", zero)

            support_256 = F.interpolate(info["post_satd_support"], size=pred.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
            main_hair_guard = dilate_mask((info["HM_X"] + add + keep + ref_overlap).clamp(0, 1), 2)
            cleanup_region = (support_256 * (1.0 - 0.98 * main_hair_guard)).clamp(0, 1)
            preserve_region = (1.0 - cleanup_region).clamp(0, 1)
            preserve_region = (preserve_region + main_hair_guard + 0.65 * body_preserve + 0.70 * detail_protect).clamp(0, 1)

            halo_region = (remove_halo + 0.12 * remove + 0.12 * boundary).clamp(0, 1)
            tail_region = (remove_tail + 0.35 * remove_neck + 0.20 * remove_halo).clamp(0, 1)
            face_region = (remove_face * (1.0 - 0.80 * detail_protect)).clamp(0, 1)
            neck_region = (remove_neck + 0.40 * remove_tail).clamp(0, 1)
            cleanup_target_region = (
                halo_region + schedule["cleanup_scale"] * (tail_region + face_region + neck_region)
            ).clamp(0, 1)
            cleanup_target_region = (cleanup_target_region * (1.0 - 0.98 * main_hair_guard)).clamp(0, 1)

            source_inpaint = info["source_inpaint_256"]
            cleanup_target = 0.72 * source_inpaint + 0.28 * blended
            face_target = 0.30 * source_inpaint + 0.70 * blended
            neck_target = 0.82 * source_inpaint + 0.18 * blended
            low_target = torch.maximum(gray_low(source_inpaint), gray_low(blended))
            keep32 = F.interpolate(preserve_region, size=latent_F.shape[-2:], mode="nearest")

            losses["preserve"] += masked_l1(pred, blended, preserve_region)
            losses["main_hair_keep"] += masked_l1(pred, blended, main_hair_guard)
            losses["body_preserve"] += masked_l1(pred, blended, body_preserve)
            losses["cleanup_target"] += masked_l1(pred, cleanup_target, cleanup_target_region)
            losses["halo"] += masked_l1(gray(pred), gray(source_inpaint), halo_region)
            losses["tail"] += masked_l1(pred, cleanup_target, tail_region)
            losses["face"] += masked_l1(pred, face_target, face_region)
            losses["neck"] += masked_l1(pred, neck_target, neck_region)
            losses["non_dark"] += masked_non_darker(gray_low(pred), low_target, cleanup_target_region)
            losses["latent_keep"] += masked_l1(latent_F, info["latent_F_cleanup_base"], keep32)

        count = max(len(info_list), 1)
        for key in list(losses.keys()):
            losses[key] = losses[key] / count

        total_loss = (
            USER_LAMBDA_PRESERVE_V18 * losses["preserve"]
            + USER_LAMBDA_MAIN_HAIR_KEEP_V18 * losses["main_hair_keep"]
            + USER_LAMBDA_BODY_PRESERVE_V18 * losses["body_preserve"]
            + USER_LAMBDA_CLEANUP_TARGET_V18 * losses["cleanup_target"]
            + USER_LAMBDA_HALO_V18 * losses["halo"]
            + USER_LAMBDA_TAIL_V18 * schedule["cleanup_scale"] * losses["tail"]
            + USER_LAMBDA_FACE_V18 * schedule["cleanup_scale"] * losses["face"]
            + USER_LAMBDA_NECK_V18 * schedule["cleanup_scale"] * losses["neck"]
            + USER_LAMBDA_NON_DARK_V18 * schedule["non_dark_scale"] * losses["non_dark"]
            + USER_LAMBDA_LATENT_KEEP_V18 * losses["latent_keep"]
        )
        losses["loss"] = total_loss
        return total_loss, losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "satd_v18_state_dict": self.satd.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "cur_iter": self.cur_iter,
        }
        torch.save(state, self.output_ckpt_dir / "last.pth")
        if epoch % USER_SAVE_CHECKPOINT_EVERY == 0:
            torch.save(state, self.output_ckpt_dir / "checkpoint.pth")
            torch.save(state, self.output_ckpt_dir / f"epoch_{epoch:03d}.pth")
        if is_best:
            torch.save(state, self.output_ckpt_dir / "best.pth")

        torch.save({"satd_v18_state_dict": self.satd.state_dict()}, self.output_ckpt_dir / "satd_for_infer_v18.pth")

    def save_preview(self, epoch: int, preview_rows: list[list[torch.Tensor]]):
        if epoch % USER_SAVE_VAL_IMAGES_EVERY != 0:
            return
        save_dir = self.output_val_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        to_pil = T.ToPILImage()
        for idx, row in enumerate(preview_rows[:USER_LOG_IMAGE_COUNT]):
            grid = torch.cat([((img[0] + 1) / 2).clamp(0, 1) for img in row], dim=2)
            to_pil(grid.cpu()).save(save_dir / f"sample_{idx:03d}.png")

    def run_epoch(self, loader, training: bool, epoch: int):
        self.satd.train(training)
        if not training:
            self.satd.eval()

        schedule = self.stage_schedule(epoch)
        total = defaultdict(float)
        preview_rows = []
        steps_in_epoch = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for step, batch in enumerate(tqdm(loader)):
                steps_in_epoch += 1
                pred_images, blended_images, latent_list, info_list = self.forward_batch(
                    batch,
                    blend_strength=schedule["blend"],
                )
                loss, losses = self.calc_losses(pred_images, blended_images, latent_list, info_list, schedule)

                if training:
                    (loss / self.grad_accum_steps).backward()
                    if (step + 1) % self.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.satd.parameters(), 5.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.logger.next_step()
                    self.cur_iter += 1

                for key, val in losses.items():
                    total[key] += float(val.detach().cpu())

                if (not training) and len(preview_rows) < USER_LOG_IMAGE_COUNT:
                    for idx, (info, _) in enumerate(info_list):
                        preview_rows.append(
                            [
                                info["source_image_256"],
                                info["shape_image_256"],
                                info["color_image_256"],
                                blended_images[idx : idx + 1],
                                pred_images[idx : idx + 1],
                            ]
                        )

                del pred_images, blended_images, latent_list, info_list, loss, losses
                if (step + 1) % 20 == 0:
                    gc.collect()

        if training and steps_in_epoch % self.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.satd.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        for key in list(total.keys()):
            total[key] /= max(steps_in_epoch, 1)
        return total, preview_rows

    def train_loop(self):
        start_epoch = 0
        if USER_RESUME_CHECKPOINT:
            ckpt = torch.load(USER_RESUME_CHECKPOINT, map_location=self.device)
            load_compatible_state_dict(self.satd, resolve_checkpoint_state_dict(ckpt))
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, RuntimeError):
                print("Warning: optimizer state is incompatible with SATD_v18; reinitializing optimizer.")
            self.best_loss = ckpt.get("best_loss", self.best_loss)
            self.cur_iter = ckpt.get("cur_iter", 0)
            start_epoch = ckpt.get("epoch", -1) + 1

        for epoch in range(start_epoch, USER_EPOCHS):
            train_losses, _ = self.run_epoch(self.train_loader, training=True, epoch=epoch)
            val_losses, preview_rows = self.run_epoch(self.val_loader, training=False, epoch=epoch)

            self.logger.log("epoch", epoch)
            for key, val in train_losses.items():
                self.logger.log(f"train/{key}", val)
            for key, val in val_losses.items():
                self.logger.log(f"val/{key}", val)
            self.save_preview(epoch, preview_rows)

            is_best = val_losses["loss"] <= self.best_loss
            if is_best:
                self.best_loss = val_losses["loss"]
            self.save_checkpoint(epoch, is_best=is_best)


def main():
    set_seed(USER_RANDOM_SEED)
    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"dataset dir: {ACTIVE_DATASET_DIR}")
    print(f"face root: {ACTIVE_FACE_IMAGE_ROOT}")
    print(f"shape root: {ACTIVE_SHAPE_IMAGE_ROOT}")
    print(f"color root: {ACTIVE_COLOR_IMAGE_ROOT}")
    triplets = load_triplets(ACTIVE_DATASET_DIR)
    if len(triplets) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(
            f"dataset.exps is smaller than the validation split size ({ACTIVE_VAL_SIZE}) "
            f"for profile {USER_DATASET_PROFILE!r}."
        )

    train_triplets, val_triplets = train_test_split(triplets, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = SATDTripletDataset_v18(
        train_triplets,
        ACTIVE_FACE_IMAGE_ROOT,
        ACTIVE_SHAPE_IMAGE_ROOT,
        ACTIVE_COLOR_IMAGE_ROOT,
    )
    val_dataset = SATDTripletDataset_v18(
        val_triplets,
        ACTIVE_FACE_IMAGE_ROOT,
        ACTIVE_SHAPE_IMAGE_ROOT,
        ACTIVE_COLOR_IMAGE_ROOT,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        collate_fn=collate_triplets,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        collate_fn=collate_triplets,
        drop_last=False,
    )

    trainer = TrainerSATD_v18(train_loader, val_loader)
    trainer.train_loop()
    trainer.logger.wandb.finish()


if __name__ == "__main__":
    main()
