"""Train a SATD variant conditioned on the production V6 S_blend path.

This is intentionally a separate experiment script. ``dataset.exps`` must
contain explicit ``face shape color`` rows; no legacy two-column fallback is
allowed because it would silently change the production S_blend input.
``satd_train_v8.py`` is left untouched; checkpoints retain the
``satd_v8_state_dict`` key so the existing V6 inference loader can consume
``satd_for_infer_v5s.pth``.
"""

import gc
import os
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

from models.Alignment_v8 import Alignment_v8
from models.Encoders import ClipBlendingModel
from models.Embedding import Embedding
from models.Net import Net
from models.SATD_v8 import SATD_v8
from utils.bicubic import BicubicDownSample
from utils.image_utils import equal_replacer
from utils.train import WandbLogger, toggle_grad


# ========================= 用户配置区域：只改这里 =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("images/satd_dataset_v5s_3000")
USER_FFHQ_ROOT = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/satd_train_v5s_3000")

USER_SMALL_DATASET_DIR = Path("images/satd_dataset_v5s_small")
# Must match pp_gen_v6's production small profile so the trained SATD sees
# the same source/shape/colour roles at inference.
USER_SMALL_FACE_ROOT = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_long")
USER_SMALL_SHAPE_ROOT = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short")
USER_SMALL_COLOR_ROOT = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short")
USER_SMALL_OUTPUT_DIR = Path("output/satd_train_v5s_small")

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"
USER_SATD_INIT_CKPT = ""

USER_BATCH_SIZE = 16
USER_EFFECTIVE_BATCH_SIZE = 16
USER_ENCODER_BATCH_SIZE = 1
USER_CPU_THREADS = 1
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_DISABLE_CUDNN_BENCHMARK = True

USER_EPOCHS = 30
USER_LR = 1e-4
USER_WEIGHT_DECAY = 1e-6
USER_VAL_SIZE_FFHQ = 30
USER_SMALL_VAL_SIZE = 20
USER_RANDOM_SEED = 3407

USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_VAL_IMAGES_EVERY = 1
USER_LOG_IMAGE_COUNT = 30

USER_USE_WANDB = False
USER_WANDB_RUN_NAME = "satd_train_v5s"
USER_WANDB_PROJECT = "HairFast-SATD-v5s"

USER_SATD_BLEND_V8 = 0.34
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0

USER_STAGE1_EPOCHS_V8 = 4
USER_STAGE2_EPOCHS_V8 = 8

USER_LAMBDA_PRESERVE_V8 = 1.05
USER_LAMBDA_BODY_PRESERVE_V8 = 1.20
USER_LAMBDA_VISIBLE_BODY_ANCHOR_V8 = 1.55
USER_LAMBDA_BODY_REVEAL_V8 = 1.20
USER_LAMBDA_CONTEXT_REVEAL_V8 = 1.00
USER_LAMBDA_REVEAL_OVERLAP_V8 = 1.20
USER_LAMBDA_REVEAL_IMPROVE_V8 = 0.85
USER_LAMBDA_CHANGE_FROM_AUTHOR_V8 = 0.95
USER_LAMBDA_SHADOW_V8 = 1.20
USER_LAMBDA_HALO_V8 = 1.45
USER_LAMBDA_TAIL_V8 = 0.95
USER_LAMBDA_FACE_V8 = 0.70
USER_LAMBDA_FACE_RESIDUE_V8 = 1.05
USER_LAMBDA_NECK_V8 = 1.20
USER_LAMBDA_NECK_RESIDUE_V8 = 1.10
USER_LAMBDA_IMPROVE_V8 = 0.55
USER_LAMBDA_FACE_SHADOW_V8 = 1.55
USER_LAMBDA_BG_SHADOW_V8 = 1.10
USER_LAMBDA_NON_DARK_V8 = 1.25
USER_LAMBDA_BG_NON_DARK_V8 = 1.10
USER_LAMBDA_OVERLAP_NON_DARK_V8 = 1.05
USER_LAMBDA_FACE_DETAIL_KEEP_V8 = 1.55
USER_LAMBDA_BODY_DETAIL_KEEP_V8 = 1.75
USER_LAMBDA_VISIBLE_BODY_DETAIL_KEEP_V8 = 1.35
USER_LAMBDA_FACE_IMPROVE_V8 = 0.95
USER_LAMBDA_BG_IMPROVE_V8 = 0.55
USER_LAMBDA_NECK_IMPROVE_V8 = 1.00
USER_LAMBDA_LATENT_KEEP_V8 = 0.03

USER_RESUME_CHECKPOINT = ""
# ========================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "face_root": USER_FFHQ_ROOT,
            "shape_root": USER_FFHQ_ROOT,
            "color_root": USER_COLOR_ROOT_FFHQ,
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
    raise RuntimeError("USER_EFFECTIVE_BATCH_SIZE 必须大于等于 USER_BATCH_SIZE，且能被 USER_BATCH_SIZE 整除。")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gray(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    g = gray(image)
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
    grad_x = F.conv2d(g, kernel_x, padding=1)
    grad_y = F.conv2d(g, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


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


def normalize_map(mask: torch.Tensor) -> torch.Tensor:
    return (mask / mask.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)).clamp(0, 1)


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


def masked_relative_improvement(
    pred: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    pred_err = ((pred - target).abs() * mask).sum() / denom
    baseline_err = ((baseline - target).abs() * mask).sum() / denom
    return torch.relu(pred_err - baseline_err + margin)


def masked_non_darker(
    pred: torch.Tensor,
    baseline: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    return (torch.relu(baseline - pred) * mask).sum() / denom


def masked_change_from_baseline(
    pred: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    change_ratio: float = 0.35,
    gap_floor: float = 0.02,
) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    baseline_gap = (baseline - target).abs()
    pred_shift = (pred - baseline).abs()
    desired_shift = torch.relu(baseline_gap - gap_floor) * change_ratio
    return (torch.relu(desired_shift - pred_shift) * mask).sum() / denom


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


class SATDPairDataset_v5s(Dataset):
    """SATD pairs with an explicit color reference for S_blend conditioning."""
    def __init__(self, entries: list[tuple[str, str, str]], face_root: Path, shape_root: Path, color_root: Path):
        self.entries = entries
        self.face_root = face_root
        self.shape_root = shape_root
        self.color_root = color_root
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.entries)

    def _read_image(self, root: Path, name: str) -> torch.Tensor:
        path = root / name
        if not path.exists():
            path = root / f"{Path(name).stem}.png"
        if not path.exists():
            path = root / f"{Path(name).stem}.jpg"
        if not path.exists():
            path = root / f"{Path(name).stem}.jpeg"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {name}.png/.jpg in {root}")
        with Image.open(path) as image:
            return self.to_tensor(image.convert("RGB"))

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


def collate_pairs(batch):
    return {
        "face_name": [item["face_name"] for item in batch],
        "shape_name": [item["shape_name"] for item in batch],
        "color_name": [item["color_name"] for item in batch],
        "face": torch.stack([item["face"] for item in batch]),
        "shape": torch.stack([item["shape"] for item in batch]),
        "color": torch.stack([item["color"] for item in batch]),
    }


def _image_stem_exists(root: Path, name: str) -> bool:
    if not name or name.strip().lower() in {"-", "_", "none", "null"}:
        return False
    candidate = root / name
    if candidate.is_file():
        return True
    stem = Path(name).stem
    return any((root / f"{stem}{suffix}").is_file() for suffix in (".png", ".jpg", ".jpeg"))


def load_pairs(
    dataset_dir: Path,
    face_root: Path,
    shape_root: Path,
    color_root: Path,
) -> list[tuple[str, str, str]]:
    rows = []
    invalid_rows = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            items = line.strip().split()
            if not items or items[0].startswith("#"):
                continue
            # A missing colour is not recoverable: silently substituting a
            # shape image changes S_blend and makes this SATD checkpoint
            # incompatible with the production PP/V6 triplet.
            if len(items) != 3:
                invalid_rows.append(f"line {line_number}: expected 3 columns, got {len(items)}")
                continue
            row = (items[0], items[1], items[2])
            if not (
                _image_stem_exists(face_root, row[0])
                and _image_stem_exists(shape_root, row[1])
                and _image_stem_exists(color_root, row[2])
            ):
                invalid_rows.append(
                    f"line {line_number}: missing face/shape/color image for {row!r}"
                )
                continue
            rows.append(row)
    if invalid_rows:
        details = "; ".join(invalid_rows[:5])
        suffix = "" if len(invalid_rows) <= 5 else f"; ... and {len(invalid_rows) - 5} more"
        raise RuntimeError(
            f"Invalid SATD-v5s manifest {dataset_dir / 'dataset.exps'}: {details}{suffix}. "
            "Regenerate it with scripts/satd_gen_v5s.py so every row is 'face shape color'."
        )
    if not rows:
        raise RuntimeError(f"No valid face/shape/color rows found in {dataset_dir / 'dataset.exps'}")
    return rows


class TrainerSATD_v5s:
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
            blending_checkpoint=USER_BLENDING_CKPT,
            pp_checkpoint=USER_PP_CKPT,
            use_satd_v8=False,
            satd_checkpoint_v8="",
            satd_blend_v8=USER_SATD_BLEND_V8,
            satd_boundary_v8=USER_SATD_BOUNDARY_V8,
            eq8_reference_blend_v8=USER_EQ8_REFERENCE_BLEND_V8,
        )

        self.net = Net(self.opts)
        self.embed = Embedding(self.opts, net=self.net).eval()
        self.align = Alignment_v8(self.opts, latent_encoder=self.embed.get_e4e_embed, net=self.net, satd_model_v8=None)
        # The author's frozen blending encoder defines the S_blend distribution
        # used by V6 inference.  It is never optimized in this trainer.
        blending_checkpoint = torch.load(USER_BLENDING_CKPT, map_location="cpu")
        self.blending_encoder = ClipBlendingModel(blending_checkpoint.get("clip", "ViT-B/32"))
        self.blending_encoder.load_state_dict(blending_checkpoint["model_state_dict"], strict=False)
        self.blending_encoder.to(self.device).eval()
        self.satd = SATD_v8().to(self.device)
        if USER_SATD_INIT_CKPT:
            ckpt = torch.load(USER_SATD_INIT_CKPT, map_location=self.device)
            load_compatible_state_dict(self.satd, ckpt.get("satd_v8_state_dict", ckpt.get("model_state_dict", ckpt)))

        self.downsample_256 = BicubicDownSample(factor=4)
        toggle_grad(self.net.generator, False)
        toggle_grad(self.blending_encoder, False)
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
        if epoch < USER_STAGE1_EPOCHS_V8:
            return {
                "blend": 0.22,
                "tail_scale": 0.20,
                "face_scale": 0.18,
                "neck_scale": 0.45,
                "improve_scale": 0.10,
                "face_shadow_scale": 0.80,
                "non_dark_scale": 1.00,
            }
        if epoch < USER_STAGE2_EPOCHS_V8:
            return {
                "blend": 0.28,
                "tail_scale": 0.78,
                "face_scale": 0.65,
                "neck_scale": 1.00,
                "improve_scale": 0.45,
                "face_shadow_scale": 1.00,
                "non_dark_scale": 1.00,
            }
        return {
            "blend": USER_SATD_BLEND_V8,
            "tail_scale": 1.00,
            "face_scale": 1.00,
            "neck_scale": 1.00,
            "improve_scale": 0.60,
            "face_shadow_scale": 1.00,
            "non_dark_scale": 1.00,
        }

    def build_name_to_embed(self, batch):
        images_to_name = defaultdict(list)
        sample_names = []
        bsz = batch["face"].shape[0]
        for idx in range(bsz):
            face, shape, color = equal_replacer(
                [batch["face"][idx], batch["shape"][idx], batch["color"][idx]]
            )
            names = {
                "face": f"face_{idx}",
                "shape": f"shape_{idx}",
                "color": f"color_{idx}",
            }
            images_to_name[face].append(names["face"])
            images_to_name[shape].append(names["shape"])
            images_to_name[color].append(names["color"])
            sample_names.append(names)
        with torch.no_grad():
            name_to_embed = self.embed.embedding_images(images_to_name)
        return name_to_embed, sample_names

    @staticmethod
    def _slice_batch(batch, start: int, end: int):
        sliced = {}
        for key, value in batch.items():
            sliced[key] = value[start:end]
        return sliced

    def forward_batch(self, batch, blend_strength: float):
        pred_images = []
        author_images = []
        color_images = []
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
                    info = self.align.prepare_satd_features(
                        names["face"],
                        names["shape"],
                        name_to_embed,
                        satd_boundary_v8=USER_SATD_BOUNDARY_V8,
                        eq8_reference_blend_v8=USER_EQ8_REFERENCE_BLEND_V8,
                    )
                    author_align = self.align.author_align_images(names["face"], names["shape"], name_to_embed)
                    # Match the author's color branch exactly: only the
                    # frozen S-space encoder is used to form S_blend.
                    I_1 = name_to_embed[names["face"]]["image_norm_256"]
                    I_3 = name_to_embed[names["color"]]["image_norm_256"]
                    mask_de = self.align.dilate_erosion.hair_from_mask(
                        torch.cat(
                            [
                                name_to_embed[names["face"]]["mask"],
                                name_to_embed[names["color"]]["mask"],
                            ],
                            dim=0,
                        )
                    )
                    hm_1d = mask_de[0][0].unsqueeze(0)
                    hm_3d, hm_3e = mask_de[0][1].unsqueeze(0), mask_de[1][1].unsqueeze(0)
                    hm_x = author_align["HM_X"]
                    hm_xd, _ = self.align.dilate_erosion.mask(hm_x)
                    target_mask = (1 - hm_1d) * (1 - hm_3d) * (1 - hm_xd)
                    latent_s_1 = name_to_embed[names["face"]]["S"]
                    latent_s_3 = name_to_embed[names["color"]]["S"]
                    if I_1 is not I_3 or I_1 is not name_to_embed[names["shape"]]["image_norm_256"]:
                        s_blend_tail = self.blending_encoder(
                            latent_s_1[:, 6:],
                            latent_s_3[:, 6:],
                            I_1 * target_mask,
                            I_3 * hm_3e,
                        )
                        s_blend = torch.cat((latent_s_1[:, :6], s_blend_tail), dim=1)
                    else:
                        s_blend = latent_s_1
                info = materialize_tensor_tree(info)
                info["latent_F_author"] = author_align["latent_F_align"].clone()

                # SATD was trained with the source S code.  V5S deliberately
                # uses the real V6 S_blend code for both baseline and cleaned
                # renders, while gradients flow only through SATD.
                author_image, _ = self.net.generator(
                    [s_blend.detach()],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=4,
                    end_layer=8,
                    layer_in=info["latent_F_author"],
                )
                author_images.append(self.downsample_256(author_image))
                color_images.append(I_3.detach())

                satd_out, satd_aux = self.satd(
                    F_base=info["latent_F_author"],
                    F_src=info["latent_F_src"],
                    F_src_inpaint=info["latent_F_src_inpaint"],
                    cleanup_masks_256=info["cleanup_masks_256"],
                    source_rgb_256=info["source_image_256"],
                )
                cleanup_support = self.align._cleanup_support_from_delta_masks(
                    info["delta_masks"],
                    out_hw=info["latent_F_author"].shape[-2:],
                )
                latent_F = info["latent_F_author"] + blend_strength * cleanup_support * (satd_out - info["latent_F_author"])

                pred_image, _ = self.net.generator(
                    [s_blend.detach()],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=4,
                    end_layer=8,
                    layer_in=latent_F,
                )
                pred_images.append(self.downsample_256(pred_image))
                latent_list.append(latent_F)
                info_list.append((info, satd_aux))

            del name_to_embed, sample_names, batch_chunk
            gc.collect()

        return (
            torch.cat(pred_images, dim=0),
            torch.cat(author_images, dim=0),
            torch.cat(color_images, dim=0),
            latent_list,
            info_list,
        )

    def calc_losses(self, pred_images, author_images, latent_list, info_list, schedule: dict[str, float]):
        losses = defaultdict(float)
        for idx, (latent_F, (info, satd_aux)) in enumerate(zip(latent_list, info_list)):
            pred = pred_images[idx:idx + 1]
            author = author_images[idx:idx + 1]
            delta_masks = info["delta_masks"]
            boundary = delta_masks["M_boundary"]
            remove = delta_masks["M_remove"]
            remove_halo = delta_masks.get("M_remove_halo", torch.zeros_like(remove))
            remove_tail = delta_masks.get("M_remove_tail", torch.zeros_like(remove))
            remove_face = delta_masks.get("M_remove_face", torch.zeros_like(remove))
            remove_neck = delta_masks.get("M_remove_neck", torch.zeros_like(remove))
            remove_context = delta_masks.get("M_remove_context", torch.zeros_like(remove))
            face_surface = delta_masks.get("M_face_surface", torch.zeros_like(remove))
            ear_surface = delta_masks.get("M_ear_surface", torch.zeros_like(remove))
            detail_protect = delta_masks.get("M_detail_protect", torch.zeros_like(remove))
            face_cleanup_surface = delta_masks.get("M_face_cleanup_surface", torch.zeros_like(remove))
            body_preserve = delta_masks.get("M_body_preserve", torch.zeros_like(remove))
            cloth_region = delta_masks.get("M_cloth_region", torch.zeros_like(remove))
            body_region = delta_masks.get("M_body_region", torch.zeros_like(remove))
            visible_body_anchor = delta_masks.get("M_visible_body_anchor", torch.zeros_like(remove))
            body_reveal = delta_masks.get("M_body_reveal", torch.zeros_like(remove))
            context_reveal = delta_masks.get("M_context_reveal", torch.zeros_like(remove))
            reveal_overlap = delta_masks.get("M_reveal_overlap", torch.zeros_like(remove))
            body_reveal_only = delta_masks.get("M_body_reveal_only", body_reveal)
            context_reveal_only = delta_masks.get("M_context_reveal_only", context_reveal)
            context_region = delta_masks.get("M_context_region", torch.zeros_like(remove))

            author_detail = normalize_map(gaussian_blur(sobel_edges(author), kernel_size=5, sigma=1.2))
            source_detail = normalize_map(gaussian_blur(sobel_edges(info["source_image_256"]), kernel_size=5, sigma=1.2))
            detail_energy = torch.maximum(author_detail, source_detail)
            face_detail_keep = (
                detail_protect
                + 0.30 * detail_energy * (face_cleanup_surface + 0.35 * ear_surface).clamp(0, 1)
            ).clamp(0, 1)
            body_detail_keep = (
                body_preserve
                + 0.35 * detail_energy * (body_region + cloth_region).clamp(0, 1)
            ).clamp(0, 1)
            stable_body_preserve = (
                body_preserve
                * (1.0 - 0.88 * visible_body_anchor)
                * (1.0 - 0.86 * body_reveal_only)
                * (1.0 - 0.74 * reveal_overlap)
                * (1.0 - 0.62 * context_reveal_only)
            ).clamp(0, 1)
            visible_body_detail_keep = (body_detail_keep * visible_body_anchor).clamp(0, 1)
            stable_body_detail_keep = (
                body_detail_keep
                * (1.0 - 0.88 * visible_body_anchor)
                * (1.0 - 0.78 * body_reveal_only)
                * (1.0 - 0.64 * reveal_overlap)
                * (1.0 - 0.52 * context_reveal_only)
            ).clamp(0, 1)
            face_cleanup_core = (
                0.60 * erode_mask(face_cleanup_surface, 1)
                + 0.40 * face_cleanup_surface
            ).clamp(0, 1)
            face_residue_seed = (remove_face + 0.68 * remove_halo).clamp(0, 1)
            face_residue_focus = (
                dilate_mask(face_residue_seed, 1)
                * face_cleanup_core
            ).clamp(0, 1)
            face_shadow_focus = (
                0.55 * erode_mask(face_residue_focus, 1)
                + 0.45 * face_residue_focus
            ).clamp(0, 1)
            neck_residue_focus = (
                dilate_mask((remove_neck + 0.78 * remove_tail + 0.20 * remove_halo).clamp(0, 1), 1)
                * (1.0 - 0.76 * body_detail_keep)
                * (1.0 - 0.45 * cloth_region)
            ).clamp(0, 1)

            shadow_region = (
                0.20 * boundary
                + 0.16 * remove
                + 0.80 * remove_halo
                + 0.08 * remove_face
                + 0.34 * remove_neck
            ).clamp(0, 1)
            shadow_region = (
                shadow_region
                * (1.0 - 0.90 * face_cleanup_surface)
                * (1.0 - 0.78 * face_detail_keep)
                * (1.0 - 0.82 * body_detail_keep)
            ).clamp(0, 1)
            halo_region = (remove_halo + 0.18 * remove + 0.16 * remove_face + 0.18 * remove_neck).clamp(0, 1)
            halo_region = (halo_region * (1.0 - 0.82 * face_detail_keep) * (1.0 - 0.85 * body_detail_keep)).clamp(0, 1)
            tail_region = ((remove_tail + 0.36 * remove_neck + 0.16 * remove_halo) * (1.0 - 0.80 * body_detail_keep)).clamp(0, 1)
            face_region = (
                face_residue_focus
                * (1.0 - 0.82 * face_detail_keep)
            ).clamp(0, 1)
            neck_region = (
                (remove_neck + 0.48 * remove_tail + 0.18 * remove_halo)
                * (1.0 - 0.82 * body_detail_keep)
            ).clamp(0, 1)
            face_shadow_region = (
                (0.82 * remove_face + 0.28 * remove_halo + 0.04 * boundary)
                * face_shadow_focus
                * (1.0 - 0.94 * face_detail_keep)
                * (1.0 - 0.15 * body_preserve)
            ).clamp(0, 1)
            background_shadow_region = (
                (0.72 * context_reveal_only + 0.28 * reveal_overlap + 0.45 * remove_context + 0.30 * remove_halo * context_region + 0.15 * boundary * context_region)
                * (1.0 - 0.50 * detail_energy * context_region)
                * (1.0 - 0.58 * visible_body_anchor)
                * (1.0 - 0.70 * body_detail_keep)
            ).clamp(0, 1)
            non_dark_region = (face_shadow_region + 0.12 * halo_region).clamp(0, 1)
            bg_non_dark_region = (background_shadow_region + 0.15 * halo_region * context_region).clamp(0, 1)
            neck_improve_region = (neck_region + 0.95 * tail_region + 0.35 * neck_residue_focus).clamp(0, 1)
            face_target = 0.42 * info["source_inpaint_256"] + 0.58 * author
            face_residue_target = gray(0.80 * info["source_inpaint_256"] + 0.20 * author)
            tail_target = 0.96 * info["source_inpaint_256"] + 0.04 * author
            neck_target = 0.94 * info["source_inpaint_256"] + 0.06 * author
            neck_residue_target = gray(0.90 * info["source_inpaint_256"] + 0.10 * author)
            visible_body_anchor_target = info["source_image_256"]
            body_reveal_target = 0.82 * info["source_inpaint_256"] + 0.18 * author
            context_reveal_target = 0.92 * info["source_inpaint_256"] + 0.08 * author
            overlap_body_logit = 1.55 * body_reveal + 0.35 * visible_body_anchor
            overlap_context_logit = context_reveal
            overlap_norm = (overlap_body_logit + overlap_context_logit).clamp(min=1e-6)
            w_body = (overlap_body_logit / overlap_norm).clamp(0, 1)
            w_context = 1.0 - w_body
            overlap_target = w_body * body_reveal_target + w_context * context_reveal_target
            overlap_non_dark_baseline = torch.maximum(gray(body_reveal_target), gray(author))
            reveal_mix_mask = (body_reveal_only + reveal_overlap + context_reveal_only).clamp(0, 1)
            reveal_mix_target = (
                body_reveal_only * body_reveal_target
                + reveal_overlap * overlap_target
                + context_reveal_only * context_reveal_target
            ) / (body_reveal_only + reveal_overlap + context_reveal_only).clamp(min=1e-6)
            author_change_mask = (
                0.95 * reveal_mix_mask
                + 0.80 * face_region
                + 0.90 * neck_region
                + 0.75 * tail_region
                + 0.45 * face_shadow_region
            ).clamp(0, 1)
            author_change_target = (
                0.42 * reveal_mix_target
                + 0.22 * face_target
                + 0.18 * neck_target
                + 0.18 * tail_target
            )
            shadow_target_low = torch.maximum(gray_low(info["source_inpaint_256"]), gray_low(author))
            face_shadow_target = shadow_target_low
            background_shadow_target = shadow_target_low
            non_dark_baseline = gray(author)

            cleanup_focus = (
                shadow_region
                + 0.85 * tail_region
                + 0.60 * face_region
                + 0.74 * neck_region
                + 0.30 * neck_residue_focus
                + 0.68 * body_reveal_only
                + 0.60 * reveal_overlap
                + 0.52 * context_reveal_only
                + 0.38 * visible_body_anchor
            ).clamp(0, 1)
            preserve_region = (1.0 - cleanup_focus).clamp(0, 1)
            keep32 = F.interpolate(preserve_region, size=(32, 32), mode="nearest")

            losses["preserve"] += masked_l1(pred, author, preserve_region)
            losses["body_preserve"] += masked_l1(pred, author, stable_body_preserve)
            losses["visible_body_anchor"] += masked_l1(pred, visible_body_anchor_target, visible_body_anchor)
            losses["body_reveal"] += masked_l1(pred, body_reveal_target, body_reveal_only)
            losses["context_reveal"] += masked_l1(pred, context_reveal_target, context_reveal_only)
            losses["reveal_overlap"] += masked_l1(pred, overlap_target, reveal_overlap)
            losses["reveal_improve"] += masked_relative_improvement(pred, reveal_mix_target, author, reveal_mix_mask)
            losses["change_from_author"] += masked_change_from_baseline(
                pred,
                author,
                author_change_target,
                author_change_mask,
            )
            losses["shadow"] += masked_l1(gray_low(pred), shadow_target_low, shadow_region)
            losses["halo"] += masked_l1(gray(pred), gray(info["source_inpaint_256"]), halo_region)
            losses["tail"] += masked_l1(pred, tail_target, tail_region)
            losses["face"] += masked_l1(pred, face_target, face_region)
            losses["face_residue"] += masked_l1(gray(pred), face_residue_target, face_region)
            losses["neck"] += masked_l1(pred, neck_target, neck_region)
            losses["neck_residue"] += masked_l1(gray(pred), neck_residue_target, neck_residue_focus)
            losses["face_shadow"] += masked_l1(gray_low(pred), face_shadow_target, face_shadow_region)
            losses["bg_shadow"] += masked_l1(gray_low(pred), background_shadow_target, background_shadow_region)
            losses["non_dark"] += masked_non_darker(gray(pred), non_dark_baseline, non_dark_region)
            losses["bg_non_dark"] += masked_non_darker(gray(pred), non_dark_baseline, bg_non_dark_region)
            losses["overlap_non_dark"] += masked_non_darker(gray(pred), overlap_non_dark_baseline, reveal_overlap)
            losses["face_improve"] += masked_relative_improvement(gray(pred), face_residue_target, gray(author), face_region)
            losses["bg_improve"] += masked_relative_improvement(gray_low(pred), background_shadow_target, gray_low(author), background_shadow_region)
            losses["neck_improve"] += masked_relative_improvement(
                gray(pred),
                gray(0.55 * tail_target + 0.45 * neck_target),
                gray(author),
                neck_improve_region,
            )
            losses["face_detail_keep"] += masked_l1(pred, author, face_detail_keep)
            losses["body_detail_keep"] += masked_l1(pred, author, stable_body_detail_keep)
            losses["visible_body_detail_keep"] += masked_l1(pred, visible_body_anchor_target, visible_body_detail_keep)
            losses["latent_keep"] += masked_l1(latent_F, info["latent_F_author"], keep32)

        count = max(len(info_list), 1)
        for key in list(losses.keys()):
            losses[key] = losses[key] / count

        total_loss = (
            USER_LAMBDA_PRESERVE_V8 * losses["preserve"]
            + USER_LAMBDA_BODY_PRESERVE_V8 * losses["body_preserve"]
            + USER_LAMBDA_VISIBLE_BODY_ANCHOR_V8 * losses["visible_body_anchor"]
            + USER_LAMBDA_BODY_REVEAL_V8 * schedule["neck_scale"] * losses["body_reveal"]
            + USER_LAMBDA_CONTEXT_REVEAL_V8 * schedule["neck_scale"] * losses["context_reveal"]
            + USER_LAMBDA_REVEAL_OVERLAP_V8 * schedule["neck_scale"] * losses["reveal_overlap"]
            + USER_LAMBDA_REVEAL_IMPROVE_V8 * schedule["improve_scale"] * losses["reveal_improve"]
            + USER_LAMBDA_CHANGE_FROM_AUTHOR_V8 * schedule["improve_scale"] * losses["change_from_author"]
            + USER_LAMBDA_SHADOW_V8 * losses["shadow"]
            + USER_LAMBDA_HALO_V8 * losses["halo"]
            + USER_LAMBDA_TAIL_V8 * schedule["tail_scale"] * losses["tail"]
            + USER_LAMBDA_FACE_V8 * schedule["face_scale"] * losses["face"]
            + USER_LAMBDA_FACE_RESIDUE_V8 * schedule["face_scale"] * losses["face_residue"]
            + USER_LAMBDA_NECK_V8 * schedule["neck_scale"] * losses["neck"]
            + USER_LAMBDA_NECK_RESIDUE_V8 * schedule["neck_scale"] * losses["neck_residue"]
            + USER_LAMBDA_FACE_SHADOW_V8 * schedule["face_shadow_scale"] * losses["face_shadow"]
            + USER_LAMBDA_BG_SHADOW_V8 * schedule["face_shadow_scale"] * losses["bg_shadow"]
            + USER_LAMBDA_NON_DARK_V8 * schedule["non_dark_scale"] * losses["non_dark"]
            + USER_LAMBDA_BG_NON_DARK_V8 * schedule["non_dark_scale"] * losses["bg_non_dark"]
            + USER_LAMBDA_OVERLAP_NON_DARK_V8 * schedule["non_dark_scale"] * losses["overlap_non_dark"]
            + USER_LAMBDA_FACE_DETAIL_KEEP_V8 * losses["face_detail_keep"]
            + USER_LAMBDA_BODY_DETAIL_KEEP_V8 * losses["body_detail_keep"]
            + USER_LAMBDA_VISIBLE_BODY_DETAIL_KEEP_V8 * losses["visible_body_detail_keep"]
            + USER_LAMBDA_FACE_IMPROVE_V8 * schedule["improve_scale"] * losses["face_improve"]
            + USER_LAMBDA_BG_IMPROVE_V8 * schedule["improve_scale"] * losses["bg_improve"]
            + USER_LAMBDA_NECK_IMPROVE_V8 * schedule["improve_scale"] * losses["neck_improve"]
            + USER_LAMBDA_LATENT_KEEP_V8 * losses["latent_keep"]
        )
        losses["loss"] = total_loss
        return total_loss, losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "satd_v8_state_dict": self.satd.state_dict(),
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

        # Keep the v8 state-dict key for V6 loader compatibility, but use a
        # distinct filename so this experiment can never overwrite V8 output.
        torch.save({"satd_v8_state_dict": self.satd.state_dict()}, self.output_ckpt_dir / "satd_for_infer_v5s.pth")

    def save_preview(self, epoch: int, preview_rows: list[list[torch.Tensor]]):
        if epoch % USER_SAVE_VAL_IMAGES_EVERY != 0:
            return
        save_dir = self.output_val_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        to_pil = T.ToPILImage()
        for idx, row in enumerate(preview_rows[:USER_LOG_IMAGE_COUNT]):
            # source | shape | explicit colour reference | author | SATD
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
                pred_images, author_images, color_images, latent_list, info_list = self.forward_batch(
                    batch,
                    blend_strength=schedule["blend"],
                )
                loss, losses = self.calc_losses(pred_images, author_images, latent_list, info_list, schedule)

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
                                color_images[idx:idx + 1],
                                author_images[idx:idx + 1],
                                pred_images[idx:idx + 1],
                            ]
                        )

                del pred_images, author_images, color_images, latent_list, info_list, loss, losses
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
            load_compatible_state_dict(self.satd, ckpt["satd_v8_state_dict"])
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, RuntimeError):
                print("Warning: optimizer state is incompatible with the current SATD_v8 architecture; reinitializing optimizer.")
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
    pairs = load_pairs(
        ACTIVE_DATASET_DIR,
        ACTIVE_FACE_IMAGE_ROOT,
        ACTIVE_SHAPE_IMAGE_ROOT,
        ACTIVE_COLOR_IMAGE_ROOT,
    )
    if len(pairs) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(
            f"dataset.exps is smaller than the validation split size ({ACTIVE_VAL_SIZE}) "
            f"for profile {USER_DATASET_PROFILE!r}."
        )

    train_pairs, val_pairs = train_test_split(pairs, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = SATDPairDataset_v5s(
        train_pairs,
        ACTIVE_FACE_IMAGE_ROOT,
        ACTIVE_SHAPE_IMAGE_ROOT,
        ACTIVE_COLOR_IMAGE_ROOT,
    )
    val_dataset = SATDPairDataset_v5s(
        val_pairs,
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
        collate_fn=collate_pairs,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        collate_fn=collate_pairs,
        drop_last=False,
    )

    trainer = TrainerSATD_v5s(train_loader, val_loader)
    trainer.train_loop()
    trainer.logger.wandb.finish()


if __name__ == "__main__":
    main()
