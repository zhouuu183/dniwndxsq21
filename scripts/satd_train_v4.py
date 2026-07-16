import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

from losses.pp_losses import LPIPSScaleLoss
from models.Alignment_v4 import Alignment_v4
from models.Embedding import Embedding
from models.Net import Net
from models.SATD_v4 import SATD_v4
from utils.bicubic import BicubicDownSample
from utils.image_utils import equal_replacer
from utils.train import WandbLogger, toggle_grad


# ========================= 用户配置区域：只改这里 =========================
# 指定当前进程可见的物理 GPU。
USER_CUDA_VISIBLE_DEVICES = "0"

# PyTorch 实际使用的设备。通常写 "cuda"。
USER_DEVICE = "cuda"

# 可选 "ffhq" 或 "small"。
USER_DATASET_PROFILE = "small"

# FFHQ 大数据集配置。
USER_DATASET_DIR_FFHQ = Path("images/satd_dataset_v4_3000")
USER_FFHQ_ROOT = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/satd_train_v4_3000")

# small 小数据集配置。
USER_SMALL_DATASET_DIR = Path("images/satd_dataset_v4_small7")
USER_SMALL_FACE_ROOT = Path("images/FFHQ_long")
USER_SMALL_SHAPE_ROOT = Path("images/FFHQ_short")
USER_SMALL_OUTPUT_DIR = Path("output/satd_train_v4_small7")

# StyleGAN2 权重路径。
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"

# rotate 阶段最佳权重。
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"

# SATD_v4 初始化权重。第一次训练保持空字符串即可。
USER_SATD_INIT_CKPT = ""

# 单次真实 batch size。
USER_BATCH_SIZE = 8

# 目标有效 batch size。比如 batch=1, effective=4 表示累计 4 次梯度后更新。
USER_EFFECTIVE_BATCH_SIZE = 16

# Embedding 阶段的内部 batch。公共服务器内存紧张时建议 1。
USER_ENCODER_BATCH_SIZE = 1

# CPU 线程上限。共享服务器上建议先压到 1，减少线程栈和并行算子的内存峰值。
USER_CPU_THREADS = 1

# DataLoader worker 数量。为稳妥建议先用 0。
USER_NUM_WORKERS = 0

# 是否启用 pin_memory。当前更建议关掉，优先稳定。
USER_PIN_MEMORY = False

# 是否关闭 cudnn benchmark。当前更建议关掉，优先稳定。
USER_DISABLE_CUDNN_BENCHMARK = True

# 总 epoch 数。
USER_EPOCHS = 30

# 学习率。
USER_LR = 1e-4

# 权重衰减。
USER_WEIGHT_DECAY = 1e-6

# 验证集样本数。
USER_VAL_SIZE_FFHQ = 20
USER_SMALL_VAL_SIZE = 4

# 随机种子。
USER_RANDOM_SEED = 3407

# 每隔多少个 epoch 保存一次 checkpoint.pth 和 epoch_xxx.pth。
USER_SAVE_CHECKPOINT_EVERY = 1

# 每隔多少个 epoch 保存一次本地验证图。
USER_SAVE_VAL_IMAGES_EVERY = 1

# 每轮最多保存多少组验证可视化。
USER_LOG_IMAGE_COUNT = 10

# 是否启用 wandb。服务器没配好时建议 False。
USER_USE_WANDB = False
USER_WANDB_RUN_NAME = "satd_train_v4"
USER_WANDB_PROJECT = "HairFast-SATD-v4"

# SATD 输出和原始 Eq.(8) 对齐结果的融合比例。
USER_SATD_BLEND_V4 = 0.40
USER_EQ8_REFERENCE_BLEND_V4 = 0.30
USER_SATD_CLEANUP_ONLY_V4 = True
USER_SATD_CLEANUP_BLEND_V4 = 0.24

# 显式差值边界带宽度。
USER_SATD_BOUNDARY_V4 = 8

# 各损失项权重。
USER_LAMBDA_PRESERVE = 1.5
USER_LAMBDA_REMOVE = 2.25
USER_LAMBDA_REMOVE_HALO = 0.75
USER_LAMBDA_REMOVE_GRAY = 0.60
USER_LAMBDA_SHAPE_EDGE = 2.60
USER_LAMBDA_SHAPE_GRAY = 1.35
USER_LAMBDA_LATENT_KEEP = 0.06
USER_LAMBDA_SHAPE_RGB = 1.35
USER_LAMBDA_BODY_PRESERVE = 1.40
USER_LAMBDA_TAIL_GRAY = 0.90
USER_LAMBDA_SHAPE_SELF_SIM = 0.0
USER_LAMBDA_TRANSFER_LPIPS = 0.20
USER_LAMBDA_BANGS_DIRECTION = 0.25
USER_BANGS_FACE_EXPAND = 17
USER_BANGS_UPPER_Y_RATIO = 0.62
USER_BANGS_EDGE_THRESHOLD = 0.03
USER_SELF_SIM_MAX_POINTS = 196
USER_LAMBDA_HAIR_HIGH_FREQ = 0.90
USER_LAMBDA_HAIR_DIRECTION = 0.30
USER_LAMBDA_NO_BANGS = 0.35
USER_LAMBDA_REMOVE_FACE = 1.10
USER_LAMBDA_REMOVE_NECK = 0.95
USER_NO_BANGS_FACE_EXPAND = 11
USER_NO_BANGS_UPPER_Y_RATIO = 0.55

# 断点恢复。为空表示不恢复。
USER_RESUME_CHECKPOINT = ""
# ========================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "face_root": USER_FFHQ_ROOT,
            "shape_root": USER_FFHQ_ROOT,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "val_size": USER_VAL_SIZE_FFHQ,
        },
        "small": {
            "dataset_dir": USER_SMALL_DATASET_DIR,
            "face_root": USER_SMALL_FACE_ROOT,
            "shape_root": USER_SMALL_SHAPE_ROOT,
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
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image.device).view(1, 1, 3, 3)
    edge_x = F.conv2d(g, kernel_x, padding=1)
    edge_y = F.conv2d(g, kernel_y, padding=1)
    return torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + 1e-6)


def sobel_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = gray(image)
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(g, kernel_x, padding=1)
    grad_y = F.conv2d(g, kernel_y, padding=1)
    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
    return grad_x, grad_y, magnitude


def dilate_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width)


def gaussian_blur(image: torch.Tensor, kernel_size: int = 11, sigma: float = 3.0) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return image

    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords.pow(2)) / max(2 * sigma ** 2, 1e-6))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(image.shape[1], 1, -1, -1)
    return F.conv2d(image, kernel_2d, padding=kernel_size // 2, groups=image.shape[1])


def low_pass_filter(image: torch.Tensor, kernel_size: int = 11, sigma: float = 3.0) -> torch.Tensor:
    return gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)


def high_pass_filter(image: torch.Tensor, kernel_size: int = 11, sigma: float = 3.0) -> torch.Tensor:
    return image - low_pass_filter(image, kernel_size=kernel_size, sigma=sigma)


def build_bangs_region(
    delta_masks: dict[str, torch.Tensor],
    face_expand: int = USER_BANGS_FACE_EXPAND,
    upper_y_ratio: float = USER_BANGS_UPPER_Y_RATIO,
) -> torch.Tensor:
    tgt = delta_masks["M_tgt"]
    add = delta_masks["M_add"]
    keep = delta_masks.get("M_keep", torch.zeros_like(tgt))
    boundary = delta_masks.get("M_boundary", torch.zeros_like(tgt))
    ref_overlap = delta_masks.get("M_ref_overlap", torch.zeros_like(tgt))
    body_preserve = delta_masks.get("M_body_preserve", torch.zeros_like(tgt))
    face_region = delta_masks.get("M_face_region", torch.zeros_like(tgt))

    face_support = dilate_mask(face_region, face_expand)
    _, _, height, _ = tgt.shape
    y_coords = torch.linspace(0.0, 1.0, steps=height, device=tgt.device).view(1, 1, height, 1)
    upper_region = (y_coords < upper_y_ratio).float()

    transfer_focus = (add + 0.55 * keep + 0.90 * ref_overlap + 0.75 * boundary).clamp(0, 1)
    bangs_region = (tgt * face_support * upper_region * transfer_focus).clamp(0, 1)
    bangs_region = (bangs_region * (1 - 0.15 * body_preserve)).clamp(0, 1)
    return bangs_region


def build_hair_direction_region(delta_masks: dict[str, torch.Tensor]) -> torch.Tensor:
    tgt = delta_masks["M_tgt"]
    boundary = delta_masks.get("M_boundary", torch.zeros_like(tgt))
    ref_overlap = delta_masks.get("M_ref_overlap", torch.zeros_like(tgt))
    body_preserve = delta_masks.get("M_body_preserve", torch.zeros_like(tgt))

    hair_region = (tgt + 0.75 * ref_overlap + 0.45 * boundary).clamp(0, 1)
    hair_region = (hair_region * (1 - 0.20 * body_preserve)).clamp(0, 1)
    return hair_region


def build_forehead_clear_region(
    delta_masks: dict[str, torch.Tensor],
    face_expand: int = USER_NO_BANGS_FACE_EXPAND,
    upper_y_ratio: float = USER_NO_BANGS_UPPER_Y_RATIO,
) -> torch.Tensor:
    tgt = delta_masks["M_tgt"]
    face_region = delta_masks.get("M_face_region", torch.zeros_like(tgt))
    body_preserve = delta_masks.get("M_body_preserve", torch.zeros_like(tgt))

    face_shell = (dilate_mask(face_region, face_expand) - face_region).clamp(0, 1)
    _, _, height, _ = tgt.shape
    y_coords = torch.linspace(0.0, 1.0, steps=height, device=tgt.device).view(1, 1, height, 1)
    upper_region = (y_coords < upper_y_ratio).float()

    clear_region = (face_shell * upper_region * (1 - tgt)).clamp(0, 1)
    clear_region = (clear_region * (1 - 0.25 * body_preserve)).clamp(0, 1)
    return clear_region


def masked_orientation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    edge_threshold: float = USER_BANGS_EDGE_THRESHOLD,
) -> torch.Tensor:
    pred_gx, pred_gy, pred_mag = sobel_gradients(pred)
    target_gx, target_gy, target_mag = sobel_gradients(target)

    dot = (pred_gx * target_gx + pred_gy * target_gy) / (pred_mag * target_mag + 1e-6)
    orientation_error = 1.0 - dot.abs().clamp(0, 1)
    edge_weight = (target_mag - edge_threshold).clamp(min=0.0)
    edge_weight = edge_weight / edge_weight.mean().clamp(min=1e-6)
    weight = mask * edge_weight
    denom = weight.sum().clamp(min=1.0)
    return (orientation_error * weight).sum() / denom


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target).abs() * mask).sum() / denom


def masked_affinity_loss(
    pred_feat: torch.Tensor,
    target_feat: torch.Tensor,
    mask: torch.Tensor,
    max_points: int = USER_SELF_SIM_MAX_POINTS,
) -> torch.Tensor:
    if mask.shape[-2:] != pred_feat.shape[-2:]:
        mask = F.interpolate(mask.float(), size=pred_feat.shape[-2:], mode="nearest")

    per_sample = []
    for batch_idx in range(pred_feat.shape[0]):
        mask_flat = mask[batch_idx, 0].reshape(-1)
        valid = torch.nonzero(mask_flat > 1e-4, as_tuple=False).flatten()
        if valid.numel() < 2:
            per_sample.append(pred_feat.new_zeros(()))
            continue

        if valid.numel() > max_points:
            weights = mask_flat[valid]
            valid = valid[torch.topk(weights, k=max_points).indices]

        pred_tokens = pred_feat[batch_idx].reshape(pred_feat.shape[1], -1).transpose(0, 1)[valid]
        target_tokens = target_feat[batch_idx].reshape(target_feat.shape[1], -1).transpose(0, 1)[valid]

        pred_tokens = F.normalize(pred_tokens, dim=1, eps=1e-6)
        target_tokens = F.normalize(target_tokens, dim=1, eps=1e-6)

        pred_affinity = pred_tokens @ pred_tokens.transpose(0, 1)
        target_affinity = target_tokens @ target_tokens.transpose(0, 1)
        per_sample.append(F.smooth_l1_loss(pred_affinity, target_affinity))

    return torch.stack(per_sample).mean()


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


class SATDPairDataset_v4(Dataset):
    def __init__(self, entries: list[tuple[str, str]], face_root: Path, shape_root: Path):
        self.entries = entries
        self.face_root = face_root
        self.shape_root = shape_root
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.entries)

    def _read_image(self, root: Path, name: str) -> torch.Tensor:
        path = root / f"{name}.png"
        if not path.exists():
            path = root / f"{name}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {name}.png/.jpg in {root}")
        with Image.open(path) as image:
            return self.to_tensor(image.convert("RGB"))

    def __getitem__(self, idx):
        face, shape = self.entries[idx]
        return {
            "face_name": face,
            "shape_name": shape,
            "face": self._read_image(self.face_root, face),
            "shape": self._read_image(self.shape_root, shape),
        }


def collate_pairs(batch):
    return {
        "face_name": [item["face_name"] for item in batch],
        "shape_name": [item["shape_name"] for item in batch],
        "face": torch.stack([item["face"] for item in batch]),
        "shape": torch.stack([item["shape"] for item in batch]),
    }


def load_pairs(dataset_dir: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split()
            if len(items) == 2:
                pairs.append((items[0], items[1]))
    return pairs


class TrainerSATD_v4:
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
            use_satd_v4=False,
            satd_checkpoint_v4="",
            satd_blend_v4=USER_SATD_BLEND_V4,
            eq8_reference_blend_v4=USER_EQ8_REFERENCE_BLEND_V4,
            satd_cleanup_only_v4=USER_SATD_CLEANUP_ONLY_V4,
            satd_cleanup_blend_v4=USER_SATD_CLEANUP_BLEND_V4,
            satd_boundary_v4=USER_SATD_BOUNDARY_V4,
        )

        self.net = Net(self.opts)
        self.embed = Embedding(self.opts, net=self.net).eval()
        self.align = Alignment_v4(self.opts, latent_encoder=self.embed.get_e4e_embed, net=self.net, satd_model=None)
        self.satd = SATD_v4().to(self.device)
        self.cleanup_only = USER_SATD_CLEANUP_ONLY_V4
        self.cleanup_blend = USER_SATD_CLEANUP_BLEND_V4
        if USER_SATD_INIT_CKPT:
            ckpt = torch.load(USER_SATD_INIT_CKPT, map_location=self.device)
            load_compatible_state_dict(self.satd, ckpt.get("satd_state_dict", ckpt.get("model_state_dict", ckpt)))

        self.downsample_256 = BicubicDownSample(factor=4)
        self.transfer_lpips = None
        if (not self.cleanup_only) and USER_LAMBDA_TRANSFER_LPIPS > 0:
            self.transfer_lpips = LPIPSScaleLoss().to(self.device).eval()
            toggle_grad(self.transfer_lpips, False)
        toggle_grad(self.net.generator, False)
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

    def build_name_to_embed(self, batch):
        images_to_name = defaultdict(list)
        sample_names = []
        bsz = batch["face"].shape[0]
        for idx in range(bsz):
            face, shape = equal_replacer([
                batch["face"][idx],
                batch["shape"][idx],
            ])
            names = {"face": f"face_{idx}", "shape": f"shape_{idx}"}
            images_to_name[face].append(names["face"])
            images_to_name[shape].append(names["shape"])
            sample_names.append(names)
        with torch.no_grad():
            name_to_embed = self.embed.embedding_images(images_to_name)
        return name_to_embed, sample_names

    def forward_batch(
        self,
        batch,
        render_transfer_images: bool = False,
        render_author_images: bool = False,
    ):
        name_to_embed, sample_names = self.build_name_to_embed(batch)

        pred_images = []
        transfer_images = [] if (render_transfer_images and not self.cleanup_only) else None
        author_images = [] if render_author_images else None
        latent_list = []
        info_list = []
        for names in sample_names:
            with torch.no_grad():
                info = self.align.prepare_satd_features(names["face"], names["shape"], name_to_embed)
            info = materialize_tensor_tree(info)
            author_align = None
            if self.cleanup_only or render_author_images:
                with torch.no_grad():
                    author_align = self.align.author_align_images(names["face"], names["shape"], name_to_embed)
                info["latent_F_author"] = author_align["latent_F_align"].clone()

            base_latent = info["latent_F_author"] if self.cleanup_only else info.get("latent_F_base", info["latent_F_eq8"])

            if render_author_images:
                author_image, _ = self.net.generator(
                    [info["latent_S_src"]],
                    input_is_latent=True,
                    return_latents=False,
                    start_layer=4,
                    end_layer=8,
                    layer_in=info["latent_F_author"],
                )
                author_images.append(self.downsample_256(author_image))

            _, satd_aux = self.satd(
                F_eq8=base_latent,
                F_src=info["latent_F_src"],
                F_src_inpaint=info["latent_F_src_inpaint"],
                F_shape_inpaint=info["latent_F_shape_inpaint"],
                F_ref=info["latent_F_ref"],
                masks_256=info["satd_masks_256"],
                source_rgb_256=info["source_image_256"],
            )
            transfer_mask, cleanup_mask = self.align._satd_residual_masks(
                info["delta_masks"],
                out_hw=base_latent.shape[-2:],
                cleanup_only=self.cleanup_only,
            )
            cleanup_delta = satd_aux["cleanup_delta"]
            if self.cleanup_only:
                latent_F = base_latent + self.cleanup_blend * cleanup_mask * cleanup_delta
            else:
                transfer_delta = satd_aux["transfer_delta"]
                transfer_latent = base_latent + USER_SATD_BLEND_V4 * transfer_mask * transfer_delta
                if render_transfer_images:
                    transfer_image, _ = self.net.generator(
                        [info["latent_S_src"]],
                        input_is_latent=True,
                        return_latents=False,
                        start_layer=4,
                        end_layer=8,
                        layer_in=transfer_latent,
                    )
                    transfer_images.append(self.downsample_256(transfer_image))

                latent_F = transfer_latent
                latent_F = latent_F + (0.60 * USER_SATD_BLEND_V4) * cleanup_mask * cleanup_delta
            pred_image, _ = self.net.generator(
                [info["latent_S_src"]],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F,
            )
            pred_images.append(self.downsample_256(pred_image))
            latent_list.append(latent_F)
            info_list.append((info, satd_aux))

        return (
            torch.cat(pred_images, dim=0),
            torch.cat(transfer_images, dim=0) if transfer_images is not None else None,
            torch.cat(author_images, dim=0) if author_images is not None else None,
            latent_list,
            info_list,
        )

    def calc_losses(self, pred_images, transfer_images, latent_list, info_list):
        losses = defaultdict(float)
        for idx, (latent_F, (info, satd_aux)) in enumerate(zip(latent_list, info_list)):
            pred = pred_images[idx:idx + 1]
            transfer_pred = transfer_images[idx:idx + 1] if transfer_images is not None else None
            delta_masks = info["delta_masks"]
            M_add = delta_masks["M_add"]
            M_remove = delta_masks["M_remove"]
            M_keep = delta_masks["M_keep"]
            M_boundary = delta_masks["M_boundary"]
            M_remove_halo = delta_masks.get("M_remove_halo", torch.zeros_like(M_remove))
            M_ref_overlap = delta_masks.get("M_ref_overlap", torch.zeros_like(M_add))
            M_body_preserve = delta_masks.get("M_body_preserve", torch.zeros_like(M_add))
            M_remove_tail = delta_masks.get("M_remove_tail", torch.zeros_like(M_add))
            M_remove_face = delta_masks.get("M_remove_face", torch.zeros_like(M_add))
            M_remove_neck = delta_masks.get("M_remove_neck", torch.zeros_like(M_add))

            preserve_mask = (1 - (M_add + M_remove + 0.75 * M_remove_halo + 0.50 * M_boundary).clamp(0, 1)).clamp(0, 1)
            shape_region = (M_add + 0.45 * M_keep + M_ref_overlap + 0.75 * M_boundary).clamp(0, 1)
            remove_region = (M_remove + 0.55 * M_remove_halo + 0.35 * M_remove_tail).clamp(0, 1)
            remove_region = (remove_region * (1 - 0.92 * M_body_preserve)).clamp(0, 1)
            tail_region = (M_remove_tail * (1 - 0.92 * M_body_preserve)).clamp(0, 1)
            transfer_texture_region = (M_add + 0.80 * M_keep + 1.10 * M_ref_overlap + 0.85 * M_boundary).clamp(0, 1)
            transfer_texture_region = (transfer_texture_region * (1 - 0.20 * M_body_preserve)).clamp(0, 1)
            remove_mask = (M_remove * (1 - 0.85 * M_body_preserve)).clamp(0, 1)
            remove_halo_mask = (M_remove_halo * (1 - 0.92 * M_body_preserve)).clamp(0, 1)
            face_cleanup_region = M_remove_face.clamp(0, 1)
            neck_cleanup_region = ((M_remove_neck + 0.20 * M_remove_tail) * (1 - 0.90 * M_body_preserve)).clamp(0, 1)

            losses["preserve"] += masked_l1(pred, info["source_image_256"], preserve_mask)
            losses["body_preserve"] += masked_l1(pred, info["source_image_256"], M_body_preserve)
            losses["remove"] += masked_l1(pred, info["source_inpaint_256"], remove_mask)
            losses["remove_halo"] += masked_l1(pred, info["source_inpaint_256"], remove_halo_mask)
            losses["remove_gray"] += masked_l1(gray(pred), gray(info["source_inpaint_256"]), remove_region)
            losses["tail_gray"] += masked_l1(gray(pred), gray(info["source_inpaint_256"]), tail_region)
            losses["remove_face"] += masked_l1(pred, info["source_inpaint_256"], face_cleanup_region)
            losses["remove_neck"] += masked_l1(pred, info["source_inpaint_256"], neck_cleanup_region)

            if self.cleanup_only:
                cleanup_focus = (
                    M_remove
                    + 0.90 * M_remove_halo
                    + 0.60 * M_remove_tail
                    + 0.65 * M_remove_face
                    + 0.50 * M_remove_neck
                ).clamp(0, 1)
                cleanup_focus = (cleanup_focus * (1 - 0.92 * M_body_preserve)).clamp(0, 1)
                keep32 = F.interpolate((1 - cleanup_focus).clamp(0, 1), size=(32, 32), mode="nearest")
                latent_keep_target = info.get("latent_F_author", info["latent_F_eq8"])
                zero = latent_F.new_zeros(())
                losses["shape_edge"] += zero
                losses["shape_gray"] += zero
                losses["shape_rgb"] += zero
                losses["shape_self_sim"] += zero
                losses["transfer_lpips"] += zero
                losses["bangs_direction"] += zero
                losses["hair_direction"] += zero
                losses["hair_high_freq"] += zero
                losses["no_bangs"] += zero
            else:
                pred_edges = sobel_edges(pred)
                shape_edges = sobel_edges(info["shape_inpaint_256"])
                losses["shape_edge"] += masked_l1(pred_edges, shape_edges, shape_region)

                losses["shape_gray"] += masked_l1(gray(pred), gray(info["shape_inpaint_256"]), shape_region)
                losses["shape_rgb"] += masked_l1(
                    pred,
                    info["shape_inpaint_256"],
                    (M_add + 0.45 * M_keep + 0.60 * M_ref_overlap + 0.75 * M_boundary).clamp(0, 1),
                )

                keep32 = F.interpolate((M_keep * (1 - 0.55 * M_ref_overlap)).clamp(0, 1), size=(32, 32), mode="nearest")
                latent_keep_target = info.get("latent_F_base", info["latent_F_eq8"])
            losses["latent_keep"] += masked_l1(
                latent_F,
                latent_keep_target,
                keep32,
            )

            if (not self.cleanup_only) and USER_LAMBDA_SHAPE_SELF_SIM > 0:
                transfer32 = F.interpolate(
                    (M_add + 0.35 * M_keep + 0.75 * M_ref_overlap + 0.50 * M_boundary).clamp(0, 1),
                    size=(32, 32),
                    mode="nearest",
                )
                transfer_latent = info.get("latent_F_base", info["latent_F_eq8"]) + satd_aux["transfer_delta"]
                losses["shape_self_sim"] += masked_affinity_loss(
                    transfer_latent,
                    info["latent_F_shape_inpaint"],
                    transfer32,
                )
            else:
                losses["shape_self_sim"] += latent_F.new_zeros(())

            if (not self.cleanup_only) and self.transfer_lpips is not None and transfer_pred is not None:
                lpips_target = 0.70 * info["shape_inpaint_256"] + 0.30 * info["shape_image_256"]
                losses["transfer_lpips"] += self.transfer_lpips(
                    transfer_pred * transfer_texture_region,
                    lpips_target * transfer_texture_region,
                )
            else:
                losses["transfer_lpips"] += latent_F.new_zeros(())

            if (not self.cleanup_only) and USER_LAMBDA_BANGS_DIRECTION > 0 and transfer_pred is not None:
                bangs_region = build_bangs_region(delta_masks)
                losses["bangs_direction"] += masked_orientation_loss(
                    transfer_pred,
                    info["shape_inpaint_256"],
                    bangs_region,
                )
            else:
                losses["bangs_direction"] += latent_F.new_zeros(())

            if (not self.cleanup_only) and USER_LAMBDA_HAIR_DIRECTION > 0 and transfer_pred is not None:
                hair_direction_region = build_hair_direction_region(delta_masks)
                losses["hair_direction"] += masked_orientation_loss(
                    transfer_pred,
                    info["shape_inpaint_256"],
                    hair_direction_region,
                )
            else:
                losses["hair_direction"] += latent_F.new_zeros(())

            if (not self.cleanup_only) and USER_LAMBDA_HAIR_HIGH_FREQ > 0 and transfer_pred is not None:
                transfer_01 = ((transfer_pred + 1) / 2).clamp(0, 1)
                shape_ref_01 = ((info["shape_image_256"] + 1) / 2).clamp(0, 1)
                losses["hair_high_freq"] += masked_l1(
                    high_pass_filter(transfer_01),
                    high_pass_filter(shape_ref_01),
                    transfer_texture_region,
                )
            else:
                losses["hair_high_freq"] += latent_F.new_zeros(())

            if (not self.cleanup_only) and USER_LAMBDA_NO_BANGS > 0 and transfer_pred is not None:
                forehead_clear_region = build_forehead_clear_region(delta_masks)
                losses["no_bangs"] += masked_l1(
                    gray(transfer_pred),
                    gray(info["shape_inpaint_256"]),
                    forehead_clear_region,
                )
            else:
                losses["no_bangs"] += latent_F.new_zeros(())

        count = max(len(info_list), 1)
        for key in list(losses.keys()):
            losses[key] = losses[key] / count

        total_loss = (
            USER_LAMBDA_PRESERVE * losses["preserve"]
            + USER_LAMBDA_BODY_PRESERVE * losses["body_preserve"]
            + USER_LAMBDA_REMOVE * losses["remove"]
            + USER_LAMBDA_REMOVE_HALO * losses["remove_halo"]
            + USER_LAMBDA_REMOVE_GRAY * losses["remove_gray"]
            + USER_LAMBDA_TAIL_GRAY * losses["tail_gray"]
            + USER_LAMBDA_SHAPE_EDGE * losses["shape_edge"]
            + USER_LAMBDA_SHAPE_GRAY * losses["shape_gray"]
            + USER_LAMBDA_LATENT_KEEP * losses["latent_keep"]
            + USER_LAMBDA_SHAPE_RGB * losses["shape_rgb"]
            + USER_LAMBDA_SHAPE_SELF_SIM * losses["shape_self_sim"]
            + USER_LAMBDA_TRANSFER_LPIPS * losses["transfer_lpips"]
            + USER_LAMBDA_BANGS_DIRECTION * losses["bangs_direction"]
            + USER_LAMBDA_HAIR_HIGH_FREQ * losses["hair_high_freq"]
            + USER_LAMBDA_HAIR_DIRECTION * losses["hair_direction"]
            + USER_LAMBDA_NO_BANGS * losses["no_bangs"]
            + USER_LAMBDA_REMOVE_FACE * losses["remove_face"]
            + USER_LAMBDA_REMOVE_NECK * losses["remove_neck"]
        )
        losses["loss"] = total_loss
        return total_loss, losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "satd_state_dict": self.satd.state_dict(),
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

        torch.save({"satd_state_dict": self.satd.state_dict()}, self.output_ckpt_dir / "satd_for_infer.pth")

    def save_preview(self, epoch: int, preview_rows: list[list[torch.Tensor]]):
        if epoch % USER_SAVE_VAL_IMAGES_EVERY != 0:
            return
        save_dir = self.output_val_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        to_pil = T.ToPILImage()
        for idx, row in enumerate(preview_rows[:USER_LOG_IMAGE_COUNT]):
            grid = torch.cat([((img[0] + 1) / 2).clamp(0, 1) for img in row], dim=2)
            to_pil(grid.cpu()).save(save_dir / f"sample_{idx:03d}.png")

    def run_epoch(self, loader, training: bool):
        self.satd.train(training)
        if not training:
            self.satd.eval()

        total = defaultdict(float)
        preview_rows = []
        collect_preview = (not training) and USER_LOG_IMAGE_COUNT > 0
        steps_in_epoch = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for step, batch in enumerate(tqdm(loader)):
                steps_in_epoch += 1
                need_transfer_images = (not self.cleanup_only) and (
                    USER_LAMBDA_TRANSFER_LPIPS > 0
                    or USER_LAMBDA_BANGS_DIRECTION > 0
                    or USER_LAMBDA_HAIR_HIGH_FREQ > 0
                    or USER_LAMBDA_HAIR_DIRECTION > 0
                    or USER_LAMBDA_NO_BANGS > 0
                )
                pred_images, transfer_images, author_images, latent_list, info_list = self.forward_batch(
                    batch,
                    render_transfer_images=need_transfer_images,
                    render_author_images=collect_preview,
                )
                loss, losses = self.calc_losses(pred_images, transfer_images, latent_list, info_list)

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

                if collect_preview and len(preview_rows) < USER_LOG_IMAGE_COUNT:
                    for idx, (info, _) in enumerate(info_list):
                        preview_rows.append(
                            [
                                info["source_image_256"],
                                info["shape_image_256"],
                                author_images[idx:idx + 1] if author_images is not None else pred_images[idx:idx + 1],
                                pred_images[idx:idx + 1],
                            ]
                        )

                del pred_images, transfer_images, author_images, latent_list, info_list, loss, losses
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
            load_compatible_state_dict(self.satd, ckpt["satd_state_dict"])
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, RuntimeError):
                print("Warning: optimizer state is incompatible with the current SATD_v4 architecture; reinitializing optimizer.")
            self.best_loss = ckpt.get("best_loss", self.best_loss)
            self.cur_iter = ckpt.get("cur_iter", 0)
            start_epoch = ckpt.get("epoch", -1) + 1

        for epoch in range(start_epoch, USER_EPOCHS):
            train_losses, _ = self.run_epoch(self.train_loader, training=True)
            val_losses, preview_rows = self.run_epoch(self.val_loader, training=False)

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
    pairs = load_pairs(ACTIVE_DATASET_DIR)
    if len(pairs) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(
            f"dataset.exps is smaller than the validation split size ({ACTIVE_VAL_SIZE}) "
            f"for profile {USER_DATASET_PROFILE!r}."
        )

    train_pairs, val_pairs = train_test_split(pairs, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = SATDPairDataset_v4(train_pairs, ACTIVE_FACE_IMAGE_ROOT, ACTIVE_SHAPE_IMAGE_ROOT)
    val_dataset = SATDPairDataset_v4(val_pairs, ACTIVE_FACE_IMAGE_ROOT, ACTIVE_SHAPE_IMAGE_ROOT)

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

    trainer = TrainerSATD_v4(train_loader, val_loader)
    trainer.train_loop()
    trainer.logger.wandb.finish()


if __name__ == "__main__":
    main()
