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
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Alignment_v9 import Alignment_v9
from models.DGSTA_v9 import DGSTA_v9
from models.Embedding import Embedding
from models.Net import Net
from utils.bicubic import BicubicDownSample
from utils.image_utils import equal_replacer
from utils.train import WandbLogger, toggle_grad


# ========================= 用户配置区域：只改这里 =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("images/satd_dataset_v8_3000")
USER_FFHQ_ROOT = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/dgsta_train_v9_3000")

USER_SMALL_DATASET_DIR = Path("images/satd_dataset_v8_small8")
USER_SMALL_FACE_ROOT = Path("images/FFHQ_long")
USER_SMALL_SHAPE_ROOT = Path("images/FFHQ_short")
USER_SMALL_OUTPUT_DIR = Path("output/dgsta_train_v9_small")

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_DGSTA_INIT_CKPT = ""

USER_BATCH_SIZE = 4
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
USER_SMALL_VAL_SIZE = 30
USER_RANDOM_SEED = 3407

USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_VAL_IMAGES_EVERY = 1
USER_LOG_IMAGE_COUNT = 30

USER_USE_WANDB = False
USER_WANDB_RUN_NAME = "dgsta_train_v9"
USER_WANDB_PROJECT = "HairFast-DGSTA-v9"

USER_DGSTA_BLEND_V9 = 1.0
USER_DGSTA_BOUNDARY_V9 = 8
USER_EQ8_REFERENCE_BLEND_V9 = 0.0

USER_STAGE1_EPOCHS_V9 = 4
USER_STAGE2_EPOCHS_V9 = 8

USER_LAMBDA_PRESERVE_V9 = 0.90
USER_LAMBDA_VISIBLE_ANCHOR_V9 = 1.80
USER_LAMBDA_CLEAN_V9 = 1.45
USER_LAMBDA_CLEAN_GRAY_V9 = 1.20
USER_LAMBDA_SHADOW_IMPROVE_V9 = 1.10
USER_LAMBDA_NON_DARK_V9 = 1.20
USER_LAMBDA_TOPOLOGY_V9 = 1.20
USER_LAMBDA_TOPOLOGY_IMPROVE_V9 = 0.85
USER_LAMBDA_BANG_V9 = 1.15
USER_LAMBDA_DETAIL_KEEP_V9 = 1.45
USER_LAMBDA_LATENT_KEEP_V9 = 0.03

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
        raise RuntimeError(f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}.")
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


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    g = gray(image)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
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


class DGSTAPairDataset_v9(torch.utils.data.Dataset):
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


class TrainerDGSTA_v9:
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
            use_dgsta_v9=False,
            dgsta_checkpoint_v9="",
            dgsta_blend_v9=USER_DGSTA_BLEND_V9,
            satd_boundary_v8=USER_DGSTA_BOUNDARY_V9,
            eq8_reference_blend_v8=USER_EQ8_REFERENCE_BLEND_V9,
        )

        self.net = Net(self.opts)
        self.embed = Embedding(self.opts, net=self.net).eval()
        self.align = Alignment_v9(self.opts, latent_encoder=self.embed.get_e4e_embed, net=self.net, dgsta_model_v9=None)
        self.dgsta = DGSTA_v9().to(self.device)
        if USER_DGSTA_INIT_CKPT:
            ckpt = torch.load(USER_DGSTA_INIT_CKPT, map_location=self.device)
            load_compatible_state_dict(self.dgsta, ckpt.get("dgsta_v9_state_dict", ckpt.get("model_state_dict", ckpt)))

        self.downsample_256 = BicubicDownSample(factor=4)
        toggle_grad(self.net.generator, False)
        toggle_grad(self.dgsta, True)

        self.optimizer = torch.optim.Adam(self.dgsta.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
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
        if epoch < USER_STAGE1_EPOCHS_V9:
            return {"repair": 1.00, "topology": 0.45, "bang": 0.55, "improve": 0.35, "blend": 0.90}
        if epoch < USER_STAGE2_EPOCHS_V9:
            return {"repair": 1.00, "topology": 0.85, "bang": 0.95, "improve": 0.70, "blend": 1.00}
        return {"repair": 1.00, "topology": 1.00, "bang": 1.00, "improve": 0.90, "blend": USER_DGSTA_BLEND_V9}

    def build_name_to_embed(self, batch):
        images_to_name = defaultdict(list)
        sample_names = []
        bsz = batch["face"].shape[0]
        for idx in range(bsz):
            face, shape = equal_replacer([batch["face"][idx], batch["shape"][idx]])
            names = {"face": f"face_{idx}", "shape": f"shape_{idx}"}
            images_to_name[face].append(names["face"])
            images_to_name[shape].append(names["shape"])
            sample_names.append(names)
        with torch.no_grad():
            name_to_embed = self.embed.embedding_images(images_to_name)
        return name_to_embed, sample_names

    @staticmethod
    def _slice_batch(batch, start: int, end: int):
        return {key: value[start:end] for key, value in batch.items()}

    def forward_batch(self, batch, blend_strength: float):
        pred_images = []
        author_images = []
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
                    info = self.align.prepare_dgsta_features(
                        names["face"],
                        names["shape"],
                        name_to_embed,
                        satd_boundary_v8=USER_DGSTA_BOUNDARY_V9,
                        eq8_reference_blend_v8=USER_EQ8_REFERENCE_BLEND_V9,
                    )
                    author_align = self.align.author_align_images(names["face"], names["shape"], name_to_embed)
                info = materialize_tensor_tree(info)
                info["latent_F_author"] = author_align["latent_F_align"].clone()

                with torch.no_grad():
                    author_image, _ = self.net.generator(
                        [info["latent_S_src"]],
                        input_is_latent=True,
                        return_latents=False,
                        start_layer=4,
                        end_layer=8,
                        layer_in=info["latent_F_author"],
                    )
                author_images.append(self.downsample_256(author_image))

                dgsta_out, dgsta_aux = self.dgsta(
                    F_base=info["latent_F_author"],
                    F_src=info["latent_F_src"],
                    F_ref=info["latent_F_ref"],
                    F_src_inpaint=info["latent_F_src_inpaint"],
                    F_shape_inpaint=info["latent_F_shape_inpaint"],
                    dgsta_masks_256=info["dgsta_masks_256"],
                    source_rgb_256=info["source_image_256"],
                    shape_rgb_256=info["shape_image_256"],
                )
                latent_F = info["latent_F_author"] + blend_strength * (dgsta_out - info["latent_F_author"])

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
                info_list.append((info, dgsta_aux))

            del name_to_embed, sample_names, batch_chunk
            gc.collect()

        return torch.cat(pred_images, dim=0), torch.cat(author_images, dim=0), latent_list, info_list

    def calc_losses(self, pred_images, author_images, latent_list, info_list, schedule: dict[str, float]):
        losses = defaultdict(float)
        for idx, (latent_F, (info, dgsta_aux)) in enumerate(zip(latent_list, info_list)):
            pred = pred_images[idx:idx + 1]
            author = author_images[idx:idx + 1]
            source = info["source_image_256"]
            shape = info["shape_image_256"]
            source_inpaint = info["source_inpaint_256"]
            shape_inpaint = info["shape_inpaint_256"]
            delta_masks = info["delta_masks"]
            remove = delta_masks["M_remove"]
            add = delta_masks["M_add"]
            keep = delta_masks["M_keep"]
            boundary = delta_masks["M_boundary"]
            remove_halo = delta_masks.get("M_remove_halo", torch.zeros_like(remove))
            remove_face = delta_masks.get("M_remove_face", torch.zeros_like(remove))
            remove_neck = delta_masks.get("M_remove_neck", torch.zeros_like(remove))
            remove_tail = delta_masks.get("M_remove_tail", torch.zeros_like(remove))
            visible_body_anchor = delta_masks.get("M_visible_body_anchor", torch.zeros_like(remove))
            body_reveal_only = delta_masks.get("M_body_reveal_only", torch.zeros_like(remove))
            context_reveal_only = delta_masks.get("M_context_reveal_only", torch.zeros_like(remove))
            reveal_overlap = delta_masks.get("M_reveal_overlap", torch.zeros_like(remove))
            detail_protect = delta_masks.get("M_detail_protect", torch.zeros_like(remove))
            bang = delta_masks.get("M_bang", torch.zeros_like(remove))
            bundle = delta_masks.get("M_bundle", torch.zeros_like(remove))

            source_detail = normalize_map(gaussian_blur(sobel_edges(source), kernel_size=5, sigma=1.2))
            shape_detail = normalize_map(gaussian_blur(sobel_edges(shape), kernel_size=5, sigma=1.2))
            detail_energy = torch.maximum(source_detail, shape_detail)

            clean_region = (
                remove
                + 0.90 * remove_halo
                + 0.95 * remove_face
                + 0.92 * remove_neck
                + 0.88 * remove_tail
                + 0.70 * body_reveal_only
                + 0.70 * context_reveal_only
                + 0.55 * reveal_overlap
            ).clamp(0, 1)
            topology_region = (add + 0.45 * keep + 0.35 * boundary + 0.70 * bundle).clamp(0, 1)
            bang_region = (bang * (add + keep + 0.30 * boundary).clamp(0, 1)).clamp(0, 1)
            protected_detail = (detail_protect + 0.35 * detail_energy * (1.0 - add).clamp(0, 1)).clamp(0, 1)
            edit_region = (clean_region + topology_region + bang_region).clamp(0, 1)
            preserve_region = (1.0 - edit_region).clamp(0, 1)
            keep32 = F.interpolate(preserve_region, size=(32, 32), mode="nearest")

            clean_target = (0.88 * source_inpaint + 0.12 * source).clamp(-1, 1)
            topology_target = (0.72 * shape_inpaint + 0.28 * shape).clamp(-1, 1)
            bang_target = (0.82 * shape_inpaint + 0.18 * shape).clamp(-1, 1)
            shadow_target = torch.maximum(gray_low(source_inpaint), gray_low(author))
            non_dark_baseline = torch.maximum(gray(clean_target), gray(author))

            losses["preserve"] += masked_l1(pred, author, preserve_region)
            losses["visible_anchor"] += masked_l1(pred, source, visible_body_anchor)
            losses["clean"] += masked_l1(pred, clean_target, clean_region)
            losses["clean_gray"] += masked_l1(gray_low(pred), shadow_target, clean_region)
            losses["shadow_improve"] += masked_relative_improvement(gray_low(pred), shadow_target, gray_low(author), clean_region)
            losses["non_dark"] += masked_non_darker(gray(pred), non_dark_baseline, clean_region)
            losses["topology"] += masked_l1(pred, topology_target, topology_region)
            losses["topology_improve"] += masked_relative_improvement(pred, topology_target, author, topology_region)
            losses["bang"] += masked_l1(pred, bang_target, bang_region)
            losses["detail_keep"] += masked_l1(pred, source, (protected_detail + visible_body_anchor).clamp(0, 1))
            losses["latent_keep"] += masked_l1(latent_F, info["latent_F_author"], keep32)

        count = max(len(info_list), 1)
        for key in list(losses.keys()):
            losses[key] = losses[key] / count

        total_loss = (
            USER_LAMBDA_PRESERVE_V9 * losses["preserve"]
            + USER_LAMBDA_VISIBLE_ANCHOR_V9 * losses["visible_anchor"]
            + USER_LAMBDA_CLEAN_V9 * schedule["repair"] * losses["clean"]
            + USER_LAMBDA_CLEAN_GRAY_V9 * schedule["repair"] * losses["clean_gray"]
            + USER_LAMBDA_SHADOW_IMPROVE_V9 * schedule["improve"] * losses["shadow_improve"]
            + USER_LAMBDA_NON_DARK_V9 * losses["non_dark"]
            + USER_LAMBDA_TOPOLOGY_V9 * schedule["topology"] * losses["topology"]
            + USER_LAMBDA_TOPOLOGY_IMPROVE_V9 * schedule["improve"] * losses["topology_improve"]
            + USER_LAMBDA_BANG_V9 * schedule["bang"] * losses["bang"]
            + USER_LAMBDA_DETAIL_KEEP_V9 * losses["detail_keep"]
            + USER_LAMBDA_LATENT_KEEP_V9 * losses["latent_keep"]
        )
        losses["loss"] = total_loss
        return total_loss, losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "dgsta_v9_state_dict": self.dgsta.state_dict(),
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
        torch.save({"dgsta_v9_state_dict": self.dgsta.state_dict()}, self.output_ckpt_dir / "dgsta_for_infer_v9.pth")

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
        self.dgsta.train(training)
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
                pred_images, author_images, latent_list, info_list = self.forward_batch(batch, blend_strength=schedule["blend"])
                loss, losses = self.calc_losses(pred_images, author_images, latent_list, info_list, schedule)

                if training:
                    (loss / self.grad_accum_steps).backward()
                    if (step + 1) % self.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.dgsta.parameters(), 5.0)
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
                                author_images[idx:idx + 1],
                                pred_images[idx:idx + 1],
                            ]
                        )

                del pred_images, author_images, latent_list, info_list, loss, losses
                if (step + 1) % 20 == 0:
                    gc.collect()

        if training and steps_in_epoch % self.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.dgsta.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        for key in list(total.keys()):
            total[key] /= max(steps_in_epoch, 1)
        return total, preview_rows

    def train_loop(self):
        start_epoch = 0
        if USER_RESUME_CHECKPOINT:
            ckpt = torch.load(USER_RESUME_CHECKPOINT, map_location=self.device)
            load_compatible_state_dict(self.dgsta, ckpt["dgsta_v9_state_dict"])
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, RuntimeError):
                print("Warning: optimizer state is incompatible; reinitializing optimizer.")
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
    pairs = load_pairs(ACTIVE_DATASET_DIR)
    if len(pairs) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(
            f"dataset.exps is smaller than the validation split size ({ACTIVE_VAL_SIZE}) "
            f"for profile {USER_DATASET_PROFILE!r}."
        )

    train_pairs, val_pairs = train_test_split(pairs, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = DGSTAPairDataset_v9(train_pairs, ACTIVE_FACE_IMAGE_ROOT, ACTIVE_SHAPE_IMAGE_ROOT)
    val_dataset = DGSTAPairDataset_v9(val_pairs, ACTIVE_FACE_IMAGE_ROOT, ACTIVE_SHAPE_IMAGE_ROOT)

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

    trainer = TrainerDGSTA_v9(train_loader, val_loader)
    trainer.train_loop()
    trainer.logger.wandb.finish()


if __name__ == "__main__":
    main()
