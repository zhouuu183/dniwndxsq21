from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

from models.Blending_v10 import Blending_v10
from models.Net import Net
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.train import WandbLogger, seed_everything, toggle_grad


# ========================= 用户配置区域：只改这里 =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("images/blending_dataset_v10_3000")
USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/blending_train_v10_3000")
USER_VAL_SIZE_FFHQ = 512

USER_DATASET_DIR_SMALL = Path("images/blending_dataset_v11_noF64")
USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ")
USER_OUTPUT_DIR_SMALL = Path("output/blending_train_v10_noF64_disabled")
USER_VAL_SIZE_SMALL = 30

USER_RANDOM_SEED = 3407
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"
USER_V10_INIT_CKPT = ""

USER_BATCH_SIZE = 4
USER_EFFECTIVE_BATCH_SIZE = 16
USER_CPU_THREADS = 1
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_DISABLE_CUDNN_BENCHMARK = True
USER_VALIDATE_DATASET_FILES = True

USER_EPOCHS = 30
USER_LR = 1e-4
USER_WEIGHT_DECAY = 1e-6
USER_GRAD_CLIP = 1.0

USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_VAL_IMAGES_EVERY = 1
USER_LOG_IMAGE_COUNT = 30

USER_USE_WANDB = False
USER_WANDB_RUN_NAME = "blending_train_v10"
USER_WANDB_PROJECT = "HairFast-Blending-v10"

USER_ALPHA_BOUNDARY_WIDTH_V10 = 9
USER_ALPHA_BOUNDARY_STRENGTH_V10 = 0.65
USER_ALPHA_FALLBACK_BLUR_V10 = 9
USER_ENABLE_F64_TRAINING = False
USER_F64_BYPASS_STRENGTH_V10 = 1.0
USER_F64_HIDDEN_CHANNELS_V10 = 512

USER_LAMBDA_FACE_CLIP = 1.0
USER_LAMBDA_HAIR_CLIP = 1.0
USER_LAMBDA_PRESERVE = 0.80
USER_LAMBDA_BOUNDARY_ALPHA = 0.35
USER_LAMBDA_TEXTURE_GRAD = 0.25
USER_LAMBDA_TEXTURE_GRAM = 0.08
USER_LAMBDA_HAIR_COLOR = 0.20
USER_LAMBDA_DELTA_REG = 0.03
USER_LAMBDA_OUTSIDE_REG = 0.05

USER_RESUME_CHECKPOINT = ""
# ========================================================================


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


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_DATASET_DIR = _DATASET_CFG["dataset_dir"]
ACTIVE_FACE_ROOT = _DATASET_CFG["face_root"]
ACTIVE_SHAPE_ROOT = _DATASET_CFG["shape_root"]
ACTIVE_COLOR_ROOT = _DATASET_CFG["color_root"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_VAL_SIZE = _DATASET_CFG["val_size"]


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

if USER_DISABLE_CUDNN_BENCHMARK and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False

torch.set_num_threads(max(1, USER_CPU_THREADS))

if USER_EFFECTIVE_BATCH_SIZE < USER_BATCH_SIZE or USER_EFFECTIVE_BATCH_SIZE % USER_BATCH_SIZE != 0:
    raise RuntimeError("USER_EFFECTIVE_BATCH_SIZE must be >= USER_BATCH_SIZE and divisible by it.")


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_image_path(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find image for {stem} under {root}")


def to_tensor_256(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = T.functional.resize(
            image,
            [256, 256],
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )
        return T.functional.normalize(T.functional.to_tensor(image), [0.5], [0.5])


def load_npz_tensor(path: Path, key: str) -> torch.Tensor:
    with np.load(path) as data:
        return torch.from_numpy(data[key]).float().squeeze(0)


def ensure_bchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(1) if tensor.shape[0] == 1 else tensor.unsqueeze(0)
    return tensor


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


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target).abs() * mask).sum() / denom


def local_gram_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    edge_pred = sobel_edges(pred)
    edge_target = sobel_edges(target)
    pred_features = torch.cat([pred, edge_pred], dim=1)
    target_features = torch.cat([target, edge_target], dim=1)
    if mask.shape[1] == 1:
        feature_mask = mask.expand(-1, pred_features.shape[1], -1, -1)
    else:
        feature_mask = mask
    pred_features = pred_features * feature_mask
    target_features = target_features * feature_mask
    bsz, channels, height, width = pred_features.shape
    pred_flat = pred_features.view(bsz, channels, height * width)
    target_flat = target_features.view(bsz, channels, height * width)
    denom = feature_mask[:, :1].sum(dim=(-2, -1), keepdim=True).clamp(min=1.0) * channels
    gram_pred = torch.bmm(pred_flat, pred_flat.transpose(1, 2)) / denom.view(bsz, 1, 1)
    gram_target = torch.bmm(target_flat, target_flat.transpose(1, 2)) / denom.view(bsz, 1, 1)
    return (gram_pred - gram_target.detach()).pow(2).mean()


def load_exps(dataset_dir: Path) -> list[list[str]]:
    exps = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as file:
        for line in file:
            items = line.strip().split()
            if len(items) == 3:
                exps.append(items)
    return exps


class BlendingDatasetV10(Dataset):
    def __init__(self, exps: list[list[str]], dataset_dir: Path, face_root: Path, shape_root: Path, color_root: Path):
        self.dataset_dir = dataset_dir
        self.face_root = face_root
        self.shape_root = shape_root
        self.color_root = color_root
        self.entries: list[tuple[str, str, str]] = []

        iterator = tqdm(exps, desc="index v10 dataset") if USER_VALIDATE_DATASET_FILES else exps
        for face_stem, shape_stem, color_stem in iterator:
            entry = (face_stem, shape_stem, color_stem)
            if USER_VALIDATE_DATASET_FILES and not self._has_required_files(*entry):
                continue
            self.entries.append(entry)

        print(
            f"dataset index: {len(self.entries)}/{len(exps)} "
            f"(lazy loading tensors/images per batch)",
            file=sys.stderr,
        )

    def _align_path(self, kind: str, face_stem: str, other_stem: str) -> Path:
        return self.dataset_dir / kind / f"face_{face_stem}__{kind.lower()}_{other_stem}.npz"

    def _has_required_files(self, face_stem: str, shape_stem: str, color_stem: str) -> bool:
        required = [
            self.dataset_dir / "FS" / f"face_{face_stem}.npz",
            self.dataset_dir / "FS" / f"color_{color_stem}.npz",
            self._align_path("AlignShape", face_stem, shape_stem),
            self._align_path("AlignColor", face_stem, color_stem),
        ]
        missing = [str(path) for path in required if not path.exists()]
        try:
            resolve_image_path(self.face_root, face_stem)
            resolve_image_path(self.shape_root, shape_stem)
            resolve_image_path(self.color_root, color_stem)
        except FileNotFoundError as exc:
            missing.append(str(exc))

        if missing:
            print(
                f"[skip] {face_stem} {shape_stem} {color_stem}: missing {missing[0]}",
                file=sys.stderr,
            )
            return False
        return True

    def _prepare_item(self, face_stem: str, shape_stem: str, color_stem: str):
        try:
            face_fs = self.dataset_dir / "FS" / f"face_{face_stem}.npz"
            color_fs = self.dataset_dir / "FS" / f"color_{color_stem}.npz"
            shape_align = self._align_path("AlignShape", face_stem, shape_stem)
            color_align = self._align_path("AlignColor", face_stem, color_stem)

            with np.load(face_fs) as data:
                face_s = torch.from_numpy(data["latent_S"]).float().squeeze(0)
            with np.load(color_fs) as data:
                color_s = torch.from_numpy(data["latent_S"]).float().squeeze(0)
                color_f64 = torch.from_numpy(data["latent_F64"]).float().squeeze(0)

            with np.load(shape_align) as data:
                align_f = torch.from_numpy(data["latent_F"]).float().squeeze(0)
                shape_target_hair_mask = torch.from_numpy(data["target_hair_mask"]).float().squeeze(0)
                shape_boundary_mask = torch.from_numpy(data["boundary_mask_256"]).float().squeeze(0)

            with np.load(color_align) as data:
                color_target_hair_mask = torch.from_numpy(data["target_hair_mask"]).float().squeeze(0)
                source_hair_mask = torch.from_numpy(data["source_hair_mask"]).float().squeeze(0)
                donor_hair_mask = torch.from_numpy(data["donor_hair_mask"]).float().squeeze(0)
                color_boundary_mask = torch.from_numpy(data["boundary_mask_256"]).float().squeeze(0)
                color_boundary_alpha = torch.from_numpy(data["boundary_alpha_256"]).float().squeeze(0)
                if "latent_F64_hair" in data:
                    color_f64 = torch.from_numpy(data["latent_F64_hair"]).float().squeeze(0)

            return {
                "face_stem": face_stem,
                "shape_stem": shape_stem,
                "color_stem": color_stem,
                "face_s": face_s,
                "color_s": color_s,
                "align_f": align_f,
                "color_f64": color_f64,
                "source_hair_mask": source_hair_mask,
                "donor_hair_mask": donor_hair_mask,
                "shape_target_hair_mask": shape_target_hair_mask,
                "color_target_hair_mask": color_target_hair_mask,
                "shape_boundary_mask": shape_boundary_mask,
                "color_boundary_mask": color_boundary_mask,
                "color_boundary_alpha": color_boundary_alpha,
                "face_i": to_tensor_256(resolve_image_path(self.face_root, face_stem)),
                "shape_i": to_tensor_256(resolve_image_path(self.shape_root, shape_stem)),
                "color_i": to_tensor_256(resolve_image_path(self.color_root, color_stem)),
            }
        except Exception as exc:
            raise RuntimeError(f"Failed to load v10 sample: {face_stem} {shape_stem} {color_stem}") from exc

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        face_stem, shape_stem, color_stem = self.entries[idx]
        return self._prepare_item(face_stem, shape_stem, color_stem)


class TrainerV10:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader):
        self.device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = ACTIVE_OUTPUT_DIR
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.preview_dir = self.output_dir / "val_images"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger()
        self.logger.start_logging()

        opts = Namespace(
            size=1024,
            ckpt=USER_STYLEGAN_CKPT,
            channel_multiplier=2,
            latent=512,
            n_mlp=8,
            device=str(self.device),
            smooth=5,
            blending_checkpoint=USER_BLENDING_CKPT,
            pp_checkpoint=USER_PP_CKPT,
            blending_v10_checkpoint=USER_V10_INIT_CKPT,
            save_all=False,
            save_all_dir=Path("output"),
            alpha_boundary_width_v10=USER_ALPHA_BOUNDARY_WIDTH_V10,
            alpha_boundary_strength_v10=USER_ALPHA_BOUNDARY_STRENGTH_V10,
            alpha_fallback_blur_v10=USER_ALPHA_FALLBACK_BLUR_V10,
            use_f64_bypass_v10=USER_ENABLE_F64_TRAINING,
            f64_bypass_strength_v10=USER_F64_BYPASS_STRENGTH_V10,
            f64_hidden_channels_v10=USER_F64_HIDDEN_CHANNELS_V10,
        )
        self.net = Net(opts)
        self.model = Blending_v10(opts, net=self.net).to(self.device)

        toggle_grad(self.net.generator, False)
        toggle_grad(self.model.blending_encoder, False)
        toggle_grad(self.model.post_process, False)
        toggle_grad(self.model.f64_bypass, True)
        self.net.generator.eval()
        self.model.blending_encoder.eval()
        self.model.post_process.eval()
        self.model.f64_bypass.train()

        self.optimizer = torch.optim.Adam(
            self.model.f64_bypass.parameters(),
            lr=USER_LR,
            weight_decay=USER_WEIGHT_DECAY,
        )
        self.grad_accum_steps = max(1, USER_EFFECTIVE_BATCH_SIZE // USER_BATCH_SIZE)
        self.dilate_erosion = DilateErosion(device=str(self.device))
        self.downsample_256 = BicubicDownSample(factor=4, cuda="cuda" in str(self.device))
        self.best_loss = float("inf")
        self.cur_iter = 0

    def _build_logger(self):
        if USER_USE_WANDB:
            return WandbLogger(name=USER_WANDB_RUN_NAME, project=USER_WANDB_PROJECT)
        return NullLogger()

    @staticmethod
    def _to_device(batch: dict) -> dict:
        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                out[key] = value
            else:
                out[key] = value
        return out

    def clip_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred_embed = self.model.blending_encoder.get_image_embed(pred * mask)
        with torch.no_grad():
            target_embed = self.model.blending_encoder.get_image_embed(target * mask)
        return (1 - F.cosine_similarity(pred_embed.float(), target_embed.float())).mean()

    def forward_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        face_i = batch["face_i"].to(self.device)
        shape_i = batch["shape_i"].to(self.device)
        color_i = batch["color_i"].to(self.device)
        face_s = batch["face_s"].to(self.device)
        color_s = batch["color_s"].to(self.device)
        align_f = batch["align_f"].to(self.device)
        color_f64 = batch["color_f64"].to(self.device)

        source_hair_mask = batch["source_hair_mask"].to(self.device)
        donor_hair_mask = batch["donor_hair_mask"].to(self.device)
        shape_target_hair_mask = batch["shape_target_hair_mask"].to(self.device)
        color_target_hair_mask = batch["color_target_hair_mask"].to(self.device)
        shape_boundary_mask = batch["shape_boundary_mask"].to(self.device)
        color_boundary_mask = batch["color_boundary_mask"].to(self.device)
        color_boundary_alpha = batch["color_boundary_alpha"].to(self.device)

        if source_hair_mask.dim() == 3:
            source_hair_mask = source_hair_mask.unsqueeze(1)
            donor_hair_mask = donor_hair_mask.unsqueeze(1)
            shape_target_hair_mask = shape_target_hair_mask.unsqueeze(1)
            color_target_hair_mask = color_target_hair_mask.unsqueeze(1)
            shape_boundary_mask = shape_boundary_mask.unsqueeze(1)
            color_boundary_mask = color_boundary_mask.unsqueeze(1)
            color_boundary_alpha = color_boundary_alpha.unsqueeze(1)

        with torch.no_grad():
            mask_de = self.dilate_erosion.mask(torch.cat([source_hair_mask, donor_hair_mask], dim=0))
            batch_size = face_i.shape[0]
            hm_1d = mask_de[0][:batch_size]
            hm_3d = mask_de[0][batch_size:]
            hm_3e = mask_de[1][batch_size:]
            hm_xd, _ = self.dilate_erosion.mask(color_target_hair_mask)
            target_mask = ((1 - hm_1d) * (1 - hm_3d) * (1 - hm_xd)).clamp(0.0, 1.0)

            blend_s_6_18 = self.model.blending_encoder(
                face_s[:, 6:],
                color_s[:, 6:],
                face_i * target_mask,
                color_i * hm_3e,
            )
            blend_s = torch.cat((face_s[:, :6], blend_s_6_18), dim=1)
            i_blend, _ = self.net.generator(
                [blend_s],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=align_f,
            )
            i_blend_256 = self.downsample_256(i_blend)
            s_final, f_final = self.model.post_process(face_i, i_blend_256)
            baseline_image, _ = self.net.generator(
                [s_final],
                input_is_latent=True,
                return_latents=False,
                start_layer=5,
                end_layer=8,
                layer_in=f_final,
            )
            baseline_256 = self.downsample_256(baseline_image)

        bypass_outputs = self.model.f64_bypass(
            current_f64=f_final.detach(),
            hair_f64=color_f64.detach(),
            hair_mask_256=color_target_hair_mask,
            boundary_mask_256=color_boundary_mask,
            boundary_alpha_256=color_boundary_alpha,
            strength=USER_F64_BYPASS_STRENGTH_V10,
        )
        i_final, _ = self.net.generator(
            [s_final.detach()],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=bypass_outputs["fused_f64"],
        )
        i_final_256 = self.downsample_256(i_final)

        texture_mask = (
            color_target_hair_mask + shape_target_hair_mask + color_boundary_mask + shape_boundary_mask
        ).clamp(0.0, 1.0)
        reference_i = (0.5 * shape_i + 0.5 * color_i).clamp(-1.0, 1.0)
        boundary_target = (
            color_boundary_alpha * color_i + (1.0 - color_boundary_alpha) * face_i
        ).clamp(-1.0, 1.0)

        return {
            "i_final": i_final,
            "i_final_256": i_final_256,
            "baseline_256": baseline_256,
            "face_i": face_i,
            "shape_i": shape_i,
            "color_i": color_i,
            "reference_i": reference_i,
            "boundary_target": boundary_target,
            "target_mask": target_mask,
            "hm_3e": hm_3e,
            "texture_mask": texture_mask,
            "color_boundary_mask": color_boundary_mask,
            "color_target_hair_mask": color_target_hair_mask,
            "f64_current": f_final.detach(),
            **bypass_outputs,
        }

    def calc_loss(self, outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        i_final_256 = outputs["i_final_256"]
        face_i = outputs["face_i"]
        color_i = outputs["color_i"]
        reference_i = outputs["reference_i"]
        baseline_256 = outputs["baseline_256"]
        target_mask = outputs["target_mask"]
        hm_3e = outputs["hm_3e"]
        texture_mask = outputs["texture_mask"]
        boundary_mask = outputs["color_boundary_mask"]
        boundary_target = outputs["boundary_target"]
        color_target_hair_mask = outputs["color_target_hair_mask"]

        face_clip = self.clip_loss(i_final_256, face_i, target_mask)
        hair_clip = self.clip_loss(i_final_256, color_i, hm_3e)
        preserve = masked_l1(i_final_256, baseline_256.detach(), (1 - texture_mask).clamp(0.0, 1.0))
        boundary_alpha = masked_l1(i_final_256, boundary_target, boundary_mask)
        texture_grad = masked_l1(sobel_edges(i_final_256), sobel_edges(reference_i), texture_mask)
        texture_gram = local_gram_loss(i_final_256, reference_i, texture_mask)
        hair_color = masked_l1(i_final_256, color_i, color_target_hair_mask)
        delta_reg = outputs["f64_delta"].abs().mean()
        outside_reg = (
            (outputs["fused_f64"] - outputs["f64_current"]).abs() * (1 - outputs["f64_inject_mask"])
        ).mean()

        losses = {
            "face_clip": USER_LAMBDA_FACE_CLIP * face_clip,
            "hair_clip": USER_LAMBDA_HAIR_CLIP * hair_clip,
            "preserve": USER_LAMBDA_PRESERVE * preserve,
            "boundary_alpha": USER_LAMBDA_BOUNDARY_ALPHA * boundary_alpha,
            "texture_grad": USER_LAMBDA_TEXTURE_GRAD * texture_grad,
            "texture_gram": USER_LAMBDA_TEXTURE_GRAM * texture_gram,
            "hair_color": USER_LAMBDA_HAIR_COLOR * hair_color,
            "delta_reg": USER_LAMBDA_DELTA_REG * delta_reg,
            "outside_reg": USER_LAMBDA_OUTSIDE_REG * outside_reg,
        }
        losses["loss"] = sum(losses.values())
        return losses["loss"], losses

    def save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            "epoch": epoch,
            "f64_bypass_state_dict": self.model.f64_bypass.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "cur_iter": self.cur_iter,
        }
        torch.save(state, self.ckpt_dir / "last.pth")
        if epoch % USER_SAVE_CHECKPOINT_EVERY == 0:
            torch.save(state, self.ckpt_dir / "checkpoint.pth")
            torch.save(state, self.ckpt_dir / f"epoch_{epoch:03d}.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")
        torch.save({"f64_bypass_state_dict": self.model.f64_bypass.state_dict()}, self.ckpt_dir / "blending_v10_for_infer.pth")

    @staticmethod
    def _to_pil(image: torch.Tensor) -> Image.Image:
        image = ((image + 1) / 2).clamp(0, 1)
        return T.functional.to_pil_image(image.detach().cpu())

    def save_preview(self, epoch: int, preview_rows: list[dict]):
        if epoch % USER_SAVE_VAL_IMAGES_EVERY != 0:
            return
        save_dir = self.preview_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(preview_rows[:USER_LOG_IMAGE_COUNT]):
            panels = [
                self._to_pil(row["face"]),
                self._to_pil(row["shape"]),
                self._to_pil(row["color"]),
                self._to_pil(row["baseline"]),
                self._to_pil(row["v10"]),
            ]
            width, height = panels[0].size
            canvas = Image.new("RGB", (width * len(panels), height), color=(255, 255, 255))
            for panel_idx, panel in enumerate(panels):
                canvas.paste(panel, (panel_idx * width, 0))
            canvas.save(save_dir / f"sample_{idx:03d}.png")

    def run_epoch(self, training: bool, epoch: int) -> tuple[dict[str, float], list[dict]]:
        self.model.f64_bypass.train(training)
        loader = self.train_loader if training else self.val_loader
        totals = defaultdict(float)
        preview_rows = []
        steps = 0

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for step, batch in enumerate(tqdm(loader, desc="train" if training else "val")):
                outputs = self.forward_batch(batch)
                loss, losses = self.calc_loss(outputs)
                steps += 1

                if training:
                    (loss / self.grad_accum_steps).backward()
                    if (step + 1) % self.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.f64_bypass.parameters(), USER_GRAD_CLIP)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.cur_iter += 1
                    self.logger.next_step()

                for key, value in losses.items():
                    totals[key] += float(value.detach().cpu())

                if not training and len(preview_rows) < USER_LOG_IMAGE_COUNT:
                    for idx in range(outputs["i_final_256"].shape[0]):
                        if len(preview_rows) >= USER_LOG_IMAGE_COUNT:
                            break
                        preview_rows.append(
                            {
                                "face": outputs["face_i"][idx],
                                "shape": outputs["shape_i"][idx],
                                "color": outputs["color_i"][idx],
                                "baseline": outputs["baseline_256"][idx],
                                "v10": outputs["i_final_256"][idx],
                            }
                        )

                del outputs, loss, losses

        if training and steps % self.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.f64_bypass.parameters(), USER_GRAD_CLIP)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        for key in list(totals.keys()):
            totals[key] /= max(steps, 1)
        return dict(totals), preview_rows

    def train_loop(self):
        start_epoch = 0
        if USER_RESUME_CHECKPOINT:
            checkpoint = torch.load(USER_RESUME_CHECKPOINT, map_location=self.device)
            self.model.f64_bypass.load_state_dict(checkpoint["f64_bypass_state_dict"], strict=False)
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, RuntimeError):
                print("Warning: optimizer state is incompatible; reinitializing optimizer.")
            self.best_loss = checkpoint.get("best_loss", self.best_loss)
            self.cur_iter = checkpoint.get("cur_iter", 0)
            start_epoch = checkpoint.get("epoch", -1) + 1

        for epoch in range(start_epoch, USER_EPOCHS):
            train_losses, _ = self.run_epoch(training=True, epoch=epoch)
            val_losses, preview_rows = self.run_epoch(training=False, epoch=epoch)
            self.save_preview(epoch, preview_rows)

            for key, value in train_losses.items():
                self.logger.log(f"train/{key}", value)
            for key, value in val_losses.items():
                self.logger.log(f"val/{key}", value)

            is_best = val_losses["loss"] <= self.best_loss
            if is_best:
                self.best_loss = val_losses["loss"]
            self.save_checkpoint(epoch, is_best=is_best)

            print(
                f"[epoch {epoch:03d}] "
                f"train={train_losses['loss']:.6f} val={val_losses['loss']:.6f} best={self.best_loss:.6f}"
            )


def main():
    seed_everything(USER_RANDOM_SEED)
    set_seed(USER_RANDOM_SEED)
    if not USER_ENABLE_F64_TRAINING:
        raise RuntimeError(
            "F64 bypass training is disabled for the noF64 alpha path. "
            "Run scripts/blending_gen_v10.py to build images/blending_dataset_v11_noF64, "
            "then evaluate/infer with hair_swap_v10.py without --use_f64_bypass_v10. "
            "Set USER_ENABLE_F64_TRAINING=True only if you intentionally resume the old F64 experiment."
        )
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"dataset dir: {ACTIVE_DATASET_DIR}")
    print(f"face root: {ACTIVE_FACE_ROOT}")
    print(f"shape root: {ACTIVE_SHAPE_ROOT}")
    print(f"color root: {ACTIVE_COLOR_ROOT}")

    exps = load_exps(ACTIVE_DATASET_DIR)
    if len(exps) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(
            f"dataset.exps is smaller than the validation split size ({ACTIVE_VAL_SIZE}) "
            f"for profile {USER_DATASET_PROFILE!r}."
        )

    train_exps, val_exps = train_test_split(exps, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = BlendingDatasetV10(train_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_SHAPE_ROOT, ACTIVE_COLOR_ROOT)
    val_dataset = BlendingDatasetV10(val_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_SHAPE_ROOT, ACTIVE_COLOR_ROOT)

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

    trainer = TrainerV10(train_loader, val_loader)
    trainer.train_loop()
    trainer.logger.wandb.finish()


if __name__ == "__main__":
    main()
