from __future__ import annotations

import gc
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ========================= User Config: edit only here =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("images/shape_dataset_v19_3000")
USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/shape_train_v19_3000")
USER_VAL_SIZE_FFHQ = 256

USER_DATASET_DIR_SMALL = Path("images/shape_dataset_v19_small")
USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_OUTPUT_DIR_SMALL = Path("output/shape_train_v19_small")
USER_VAL_SIZE_SMALL = 30

USER_RANDOM_SEED = 3407
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_INIT_CKPT = ""
USER_RESUME_CHECKPOINT = ""
USER_ARCFACE_CKPT = "pretrained_models/ArcFace/backbone_ir50.pth"

USER_BATCH_SIZE = 1
USER_EFFECTIVE_BATCH_SIZE = 4
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_CPU_THREADS = 1
USER_DISABLE_CUDNN_BENCHMARK = True
USER_LOW_MEMORY_MODE = True
USER_USE_AMP = True
USER_EMPTY_CACHE_EVERY = 10
USER_BASELINE_RENDER_MODE = "detail"

USER_EPOCHS = 30
USER_LR = 1e-4
USER_WEIGHT_DECAY = 1e-6
USER_GRAD_CLIP = 1.0

USER_SHAPE_ADAPTER_HIDDEN_V19 = 256
USER_SHAPE_ADAPTER_STRENGTH_V19 = 1.0
USER_SHAPE_PRIOR_STRENGTH_V19 = 0.20
USER_SHAPE_DETAIL_STRENGTH_V19 = 1.0

USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_PREVIEW_EVERY = 1
USER_PREVIEW_COUNT = 8

USER_LAMBDA_MASK_COARSE = 0.20
USER_LAMBDA_MASK_TARGET = 0.60
USER_LAMBDA_MASK_RESIDUAL = 0.35
USER_LAMBDA_LATENT_SHAPE32 = 0.70
USER_LAMBDA_LATENT_PRESERVE32 = 0.90
USER_LAMBDA_LATENT_DETAIL64 = 0.55
USER_LAMBDA_LATENT_PRESERVE64 = 0.65
USER_LAMBDA_IMAGE_SHAPE = 0.45
USER_LAMBDA_IMAGE_PRESERVE = 0.70
USER_LAMBDA_BOUNDARY_SOBEL = 0.30
USER_LAMBDA_BOUNDARY_LAPLACIAN = 0.20
USER_LAMBDA_BOUNDARY_CHAMFER = 0.35
USER_LAMBDA_BOUNDARY_HAUSDORFF = 0.15
USER_LAMBDA_HIGH_FREQ = 0.35
USER_LAMBDA_LOCAL_FEATURE = 0.0
USER_LAMBDA_FACE_IDENTITY = 0.55
USER_LAMBDA_DELTA32 = 0.02
USER_LAMBDA_DELTA64 = 0.02
# ========================================================================


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Net import Net, iresnet50
from models.ShapeAdaptor_v19 import (
    BoundaryLayerShapeAdaptor_v19,
    V19_PRIOR_KEYS,
    ensure_bchw_mask_v19,
)
from utils.bicubic import BicubicDownSample
from utils.shape_metrics_v19 import (
    boundary_fscore,
    chamfer_distance,
    cosine_feature_similarity,
    hausdorff_distance,
    high_frequency_spectrum_similarity,
    iou_score,
    laplacian_edges,
    masked_l1,
    sobel_edges,
)
from utils.train import seed_everything, toggle_grad

TRAIN_PRIOR_KEYS = V19_PRIOR_KEYS + ("pseudo_target_mask", "pseudo_boundary_residual")


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "val_size": USER_VAL_SIZE_FFHQ,
        },
        "small": {
            "dataset_dir": USER_DATASET_DIR_SMALL,
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "val_size": USER_VAL_SIZE_SMALL,
        },
    }
    if USER_DATASET_PROFILE not in profiles:
        raise RuntimeError(f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}.")
    return profiles[USER_DATASET_PROFILE]


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_DATASET_DIR = _DATASET_CFG["dataset_dir"]
ACTIVE_FACE_ROOT = _DATASET_CFG["face_root"]
ACTIVE_SHAPE_ROOT = _DATASET_CFG["shape_root"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_VAL_SIZE = _DATASET_CFG["val_size"]

torch.set_num_threads(max(1, USER_CPU_THREADS))
if USER_DISABLE_CUDNN_BENCHMARK and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
if USER_EFFECTIVE_BATCH_SIZE < USER_BATCH_SIZE or USER_EFFECTIVE_BATCH_SIZE % USER_BATCH_SIZE != 0:
    raise RuntimeError("USER_EFFECTIVE_BATCH_SIZE must be >= USER_BATCH_SIZE and divisible by it.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_opts(device: torch.device) -> Namespace:
    return Namespace(
        size=1024,
        ckpt=USER_STYLEGAN_CKPT,
        channel_multiplier=2,
        latent=512,
        n_mlp=8,
        device=str(device),
        batch_size=1,
        save_all=False,
        save_all_dir=Path("output"),
        mixing=0.95,
        smooth=5,
        rotate_checkpoint=USER_ROTATE_CKPT,
    )


def load_exps(dataset_dir: Path) -> list[tuple[str, str, str]]:
    exps = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as file:
        for line in file:
            items = line.strip().split()
            if len(items) == 3:
                exps.append((items[0], items[1], items[2]))
    return exps


def resolve_image_path(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find image for {stem} under {root}")


def load_image_256(root: Path, stem: str) -> torch.Tensor:
    with Image.open(resolve_image_path(root, stem)) as image:
        image = image.convert("RGB").resize((256, 256), Image.BICUBIC)
        return T.functional.to_tensor(image) * 2.0 - 1.0


def ensure_chw_mask_train_v19(mask: torch.Tensor) -> torch.Tensor:
    mask = ensure_bchw_mask_v19(mask)
    if mask.dim() == 4 and mask.shape[0] == 1:
        return mask.squeeze(0)
    if mask.dim() == 3:
        return mask
    raise RuntimeError(f"Expected CHW-compatible mask, but got shape={tuple(mask.shape)}")


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def role_fs_path(role: str, stem: str) -> Path:
    return ACTIVE_DATASET_DIR / "FS" / f"{role}_{stem}.npz"


def align_shape_path(face_stem: str, shape_stem: str) -> Path:
    return ACTIVE_DATASET_DIR / "AlignShape" / f"face_{face_stem}__alignshape_{shape_stem}.npz"


class ShapeDatasetV19(Dataset):
    def __init__(self, exps: list[tuple[str, str, str]], dataset_dir: Path, face_root: Path, shape_root: Path):
        self.exps = exps
        self.dataset_dir = dataset_dir
        self.face_root = face_root
        self.shape_root = shape_root

    def __len__(self):
        return len(self.exps)

    def __getitem__(self, idx):
        face_stem, shape_stem, color_stem = self.exps[idx]
        face_fs = role_fs_path("face", face_stem)
        align_path = align_shape_path(face_stem, shape_stem)
        with np.load(face_fs) as face_data:
            face_s = torch.from_numpy(face_data["latent_S"]).float().squeeze(0)
            face_f32 = torch.from_numpy(face_data["latent_F32"]).float().squeeze(0)

        with np.load(align_path) as align_data:
            def read_align(*keys: str) -> torch.Tensor:
                for key in keys:
                    if key in align_data:
                        return torch.from_numpy(align_data[key]).float().squeeze(0)
                raise KeyError(f"{align_path} does not contain any of keys={keys!r}")

            priors = {
                key: ensure_chw_mask_train_v19(torch.from_numpy(align_data[key]).float().squeeze(0))
                for key in TRAIN_PRIOR_KEYS
            }
            f_base = read_align("latent_F_base")
            f_shape = read_align("latent_F_shape")
            f_reference = read_align("latent_F_reference", "latent_F_shape")
            f_base64 = read_align("latent_F_base64")
            f_shape64 = read_align("latent_F_shape64")
            f_reference64 = read_align("latent_F_reference64", "latent_F_shape64", "latent_F64_detail_target")
            reference_i = read_align("reference_image_256") if "reference_image_256" in align_data else None
            aligned_reference_i = (
                read_align("aligned_reference_image_256", "reference_image_256")
                if "reference_image_256" in align_data or "aligned_reference_image_256" in align_data
                else None
            )

        return {
            "face_stem": face_stem,
            "shape_stem": shape_stem,
            "color_stem": color_stem,
            "face_s": face_s,
            "F_src": face_f32,
            "F_base": f_base,
            "F_shape": f_shape,
            "F_reference": f_reference,
            "F_base64": f_base64,
            "F_shape64": f_shape64,
            "F_reference64": f_reference64,
            "face_i": load_image_256(self.face_root, face_stem),
            "reference_i": reference_i if reference_i is not None else load_image_256(self.shape_root, shape_stem),
            "aligned_reference_i": aligned_reference_i if aligned_reference_i is not None else load_image_256(self.shape_root, shape_stem),
            "priors": priors,
        }


def collate_shape_v19(batch):
    priors = {
        key: torch.stack([ensure_chw_mask_train_v19(item["priors"][key]) for item in batch], dim=0)
        for key in TRAIN_PRIOR_KEYS
    }
    return {
        "face_stem": [item["face_stem"] for item in batch],
        "shape_stem": [item["shape_stem"] for item in batch],
        "color_stem": [item["color_stem"] for item in batch],
        "face_s": torch.stack([item["face_s"] for item in batch], dim=0),
        "F_src": torch.stack([item["F_src"] for item in batch], dim=0),
        "F_base": torch.stack([item["F_base"] for item in batch], dim=0),
        "F_shape": torch.stack([item["F_shape"] for item in batch], dim=0),
        "F_reference": torch.stack([item["F_reference"] for item in batch], dim=0),
        "F_base64": torch.stack([item["F_base64"] for item in batch], dim=0),
        "F_shape64": torch.stack([item["F_shape64"] for item in batch], dim=0),
        "F_reference64": torch.stack([item["F_reference64"] for item in batch], dim=0),
        "face_i": torch.stack([item["face_i"] for item in batch], dim=0),
        "reference_i": torch.stack([item["reference_i"] for item in batch], dim=0),
        "aligned_reference_i": torch.stack([item["aligned_reference_i"] for item in batch], dim=0),
        "priors": priors,
    }


class ArcFacePatchEncoder_v19(torch.nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        self.backbone = iresnet50()
        self.backbone.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.backbone.to(device).eval()
        self.pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        toggle_grad(self.backbone, False)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        image = ((image + 1.0) / 2.0).clamp(0, 1)
        mask = F.interpolate(mask.float(), size=image.shape[-2:], mode="bilinear", align_corners=False)
        patch = image * mask + 0.5 * (1.0 - mask)
        patch = self.pool(patch) * 2.0 - 1.0
        return self.backbone(patch)


class TrainerV19:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader):
        self.device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(USER_USE_AMP and self.device.type == "cuda")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = ACTIVE_OUTPUT_DIR
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.preview_dir = self.output_dir / "previews"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

        opts = build_opts(self.device)
        self.net = Net(opts)
        toggle_grad(self.net.generator, False)
        self.net.generator.eval()
        self.downsample_256 = BicubicDownSample(factor=4, cuda="cuda" in str(self.device))

        self.model = BoundaryLayerShapeAdaptor_v19(hidden_channels=USER_SHAPE_ADAPTER_HIDDEN_V19).to(self.device)
        if USER_INIT_CKPT:
            checkpoint = torch.load(USER_INIT_CKPT, map_location=self.device)
            state = checkpoint.get("shape_adapter_v19_state_dict", checkpoint)
            self.model.load_state_dict(state, strict=False)

        self.feature_encoder = None
        if USER_LAMBDA_LOCAL_FEATURE > 0 or USER_LAMBDA_FACE_IDENTITY > 0:
            self.feature_encoder = ArcFacePatchEncoder_v19(USER_ARCFACE_CKPT, self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_accum_steps = max(1, USER_EFFECTIVE_BATCH_SIZE // USER_BATCH_SIZE)
        self.best_loss = float("inf")
        self.cur_iter = 0

    def _autocast_context(self):
        if self.use_amp:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return nullcontext()

    def _generator_autocast_context(self):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    def _maybe_release_memory(self, step_idx: int):
        if not USER_LOW_MEMORY_MODE:
            return
        if self.device.type != "cuda":
            return
        if USER_EMPTY_CACHE_EVERY <= 0:
            return
        if step_idx % USER_EMPTY_CACHE_EVERY == 0:
            release_memory()

    def _to_device(self, batch: dict) -> dict:
        return {
            "face_s": batch["face_s"].to(self.device),
            "F_src": batch["F_src"].to(self.device),
            "F_base": batch["F_base"].to(self.device),
            "F_shape": batch["F_shape"].to(self.device),
            "F_reference": batch["F_reference"].to(self.device),
            "F_base64": batch["F_base64"].to(self.device),
            "F_shape64": batch["F_shape64"].to(self.device),
            "F_reference64": batch["F_reference64"].to(self.device),
            "face_i": batch["face_i"].to(self.device),
            "reference_i": batch["reference_i"].to(self.device),
            "aligned_reference_i": batch["aligned_reference_i"].to(self.device),
            "priors": {key: ensure_bchw_mask_v19(value.to(self.device)) for key, value in batch["priors"].items()},
        }

    def render_with_detail(self, latent_s: torch.Tensor, latent_f32: torch.Tensor, latent_f64: torch.Tensor) -> torch.Tensor:
        latent_s = latent_s.float()
        latent_f32 = latent_f32.float()
        latent_f64 = latent_f64.float()
        with self._generator_autocast_context():
            feature_64_seed, skip_64 = self.net.generator(
                [latent_s],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=4,
                layer_in=latent_f32,
            )
            if latent_f64.shape[-2:] != feature_64_seed.shape[-2:]:
                latent_f64 = F.interpolate(latent_f64, size=feature_64_seed.shape[-2:], mode="bilinear", align_corners=False)
            image, _ = self.net.generator(
                [latent_s],
                input_is_latent=True,
                return_latents=False,
                start_layer=5,
                end_layer=8,
                layer_in=latent_f64,
                skip=skip_64,
            )
            return self.downsample_256(image).float()

    def render_baseline_256(self, latent_s: torch.Tensor, latent_f32: torch.Tensor, latent_f64: torch.Tensor) -> torch.Tensor:
        if USER_BASELINE_RENDER_MODE == "detail":
            return self.render_with_detail(latent_s, latent_f32, latent_f64)
        latent_s = latent_s.float()
        latent_f32 = latent_f32.float()
        with self._generator_autocast_context():
            image, _ = self.net.generator(
                [latent_s],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_f32,
            )
            return self.downsample_256(image).float()

    def forward_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        with self._autocast_context():
            out = self.model(
                F_base=batch["F_base"],
                F_src=batch["F_src"],
                F_shape=batch["F_shape"],
                F_base64=batch["F_base64"],
                F_shape64=batch["F_shape64"],
                F_reference=batch["F_reference"],
                F_reference64=batch["F_reference64"],
                priors_256=batch["priors"],
                strength=USER_SHAPE_ADAPTER_STRENGTH_V19,
                shape_prior_strength=USER_SHAPE_PRIOR_STRENGTH_V19,
                detail_strength=USER_SHAPE_DETAIL_STRENGTH_V19,
            )
        pred_render_256 = self.render_with_detail(batch["face_s"], out["latent_F_refined"], out["latent_F64_detail"])
        with torch.no_grad():
            baseline_render_256 = self.render_baseline_256(batch["face_s"], batch["F_base"], batch["F_base64"])
        out.update(
            {
                "pred_render_256": pred_render_256,
                "baseline_render_256": baseline_render_256,
                "pred_256": pred_render_256,
                "baseline_256": baseline_render_256,
            }
        )
        return out

    def calc_losses(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        priors = batch["priors"]
        pred = outputs["pred_256"].float()
        baseline = outputs["baseline_256"].float()
        aligned_reference = batch["aligned_reference_i"].float()
        source = batch["face_i"].float()
        coarse_pred = outputs["coarse_target_mask"].float()
        target_pred = outputs["target_hair_mask_soft"].float()
        residual_pred = outputs["boundary_residual"].float()
        latent_f32 = outputs["latent_F_refined"].float()
        latent_f64 = outputs["latent_F64_detail"].float()

        coarse_target = priors["coarse_target_mask"].float()
        target_mask = priors["pseudo_target_mask"].float()
        boundary = priors["boundary_band"].float()
        source_hair = priors["source_hair_mask"].float()
        detail_map = priors["hair_detail_map"].float()
        source_face_protect = priors["source_face_protect"].float()
        pseudo_residual = priors["pseudo_boundary_residual"].float()

        remove_region = (source_hair * (1.0 - target_mask)).clamp(0, 1)
        feature_gap32 = (batch["F_reference"].float() - batch["F_base"].float()).abs().mean(dim=1, keepdim=True)
        feature_gap32 = feature_gap32 / feature_gap32.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        repair_gate = F.interpolate(feature_gap32, size=target_mask.shape[-2:], mode="bilinear", align_corners=False)
        repair_mask = torch.maximum(
            torch.maximum(boundary, (0.85 * remove_region).clamp(0, 1)),
            torch.maximum((0.60 * (target_mask - coarse_target).abs()).clamp(0, 1), (0.55 * target_mask * repair_gate).clamp(0, 1)),
        )
        preserve = (source_face_protect + (1.0 - torch.maximum(source_hair, target_mask))).clamp(0, 1)
        coarse64 = F.interpolate(coarse_target, size=latent_f64.shape[-2:], mode="bilinear", align_corners=False)
        boundary32 = F.interpolate(boundary, size=latent_f32.shape[-2:], mode="bilinear", align_corners=False)
        boundary64 = F.interpolate(boundary, size=latent_f64.shape[-2:], mode="bilinear", align_corners=False)
        target32 = F.interpolate(target_mask, size=latent_f32.shape[-2:], mode="bilinear", align_corners=False)
        target64 = F.interpolate(target_mask, size=latent_f64.shape[-2:], mode="bilinear", align_corners=False)
        repair32 = F.interpolate(repair_mask, size=latent_f32.shape[-2:], mode="bilinear", align_corners=False)
        preserve32 = F.interpolate(preserve, size=latent_f32.shape[-2:], mode="bilinear", align_corners=False)
        preserve64 = F.interpolate(preserve, size=latent_f64.shape[-2:], mode="bilinear", align_corners=False)
        remove64 = F.interpolate(remove_region, size=latent_f64.shape[-2:], mode="bilinear", align_corners=False)
        detail_map64 = F.interpolate(detail_map, size=latent_f64.shape[-2:], mode="bilinear", align_corners=False)
        feature_gap64 = (batch["F_reference64"].float() - batch["F_base64"].float()).abs().mean(dim=1, keepdim=True)
        feature_gap64 = feature_gap64 / feature_gap64.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        detail_support64 = torch.maximum(
            torch.maximum(boundary64, (0.55 * remove64).clamp(0, 1)),
            torch.maximum((0.60 * (target64 - coarse64).abs()).clamp(0, 1), torch.maximum((0.50 * target64 * feature_gap64).clamp(0, 1), (0.40 * target64 * detail_map64).clamp(0, 1))),
        )

        losses = {
            "mask_coarse": USER_LAMBDA_MASK_COARSE * F.binary_cross_entropy(coarse_pred, coarse_target.clamp(0, 1)),
            "mask_target": USER_LAMBDA_MASK_TARGET * F.binary_cross_entropy(target_pred, target_mask.clamp(0, 1)),
            "mask_residual": USER_LAMBDA_MASK_RESIDUAL * masked_l1(residual_pred, pseudo_residual, boundary),
            "latent_shape32": USER_LAMBDA_LATENT_SHAPE32 * masked_l1(latent_f32, batch["F_reference"].float(), repair32),
            "latent_preserve32": USER_LAMBDA_LATENT_PRESERVE32 * masked_l1(latent_f32, batch["F_base"].float(), preserve32),
            "latent_detail64": USER_LAMBDA_LATENT_DETAIL64 * masked_l1(latent_f64, batch["F_reference64"].float(), detail_support64),
            "latent_preserve64": USER_LAMBDA_LATENT_PRESERVE64 * masked_l1(latent_f64, batch["F_base64"].float(), preserve64),
            "image_shape": USER_LAMBDA_IMAGE_SHAPE * masked_l1(pred, aligned_reference, repair_mask),
            "image_preserve": USER_LAMBDA_IMAGE_PRESERVE * masked_l1(pred, baseline.detach(), preserve),
            "boundary_sobel": USER_LAMBDA_BOUNDARY_SOBEL * masked_l1(sobel_edges(pred), sobel_edges(aligned_reference), boundary),
            "boundary_laplacian": USER_LAMBDA_BOUNDARY_LAPLACIAN * masked_l1(laplacian_edges(pred), laplacian_edges(aligned_reference), boundary),
            "boundary_chamfer": USER_LAMBDA_BOUNDARY_CHAMFER * chamfer_distance(target_pred, target_mask),
            "boundary_hausdorff": USER_LAMBDA_BOUNDARY_HAUSDORFF * hausdorff_distance(target_pred, target_mask),
            "high_freq": USER_LAMBDA_HIGH_FREQ * (1.0 - high_frequency_spectrum_similarity(pred, aligned_reference, boundary)),
            "delta32": USER_LAMBDA_DELTA32 * outputs["learned_delta_F32"].float().abs().mean(),
            "delta64": USER_LAMBDA_DELTA64 * outputs["learned_delta_F64"].float().abs().mean(),
        }

        face_identity_similarity = pred.new_tensor(1.0)
        local_feature_similarity = pred.new_tensor(1.0)
        if self.feature_encoder is not None:
            if USER_LAMBDA_FACE_IDENTITY > 0:
                pred_face_feat = self.feature_encoder(pred, source_face_protect)
                source_face_feat = self.feature_encoder(source, source_face_protect)
                face_identity_similarity = cosine_feature_similarity(pred_face_feat, source_face_feat)
                losses["face_identity"] = USER_LAMBDA_FACE_IDENTITY * (1.0 - face_identity_similarity)
            if USER_LAMBDA_LOCAL_FEATURE > 0:
                pred_feat = self.feature_encoder(pred, target_mask)
                reference_feat = self.feature_encoder(aligned_reference, target_mask)
                local_feature_similarity = cosine_feature_similarity(pred_feat, reference_feat)
                losses["local_feature"] = USER_LAMBDA_LOCAL_FEATURE * (1.0 - local_feature_similarity)

        losses["loss"] = sum(losses.values())
        with torch.no_grad():
            metrics = {
                "boundary_fscore": boundary_fscore(target_pred.detach(), target_mask.detach()),
                "boundary_iou": iou_score(target_pred.detach(), target_mask.detach()),
                "boundary_chamfer_metric": chamfer_distance(target_pred.detach(), target_mask.detach()),
                "boundary_hausdorff_metric": hausdorff_distance(target_pred.detach(), target_mask.detach()),
                "high_freq_similarity": high_frequency_spectrum_similarity(pred.detach(), aligned_reference.detach(), boundary.detach()),
                "face_identity_similarity": face_identity_similarity.detach(),
                "local_feature_similarity": local_feature_similarity.detach(),
            }
        return losses["loss"], losses, metrics

    def save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            "epoch": epoch,
            "shape_adapter_v19_state_dict": self.model.state_dict(),
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
        torch.save({"shape_adapter_v19_state_dict": self.model.state_dict()}, self.ckpt_dir / "shape_adapter_v19_for_infer.pth")

    @staticmethod
    def _to_pil(image: torch.Tensor) -> Image.Image:
        return T.functional.to_pil_image(((image + 1) / 2).detach().cpu().clamp(0, 1))

    @staticmethod
    def _mask_pil(mask: torch.Tensor) -> Image.Image:
        mask = mask.detach().cpu().float().clamp(0, 1)
        if mask.dim() == 3:
            mask = mask[0]
        return T.functional.to_pil_image(mask)

    def save_preview(self, epoch: int, previews: list[dict]):
        if epoch % USER_SAVE_PREVIEW_EVERY != 0:
            return
        save_dir = self.preview_dir / f"epoch_{epoch:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(previews[:USER_PREVIEW_COUNT]):
            panels = [
                self._to_pil(row["face"]),
                self._to_pil(row["shape"]),
                self._to_pil(row["baseline"]),
                self._to_pil(row["pred"]),
                self._mask_pil(row["target"]).convert("RGB"),
                self._mask_pil(row["boundary"]).convert("RGB"),
            ]
            width, height = panels[0].size
            canvas = Image.new("RGB", (width * len(panels), height), color=(255, 255, 255))
            for panel_idx, panel in enumerate(panels):
                canvas.paste(panel.convert("RGB"), (panel_idx * width, 0))
            canvas.save(save_dir / f"sample_{idx:03d}.png")

    def run_epoch(self, loader: DataLoader, training: bool, epoch: int) -> tuple[dict[str, float], dict[str, float], list[dict]]:
        self.model.train(training)
        totals = defaultdict(float)
        metric_totals = defaultdict(float)
        previews = []
        steps = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for step, raw_batch in enumerate(tqdm(loader, desc="train" if training else "val")):
                steps += 1
                batch = self._to_device(raw_batch)
                outputs = self.forward_batch(batch)
                loss, losses, metrics = self.calc_losses(outputs, batch)

                if training:
                    self.scaler.scale(loss / self.grad_accum_steps).backward()
                    if (step + 1) % self.grad_accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.cur_iter += 1

                for key, value in losses.items():
                    totals[key] += float(value.detach().cpu())
                for key, value in metrics.items():
                    metric_totals[key] += float(value.detach().cpu())

                if not training and len(previews) < USER_PREVIEW_COUNT:
                    max_add = min(batch["face_i"].shape[0], USER_PREVIEW_COUNT - len(previews))
                    for idx in range(max_add):
                        previews.append(
                            {
                                "face": batch["face_i"][idx].detach().cpu(),
                                "shape": batch["reference_i"][idx].detach().cpu(),
                                "baseline": outputs["baseline_256"][idx].detach().cpu(),
                                "pred": outputs["pred_256"][idx].detach().cpu(),
                                "target": batch["priors"]["pseudo_target_mask"][idx].detach().cpu(),
                                "boundary": batch["priors"]["boundary_band"][idx].detach().cpu(),
                            }
                        )

                del batch, outputs, loss, losses, metrics
                self._maybe_release_memory(steps)

        if training and steps % self.grad_accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        if USER_LOW_MEMORY_MODE:
            release_memory()
        avg_losses = {key: value / max(steps, 1) for key, value in totals.items()}
        avg_metrics = {key: value / max(steps, 1) for key, value in metric_totals.items()}
        return avg_losses, avg_metrics, previews

    def train_loop(self):
        start_epoch = 0
        if USER_RESUME_CHECKPOINT:
            checkpoint = torch.load(USER_RESUME_CHECKPOINT, map_location=self.device)
            state = checkpoint.get("shape_adapter_v19_state_dict", checkpoint)
            self.model.load_state_dict(state, strict=False)
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_loss = checkpoint.get("best_loss", self.best_loss)
            self.cur_iter = checkpoint.get("cur_iter", 0)
            start_epoch = checkpoint.get("epoch", -1) + 1

        for epoch in range(start_epoch, USER_EPOCHS):
            train_losses, train_metrics, _ = self.run_epoch(self.train_loader, training=True, epoch=epoch)
            val_losses, val_metrics, previews = self.run_epoch(self.val_loader, training=False, epoch=epoch)
            self.save_preview(epoch, previews)
            is_best = val_losses["loss"] <= self.best_loss
            if is_best:
                self.best_loss = val_losses["loss"]
            self.save_checkpoint(epoch, is_best=is_best)
            print(
                f"[epoch {epoch:03d}] "
                f"train={train_losses['loss']:.6f} "
                f"val={val_losses['loss']:.6f} "
                f"best={self.best_loss:.6f} "
                f"bf={val_metrics['boundary_fscore']:.4f} "
                f"iou={val_metrics['boundary_iou']:.4f} "
                f"hf={val_metrics['high_freq_similarity']:.4f}"
            )


def main():
    seed_everything(USER_RANDOM_SEED)
    set_seed(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exps = load_exps(ACTIVE_DATASET_DIR)
    if len(exps) <= ACTIVE_VAL_SIZE:
        raise RuntimeError(f"Not enough samples in {ACTIVE_DATASET_DIR}: {len(exps)}")

    train_exps, val_exps = train_test_split(exps, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    train_dataset = ShapeDatasetV19(train_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_SHAPE_ROOT)
    val_dataset = ShapeDatasetV19(val_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_SHAPE_ROOT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=True,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        collate_fn=collate_shape_v19,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=USER_BATCH_SIZE,
        shuffle=False,
        num_workers=USER_NUM_WORKERS,
        pin_memory=USER_PIN_MEMORY and torch.cuda.is_available(),
        collate_fn=collate_shape_v19,
        drop_last=False,
    )

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"dataset dir: {ACTIVE_DATASET_DIR}")
    print(f"train/val: {len(train_dataset)}/{len(val_dataset)}")
    trainer = TrainerV19(train_loader, val_loader)
    trainer.train_loop()


if __name__ == "__main__":
    main()
