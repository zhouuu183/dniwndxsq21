import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v17 import HairFast_v17, get_parser_v17
from models.Blending_v17 import build_lock_mask_v17, build_safe_mask_v17
from models.Net import Net
from models.SID_Blending_v17 import SIDBlendingModel_v17, gray01, image_from_norm, rgb01_to_hsv, rgb01_to_lab_ab
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.save_utils import save_latents
from utils.train import toggle_grad


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("input/blending_dataset_v17")
USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/blending_train_v17")
USER_VAL_SIZE_FFHQ = 512

USER_DATASET_DIR_SMALL = Path("iamges/blending_dataset_v17_small")
USER_FACE_ROOT_SMALL = Path("images/FFHQ_long")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_short")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ_color")
USER_OUTPUT_DIR_SMALL = Path("output/blending_train_v17_small")
USER_VAL_SIZE_SMALL = 64

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_BATCH_SIZE = 12
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_EPOCHS = 20
USER_LR = 1e-4
USER_WEIGHT_DECAY = 1e-6
USER_GRAD_CLIP = 5.0

USER_INIT_BLENDING_CKPT = ""
USER_CLIP_MODEL_V17 = "ViT-B/32"
USER_USE_SATD_V17 = True
USER_SATD_CHECKPOINT_V17 = "checkpoints/best.pth"
USER_SATD_BLEND_V17 = 0.34
USER_SATD_BOUNDARY_V17 = 8
USER_EQ_REFERENCE_BLEND_V17 = 0.0
USER_BUILD_CACHE_WITH_CURRENT_SATD = True
USER_FORCE_REFRESH_ALIGN_CACHE = False

USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_PREVIEW_EVERY = 1
USER_LOG_IMAGE_COUNT = 30

USER_LAMBDA_FACE_CLIP_V17 = 1.0
USER_LAMBDA_CHROMA_V17 = 18.0
USER_LAMBDA_LUMA_KEEP_V17 = 1.0
USER_LAMBDA_LUMA_STYLE_V17 = 0.0
USER_LAMBDA_SATD_PRESERVE_V17 = 1.2
USER_LAMBDA_NON_DARK_V17 = 0.2
USER_LAMBDA_NON_HAIR_PRESERVE_V17 = 6.0
USER_LAMBDA_NON_HAIR_CHROMA_V17 = 8.0
USER_LAMBDA_HAIR_DETAIL_V17 = 0.6

USER_FEATURE_STRENGTH_V17 = 1.5
# ============================================================================


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def masked_mean_std(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.shape[1] == 1 and features.shape[1] != 1:
        mask = mask.expand(-1, features.shape[1], -1, -1)
    denom = mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
    mean = (features * mask).sum(dim=(-2, -1), keepdim=True) / denom
    var = ((features - mean).pow(2) * mask).sum(dim=(-2, -1), keepdim=True) / denom
    return mean, torch.sqrt(var + 1e-6)


def masked_pair_stats_loss(
    pred_features: torch.Tensor,
    pred_mask: torch.Tensor,
    target_features: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    pred_mean, pred_std = masked_mean_std(pred_features, pred_mask)
    target_mean, target_std = masked_mean_std(target_features, target_mask)
    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


def masked_soft_histogram(values: torch.Tensor, mask: torch.Tensor, bins: int, value_min: float, value_max: float) -> torch.Tensor:
    values = values.view(values.size(0), -1, 1)
    mask = mask.view(mask.size(0), -1, 1).float()
    centers = torch.linspace(value_min, value_max, steps=bins, device=values.device, dtype=values.dtype).view(1, 1, bins)
    sigma = (value_max - value_min) / max(bins - 1, 1)
    weights = torch.exp(-0.5 * ((values - centers) / max(sigma, 1e-6)) ** 2) * mask
    hist = weights.sum(dim=1)
    return hist / hist.sum(dim=1, keepdim=True).clamp(min=1e-6)


def masked_pair_lab_hist_loss(
    pred: torch.Tensor,
    pred_mask: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    pred_ab = rgb01_to_lab_ab(image_from_norm(pred))
    target_ab = rgb01_to_lab_ab(image_from_norm(target))
    pred_hist_a = masked_soft_histogram(pred_ab[:, 0:1], pred_mask, bins=8, value_min=-110.0, value_max=110.0)
    pred_hist_b = masked_soft_histogram(pred_ab[:, 1:2], pred_mask, bins=8, value_min=-110.0, value_max=110.0)
    target_hist_a = masked_soft_histogram(target_ab[:, 0:1], target_mask, bins=8, value_min=-110.0, value_max=110.0)
    target_hist_b = masked_soft_histogram(target_ab[:, 1:2], target_mask, bins=8, value_min=-110.0, value_max=110.0)
    return F.l1_loss(pred_hist_a, target_hist_a) + F.l1_loss(pred_hist_b, target_hist_b)


def chroma_transfer_loss(
    pred: torch.Tensor,
    pred_mask: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    pred_ab = rgb01_to_lab_ab(image_from_norm(pred))
    target_ab = rgb01_to_lab_ab(image_from_norm(target))
    pred_hs = rgb01_to_hsv(image_from_norm(pred))[:, :2]
    target_hs = rgb01_to_hsv(image_from_norm(target))[:, :2]

    ab_loss = masked_pair_stats_loss(pred_ab, pred_mask, target_ab, target_mask)
    hs_loss = masked_pair_stats_loss(pred_hs, pred_mask, target_hs, target_mask)
    hist_loss = masked_pair_lab_hist_loss(pred, pred_mask, target, target_mask)
    return ab_loss + 0.5 * hs_loss + 0.5 * hist_loss


def chroma_preserve_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_rgb = image_from_norm(pred)
    target_rgb = image_from_norm(target)
    pred_ab = rgb01_to_lab_ab(pred_rgb)
    target_ab = rgb01_to_lab_ab(target_rgb)
    pred_hs = rgb01_to_hsv(pred_rgb)[:, :2]
    target_hs = rgb01_to_hsv(target_rgb)[:, :2]
    return masked_l1(pred_ab, target_ab, mask) + 0.5 * masked_l1(pred_hs, target_hs, mask)


def masked_gradient_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().clamp(0, 1)
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    loss_x = ((pred_dx - target_dx).abs() * mask_x).sum() / mask_x.sum().clamp(min=1.0)
    loss_y = ((pred_dy - target_dy).abs() * mask_y).sum() / mask_y.sum().clamp(min=1.0)
    return loss_x + loss_y


def find_image_path(root: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find an image for stem={stem!r} in {root}")


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


def build_cache_model() -> HairFast_v17:
    model_args = get_parser_v17().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    model_args.use_satd_v17 = bool(USER_USE_SATD_V17)
    model_args.satd_checkpoint_v17 = USER_SATD_CHECKPOINT_V17
    model_args.satd_blend_v17 = USER_SATD_BLEND_V17
    model_args.satd_boundary_v17 = USER_SATD_BOUNDARY_V17
    model_args.eq_reference_blend_v17 = USER_EQ_REFERENCE_BLEND_V17

    hair_fast = HairFast_v17(model_args)
    hair_fast.blend.blend_images = identity_func
    return hair_fast


def ensure_dataset_cache_v17(triplets: list[tuple[str, str, str]]):
    if not USER_BUILD_CACHE_WITH_CURRENT_SATD:
        return

    if USER_USE_SATD_V17:
        if not USER_SATD_CHECKPOINT_V17:
            raise RuntimeError("USER_SATD_CHECKPOINT_V17 is empty while USER_USE_SATD_V17=True.")
        if not Path(USER_SATD_CHECKPOINT_V17).exists():
            raise FileNotFoundError(f"Cannot find USER_SATD_CHECKPOINT_V17: {USER_SATD_CHECKPOINT_V17}")

    fs_dir = ACTIVE_DATASET_DIR / "FS"
    align_dir = ACTIVE_DATASET_DIR / "Align"
    mask_dir = ACTIVE_DATASET_DIR / "Masks"
    fs_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    def fs_path(stem: str) -> Path:
        return fs_dir / f"{stem}.npz"

    def align_path(face_stem: str, shape_stem: str) -> Path:
        return align_dir / f"{face_stem}_{shape_stem}.npz"

    def mask_path(face_stem: str, shape_stem: str) -> Path:
        return mask_dir / f"{face_stem}_{shape_stem}.npz"

    hair_fast = build_cache_model()
    required_triplets = []
    for face_name, shape_name, color_name in triplets:
        need_align = USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, shape_name).exists())
        need_mask = USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_path(face_name, shape_name).exists())
        need_face_fs = not fs_path(face_name).exists()
        need_shape_fs = not fs_path(shape_name).exists()
        need_color_fs = not fs_path(color_name).exists()
        if need_align or need_mask or need_face_fs or need_shape_fs or need_color_fs:
            required_triplets.append((face_name, shape_name, color_name))

    if not required_triplets:
        print("[blending_v17] FS/Align/Masks cache already matches current training needs.", file=sys.stderr)
        return

    print(
        f"[blending_v17] rebuilding cache for {len(required_triplets)} triplets "
        f"(use_satd_v17={USER_USE_SATD_V17}, satd_checkpoint={USER_SATD_CHECKPOINT_V17})",
        file=sys.stderr,
    )

    for face_name, shape_name, color_name in tqdm(required_triplets, desc="Build v17 FS/Align cache", leave=False):
        face_path = find_image_path(ACTIVE_FACE_ROOT, face_name)
        shape_path = find_image_path(ACTIVE_SHAPE_ROOT, shape_name)
        color_path = find_image_path(ACTIVE_COLOR_ROOT, color_name)

        align_shape, _, name_to_embed = hair_fast(
            face_path,
            shape_path,
            color_path,
            align_flag=True,
        )

        if not fs_path(face_name).exists():
            save_latents(ACTIVE_DATASET_DIR, "FS", f"{face_name}.npz", latent_in=name_to_embed["face"]["S"])
        if not fs_path(shape_name).exists():
            save_latents(ACTIVE_DATASET_DIR, "FS", f"{shape_name}.npz", latent_in=name_to_embed["shape"]["S"])
        if not fs_path(color_name).exists():
            save_latents(ACTIVE_DATASET_DIR, "FS", f"{color_name}.npz", latent_in=name_to_embed["color"]["S"])

        if USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, shape_name).exists()):
            save_latents(
                ACTIVE_DATASET_DIR,
                "Align",
                f"{face_name}_{shape_name}.npz",
                latent_F=align_shape["latent_F_align"],
            )

        if USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_path(face_name, shape_name).exists()):
            hair_mask = align_shape["HM_X"].float()
            lock_mask = build_lock_mask_v17(align_shape)
            safe_mask = build_safe_mask_v17(hair_mask, lock_mask)
            save_latents(
                ACTIVE_DATASET_DIR,
                "Masks",
                f"{face_name}_{shape_name}.npz",
                hair_mask=hair_mask,
                lock_mask=lock_mask,
                safe_mask=safe_mask,
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
        tiles.append(((tensor + 1) / 2).detach().cpu().clamp(0, 1))
    panel = torch.cat(tiles, dim=2)
    save_image(panel, path)


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
    def generate_mask(self, image: torch.Tensor):
        image_512 = (self.downsample_512((image + 1) / 2) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(image_512)
        current_mask = torch.argmax(down_seg, dim=1).long().float()
        hair_mask = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
        hair_mask = F.interpolate(hair_mask.unsqueeze(1), size=(256, 256), mode="nearest")
        hair_mask_dilate, hair_mask_erode = self.dilate_erosion.mask(hair_mask)
        return hair_mask_dilate, hair_mask_erode


def prepare_item(exp, dataset_dir: Path, face_root: Path, color_root: Path):
    face_name, shape_name, color_name = exp

    try:
        color_s = torch.from_numpy(np.load(dataset_dir / "FS" / f"{color_name}.npz")["latent_in"]).squeeze(0)
        align_s = torch.from_numpy(np.load(dataset_dir / "FS" / f"{face_name}.npz")["latent_in"]).squeeze(0)
        align_f = torch.from_numpy(np.load(dataset_dir / "Align" / f"{face_name}_{shape_name}.npz")["latent_F"]).squeeze(0)
        mask_npz = np.load(dataset_dir / "Masks" / f"{face_name}_{shape_name}.npz")
        hair_mask = torch.from_numpy(mask_npz["hair_mask"]).float().squeeze(0)
        lock_mask = torch.from_numpy(mask_npz["lock_mask"]).float().squeeze(0)
        safe_mask = torch.from_numpy(mask_npz["safe_mask"]).float().squeeze(0)

        with Image.open(find_image_path(color_root, color_name)) as color_image:
            color_i = T.functional.normalize(T.functional.to_tensor(color_image.convert("RGB")), [0.5], [0.5])
        with Image.open(find_image_path(face_root, face_name)) as face_image:
            face_i = T.functional.normalize(T.functional.to_tensor(face_image.convert("RGB")), [0.5], [0.5])
        return color_s, align_s, align_f, hair_mask, lock_mask, safe_mask, color_i, face_i
    except Exception as exc:
        print(exc, file=sys.stderr)
        return None


class BlendingDatasetV17(Dataset):
    def __init__(
        self,
        exps: list[tuple[str, str, str]],
        dataset_dir: Path,
        face_root: Path,
        color_root: Path,
    ):
        super().__init__()
        self.exps = list(exps)
        self.dataset_dir = dataset_dir
        self.face_root = face_root
        self.color_root = color_root
        print(f"dataset pairs: {len(self.exps)}", file=sys.stderr)

    def __len__(self):
        return len(self.exps)

    def __getitem__(self, idx):
        item = prepare_item(self.exps[idx], self.dataset_dir, self.face_root, self.color_root)
        if item is None:
            raise RuntimeError(f"Failed to prepare blending item at index {idx}")
        return item


class BlendingTrainerV17:
    def __init__(
        self,
        model: SIDBlendingModel_v17,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        helper: MaskPrepHelper,
    ):
        self.device = helper.device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.helper = helper
        self.best_loss = float("inf")
        self.output_ckpt_dir = ACTIVE_OUTPUT_DIR / "checkpoints"
        self.output_val_dir = ACTIVE_OUTPUT_DIR / "val_images"
        self.output_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_val_dir.mkdir(parents=True, exist_ok=True)

    def prepare_batch(self, batch):
        color_s, align_s, align_f, hair_mask, lock_mask, safe_mask, color_i, face_i = [
            item.to(self.device, non_blocking=True) for item in batch
        ]
        safe_mask = build_safe_mask_v17(hair_mask, lock_mask)

        hm_color_d, hm_color_e = self.helper.generate_mask(color_i)
        hm_face_d, _ = self.helper.generate_mask(face_i)
        hm_align_d, _ = self.helper.dilate_erosion.mask(hair_mask)
        face_target_mask = (1.0 - hm_face_d) * (1.0 - hm_align_d)

        i_satd, _ = self.helper.net.generator(
            [align_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=align_f,
        )
        i_satd_256 = self.helper.downsample_256(i_satd)
        color_i_256 = self.helper.downsample_256(color_i)
        face_i_256 = self.helper.downsample_256(face_i)
        satd_luma = gray01(image_from_norm(i_satd_256))

        valid = hm_color_e.flatten(1).any(dim=1) & safe_mask.flatten(1).any(dim=1)
        if not valid.any():
            return None

        return (
            color_s[valid],
            align_s[valid],
            align_f[valid],
            hair_mask[valid],
            lock_mask[valid],
            safe_mask[valid],
            color_i_256[valid],
            face_i_256[valid],
            hm_color_e[valid],
            face_target_mask[valid],
            i_satd_256[valid],
            satd_luma[valid],
        )

    def calc_loss(
        self,
        i_gen: torch.Tensor,
        i_satd: torch.Tensor,
        i_face: torch.Tensor,
        i_color: torch.Tensor,
        face_target_mask: torch.Tensor,
        ref_hair_mask: torch.Tensor,
        hair_mask: torch.Tensor,
        lock_mask: torch.Tensor,
        safe_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        face_embed = self.model.get_image_embed(i_gen * face_target_mask)
        face_gt = self.model.get_image_embed(i_face * face_target_mask)
        face_loss = (1 - F.cosine_similarity(face_embed, face_gt)).mean()

        chroma_mask = torch.maximum(safe_mask, 0.35 * (hair_mask * (1.0 - lock_mask)).clamp(0, 1)).clamp(0, 1)
        chroma_loss = chroma_transfer_loss(i_gen, chroma_mask, i_color, ref_hair_mask)
        luma_keep_loss = masked_l1(gray01(image_from_norm(i_gen)), gray01(image_from_norm(i_satd)), safe_mask)
        luma_style_loss = masked_pair_stats_loss(
            gray01(image_from_norm(i_gen)),
            chroma_mask,
            gray01(image_from_norm(i_color)),
            ref_hair_mask,
        )

        residual_hair = (hair_mask - safe_mask).clamp(0, 1)
        preserve_mask = (lock_mask + 0.15 * residual_hair).clamp(0, 1)
        satd_preserve_loss = masked_l1(i_gen, i_satd, preserve_mask)
        non_dark_mask = (lock_mask + 0.10 * residual_hair).clamp(0, 1)
        non_dark_loss = masked_non_darker(
            gray01(image_from_norm(i_gen)),
            gray01(image_from_norm(i_satd)),
            non_dark_mask,
        )
        non_hair_preserve_loss = masked_l1(i_gen, i_satd, face_target_mask)
        non_hair_chroma_loss = chroma_preserve_loss(i_gen, i_satd, face_target_mask)
        hair_detail_mask = (hair_mask * (1.0 - lock_mask)).clamp(0, 1)
        hair_detail_loss = masked_gradient_l1(
            gray01(image_from_norm(i_gen)),
            gray01(image_from_norm(i_satd)),
            hair_detail_mask,
        )

        total_loss = (
            USER_LAMBDA_FACE_CLIP_V17 * face_loss
            + USER_LAMBDA_CHROMA_V17 * chroma_loss
            + USER_LAMBDA_LUMA_KEEP_V17 * luma_keep_loss
            + USER_LAMBDA_LUMA_STYLE_V17 * luma_style_loss
            + USER_LAMBDA_SATD_PRESERVE_V17 * satd_preserve_loss
            + USER_LAMBDA_NON_DARK_V17 * non_dark_loss
            + USER_LAMBDA_NON_HAIR_PRESERVE_V17 * non_hair_preserve_loss
            + USER_LAMBDA_NON_HAIR_CHROMA_V17 * non_hair_chroma_loss
            + USER_LAMBDA_HAIR_DETAIL_V17 * hair_detail_loss
        )
        return total_loss, {
            "face_loss": face_loss,
            "chroma_loss": chroma_loss,
            "luma_keep_loss": luma_keep_loss,
            "luma_style_loss": luma_style_loss,
            "satd_preserve_loss": satd_preserve_loss,
            "non_dark_loss": non_dark_loss,
            "non_hair_preserve_loss": non_hair_preserve_loss,
            "non_hair_chroma_loss": non_hair_chroma_loss,
            "hair_detail_loss": hair_detail_loss,
            "loss": total_loss,
        }

    def save_checkpoint(self, epoch: int, best_loss: float, name: str):
        model_state_dict = self.model.state_dict()
        saved_state_dict = {key: value for key, value in model_state_dict.items() if not key.startswith("clip_model.")}
        torch.save(
            {
                "epoch": epoch,
                "best_loss": best_loss,
                "clip": USER_CLIP_MODEL_V17,
                "model_state_dict": saved_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.output_ckpt_dir / f"{name}.pth",
        )

    def _forward_generator(self, batch_prepared):
        (
            color_s,
            align_s,
            align_f,
            hair_mask,
            lock_mask,
            safe_mask,
            color_i,
            face_i,
            ref_hair_mask,
            face_target_mask,
            i_satd,
            satd_luma,
        ) = batch_prepared

        s_blend_6_18, f_blend, _ = self.model(
            latent_face=align_s[:, 6:],
            latent_color=color_s[:, 6:],
            target_face=face_i * face_target_mask,
            reference_hair_image=color_i,
            reference_hair_mask=ref_hair_mask,
            safe_mask=safe_mask,
            lock_mask=lock_mask,
            align_hair_mask=hair_mask,
            satd_luma=satd_luma,
            align_f=align_f,
            feature_strength=USER_FEATURE_STRENGTH_V17,
        )
        latent_in = torch.cat((align_s[:, :6], s_blend_6_18), dim=1)
        i_g, _ = self.helper.net.generator(
            [latent_in],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=f_blend,
        )
        i_g_256 = self.helper.downsample_256(i_g)
        return i_g_256, (
            hair_mask,
            lock_mask,
            safe_mask,
            color_i,
            face_i,
            ref_hair_mask,
            face_target_mask,
            i_satd,
        )

    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        running_steps = 0
        progress = tqdm(self.train_loader, desc=f"SID blend train {epoch + 1}/{USER_EPOCHS}", leave=False)
        for batch in progress:
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue

            i_g_256, loss_items = self._forward_generator(prepared)
            hair_mask, lock_mask, safe_mask, color_i, face_i, ref_hair_mask, face_target_mask, i_satd = loss_items
            loss, loss_info = self.calc_loss(
                i_gen=i_g_256,
                i_satd=i_satd,
                i_face=face_i,
                i_color=color_i,
                face_target_mask=face_target_mask,
                ref_hair_mask=ref_hair_mask,
                hair_mask=hair_mask,
                lock_mask=lock_mask,
                safe_mask=safe_mask,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
            self.optimizer.step()

            running_loss += float(loss.item())
            running_steps += 1
            progress.set_postfix(loss=float(loss.item()), grad=float(grad_norm))

        return running_loss / max(running_steps, 1)

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        total_losses = {
            "face_loss": 0.0,
            "chroma_loss": 0.0,
            "luma_keep_loss": 0.0,
            "luma_style_loss": 0.0,
            "satd_preserve_loss": 0.0,
            "non_dark_loss": 0.0,
            "non_hair_preserve_loss": 0.0,
            "non_hair_chroma_loss": 0.0,
            "hair_detail_loss": 0.0,
            "loss": 0.0,
        }
        total_steps = 0
        preview_rows = []

        for batch in tqdm(self.val_loader, desc=f"SID blend val {epoch + 1}/{USER_EPOCHS}", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue

            i_g_256, loss_items = self._forward_generator(prepared)
            hair_mask, lock_mask, safe_mask, color_i, face_i, ref_hair_mask, face_target_mask, i_satd = loss_items
            loss, loss_info = self.calc_loss(
                i_gen=i_g_256,
                i_satd=i_satd,
                i_face=face_i,
                i_color=color_i,
                face_target_mask=face_target_mask,
                ref_hair_mask=ref_hair_mask,
                hair_mask=hair_mask,
                lock_mask=lock_mask,
                safe_mask=safe_mask,
            )

            del loss
            for key, value in loss_info.items():
                total_losses[key] += float(value.item())
            total_steps += 1

            if len(preview_rows) < USER_LOG_IMAGE_COUNT:
                for idx in range(i_g_256.size(0)):
                    preview_rows.append([face_i[idx : idx + 1], color_i[idx : idx + 1], i_satd[idx : idx + 1], i_g_256[idx : idx + 1]])
                    if len(preview_rows) >= USER_LOG_IMAGE_COUNT:
                        break

        avg_losses = {key: value / max(total_steps, 1) for key, value in total_losses.items()}

        if epoch % USER_SAVE_PREVIEW_EVERY == 0:
            epoch_dir = self.output_val_dir / f"epoch_{epoch + 1:03d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            for idx, row in enumerate(preview_rows):
                save_preview(epoch_dir / f"sample_{idx:03d}.png", row)

        print(
            f"[blending_v17] epoch={epoch + 1} "
            f"val_loss={avg_losses['loss']:.6f} "
            f"val_face={avg_losses['face_loss']:.6f} "
            f"val_chroma={avg_losses['chroma_loss']:.6f} "
            f"val_luma={avg_losses['luma_keep_loss']:.6f}"
        )
        return avg_losses["loss"]

    def train_loop(self):
        for epoch in range(USER_EPOCHS):
            print(
                f"[blending_v17] epoch={epoch + 1} "
                f"feature_only=True feature_strength={USER_FEATURE_STRENGTH_V17:.2f}"
            )
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            print(f"[blending_v17] epoch={epoch + 1} train_loss={train_loss:.6f}")

            if (epoch + 1) % USER_SAVE_CHECKPOINT_EVERY == 0:
                self.save_checkpoint(epoch + 1, self.best_loss, "last")
            if val_loss <= self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch + 1, self.best_loss, "best")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
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

    ensure_dataset_cache_v17(triplets)
    train_exps, val_exps = train_test_split(triplets, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    helper = MaskPrepHelper(device)

    train_dataset = BlendingDatasetV17(train_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_COLOR_ROOT)
    val_dataset = BlendingDatasetV17(val_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_COLOR_ROOT)
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

    model = SIDBlendingModel_v17(USER_CLIP_MODEL_V17)
    if USER_INIT_BLENDING_CKPT:
        ckpt = torch.load(USER_INIT_BLENDING_CKPT, map_location=device)
        load_compatible_state_dict(model, ckpt.get("model_state_dict", ckpt))
    optimizer = torch.optim.Adam(model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)

    trainer = BlendingTrainerV17(model, optimizer, train_loader, val_loader, helper)
    print(
        f"[blending_v17] sid_blending=True use_satd_v17={USER_USE_SATD_V17} "
        f"satd_checkpoint={USER_SATD_CHECKPOINT_V17}",
        file=sys.stderr,
    )
    trainer.train_loop()


if __name__ == "__main__":
    main()
