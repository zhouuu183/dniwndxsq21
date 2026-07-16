import gc
import os
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

from hair_swap_v18 import HairFast_v18, get_parser_v18
from models.Encoders import ClipBlendingModel as BlendingModel
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.save_utils import save_latents
from utils.train import toggle_grad


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("input/blending_dataset_v18")
USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/blending_train_v18")
USER_VAL_SIZE_FFHQ = 512

USER_DATASET_DIR_SMALL = Path("input/blending_dataset_v18_small")
USER_FACE_ROOT_SMALL = Path("images/FFHQ_long")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_short")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ_color")
USER_OUTPUT_DIR_SMALL = Path("output/blending_train_v18_small")
USER_VAL_SIZE_SMALL = 64

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_BATCH_SIZE = 16
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_EPOCHS = 20
USER_LR = 2e-5
USER_WEIGHT_DECAY = 1e-6
USER_GRAD_CLIP = 5.0

USER_INIT_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_FALLBACK_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_CLIP_MODEL = "ViT-B/32"
USER_BUILD_CACHE = True
USER_FORCE_REFRESH_ALIGN_CACHE = False
USER_USE_SATD_V18 = False
USER_SATD_CKPT_V18 = ""
USER_SATD_BLEND_V18 = 0.28

USER_SAVE_CHECKPOINT_EVERY = 1
USER_SAVE_PREVIEW_EVERY = 1
USER_LOG_IMAGE_COUNT = 16

USER_LAMBDA_CLIP_HAIR_V18 = 0.0
USER_LAMBDA_COLOR_CHROMA_V18 = 10.00
USER_LAMBDA_COLOR_RGB_STATS_V18 = 6.00
USER_LAMBDA_HAIR_LUMA_KEEP_V18 = 0.45
USER_LAMBDA_TONE_RGB_KEEP_V18 = 2.50
USER_LAMBDA_TONE_LUMA_KEEP_V18 = 2.00
USER_LAMBDA_TONE_CHROMA_KEEP_V18 = 1.50
USER_LAMBDA_ALIGN_PRESERVE_V18 = 1.50
USER_LAMBDA_ALIGN_NON_DARK_V18 = 2.00
USER_LAMBDA_REMOVE_PRESERVE_V18 = 4.00
USER_LAMBDA_REMOVE_NON_DARK_V18 = 3.00
# =============================================================================


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


def gray(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def rgb01(image: torch.Tensor) -> torch.Tensor:
    return ((image + 1.0) * 0.5).clamp(0.0, 1.0)


def chroma_uv(image: torch.Tensor) -> torch.Tensor:
    image = rgb01(image)
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    u = -0.14713 * r - 0.28886 * g + 0.43600 * b
    v = 0.61500 * r - 0.51499 * g - 0.10001 * b
    return torch.cat([u, v], dim=1)


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


def masked_pair_chroma_stats_loss(
    pred: torch.Tensor,
    pred_mask: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    return masked_pair_stats_loss(chroma_uv(pred), pred_mask, chroma_uv(target), target_mask)


def masked_pair_rgb_stats_loss(
    pred: torch.Tensor,
    pred_mask: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    return masked_pair_stats_loss(rgb01(pred), pred_mask, rgb01(target), target_mask)


def resolve_init_blending_checkpoint() -> Path:
    ckpt_path = Path(USER_INIT_BLENDING_CKPT) if USER_INIT_BLENDING_CKPT else Path(USER_FALLBACK_BLENDING_CKPT)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Cannot find blending init checkpoint: {ckpt_path}. "
            "Use the author's pretrained blending checkpoint as initialization."
        )
    return ckpt_path


def find_image_path(root: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find {stem}.png/.jpg/.jpeg/.webp in {root}")


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


def build_remove_protect_mask_v18(align_info: dict[str, object]) -> torch.Tensor:
    delta_masks = align_info.get("delta_masks")
    if not isinstance(delta_masks, dict):
        hm_x = align_info["HM_X"]
        return torch.zeros_like(hm_x).float()

    remove = delta_masks["M_remove"].float()
    zero = torch.zeros_like(remove)
    protect = (
        1.00 * remove
        + 0.95 * delta_masks.get("M_remove_halo", zero).float()
        + 0.90 * delta_masks.get("M_remove_face", zero).float()
        + 0.90 * delta_masks.get("M_remove_neck", zero).float()
        + 0.85 * delta_masks.get("M_remove_tail", zero).float()
        + 0.80 * delta_masks.get("M_remove_context", zero).float()
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


def build_cache_model() -> HairFast_v18:
    model_args = get_parser_v18().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    model_args.use_satd_v18 = bool(USER_USE_SATD_V18 and USER_SATD_CKPT_V18)
    model_args.satd_checkpoint_v18 = USER_SATD_CKPT_V18
    model_args.satd_blend_v18 = USER_SATD_BLEND_V18
    model_args.use_post_satd_v18 = False
    model_args.use_postprocess_v18 = False

    hair_fast = HairFast_v18(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)
    return hair_fast


def ensure_dataset_cache_v18(triplets: list[tuple[str, str, str]]):
    if not USER_BUILD_CACHE:
        return

    fs_dir = ACTIVE_DATASET_DIR / "FS"
    align_dir = ACTIVE_DATASET_DIR / "Align"
    mask_dir = ACTIVE_DATASET_DIR / "Masks"
    fs_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    def fs_path(stem: str) -> Path:
        return fs_dir / f"{stem}.npz"

    def align_path(face_stem: str, ref_stem: str) -> Path:
        return align_dir / f"{face_stem}_{ref_stem}.npz"

    def mask_path(face_stem: str, ref_stem: str) -> Path:
        return mask_dir / f"{face_stem}_{ref_stem}.npz"

    hair_fast = build_cache_model()
    required_triplets = []
    for face_name, shape_name, color_name in triplets:
        need_align_shape = USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, shape_name).exists())
        need_align_color = USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, color_name).exists())
        need_mask_shape = USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_path(face_name, shape_name).exists())
        need_mask_color = USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_path(face_name, color_name).exists())
        need_face_fs = not fs_path(face_name).exists()
        need_shape_fs = not fs_path(shape_name).exists()
        need_color_fs = not fs_path(color_name).exists()
        if need_align_shape or need_align_color or need_mask_shape or need_mask_color or need_face_fs or need_shape_fs or need_color_fs:
            required_triplets.append((face_name, shape_name, color_name))

    if not required_triplets:
        print("[blending_v18] FS/Align cache already matches current training needs.", file=sys.stderr)
        return

    print(f"[blending_v18] rebuilding cache for {len(required_triplets)} triplets", file=sys.stderr)

    for face_name, shape_name, color_name in tqdm(required_triplets, desc="Build v18 FS/Align cache", leave=False):
        face_path = find_image_path(ACTIVE_FACE_ROOT, face_name)
        shape_path = find_image_path(ACTIVE_SHAPE_ROOT, shape_name)
        color_path = find_image_path(ACTIVE_COLOR_ROOT, color_name)

        align_shape, align_color, name_to_embed = hair_fast(
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
            save_latents(ACTIVE_DATASET_DIR, "Align", f"{face_name}_{shape_name}.npz", latent_F=align_shape["latent_F_align"])
        if USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, color_name).exists()):
            save_latents(ACTIVE_DATASET_DIR, "Align", f"{face_name}_{color_name}.npz", latent_F=align_color["latent_F_align"])
        if USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_path(face_name, shape_name).exists()):
            save_latents(ACTIVE_DATASET_DIR, "Masks", f"{face_name}_{shape_name}.npz", remove_mask=build_remove_protect_mask_v18(align_shape))
        if USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_path(face_name, color_name).exists()):
            save_latents(ACTIVE_DATASET_DIR, "Masks", f"{face_name}_{color_name}.npz", remove_mask=build_remove_protect_mask_v18(align_color))

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
        align_f = torch.from_numpy(np.load(dataset_dir / "Align" / f"{face_name}_{color_name}.npz")["latent_F"]).squeeze(0)
        remove_mask = torch.from_numpy(np.load(dataset_dir / "Masks" / f"{face_name}_{color_name}.npz")["remove_mask"]).squeeze(0)

        with Image.open(find_image_path(color_root, color_name)) as color_image:
            color_i = T.functional.normalize(T.functional.to_tensor(color_image.convert("RGB")), [0.5], [0.5])
        with Image.open(find_image_path(face_root, face_name)) as face_image:
            face_i = T.functional.normalize(T.functional.to_tensor(face_image.convert("RGB")), [0.5], [0.5])
        return color_s, align_s, align_f, remove_mask, color_i, face_i
    except Exception as exc:
        print(exc, file=sys.stderr)
        return None


class BlendingDatasetV18(Dataset):
    def __init__(
        self,
        exps: list[tuple[str, str, str]],
        dataset_dir: Path,
        face_root: Path,
        color_root: Path,
    ):
        super().__init__()
        self.exps = [(p1, p2, p3) for (p1, p2, p3) in exps] + [(p1, p3, p2) for (p1, p2, p3) in exps]
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


class BlendingTrainerV18:
    def __init__(self, model: BlendingModel, optimizer, train_loader: DataLoader, val_loader: DataLoader, helper: MaskPrepHelper):
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
        color_s, align_s, align_f, remove_mask, color_i, face_i = [item.to(self.device, non_blocking=True) for item in batch]
        remove_mask = remove_mask.float().clamp(0, 1)
        if remove_mask.dim() == 3:
            remove_mask = remove_mask.unsqueeze(1)

        hm_3d, hm_3e = self.helper.generate_mask(color_i)
        hm_1d, _ = self.helper.generate_mask(face_i)
        i_x, _ = self.helper.net.generator(
            [align_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=align_f,
        )
        hm_xd, _ = self.helper.generate_mask(i_x)
        target_mask = (1 - hm_1d) * (1 - hm_3d) * (1 - hm_xd)
        align_protect_mask = (remove_mask + 0.35 * target_mask + 0.25 * (1 - hm_xd) * (1 - hm_3e)).clamp(0, 1)
        color_transfer_mask = (hm_xd * (1.0 - remove_mask)).clamp(0, 1)
        color_reference_mask = hm_3d.clamp(0, 1)
        i_x_256 = self.helper.downsample_256(i_x)
        face_i_256 = self.helper.downsample_256(face_i)
        color_i_256 = self.helper.downsample_256(color_i)

        valid = color_reference_mask.flatten(1).any(dim=1) & color_transfer_mask.flatten(1).any(dim=1)
        if not valid.any():
            return None

        return (
            color_s[valid],
            align_s[valid],
            align_f[valid],
            color_i_256[valid],
            face_i_256[valid],
            i_x_256[valid],
            target_mask[valid],
            align_protect_mask[valid],
            remove_mask[valid],
            color_transfer_mask[valid],
            color_reference_mask[valid],
        )

    def calc_loss(self, i_gen, i_face, i_color, i_align, mask_face, mask_align, mask_remove, mask_color_transfer, mask_color_ref):
        gen_embed = self.model.get_image_embed(i_gen * mask_face)
        gt_embed = self.model.get_image_embed(i_face * mask_face)
        face_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        gen_embed = self.model.get_image_embed(i_gen * mask_color_transfer)
        gt_embed = self.model.get_image_embed(i_color * mask_color_ref)
        clip_hair_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        color_chroma_loss = masked_pair_chroma_stats_loss(i_gen, mask_color_transfer, i_color, mask_color_ref)
        color_rgb_stats_loss = masked_pair_rgb_stats_loss(i_gen, mask_color_transfer, i_color, mask_color_ref)
        hair_luma_keep_loss = masked_l1(gray(i_gen), gray(i_align), mask_color_transfer)
        tone_rgb_keep_loss = masked_l1(i_gen, i_align, mask_face)
        tone_luma_keep_loss = masked_l1(gray(i_gen), gray(i_align), mask_face)
        tone_chroma_keep_loss = masked_l1(chroma_uv(i_gen), chroma_uv(i_align), mask_face)
        align_preserve_loss = masked_l1(i_gen, i_align, mask_align)
        align_non_dark_loss = masked_non_darker(gray(i_gen), gray(i_align), (mask_align + mask_color_transfer).clamp(0, 1))
        remove_preserve_loss = masked_l1(i_gen, i_align, mask_remove)
        remove_non_dark_loss = masked_non_darker(gray(i_gen), gray(i_align), mask_remove)

        total_loss = (
            face_loss
            + USER_LAMBDA_CLIP_HAIR_V18 * clip_hair_loss
            + USER_LAMBDA_COLOR_CHROMA_V18 * color_chroma_loss
            + USER_LAMBDA_COLOR_RGB_STATS_V18 * color_rgb_stats_loss
            + USER_LAMBDA_HAIR_LUMA_KEEP_V18 * hair_luma_keep_loss
            + USER_LAMBDA_TONE_RGB_KEEP_V18 * tone_rgb_keep_loss
            + USER_LAMBDA_TONE_LUMA_KEEP_V18 * tone_luma_keep_loss
            + USER_LAMBDA_TONE_CHROMA_KEEP_V18 * tone_chroma_keep_loss
            + USER_LAMBDA_ALIGN_PRESERVE_V18 * align_preserve_loss
            + USER_LAMBDA_ALIGN_NON_DARK_V18 * align_non_dark_loss
            + USER_LAMBDA_REMOVE_PRESERVE_V18 * remove_preserve_loss
            + USER_LAMBDA_REMOVE_NON_DARK_V18 * remove_non_dark_loss
        )
        return total_loss, {
            "face_loss": face_loss,
            "clip_hair_loss": clip_hair_loss,
            "color_chroma_loss": color_chroma_loss,
            "color_rgb_stats_loss": color_rgb_stats_loss,
            "hair_luma_keep_loss": hair_luma_keep_loss,
            "tone_rgb_keep_loss": tone_rgb_keep_loss,
            "tone_luma_keep_loss": tone_luma_keep_loss,
            "tone_chroma_keep_loss": tone_chroma_keep_loss,
            "align_preserve_loss": align_preserve_loss,
            "align_non_dark_loss": align_non_dark_loss,
            "remove_preserve_loss": remove_preserve_loss,
            "remove_non_dark_loss": remove_non_dark_loss,
            "loss": total_loss,
        }

    def save_checkpoint(self, epoch: int, best_loss: float, name: str):
        model_state_dict = self.model.state_dict()
        saved_state_dict = {key: value for key, value in model_state_dict.items() if not key.startswith("clip_model.")}
        torch.save(
            {
                "epoch": epoch,
                "best_loss": best_loss,
                "clip": USER_CLIP_MODEL,
                "model_state_dict": saved_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.output_ckpt_dir / f"{name}.pth",
        )

    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        running_steps = 0
        progress = tqdm(self.train_loader, desc=f"Blend-v18 train {epoch + 1}/{USER_EPOCHS}", leave=False)
        for batch in progress:
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue

            (
                color_s,
                align_s,
                align_f,
                color_i,
                face_i,
                i_x_256,
                target_mask,
                align_protect_mask,
                remove_mask,
                color_transfer_mask,
                hm_3e,
            ) = prepared
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * hm_3e)
            latent_in = torch.cat((align_s[:, :6], blend_s), dim=1)
            i_g, _ = self.helper.net.generator(
                [latent_in],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=align_f,
            )
            loss, loss_info = self.calc_loss(
                self.helper.downsample_256(i_g),
                face_i,
                color_i,
                i_x_256,
                target_mask,
                align_protect_mask,
                remove_mask,
                color_transfer_mask,
                hm_3e,
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
            "clip_hair_loss": 0.0,
            "color_chroma_loss": 0.0,
            "color_rgb_stats_loss": 0.0,
            "hair_luma_keep_loss": 0.0,
            "tone_rgb_keep_loss": 0.0,
            "tone_luma_keep_loss": 0.0,
            "tone_chroma_keep_loss": 0.0,
            "align_preserve_loss": 0.0,
            "align_non_dark_loss": 0.0,
            "remove_preserve_loss": 0.0,
            "remove_non_dark_loss": 0.0,
            "loss": 0.0,
        }
        total_steps = 0
        preview_rows = []

        for batch in tqdm(self.val_loader, desc=f"Blend-v18 val {epoch + 1}/{USER_EPOCHS}", leave=False):
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue

            (
                color_s,
                align_s,
                align_f,
                color_i,
                face_i,
                i_x_256,
                target_mask,
                align_protect_mask,
                remove_mask,
                color_transfer_mask,
                hm_3e,
            ) = prepared
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * hm_3e)
            latent_in = torch.cat((align_s[:, :6], blend_s), dim=1)
            i_g, _ = self.helper.net.generator(
                [latent_in],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=align_f,
            )
            i_g_256 = self.helper.downsample_256(i_g)
            loss, loss_info = self.calc_loss(
                i_g_256,
                face_i,
                color_i,
                i_x_256,
                target_mask,
                align_protect_mask,
                remove_mask,
                color_transfer_mask,
                hm_3e,
            )

            del loss
            for key, value in loss_info.items():
                total_losses[key] += float(value.item())
            total_steps += 1

            if len(preview_rows) < USER_LOG_IMAGE_COUNT:
                for idx in range(i_g_256.size(0)):
                    preview_rows.append([face_i[idx : idx + 1], color_i[idx : idx + 1], i_x_256[idx : idx + 1], i_g_256[idx : idx + 1]])
                    if len(preview_rows) >= USER_LOG_IMAGE_COUNT:
                        break

        avg_losses = {key: value / max(total_steps, 1) for key, value in total_losses.items()}

        if epoch % USER_SAVE_PREVIEW_EVERY == 0:
            epoch_dir = self.output_val_dir / f"epoch_{epoch + 1:03d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            for idx, row in enumerate(preview_rows):
                save_preview(epoch_dir / f"sample_{idx:03d}.png", row)

        print(
            f"[blending_v18] epoch={epoch + 1} "
            f"val_loss={avg_losses['loss']:.6f} "
            f"val_face={avg_losses['face_loss']:.6f} "
            f"val_color={avg_losses['color_chroma_loss']:.6f} "
            f"val_luma={avg_losses['hair_luma_keep_loss']:.6f}"
        )
        return avg_losses["loss"]

    def train_loop(self):
        for epoch in range(USER_EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            print(f"[blending_v18] epoch={epoch + 1} train_loss={train_loss:.6f}")

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

    ensure_dataset_cache_v18(triplets)
    train_exps, val_exps = train_test_split(triplets, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    helper = MaskPrepHelper(device)

    train_dataset = BlendingDatasetV18(train_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_COLOR_ROOT)
    val_dataset = BlendingDatasetV18(val_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_COLOR_ROOT)
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

    model = BlendingModel(USER_CLIP_MODEL)
    ckpt = torch.load(resolve_init_blending_checkpoint(), map_location=device)
    load_compatible_state_dict(model, ckpt.get("model_state_dict", ckpt))
    optimizer = torch.optim.Adam(model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)

    trainer = BlendingTrainerV18(model, optimizer, train_loader, val_loader, helper)
    trainer.train_loop()


if __name__ == "__main__":
    main()
