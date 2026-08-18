import atexit
import gc
import multiprocessing
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import random
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
from models.Encoders import ClipBlendingModel as BlendingModel
from models.Net import Net
from models.SG_IDCT_v16 import rgb_to_lab
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.hair_color_completion_v8 import reference_color_completion_v8
from utils.image_utils import DilateErosion
from utils.save_utils import save_latents
from utils.train import get_fid_calc, toggle_grad


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
USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("input/blending_dataset_v8")
USER_FACE_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/blending_train_v8_v2")
USER_VAL_SIZE_FFHQ = 512

USER_DATASET_DIR_SMALL = Path("input/blending_dataset_v8_small")
USER_FACE_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_long")
USER_SHAPE_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short")
USER_COLOR_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_color")
USER_OUTPUT_DIR_SMALL = Path("output/blending_train_v8_small_v2")
USER_VAL_SIZE_SMALL = 64

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_BATCH_SIZE = 16
USER_GRAD_ACCUM_STEPS = 1  # effective batch size = USER_BATCH_SIZE * USER_GRAD_ACCUM_STEPS
USER_NUM_WORKERS = 0
USER_PIN_MEMORY = False
USER_EPOCHS = 20
USER_LR = 2e-5
USER_WEIGHT_DECAY = 1e-6
USER_GRAD_CLIP = 5.0
USER_FACE_CLIP_LOSS_WEIGHT = 1.0
USER_HAIR_CLIP_LOSS_WEIGHT = 0.25
USER_HAIR_CHROMA_STATS_LOSS_WEIGHT = 16.0
USER_HAIR_CHROMA_HIST_LOSS_WEIGHT = 5.0
USER_HAIR_RGB_STATS_LOSS_WEIGHT = 2.0
USER_HAIR_LAB_MEAN_LOSS_WEIGHT = 3.0
USER_HAIR_LUMA_STYLE_LOSS_WEIGHT = 0.75
USER_HAIR_LUMA_OVER_LOSS_WEIGHT = 1.10
USER_HAIR_LUMA_OVER_STD = 1.35
USER_HAIR_LUMA_OVER_MARGIN = 4.0
USER_HAIR_LUMA_BASE_GRAD_LOSS_WEIGHT = 0.20
USER_HAIR_HIGHLIGHT_CHROMA_LOSS_WEIGHT = 3.0
USER_HAIR_HIGHLIGHT_MIN_PIXELS = 16.0
USER_HAIR_BASE_STATS_QUANTILE = 0.85
USER_HAIR_VIVID_COVERAGE_LOSS_WEIGHT = 2.5
USER_HAIR_VIVID_PROGRESS_FLOOR = 0.72
USER_HAIR_VIVID_MIN_PIXELS = 48.0
USER_FACE_KEEP_L1_LOSS_WEIGHT = 1.0
USER_REMOVE_KEEP_L1_LOSS_WEIGHT = 1.1
USER_PROTECT_CHROMA_KEEP_LOSS_WEIGHT = 2.5
USER_SKIN_CHROMA_KEEP_LOSS_WEIGHT = 10.0
USER_SKIN_RGB_KEEP_LOSS_WEIGHT = 4.0
USER_HAIR_VERTICAL_LAB_MEAN_LOSS_WEIGHT = 2.0
USER_HAIR_VERTICAL_REGIONS = 3
USER_HAIR_VERTICAL_MIN_PIXELS = 24.0
USER_SAFE_HAIR_MIN_PIXELS = 64.0
USER_TARGET_HAIR_NECK_OVERRIDE = 0.94
USER_AUTHOR_COLOR_ALIGN_BATCH_PROB = 0.35
USER_AUTHOR_ZERO_PREFIX_TRAIN = True

USER_INIT_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_FALLBACK_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_CLIP_MODEL = "ViT-B/32"
USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = "/data/coding/HairFastGAN/HairFastGAN-main/best.pth"
USER_SATD_BLEND_V8 = 0.34
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0
USER_BUILD_CACHE_WITH_CURRENT_SATD = False
USER_FORCE_REFRESH_ALIGN_CACHE = False

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
if not 0.0 <= USER_AUTHOR_COLOR_ALIGN_BATCH_PROB <= 1.0:
    raise RuntimeError("USER_AUTHOR_COLOR_ALIGN_BATCH_PROB must be in [0, 1].")
if not 0.0 <= USER_TARGET_HAIR_NECK_OVERRIDE <= 1.0:
    raise RuntimeError("USER_TARGET_HAIR_NECK_OVERRIDE must be in [0, 1].")
if not 0.5 <= USER_HAIR_BASE_STATS_QUANTILE < 1.0:
    raise RuntimeError("USER_HAIR_BASE_STATS_QUANTILE must be in [0.5, 1.0).")
if USER_HAIR_VERTICAL_REGIONS < 1:
    raise RuntimeError("USER_HAIR_VERTICAL_REGIONS must be >= 1.")
if USER_HAIR_VERTICAL_MIN_PIXELS < 1:
    raise RuntimeError("USER_HAIR_VERTICAL_MIN_PIXELS must be >= 1.")
if USER_HAIR_HIGHLIGHT_MIN_PIXELS < 1:
    raise RuntimeError("USER_HAIR_HIGHLIGHT_MIN_PIXELS must be >= 1.")
if USER_HAIR_VIVID_MIN_PIXELS < 1:
    raise RuntimeError("USER_HAIR_VIVID_MIN_PIXELS must be >= 1.")
if not 0.0 <= USER_HAIR_VIVID_PROGRESS_FLOOR <= 1.0:
    raise RuntimeError("USER_HAIR_VIVID_PROGRESS_FLOOR must be in [0, 1].")


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


def resolve_init_blending_checkpoint() -> Path:
    ckpt_path = Path(USER_INIT_BLENDING_CKPT) if USER_INIT_BLENDING_CKPT else Path(USER_FALLBACK_BLENDING_CKPT)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Cannot find blending init checkpoint: {ckpt_path}. "
            "Do not train v8 blending from scratch; use the author's pretrained blending checkpoint."
        )
    if not USER_INIT_BLENDING_CKPT:
        print(f"[blending_v8] USER_INIT_BLENDING_CKPT is empty; using fallback {ckpt_path}", file=sys.stderr)
    return ckpt_path


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
    return f"{role_key(role, stem)}.npz"


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
                "Run scripts/blending_gen_v8.py again or set USER_BUILD_CACHE_WITH_CURRENT_SATD=True "
                f"to rebuild it.\nMissing examples:\n{preview}"
            )
        return

    hair_fast = build_cache_model()
    required_triplets = []
    for face_name, shape_name, color_name in triplets:
        need_align_shape = USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, "shape", shape_name).exists())
        need_align_color = USER_FORCE_REFRESH_ALIGN_CACHE or (not align_path(face_name, "color", color_name).exists())
        need_mask_shape = USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_has_target_hair(mask_path(face_name, "shape", shape_name)))
        need_mask_color = USER_FORCE_REFRESH_ALIGN_CACHE or (not mask_has_target_hair(mask_path(face_name, "color", color_name)))
        need_face_fs = not fs_path("face", face_name).exists()
        need_shape_fs = not fs_path("shape", shape_name).exists()
        need_color_fs = not fs_path("color", color_name).exists()
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
        tiles.append(((tensor + 1) / 2).detach().cpu().clamp(0, 1))
    panel = torch.cat(tiles, dim=2)
    save_image(panel, path)


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
        self,
        image: torch.Tensor,
        return_keep: bool = False,
        return_raw: bool = False,
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
        else:
            result = (hair_mask_dilate, hair_mask_erode)
        if return_raw:
            result = (*result, hair_mask.clamp(0, 1))
        return result


def prepare_item(exp, dataset_dir: Path, face_root: Path, color_root: Path):
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
        return (
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
        print(f"dataset pairs: {len(self.exps)}", file=sys.stderr)

    def __len__(self):
        return len(self.exps)

    def __getitem__(self, idx):
        item = prepare_item(self.exps[idx], self.dataset_dir, self.face_root, self.color_root)
        if item is None:
            raise RuntimeError(f"Failed to prepare blending item at index {idx}")
        return item


class BlendingTrainerV8:
    def __init__(
        self,
        model: BlendingModel,
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
        self.grad_accum_steps = int(USER_GRAD_ACCUM_STEPS)
        self.best_loss = float("inf")
        self.output_ckpt_dir = ACTIVE_OUTPUT_DIR / "checkpoints"
        self.output_val_dir = ACTIVE_OUTPUT_DIR / "val_images"
        self.output_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_val_dir.mkdir(parents=True, exist_ok=True)
        self.fid_calc = None
        if USER_USE_FID and Path(USER_FID_DATASET).exists():
            self.fid_calc = get_fid_calc(USER_FID_CACHE, str(USER_FID_DATASET), device=self.device)

    def prepare_batch(self, batch, use_author_color_align: bool = False):
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
        ) = batch
        align_f = align_f_color if use_author_color_align else align_f_shape
        remove_mask = remove_mask_color if use_author_color_align else remove_mask_shape
        target_hair = target_hair_color if use_author_color_align else target_hair_shape
        color_s, align_s, align_f, remove_mask, target_hair, color_i, face_i = [
            item.to(self.device, non_blocking=True)
            for item in (color_s, align_s, align_f, remove_mask, target_hair, color_i, face_i)
        ]
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
                hm_x_raw,
            ) = self.helper.generate_mask(
                i_x,
                return_keep=True,
                return_raw=True,
            )
            i_x_256 = self.helper.downsample_256(i_x)
            face_i_256 = self.helper.downsample_256(face_i)
            color_i_256 = self.helper.downsample_256(color_i)

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
        target_hair_full = torch.where(
            has_cached_hair.view(-1, 1, 1, 1),
            target_hair,
            hm_xd,
        ).clamp(0, 1)

        target_mask = (1 - hm_1d) * (1 - hm_3d) * (1 - target_hair_d)
        rendered_hair_support = torch.maximum(hm_x_raw, 0.18 * hm_xd)
        color_transfer_mask = (target_hair_full * rendered_hair_support).clamp(0, 1)
        neck_hair_override = (USER_TARGET_HAIR_NECK_OVERRIDE * color_transfer_mask).clamp(0, 1)
        target_neck_visible = (target_neck_guard * (1.0 - neck_hair_override)).clamp(0, 1)
        source_neck_visible = (source_neck_guard * (1.0 - neck_hair_override)).clamp(0, 1)
        skin_protect_mask = (
            target_face_guard
            + target_neck_visible
            + 0.45 * source_face_guard
            + 0.45 * source_neck_visible
        ).clamp(0, 1)
        remove_color_block = (remove_mask * (1.0 - color_transfer_mask)).clamp(0, 1)
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
        color_reference_mask = hm_3e.clamp(0, 1)

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
            satd_protect_mask[valid],
            remove_color_block[valid],
            color_transfer_mask[valid],
            face_keep_mask[valid],
            skin_protect_mask[valid],
            hm_3e[valid],
            color_reference_mask[valid],
            hm_1d[valid],
            target_hair_full[valid],
        )

    @torch.no_grad()
    def complete_validation_image(
        self,
        i_g: torch.Tensor,
        face_i: torch.Tensor,
        color_i: torch.Tensor,
        hm_1d: torch.Tensor,
        target_hair_full: torch.Tensor,
        hm_3e: torch.Tensor,
        face_keep_mask: torch.Tensor,
        skin_protect_mask: torch.Tensor,
        remove_mask: torch.Tensor,
        generated_hair: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the same inference completion to validation previews."""
        source_hair = hm_1d.clamp(0, 1)
        target_hair = target_hair_full.clamp(0, 1)
        fixed, _, _ = reference_color_completion_v8(
            source=face_i,
            reference=color_i,
            blend_raw=i_g,
            source_hair=source_hair,
            reference_hair=hm_3e,
            target_hair=target_hair,
            generated_hair=generated_hair,
            protect_masks={
                "M_face_region": skin_protect_mask,
                "M_body_preserve": (face_keep_mask + remove_mask).clamp(0, 1),
            },
        )
        return fixed

    @staticmethod
    def masked_l1(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        denom = mask.sum().clamp_min(1.0) * source.size(1)
        return (torch.abs(source - target) * mask).sum() / denom

    @staticmethod
    def masked_mean_std(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.float().clamp(0, 1)
        if mask.size(1) == 1 and image.size(1) != 1:
            mask = mask.expand(-1, image.size(1), -1, -1)
        denom = mask.flatten(2).sum(dim=2).clamp_min(1.0)
        mean = (image * mask).flatten(2).sum(dim=2) / denom
        centered = image - mean[:, :, None, None]
        var = (centered.square() * mask).flatten(2).sum(dim=2) / denom
        return mean, torch.sqrt(var.clamp_min(1e-8))

    def masked_pair_stats_loss(
        self,
        pred_features: torch.Tensor,
        pred_mask: torch.Tensor,
        target_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_mean, pred_std = self.masked_mean_std(pred_features, pred_mask)
        target_mean, target_std = self.masked_mean_std(target_features, target_mask)
        return tnf.l1_loss(pred_mean, target_mean) + tnf.l1_loss(pred_std, target_std)

    def masked_pair_mean_loss(
        self,
        pred_features: torch.Tensor,
        pred_mask: torch.Tensor,
        target_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_mean, _ = self.masked_mean_std(pred_features, pred_mask)
        target_mean, _ = self.masked_mean_std(target_features, target_mask)
        return tnf.l1_loss(pred_mean, target_mean)

    @staticmethod
    def masked_quantile(
        features: torch.Tensor,
        mask: torch.Tensor,
        quantile: float,
        fallback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        result = []
        for index in range(features.size(0)):
            selected = features[index].flatten(1)[:, mask[index, 0].flatten() > 0.05]
            if selected.size(1) == 0:
                if fallback is None:
                    result.append(features[index].new_zeros(features.size(1)))
                else:
                    result.append(fallback[index])
            else:
                result.append(torch.quantile(selected.float(), quantile, dim=1))
        return torch.stack(result, dim=0)

    def masked_highlight_chroma_loss(
        self,
        pred_chroma: torch.Tensor,
        pred_luma: torch.Tensor,
        pred_mask: torch.Tensor,
        target_chroma: torch.Tensor,
        target_luma: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_q75 = self.masked_quantile(pred_luma, pred_mask, 0.75)
        target_q75 = self.masked_quantile(target_luma, target_mask, 0.75)
        pred_highlight = pred_mask * (pred_luma >= pred_q75[:, :, None, None]).float()
        target_highlight = target_mask * (target_luma >= target_q75[:, :, None, None]).float()
        pred_valid = pred_highlight.flatten(1).sum(dim=1) >= USER_HAIR_HIGHLIGHT_MIN_PIXELS
        target_valid = target_highlight.flatten(1).sum(dim=1) >= USER_HAIR_HIGHLIGHT_MIN_PIXELS
        pred_highlight = torch.where(
            pred_valid[:, None, None, None],
            pred_highlight,
            pred_mask,
        )
        target_highlight = torch.where(
            target_valid[:, None, None, None],
            target_highlight,
            target_mask,
        )
        return self.masked_pair_stats_loss(
            pred_chroma,
            pred_highlight,
            target_chroma,
            target_highlight,
        )

    def robust_base_hair_mask(
        self,
        luma: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Exclude the brightest reference-dependent tail from global color statistics."""
        threshold = self.masked_quantile(
            luma,
            mask,
            USER_HAIR_BASE_STATS_QUANTILE,
        )
        base_mask = mask * (luma <= threshold[:, :, None, None]).float()
        enough = base_mask.flatten(1).sum(dim=1) >= USER_SAFE_HAIR_MIN_PIXELS
        return torch.where(enough[:, None, None, None], base_mask, mask).clamp(0, 1)

    def masked_vivid_coverage_loss(
        self,
        pred_chroma: torch.Tensor,
        base_chroma: torch.Tensor,
        pred_mask: torch.Tensor,
        target_chroma: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        base_mean, _ = self.masked_mean_std(base_chroma, pred_mask)
        target_mean, _ = self.masked_mean_std(target_chroma, target_mask)
        direction = target_mean - base_mean
        distance = direction.norm(dim=1).clamp_min(1e-6)
        target_chroma_size = target_mean.norm(dim=1)
        vividness = (
            ((target_chroma_size - 12.0) / 30.0).clamp(0, 1)
            * ((distance - 7.0) / 35.0).clamp(0, 1)
        ).detach()

        unit_direction = direction / distance[:, None]
        progress_map = (
            (pred_chroma - base_mean[:, :, None, None])
            * unit_direction[:, :, None, None]
        ).sum(dim=1, keepdim=True) / distance[:, None, None, None]
        low_progress = self.masked_quantile(progress_map, pred_mask, 0.25)[:, 0]
        valid = (
            (pred_mask.flatten(1).sum(dim=1) >= USER_HAIR_VIVID_MIN_PIXELS)
            & (target_mask.flatten(1).sum(dim=1) >= USER_HAIR_VIVID_MIN_PIXELS)
            & (vividness > 0)
        ).float()
        per_sample = torch.relu(USER_HAIR_VIVID_PROGRESS_FLOOR - low_progress) * vividness
        return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)

    def masked_vertical_pair_mean_loss(
        self,
        pred_features: torch.Tensor,
        pred_mask: torch.Tensor,
        target_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        regions = max(int(USER_HAIR_VERTICAL_REGIONS), 1)
        min_pixels = float(USER_HAIR_VERTICAL_MIN_PIXELS)
        pred_mask = pred_mask.float().clamp(0, 1)
        target_mask = target_mask.float().clamp(0, 1)

        def normalized_vertical_position(mask: torch.Tensor) -> torch.Tensor:
            height = mask.size(2)
            rows = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
            present = mask > 0.05
            y_min = torch.where(present, rows, torch.full_like(rows, float(height))).flatten(2).min(dim=2)[0]
            y_max = torch.where(present, rows, torch.full_like(rows, -1.0)).flatten(2).max(dim=2)[0] + 1.0
            y_min = y_min[:, :, None, None]
            span = (y_max[:, :, None, None] - y_min).clamp_min(1.0)
            return (rows - y_min) / span

        pred_position = normalized_vertical_position(pred_mask)
        target_position = normalized_vertical_position(target_mask)
        total_loss = pred_features.sum() * 0.0
        valid_regions = pred_features.new_zeros(())

        pred_present = pred_mask > 0.05
        height = pred_mask.size(2)
        rows = torch.arange(height, device=pred_mask.device, dtype=pred_mask.dtype).view(1, 1, height, 1)
        pred_y_min = torch.where(
            pred_present,
            rows,
            torch.full_like(rows, float(height)),
        ).flatten(2).min(dim=2)[0]
        pred_y_max = torch.where(
            pred_present,
            rows,
            torch.full_like(rows, -1.0),
        ).flatten(2).max(dim=2)[0]
        pred_extent = (pred_y_max - pred_y_min + 1.0).clamp_min(0.0) / max(height, 1)
        pred_bottom = pred_y_max / max(height - 1, 1)
        long_hair = (pred_extent >= 0.48) & (pred_bottom >= 0.72)

        pred_global_mean, _ = self.masked_mean_std(pred_features, pred_mask)
        target_global_mean, _ = self.masked_mean_std(target_features, target_mask)
        global_valid = (
            long_hair[:, 0]
            & (pred_mask.flatten(1).sum(dim=1) >= min_pixels)
            & (target_mask.flatten(1).sum(dim=1) >= min_pixels)
        ).float()
        global_loss = torch.abs(pred_global_mean - target_global_mean).mean(dim=1)
        total_loss = total_loss + (global_loss * global_valid).sum()
        valid_regions = valid_regions + global_valid.sum()

        for region_idx in range(regions):
            center = (float(region_idx) + 0.5) / regions
            half_width = 0.52 if regions == 3 else min(0.75, 1.6 / regions)
            pred_weight = (1.0 - torch.abs(pred_position - center) / half_width).clamp_min(0.0)
            target_weight = (1.0 - torch.abs(target_position - center) / half_width).clamp_min(0.0)
            pred_region_mask = pred_mask * pred_weight
            target_region_mask = target_mask * target_weight
            pred_mean, _ = self.masked_mean_std(pred_features, pred_region_mask)
            target_mean, _ = self.masked_mean_std(target_features, target_region_mask)
            valid = (
                (pred_region_mask.flatten(1).sum(dim=1) >= min_pixels)
                & (target_region_mask.flatten(1).sum(dim=1) >= min_pixels)
                & (~long_hair[:, 0])
            ).float()
            per_sample_loss = torch.abs(pred_mean - target_mean).mean(dim=1)
            total_loss = total_loss + (per_sample_loss * valid).sum()
            valid_regions = valid_regions + valid.sum()

        return total_loss / valid_regions.clamp_min(1.0)

    @staticmethod
    def masked_soft_histogram(
        values: torch.Tensor,
        mask: torch.Tensor,
        bins: int,
        value_min: float,
        value_max: float,
    ) -> torch.Tensor:
        values = values.flatten(2)
        mask = mask.float().clamp(0, 1)
        if mask.size(1) == 1 and values.size(1) != 1:
            mask = mask.expand(-1, values.size(1), -1, -1)
        mask = mask.flatten(2)

        centers = torch.linspace(value_min, value_max, steps=bins, device=values.device, dtype=values.dtype)
        centers = centers.view(1, 1, 1, bins)
        sigma = (value_max - value_min) / max(bins - 1, 1)
        weights = torch.exp(-0.5 * ((values.unsqueeze(-1) - centers) / max(sigma, 1e-6)).square())
        weights = weights * mask.unsqueeze(-1)
        hist = weights.sum(dim=2)
        return hist / hist.sum(dim=2, keepdim=True).clamp_min(1e-6)

    def masked_pair_hist_loss(
        self,
        pred_features: torch.Tensor,
        pred_mask: torch.Tensor,
        target_features: torch.Tensor,
        target_mask: torch.Tensor,
        bins: int,
        value_min: float,
        value_max: float,
    ) -> torch.Tensor:
        pred_hist = self.masked_soft_histogram(pred_features, pred_mask, bins, value_min, value_max)
        target_hist = self.masked_soft_histogram(target_features, target_mask, bins, value_min, value_max)
        return tnf.l1_loss(pred_hist, target_hist)

    def masked_luma_over_loss(
        self,
        pred_luma: torch.Tensor,
        pred_mask: torch.Tensor,
        target_luma: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_mean, target_std = self.masked_mean_std(target_luma, target_mask)
        upper = target_mean[:, :, None, None] + USER_HAIR_LUMA_OVER_STD * target_std[:, :, None, None]
        upper = upper + USER_HAIR_LUMA_OVER_MARGIN
        pred_mask = pred_mask.float().clamp(0, 1)
        denom = pred_mask.sum().clamp_min(1.0)
        return (torch.relu(pred_luma - upper) * pred_mask).sum() / denom

    @staticmethod
    def masked_gradient_l1(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().clamp(0, 1)
        if mask.size(1) == 1 and source.size(1) != 1:
            mask = mask.expand(-1, source.size(1), -1, -1)

        source_dx = source[:, :, :, 1:] - source[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]

        source_dy = source[:, :, 1:, :] - source[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        loss_x = (torch.abs(source_dx - target_dx) * mask_x).sum() / mask_x.sum().clamp_min(1.0)
        loss_y = (torch.abs(source_dy - target_dy) * mask_y).sum() / mask_y.sum().clamp_min(1.0)
        return loss_x + loss_y

    def calc_loss(
        self,
        i_gen,
        i_face,
        i_color,
        i_base,
        mask_face,
        mask_gen_hair,
        mask_color_ref,
        satd_protect_mask,
        face_keep_mask,
        skin_protect_mask,
        remove_mask,
    ):
        mask_gen_hair = mask_gen_hair.float().clamp(0, 1)
        mask_color_ref = mask_color_ref.float().clamp(0, 1)
        satd_protect_mask = satd_protect_mask.float().clamp(0, 1)
        face_keep_mask = face_keep_mask.float().clamp(0, 1)
        skin_protect_mask = skin_protect_mask.float().clamp(0, 1)
        remove_mask = remove_mask.float().clamp(0, 1)

        gen_face_embed = self.model.get_image_embed(i_gen * mask_face)
        face_embed = self.model.get_image_embed(i_face * mask_face)
        face_loss = (1 - tnf.cosine_similarity(gen_face_embed, face_embed)).mean()

        gen_hair_embed = self.model.get_image_embed(i_gen * mask_gen_hair)
        color_hair_embed = self.model.get_image_embed(i_color * mask_color_ref)
        hair_loss = (1 - tnf.cosine_similarity(gen_hair_embed, color_hair_embed)).mean()

        gen_lab = rgb_to_lab(i_gen)
        color_lab = rgb_to_lab(i_color)
        base_lab = rgb_to_lab(i_base)
        gen_luma = gen_lab[:, 0:1]
        color_luma = color_lab[:, 0:1]
        base_luma = base_lab[:, 0:1]
        gen_chroma = gen_lab[:, 1:3]
        color_chroma = color_lab[:, 1:3]
        base_chroma = base_lab[:, 1:3]
        gen_rgb01 = ((i_gen + 1.0) * 0.5).clamp(0, 1)
        color_rgb01 = ((i_color + 1.0) * 0.5).clamp(0, 1)
        gen_base_hair_mask = self.robust_base_hair_mask(gen_luma, mask_gen_hair)
        color_base_hair_mask = self.robust_base_hair_mask(color_luma, mask_color_ref)

        hair_chroma_stats_loss = self.masked_pair_stats_loss(
            gen_chroma,
            mask_gen_hair,
            color_chroma,
            mask_color_ref,
        )
        hair_chroma_hist_loss = self.masked_pair_hist_loss(
            gen_chroma,
            mask_gen_hair,
            color_chroma,
            mask_color_ref,
            bins=12,
            value_min=-110.0,
            value_max=110.0,
        )
        hair_rgb_stats_loss = self.masked_pair_stats_loss(
            gen_rgb01,
            gen_base_hair_mask,
            color_rgb01,
            color_base_hair_mask,
        )
        hair_lab_mean_loss = self.masked_pair_mean_loss(
            gen_lab,
            gen_base_hair_mask,
            color_lab,
            color_base_hair_mask,
        )
        hair_vertical_lab_mean_loss = self.masked_vertical_pair_mean_loss(
            gen_chroma,
            mask_gen_hair,
            color_chroma,
            mask_color_ref,
        )
        hair_luma_style_loss = self.masked_pair_stats_loss(
            gen_luma,
            gen_base_hair_mask,
            color_luma,
            color_base_hair_mask,
        )
        hair_luma_over_loss = self.masked_luma_over_loss(
            gen_luma,
            mask_gen_hair,
            color_luma,
            mask_color_ref,
        )
        hair_luma_base_grad_loss = self.masked_gradient_l1(gen_luma, base_luma, mask_gen_hair)
        hair_highlight_chroma_loss = self.masked_highlight_chroma_loss(
            gen_chroma,
            gen_luma,
            mask_gen_hair,
            color_chroma,
            color_luma,
            mask_color_ref,
        )
        hair_vivid_coverage_loss = self.masked_vivid_coverage_loss(
            gen_chroma,
            base_chroma,
            mask_gen_hair,
            color_chroma,
            mask_color_ref,
        )

        face_keep_region = (face_keep_mask * satd_protect_mask).clamp(0, 1)
        protect_region = (face_keep_region + remove_mask).clamp(0, 1)
        face_keep_loss = self.masked_l1(i_gen, i_base, face_keep_region)
        remove_keep_loss = self.masked_l1(i_gen, i_base, remove_mask)
        protect_chroma_keep_loss = self.masked_l1(gen_chroma, base_chroma, protect_region)
        skin_chroma_keep_loss = self.masked_l1(gen_chroma, base_chroma, skin_protect_mask)
        skin_rgb_keep_loss = self.masked_l1(i_gen, i_base, skin_protect_mask)

        total_loss = (
            USER_FACE_CLIP_LOSS_WEIGHT * face_loss
            + USER_HAIR_CLIP_LOSS_WEIGHT * hair_loss
            + USER_HAIR_CHROMA_STATS_LOSS_WEIGHT * hair_chroma_stats_loss
            + USER_HAIR_CHROMA_HIST_LOSS_WEIGHT * hair_chroma_hist_loss
            + USER_HAIR_RGB_STATS_LOSS_WEIGHT * hair_rgb_stats_loss
            + USER_HAIR_LAB_MEAN_LOSS_WEIGHT * hair_lab_mean_loss
            + USER_HAIR_VERTICAL_LAB_MEAN_LOSS_WEIGHT * hair_vertical_lab_mean_loss
            + USER_HAIR_LUMA_STYLE_LOSS_WEIGHT * hair_luma_style_loss
            + USER_HAIR_LUMA_OVER_LOSS_WEIGHT * hair_luma_over_loss
            + USER_HAIR_LUMA_BASE_GRAD_LOSS_WEIGHT * hair_luma_base_grad_loss
            + USER_HAIR_HIGHLIGHT_CHROMA_LOSS_WEIGHT * hair_highlight_chroma_loss
            + USER_HAIR_VIVID_COVERAGE_LOSS_WEIGHT * hair_vivid_coverage_loss
            + USER_FACE_KEEP_L1_LOSS_WEIGHT * face_keep_loss
            + USER_REMOVE_KEEP_L1_LOSS_WEIGHT * remove_keep_loss
            + USER_PROTECT_CHROMA_KEEP_LOSS_WEIGHT * protect_chroma_keep_loss
            + USER_SKIN_CHROMA_KEEP_LOSS_WEIGHT * skin_chroma_keep_loss
            + USER_SKIN_RGB_KEEP_LOSS_WEIGHT * skin_rgb_keep_loss
        )
        return total_loss, {
            "face_loss": face_loss,
            "hair_loss": hair_loss,
            "hair_chroma_stats": hair_chroma_stats_loss,
            "hair_chroma_hist": hair_chroma_hist_loss,
            "hair_rgb_stats": hair_rgb_stats_loss,
            "hair_lab_mean": hair_lab_mean_loss,
            "hair_vertical_lab_mean": hair_vertical_lab_mean_loss,
            "hair_luma_style": hair_luma_style_loss,
            "hair_luma_over": hair_luma_over_loss,
            "hair_luma_base_grad": hair_luma_base_grad_loss,
            "hair_highlight_chroma": hair_highlight_chroma_loss,
            "hair_vivid_coverage": hair_vivid_coverage_loss,
            "face_keep_l1": face_keep_loss,
            "remove_keep_l1": remove_keep_loss,
            "protect_chroma_keep": protect_chroma_keep_loss,
            "skin_chroma_keep": skin_chroma_keep_loss,
            "skin_rgb_keep": skin_rgb_keep_loss,
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

    def load_resume_checkpoint(self) -> int:
        if not USER_RESUME_CHECKPOINT:
            return 0

        resume_path = Path(USER_RESUME_CHECKPOINT)
        if not resume_path.exists():
            raise FileNotFoundError(f"Cannot find USER_RESUME_CHECKPOINT: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        loaded_keys = load_compatible_state_dict(self.model, state_dict)

        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, RuntimeError) as exc:
                print(
                    f"[blending_v8] optimizer state is incompatible; reinitializing optimizer. {exc}",
                    file=sys.stderr,
                )

        self.best_loss = float(checkpoint.get("best_loss", self.best_loss))
        start_epoch = int(checkpoint.get("epoch", 0))
        print(
            f"[blending_v8] resumed from {resume_path} "
            f"with {len(loaded_keys)} compatible tensors; "
            f"start_epoch={start_epoch + 1} best_loss={self.best_loss:.6f}",
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

    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        running_steps = 0
        accumulated_batches = 0
        last_grad_norm = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(self.train_loader, desc=f"Blend train {epoch + 1}/{USER_EPOCHS}", leave=False)
        for batch in progress:
            use_author_color_align = (
                USER_AUTHOR_COLOR_ALIGN_BATCH_PROB > 0
                and random.random() < USER_AUTHOR_COLOR_ALIGN_BATCH_PROB
            )
            prepared = self.prepare_batch(batch, use_author_color_align=use_author_color_align)
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
                satd_protect_mask,
                remove_mask,
                color_transfer_mask,
                face_keep_mask,
                skin_protect_mask,
                hm_3e,
                color_ref_mask,
                hm_1d,
                target_hair_full,
            ) = prepared
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * hm_3e)
            latent_in = self.build_generator_latent(align_s, blend_s)
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
                color_transfer_mask,
                color_ref_mask,
                satd_protect_mask,
                face_keep_mask,
                skin_protect_mask,
                remove_mask,
            )

            (loss / self.grad_accum_steps).backward()
            accumulated_batches += 1
            if accumulated_batches == self.grad_accum_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                last_grad_norm = float(grad_norm)

            running_loss += float(loss.item())
            running_steps += 1
            progress.set_postfix(
                loss=float(loss.item()),
                chroma=float(loss_info["hair_chroma_stats"].item()),
                hist=float(loss_info["hair_chroma_hist"].item()),
                rgb=float(loss_info["hair_rgb_stats"].item()),
                over=float(loss_info["hair_luma_over"].item()),
                highlight=float(loss_info["hair_highlight_chroma"].item()),
                vivid=float(loss_info["hair_vivid_coverage"].item()),
                vertical=float(loss_info["hair_vertical_lab_mean"].item()),
                skin=float(loss_info["skin_chroma_keep"].item()),
                branch="color" if use_author_color_align else "shape",
                grad=last_grad_norm,
                accum=f"{accumulated_batches}/{self.grad_accum_steps}",
            )

        if accumulated_batches:
            scale = self.grad_accum_steps / accumulated_batches
            if scale != 1.0:
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), USER_GRAD_CLIP)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return running_loss / max(running_steps, 1)

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        total_losses = {
            "face_loss": 0.0,
            "hair_loss": 0.0,
            "hair_chroma_stats": 0.0,
            "hair_chroma_hist": 0.0,
            "hair_rgb_stats": 0.0,
            "hair_lab_mean": 0.0,
            "hair_vertical_lab_mean": 0.0,
            "hair_luma_style": 0.0,
            "hair_luma_over": 0.0,
            "hair_luma_base_grad": 0.0,
            "hair_highlight_chroma": 0.0,
            "hair_vivid_coverage": 0.0,
            "face_keep_l1": 0.0,
            "remove_keep_l1": 0.0,
            "protect_chroma_keep": 0.0,
            "skin_chroma_keep": 0.0,
            "skin_rgb_keep": 0.0,
            "loss": 0.0,
        }
        total_steps = 0
        images_to_fid = []
        preview_rows = []

        for batch in tqdm(self.val_loader, desc=f"Blend val {epoch + 1}/{USER_EPOCHS}", leave=False):
            prepared = self.prepare_batch(batch, use_author_color_align=False)
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
                satd_protect_mask,
                remove_mask,
                color_transfer_mask,
                face_keep_mask,
                skin_protect_mask,
                hm_3e,
                color_ref_mask,
                hm_1d,
                target_hair_full,
            ) = prepared
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * hm_3e)
            latent_in = self.build_generator_latent(align_s, blend_s)
            i_g, _ = self.helper.net.generator(
                [latent_in],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=align_f,
            )
            i_g_256 = self.helper.downsample_256(i_g)
            _, _, generated_hair_256 = self.helper.generate_mask(i_g, return_raw=True)
            i_fixed_256 = self.complete_validation_image(
                i_g_256,
                face_i,
                color_i,
                hm_1d,
                target_hair_full,
                hm_3e,
                face_keep_mask,
                skin_protect_mask,
                remove_mask,
                generated_hair_256,
            )
            loss, loss_info = self.calc_loss(
                i_fixed_256,
                face_i,
                color_i,
                i_x_256,
                target_mask,
                color_transfer_mask,
                color_ref_mask,
                satd_protect_mask,
                face_keep_mask,
                skin_protect_mask,
                remove_mask,
            )

            for key, value in loss_info.items():
                total_losses[key] = total_losses.get(key, 0.0) + float(value.item())
            total_steps += 1

            if self.fid_calc is not None:
                images_to_fid.append(T.Resize((299, 299))(((i_g + 1) / 2).clamp(0, 1)))

            if len(preview_rows) < USER_LOG_IMAGE_COUNT:
                for idx in range(bsz):
                    preview_rows.append([
                        face_i[idx : idx + 1],
                        color_i[idx : idx + 1],
                        i_x_256[idx : idx + 1],
                        i_fixed_256[idx : idx + 1],
                        mask_to_preview(target_hair_full[idx : idx + 1]),
                    ])
                    if len(preview_rows) >= USER_LOG_IMAGE_COUNT:
                        break

        avg_losses = {key: value / max(total_steps, 1) for key, value in total_losses.items()}
        if self.fid_calc is not None and images_to_fid:
            avg_losses["fid_clip"] = float(self.fid_calc(torch.cat(images_to_fid)).item())

        if epoch % USER_SAVE_PREVIEW_EVERY == 0:
            epoch_dir = self.output_val_dir / f"epoch_{epoch + 1:03d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            for idx, row in enumerate(preview_rows):
                save_preview(epoch_dir / f"sample_{idx:03d}.png", row)

        print(
            f"[blending_v8] epoch={epoch + 1} "
            f"val_loss={avg_losses['loss']:.6f} "
            f"val_face={avg_losses['face_loss']:.6f} "
            f"val_hair={avg_losses['hair_loss']:.6f} "
            f"val_chroma={avg_losses['hair_chroma_stats']:.6f} "
            f"val_hist={avg_losses['hair_chroma_hist']:.6f} "
            f"val_rgb={avg_losses['hair_rgb_stats']:.6f} "
            f"val_lab={avg_losses['hair_lab_mean']:.6f} "
            f"val_vertical={avg_losses['hair_vertical_lab_mean']:.6f} "
            f"val_luma={avg_losses['hair_luma_style']:.6f} "
            f"val_over={avg_losses['hair_luma_over']:.6f} "
            f"val_highlight={avg_losses['hair_highlight_chroma']:.6f} "
            f"val_vivid={avg_losses['hair_vivid_coverage']:.6f} "
            f"val_skin={avg_losses['skin_chroma_keep']:.6f}"
        )
        return avg_losses["loss"]

    def train_loop(self):
        start_epoch = self.load_resume_checkpoint()
        if start_epoch >= USER_EPOCHS:
            print(
                f"[blending_v8] resume checkpoint is already at epoch {start_epoch}; "
                f"USER_EPOCHS={USER_EPOCHS}, nothing to train.",
                file=sys.stderr,
            )
            return

        for epoch in range(start_epoch, USER_EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            print(f"[blending_v8] epoch={epoch + 1} train_loss={train_loss:.6f}")

            is_best = val_loss <= self.best_loss
            if is_best:
                self.best_loss = val_loss
            if (epoch + 1) % USER_SAVE_CHECKPOINT_EVERY == 0:
                self.save_checkpoint(epoch + 1, self.best_loss, "last")
            if is_best:
                self.save_checkpoint(epoch + 1, self.best_loss, "best")
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
    train_exps, val_exps = train_test_split(triplets, test_size=ACTIVE_VAL_SIZE, random_state=USER_RANDOM_SEED)
    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    helper = MaskPrepHelper(device)

    train_dataset = BlendingDatasetV8(train_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_COLOR_ROOT)
    val_dataset = BlendingDatasetV8(val_exps, ACTIVE_DATASET_DIR, ACTIVE_FACE_ROOT, ACTIVE_COLOR_ROOT)
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
    init_ckpt = resolve_init_blending_checkpoint()
    ckpt = torch.load(init_ckpt, map_location=device)
    loaded_keys = load_compatible_state_dict(model, ckpt.get("model_state_dict", ckpt))
    print(f"[blending_v8] initialized from {init_ckpt} with {len(loaded_keys)} compatible tensors", file=sys.stderr)
    optimizer = torch.optim.Adam(model.parameters(), lr=USER_LR, weight_decay=USER_WEIGHT_DECAY)

    trainer = BlendingTrainerV8(model, optimizer, train_loader, val_loader, helper)
    print(
        f"[blending_v8] train_on_shape_satd_align=True "
        f"author_color_align_aux={USER_AUTHOR_COLOR_ALIGN_BATCH_PROB > 0} use_satd_v8={USER_USE_SATD_V8} "
        f"satd_checkpoint={USER_SATD_CHECKPOINT_V8} "
        f"batch_size={USER_BATCH_SIZE} grad_accum_steps={USER_GRAD_ACCUM_STEPS} "
        f"effective_batch_size={USER_BATCH_SIZE * USER_GRAD_ACCUM_STEPS} "
        f"hair_chroma_stats_w={USER_HAIR_CHROMA_STATS_LOSS_WEIGHT} "
        f"hair_chroma_hist_w={USER_HAIR_CHROMA_HIST_LOSS_WEIGHT} "
        f"hair_rgb_stats_w={USER_HAIR_RGB_STATS_LOSS_WEIGHT} "
        f"hair_lab_mean_w={USER_HAIR_LAB_MEAN_LOSS_WEIGHT} "
        f"hair_vertical_lab_mean_w={USER_HAIR_VERTICAL_LAB_MEAN_LOSS_WEIGHT} "
        f"hair_luma_over_w={USER_HAIR_LUMA_OVER_LOSS_WEIGHT} "
        f"hair_highlight_chroma_w={USER_HAIR_HIGHLIGHT_CHROMA_LOSS_WEIGHT} "
        f"hair_vivid_coverage_w={USER_HAIR_VIVID_COVERAGE_LOSS_WEIGHT} "
        f"remove_keep_w={USER_REMOVE_KEEP_L1_LOSS_WEIGHT} "
        f"protect_chroma_keep_w={USER_PROTECT_CHROMA_KEEP_LOSS_WEIGHT} "
        f"skin_chroma_keep_w={USER_SKIN_CHROMA_KEEP_LOSS_WEIGHT} "
        f"skin_rgb_keep_w={USER_SKIN_RGB_KEEP_LOSS_WEIGHT} "
        f"target_hair_neck_override={USER_TARGET_HAIR_NECK_OVERRIDE} "
        f"author_color_align_batch_prob={USER_AUTHOR_COLOR_ALIGN_BATCH_PROB} "
        f"author_zero_prefix_train={USER_AUTHOR_ZERO_PREFIX_TRAIN} "
        f"resume_checkpoint={USER_RESUME_CHECKPOINT or '<none>'}",
        file=sys.stderr,
    )
    trainer.train_loop()


if __name__ == "__main__":
    main()
