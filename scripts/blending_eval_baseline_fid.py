import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap import HairFast, get_parser
from models.Encoders import ClipBlendingModel as BlendingModel
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion, list_image_files
from utils.train import get_fid_calc, toggle_grad


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "celebahq_fixed3000"
USER_REFERENCE_MODE = "both"  # "both" or "full"

USER_FACE_ROOT_CELEBAHQ = Path("images/CelebA-HQ")
USER_SHAPE_ROOT_CELEBAHQ = Path("images/CelebA-HQ")
USER_COLOR_ROOT_CELEBAHQ = Path("images/CelebA-HQ")
USER_FID_REFERENCE_ROOT_CELEBAHQ = Path("images/CelebA-HQ")
USER_OUTPUT_DIR_CELEBAHQ = Path("output/blending_eval_baseline_fid_celebahq_fixed3000")
USER_SAMPLE_SIZE_CELEBAHQ = 3000

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_SAVE_PANEL = True
USER_ALLOW_REUSE_ACROSS_TRIPLETS = False

USER_BLENDING_CHECKPOINT = Path("pretrained_models/Blending/checkpoint.pth")
USER_CLIP_MODEL = "ViT-B/32"

USER_SAVE_PREVIEW = True
USER_LOG_IMAGE_COUNT = 16
# ============================================================================


def resolve_profile_defaults():
    if USER_DATASET_PROFILE == "celebahq_fixed3000":
        return {
            "face_root": USER_FACE_ROOT_CELEBAHQ,
            "shape_root": USER_SHAPE_ROOT_CELEBAHQ,
            "color_root": USER_COLOR_ROOT_CELEBAHQ,
            "fid_reference_root": USER_FID_REFERENCE_ROOT_CELEBAHQ,
            "output_dir": USER_OUTPUT_DIR_CELEBAHQ,
            "sample_size": USER_SAMPLE_SIZE_CELEBAHQ,
        }
    raise ValueError(f"Unsupported USER_DATASET_PROFILE: {USER_DATASET_PROFILE}")


PROFILE = resolve_profile_defaults()
ACTIVE_FACE_ROOT = PROFILE["face_root"]
ACTIVE_SHAPE_ROOT = PROFILE["shape_root"]
ACTIVE_COLOR_ROOT = PROFILE["color_root"]
ACTIVE_FID_REFERENCE_ROOT = PROFILE["fid_reference_root"]
ACTIVE_OUTPUT_DIR = PROFILE["output_dir"]
ACTIVE_SAMPLE_SIZE = PROFILE["sample_size"]


def set_seed(seed: int):
    random.seed(seed)
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


def identity_func(align_shape, align_color, name_to_embed, **kwargs):
    return align_shape, align_color, name_to_embed


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


def _sample_excluding_stems(rng: random.Random, candidates: list[str], forbidden_stems: set[str]) -> str:
    valid = [item for item in candidates if Path(item).stem not in forbidden_stems]
    if not valid:
        raise RuntimeError("No valid candidates left after excluding duplicate stems.")
    return rng.choice(valid)


def _assert_unique_stems(images: list[str], label: str):
    counts: dict[str, int] = {}
    for item in images:
        stem = Path(item).stem
        counts[stem] = counts.get(stem, 0) + 1

    duplicates = [stem for stem, count in counts.items() if count > 1]
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:10])
        raise RuntimeError(
            f"{label} contains duplicate stems inside the same directory, which would make loading ambiguous: {sample}"
        )


def sample_triplets(
    face_images: list[str],
    shape_images: list[str],
    color_images: list[str],
    size: int,
    allow_reuse: bool,
    seed: int,
    reference_mode: str,
) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    triplets: list[tuple[str, str, str]] = []
    reference_mode = str(reference_mode).lower()

    if not face_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_FACE_ROOT}")
    if not shape_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_SHAPE_ROOT}")
    if not color_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_COLOR_ROOT}")
    if reference_mode not in {"both", "full"}:
        raise RuntimeError(f"Unsupported USER_REFERENCE_MODE={reference_mode!r}. Choose one of: both, full.")

    max_needed = max(len(face_images), len(shape_images), len(color_images))
    if reference_mode == "both":
        max_needed = max(len(face_images), len(shape_images))

    if not allow_reuse and size > max_needed:
        raise RuntimeError("Not enough unique images to sample all triplets without reuse.")

    face_pool = face_images.copy()
    shape_pool = shape_images.copy()
    color_pool = color_images.copy()

    for _ in range(size):
        if allow_reuse:
            face = rng.choice(face_images)
            if reference_mode == "both":
                reference = _sample_excluding_stems(rng, shape_images, {Path(face).stem})
                shape = reference
                color = reference
            else:
                shape = _sample_excluding_stems(rng, shape_images, {Path(face).stem})
                color = _sample_excluding_stems(rng, color_images, {Path(face).stem, Path(shape).stem})
        else:
            if not face_pool or not shape_pool or (reference_mode == "full" and not color_pool):
                raise RuntimeError("The image pool has been exhausted. Reduce ACTIVE_SAMPLE_SIZE or allow reuse.")
            face = rng.choice(face_pool)
            if reference_mode == "both":
                reference = _sample_excluding_stems(rng, shape_pool, {Path(face).stem})
                shape = reference
                color = reference
            else:
                shape = _sample_excluding_stems(rng, shape_pool, {Path(face).stem})
                color = _sample_excluding_stems(rng, color_pool, {Path(face).stem, Path(shape).stem})
            face_pool.remove(face)
            shape_pool.remove(shape)
            if reference_mode == "full":
                color_pool.remove(color)

        triplets.append((Path(face).stem, Path(shape).stem, Path(color).stem))

    return triplets


def resolve_image_path(root: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for stem={stem!r} under {root}")


def load_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return T.ToTensor()(image.convert("RGB"))


def load_normalized_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = T.functional.to_tensor(image.convert("RGB"))
    return T.functional.normalize(tensor, [0.5], [0.5])


def ensure_chw(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 4:
        image = image[0]
    return image.detach().cpu().clamp(0.0, 1.0)


def resize_like(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(image.shape[-2:]) == (height, width):
        return image
    return F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]


def save_panel(path: Path, source: torch.Tensor, shape: torch.Tensor, color: torch.Tensor, result: torch.Tensor) -> None:
    result = ensure_chw(result)
    height, width = result.shape[-2:]
    source = resize_like(ensure_chw(source), height, width)
    shape = resize_like(ensure_chw(shape), height, width)
    color = resize_like(ensure_chw(color), height, width)
    panel = torch.cat([source, shape, color, result], dim=2)
    save_image(panel, path)


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
            argparse.Namespace(
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


def build_alignment_model() -> HairFast:
    if not USER_BLENDING_CHECKPOINT.exists():
        raise FileNotFoundError(f"Cannot find blending checkpoint: {USER_BLENDING_CHECKPOINT}")

    model_args = get_parser().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    model_args.blending_checkpoint = str(USER_BLENDING_CHECKPOINT)
    model_args.use_shadow_cleanup = False

    hair_fast = HairFast(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)
    return hair_fast


def build_blending_model(device: torch.device) -> BlendingModel:
    if not USER_BLENDING_CHECKPOINT.exists():
        raise FileNotFoundError(f"Cannot find blending checkpoint: {USER_BLENDING_CHECKPOINT}")

    checkpoint = torch.load(USER_BLENDING_CHECKPOINT, map_location=device)
    clip_model = checkpoint.get("clip", USER_CLIP_MODEL)
    model = BlendingModel(clip_model)
    load_compatible_state_dict(model, checkpoint.get("model_state_dict", checkpoint))
    return model.to(device).eval()


def calc_loss(model: BlendingModel, i_gen, i_face, i_color, mask_face, mask_hair):
    gen_embed = model.get_image_embed(i_gen * mask_face)
    gt_embed = model.get_image_embed(i_face * mask_face)
    face_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

    gen_embed = model.get_image_embed(i_gen * mask_hair)
    gt_embed = model.get_image_embed(i_color * mask_hair)
    hair_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()
    total_loss = face_loss + hair_loss
    return total_loss, {"face_loss": face_loss, "hair_loss": hair_loss, "loss": total_loss}


@torch.no_grad()
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    set_seed(USER_RANDOM_SEED)

    face_images = list_image_files(ACTIVE_FACE_ROOT)
    shape_images = list_image_files(ACTIVE_SHAPE_ROOT)
    color_images = list_image_files(ACTIVE_COLOR_ROOT)
    _assert_unique_stems(face_images, "ACTIVE_FACE_ROOT")
    _assert_unique_stems(shape_images, "ACTIVE_SHAPE_ROOT")
    _assert_unique_stems(color_images, "ACTIVE_COLOR_ROOT")

    triplets = sample_triplets(
        face_images,
        shape_images,
        color_images,
        ACTIVE_SAMPLE_SIZE,
        USER_ALLOW_REUSE_ACROSS_TRIPLETS,
        USER_RANDOM_SEED,
        USER_REFERENCE_MODE,
    )
    if not triplets:
        raise RuntimeError("No triplets selected for evaluation.")

    mode_tag = str(USER_REFERENCE_MODE).lower()
    result_dir = ACTIVE_OUTPUT_DIR / mode_tag / "results"
    real_dir = ACTIVE_OUTPUT_DIR / mode_tag / "real_images"
    panel_dir = ACTIVE_OUTPUT_DIR / mode_tag / "panels"
    preview_dir = ACTIVE_OUTPUT_DIR / mode_tag / "val_images"
    metrics_path = ACTIVE_OUTPUT_DIR / mode_tag / "metrics.json"
    manifest_path = ACTIVE_OUTPUT_DIR / mode_tag / "export_manifest.jsonl"
    fid_cache_path = ACTIVE_OUTPUT_DIR / mode_tag / "fid_reference_full_celebahq.pkl"

    result_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_PANEL:
        panel_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    helper = MaskPrepHelper(device)
    alignment_model = build_alignment_model()
    blending_model = build_blending_model(device)

    metrics = {"face_loss": 0.0, "hair_loss": 0.0, "loss": 0.0}
    total_steps = 0
    images_to_fid = []
    preview_rows = []
    manifest_lines = []

    for index, (face_stem, shape_stem, color_stem) in enumerate(
        tqdm(triplets, desc=f"Baseline blending eval ({mode_tag})"),
        start=1,
    ):
        face_path = resolve_image_path(ACTIVE_FACE_ROOT, face_stem)
        shape_path = resolve_image_path(ACTIVE_SHAPE_ROOT, shape_stem)
        color_path = resolve_image_path(ACTIVE_COLOR_ROOT, color_stem)

        align_shape, align_color, name_to_embed = alignment_model(
            face_path,
            shape_path,
            color_path,
            align_flag=True,
            use_shadow_cleanup=False,
        )

        color_s = name_to_embed["color"]["S"].to(device)
        align_s = name_to_embed["face"]["S"].to(device)
        align_f = align_color["latent_F_align"].to(device)

        face_i = load_normalized_tensor(face_path).unsqueeze(0).to(device)
        color_i = load_normalized_tensor(color_path).unsqueeze(0).to(device)

        hm_3d, hm_3e = helper.generate_mask(color_i)
        hm_1d, _ = helper.generate_mask(face_i)
        i_x, _ = helper.net.generator(
            [align_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=align_f,
        )
        hm_xd, hm_xe = helper.generate_mask(i_x)

        if not hm_3e.flatten(1).any(dim=1).item() or not hm_xe.flatten(1).any(dim=1).item():
            continue

        target_mask = (1 - hm_1d) * (1 - hm_3d) * (1 - hm_xd)
        face_i_256 = helper.downsample_256(face_i)
        color_i_256 = helper.downsample_256(color_i)

        blend_s = blending_model(align_s[:, 6:], color_s[:, 6:], face_i_256 * target_mask, color_i_256 * hm_3e)
        latent_in = torch.cat((torch.zeros(1, 6, 512, device=device), blend_s), dim=1)
        i_g, _ = helper.net.generator(
            [latent_in],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=align_f,
        )
        i_g_256 = helper.downsample_256(i_g)

        _, loss_info = calc_loss(blending_model, i_g_256, face_i_256, color_i_256, target_mask, hm_3e)
        for key, value in loss_info.items():
            metrics[key] += float(value.item())
        total_steps += 1

        images_to_fid.append(T.Resize((299, 299))(((i_g + 1) / 2).clamp(0, 1)).cpu())

        result_name = f"{index:04d}__{face_stem}__{shape_stem}__{color_stem}.png"
        real_name = f"{index:04d}__{face_stem}.png"
        result_path = result_dir / result_name
        real_path = real_dir / real_name
        source_tensor = load_tensor(face_path)
        save_image(ensure_chw(i_g), result_path)
        save_image(ensure_chw(source_tensor), real_path)

        if USER_SAVE_PANEL:
            save_panel(
                panel_dir / result_name,
                source_tensor,
                load_tensor(shape_path),
                load_tensor(color_path),
                i_g,
            )

        if USER_SAVE_PREVIEW and len(preview_rows) < USER_LOG_IMAGE_COUNT:
            preview_rows.append([face_i_256[0:1], color_i_256[0:1], i_g_256[0:1]])

        manifest_lines.append(
            json.dumps(
                {
                    "index": index,
                    "sample_id": f"{face_stem}__{shape_stem}__{color_stem}",
                    "source_path": str(face_path),
                    "shape_path": str(shape_path),
                    "color_path": str(color_path),
                    "real_path": real_path.as_posix(),
                    "result_path": result_path.as_posix(),
                },
                ensure_ascii=True,
            )
            + "\n"
        )

    if total_steps == 0:
        raise RuntimeError("No valid samples were produced during baseline blending evaluation.")

    for key in metrics:
        metrics[key] /= total_steps

    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.writelines(manifest_lines)

    fid_calc = get_fid_calc(str(fid_cache_path), str(ACTIVE_FID_REFERENCE_ROOT), device=device)
    metrics["fid_clip"] = float(fid_calc(torch.cat(images_to_fid, dim=0)).item())

    if USER_SAVE_PREVIEW:
        preview_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(preview_rows):
            save_preview(preview_dir / f"sample_{idx:03d}.png", row)

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=True, indent=2)

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"reference mode: {mode_tag}")
    print(f"face root: {ACTIVE_FACE_ROOT}")
    print(f"shape root: {ACTIVE_SHAPE_ROOT}")
    print(f"color root: {ACTIVE_COLOR_ROOT}")
    print(f"fid reference root: {ACTIVE_FID_REFERENCE_ROOT}")
    print(f"fixed sample size: {len(triplets)}")
    print(f"valid evaluated samples: {total_steps}")
    print(f"results dir: {result_dir}")
    print(f"real dir: {real_dir}")
    print(f"manifest: {manifest_path}")
    print(f"blending checkpoint: {USER_BLENDING_CHECKPOINT}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"metrics saved to: {metrics_path}")
    if USER_SAVE_PANEL:
        print(f"panels dir: {panel_dir}")


if __name__ == "__main__":
    main()
