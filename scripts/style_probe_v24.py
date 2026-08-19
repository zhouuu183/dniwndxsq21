import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v22 import HairFast_v22, get_parser_v22
from models.Net import get_segmentation
from utils.image_utils import equal_replacer, list_image_files
from utils.mask_delta_v8 import filter_parsing_to_primary_subject


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")

USER_FACE_ROOT_SMALL = Path("images/FFHQ_long")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_short")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ_color")

USER_OUTPUT_DIR = Path("output/style_probe_v24")
USER_SAMPLE_COUNT = 12
USER_RANDOM_SEED = 3407
USER_DEVICE = "cuda"

# Optional explicit triplets by stem, e.g. [("00001", "00002", "00003")].
# If empty, the script samples from the selected profile roots.
USER_TRIPLETS: list[tuple[str, str, str]] = []
USER_ALLOW_REUSE_ACROSS_TRIPLETS = True

USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = "output/satd_train_v8_small/checkpoints/satd_for_infer_v8.pth"
USER_SATD_BLEND_V8 = 0.34
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0

# Tail mixing means: S_new[:, start_layer:] = lerp(face_S, color_S, alpha).
USER_MIX_START_LAYERS = (6, 8, 10, 12, 14)
USER_MIX_ALPHAS = (0.15, 0.30, 0.45, 0.60)
USER_HARD_PRESERVE_NON_HAIR = True
USER_HAIR_MASK_SOURCE = "satd_or_hmx"  # satd, hmx, satd_or_hmx
USER_ALPHA_BLUR_RADIUS = 3
USER_PANEL_TILE_SIZE = 192
USER_SAVE_RAW_GENERATIONS = False
# ============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_profile() -> dict[str, Path]:
    profiles = {
        "ffhq": {
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
        },
        "small": {
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "color_root": USER_COLOR_ROOT_SMALL,
        },
    }
    if USER_DATASET_PROFILE not in profiles:
        raise RuntimeError(
            f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}. "
            f"Choose one of: {', '.join(sorted(profiles))}."
        )
    return profiles[USER_DATASET_PROFILE]


PROFILE = resolve_dataset_profile()
ACTIVE_FACE_ROOT = PROFILE["face_root"]
ACTIVE_SHAPE_ROOT = PROFILE["shape_root"]
ACTIVE_COLOR_ROOT = PROFILE["color_root"]


def image_stem(file_name: str) -> str:
    return Path(file_name).stem


def find_image_path(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem}.png/.jpg/.jpeg in {root}")


def sample_excluding(rng: random.Random, pool: list[str], forbidden_stems: set[str]) -> str:
    candidates = [item for item in pool if image_stem(item) not in forbidden_stems]
    if not candidates:
        raise RuntimeError("No candidate left after excluding same-stem images.")
    return rng.choice(candidates)


def sample_triplets() -> list[tuple[str, str, str]]:
    if USER_TRIPLETS:
        return USER_TRIPLETS

    face_files = list_image_files(ACTIVE_FACE_ROOT)
    shape_files = list_image_files(ACTIVE_SHAPE_ROOT)
    color_files = list_image_files(ACTIVE_COLOR_ROOT)
    if not face_files or not shape_files or not color_files:
        raise RuntimeError("One of the selected input roots is empty.")

    rng = random.Random(USER_RANDOM_SEED)
    face_pool = face_files.copy()
    shape_pool = shape_files.copy()
    color_pool = color_files.copy()
    triplets: list[tuple[str, str, str]] = []

    for _ in range(USER_SAMPLE_COUNT):
        if USER_ALLOW_REUSE_ACROSS_TRIPLETS:
            face = rng.choice(face_files)
            shape = sample_excluding(rng, shape_files, {image_stem(face)})
            color = sample_excluding(rng, color_files, {image_stem(face), image_stem(shape)})
        else:
            if not face_pool or not shape_pool or not color_pool:
                raise RuntimeError("The image pool has been exhausted.")
            face = rng.choice(face_pool)
            shape = sample_excluding(rng, shape_pool, {image_stem(face)})
            color = sample_excluding(rng, color_pool, {image_stem(face), image_stem(shape)})
            face_pool.remove(face)
            shape_pool.remove(shape)
            color_pool.remove(color)
        triplets.append((image_stem(face), image_stem(shape), image_stem(color)))
    return triplets


def build_model() -> HairFast_v22:
    args = get_parser_v22().parse_args([])
    args.device = USER_DEVICE
    args.save_all = False
    args.use_satd_v8 = bool(USER_USE_SATD_V8)
    args.satd_checkpoint_v8 = USER_SATD_CHECKPOINT_V8
    args.satd_blend_v8 = USER_SATD_BLEND_V8
    args.satd_boundary_v8 = USER_SATD_BOUNDARY_V8
    args.eq8_reference_blend_v8 = USER_EQ8_REFERENCE_BLEND_V8
    return HairFast_v22(args)


def load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return T.functional.to_tensor(image.convert("RGB"))


def gaussian_blur2d(image: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return image
    sigma = max(radius / 2.0, 1e-4)
    size = 2 * radius + 1
    coords = torch.arange(size, device=image.device, dtype=image.dtype) - radius
    kernel_1d = torch.exp(-0.5 * (coords / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)
    kernel_x = kernel_1d.view(1, 1, 1, size).expand(image.size(1), 1, 1, size)
    kernel_y = kernel_1d.view(1, 1, size, 1).expand(image.size(1), 1, size, 1)
    image = F.conv2d(image, kernel_x, padding=(0, radius), groups=image.size(1))
    return F.conv2d(image, kernel_y, padding=(radius, 0), groups=image.size(1))


def mask_to_rgb(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask.float().clamp(0, 1).repeat(1, 3, 1, 1)


def resize_for_panel(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.shape[-2:] == (size, size):
        return image
    return F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False).clamp(0, 1)


def tensor_to_pil(image: torch.Tensor, tile_size: int) -> Image.Image:
    image = resize_for_panel(image, tile_size)[0].detach().cpu().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def make_labeled_panel(rows: list[list[tuple[str, torch.Tensor]]], path: Path, tile_size: int) -> None:
    label_h = 22
    cols = max(len(row) for row in rows)
    canvas = Image.new("RGB", (cols * tile_size, len(rows) * (tile_size + label_h)), "black")
    draw = ImageDraw.Draw(canvas)

    for row_idx, row in enumerate(rows):
        y = row_idx * (tile_size + label_h)
        for col_idx, (label, tensor) in enumerate(row):
            x = col_idx * tile_size
            draw.rectangle([x, y, x + tile_size, y + label_h], fill=(20, 20, 20))
            draw.text((x + 4, y + 4), label[:28], fill=(235, 235, 235))
            canvas.paste(tensor_to_pil(tensor, tile_size), (x, y + label_h))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_target_hair_alpha(i_satd_01: torch.Tensor, hm_x_256: torch.Tensor) -> torch.Tensor:
    hm_x = hm_x_256.float().clamp(0, 1)
    if USER_HAIR_MASK_SOURCE in {"satd", "satd_or_hmx"}:
        satd_mask, _ = filter_parsing_to_primary_subject(get_segmentation(i_satd_01, resize=True))
        hm_satd = torch.where(satd_mask == 13, torch.ones_like(satd_mask), torch.zeros_like(satd_mask)).float()
    else:
        hm_satd = torch.zeros_like(hm_x)

    if USER_HAIR_MASK_SOURCE == "satd":
        hair_256 = hm_satd
    elif USER_HAIR_MASK_SOURCE == "hmx":
        hair_256 = hm_x
    elif USER_HAIR_MASK_SOURCE == "satd_or_hmx":
        hair_256 = (hm_satd + hm_x).clamp(0, 1)
    else:
        raise RuntimeError(f"Unsupported USER_HAIR_MASK_SOURCE={USER_HAIR_MASK_SOURCE!r}")

    alpha = F.interpolate(hair_256, size=i_satd_01.shape[-2:], mode="nearest").clamp(0, 1)
    if USER_ALPHA_BLUR_RADIUS > 0:
        alpha = gaussian_blur2d(alpha, USER_ALPHA_BLUR_RADIUS).clamp(0, 1)
    return alpha


@torch.inference_mode()
def run_one_probe(
    hair_fast: HairFast_v22,
    face_path: Path,
    shape_path: Path,
    color_path: Path,
    sample_dir: Path,
) -> None:
    face_tensor = load_rgb_tensor(face_path)
    shape_tensor = load_rgb_tensor(shape_path)
    color_tensor = load_rgb_tensor(color_path)
    images = equal_replacer([face_tensor, shape_tensor, color_tensor])

    images_to_name: dict[torch.Tensor, list[str]] = defaultdict(list)
    for image, name in zip(images, ("face", "shape", "color")):
        images_to_name[image].append(name)

    name_to_embed = hair_fast.embed.embedding_images(images_to_name)
    align_shape = hair_fast.align.align_images(
        "face",
        "shape",
        name_to_embed,
        use_satd_v8=USER_USE_SATD_V8,
        satd_blend_v8=USER_SATD_BLEND_V8,
        satd_boundary_v8=USER_SATD_BOUNDARY_V8,
        eq8_reference_blend_v8=USER_EQ8_REFERENCE_BLEND_V8,
    )

    face_s = name_to_embed["face"]["S"]
    color_s = name_to_embed["color"]["S"]
    latent_f_align = align_shape["latent_F_align"]

    i_satd, _ = hair_fast.net.generator(
        [face_s],
        input_is_latent=True,
        return_latents=False,
        start_layer=4,
        end_layer=8,
        layer_in=latent_f_align,
    )
    i_satd_01 = ((i_satd + 1.0) * 0.5).clamp(0, 1)
    alpha = build_target_hair_alpha(i_satd_01, align_shape["HM_X"])
    mask_preview = mask_to_rgb(alpha)

    face_ref = name_to_embed["face"]["image_norm_256"] * 0.5 + 0.5
    shape_ref = name_to_embed["shape"]["image_norm_256"] * 0.5 + 0.5
    color_ref = name_to_embed["color"]["image_norm_256"] * 0.5 + 0.5

    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image(i_satd_01[0], sample_dir / "satd.png")
    save_image(color_ref[0], sample_dir / "color_ref.png")
    save_image(mask_preview[0], sample_dir / "target_hair_alpha.png")

    rows: list[list[tuple[str, torch.Tensor]]] = [
        [
            ("face", face_ref),
            ("shape", shape_ref),
            ("color", color_ref),
            ("satd", i_satd_01),
            ("hair_alpha", mask_preview),
        ]
    ]

    max_layers = face_s.size(1)
    for start_layer in USER_MIX_START_LAYERS:
        if start_layer < 0 or start_layer >= max_layers:
            continue

        row: list[tuple[str, torch.Tensor]] = [
            (f"tail S[{start_layer}:]", i_satd_01),
            ("color", color_ref),
            ("alpha", mask_preview),
        ]
        for alpha_value in USER_MIX_ALPHAS:
            mixed_s = face_s.clone()
            mixed_s[:, start_layer:] = torch.lerp(face_s[:, start_layer:], color_s[:, start_layer:], float(alpha_value))
            i_gen, _ = hair_fast.net.generator(
                [mixed_s],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_f_align,
            )
            i_gen_01 = ((i_gen + 1.0) * 0.5).clamp(0, 1)
            if USER_HARD_PRESERVE_NON_HAIR:
                i_out_01 = (i_gen_01 * alpha + i_satd_01 * (1.0 - alpha)).clamp(0, 1)
            else:
                i_out_01 = i_gen_01

            stem = f"S{start_layer:02d}_a{alpha_value:.2f}".replace(".", "p")
            save_image(i_out_01[0], sample_dir / f"{stem}.png")
            if USER_SAVE_RAW_GENERATIONS:
                save_image(i_gen_01[0], sample_dir / f"{stem}_raw.png")
            row.append((f"S{start_layer} a{alpha_value:.2f}", i_out_01))
        rows.append(row)

    make_labeled_panel(rows, sample_dir / "panel.png", USER_PANEL_TILE_SIZE)


@torch.inference_mode()
def main() -> None:
    set_seed(USER_RANDOM_SEED)
    USER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    triplets = sample_triplets()
    hair_fast = build_model()

    manifest_path = USER_OUTPUT_DIR / "manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        print(f"profile={USER_DATASET_PROFILE}", file=manifest)
        print(f"mix_start_layers={USER_MIX_START_LAYERS}", file=manifest)
        print(f"mix_alphas={USER_MIX_ALPHAS}", file=manifest)
        print(f"hard_preserve_non_hair={USER_HARD_PRESERVE_NON_HAIR}", file=manifest)
        print(f"hair_mask_source={USER_HAIR_MASK_SOURCE}", file=manifest)
        print("", file=manifest)

        for idx, (face_name, shape_name, color_name) in enumerate(tqdm(triplets, desc="Style probe v24")):
            face_path = find_image_path(ACTIVE_FACE_ROOT, face_name)
            shape_path = find_image_path(ACTIVE_SHAPE_ROOT, shape_name)
            color_path = find_image_path(ACTIVE_COLOR_ROOT, color_name)
            sample_dir = USER_OUTPUT_DIR / f"sample_{idx:03d}_{face_name}__{shape_name}__{color_name}"
            print(f"{idx:03d} {face_name} {shape_name} {color_name} -> {sample_dir}", file=manifest, flush=True)
            run_one_probe(hair_fast, face_path, shape_path, color_path, sample_dir)

    print(f"[style_probe_v24] saved {len(triplets)} samples to {USER_OUTPUT_DIR}")
    print(f"[style_probe_v24] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
