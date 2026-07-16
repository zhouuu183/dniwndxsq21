from __future__ import annotations

import os
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

# ========================= 用户配置区域：只改这里 =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("images/blending_dataset_v10_3000")
USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/blending_eval_v10_noF64_3000")

USER_DATASET_DIR_SMALL = Path("images/blending_dataset_v11_noF64")
USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ")
USER_OUTPUT_DIR_SMALL = Path("output/blending_eval_v10_noF64_small")

USER_RANDOM_SEED = 3407
USER_MAX_SAMPLES = 300  # 0 表示跑完整 dataset.exps
USER_START_INDEX = 0
USER_SKIP_EXISTING = True
USER_SAVE_RESULT_IMAGE = True
USER_SAVE_BASELINE_IMAGE = True
USER_SAVE_DIFF_IMAGE = True
USER_SAVE_PREVIEW_GRID = True
USER_SAVE_ALPHA_DEBUG = True
USER_COMPUTE_BASELINE_COMPARE = True
USER_DIFF_GAIN = 4.0

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"

USER_SMOOTH = 5
USER_ALPHA_BOUNDARY_WIDTH_V10 = 9
USER_ALPHA_BOUNDARY_STRENGTH_V10 = 0.65
USER_ALPHA_FALLBACK_BLUR_V10 = 9
USER_DISABLE_CUDNN_BENCHMARK = True
# ========================================================================


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Blending_v10 import Blending_v10
from models.Alignment_v10 import Alignment_v10
from models.Embedding import Embedding
from models.Net import Net
from utils.image_utils import equal_replacer
from utils.train import seed_everything, toggle_grad


def resolve_dataset_profile() -> dict[str, Path]:
    profiles = {
        "ffhq": {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
        },
        "small": {
            "dataset_dir": USER_DATASET_DIR_SMALL,
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "color_root": USER_COLOR_ROOT_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
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
        smooth=USER_SMOOTH,
        rotate_checkpoint=USER_ROTATE_CKPT,
        blending_checkpoint=USER_BLENDING_CKPT,
        pp_checkpoint=USER_PP_CKPT,
        blending_v10_checkpoint="",
        alpha_boundary_width_v10=USER_ALPHA_BOUNDARY_WIDTH_V10,
        alpha_boundary_strength_v10=USER_ALPHA_BOUNDARY_STRENGTH_V10,
        alpha_fallback_blur_v10=USER_ALPHA_FALLBACK_BLUR_V10,
        use_f64_bypass_v10=False,
        f64_bypass_strength_v10=0.0,
        f64_hidden_channels_v10=512,
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


def load_image_norm_256(root: Path, stem: str, device: torch.device) -> torch.Tensor:
    with Image.open(resolve_image_path(root, stem)) as image:
        image = image.convert("RGB").resize((256, 256), Image.BICUBIC)
        tensor = T.functional.to_tensor(image).unsqueeze(0)
        return (tensor * 2.0 - 1.0).to(device)


def load_preview_image(root: Path, stem: str) -> Image.Image:
    with Image.open(resolve_image_path(root, stem)) as image:
        return image.convert("RGB").resize((256, 256), Image.BICUBIC)


def load_image_tensor_1024(root: Path, stem: str) -> torch.Tensor:
    with Image.open(resolve_image_path(root, stem)) as image:
        image = image.convert("RGB").resize((1024, 1024), Image.BICUBIC)
        return T.functional.to_tensor(image)


def load_npz_tensor(path: Path, key: str, device: torch.device) -> torch.Tensor:
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain key {key!r}")
        return torch.from_numpy(data[key]).to(device=device).float()


def ensure_batch_latent_s(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    return tensor


def ensure_bchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3:
        return tensor.unsqueeze(1) if tensor.shape[0] == 1 else tensor.unsqueeze(0)
    return tensor


def role_fs_path(role: str, stem: str) -> Path:
    return ACTIVE_DATASET_DIR / "FS" / f"{role}_{stem}.npz"


def role_mask_path(role: str, stem: str) -> Path:
    return ACTIVE_DATASET_DIR / "Mask" / f"{role}_{stem}.npz"


def align_path(kind: str, face_stem: str, other_stem: str) -> Path:
    return ACTIVE_DATASET_DIR / kind / f"face_{face_stem}__{kind.lower()}_{other_stem}.npz"


def load_role_cache(role: str, stem: str, image_root: Path, device: torch.device) -> dict[str, torch.Tensor]:
    fs_path = role_fs_path(role, stem)
    mask_path = role_mask_path(role, stem)
    return {
        "S": ensure_batch_latent_s(load_npz_tensor(fs_path, "latent_S", device)),
        "mask": ensure_bchw(load_npz_tensor(mask_path, "parsing_mask", device)),
        "image_norm_256": load_image_norm_256(image_root, stem, device),
        "alpha_256": ensure_bchw(load_npz_tensor(mask_path, "alpha_256", device)),
        "hair_mask_256": ensure_bchw(load_npz_tensor(mask_path, "hair_mask_256", device)),
    }


def load_align_cache(kind: str, face_stem: str, other_stem: str, device: torch.device) -> dict[str, torch.Tensor]:
    path = align_path(kind, face_stem, other_stem)
    with np.load(path) as data:
        result = {
            "target_hair_mask": ensure_bchw(torch.from_numpy(data["target_hair_mask"]).to(device=device).float()),
            "source_hair_mask": ensure_bchw(torch.from_numpy(data["source_hair_mask"]).to(device=device).float()),
            "donor_hair_mask": ensure_bchw(torch.from_numpy(data["donor_hair_mask"]).to(device=device).float()),
            "boundary_mask_256": ensure_bchw(torch.from_numpy(data["boundary_mask_256"]).to(device=device).float()),
            "boundary_alpha_256": ensure_bchw(torch.from_numpy(data["boundary_alpha_256"]).to(device=device).float()),
        }
        if "latent_F" in data:
            result["latent_F_align"] = torch.from_numpy(data["latent_F"]).to(device=device).float()
        if "latent_F_hair" in data:
            result["latent_F_hair"] = torch.from_numpy(data["latent_F_hair"]).to(device=device).float()
    return result


def tensor01_to_pil(image: torch.Tensor, size: tuple[int, int] | None = None) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    if image.dim() == 4:
        image = image[0]
    if size is not None:
        image = T.functional.resize(image, [size[1], size[0]], antialias=True)
    return T.functional.to_pil_image(image)


def diff_to_pil(image_a: torch.Tensor, image_b: torch.Tensor, gain: float = 4.0) -> Image.Image:
    diff = (image_a.detach().cpu().float() - image_b.detach().cpu().float()).abs() * float(gain)
    diff = diff.clamp(0, 1)
    if diff.dim() == 4:
        diff = diff[0]
    return T.functional.to_pil_image(diff)


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    mask = mask.detach().cpu().float().clamp(0, 1)
    if mask.dim() == 4:
        mask = mask[0]
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask[0]
    return Image.fromarray((mask.numpy() * 255.0).round().astype(np.uint8), mode="L").resize((256, 256))


def save_grid(panels: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height), color=(255, 255, 255))
    for idx, panel in enumerate(panels):
        canvas.paste(panel.convert("RGB"), (idx * width, 0))
    canvas.save(path)


def make_sample(
    face_stem: str,
    shape_stem: str,
    color_stem: str,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    name_to_embed = {
        "face": load_role_cache("face", face_stem, ACTIVE_FACE_ROOT, device),
        "shape": load_role_cache("shape", shape_stem, ACTIVE_SHAPE_ROOT, device),
        "color": load_role_cache("color", color_stem, ACTIVE_COLOR_ROOT, device),
    }
    align_shape = load_align_cache("AlignShape", face_stem, shape_stem, device)
    align_color = load_align_cache("AlignColor", face_stem, color_stem, device)
    return name_to_embed, align_shape, align_color


def make_baseline_compare(
    face_stem: str,
    shape_stem: str,
    color_stem: str,
    embedder: Embedding,
    aligner: Alignment_v10,
    model: Blending_v10,
) -> torch.Tensor:
    face_img, shape_img, color_img = equal_replacer(
        [
            load_image_tensor_1024(ACTIVE_FACE_ROOT, face_stem),
            load_image_tensor_1024(ACTIVE_SHAPE_ROOT, shape_stem),
            load_image_tensor_1024(ACTIVE_COLOR_ROOT, color_stem),
        ]
    )
    images_to_name = defaultdict(list)
    for image, name in zip((face_img, shape_img, color_img), ("face", "shape", "color")):
        images_to_name[image].append(name)

    with torch.inference_mode():
        name_to_embed = embedder.embedding_images(images_to_name)
        align_shape = aligner.align_images(
            "face",
            "shape",
            name_to_embed,
            alpha_boundary_strength_v10=0.0,
        )
        if shape_img is not color_img:
            align_color = aligner.shape_module(
                "face",
                "color",
                name_to_embed,
                only_target=True,
                alpha_boundary_strength_v10=0.0,
            )
        else:
            align_color = align_shape
        return model.blend_images(
            align_shape,
            align_color,
            name_to_embed,
            use_f64_bypass_v10=False,
            f64_bypass_strength_v10=0.0,
        )


def main() -> None:
    seed_everything(USER_RANDOM_SEED)
    if USER_DISABLE_CUDNN_BENCHMARK and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    opts = build_opts(device)
    net = Net(opts)
    model = Blending_v10(opts, net=net).to(device).eval()
    embedder = None
    aligner = None
    if USER_COMPUTE_BASELINE_COMPARE:
        embedder = Embedding(opts, net=net).eval()
        aligner = Alignment_v10(opts, latent_encoder=embedder.get_e4e_embed, net=net).eval()
    toggle_grad(net.generator, False)
    toggle_grad(model, False)
    if aligner is not None:
        toggle_grad(aligner, False)

    exps = load_exps(ACTIVE_DATASET_DIR)
    start = max(0, USER_START_INDEX)
    end = len(exps) if USER_MAX_SAMPLES == 0 else min(len(exps), start + USER_MAX_SAMPLES)
    selected = exps[start:end]
    if not selected:
        raise RuntimeError(f"No samples selected from {ACTIVE_DATASET_DIR / 'dataset.exps'}")

    result_dir = ACTIVE_OUTPUT_DIR / "results"
    baseline_dir = ACTIVE_OUTPUT_DIR / "baseline_compare"
    diff_dir = ACTIVE_OUTPUT_DIR / "diff"
    grid_dir = ACTIVE_OUTPUT_DIR / "grids"
    alpha_dir = ACTIVE_OUTPUT_DIR / "alpha_debug"
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"dataset dir: {ACTIVE_DATASET_DIR}")
    print(f"output dir: {ACTIVE_OUTPUT_DIR}")
    print(f"samples: {start}..{end - 1} ({len(selected)})")
    print("F64 bypass: disabled")
    print(f"baseline compare: {USER_COMPUTE_BASELINE_COMPARE}")

    for offset, (face_stem, shape_stem, color_stem) in enumerate(tqdm(selected, desc="eval noF64"), start=start):
        sample_name = f"{offset:04d}_face_{face_stem}__shape_{shape_stem}__color_{color_stem}.png"
        result_path = result_dir / sample_name
        baseline_path = baseline_dir / sample_name
        diff_path = diff_dir / sample_name
        grid_path = grid_dir / sample_name
        expected_paths = []
        if USER_SAVE_RESULT_IMAGE:
            expected_paths.append(result_path)
        if USER_COMPUTE_BASELINE_COMPARE and USER_SAVE_BASELINE_IMAGE:
            expected_paths.append(baseline_path)
        if USER_COMPUTE_BASELINE_COMPARE and USER_SAVE_DIFF_IMAGE:
            expected_paths.append(diff_path)
        if USER_SAVE_PREVIEW_GRID:
            expected_paths.append(grid_path)
        if USER_SKIP_EXISTING and expected_paths and all(path.exists() for path in expected_paths):
            continue

        try:
            name_to_embed, align_shape, align_color = make_sample(face_stem, shape_stem, color_stem, device)
            with torch.inference_mode():
                final_image = model.blend_images(
                    align_shape,
                    align_color,
                    name_to_embed,
                    use_f64_bypass_v10=False,
                    f64_bypass_strength_v10=0.0,
                )
                baseline_image = None
                if USER_COMPUTE_BASELINE_COMPARE:
                    if embedder is None or aligner is None:
                        raise RuntimeError("baseline compare requested but embedder/aligner were not initialized")
                    baseline_image = make_baseline_compare(
                        face_stem,
                        shape_stem,
                        color_stem,
                        embedder,
                        aligner,
                        model,
                    )

            result_pil = tensor01_to_pil(final_image)
            if USER_SAVE_RESULT_IMAGE:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_pil.save(result_path)

            baseline_pil = None
            diff_pil = None
            if baseline_image is not None:
                baseline_pil = tensor01_to_pil(baseline_image)
                diff_pil = diff_to_pil(final_image, baseline_image, gain=USER_DIFF_GAIN)
                if USER_SAVE_BASELINE_IMAGE:
                    baseline_path.parent.mkdir(parents=True, exist_ok=True)
                    baseline_pil.save(baseline_path)
                if USER_SAVE_DIFF_IMAGE:
                    diff_path.parent.mkdir(parents=True, exist_ok=True)
                    diff_pil.save(diff_path)

            if USER_SAVE_PREVIEW_GRID:
                panels = [
                    load_preview_image(ACTIVE_FACE_ROOT, face_stem),
                    load_preview_image(ACTIVE_SHAPE_ROOT, shape_stem),
                    load_preview_image(ACTIVE_COLOR_ROOT, color_stem),
                ]
                if baseline_pil is not None:
                    panels.append(baseline_pil.resize((256, 256), Image.BICUBIC))
                panels.append(tensor01_to_pil(final_image, size=(256, 256)))
                if diff_pil is not None:
                    panels.append(diff_pil.resize((256, 256), Image.BICUBIC))
                panels.append(mask_to_pil(align_shape["boundary_alpha_256"]).convert("RGB"))
                save_grid(panels, grid_path)

            if USER_SAVE_ALPHA_DEBUG:
                alpha_dir.mkdir(parents=True, exist_ok=True)
                mask_to_pil(align_shape["boundary_alpha_256"]).save(alpha_dir / sample_name)
        except Exception as exc:
            print(f"[skip] {face_stem} {shape_stem} {color_stem}: {exc}", file=sys.stderr)

    print(f"done: {ACTIVE_OUTPUT_DIR}")
    print("grid panels: face | shape | color | baseline(strength=0) | noF64 alpha | diff x gain | shape boundary alpha")


if __name__ == "__main__":
    main()
