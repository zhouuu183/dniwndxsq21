from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

# ========================= 用户配置区域：只改这里 =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_DATASET_DIR_FFHQ = Path("images/shape_dataset_v12_3000")
USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/shape_eval_v12_3000")

USER_DATASET_DIR_SMALL = Path("images/shape_dataset_v12_small")
USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ")
USER_OUTPUT_DIR_SMALL = Path("output/shape_eval_v12_small")

USER_SHAPE_ADAPTER_CKPT = Path("output/shape_train_v12_small/checkpoints/shape_adapter_v12_for_infer.pth")
USER_MAX_SAMPLES = 300
USER_START_INDEX = 0
USER_SKIP_EXISTING = True
USER_DIFF_GAIN = 4.0

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_SMOOTH = 5
USER_SHAPE_ADAPTER_HIDDEN_V12 = 512
USER_SHAPE_ADAPTER_STRENGTH_V12 = 1.0
USER_SHAPE_PRIOR_STRENGTH_V12 = 0.0
# ========================================================================


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Blending_v12 import Blending_v12
from models.Net import Net
from models.ShapeAdapter_v12 import TopologyF32Adapter_v12, V12_MASK_KEYS, load_shape_adapter_v12
from utils.train import toggle_grad


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
        raise RuntimeError(f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}.")
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


def preview_image(root: Path, stem: str) -> Image.Image:
    with Image.open(resolve_image_path(root, stem)) as image:
        return image.convert("RGB").resize((256, 256), Image.BICUBIC)


def image_norm_256(root: Path, stem: str, device: torch.device) -> torch.Tensor:
    with Image.open(resolve_image_path(root, stem)) as image:
        image = image.convert("RGB").resize((256, 256), Image.BICUBIC)
        tensor = T.functional.to_tensor(image).unsqueeze(0)
        return (tensor * 2.0 - 1.0).to(device)


def load_npz(path: Path, key: str, device: torch.device) -> torch.Tensor:
    with np.load(path) as data:
        return torch.from_numpy(data[key]).float().to(device)


def ensure_bchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3:
        return tensor.unsqueeze(1) if tensor.shape[0] != 1 else tensor.unsqueeze(0)
    return tensor


def role_fs(role: str, stem: str) -> Path:
    return ACTIVE_DATASET_DIR / "FS" / f"{role}_{stem}.npz"


def role_mask(role: str, stem: str) -> Path:
    return ACTIVE_DATASET_DIR / "Mask" / f"{role}_{stem}.npz"


def align_shape_path(face_stem: str, shape_stem: str) -> Path:
    return ACTIVE_DATASET_DIR / "AlignShape" / f"face_{face_stem}__alignshape_{shape_stem}.npz"


def align_color_path(face_stem: str, color_stem: str) -> Path:
    return ACTIVE_DATASET_DIR / "AlignColor" / f"face_{face_stem}__aligncolor_{color_stem}.npz"


def load_role(role: str, stem: str, root: Path, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "S": load_npz(role_fs(role, stem), "latent_S", device),
        "mask": ensure_bchw(load_npz(role_mask(role, stem), "parsing_mask", device)),
        "image_norm_256": image_norm_256(root, stem, device),
    }


def load_align_shape(face_stem: str, shape_stem: str, device: torch.device) -> dict[str, torch.Tensor]:
    path = align_shape_path(face_stem, shape_stem)
    with np.load(path) as data:
        result = {
            "latent_F_base": torch.from_numpy(data["latent_F_base"]).float().to(device),
            "latent_F_src": torch.from_numpy(data["latent_F_src"]).float().to(device),
            "latent_F_shape": torch.from_numpy(data["latent_F_shape"]).float().to(device),
            "latent_F_align": torch.from_numpy(data["latent_F_base"]).float().to(device),
            "HM_X": ensure_bchw(torch.from_numpy(data["target_hair_mask"]).float().to(device)),
            "target_hair_mask": ensure_bchw(torch.from_numpy(data["target_hair_mask"]).float().to(device)),
        }
        for key in V12_MASK_KEYS:
            result[key] = ensure_bchw(torch.from_numpy(data[key]).float().to(device))
    return result


def load_align_color(face_stem: str, color_stem: str, device: torch.device) -> dict[str, torch.Tensor]:
    path = align_color_path(face_stem, color_stem)
    with np.load(path) as data:
        target = ensure_bchw(torch.from_numpy(data["target_hair_mask"]).float().to(device))
    return {"HM_X": target, "target_hair_mask": target}


def tensor01_to_pil(image: torch.Tensor, size: tuple[int, int] | None = None) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    if image.dim() == 4:
        image = image[0]
    if size is not None:
        image = T.functional.resize(image, [size[1], size[0]], antialias=True)
    return T.functional.to_pil_image(image)


def diff_to_pil(image_a: torch.Tensor, image_b: torch.Tensor, size: tuple[int, int] | None = None) -> Image.Image:
    diff = (image_a.detach().cpu().float() - image_b.detach().cpu().float()).abs() * USER_DIFF_GAIN
    if diff.dim() == 4:
        diff = diff[0]
    if size is not None:
        diff = T.functional.resize(diff, [size[1], size[0]], antialias=True)
    return T.functional.to_pil_image(diff.clamp(0, 1))


def mask_to_pil(mask: torch.Tensor, size: tuple[int, int] | None = None) -> Image.Image:
    mask = mask.detach().cpu().float().clamp(0, 1)
    if mask.dim() == 4:
        mask = mask[0]
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask[0]
    pil = T.functional.to_pil_image(mask)
    return pil if size is None else pil.resize(size, Image.BICUBIC)


def save_grid(panels: list[Image.Image], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height), color=(255, 255, 255))
    for idx, panel in enumerate(panels):
        canvas.paste(panel.convert("RGB"), (idx * width, 0))
    canvas.save(path)


def main():
    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    opts = build_opts(device)
    net = Net(opts)
    blending = Blending_v12(opts, net=net).to(device).eval()
    adapter = TopologyF32Adapter_v12(hidden_channels=USER_SHAPE_ADAPTER_HIDDEN_V12).to(device).eval()
    if not USER_SHAPE_ADAPTER_CKPT.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {USER_SHAPE_ADAPTER_CKPT}")
    checkpoint = torch.load(USER_SHAPE_ADAPTER_CKPT, map_location=device)
    loaded, skipped = load_shape_adapter_v12(adapter, checkpoint)
    print(f"loaded adapter keys: {len(loaded)} | skipped incompatible: {len(skipped)}")
    toggle_grad(net.generator, False)
    toggle_grad(blending, False)
    toggle_grad(adapter, False)

    exps = load_exps(ACTIVE_DATASET_DIR)
    start = max(0, USER_START_INDEX)
    end = len(exps) if USER_MAX_SAMPLES == 0 else min(len(exps), start + USER_MAX_SAMPLES)
    selected = exps[start:end]
    grid_dir = ACTIVE_OUTPUT_DIR / "grids"
    result_dir = ACTIVE_OUTPUT_DIR / "results"
    baseline_dir = ACTIVE_OUTPUT_DIR / "baseline"
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"dataset dir: {ACTIVE_DATASET_DIR}")
    print(f"checkpoint: {USER_SHAPE_ADAPTER_CKPT}")
    print(f"samples: {start}..{end - 1} ({len(selected)})")

    for offset, (face_stem, shape_stem, color_stem) in enumerate(tqdm(selected, desc="eval v12"), start=start):
        sample_name = f"{offset:04d}_face_{face_stem}__shape_{shape_stem}__color_{color_stem}.png"
        grid_path = grid_dir / sample_name
        if USER_SKIP_EXISTING and grid_path.exists():
            continue

        name_to_embed = {
            "face": load_role("face", face_stem, ACTIVE_FACE_ROOT, device),
            "shape": load_role("shape", shape_stem, ACTIVE_SHAPE_ROOT, device),
            "color": load_role("color", color_stem, ACTIVE_COLOR_ROOT, device),
        }
        align_shape = load_align_shape(face_stem, shape_stem, device)
        align_color = load_align_color(face_stem, color_stem, device)
        masks_256 = {key: align_shape[key] for key in V12_MASK_KEYS}

        with torch.inference_mode():
            baseline = blending.blend_images(align_shape, align_color, name_to_embed)
            adapter_out = adapter(
                F_base=align_shape["latent_F_base"],
                F_src=align_shape["latent_F_src"],
                F_shape=align_shape["latent_F_shape"],
                masks_256=masks_256,
                strength=USER_SHAPE_ADAPTER_STRENGTH_V12,
                shape_prior_strength=USER_SHAPE_PRIOR_STRENGTH_V12,
            )
            align_shape_v12 = dict(align_shape)
            align_shape_v12["latent_F_align"] = adapter_out["latent_F_refined"]
            result = blending.blend_images(align_shape_v12, align_color, name_to_embed)

        result_path = result_dir / sample_name
        baseline_path = baseline_dir / sample_name
        result_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        tensor01_to_pil(result).save(result_path)
        tensor01_to_pil(baseline).save(baseline_path)
        save_grid(
            [
                preview_image(ACTIVE_FACE_ROOT, face_stem),
                preview_image(ACTIVE_SHAPE_ROOT, shape_stem),
                preview_image(ACTIVE_COLOR_ROOT, color_stem),
                tensor01_to_pil(baseline, size=(256, 256)),
                tensor01_to_pil(result, size=(256, 256)),
                diff_to_pil(result, baseline, size=(256, 256)),
                mask_to_pil(align_shape["M_edit"], size=(256, 256)).convert("RGB"),
            ],
            grid_path,
        )

    print(f"done: {ACTIVE_OUTPUT_DIR}")
    print("grid panels: face | shape | color | baseline | v12 | diff | edit mask")


if __name__ == "__main__":
    main()
