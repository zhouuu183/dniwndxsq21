import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image


# ============================================================
# USER CONFIG AREA: edit only this block
# ------------------------------------------------------------
# FFHQ image root used by blending_gen_v1.py
USER_FFHQ_ROOT = Path("input/FFHQ")
#
# blending_gen_v1.py output directory
USER_DATASET_DIR = Path("input/blending_dataset_v1")
#
# Where to save inspection images and the json report
USER_OUTPUT_DIR = Path("output/check_blending_dataset_v1")
#
# How many samples to inspect
USER_NUM_SAMPLES = 12
#
# Set True to sample randomly, False to inspect the first N samples
USER_RANDOM_SAMPLE = True
#
# Random seed for sampling
USER_RANDOM_SEED = 3407
# ============================================================


REQUIRED_FS_KEYS = ["latent_in", "mask"]
REQUIRED_ALIGN_KEYS = ["latent_F_align"]
REQUIRED_DELTA_KEYS = [
    "M_src",
    "M_src_hair",
    "M_tgt",
    "M_add",
    "M_remove",
    "M_keep",
    "M_boundary",
    "M_accessory",
    "M_color",
]


def resolve_image_path(root: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find image for stem={stem} under {root}")


def load_image_256(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((256, 256), resample=Image.LANCZOS)
    return TF.to_tensor(image)


def load_mask_2d(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(array).float().squeeze()
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D mask after squeeze, got shape={tuple(tensor.shape)}")
    return tensor


def mask_to_rgb(mask: torch.Tensor) -> torch.Tensor:
    return mask.unsqueeze(0).repeat(3, 1, 1).clamp(0, 1)


def approx_binary(mask: torch.Tensor, atol: float = 1e-4) -> bool:
    return bool((((mask - 0.0).abs() < atol) | ((mask - 1.0).abs() < atol)).all())


def close_mask(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-4) -> bool:
    return bool(torch.allclose(a, b, atol=atol, rtol=0.0))


def save_visual_grid(
    face_img: torch.Tensor,
    shape_img: torch.Tensor,
    color_img: torch.Tensor,
    masks: dict[str, torch.Tensor],
    output_path: Path,
):
    grid_items = [
        face_img,
        shape_img,
        color_img,
        mask_to_rgb(masks["M_src"]),
        mask_to_rgb(masks["M_tgt"]),
        mask_to_rgb(masks["M_add"]),
        mask_to_rgb(masks["M_remove"]),
        mask_to_rgb(masks["M_keep"]),
        mask_to_rgb(masks["M_boundary"]),
        mask_to_rgb(masks["M_accessory"]),
        mask_to_rgb(masks["M_color"]),
    ]
    canvas = torch.stack(grid_items, dim=0)
    save_image(canvas, output_path, nrow=6)


def inspect_sample(dataset_dir: Path, ffhq_root: Path, exp: tuple[str, str, str]) -> dict:
    face_stem, shape_stem, color_stem = exp
    issues: list[str] = []

    fs_paths = {
        "face": dataset_dir / "FS" / f"{face_stem}.npz",
        "shape": dataset_dir / "FS" / f"{shape_stem}.npz",
        "color": dataset_dir / "FS" / f"{color_stem}.npz",
    }
    align_shape_path = dataset_dir / "Align" / f"{face_stem}_{shape_stem}.npz"
    align_color_path = dataset_dir / "Align" / f"{face_stem}_{color_stem}.npz"
    delta_path = dataset_dir / "Delta" / f"{face_stem}_{shape_stem}_{color_stem}.npz"

    for name, path in fs_paths.items():
        if not path.is_file():
            issues.append(f"missing FS file: {name} -> {path}")
    for path in (align_shape_path, align_color_path, delta_path):
        if not path.is_file():
            issues.append(f"missing file: {path}")

    if issues:
        return {
            "exp": {"face": face_stem, "shape": shape_stem, "color": color_stem},
            "ok": False,
            "issues": issues,
        }

    stats = {}
    for name, path in fs_paths.items():
        npz = np.load(path)
        for key in REQUIRED_FS_KEYS:
            if key not in npz:
                issues.append(f"{path.name} missing key: {key}")
        if "latent_in" in npz:
            stats[f"{name}_latent_shape"] = list(npz["latent_in"].shape)
        if "mask" in npz:
            stats[f"{name}_mask_shape"] = list(npz["mask"].shape)

    for path in (align_shape_path, align_color_path):
        npz = np.load(path)
        for key in REQUIRED_ALIGN_KEYS:
            if key not in npz:
                issues.append(f"{path.name} missing key: {key}")
        if "latent_F_align" in npz:
            shape = list(npz["latent_F_align"].shape)
            stats[f"{path.stem}_latent_F_align_shape"] = shape
            if shape[-2:] != [32, 32]:
                issues.append(f"{path.name} latent_F_align last dims are not [32, 32]: {shape}")

    delta_npz = np.load(delta_path)
    masks = {}
    for key in REQUIRED_DELTA_KEYS:
        if key not in delta_npz:
            issues.append(f"{delta_path.name} missing key: {key}")
            continue
        try:
            masks[key] = load_mask_2d(delta_npz[key])
        except Exception as exc:
            issues.append(f"{delta_path.name} failed to parse {key}: {exc}")

    if len(masks) == len(REQUIRED_DELTA_KEYS):
        for key, mask in masks.items():
            stats[f"{key}_shape"] = list(mask.shape)
            stats[f"{key}_sum"] = float(mask.sum().item())
            if mask.shape != (256, 256):
                issues.append(f"{key} shape is not [256, 256]: {tuple(mask.shape)}")
            if not approx_binary(mask):
                issues.append(f"{key} is not approximately binary")

        src = masks["M_src"]
        src_hair = masks["M_src_hair"]
        tgt = masks["M_tgt"]
        add = masks["M_add"]
        remove = masks["M_remove"]
        keep = masks["M_keep"]
        boundary = masks["M_boundary"]
        accessory = masks["M_accessory"]

        if not close_mask((add + keep).clamp(0, 1), tgt):
            issues.append("M_add + M_keep != M_tgt")
        if not close_mask((remove + keep).clamp(0, 1), src):
            issues.append("M_remove + M_keep != M_src")
        if float((add * remove).max().item()) > 1e-4:
            issues.append("M_add and M_remove overlap")
        if float((boundary * add).sum().item()) <= 0 and float((boundary * remove).sum().item()) <= 0:
            issues.append("M_boundary does not touch change regions")
        if float((accessory * src_hair).max().item()) > 1e-4:
            issues.append("M_accessory overlaps with M_src_hair")

    return {
        "exp": {"face": face_stem, "shape": shape_stem, "color": color_stem},
        "ok": len(issues) == 0,
        "issues": issues,
        "stats": stats,
    }


def main():
    random.seed(USER_RANDOM_SEED)
    np.random.seed(USER_RANDOM_SEED)
    USER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_exps = USER_DATASET_DIR / "dataset.exps"
    if not dataset_exps.is_file():
        raise FileNotFoundError(f"Cannot find dataset.exps: {dataset_exps}")

    with open(dataset_exps, "r", encoding="utf-8") as file:
        exps = [tuple(line.split()) for line in file if line.strip()]

    if not exps:
        raise RuntimeError(f"No experiments found in {dataset_exps}")

    if USER_RANDOM_SAMPLE:
        selected = random.sample(exps, k=min(USER_NUM_SAMPLES, len(exps)))
    else:
        selected = exps[: USER_NUM_SAMPLES]

    results = []
    for idx, exp in enumerate(selected):
        result = inspect_sample(USER_DATASET_DIR, USER_FFHQ_ROOT, exp)
        results.append(result)

        if result.get("ok"):
            face_stem = result["exp"]["face"]
            shape_stem = result["exp"]["shape"]
            color_stem = result["exp"]["color"]

            delta_npz = np.load(USER_DATASET_DIR / "Delta" / f"{face_stem}_{shape_stem}_{color_stem}.npz")
            masks = {key: load_mask_2d(delta_npz[key]) for key in REQUIRED_DELTA_KEYS}

            face_img = load_image_256(resolve_image_path(USER_FFHQ_ROOT, face_stem))
            shape_img = load_image_256(resolve_image_path(USER_FFHQ_ROOT, shape_stem))
            color_img = load_image_256(resolve_image_path(USER_FFHQ_ROOT, color_stem))

            vis_path = USER_OUTPUT_DIR / f"{idx:03d}_{face_stem}_{shape_stem}_{color_stem}.png"
            save_visual_grid(face_img, shape_img, color_img, masks, vis_path)

    summary = {
        "dataset_dir": str(USER_DATASET_DIR),
        "ffhq_root": str(USER_FFHQ_ROOT),
        "num_total_exps": len(exps),
        "num_checked": len(results),
        "num_ok": sum(int(item["ok"]) for item in results),
        "num_bad": sum(int(not item["ok"]) for item in results),
        "legend": [
            "row order in visualization:",
            "1 face, 2 shape, 3 color, 4 M_src, 5 M_tgt,",
            "6 M_add, 7 M_remove, 8 M_keep, 9 M_boundary, 10 M_accessory, 11 M_color",
        ],
        "results": results,
    }

    with open(USER_OUTPUT_DIR / "report.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Checked {summary['num_checked']} samples from {USER_DATASET_DIR}")
    print(f"OK: {summary['num_ok']}, BAD: {summary['num_bad']}")
    print(f"Report saved to: {USER_OUTPUT_DIR / 'report.json'}")
    print(f"Visualizations saved to: {USER_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
