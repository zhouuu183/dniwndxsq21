import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap import HairFast, get_parser


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"
USER_EXPORT_SPLIT = "val"  # "train", "val", or "all"

USER_DATASET_DIR_FFHQ = Path("images/satd_dataset_v8_3000")
USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("output/author_export_v8_3000")
USER_VAL_SIZE_FFHQ = 30

USER_DATASET_DIR_SMALL = Path("images/satd_dataset_v8_small8")
USER_FACE_ROOT_SMALL = Path("images/FFHQ_long")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_short")
USER_OUTPUT_DIR_SMALL = Path("output/author_export_v8_small")
USER_VAL_SIZE_SMALL = 30

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_MAX_PAIRS = 0  # 0 means export all selected pairs
USER_SAVE_PANEL = True
USER_USE_SHADOW_CLEANUP = False
# ============================================================================


def resolve_profile_defaults():
    if USER_DATASET_PROFILE == "ffhq":
        return {
            "dataset_dir": USER_DATASET_DIR_FFHQ,
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "val_size": USER_VAL_SIZE_FFHQ,
        }
    if USER_DATASET_PROFILE == "small":
        return {
            "dataset_dir": USER_DATASET_DIR_SMALL,
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "val_size": USER_VAL_SIZE_SMALL,
        }
    raise ValueError(f"Unsupported USER_DATASET_PROFILE: {USER_DATASET_PROFILE}")


PROFILE = resolve_profile_defaults()
ACTIVE_DATASET_DIR = PROFILE["dataset_dir"]
ACTIVE_FACE_ROOT = PROFILE["face_root"]
ACTIVE_SHAPE_ROOT = PROFILE["shape_root"]
ACTIVE_OUTPUT_DIR = PROFILE["output_dir"]
ACTIVE_VAL_SIZE = PROFILE["val_size"]


def load_pairs(dataset_dir: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(dataset_dir / "dataset.exps", "r", encoding="utf-8") as handle:
        for line in handle:
            items = line.strip().split()
            if len(items) == 2:
                pairs.append((items[0], items[1]))
    return pairs


def select_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    export_split = USER_EXPORT_SPLIT.lower()
    if export_split == "all":
        selected = list(pairs)
    else:
        if len(pairs) <= ACTIVE_VAL_SIZE:
            raise RuntimeError(
                f"dataset.exps has only {len(pairs)} pairs, which is not larger than val_size={ACTIVE_VAL_SIZE}."
            )
        train_pairs, val_pairs = train_test_split(
            pairs,
            test_size=ACTIVE_VAL_SIZE,
            random_state=USER_RANDOM_SEED,
        )
        if export_split == "train":
            selected = train_pairs
        elif export_split == "val":
            selected = val_pairs
        else:
            raise ValueError(f"Unsupported USER_EXPORT_SPLIT: {USER_EXPORT_SPLIT}")

    max_pairs = int(USER_MAX_PAIRS)
    if max_pairs > 0:
        selected = selected[:max_pairs]
    return selected


def resolve_image_path(root: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for stem={stem!r} under {root}")


def load_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return T.ToTensor()(image.convert("RGB"))


def ensure_chw(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 4:
        image = image[0]
    return image.detach().cpu().clamp(0.0, 1.0)


def resize_like(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(image.shape[-2:]) == (height, width):
        return image
    return F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]


def save_panel(path: Path, source: torch.Tensor, reference: torch.Tensor, result: torch.Tensor) -> None:
    result = ensure_chw(result)
    height, width = result.shape[-2:]
    source = resize_like(ensure_chw(source), height, width)
    reference = resize_like(ensure_chw(reference), height, width)
    panel = torch.cat([source, reference, result], dim=2)
    save_image(panel, path)


def build_author_model() -> HairFast:
    model_args = get_parser().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    if USER_USE_SHADOW_CLEANUP:
        model_args.use_shadow_cleanup = True
    return HairFast(model_args)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    pairs = load_pairs(ACTIVE_DATASET_DIR)
    selected_pairs = select_pairs(pairs)
    if not selected_pairs:
        raise RuntimeError("No pairs selected for export.")

    split_tag = USER_EXPORT_SPLIT.lower()
    result_dir = ACTIVE_OUTPUT_DIR / split_tag / "results"
    panel_dir = ACTIVE_OUTPUT_DIR / split_tag / "panels"
    result_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_PANEL:
        panel_dir.mkdir(parents=True, exist_ok=True)

    model = build_author_model()
    manifest_lines = []

    for index, (face_stem, shape_stem) in enumerate(tqdm(selected_pairs, desc=f"Author export {split_tag}"), start=1):
        face_path = resolve_image_path(ACTIVE_FACE_ROOT, face_stem)
        shape_path = resolve_image_path(ACTIVE_SHAPE_ROOT, shape_stem)
        result_name = f"{index:04d}__{face_stem}__{shape_stem}.png"
        result_path = result_dir / result_name

        result = model.swap(
            face_path,
            shape_path,
            shape_path,
            use_shadow_cleanup=USER_USE_SHADOW_CLEANUP,
        )
        save_image(ensure_chw(result), result_path)

        if USER_SAVE_PANEL:
            save_panel(panel_dir / result_name, load_tensor(face_path), load_tensor(shape_path), result)

        manifest_lines.append(f"{face_stem} {shape_stem} {result_path.as_posix()}\n")

    with open(ACTIVE_OUTPUT_DIR / split_tag / "export_manifest.txt", "w", encoding="utf-8") as handle:
        handle.writelines(manifest_lines)

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"export split: {split_tag}")
    print(f"dataset dir: {ACTIVE_DATASET_DIR}")
    print(f"face root: {ACTIVE_FACE_ROOT}")
    print(f"shape root: {ACTIVE_SHAPE_ROOT}")
    print(f"exported pairs: {len(selected_pairs)}")
    print(f"results dir: {result_dir}")
    if USER_SAVE_PANEL:
        print(f"panels dir: {panel_dir}")


if __name__ == "__main__":
    main()
