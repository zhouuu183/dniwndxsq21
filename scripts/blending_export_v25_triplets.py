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

from hair_swap_v8 import HairFast_v8, get_parser_v8
from utils.image_utils import list_image_files


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "celebahq_fixed3000"
USER_REFERENCE_MODE = "both"  # "both" or "full"

USER_FACE_ROOT_CELEBAHQ = Path("images/CelebA-HQ")
USER_SHAPE_ROOT_CELEBAHQ = Path("images/CelebA-HQ")
USER_COLOR_ROOT_CELEBAHQ = Path("images/CelebA-HQ")
USER_OUTPUT_DIR_CELEBAHQ = Path("output/blending_export_v8_celebahq_fixed3000")
USER_SAMPLE_SIZE_CELEBAHQ = 3000

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_SAVE_PANEL = True
USER_ALLOW_REUSE_ACROSS_TRIPLETS = False

USER_BLENDING_CHECKPOINT = Path("output/blending_train_v8/checkpoints/best.pth")
USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = Path("output/satd_train_v8_3000/checkpoints/best.pth")
USER_SATD_BLEND_V8 = 0.34
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0
# ============================================================================


def resolve_profile_defaults():
    if USER_DATASET_PROFILE == "celebahq_fixed3000":
        return {
            "face_root": USER_FACE_ROOT_CELEBAHQ,
            "shape_root": USER_SHAPE_ROOT_CELEBAHQ,
            "color_root": USER_COLOR_ROOT_CELEBAHQ,
            "output_dir": USER_OUTPUT_DIR_CELEBAHQ,
            "sample_size": USER_SAMPLE_SIZE_CELEBAHQ,
        }
    raise ValueError(f"Unsupported USER_DATASET_PROFILE: {USER_DATASET_PROFILE}")


PROFILE = resolve_profile_defaults()
ACTIVE_FACE_ROOT = PROFILE["face_root"]
ACTIVE_SHAPE_ROOT = PROFILE["shape_root"]
ACTIVE_COLOR_ROOT = PROFILE["color_root"]
ACTIVE_OUTPUT_DIR = PROFILE["output_dir"]
ACTIVE_SAMPLE_SIZE = PROFILE["sample_size"]


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


def ensure_chw(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 4:
        image = image[0]
    return image.detach().cpu().clamp(0.0, 1.0)


def resize_like(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(image.shape[-2:]) == (height, width):
        return image
    return F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]


def save_panel(
    path: Path,
    source: torch.Tensor,
    shape: torch.Tensor,
    color: torch.Tensor,
    result: torch.Tensor,
) -> None:
    result = ensure_chw(result)
    height, width = result.shape[-2:]
    source = resize_like(ensure_chw(source), height, width)
    shape = resize_like(ensure_chw(shape), height, width)
    color = resize_like(ensure_chw(color), height, width)
    panel = torch.cat([source, shape, color, result], dim=2)
    save_image(panel, path)


def build_model() -> HairFast_v8:
    if not USER_BLENDING_CHECKPOINT.exists():
        raise FileNotFoundError(f"Cannot find blending checkpoint: {USER_BLENDING_CHECKPOINT}")
    if USER_USE_SATD_V8 and not USER_SATD_CHECKPOINT_V8.exists():
        raise FileNotFoundError(f"Cannot find SATD checkpoint: {USER_SATD_CHECKPOINT_V8}")

    model_args = get_parser_v8().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    model_args.blending_checkpoint = str(USER_BLENDING_CHECKPOINT)
    model_args.use_satd_v8 = bool(USER_USE_SATD_V8)
    model_args.satd_checkpoint_v8 = str(USER_SATD_CHECKPOINT_V8)
    model_args.satd_blend_v8 = USER_SATD_BLEND_V8
    model_args.satd_boundary_v8 = USER_SATD_BOUNDARY_V8
    model_args.eq8_reference_blend_v8 = USER_EQ8_REFERENCE_BLEND_V8
    return HairFast_v8(model_args)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        raise RuntimeError("No triplets selected for export.")

    mode_tag = str(USER_REFERENCE_MODE).lower()
    result_dir = ACTIVE_OUTPUT_DIR / mode_tag / "results"
    panel_dir = ACTIVE_OUTPUT_DIR / mode_tag / "panels"
    real_dir = ACTIVE_OUTPUT_DIR / mode_tag / "real_images"
    result_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_PANEL:
        panel_dir.mkdir(parents=True, exist_ok=True)

    model = build_model()
    manifest_lines = []

    for index, (face_stem, shape_stem, color_stem) in enumerate(
        tqdm(triplets, desc=f"Blending export celebahq_fixed3000 ({mode_tag})"),
        start=1,
    ):
        face_path = resolve_image_path(ACTIVE_FACE_ROOT, face_stem)
        shape_path = resolve_image_path(ACTIVE_SHAPE_ROOT, shape_stem)
        color_path = resolve_image_path(ACTIVE_COLOR_ROOT, color_stem)

        result_name = f"{index:04d}__{face_stem}__{shape_stem}__{color_stem}.png"
        real_name = f"{index:04d}__{face_stem}.png"
        result_path = result_dir / result_name
        real_path = real_dir / real_name

        result = model.swap(
            face_path,
            shape_path,
            color_path,
            use_satd_v8=USER_USE_SATD_V8,
            satd_blend_v8=USER_SATD_BLEND_V8,
            satd_boundary_v8=USER_SATD_BOUNDARY_V8,
            eq8_reference_blend_v8=USER_EQ8_REFERENCE_BLEND_V8,
        )
        source_tensor = load_tensor(face_path)
        save_image(ensure_chw(result), result_path)
        save_image(ensure_chw(source_tensor), real_path)

        if USER_SAVE_PANEL:
            save_panel(
                panel_dir / result_name,
                source_tensor,
                load_tensor(shape_path),
                load_tensor(color_path),
                result,
            )

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

    manifest_path = ACTIVE_OUTPUT_DIR / mode_tag / "export_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.writelines(manifest_lines)

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"reference mode: {mode_tag}")
    print(f"face root: {ACTIVE_FACE_ROOT}")
    print(f"shape root: {ACTIVE_SHAPE_ROOT}")
    print(f"color root: {ACTIVE_COLOR_ROOT}")
    print(f"fixed sample size: {len(triplets)}")
    print(f"results dir: {result_dir}")
    print(f"real dir: {real_dir}")
    print(f"manifest: {manifest_path}")
    print(f"blending checkpoint: {USER_BLENDING_CHECKPOINT}")
    if USER_USE_SATD_V8:
        print(f"satd checkpoint: {USER_SATD_CHECKPOINT_V8}")
    if USER_SAVE_PANEL:
        print(f"panels dir: {panel_dir}")


if __name__ == "__main__":
    main()
