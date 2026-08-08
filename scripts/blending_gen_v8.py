import os
import random
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from hair_swap_v8 import HairFast_v8, get_parser_v8
from utils.image_utils import list_image_files
from utils.save_utils import save_latents
from utils.train import seed_everything


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("input/blending_dataset_v8")
USER_DATASET_SIZE_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("/root/shared-nvme/HairFastGAN/images/mix_ear/")
USER_SHAPE_ROOT_SMALL = Path("/root/shared-nvme/HairFastGAN/images/FFHQ_color/")
USER_COLOR_ROOT_SMALL = Path("/root/shared-nvme/HairFastGAN/images/ear/")
USER_OUTPUT_DIR_SMALL = Path("input/blending_dataset_v8_small")
USER_DATASET_SIZE_SMALL = 300

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_TRIPLETS = True

USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = "/root/shared-nvme/HairFastGAN/checkpoints/satd_3000_best.pth"
USER_SATD_BLEND_V8 = 0.34
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0
# Keep SATD skin-only: 1.0 = never modify target hair F (hair colour stays at
# baseline quality), 0 = allow SATD to modify hair F (old behaviour that caused
# the A+C hair-colour drift).
USER_SATD_HAIR_EXCLUDE_STRENGTH = 1.0
# ============================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "dataset_size": USER_DATASET_SIZE_FFHQ,
        },
        "small": {
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "color_root": USER_COLOR_ROOT_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "dataset_size": USER_DATASET_SIZE_SMALL,
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
ACTIVE_OUTPUT_DIR = PROFILE["output_dir"]
ACTIVE_DATASET_SIZE = PROFILE["dataset_size"]


def identity_func(align_shape, align_color, name_to_embed, **kwargs):
    return align_shape, align_color, name_to_embed


def role_key(role: str, stem: str) -> str:
    return f"{role}__{stem}"


def fs_cache_name(role: str, stem: str) -> str:
    return f"{role_key(role, stem)}.npz"


def align_cache_name(face_name: str, ref_role: str, ref_name: str) -> str:
    return f"{role_key('face', face_name)}_{role_key(ref_role, ref_name)}.npz"


def triplet_outputs_complete(output_dir: Path, face_name: str, shape_name: str, color_name: str) -> bool:
    """All 7 cache files for this triplet already exist -> can be skipped on resume."""
    required = [
        output_dir / "FS" / fs_cache_name("face", face_name),
        output_dir / "FS" / fs_cache_name("shape", shape_name),
        output_dir / "FS" / fs_cache_name("color", color_name),
        output_dir / "Align" / align_cache_name(face_name, "shape", shape_name),
        output_dir / "Align" / align_cache_name(face_name, "color", color_name),
        output_dir / "Masks" / align_cache_name(face_name, "shape", shape_name),
        output_dir / "Masks" / align_cache_name(face_name, "color", color_name),
    ]
    return all(path.exists() for path in required)


def load_corrupted_set(output_dir: Path) -> set[str]:
    """Image stems previously found corrupted; skipped without retrying."""
    path = output_dir / "corrupted_images.txt"
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def record_corrupted(output_dir: Path, stems: set[str]) -> None:
    path = output_dir / "corrupted_images.txt"
    with open(path, "w", encoding="utf-8") as handle:
        for stem in sorted(stems):
            handle.write(f"{stem}\n")


def is_corrupt_image_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "decode" in text
        or "corrupt" in text
        or "out of bound" in text
        or "truncated" in text
        or isinstance(error, (OSError, ValueError))
    )


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
) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    triplets: list[tuple[str, str, str]] = []

    if not face_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_FACE_ROOT}")
    if not shape_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_SHAPE_ROOT}")
    if not color_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_COLOR_ROOT}")

    if not allow_reuse and (
        size > len(face_images) or size > len(shape_images) or size > len(color_images)
    ):
        raise RuntimeError("Not enough unique images to sample all triplets without reuse.")

    face_pool = face_images.copy()
    shape_pool = shape_images.copy()
    color_pool = color_images.copy()

    for _ in range(size):
        if allow_reuse:
            face = rng.choice(face_images)
            shape = _sample_excluding_stems(rng, shape_images, {Path(face).stem})
            color = _sample_excluding_stems(rng, color_images, {Path(face).stem})
        else:
            if not face_pool or not shape_pool or not color_pool:
                raise RuntimeError("The image pool has been exhausted. Reduce ACTIVE_DATASET_SIZE or allow reuse.")
            face = rng.choice(face_pool)
            shape = _sample_excluding_stems(rng, shape_pool, {Path(face).stem})
            color = _sample_excluding_stems(rng, color_pool, {Path(face).stem})
            face_pool.remove(face)
            shape_pool.remove(shape)
            color_pool.remove(color)

        triplets.append((Path(face).stem, Path(shape).stem, Path(color).stem))

    return triplets


def build_model() -> HairFast_v8:
    if USER_USE_SATD_V8:
        if not USER_SATD_CHECKPOINT_V8:
            raise RuntimeError("USER_SATD_CHECKPOINT_V8 is empty while USER_USE_SATD_V8=True.")
        if not Path(USER_SATD_CHECKPOINT_V8).exists():
            raise FileNotFoundError(f"Cannot find USER_SATD_CHECKPOINT_V8: {USER_SATD_CHECKPOINT_V8}")

    model_args = get_parser_v8().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    model_args.use_satd_v8 = bool(USER_USE_SATD_V8)
    model_args.satd_checkpoint_v8 = USER_SATD_CHECKPOINT_V8
    model_args.satd_blend_v8 = USER_SATD_BLEND_V8
    model_args.satd_boundary_v8 = USER_SATD_BOUNDARY_V8
    model_args.eq8_reference_blend_v8 = USER_EQ8_REFERENCE_BLEND_V8
    # Keep SATD skin-only: never let it modify the target hair's F features, so
    # hair colour is rendered at baseline quality (fixes the A+C colour drift the
    # blending encoder could not recover).
    model_args.satd_hair_exclude_strength = USER_SATD_HAIR_EXCLUDE_STRENGTH

    hair_fast = HairFast_v8(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)
    return hair_fast


def main():
    seed_everything(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    face_images = list_image_files(ACTIVE_FACE_ROOT)
    shape_images = list_image_files(ACTIVE_SHAPE_ROOT)
    color_images = list_image_files(ACTIVE_COLOR_ROOT)
    _assert_unique_stems(face_images, "ACTIVE_FACE_ROOT")
    _assert_unique_stems(shape_images, "ACTIVE_SHAPE_ROOT")
    _assert_unique_stems(color_images, "ACTIVE_COLOR_ROOT")

    hair_fast = build_model()
    triplets = sample_triplets(
        face_images,
        shape_images,
        color_images,
        ACTIVE_DATASET_SIZE,
        USER_ALLOW_REUSE_ACROSS_TRIPLETS,
        USER_RANDOM_SEED,
    )

    corrupted = load_corrupted_set(ACTIVE_OUTPUT_DIR)
    done_count = 0
    skipped_done = 0
    skipped_corrupt = 0

    # dataset.exps is rebuilt each run and only lists successfully-generated
    # triplets, so a corrupted/skipped triplet never enters the training list.
    with open(ACTIVE_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps:
        for face_name, shape_name, color_name in triplets:
            # Resume: a triplet whose 7 cache files already exist is complete.
            if triplet_outputs_complete(ACTIVE_OUTPUT_DIR, face_name, shape_name, color_name):
                print(face_name, shape_name, color_name, file=f_exps, flush=True)
                skipped_done += 1
                done_count += 1
                continue

            # Skip triplets that reference a known-corrupted image without retry.
            triplet_stems = {face_name, shape_name, color_name}
            if triplet_stems & corrupted:
                skipped_corrupt += 1
                continue

            try:
                face_path = find_image_path(ACTIVE_FACE_ROOT, face_name)
                shape_path = find_image_path(ACTIVE_SHAPE_ROOT, shape_name)
                color_path = find_image_path(ACTIVE_COLOR_ROOT, color_name)

                align_shape, align_color, name_to_embed = hair_fast(
                    face_path,
                    shape_path,
                    color_path,
                    align_flag=True,
                )
            except Exception as error:  # noqa: BLE001 - skip bad images, keep going
                if is_corrupt_image_error(error):
                    # Record every stem in this triplet so the whole triplet is
                    # skipped next run; the actually-bad image is among them.
                    corrupted |= triplet_stems
                    record_corrupted(ACTIVE_OUTPUT_DIR, corrupted)
                    skipped_corrupt += 1
                    print(f"[skip] corrupted image in triplet ({face_name}, {shape_name}, {color_name}): {error}")
                    continue
                raise

            save_latents(ACTIVE_OUTPUT_DIR, "FS", fs_cache_name("face", face_name), latent_in=name_to_embed["face"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", fs_cache_name("shape", shape_name), latent_in=name_to_embed["shape"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", fs_cache_name("color", color_name), latent_in=name_to_embed["color"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "Align", align_cache_name(face_name, "shape", shape_name), latent_F=align_shape["latent_F_align"])
            save_latents(ACTIVE_OUTPUT_DIR, "Align", align_cache_name(face_name, "color", color_name), latent_F=align_color["latent_F_align"])
            save_latents(ACTIVE_OUTPUT_DIR, "Masks", align_cache_name(face_name, "shape", shape_name), remove_mask=build_remove_protect_mask(align_shape))
            save_latents(ACTIVE_OUTPUT_DIR, "Masks", align_cache_name(face_name, "color", color_name), remove_mask=build_remove_protect_mask(align_color))

            # Only record to dataset.exps AFTER all files are on disk, so an
            # interrupted triplet is re-run (not half-listed) next time.
            print(face_name, shape_name, color_name, file=f_exps, flush=True)
            done_count += 1

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"generated/kept {done_count} triplets "
          f"(resumed {skipped_done} already-done, skipped {skipped_corrupt} corrupted) "
          f"to {ACTIVE_OUTPUT_DIR / 'dataset.exps'}")
    print(f"face root: {ACTIVE_FACE_ROOT}")
    print(f"shape root: {ACTIVE_SHAPE_ROOT}")
    print(f"color root: {ACTIVE_COLOR_ROOT}")
    print(f"use satd v8: {USER_USE_SATD_V8}")
    print(f"satd checkpoint: {USER_SATD_CHECKPOINT_V8}")
    print(f"satd blend: {USER_SATD_BLEND_V8}")
    print(f"satd boundary: {USER_SATD_BOUNDARY_V8}")


if __name__ == "__main__":
    main()
