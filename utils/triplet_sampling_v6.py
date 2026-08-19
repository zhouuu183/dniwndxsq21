from __future__ import annotations

import random
from pathlib import Path


def assert_disjoint_stems(images_a: list[str], images_b: list[str], label_a: str, label_b: str):
    overlap = {Path(item).stem for item in images_a} & {Path(item).stem for item in images_b}
    if overlap:
        sample = ", ".join(sorted(list(overlap))[:10])
        raise RuntimeError(
            f"{label_a} and {label_b} contain duplicate stems that would overwrite saved files: {sample}"
        )


def _sample_excluding_stems(rng: random.Random, candidates: list[str], forbidden_stems: set[str]) -> str:
    valid = [item for item in candidates if Path(item).stem not in forbidden_stems]
    if not valid:
        raise RuntimeError("No valid image candidates remain after excluding duplicate stems.")
    return rng.choice(valid)


def sample_triplets(
    face_images: list[str],
    shape_images: list[str],
    color_images: list[str],
    size: int,
    allow_reuse: bool,
    seed: int,
    color_equals_shape: bool,
) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    triplets: list[tuple[str, str, str]] = []
    shared_shape_color_pool = shape_images is color_images

    if not face_images:
        raise RuntimeError("No face/source images were found.")
    if not shape_images:
        raise RuntimeError("No shape/reference images were found.")
    if not color_equals_shape and not color_images:
        raise RuntimeError("No color/reference images were found.")

    if not allow_reuse:
        if size > len(face_images):
            raise RuntimeError("Not enough unique face images to sample all triplets without reuse.")
        if color_equals_shape:
            if size > len(shape_images):
                raise RuntimeError("Not enough unique shape images to sample all triplets without reuse.")
        elif shared_shape_color_pool:
            if size * 2 > len(shape_images):
                raise RuntimeError("Not enough unique images to sample distinct shape/color pairs without reuse.")
        else:
            if size > len(shape_images) or size > len(color_images):
                raise RuntimeError("Not enough unique shape/color images to sample all triplets without reuse.")

    face_pool = face_images.copy()
    shape_pool = shape_images.copy()
    color_pool = shape_pool if shared_shape_color_pool else color_images.copy()
    for _ in range(size):
        if allow_reuse:
            face = rng.choice(face_images)
            shape = _sample_excluding_stems(rng, shape_images, {Path(face).stem})
            color = shape if color_equals_shape else _sample_excluding_stems(
                rng,
                color_images,
                {Path(face).stem, Path(shape).stem},
            )
        else:
            face = rng.choice(face_pool)
            shape = _sample_excluding_stems(rng, shape_pool, {Path(face).stem})
            color = shape if color_equals_shape else _sample_excluding_stems(
                rng,
                color_pool,
                {Path(face).stem, Path(shape).stem},
            )
            face_pool.remove(face)
            shape_pool.remove(shape)
            if not color_equals_shape:
                color_pool.remove(color)
        triplets.append((face, shape, color))
    return triplets
