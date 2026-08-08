"""Create a deterministic CelebA-HQ evaluation manifest.

Run this script once for each reference mode.  The resulting JSONL file is the
contract shared by every hair-transfer method in an evaluation comparison.
The generator script never creates new pairs; it only consumes this file.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from PIL import Image


# ========================= User Config: edit here only =========================
USER_MODE = "full"  # "full": source + shape ref + color ref; "both": source + one ref
USER_CELEBA_HQ_DIR = Path("/root/shared-nvme/HairFastGAN/celeba-1024/")

# 0 means all images.  For a paper comparison, keep this value fixed after the
# manifest has been created.  A few thousand images is usually practical.
USER_SAMPLE_COUNT = 3000
USER_RANDOM_SEED = 3407

# False gives sampling without replacement within each role.  True allows an
# image to appear in several rows; roles in one row remain distinct either way.
USER_ALLOW_REUSE_ACROSS_SAMPLES = False

# None uses input/eval_pairs_v5/celeba_hq_<mode>_seed<seed>_<count>.jsonl.
# Set an explicit path if the same manifest must be stored elsewhere.
USER_MANIFEST_PATH: Path | None = None
USER_REBUILD_MANIFEST = False
USER_VERIFY_IMAGES = True
USER_WRITE_TEXT_LIST = True
# ============================================================================


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def normalize_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"full", "both"}:
        raise RuntimeError(f"USER_MODE must be 'full' or 'both', got {value!r}.")
    return mode


def count_tag(count: int) -> str:
    return "all" if count <= 0 else str(count)


def default_manifest_path(mode: str) -> Path:
    return Path(
        "input/eval_pairs_v5"
    ) / f"celeba_hq_{mode}_seed{USER_RANDOM_SEED}_{count_tag(USER_SAMPLE_COUNT)}.jsonl"


def relative_image_files(root: Path) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"CelebA-HQ directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"CelebA-HQ path is not a directory: {root}")

    files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    files.sort(key=str.casefold)
    if not files:
        raise RuntimeError(f"No supported images found under {root}")
    if len(files) != len(set(files)):
        raise RuntimeError("Duplicate relative image paths were found in CelebA-HQ.")
    return files


def check_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:  # noqa: BLE001 - report the exact bad input
        raise RuntimeError(f"Unreadable image: {path}: {error}") from error


def choose_count(total: int) -> int:
    count = total if USER_SAMPLE_COUNT <= 0 else int(USER_SAMPLE_COUNT)
    if count <= 0:
        raise RuntimeError("USER_SAMPLE_COUNT must be positive or zero for all images.")
    if not USER_ALLOW_REUSE_ACROSS_SAMPLES and count > total:
        raise RuntimeError(
            f"USER_SAMPLE_COUNT={count} is larger than the {total} available images. "
            "Reduce it or set USER_ALLOW_REUSE_ACROSS_SAMPLES=True."
        )
    required_roles = 3 if normalize_mode(USER_MODE) == "full" else 2
    if total < required_roles:
        raise RuntimeError(
            f"At least {required_roles} different images are required for mode={USER_MODE}."
        )
    return count


def sample_with_constraints(
    files: list[str],
    count: int,
    forbidden: list[set[str]],
    rng: random.Random,
) -> list[str]:
    """Sample one role while enforcing row-local exclusions.

    This branch is used only when reuse is explicitly enabled.  Choosing from
    the filtered pool at each row keeps the manifest valid even when the same
    source is selected more than once.
    """
    result: list[str] = []
    for index in range(count):
        candidates = [item for item in files if item not in forbidden[index]]
        if not candidates:
            raise RuntimeError(
                f"No valid reference remains for row {index + 1}; "
                "disable the exclusion or provide more CelebA-HQ images."
            )
        result.append(rng.choice(candidates))
    return result


def build_rows(files: list[str], mode: str, count: int) -> list[dict[str, object]]:
    rng = random.Random(USER_RANDOM_SEED)

    if USER_ALLOW_REUSE_ACROSS_SAMPLES:
        source = [rng.choice(files) for _ in range(count)]
        shape = sample_with_constraints(
            files,
            count,
            [{source[index]} for index in range(count)],
            rng,
        )

        if mode == "both":
            color = shape.copy()
        else:
            color = sample_with_constraints(
                files,
                count,
                [
                    {source[index], shape[index]}
                    for index in range(count)
                ],
                rng,
            )
    else:
        # One shuffled circular order plus different non-zero offsets gives a
        # guaranteed derangement when all roles use the same CelebA-HQ root.
        # It is deterministic, uses each role without replacement, and avoids
        # the late-row dead ends of greedy random sampling.
        order = files.copy()
        rng.shuffle(order)
        source = order[:count]
        offsets = list(range(1, len(order)))
        rng.shuffle(offsets)
        shape_offset = offsets[0]
        shape = [order[(index + shape_offset) % len(order)] for index in range(count)]
        if mode == "both":
            color = shape.copy()
        else:
            color_offset = next(
                (offset for offset in offsets[1:] if offset != shape_offset),
                None,
            )
            if color_offset is None:
                raise RuntimeError("full mode needs at least three distinct images.")
            color = [order[(index + color_offset) % len(order)] for index in range(count)]

    rows: list[dict[str, object]] = []
    for index in range(count):
        source_file = source[index]
        shape_file = shape[index]
        color_file = color[index]
        if mode == "both":
            if source_file == shape_file:
                raise RuntimeError(f"Pairing collision at row {index + 1}.")
        elif len({source_file, shape_file, color_file}) != 3:
            raise RuntimeError(f"Triplet collision at row {index + 1}.")

        row: dict[str, object] = {
            "manifest_version": 1,
            "index": index + 1,
            "mode": mode,
            "output_file": f"{index + 1:06d}.png",
            "sample_seed": USER_RANDOM_SEED + index,
            "source_file": source_file,
            "shape_file": shape_file,
            "color_file": color_file,
            "source_relpath": source_file,
            "shape_relpath": shape_file,
            "color_relpath": color_file,
        }
        if mode == "both":
            row["reference_file"] = shape_file
            row["reference_relpath"] = shape_file
        rows.append(row)
    return rows


def validate_rows(path: Path, rows: list[dict[str, object]], mode: str, root: Path) -> None:
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    for expected_index, row in enumerate(rows, start=1):
        if str(row.get("mode", "")).lower() != mode:
            raise RuntimeError(f"Manifest row {expected_index} has the wrong mode.")
        if int(row.get("index", -1)) != expected_index:
            raise RuntimeError(f"Manifest row {expected_index} has a non-contiguous index.")
        source = str(row.get("source_file", row.get("source_relpath", "")))
        shape = str(row.get("shape_file", row.get("shape_relpath", "")))
        color = str(row.get("color_file", row.get("color_relpath", shape)))
        if not source or not shape or not color:
            raise RuntimeError(f"Manifest row {expected_index} is missing an image path.")
        if mode == "both":
            reference = str(row.get("reference_file", shape))
            if shape != color or reference != shape:
                raise RuntimeError(f"both row {expected_index} must use one reference image twice.")
        elif len({source, shape, color}) != 3:
            raise RuntimeError(f"full row {expected_index} contains repeated images.")
        for role, relative_path in (("source", source), ("shape", shape), ("color", color)):
            image_path = Path(relative_path)
            if image_path.is_absolute():
                image_path = Path(relative_path)
            else:
                image_path = root / image_path
            if not image_path.exists():
                raise FileNotFoundError(f"Manifest row {expected_index} {role} image is missing: {image_path}")
            if USER_VERIFY_IMAGES:
                check_image(image_path)


def load_existing(path: Path, mode: str, root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON on {path}:{line_number}: {error}") from error
    validate_rows(path, rows, mode, root)
    return rows


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_outputs(path: Path, rows: list[dict[str, object]], mode: str, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary.replace(path)

    metadata = {
        "manifest_version": 1,
        "mode": mode,
        "dataset_root": str(root),
        "sample_count": len(rows),
        "requested_sample_count": USER_SAMPLE_COUNT,
        "random_seed": USER_RANDOM_SEED,
        "allow_reuse_across_samples": USER_ALLOW_REUSE_ACROSS_SAMPLES,
        "sha256": file_digest(path),
    }
    metadata_path = path.with_suffix(".meta.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)

    if USER_WRITE_TEXT_LIST:
        text_path = path.with_suffix(".txt")
        with text_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                if mode == "both":
                    handle.write(f"{row['source_file']} {row['reference_file']}\n")
                else:
                    handle.write(
                        f"{row['source_file']} {row['shape_file']} {row['color_file']}\n"
                    )
        if mode == "both":
            # Some model APIs always accept three arguments.  Repeating the
            # single reference here preserves the same binary pair semantics.
            triplet_text_path = path.with_name(path.stem + "_as_triplets.txt")
            with triplet_text_path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    reference = row["reference_file"]
                    handle.write(f"{row['source_file']} {reference} {reference}\n")


def main() -> None:
    mode = normalize_mode(USER_MODE)
    root = USER_CELEBA_HQ_DIR
    manifest_path = USER_MANIFEST_PATH or default_manifest_path(mode)

    if manifest_path.exists() and not USER_REBUILD_MANIFEST:
        rows = load_existing(manifest_path, mode, root)
        print(f"Reused existing manifest: {manifest_path}")
        print(f"Mode: {mode}; rows: {len(rows)}; sha256: {file_digest(manifest_path)}")
        print("Keep USER_REBUILD_MANIFEST=False when regenerating outputs for comparison.")
        return

    files = relative_image_files(root)
    count = choose_count(len(files))
    rows = build_rows(files, mode, count)
    validate_rows(manifest_path, rows, mode, root)
    save_outputs(manifest_path, rows, mode, root)

    print(f"Created manifest: {manifest_path}")
    print(f"Mode: {mode}; CelebA-HQ images: {len(files)}; rows: {len(rows)}")
    print(f"Random seed: {USER_RANDOM_SEED}")
    print(f"SHA256: {file_digest(manifest_path)}")
    if USER_WRITE_TEXT_LIST:
        print(f"Text list: {manifest_path.with_suffix('.txt')}")
        if mode == "both":
            print(f"Three-column both list: {manifest_path.with_name(manifest_path.stem + '_as_triplets.txt')}")
    print(
        "Use this exact manifest for every method. Set USER_REBUILD_MANIFEST=False "
        "after this file has been created."
    )


if __name__ == "__main__":
    main()
