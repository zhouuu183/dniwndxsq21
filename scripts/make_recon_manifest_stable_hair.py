"""Create one deterministic, unique 3000-image reconstruction manifest (Stable-Hair copy).

Byte-compatible twin of ``scripts/make_recon_manifest.py``: same seed 3407,
same enumeration/sort/shuffle logic, same row schema, same default output path
``input/eval_pairs_v5/celeba_hq_recon_seed3407_3000.jsonl``.  The
reconstruction manifest is model-independent, so HairFast and Stable-Hair MUST
share this exact file for their recon numbers to be comparable.  The script is
read-only by default: if the frozen manifest exists it is validated and reused.

Every row uses the same image for source, shape, colour and reference, i.e.
the model is asked to reconstruct its own input.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from PIL import Image


# ========================= User config: edit here only ========================
USER_IMAGE_ROOT = Path("/root/shared-nvme/HairFastGAN/celeba-1024/")
USER_MANIFEST = Path("input/eval_pairs_v5/celeba_hq_recon_seed3407_3000.jsonl")
USER_SAMPLE_COUNT = 3000
USER_RANDOM_SEED = 3407
USER_REBUILD_MANIFEST = False
USER_VERIFY_IMAGES = True
# ===============================================================================

def _find_repo_root(start: Path) -> Path:
    # Works no matter which scripts subdirectory the file lives in: walk up to
    # the first ancestor that looks like the project root.
    for candidate in (start, *start.parents):
        if (candidate / "input").is_dir():
            return candidate
    return start.parents[1] if len(start.parents) > 1 else start


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_images(root: Path) -> list[str]:
    if not root.is_dir():
        raise NotADirectoryError(f"CelebA-HQ directory does not exist: {root}")
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"No images found under {root}")
    if not USER_VERIFY_IMAGES:
        return files

    good: list[str] = []
    for relative in files:
        path = root / relative
        try:
            with Image.open(path) as image:
                image.verify()
            good.append(relative)
        except Exception as error:  # noqa: BLE001 - omit only unusable inputs
            print(f"Skip unreadable image: {path}: {error}")
    return good


def validate_existing(path: Path) -> int:
    seen: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if int(row.get("index", -1)) != count or str(row.get("mode", "")).lower() != "recon":
                raise RuntimeError(f"Invalid existing reconstruction manifest row {line_number}.")
            source = str(row.get("source_file", ""))
            if not source or source != str(row.get("shape_file", "")) or source != str(row.get("color_file", "")):
                raise RuntimeError(f"Existing row {count} does not use one image for all inputs.")
            if source in seen:
                raise RuntimeError(f"Existing manifest has duplicate image: {source}")
            seen.add(source)
    if count != USER_SAMPLE_COUNT:
        raise RuntimeError(f"Existing manifest has {count} rows, expected {USER_SAMPLE_COUNT}.")
    return count


def main() -> None:
    root = USER_IMAGE_ROOT.expanduser().resolve()
    manifest = repo_path(USER_MANIFEST).resolve()
    if manifest.exists() and not USER_REBUILD_MANIFEST:
        count = validate_existing(manifest)
        print(f"Reuse existing reconstruction manifest: {manifest}")
        print(f"Rows: {count}")
        print(f"SHA256: {sha256(manifest)}")
        print(
            "HairFast and Stable-Hair must use this exact file; compare the "
            "SHA256 with the HairFast run before comparing metrics."
        )
        return

    files = valid_images(root)
    if len(files) < USER_SAMPLE_COUNT:
        raise RuntimeError(
            f"Only {len(files)} usable images found, need {USER_SAMPLE_COUNT}."
        )
    rng = random.Random(USER_RANDOM_SEED)
    rng.shuffle(files)
    selected = files[:USER_SAMPLE_COUNT]

    rows: list[dict[str, object]] = []
    for index, source in enumerate(selected, start=1):
        rows.append(
            {
                "manifest_version": 1,
                "index": index,
                "mode": "recon",
                "output_file": f"{index:06d}.png",
                "sample_seed": USER_RANDOM_SEED + index - 1,
                "source_file": source,
                "shape_file": source,
                "color_file": source,
                "reference_file": source,
            }
        )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Created reconstruction manifest: {manifest}")
    print(f"Rows: {len(rows)}; unique images: {len(set(selected))}")
    print(f"Random seed: {USER_RANDOM_SEED}")
    print(f"SHA256: {sha256(manifest)}")
    print("Every row uses the same image for source, hairstyle, and colour.")


if __name__ == "__main__":
    main()
