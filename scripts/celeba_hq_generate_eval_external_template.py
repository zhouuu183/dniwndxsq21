"""Adapter template for evaluating an external hair-transfer method on fixed pairs.

This script deliberately owns the evaluation contract, but not model inference.
Replace ``infer_one`` with one call into the other model.  Do not create pairs
here: all methods must consume the same frozen JSONL manifest.

The only directory that may be passed to scripts/fid_metric.py is ``results``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


# ========================= User config: edit here only ========================
USER_MODE = "both"  # "both" (source + one reference) or "full" (three inputs)
USER_CELEBA_HQ_DIR = Path("/root/shared-nvme/HairFastGAN/celeba-1024")
USER_MANIFEST_PATH = Path(
    "input/eval_pairs_v5/celeba_hq_both_seed3407_3000.jsonl"
)
USER_EXPECTED_SAMPLE_COUNT = 3000

# Use a new name when the external model, checkpoint, preprocessing, or any
# inference setting changes.  This prevents stale images being mixed into FID.
USER_OUTPUT_ROOT = Path("output/celeba_hq_eval_external")
USER_METHOD_RUN_NAME = "external_method_checkpoint_name"
USER_MODEL_DESCRIPTION = "Replace infer_one() with the external model and record its checkpoint/settings here."
USER_SKIP_EXISTING_IMAGES = True
USER_VERIFY_INPUT_IMAGES = True
# ==============================================================================


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_image(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def verify_rgb_image(path: Path, label: str) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.mode not in {"RGB", "RGBA", "L", "P", "CMYK"}:
                raise RuntimeError(f"unsupported image mode {image.mode!r}")
    except Exception as error:  # noqa: BLE001 - retain the exact file in the error
        raise RuntimeError(f"Unreadable {label}: {path}: {error}") from error


def load_manifest(path: Path, mode: str, image_root: Path) -> list[dict[str, object]]:
    if mode not in {"both", "full"}:
        raise RuntimeError(f"USER_MODE must be 'both' or 'full', got {mode!r}.")
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest is missing: {path}. Create it once with "
            "scripts/celeba_hq_make_eval_pairs_v5.py."
        )

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"Manifest row {line_number} is not a JSON object.")
            rows.append(row)

    if len(rows) != USER_EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected {USER_EXPECTED_SAMPLE_COUNT} rows, found {len(rows)} in {path}. "
            "Do not silently compare different sample counts."
        )

    seen_outputs: set[str] = set()
    for expected_index, row in enumerate(rows, start=1):
        if int(row.get("index", -1)) != expected_index:
            raise RuntimeError(f"Manifest index is not contiguous at row {expected_index}.")
        if str(row.get("mode", "")).strip().lower() != mode:
            raise RuntimeError(f"Manifest row {expected_index} has a different mode.")

        source = str(row.get("source_file", row.get("source_relpath", "")))
        shape = str(row.get("shape_file", row.get("shape_relpath", "")))
        color = str(row.get("color_file", row.get("color_relpath", "")))
        if not source or not shape or not color:
            raise RuntimeError(f"Manifest row {expected_index} is missing an input path.")
        if mode == "both":
            reference = str(row.get("reference_file", shape))
            if source == reference or shape != reference or color != reference:
                raise RuntimeError(
                    f"Manifest row {expected_index} is not a valid both pair."
                )
        elif len({source, shape, color}) != 3:
            raise RuntimeError(
                f"Manifest row {expected_index} is not a valid full triplet."
            )

        output_file = Path(str(row.get("output_file", f"{expected_index:06d}.png")))
        if output_file.name != str(output_file) or output_file.suffix.lower() != ".png":
            raise RuntimeError(f"Unsafe output filename at row {expected_index}: {output_file}")
        if str(output_file) in seen_outputs:
            raise RuntimeError(f"Duplicate output filename: {output_file}")
        seen_outputs.add(str(output_file))

        row["source_file"] = source
        row["shape_file"] = shape
        row["color_file"] = color
        row["output_file"] = str(output_file)
        for role, relative_path in (("source", source), ("shape", shape), ("color", color)):
            input_path = resolve_image(image_root, relative_path)
            if not input_path.is_file():
                raise FileNotFoundError(
                    f"Manifest row {expected_index} {role} image is missing: {input_path}"
                )
            if USER_VERIFY_INPUT_IMAGES:
                verify_rgb_image(input_path, f"{role} input")
    return rows


def infer_one(
    source_path: Path,
    shape_path: Path,
    color_path: Path,
    seed: int,
) -> Image.Image:
    """Run one external-model inference and return its final image.

    Replace only this function.  Examples of the expected mapping:
      * both: external(source_path, shape_path, seed=seed)
      * full: external(source_path, shape_path, color_path, seed=seed)

    In both mode ``shape_path == color_path`` by design.  If the external API
    always takes three image arguments, pass both paths unchanged; do not pick
    another colour reference.  Return the final image only, not a grid/panel.
    """
    del source_path, shape_path, color_path, seed
    raise NotImplementedError(
        "Connect infer_one() to the external model before running this template."
    )


def save_png_atomic(image: Image.Image, output_path: Path) -> None:
    # FID loader requires a consistent three-channel tensor across every image.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp.png")
    try:
        image.convert("RGB").save(temporary, format="PNG")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def image_files(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def validate_result_directory(result_dir: Path, expected: set[str]) -> None:
    actual = image_files(result_dir)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} (first: {missing[:5]})")
        if extra:
            details.append(f"extra {len(extra)} (first: {extra[:5]})")
        raise RuntimeError(f"Invalid result set in {result_dir}: {'; '.join(details)}")
    for filename in sorted(expected):
        verify_rgb_image(result_dir / filename, "result image")


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if (
            destination.stat().st_size == source.stat().st_size
            and file_sha256(destination) == file_sha256(source)
        ):
            return
        raise RuntimeError(f"Existing real-source file conflicts with manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_run_files(run_dir: Path, manifest_path: Path, rows: list[dict[str, object]]) -> None:
    config_path = run_dir / "run_config.json"
    identity = {
        "mode": USER_MODE,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_root": str(USER_CELEBA_HQ_DIR.resolve()),
        "sample_count": len(rows),
        "model_description": USER_MODEL_DESCRIPTION,
    }
    result_dir = run_dir / "results"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != identity and image_files(result_dir):
            raise RuntimeError(
                "This run directory contains results from a different manifest or model setting. "
                "Choose a new USER_METHOD_RUN_NAME."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(identity, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(manifest_path, run_dir / "pairs.jsonl")


def main() -> None:
    mode = USER_MODE.strip().lower()
    manifest_path = USER_MANIFEST_PATH.expanduser().resolve()
    image_root = USER_CELEBA_HQ_DIR.expanduser().resolve()
    rows = load_manifest(manifest_path, mode, image_root)

    run_dir = USER_OUTPUT_ROOT / USER_METHOD_RUN_NAME / mode
    result_dir = run_dir / "results"
    real_source_dir = run_dir / "real_source"
    write_run_files(run_dir, manifest_path, rows)
    result_dir.mkdir(parents=True, exist_ok=True)
    real_source_dir.mkdir(parents=True, exist_ok=True)

    expected_outputs = {str(row["output_file"]) for row in rows}
    # The real set is exactly the source image in every fixed pair/triplet.
    expected_real_sources: set[str] = set()
    for row in rows:
        source_path = resolve_image(image_root, str(row["source_file"]))
        suffix = ".jpg" if source_path.suffix.lower() in {".jpg", ".jpeg"} else ".png"
        filename = f"{int(row['index']):06d}{suffix}"
        link_or_copy(source_path, real_source_dir / filename)
        expected_real_sources.add(filename)

    for row in tqdm(rows, desc=f"Generate external evaluation images ({mode})"):
        output_path = result_dir / str(row["output_file"])
        if USER_SKIP_EXISTING_IMAGES and output_path.exists():
            verify_rgb_image(output_path, "existing result image")
            continue
        source_path = resolve_image(image_root, str(row["source_file"]))
        shape_path = resolve_image(image_root, str(row["shape_file"]))
        color_path = resolve_image(image_root, str(row["color_file"]))
        # Use the stored seed exactly; it is part of the frozen evaluation row.
        seed = int(row["sample_seed"])
        try:
            result = infer_one(source_path, shape_path, color_path, seed)
        except Exception as error:  # noqa: BLE001 - include the fixed row for reruns
            raise RuntimeError(
                f"External inference failed at row {row['index']} "
                f"(source={source_path}, shape={shape_path}, color={color_path})."
            ) from error
        if not isinstance(result, Image.Image):
            raise TypeError("infer_one() must return a PIL.Image.Image final result.")
        save_png_atomic(result, output_path)

    validate_result_directory(result_dir, expected_outputs)
    validate_result_directory(real_source_dir, expected_real_sources)
    print(f"Mode: {mode}; rows: {len(rows)}")
    print(f"Manifest SHA256: {file_sha256(manifest_path)}")
    print(f"FID real set: {real_source_dir}")
    print(f"FID generated set: {result_dir}")


if __name__ == "__main__":
    main()
