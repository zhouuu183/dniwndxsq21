"""Generate fixed-manifest evaluation images with the combined v8 + v5 model.

The result directory contains only final images and is suitable as a method
directory for scripts/fid_metric.py.  Panels and the matched real source set
are stored in sibling directories so they cannot accidentally enter FID.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


# ========================= User Config: edit here only =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_MODE = "both"  # "full" or "both"
USER_CELEBA_HQ_DIR = Path("/root/shared-nvme/HairFastGAN/celeba-1024/")

# None uses the same convention as celeba_hq_make_eval_pairs_v5.py.
USER_MANIFEST_PATH: Path | None = None
USER_MANIFEST_SAMPLE_COUNT = 3000
USER_MANIFEST_SEED = 3407

USER_OUTPUT_ROOT = Path("output/celeba_hq_eval_v5_both")
USER_RUN_NAME = "ours_v8v5"
USER_SKIP_EXISTING_IMAGES = True
USER_VERIFY_EXISTING_OUTPUTS = True
USER_SAVE_PANELS = True
USER_SAVE_DEBUG_MASKS = False
USER_EXPORT_REAL_SOURCE_IMAGES = True
USER_VERIFY_INPUT_IMAGES = True
USER_ALIGN_INPUTS = False  # official CelebA-HQ images are already aligned/cropped
USER_EMPTY_CACHE_EVERY = 25

# Base HairFast dependencies (not newly trained in pp_train_v5).
# They are still required to reconstruct the upstream image features.
USER_STYLEGAN_CHECKPOINT = Path("pretrained_models/StyleGAN/ffhq.pt")
USER_ROTATE_CHECKPOINT = Path("pretrained_models/Rotate/rotate_best.pth")

# v8 stage weights.  These are upstream inputs to the v5 PP stage, not
# replacements for the v5 checkpoint.  Use the exact versions used to create
# the PP dataset and to train the selected PP checkpoint.
USER_BLENDING_V8_CHECKPOINT = Path(
    "/root/shared-nvme/HairFastGAN/checkpoints/blending_3000best.pth")
USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = Path(
    "/root/shared-nvme/HairFastGAN/checkpoints/satd_3000_best.pth")
USER_SATD_BLEND_V8 = 0.28
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0

# The user-trained v5 post-process checkpoint.  This is the new final stage
# containing the v5 face-detail and earring restoration network.
# pp_train_v5.py saves best_<epoch>.pth; replace best_0.pth with the selected
# validation-best file from your actual checkpoint directory.
USER_PP_V5_CHECKPOINT = Path("/root/shared-nvme/hairfast_ppmodify/output/pp_v5_checkpoints_full/best_26.pth")

# Keep this at the value used while generating the PP training data.  The
# current v5 pipeline used no extra image-space chroma correction by default.
USER_BLEND_CHROMA_CORRECT_STRENGTH = 0.0

# The model reads this encoder checkpoint from its established repository path.
USER_E4E_CHECKPOINT = Path("pretrained_models/encoder4editing/e4e_ffhq_encode.pt")
# ============================================================================


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v5 import HairFastV5, get_parser
from utils.seed import set_seed


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def normalize_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"full", "both"}:
        raise RuntimeError(f"USER_MODE must be 'full' or 'both', got {value!r}.")
    return mode


def default_manifest_path(mode: str) -> Path:
    count_tag = "all" if USER_MANIFEST_SAMPLE_COUNT <= 0 else str(USER_MANIFEST_SAMPLE_COUNT)
    return Path(
        "input/eval_pairs_v5"
    ) / f"celeba_hq_{mode}_seed{USER_MANIFEST_SEED}_{count_tag}.jsonl"


def resolve_manifest_path(mode: str) -> Path:
    if USER_MANIFEST_PATH is not None:
        return USER_MANIFEST_PATH
    default_path = default_manifest_path(mode)
    if default_path.exists():
        return default_path
    candidates = sorted(
        default_path.parent.glob(f"celeba_hq_{mode}_seed{USER_MANIFEST_SEED}_*.jsonl")
    )
    if len(candidates) == 1:
        # This also supports a different sample count without requiring the
        # user to duplicate it in the generation config.
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(str(item) for item in candidates[:8])
        raise RuntimeError(
            f"Several {mode} manifests use seed={USER_MANIFEST_SEED}: {names}. "
            "Set USER_MANIFEST_PATH explicitly."
        )
    return default_path


def image_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_and_validate_manifest(path: Path, mode: str, root: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation manifest does not exist: {path}. "
            "Run celeba_hq_make_eval_pairs_v5.py first."
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

    if not rows:
        raise RuntimeError(f"Evaluation manifest is empty: {path}")

    seen_outputs: set[str] = set()
    for expected_index, row in enumerate(rows, start=1):
        row_mode = str(row.get("mode", "")).strip().lower()
        if row_mode != mode:
            raise RuntimeError(
                f"Manifest row {expected_index} has mode={row_mode!r}; "
                f"the generation config requests mode={mode!r}."
            )
        if int(row.get("index", -1)) != expected_index:
            raise RuntimeError(f"Manifest index is not contiguous at row {expected_index}.")

        source = str(row.get("source_file", row.get("source_relpath", "")))
        shape = str(row.get("shape_file", row.get("shape_relpath", "")))
        if mode == "both":
            reference = str(
                row.get(
                    "reference_file",
                    row.get("reference_relpath", shape),
                )
            )
            color = str(row.get("color_file", row.get("color_relpath", reference)))
            if not reference or source == reference or shape != reference or color != reference:
                raise RuntimeError(
                    f"Manifest row {expected_index} is not a valid both pair: "
                    "source must differ from the one reference, and shape/color must be that same file."
                )
            row["shape_file"] = reference
            row["color_file"] = reference
            row["reference_file"] = reference
        else:
            color = str(row.get("color_file", row.get("color_relpath", "")))
            if not source or not shape or not color or len({source, shape, color}) != 3:
                raise RuntimeError(
                    f"Manifest row {expected_index} is not a valid full triplet: "
                    "source, shape, and color must be three different images."
                )

        if not source or not shape or not color:
            raise RuntimeError(f"Manifest row {expected_index} is missing an image path.")

        output_file = Path(str(row.get("output_file", f"{expected_index:06d}.png")))
        if output_file.name != str(output_file) or output_file.suffix.lower() != ".png":
            raise RuntimeError(f"Manifest row {expected_index} has an unsafe output filename.")
        if str(output_file) in seen_outputs:
            raise RuntimeError(f"Duplicate output filename in manifest: {output_file}")
        seen_outputs.add(str(output_file))
        row["output_file"] = str(output_file)
        row["source_file"] = source
        row["shape_file"] = shape
        row["color_file"] = color

        for role, relative_path in (
            ("source", source),
            ("shape", shape),
            ("color", color),
        ):
            path_value = image_path(root, relative_path)
            if not path_value.exists():
                raise FileNotFoundError(
                    f"Manifest row {expected_index} {role} image is missing: {path_value}"
                )
            if USER_VERIFY_INPUT_IMAGES:
                verify_image(path_value)

    return rows

def verify_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:  # noqa: BLE001 - include the exact input path
        raise RuntimeError(f"Unreadable input image: {path}: {error}") from error


def validate_checkpoints() -> None:
    required = (
        (USER_STYLEGAN_CHECKPOINT, "USER_STYLEGAN_CHECKPOINT"),
        (USER_ROTATE_CHECKPOINT, "USER_ROTATE_CHECKPOINT"),
        (USER_BLENDING_V8_CHECKPOINT, "USER_BLENDING_V8_CHECKPOINT"),
        (USER_PP_V5_CHECKPOINT, "USER_PP_V5_CHECKPOINT"),
        (USER_E4E_CHECKPOINT, "USER_E4E_CHECKPOINT"),
    )
    for path, label in required:
        if not Path(path).exists():
            raise FileNotFoundError(f"Cannot find {label}: {path}")
    if USER_USE_SATD_V8 and not USER_SATD_CHECKPOINT_V8.exists():
        raise FileNotFoundError(f"Cannot find USER_SATD_CHECKPOINT_V8: {USER_SATD_CHECKPOINT_V8}")


def make_model_args(run_dir: Path):
    args = get_parser().parse_args([])
    args.device = USER_DEVICE
    args.save_all = bool(USER_SAVE_DEBUG_MASKS)
    args.save_all_dir = run_dir / "debug"
    args.ckpt = str(USER_STYLEGAN_CHECKPOINT)
    args.rotate_checkpoint = str(USER_ROTATE_CHECKPOINT)
    args.blending_checkpoint = str(USER_BLENDING_V8_CHECKPOINT)
    args.pp_checkpoint = str(USER_PP_V5_CHECKPOINT)
    args.pp_v5_checkpoint = str(USER_PP_V5_CHECKPOINT)
    args.use_satd_v8 = bool(USER_USE_SATD_V8)
    args.satd_checkpoint_v8 = str(USER_SATD_CHECKPOINT_V8)
    args.satd_blend_v8 = float(USER_SATD_BLEND_V8)
    args.satd_boundary_v8 = int(USER_SATD_BOUNDARY_V8)
    args.eq8_reference_blend_v8 = float(USER_EQ8_REFERENCE_BLEND_V8)
    args.blend_chroma_correct_strength = float(USER_BLEND_CHROMA_CORRECT_STRENGTH)
    return args


def load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return T.functional.to_tensor(image.convert("RGB"))


def resize_chw(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[-2:]) == size:
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]


def save_panel(
    path: Path,
    source_path: Path,
    shape_path: Path,
    color_path: Path,
    result: torch.Tensor,
) -> None:
    result = result.detach().cpu().clamp(0, 1)
    size = tuple(result.shape[-2:])
    source = resize_chw(load_rgb_tensor(source_path), size)
    shape = resize_chw(load_rgb_tensor(shape_path), size)
    color = resize_chw(load_rgb_tensor(color_path), size)
    panel = torch.cat((source, shape, color, result), dim=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(panel, path)


def save_png_atomic(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep a recognized image suffix so torchvision/PIL can infer PNG format.
    temporary = path.with_name(path.stem + ".tmp.png")
    try:
        save_image(image.detach().cpu().clamp(0, 1), temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def image_suffix_for_real(path: Path) -> str:
    suffix = path.suffix.lower()
    return ".jpg" if suffix in {".jpg", ".jpeg"} else ".png"


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def export_real_sources(
    rows: list[dict[str, object]],
    root: Path,
    real_dir: Path,
) -> dict[int, str]:
    paths: dict[int, str] = {}
    for row in rows:
        index = int(row["index"])
        source_path = image_path(root, str(row["source_file"]))
        destination = real_dir / f"{index:06d}{image_suffix_for_real(source_path)}"
        link_or_copy(source_path, destination)
        paths[index] = str(destination)
    return paths


def image_files_in_directory(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def manifest_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_files(
    run_dir: Path,
    rows: list[dict[str, object]],
    manifest_path: Path,
    real_paths: dict[int, str],
    result_dir: Path,
    panel_dir: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = run_dir / "run_manifest.jsonl"
    with run_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            index = int(row["index"])
            record = dict(row)
            record.update(
                {
                    "source_path": str(image_path(USER_CELEBA_HQ_DIR, str(row["source_file"]))),
                    "shape_path": str(image_path(USER_CELEBA_HQ_DIR, str(row["shape_file"]))),
                    "color_path": str(image_path(USER_CELEBA_HQ_DIR, str(row["color_file"]))),
                    "result_path": str(result_dir / str(row["output_file"])),
                    "panel_path": str(panel_dir / str(row["output_file"])),
                    "real_source_path": real_paths.get(index, ""),
                }
            )
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    config = {
        "reference_mode": USER_MODE,
        "dataset_root": str(USER_CELEBA_HQ_DIR),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_digest(manifest_path),
        "run_name": USER_RUN_NAME,
        "result_dir": str(result_dir),
        "panel_dir": str(panel_dir),
        "real_source_dir": str(run_dir / "real_source"),
        "stylegan_checkpoint": str(USER_STYLEGAN_CHECKPOINT),
        "rotate_checkpoint": str(USER_ROTATE_CHECKPOINT),
        "blending_v8_checkpoint": str(USER_BLENDING_V8_CHECKPOINT),
        "pp_v5_checkpoint": str(USER_PP_V5_CHECKPOINT),
        "use_satd_v8": USER_USE_SATD_V8,
        "satd_checkpoint_v8": str(USER_SATD_CHECKPOINT_V8),
        "satd_blend_v8": USER_SATD_BLEND_V8,
        "satd_boundary_v8": USER_SATD_BOUNDARY_V8,
        "eq8_reference_blend_v8": USER_EQ8_REFERENCE_BLEND_V8,
        "blend_chroma_correct_strength": USER_BLEND_CHROMA_CORRECT_STRENGTH,
        "align_inputs": USER_ALIGN_INPUTS,
        "sample_count": len(rows),
        "seed_base": USER_MANIFEST_SEED,
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=True, indent=2)


def validate_reusable_run(run_dir: Path, manifest_path: Path) -> None:
    """Prevent skip-existing mode from mixing outputs from another model run."""
    if not USER_SKIP_EXISTING_IMAGES:
        return
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot safely reuse {run_dir}: its run_config.json is unreadable. "
            "Choose a new USER_RUN_NAME or set USER_SKIP_EXISTING_IMAGES=False."
        ) from error

    expected = {
        "reference_mode": USER_MODE,
        "manifest_sha256": manifest_digest(manifest_path),
        "stylegan_checkpoint": str(USER_STYLEGAN_CHECKPOINT),
        "rotate_checkpoint": str(USER_ROTATE_CHECKPOINT),
        "blending_v8_checkpoint": str(USER_BLENDING_V8_CHECKPOINT),
        "pp_v5_checkpoint": str(USER_PP_V5_CHECKPOINT),
        "use_satd_v8": USER_USE_SATD_V8,
        "satd_checkpoint_v8": str(USER_SATD_CHECKPOINT_V8),
        "satd_blend_v8": USER_SATD_BLEND_V8,
        "satd_boundary_v8": USER_SATD_BOUNDARY_V8,
        "eq8_reference_blend_v8": USER_EQ8_REFERENCE_BLEND_V8,
        "blend_chroma_correct_strength": USER_BLEND_CHROMA_CORRECT_STRENGTH,
        "align_inputs": USER_ALIGN_INPUTS,
    }
    mismatches = [
        key for key, value in expected.items()
        if str(previous.get(key)) != str(value)
    ]
    if mismatches:
        raise RuntimeError(
            f"{run_dir} was created with different evaluation settings: {mismatches}. "
            "Choose a new USER_RUN_NAME or set USER_SKIP_EXISTING_IMAGES=False."
        )


def validate_result_directory(result_dir: Path, expected: set[str]) -> None:
    actual = image_files_in_directory(result_dir)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} result images in {result_dir}; first: {missing[:5]}"
        )
    if extra:
        raise RuntimeError(
            f"Found {len(extra)} extra image files in {result_dir}; first: {extra[:5]}. "
            "Use a fresh USER_RUN_NAME or remove stale files before computing FID."
        )


def main() -> None:
    mode = normalize_mode(USER_MODE)
    if USER_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"USER_DEVICE={USER_DEVICE!r}, but CUDA is unavailable. "
            "Use a CUDA environment or explicitly choose USER_DEVICE='cpu'."
        )
    manifest_path = resolve_manifest_path(mode)
    validate_checkpoints()
    rows = load_and_validate_manifest(manifest_path, mode, USER_CELEBA_HQ_DIR)

    run_dir = USER_OUTPUT_ROOT / USER_RUN_NAME / mode
    result_dir = run_dir / "results"
    panel_dir = run_dir / "panels"
    real_dir = run_dir / "real_source"
    result_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_PANELS:
        panel_dir.mkdir(parents=True, exist_ok=True)
    if USER_EXPORT_REAL_SOURCE_IMAGES:
        real_dir.mkdir(parents=True, exist_ok=True)

    validate_reusable_run(run_dir, manifest_path)

    real_paths = {}
    if USER_EXPORT_REAL_SOURCE_IMAGES:
        real_paths = export_real_sources(rows, USER_CELEBA_HQ_DIR, real_dir)

    expected_outputs = {str(row["output_file"]) for row in rows}
    write_run_files(run_dir, rows, manifest_path, real_paths, result_dir, panel_dir)
    shutil.copy2(manifest_path, run_dir / "pairs.jsonl")

    set_seed(USER_MANIFEST_SEED)
    model = HairFastV5(make_model_args(run_dir))
    generated = 0
    skipped = 0

    with torch.inference_mode():
        for row in tqdm(rows, desc=f"Generate v8+v5 evaluation images ({mode})"):
            output_file = str(row["output_file"])
            output_path = result_dir / output_file
            source_path = image_path(USER_CELEBA_HQ_DIR, str(row["source_file"]))
            shape_path = image_path(USER_CELEBA_HQ_DIR, str(row["shape_file"]))
            color_path = image_path(USER_CELEBA_HQ_DIR, str(row["color_file"]))

            if USER_SKIP_EXISTING_IMAGES and output_path.exists():
                if USER_VERIFY_EXISTING_OUTPUTS:
                    verify_image(output_path)
                skipped += 1
                if USER_SAVE_PANELS and not (panel_dir / output_file).exists():
                    save_panel(
                        panel_dir / output_file,
                        source_path,
                        shape_path,
                        color_path,
                        load_rgb_tensor(output_path),
                    )
                continue

            sample_seed = int(row.get("sample_seed", USER_MANIFEST_SEED + int(row["index"]) - 1))
            try:
                result = model(
                    source_path,
                    shape_path,
                    color_path,
                    align=USER_ALIGN_INPUTS,
                    seed=sample_seed,
                    use_satd_v8=USER_USE_SATD_V8,
                    satd_blend_v8=USER_SATD_BLEND_V8,
                    satd_boundary_v8=USER_SATD_BOUNDARY_V8,
                    eq8_reference_blend_v8=USER_EQ8_REFERENCE_BLEND_V8,
                    blend_chroma_correct_strength=USER_BLEND_CHROMA_CORRECT_STRENGTH,
                    exp_name=Path(output_file).stem,
                )
                if isinstance(result, tuple):
                    result = result[0]
            except Exception as error:  # noqa: BLE001 - identify the failed fixed row
                raise RuntimeError(
                    f"Generation failed at manifest row {row['index']} "
                    f"(source={source_path}, shape={shape_path}, color={color_path}): {error}"
                ) from error

            save_png_atomic(result, output_path)
            if USER_SAVE_PANELS:
                save_panel(panel_dir / output_file, source_path, shape_path, color_path, result)
            generated += 1

            if (
                USER_DEVICE.startswith("cuda")
                and USER_EMPTY_CACHE_EVERY > 0
                and generated % USER_EMPTY_CACHE_EVERY == 0
            ):
                torch.cuda.empty_cache()

    validate_result_directory(result_dir, expected_outputs)
    if USER_EXPORT_REAL_SOURCE_IMAGES:
        real_expected = {
            f"{int(row['index']):06d}{image_suffix_for_real(image_path(USER_CELEBA_HQ_DIR, str(row['source_file'])))}"
            for row in rows
        }
        validate_result_directory(real_dir, real_expected)

    print(f"Reference mode: {mode}")
    print(f"Manifest: {manifest_path}")
    print(f"Manifest SHA256: {manifest_digest(manifest_path)}")
    print(f"Rows: {len(rows)}")
    print(f"Generated this run: {generated}; skipped existing: {skipped}")
    print(f"Results (use this directory for FID): {result_dir}")
    if USER_EXPORT_REAL_SOURCE_IMAGES:
        print(f"Matched real source images (use as FID real set): {real_dir}")
    if USER_SAVE_PANELS:
        print(f"Panels (do not pass to FID): {panel_dir}")
    print(f"Run config: {run_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
