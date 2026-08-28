"""Generate fixed 3000-sample results for the original HairFast baseline.

The baseline is loaded from USER_BASELINE_ROOT and must be the untouched
no-suffix ``hair_swap.HairFast`` implementation.  The JSONL manifest is shared
with every other method; this script never creates a new pairing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms import functional as tvf


# ========================= User config: edit here only ========================
USER_MODE = "both"  # "both" or "full"
USER_DEVICE = "cuda"
USER_CUDA_VISIBLE_DEVICES = "0"

# This must point to a separate untouched author checkout.  Relative weight
# paths below are resolved from this directory; absolute paths may be entered.
USER_BASELINE_ROOT = Path("/data/coding/HairFastGAN/HairFastGAN-main")

# Fill these with the exact baseline weights you want to evaluate.  These four
# are the method/checkpoint inputs passed directly to hair_swap.HairFast.
USER_STYLEGAN_CHECKPOINT = Path("pretrained_models/StyleGAN/ffhq.pt")
USER_ROTATE_CHECKPOINT = Path("pretrained_models/Rotate/rotate_best.pth")
USER_BLENDING_CHECKPOINT = Path("pretrained_models/Blending/checkpoint.pth")
USER_PP_CHECKPOINT = Path("pretrained_models/PostProcess/pp_model.pth")
USER_CELEBA_HQ_DIR = Path("/root/shared-nvme/HairFastGAN/celeba-1024")
# None derives input/eval_pairs_v5/celeba_hq_<USER_MODE>_seed3407_3000.jsonl.
# Set an explicit path only when deliberately using a non-default manifest.
USER_MANIFEST_PATH: Path | None = None
USER_EXPECTED_SAMPLE_COUNT = 3000

USER_OUTPUT_ROOT = Path("output/celeba_hq_baseline_fid")
USER_RUN_NAME = "author_baseline"
USER_SKIP_EXISTING_IMAGES = True
USER_VERIFY_INPUT_IMAGES = True
# Missing/corrupt input rows are skipped and recorded instead of aborting.
# The resulting FID set may contain fewer than USER_EXPECTED_SAMPLE_COUNT rows.
USER_SKIP_INVALID_INPUTS = True
USER_MODEL_DESCRIPTION = "Untouched author HairFast baseline (no-suffix hair_swap.HairFast)"
# ===============================================================================


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_config_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def default_manifest_path(mode: str) -> Path:
    return REPO_ROOT / "input/eval_pairs_v5" / f"celeba_hq_{mode}_seed3407_3000.jsonl"


def resolve_image(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def verify_image(path: Path, label: str) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:  # noqa: BLE001 - retain the exact file path
        raise RuntimeError(f"Unreadable {label}: {path}: {error}") from error


def verify_rgb_image(path: Path, label: str) -> None:
    verify_image(path, label)
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise RuntimeError(f"{label} must be RGB, got {image.mode!r}: {path}")


def load_manifest(
    path: Path,
    mode: str,
    image_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if mode not in {"both", "full"}:
        raise ValueError(f"USER_MODE must be 'both' or 'full', got {mode!r}.")
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest is missing: {path}. First run "
            "scripts/celeba_hq_make_eval_pairs_v5.py with USER_SAMPLE_COUNT=3000."
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
            f"Expected exactly {USER_EXPECTED_SAMPLE_COUNT} rows, found {len(rows)}."
        )

    seen_outputs: set[str] = set()
    valid_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    for expected_index, row in enumerate(rows, start=1):
        if int(row.get("index", -1)) != expected_index:
            raise RuntimeError(f"Manifest index is not contiguous at row {expected_index}.")
        if str(row.get("mode", "")).strip().lower() != mode:
            raise RuntimeError(f"Manifest row {expected_index} has the wrong mode.")

        source = str(row.get("source_file", row.get("source_relpath", "")))
        shape = str(row.get("shape_file", row.get("shape_relpath", "")))
        color = str(row.get("color_file", row.get("color_relpath", "")))
        if not source or not shape or not color:
            raise RuntimeError(f"Manifest row {expected_index} is missing an input path.")
        if "sample_seed" not in row:
            raise RuntimeError(f"Manifest row {expected_index} is missing sample_seed.")

        if mode == "both":
            reference = str(row.get("reference_file", shape))
            if source == reference or shape != reference or color != reference:
                raise RuntimeError(f"Invalid both pair at row {expected_index}.")
        elif len({source, shape, color}) != 3:
            raise RuntimeError(f"Invalid full triplet at row {expected_index}.")

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
        invalid_reason: str | None = None
        for role, relative_path in (("source", source), ("shape", shape), ("color", color)):
            image_path = resolve_image(image_root, relative_path)
            if not image_path.is_file():
                invalid_reason = (
                    f"Manifest row {expected_index} {role} image is missing: {image_path}"
                )
                break
            if USER_VERIFY_INPUT_IMAGES:
                try:
                    verify_image(image_path, f"{role} input")
                except RuntimeError as error:
                    invalid_reason = str(error)
                    break
        if invalid_reason is not None:
            if not USER_SKIP_INVALID_INPUTS:
                raise FileNotFoundError(invalid_reason)
            skipped_rows.append(
                {
                    "index": expected_index,
                    "output_file": str(output_file),
                    "source_file": source,
                    "shape_file": shape,
                    "color_file": color,
                    "reason": invalid_reason,
                }
            )
            continue
        valid_rows.append(row)

    if skipped_rows:
        print(
            f"Skipped {len(skipped_rows)} invalid input rows; "
            f"using {len(valid_rows)} valid rows."
        )
        for skipped in skipped_rows[:10]:
            print(f"  [skip row {skipped['index']}] {skipped['reason']}")
    if not valid_rows:
        raise RuntimeError(
            "No valid manifest rows remain. Check USER_CELEBA_HQ_DIR and the input files."
        )
    return valid_rows, skipped_rows


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")


def resolve_baseline_file(root: Path, configured: Path) -> Path:
    path = configured.expanduser()
    return path if path.is_absolute() else root / path


def import_baseline(baseline_root: Path):
    require_file(baseline_root / "hair_swap.py", "Official baseline hair_swap.py")
    sys.path.insert(0, str(baseline_root))
    from hair_swap import HairFast, get_parser  # type: ignore[import-not-found]

    return HairFast, get_parser


def build_baseline(baseline_root: Path, device: str):
    HairFast, get_parser = import_baseline(baseline_root)
    checkpoint_paths = {
        "StyleGAN checkpoint": resolve_baseline_file(baseline_root, USER_STYLEGAN_CHECKPOINT),
        "rotate checkpoint": resolve_baseline_file(baseline_root, USER_ROTATE_CHECKPOINT),
        "blending checkpoint": resolve_baseline_file(baseline_root, USER_BLENDING_CHECKPOINT),
        "author PP checkpoint": resolve_baseline_file(baseline_root, USER_PP_CHECKPOINT),
    }
    for label, path in checkpoint_paths.items():
        require_file(path, label)

    args = get_parser().parse_args([])
    args.device = device
    args.ckpt = str(checkpoint_paths["StyleGAN checkpoint"])
    args.rotate_checkpoint = str(checkpoint_paths["rotate checkpoint"])
    args.blending_checkpoint = str(checkpoint_paths["blending checkpoint"])
    args.pp_checkpoint = str(checkpoint_paths["author PP checkpoint"])
    args.save_all = False
    return HairFast(args), checkpoint_paths


def tensor_to_rgb_image(value: torch.Tensor) -> Image.Image:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Baseline output must be a Tensor, got {type(value)!r}.")
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"Expected one baseline image, got {tuple(value.shape)}")
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"Expected RGB [3,H,W], got {tuple(value.shape)}")
    return tvf.to_pil_image(value.detach().cpu().float().clamp(0, 1)).convert("RGB")


def image_files(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def validate_directory(directory: Path, expected: set[str], label: str) -> None:
    actual = image_files(directory)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"Invalid {label} directory {directory}; missing={len(missing)}, extra={len(extra)}."
        )
    for filename in expected:
        verify_rgb_image(directory / filename, label)


def export_real_sources(rows: list[dict[str, object]], image_root: Path, real_dir: Path) -> set[str]:
    real_dir.mkdir(parents=True, exist_ok=True)
    expected: set[str] = set()
    for row in tqdm(rows, desc="Export real source images"):
        index = int(row["index"])
        destination = real_dir / f"{index:06d}.png"
        source_path = resolve_image(image_root, str(row["source_file"]))
        if destination.exists():
            with Image.open(destination) as existing, Image.open(source_path) as source:
                if existing.convert("RGB").tobytes() != source.convert("RGB").tobytes():
                    raise RuntimeError(f"Existing real source differs from manifest: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source_path) as image:
                image.convert("RGB").save(destination, format="PNG")
        expected.add(destination.name)
    return expected


def prepare_run_config(
    run_dir: Path,
    manifest_path: Path,
    image_root: Path,
    baseline_root: Path,
    checkpoint_paths: dict[str, Path],
    mode: str,
    sample_count: int,
) -> None:
    config = {
        "mode": mode,
        "sample_count": sample_count,
        "dataset_root": str(image_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "baseline_root": str(baseline_root),
        "model_description": USER_MODEL_DESCRIPTION,
        "checkpoints": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in checkpoint_paths.items()
        },
    }
    config_path = run_dir / "run_config.json"
    result_dir = run_dir / "results"
    if config_path.exists():
        try:
            previous = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            if image_files(result_dir):
                raise RuntimeError(
                    f"Cannot reuse {run_dir}: run_config.json is unreadable while "
                    "result images exist. Use a new USER_RUN_NAME."
                ) from error
            previous = None
        if previous != config and image_files(result_dir):
            raise RuntimeError(
                "Existing results belong to another manifest/checkpoint/config. "
                "Use a new USER_RUN_NAME."
            )
    elif image_files(result_dir):
        raise RuntimeError(
            f"Cannot reuse {run_dir}: result images exist but run_config.json is missing. "
            "Use a new USER_RUN_NAME."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(manifest_path, run_dir / "pairs.jsonl")


def main() -> None:
    if USER_CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES
    if USER_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("USER_DEVICE is CUDA, but PyTorch cannot see a CUDA device.")

    mode = USER_MODE.strip().lower()
    baseline_root = USER_BASELINE_ROOT.expanduser().resolve()
    image_root = USER_CELEBA_HQ_DIR.expanduser().resolve()
    manifest_path = (
        resolve_config_path(USER_MANIFEST_PATH.expanduser()).resolve()
        if USER_MANIFEST_PATH is not None
        else default_manifest_path(mode).resolve()
    )
    output_root = resolve_config_path(USER_OUTPUT_ROOT.expanduser()).resolve()
    rows, skipped_rows = load_manifest(manifest_path, mode, image_root)
    run_dir = output_root / USER_RUN_NAME / mode
    result_dir = run_dir / "results"
    real_dir = run_dir / "real_source"

    original_cwd = Path.cwd()
    try:
        # Official baseline modules use repository-relative pretrained_models paths.
        os.chdir(baseline_root)
        print("Initializing original HairFast baseline...", flush=True)
        model, checkpoint_paths = build_baseline(baseline_root, USER_DEVICE)
        print("Baseline model initialized.", flush=True)
        prepare_run_config(
            run_dir,
            manifest_path,
            image_root,
            baseline_root,
            checkpoint_paths,
            mode,
            len(rows),
        )
        if skipped_rows:
            (run_dir / "skipped_rows.json").write_text(
                json.dumps(skipped_rows, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        expected_outputs = {str(row["output_file"]) for row in rows}
        print(f"Preparing {len(rows)} real source images...", flush=True)
        expected_real = export_real_sources(rows, image_root, real_dir)

        with torch.inference_mode():
            for row in tqdm(rows, desc=f"Generate baseline results ({mode})"):
                output_path = result_dir / str(row["output_file"])
                if USER_SKIP_EXISTING_IMAGES and output_path.exists():
                    verify_rgb_image(output_path, "existing baseline result")
                    continue
                source_path = resolve_image(image_root, str(row["source_file"]))
                shape_path = resolve_image(image_root, str(row["shape_file"]))
                color_path = resolve_image(image_root, str(row["color_file"]))
                try:
                    result = model(
                        source_path,
                        shape_path,
                        color_path,
                        seed=int(row["sample_seed"]),
                        exp_name=Path(str(row["output_file"])).stem,
                    )
                    tensor_to_rgb_image(result).save(output_path, format="PNG")
                except Exception as error:  # noqa: BLE001 - identify fixed row
                    raise RuntimeError(
                        f"Baseline inference failed at row {row['index']} "
                        f"(source={source_path}, shape={shape_path}, color={color_path})"
                    ) from error

        validate_directory(result_dir, expected_outputs, "baseline results")
        validate_directory(real_dir, expected_real, "real source")
        print(f"Mode: {mode}; rows: {len(rows)}")
        print(f"Manifest: {manifest_path}")
        print(f"Manifest SHA256: {sha256(manifest_path)}")
        print(f"FID real set: {real_dir}")
        print(f"FID baseline set: {result_dir}")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
