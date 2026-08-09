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

# ``run_config.json`` is a safety boundary for resumable metric generation.
# Bump this whenever the identity/fingerprint format changes.  Configs from an
# older schema must never silently bless already-generated images.
RUN_CONFIG_SCHEMA_VERSION = 3
OUTPUT_RECEIPT_SCHEMA_VERSION = 1

# These files contain the non-learned v8/v5 inference policy introduced by
# this project.  The git commit alone is insufficient because evaluation is
# commonly launched from a working tree with uncommitted experiments.
INFERENCE_CODE_FILES = (
    "scripts/celeba_hq_generate_eval_v5.py",
    "hair_swap_v5.py",
    "hair_swap_v8.py",
    "datasets/image_dataset.py",
    "models/Alignment.py",
    "models/Alignment_v8.py",
    "models/Blending.py",
    "models/Blending_v5.py",
    "models/Blending_v8.py",
    "models/Embedding.py",
    "models/Encoders.py",
    "models/Net.py",
    "models/SATD_v8.py",
    "models/CtrlHair/external_code/face_parsing/model.py",
    "models/CtrlHair/external_code/face_parsing/my_parsing_util.py",
    "models/CtrlHair/shape_branch/config.py",
    "models/CtrlHair/shape_branch/model.py",
    "models/CtrlHair/shape_branch/shape_util.py",
    "models/CtrlHair/shape_branch/solver.py",
    "models/FeatureStyleEncoder/FSencoder.py",
    "models/FeatureStyleEncoder/configs/001.yaml",
    "models/FeatureStyleEncoder/trainer.py",
    "models/encoder4editing/utils/model_utils.py",
    "models/ear_modules_v5.py",
    "models/sean_codes/models/pix2pix_model.py",
    "models/sean_codes/util/util.py",
    "models/stylegan2/model.py",
    "models/postprocess_v5.py",
    "utils/bicubic.py",
    "utils/hair_color_match_v8.py",
    "utils/image_utils.py",
    "utils/mask_delta_v8.py",
    "utils/seed.py",
    "utils/shape_predictor.py",
)

# Imports inside FeatureStyleEncoder, SEAN, encoder4editing and CtrlHair are
# deep and partly dynamic.  Hashing their repository trees prevents a newly
# imported architecture/helper from escaping an otherwise hand-maintained
# short list.  Training-only files may conservatively invalidate a run, which
# is preferable to reusing pixels after an unrecorded inference-code change.
INFERENCE_CODE_DIRECTORIES = ("datasets", "models", "utils")
INFERENCE_CODE_SUFFIXES = {".py", ".yaml", ".yml"}

# HairFast also loads these weights from fixed repository-relative locations.
# Fingerprinting only the user-facing v8/v5 checkpoints is not sufficient:
# changing the parser, SEAN, shape adaptor, or feature encoder also changes the
# generated pixels while leaving those configured paths untouched.
FIXED_INFERENCE_CHECKPOINTS = {
    "embedding_e4e": "pretrained_models/encoder4editing/e4e_ffhq_encode.pt",
    "shape_adaptor": "pretrained_models/ShapeAdaptor/mask_generator.pth",
    "sean_generator": (
        "pretrained_models/sean_checkpoints/"
        "CelebA-HQ_pretrained/latest_net_G.pth"
    ),
    "feature_style_encoder": "pretrained_models/FeatureStyleEncoder/143_enc.pth",
    "feature_style_stylegan": (
        "pretrained_models/FeatureStyleEncoder/psp_ffhq_encode.pt"
    ),
    "feature_style_arcface": "pretrained_models/FeatureStyleEncoder/backbone.pth",
    "feature_style_parsing": "pretrained_models/FeatureStyleEncoder/79999_iter.pth",
    "face_parsing": "pretrained_models/BiSeNet/face_parsing_79999_iter.pth",
    "postprocess_arcface": "pretrained_models/ArcFace/backbone_ir50.pth",
    "postprocess_latent_avg": "pretrained_models/PostProcess/latent_avg.pt",
}

FIXED_INFERENCE_ASSET_DIRECTORIES = {
    "sean_median_style_codes": "models/sean_codes/styles_test/mean_style_code/median",
}

# Debug/output locations do not alter the final tensor.  Every other parser
# option is captured automatically, including future mask, colour, earring,
# and revealed-skin settings, so a newly added policy cannot be forgotten in
# the resume check.
MODEL_POLICY_EXCLUDED_ARGS = {
    "save_all",
    "save_all_dir",
    "ckpt",
    "rotate_checkpoint",
    "blending_checkpoint",
    "pp_checkpoint",
    "pp_v5_checkpoint",
    "satd_checkpoint_v8",
}


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
    repo_root = repository_root()
    runtime_e4e = resolve_repository_path(
        FIXED_INFERENCE_CHECKPOINTS["embedding_e4e"],
        repo_root,
    )
    configured_e4e = resolve_repository_path(USER_E4E_CHECKPOINT, repo_root)
    if configured_e4e != runtime_e4e:
        raise RuntimeError(
            "USER_E4E_CHECKPOINT does not control the current HairFast Embedding implementation; "
            f"it hardcodes {runtime_e4e}. Set USER_E4E_CHECKPOINT to that file or wire the "
            "checkpoint through Embedding.py before evaluating."
        )
    required: list[tuple[str | os.PathLike[str], str]] = [
        (USER_STYLEGAN_CHECKPOINT, "USER_STYLEGAN_CHECKPOINT"),
        (
            stylegan_pca_path(USER_STYLEGAN_CHECKPOINT, repo_root=repo_root),
            "StyleGAN PCA sidecar",
        ),
        (USER_ROTATE_CHECKPOINT, "USER_ROTATE_CHECKPOINT"),
        (USER_BLENDING_V8_CHECKPOINT, "USER_BLENDING_V8_CHECKPOINT"),
        (USER_PP_V5_CHECKPOINT, "USER_PP_V5_CHECKPOINT"),
        (USER_E4E_CHECKPOINT, "USER_E4E_CHECKPOINT"),
    ]
    required.extend(
        (path, f"fixed inference checkpoint {name!r}")
        for name, path in FIXED_INFERENCE_CHECKPOINTS.items()
    )
    if USER_ALIGN_INPUTS:
        required.append(
            (
                "pretrained_models/ShapeAdaptor/shape_predictor_68_face_landmarks.dat",
                "face-alignment landmark checkpoint",
            )
        )
    for path, label in required:
        resolved = resolve_repository_path(path, repo_root)
        if not resolved.is_file():
            raise FileNotFoundError(f"Cannot find {label}: {resolved}")
    if USER_USE_SATD_V8:
        resolved = resolve_repository_path(USER_SATD_CHECKPOINT_V8, repo_root)
        if not resolved.is_file():
            raise FileNotFoundError(f"Cannot find USER_SATD_CHECKPOINT_V8: {resolved}")

    for name, path in FIXED_INFERENCE_ASSET_DIRECTORIES.items():
        resolved = resolve_repository_path(path, repo_root)
        if not resolved.is_dir() or not any(item.is_file() for item in resolved.rglob("*")):
            raise FileNotFoundError(
                f"Cannot find non-empty fixed inference asset directory {name!r}: {resolved}"
            )


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
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        if (
            destination.stat().st_size == source.stat().st_size
            and file_sha256(destination, use_cache=False) == file_sha256(source)
        ):
            return
        raise RuntimeError(
            f"Existing real-source export does not match its manifest input: {destination}. "
            "Use a fresh USER_RUN_NAME or explicitly clear the stale real_source directory."
        )
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


_FILE_SHA256_CACHE: dict[str, str] = {}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repository_path(path: str | os.PathLike[str], repo_root: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def stylegan_pca_path(
    checkpoint_path: str | os.PathLike[str],
    *,
    repo_root: Path,
) -> Path:
    """Mirror ``models.Net``'s established ``ckpt[:-3] + '_PCA.npz'`` rule."""

    checkpoint = resolve_repository_path(checkpoint_path, repo_root)
    return Path(str(checkpoint)[:-3] + "_PCA.npz")


def file_sha256(path: Path, *, use_cache: bool = True) -> str:
    """Hash a file once per process, even when several records reference it."""

    resolved = path.expanduser().resolve()
    cache_key = str(resolved)
    if use_cache:
        cached = _FILE_SHA256_CACHE.get(cache_key)
        if cached is not None:
            return cached

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    if use_cache:
        _FILE_SHA256_CACHE[cache_key] = value
    return value


def manifest_digest(path: Path) -> str:
    return file_sha256(path)


def jsonable_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def checkpoint_record(
    path: str | os.PathLike[str] | None,
    *,
    repo_root: Path,
) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = resolve_repository_path(path, repo_root)
    if not resolved.is_file():
        raise FileNotFoundError(f"Cannot fingerprint checkpoint: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def directory_record(
    path: str | os.PathLike[str],
    *,
    repo_root: Path,
) -> dict[str, object]:
    """Fingerprint every file in a small, inference-critical asset tree."""

    resolved = resolve_repository_path(path, repo_root)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Cannot fingerprint inference asset directory: {resolved}")

    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"Inference asset directory is empty: {resolved}")
    file_hashes = {
        item.relative_to(resolved).as_posix(): file_sha256(item)
        for item in files
    }
    encoded = json.dumps(file_hashes, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return {
        "path": str(resolved),
        "file_count": len(file_hashes),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": file_hashes,
    }


def input_images_record(
    rows: list[dict[str, object]],
    *,
    dataset_root: Path,
) -> dict[str, object]:
    """Fingerprint each unique source/shape/color file referenced by a run."""

    files: dict[str, dict[str, object]] = {}
    for row in rows:
        for field in ("source_file", "shape_file", "color_file"):
            resolved = image_path(dataset_root, str(row[field])).expanduser().resolve()
            key = str(resolved)
            if key in files:
                continue
            if not resolved.is_file():
                raise FileNotFoundError(f"Cannot fingerprint evaluation input: {resolved}")
            files[key] = {
                "bytes": resolved.stat().st_size,
                "sha256": file_sha256(resolved),
            }

    encoded = json.dumps(files, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return {
        "file_count": len(files),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def output_receipt_path(receipt_dir: Path, output_file: str) -> Path:
    return receipt_dir / f"{output_file}.json"


def output_receipt_fields(
    row: dict[str, object],
    identity: dict[str, object],
) -> dict[str, object]:
    index = int(row["index"])
    return {
        "schema_version": OUTPUT_RECEIPT_SCHEMA_VERSION,
        "identity_sha256": str(identity["identity_sha256"]),
        "manifest_index": index,
        "output_file": str(row["output_file"]),
        "sample_seed": int(row.get("sample_seed", USER_MANIFEST_SEED + index - 1)),
    }


def write_output_receipt(
    output_path: Path,
    receipt_path: Path,
    row: dict[str, object],
    identity: dict[str, object],
) -> None:
    record = output_receipt_fields(row, identity)
    with Image.open(output_path) as image:
        dimensions = [int(image.width), int(image.height)]
    record.update(
        {
            "bytes": output_path.stat().st_size,
            "dimensions": dimensions,
            # Output files may be overwritten in the same process, so never
            # use the immutable-input/checkpoint hash cache here.
            "sha256": file_sha256(output_path, use_cache=False),
        }
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(receipt_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def reusable_output_status(
    output_path: Path,
    receipt_path: Path,
    row: dict[str, object],
    identity: dict[str, object],
) -> tuple[bool, str]:
    if not receipt_path.is_file():
        return False, "receipt is missing"
    try:
        with receipt_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return False, f"receipt is unreadable ({error})"
    if not isinstance(previous, dict):
        return False, "receipt is not a JSON object"

    expected = output_receipt_fields(row, identity)
    mismatches = [
        key for key, value in expected.items()
        if previous.get(key) != value
    ]
    if mismatches:
        return False, f"receipt fields differ: {mismatches}"

    try:
        if USER_VERIFY_EXISTING_OUTPUTS:
            verify_image(output_path)
        with Image.open(output_path) as image:
            actual_dimensions = [int(image.width), int(image.height)]
        actual_bytes = output_path.stat().st_size
        actual_sha256 = file_sha256(output_path, use_cache=False)
    except (OSError, RuntimeError) as error:
        return False, f"output cannot be verified ({error})"
    if previous.get("bytes") != actual_bytes:
        return False, "output byte size differs from receipt"
    if previous.get("dimensions") != actual_dimensions:
        return False, "output dimensions differ from receipt"
    if previous.get("sha256") != actual_sha256:
        return False, "output SHA256 differs from receipt"
    return True, ""


def build_run_identity(
    run_dir: Path,
    manifest_path: Path,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Fingerprint every setting/file that can change an evaluation image."""

    repo_root = repository_root()
    code_paths = {Path(relative) for relative in INFERENCE_CODE_FILES}
    for directory in INFERENCE_CODE_DIRECTORIES:
        root = repo_root / directory
        if not root.is_dir():
            raise FileNotFoundError(f"Cannot fingerprint inference policy directory: {root}")
        code_paths.update(
            path.relative_to(repo_root)
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in INFERENCE_CODE_SUFFIXES
        )

    code_hashes = {}
    for relative in sorted(code_paths, key=lambda item: item.as_posix()):
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Cannot fingerprint inference policy file: {path}")
        code_hashes[relative.as_posix()] = file_sha256(path)

    model_args = make_model_args(run_dir)
    model_policy = {
        key: jsonable_value(value)
        for key, value in vars(model_args).items()
        if key not in MODEL_POLICY_EXCLUDED_ARGS
    }
    checkpoints: dict[str, object] = {
        "stylegan": checkpoint_record(USER_STYLEGAN_CHECKPOINT, repo_root=repo_root),
        "stylegan_pca": checkpoint_record(
            stylegan_pca_path(USER_STYLEGAN_CHECKPOINT, repo_root=repo_root),
            repo_root=repo_root,
        ),
        "rotate": checkpoint_record(USER_ROTATE_CHECKPOINT, repo_root=repo_root),
        "blending_v8": checkpoint_record(
            USER_BLENDING_V8_CHECKPOINT,
            repo_root=repo_root,
        ),
        "pp_v5": checkpoint_record(USER_PP_V5_CHECKPOINT, repo_root=repo_root),
        "satd_v8": (
            checkpoint_record(USER_SATD_CHECKPOINT_V8, repo_root=repo_root)
            if USER_USE_SATD_V8
            else None
        ),
        "e4e": checkpoint_record(USER_E4E_CHECKPOINT, repo_root=repo_root),
    }
    checkpoints["fixed_runtime"] = {
        name: checkpoint_record(path, repo_root=repo_root)
        for name, path in FIXED_INFERENCE_CHECKPOINTS.items()
    }
    checkpoints["input_alignment_landmarks"] = (
        checkpoint_record(
            "pretrained_models/ShapeAdaptor/shape_predictor_68_face_landmarks.dat",
            repo_root=repo_root,
        )
        if USER_ALIGN_INPUTS
        else None
    )
    inference_assets = {
        name: directory_record(path, repo_root=repo_root)
        for name, path in FIXED_INFERENCE_ASSET_DIRECTORIES.items()
    }

    manifest_path = manifest_path.expanduser().resolve()
    dataset_root = USER_CELEBA_HQ_DIR.expanduser().resolve()
    manifest_sha256 = manifest_digest(manifest_path)
    input_images = input_images_record(rows, dataset_root=dataset_root)
    identity = {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "reference_mode": normalize_mode(USER_MODE),
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "input_images": input_images,
        "seed_base": int(USER_MANIFEST_SEED),
        "align_inputs": bool(USER_ALIGN_INPUTS),
        "model_policy": model_policy,
        "code_sha256": code_hashes,
        "checkpoints": checkpoints,
        "inference_assets": inference_assets,
    }
    encoded = json.dumps(identity, ensure_ascii=True, sort_keys=True).encode("utf-8")
    identity["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return identity


def write_run_files(
    run_dir: Path,
    rows: list[dict[str, object]],
    manifest_path: Path,
    real_paths: dict[int, str],
    result_dir: Path,
    panel_dir: Path,
    receipt_dir: Path,
    identity: dict[str, object],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    mode = str(identity["reference_mode"])
    manifest_sha256 = str(identity["manifest_sha256"])
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
                    "receipt_path": str(
                        output_receipt_path(receipt_dir, str(row["output_file"]))
                    ),
                    "real_source_path": real_paths.get(index, ""),
                }
            )
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    config = {
        **identity,
        "reference_mode": mode,
        "dataset_root": str(USER_CELEBA_HQ_DIR.expanduser().resolve()),
        "manifest_path": str(manifest_path.expanduser().resolve()),
        "manifest_sha256": manifest_sha256,
        "run_name": USER_RUN_NAME,
        "result_dir": str(result_dir),
        "panel_dir": str(panel_dir),
        "receipt_dir": str(receipt_dir),
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
    }
    config_path = run_dir / "run_config.json"
    temporary = config_path.with_suffix(".json.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(config, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(config_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_reusable_run(run_dir: Path, identity: dict[str, object]) -> None:
    """Prevent old result pixels from ever being blessed by a new identity."""

    existing_results = image_files_in_directory(run_dir / "results")
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        if existing_results:
            raise RuntimeError(
                f"Cannot safely reuse {run_dir}: run_config.json is missing while result images exist. "
                "Choose a fresh USER_RUN_NAME or explicitly clear that result directory."
            )
        return
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        if existing_results:
            raise RuntimeError(
                f"Cannot safely reuse {run_dir}: its run_config.json is unreadable while "
                "result images exist. Choose a new USER_RUN_NAME or explicitly clear that "
                "result directory."
            ) from error
        return

    # Compare the complete canonical identity, rather than only the digest.
    # This catches truncated/tampered configs even if a stale digest happened
    # to be copied alongside them.
    previous_identity = {
        key: previous.get(key)
        for key in identity
    }
    if previous_identity != identity and existing_results:
        mismatches = [
            key for key, value in identity.items()
            if previous.get(key) != value
        ]
        raise RuntimeError(
            f"{run_dir} was created with different evaluation identity fields: {mismatches[:12]}. "
            "Choose a new USER_RUN_NAME or explicitly clear that result directory. "
            "USER_SKIP_EXISTING_IMAGES=False is not enough because an interrupted overwrite "
            "could otherwise leave old pixels under the new identity."
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
    repo_root = repository_root()
    if Path.cwd().resolve() != repo_root:
        raise RuntimeError(
            "Run this evaluator with the repository root as the current working directory "
            f"({repo_root}); the HairFast runtime uses repository-relative weight paths."
        )
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
    receipt_dir = run_dir / "receipts"
    result_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_PANELS:
        panel_dir.mkdir(parents=True, exist_ok=True)
    if USER_EXPORT_REAL_SOURCE_IMAGES:
        real_dir.mkdir(parents=True, exist_ok=True)

    identity = build_run_identity(run_dir, manifest_path, rows)
    validate_reusable_run(run_dir, identity)

    real_paths = {}
    if USER_EXPORT_REAL_SOURCE_IMAGES:
        real_paths = export_real_sources(rows, USER_CELEBA_HQ_DIR, real_dir)

    expected_outputs = {str(row["output_file"]) for row in rows}
    write_run_files(
        run_dir,
        rows,
        manifest_path,
        real_paths,
        result_dir,
        panel_dir,
        receipt_dir,
        identity,
    )
    shutil.copy2(manifest_path, run_dir / "pairs.jsonl")

    set_seed(USER_MANIFEST_SEED)
    model = HairFastV5(make_model_args(run_dir))
    generated = 0
    skipped = 0

    with torch.inference_mode():
        for row in tqdm(rows, desc=f"Generate v8+v5 evaluation images ({mode})"):
            output_file = str(row["output_file"])
            output_path = result_dir / output_file
            receipt_path = output_receipt_path(receipt_dir, output_file)
            source_path = image_path(USER_CELEBA_HQ_DIR, str(row["source_file"]))
            shape_path = image_path(USER_CELEBA_HQ_DIR, str(row["shape_file"]))
            color_path = image_path(USER_CELEBA_HQ_DIR, str(row["color_file"]))

            if USER_SKIP_EXISTING_IMAGES and output_path.exists():
                reusable, reason = reusable_output_status(
                    output_path,
                    receipt_path,
                    row,
                    identity,
                )
                if reusable:
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
                tqdm.write(f"Regenerating unverified {output_file}: {reason}")

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
            write_output_receipt(output_path, receipt_path, row, identity)
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
    print(f"Manifest SHA256: {identity['manifest_sha256']}")
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
