"""Evaluate Stable-Hair on the frozen CelebA-HQ 3000-pair manifest.

This is the Stable-Hair adapter for the shared evaluation contract.  It is a
self-contained copy of ``celeba_hq_generate_eval_external_template.py`` whose
only model-specific surface is ``infer_one``.  All directory layout, manifest
validation, real-source materialisation, safety checks and resume behaviour
are owned by this script exactly as in the template.

Contract (do not change):
  * The JSONL manifest is the *only* evaluation contract.  It is produced once
    by ``scripts/celeba_hq_make_eval_pairs_v5.py`` (or the stable-hair copy
    ``celeba_hq_make_eval_pairs_stable_hair.py``) and is read-only here.
  * Only ``results`` may be handed to ``scripts/fid_metric.py``.
  * Stable-Hair's ``Hair_Transfer`` is a two-input API (source + one
    reference).  Therefore only ``both`` is supported.  ``full`` would require
    silently dropping the color reference, which the contract forbids, so the
    script refuses ``full`` instead of cheating.

Model inference is inlined here (without the gradio import that ``infer_full``
carries at module top-level) so that the eval environment does not need gradio.
The heavy modules come from the Stable-Hair checkout pointed to by
``USER_STABLE_HAIR_ROOT``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


# ========================= User config: edit here only ========================
USER_MODE = "both"  # Stable-Hair is 2-input: only "both" is fair. "full" is refused.

USER_CELEBA_HQ_DIR = Path("/root/shared-nvme/HairFastGAN/celeba-1024")
USER_MANIFEST_PATH = Path(
    "input/eval_pairs_v5/celeba_hq_both_seed3407_3000.jsonl"
)
USER_EXPECTED_SAMPLE_COUNT = 3000

# Use a new name when the external model, checkpoint, preprocessing, or any
# inference setting changes.  This prevents stale images being mixed into FID.
USER_OUTPUT_ROOT = Path("output/celeba_hq_eval_external")
USER_METHOD_RUN_NAME = "stable_hair_fp16_512"
USER_MODEL_DESCRIPTION = (
    "Stable-Hair: stage1 bald converter + stage2 hair encoder/adapter/controlnet. "
    "Two-input source+reference transfer at 512x512, fp16, UniPC scheduler."
)
USER_SKIP_EXISTING_IMAGES = True
USER_VERIFY_INPUT_IMAGES = True

# --- External model: Stable-Hair ------------------------------------------------
# Checkout root of Stable-Hair (contains configs/, models/, ref_encoder/, utils/).
# On the Linux eval host, point this at your Stable-Hair checkout.
USER_STABLE_HAIR_ROOT = Path("D:/stable_hair")
USER_STABLE_HAIR_CONFIG = USER_STABLE_HAIR_ROOT / "configs" / "hair_transfer.yaml"
USER_DEVICE = "cuda"
USER_WEIGHT_DTYPE = "float16"  # "float16" or "float32"
USER_STEP = 30
USER_GUIDANCE_SCALE = 1.5
USER_SCALE = 1.0
USER_CONTROLNET_CONDITIONING_SCALE = 1.0
USER_SIZE = 512
# Stable-Hair accepts source + ONE reference.  A full triplet cannot be
# transferred fairly, so the contract requires refusing "full" here.
USER_MODEL_SUPPORTS_FULL_MODE = False
# Hard-coded in the original Stable-Hair get_bald(); recorded for transparency.
USER_BALD_NUM_INFERENCE_STEPS = 30
USER_BALD_GUIDANCE_SCALE = 1.5
USER_BALD_SCALE = 0.9
# ==============================================================================


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _find_repo_root(start: Path) -> Path:
    # Works no matter which scripts subdirectory the file lives in or from
    # which CWD it is launched: walk up to the first ancestor that looks like
    # the project root (it contains input/eval_pairs_v5 manifests).
    for candidate in (start, *start.parents):
        if (candidate / "input").is_dir():
            return candidate
    return start.parents[1] if len(start.parents) > 1 else start


REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def anchor_to_root(path: Path) -> Path:
    # Relative paths are CWD-independent: use CWD when it already works,
    # otherwise anchor to the auto-detected project root.
    path = path.expanduser()
    if path.is_absolute():
        return path
    resolved = path.resolve()
    if resolved.exists():
        return resolved
    return REPO_ROOT / path


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
            "scripts/celeba_hq_make_eval_pairs_v5.py (or the stable-hair copy)."
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


# ============================ Stable-Hair model ===============================
# The model is constructed lazily and cached for the whole run.  inlining the
# loader avoids the top-level ``import gradio`` in infer_full.py, and resolves
# every relative checkpoint path against USER_STABLE_HAIR_ROOT so the script is
# independent of the current working directory.

_STABLE_HAIR_MODEL: dict[str, object] | None = None


def _resolve_under(value, anchor: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (anchor / path).resolve()


def _torch_load(path):
    import torch

    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        return torch.load(path)


def get_stable_hair_model() -> dict[str, object]:
    global _STABLE_HAIR_MODEL
    if _STABLE_HAIR_MODEL is not None:
        return _STABLE_HAIR_MODEL

    root = USER_STABLE_HAIR_ROOT.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"USER_STABLE_HAIR_ROOT does not exist: {root}. "
            "Set it to your Stable-Hair checkout on this machine."
        )
    # Import the environment's pip diffusers/transformers FIRST so they are
    # cached in sys.modules.  The Stable-Hair checkout vendors an old diffusers
    # source tree that imports huggingface_hub APIs removed in new versions
    # (hf_cache_home); caching here prevents that vendored copy from being
    # imported when Stable-Hair root is prepended to sys.path below.
    import diffusers  # noqa: F401,WPS433
    import transformers  # noqa: F401,WPS433

    if str(root) not in sys.path:
        # Stable-Hair modules use absolute imports rooted at the checkout:
        # ``from utils.pipeline import StableHairPipeline``, ``from ref_encoder...
        sys.path.insert(0, str(root))

    import torch  # noqa: WPS433 - delayed import keeps config section clean
    from omegaconf import OmegaConf  # noqa: WPS433

    from ref_encoder.adapter import adapter_injection, set_scale  # noqa: WPS433
    from ref_encoder.latent_controlnet import ControlNetModel  # noqa: WPS433
    from ref_encoder.reference_unet import ref_unet  # noqa: WPS433
    from utils.pipeline import StableHairPipeline  # noqa: WPS433
    from utils.pipeline_cn import StableDiffusionControlNetPipeline  # noqa: WPS433
    from diffusers import UniPCMultistepScheduler  # noqa: WPS433
    from diffusers.models import UNet2DConditionModel  # noqa: WPS433

    weight_dtype = torch.float16 if USER_WEIGHT_DTYPE == "float16" else torch.float32
    device = USER_DEVICE
    config_path = _resolve_under(USER_STABLE_HAIR_CONFIG, root)
    if not config_path.is_file():
        raise FileNotFoundError(f"Stable-Hair config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg.pretrained_model_path = str(_resolve_under(cfg.pretrained_model_path, root))
    cfg.pretrained_folder = str(_resolve_under(cfg.pretrained_folder, root))
    cfg.bald_converter_path = str(_resolve_under(cfg.bald_converter_path, root))

    print("Initializing Stable Hair Pipeline...")
    unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_path, subfolder="unet").to(device)
    controlnet = ControlNetModel.from_unet(unet).to(device)
    controlnet.load_state_dict(
        _torch_load(os.path.join(cfg.pretrained_folder, cfg.controlnet_path)), strict=False
    )
    controlnet.to(weight_dtype)

    pipeline = StableHairPipeline.from_pretrained(
        cfg.pretrained_model_path,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=weight_dtype,
    ).to(device)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    hair_encoder = ref_unet.from_pretrained(cfg.pretrained_model_path, subfolder="unet").to(device)
    hair_encoder.load_state_dict(
        _torch_load(os.path.join(cfg.pretrained_folder, cfg.encoder_path)), strict=False
    )
    hair_adapter = adapter_injection(
        pipeline.unet, device=device, dtype=torch.float16, use_resampler=False
    )
    hair_adapter.load_state_dict(
        _torch_load(os.path.join(cfg.pretrained_folder, cfg.adapter_path)), strict=False
    )

    bald_converter = ControlNetModel.from_unet(unet).to(device)
    bald_converter.load_state_dict(_torch_load(cfg.bald_converter_path), strict=False)
    bald_converter.to(dtype=weight_dtype)
    del unet

    remove_hair_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.pretrained_model_path,
        controlnet=bald_converter,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(
        remove_hair_pipeline.scheduler.config
    )
    remove_hair_pipeline = remove_hair_pipeline.to(device)

    hair_encoder.to(weight_dtype)
    hair_adapter.to(weight_dtype)

    if USER_ENABLE_SEQUENTIAL_CPU_OFFLOAD:
        pipeline.enable_sequential_cpu_offload()
        remove_hair_pipeline.enable_sequential_cpu_offload()
    elif USER_ENABLE_CPU_OFFLOAD:
        pipeline.enable_model_cpu_offload()
        remove_hair_pipeline.enable_model_cpu_offload()
    # Lossless: decode VAE latents slice-by-slice to cut peak memory.
    pipeline.enable_vae_slicing()

    _STABLE_HAIR_MODEL = {
        "pipeline": pipeline,
        "hair_encoder": hair_encoder,
        "hair_adapter": hair_adapter,
        "remove_hair_pipeline": remove_hair_pipeline,
        "set_scale": set_scale,
        "weight_dtype": weight_dtype,
        "device": device,
    }
    print("Stable Hair initialization done.")
    return _STABLE_HAIR_MODEL


def _get_bald(model: dict[str, object], id_image, scale: float):
    import torch  # noqa: WPS433

    remove_hair_pipeline = model["remove_hair_pipeline"]
    width, height = id_image.size
    image = remove_hair_pipeline(
        prompt="",
        negative_prompt="",
        num_inference_steps=USER_BALD_NUM_INFERENCE_STEPS,
        guidance_scale=USER_BALD_GUIDANCE_SCALE,
        width=width,
        height=height,
        image=id_image,
        controlnet_conditioning_scale=float(scale),
        generator=None,
    ).images[0]
    return image


def _stable_hair_transfer(
    model: dict[str, object],
    source_image,
    reference_image,
    random_seed: int,
    step: int,
    guidance_scale: float,
    scale: float,
    controlnet_conditioning_scale: float,
    size: int = 512,
):
    """Mirror StableHair.Hair_Transfer but on a cached model dict.

    source_image / reference_image are file paths, matching the original API.
    """
    import numpy as np  # noqa: WPS433
    import torch  # noqa: WPS433

    pipeline = model["pipeline"]
    hair_encoder = model["hair_encoder"]
    set_scale_fn = model["set_scale"]
    device = model["device"]

    source_pil = Image.open(source_image).convert("RGB").resize((size, size))
    id_arr = np.array(source_pil)
    reference_arr = np.array(
        Image.open(reference_image).convert("RGB").resize((size, size))
    )
    source_image_bald = np.array(_get_bald(model, source_pil, scale=USER_BALD_SCALE))
    height, width, _ = source_image_bald.shape

    set_scale_fn(pipeline.unet, float(scale))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_seed))
    sample = pipeline(
        "",
        negative_prompt="",
        num_inference_steps=int(step),
        guidance_scale=float(guidance_scale),
        width=width,
        height=height,
        controlnet_condition=source_image_bald,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        reference_encoder=hair_encoder,
        ref_image=reference_arr,
    ).samples
    return id_arr, sample, source_image_bald, reference_arr


def _to_pil_rgb(sample) -> Image.Image:
    import numpy as np  # noqa: WPS433

    arr = sample
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = (np.clip(arr.astype(float), 0.0, 1.0) * 255.0).astype("uint8")
    return Image.fromarray(arr, mode="RGB").convert("RGB")


def infer_one(
    source_path: Path,
    shape_path: Path,
    color_path: Path,
    seed: int,
) -> Image.Image:
    """Run one Stable-Hair inference and return the final image (RGB PIL).

    In ``both`` mode ``shape_path == color_path == reference`` by construction,
    so passing ``shape_path`` as Stable-Hair's single reference is exactly the
    contract; no colour reference is dropped.  ``full`` is refused upstream in
    ``main`` because Stable-Hair cannot consume two distinct references fairly.
    The manifest seed is used verbatim.
    """
    del color_path  # in both mode color == shape == reference; not an extra input
    model = get_stable_hair_model()
    _id, sample, _bald, _ref = _stable_hair_transfer(
        model,
        source_image=str(source_path),
        reference_image=str(shape_path),
        random_seed=int(seed),
        step=USER_STEP,
        guidance_scale=USER_GUIDANCE_SCALE,
        scale=USER_SCALE,
        controlnet_conditioning_scale=USER_CONTROLNET_CONDITIONING_SCALE,
        size=USER_SIZE,
    )
    return _to_pil_rgb(sample)


# ============================ output / validation =============================
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


def run_identity(manifest_path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "mode": USER_MODE,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_root": str(USER_CELEBA_HQ_DIR.resolve()),
        "sample_count": len(rows),
        "model": {
            "name": "stable_hair",
            "description": USER_MODEL_DESCRIPTION,
            "stable_hair_root": str(USER_STABLE_HAIR_ROOT),
            "config_path": str(USER_STABLE_HAIR_CONFIG),
            "supports_full_mode": USER_MODEL_SUPPORTS_FULL_MODE,
        },
        "inference": {
            "device": USER_DEVICE,
            "weight_dtype": USER_WEIGHT_DTYPE,
            "step": USER_STEP,
            "guidance_scale": USER_GUIDANCE_SCALE,
            "scale": USER_SCALE,
            "controlnet_conditioning_scale": USER_CONTROLNET_CONDITIONING_SCALE,
            "size": USER_SIZE,
            "bald_num_inference_steps": USER_BALD_NUM_INFERENCE_STEPS,
            "bald_guidance_scale": USER_BALD_GUIDANCE_SCALE,
            "bald_scale": USER_BALD_SCALE,
        },
        # Kept for template compatibility.
        "model_description": USER_MODEL_DESCRIPTION,
    }


def write_run_files(run_dir: Path, manifest_path: Path, rows: list[dict[str, object]]) -> None:
    config_path = run_dir / "run_config.json"
    identity = run_identity(manifest_path, rows)
    result_dir = run_dir / "results"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != identity and image_files(result_dir):
            raise RuntimeError(
                "This run directory contains results from a different manifest or model "
                "setting. Choose a new USER_METHOD_RUN_NAME."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(identity, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(manifest_path, run_dir / "pairs.jsonl")


def main() -> None:
    mode = USER_MODE.strip().lower()
    if mode == "full" and not USER_MODEL_SUPPORTS_FULL_MODE:
        raise RuntimeError(
            "USER_MODE='full' is not supported by Stable-Hair. Stable-Hair's "
            "Hair_Transfer accepts only source + one reference image, so a full "
            "triplet (source + shape ref + color ref) cannot be transferred "
            "fairly without silently ignoring the color reference, which the "
            "evaluation contract forbids. Use USER_MODE='both' instead."
        )

    manifest_path = anchor_to_root(USER_MANIFEST_PATH)
    image_root = USER_CELEBA_HQ_DIR.expanduser().resolve()
    rows = load_manifest(manifest_path, mode, image_root)

    run_dir = anchor_to_root(USER_OUTPUT_ROOT) / USER_METHOD_RUN_NAME / mode
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

    # Pre-warm the model once so the per-row progress bar is honest.
    get_stable_hair_model()

    for row in tqdm(rows, desc=f"Generate Stable-Hair evaluation images ({mode})"):
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
                f"Stable-Hair inference failed at row {row['index']} "
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
