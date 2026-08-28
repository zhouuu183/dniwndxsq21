"""Generate Stable-Hair reconstruction outputs on the frozen recon manifest.

Stable-Hair twin of ``scripts/generate_baseline_recon.py``.  It calls the
untouched Stable-Hair model with one original image for both inputs
(source + reference), i.e. the model reconstructs its own input.  The Stable
Hair loader is inlined here without the gradio import that ``infer_full.py``
carries at module top-level; all checkpoint paths are resolved against
``USER_STABLE_HAIR_ROOT`` so the script is independent of the CWD.

Contract:
  * The JSONL recon manifest is the only contract (shared with HairFast).
  * Only ``results`` may be handed to the LPIPS/PSNR script.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm


# ========================= User config: edit here only ========================
USER_MANIFEST = Path("input/eval_pairs_v5/celeba_hq_recon_seed3407_3000.jsonl")
USER_IMAGE_ROOT = Path("/root/shared-nvme/HairFastGAN/celeba-1024")
USER_EXPECTED_COUNT = 3000

# Stable-Hair checkout root (contains configs/, models/, ref_encoder/, utils/).
USER_STABLE_HAIR_ROOT = Path("D:/stable_hair")
USER_STABLE_HAIR_CONFIG = USER_STABLE_HAIR_ROOT / "configs" / "hair_transfer.yaml"
USER_DEVICE = "cuda"
USER_CUDA_VISIBLE_DEVICES = "0"
USER_WEIGHT_DTYPE = "float16"  # "float16" or "float32"
USER_STEP = 30
USER_GUIDANCE_SCALE = 1.5
USER_SCALE = 1.0
USER_CONTROLNET_CONDITIONING_SCALE = 1.0
USER_SIZE = 512
USER_BALD_NUM_INFERENCE_STEPS = 30
USER_BALD_GUIDANCE_SCALE = 1.5
USER_BALD_SCALE = 0.9

# Use a new name when the model, checkpoint or any inference setting changes.
USER_OUTPUT_DIR = Path("output/celeba_hq_stable_hair_recon/stable_hair_fp16_512/results")
USER_SKIP_EXISTING = True
# ===============================================================================

def _find_repo_root(start: Path) -> Path:
    # Works no matter which scripts subdirectory the file lives in: walk up to
    # the first ancestor that looks like the project root.
    for candidate in (start, *start.parents):
        if (candidate / "input").is_dir():
            return candidate
    return start.parents[1] if len(start.parents) > 1 else start


REPO_ROOT = _find_repo_root(Path(__file__).resolve())

_STABLE_HAIR_MODEL: dict[str, object] | None = None


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing reconstruction manifest: {path}. "
            "Run scripts/make_recon_manifest_stable_hair.py first."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"Manifest row {line_number} is not an object.")
            rows.append(row)
    if len(rows) != USER_EXPECTED_COUNT:
        raise RuntimeError(f"Expected {USER_EXPECTED_COUNT} rows, found {len(rows)}.")

    seen: set[str] = set()
    for expected_index, row in enumerate(rows, start=1):
        if int(row.get("index", -1)) != expected_index:
            raise RuntimeError(f"Manifest index is not contiguous at row {expected_index}.")
        if str(row.get("mode", "")).lower() != "recon":
            raise RuntimeError(f"Manifest row {expected_index} is not mode=recon.")
        if "sample_seed" not in row:
            raise RuntimeError(f"Manifest row {expected_index} is missing sample_seed.")
        source = str(row.get("source_file", ""))
        if not source or source != str(row.get("shape_file", "")) or source != str(row.get("color_file", "")):
            raise RuntimeError(f"Row {expected_index} must use one image for all three inputs.")
        if source in seen:
            raise RuntimeError(f"Duplicate source image in manifest: {source}")
        seen.add(source)
        output_file = Path(str(row.get("output_file", f"{expected_index:06d}.png")))
        if output_file.name != str(output_file) or output_file.suffix.lower() != ".png":
            raise RuntimeError(f"Unsafe output filename at row {expected_index}: {output_file}")
        row["source_file"] = source
        row["output_file"] = str(output_file)
    return rows


def verify_rgb(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.mode not in {"RGB", "RGBA", "L", "P", "CMYK"}:
                raise RuntimeError(f"unsupported image mode {image.mode!r}")
    except Exception as error:  # noqa: BLE001 - retain the exact file in the error
        raise RuntimeError(f"Unreadable image: {path}: {error}") from error


# ============================ Stable-Hair model ===============================
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
        sys.path.insert(0, str(root))

    import torch  # noqa: WPS433
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
    """Mirror StableHair.Hair_Transfer on a cached model dict (path inputs)."""
    import numpy as np  # noqa: WPS433
    import torch  # noqa: WPS433

    pipeline = model["pipeline"]
    hair_encoder = model["hair_encoder"]
    set_scale_fn = model["set_scale"]
    device = model["device"]

    source_pil = Image.open(source_image).convert("RGB").resize((size, size))
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
    return sample


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


def infer_recon_one(source_path: Path, seed: int) -> Image.Image:
    """One Stable-Hair reconstruction: the reference is the source itself."""
    model = get_stable_hair_model()
    sample = _stable_hair_transfer(
        model,
        source_image=str(source_path),
        reference_image=str(source_path),  # recon: reference == source
        random_seed=int(seed),
        step=USER_STEP,
        guidance_scale=USER_GUIDANCE_SCALE,
        scale=USER_SCALE,
        controlnet_conditioning_scale=USER_CONTROLNET_CONDITIONING_SCALE,
        size=USER_SIZE,
    )
    return _to_pil_rgb(sample)


def save_png_atomic(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp.png")
    try:
        image.convert("RGB").save(temporary, format="PNG")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_identity(manifest_path: Path) -> dict[str, Any]:
    return {
        "task": "reconstruction",
        "model": "stable_hair",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "dataset_root": str(USER_IMAGE_ROOT.resolve()),
        "sample_count": USER_EXPECTED_COUNT,
        "stable_hair_root": str(USER_STABLE_HAIR_ROOT),
        "config_path": str(USER_STABLE_HAIR_CONFIG),
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
    }


def write_run_config(output_dir: Path, manifest_path: Path) -> None:
    config_path = output_dir.parent / "run_config.json"
    identity = run_identity(manifest_path)
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != identity and any(output_dir.glob("*.png")):
            raise RuntimeError(
                "This run directory contains results from a different manifest or model "
                "setting. Choose a new USER_OUTPUT_DIR run name."
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(identity, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(manifest_path, output_dir.parent / "pairs.jsonl")


def main() -> None:
    if USER_CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

    manifest = repo_path(USER_MANIFEST).resolve()
    image_root = USER_IMAGE_ROOT.expanduser().resolve()
    output_dir = repo_path(USER_OUTPUT_DIR).resolve()
    rows = load_manifest(manifest)
    write_run_config(output_dir, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    get_stable_hair_model()
    with tqdm(rows, desc="Generate Stable-Hair recon") as progress:
        for row in progress:
            source = image_path(image_root, str(row["source_file"]))
            if not source.is_file():
                raise FileNotFoundError(f"Missing source image at row {row['index']}: {source}")
            verify_rgb(source)
            output = output_dir / str(row["output_file"])
            if USER_SKIP_EXISTING and output.is_file():
                verify_rgb(output)
                continue
            # Use the stored seed exactly; it is part of the frozen manifest row.
            image = infer_recon_one(source, int(row["sample_seed"]))
            save_png_atomic(image, output)

    expected = {str(row["output_file"]) for row in rows}
    actual = {path.name for path in output_dir.glob("*.png")}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(f"Invalid recon result directory: missing={len(missing)}, extra={len(extra)}")
    for filename in sorted(expected):
        verify_rgb(output_dir / filename)

    print(f"Generated/reused {len(expected)} reconstruction images: {output_dir}")
    print(f"Manifest: {manifest}")
    print(f"Manifest SHA256: {sha256(manifest)}")
    print("Next: compute LPIPS/PSNR with scripts/lpips_psnr_stable_hair.py")


if __name__ == "__main__":
    main()
