from __future__ import annotations

import gc
import os
import random
import sys
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ========================= User Config: edit only here =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_EXPS_FILE_FFHQ = Path("")
USER_OUTPUT_DIR_FFHQ = Path("images/shape_dataset_v19_3000")
USER_MAX_SAMPLES_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ")
USER_EXPS_FILE_SMALL = Path("")
USER_OUTPUT_DIR_SMALL = Path("images/shape_dataset_v19_small")
USER_MAX_SAMPLES_SMALL = 300

USER_AUTO_COLOR_MODE = "shape"
USER_RANDOM_SEED = 3407
USER_BATCH_SIZE = 1
USER_SKIP_EXISTING = True
USER_SAVE_ALIGN_COLOR = False
USER_LOW_MEMORY_MODE = True
USER_CLEAR_CACHE_EVERY = 8

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"

USER_PRECOMPUTE_WITH_TRAINED_V19 = False
USER_SHAPE_ADAPTER_V19_CKPT = ""
USER_SHAPE_ADAPTER_HIDDEN_V19 = 256
USER_SHAPE_V19_USE_ROTATION = True
USER_SHAPE_V19_STRENGTH = 1.0
USER_SHAPE_V19_PRIOR_STRENGTH = 0.35
USER_SHAPE_V19_DETAIL_STRENGTH = 1.0
USER_SHAPE_V19_MASK_RESIDUAL_GAIN = 1.0
USER_SHAPE_V19_BOUNDARY_RADIUS = 3
# ========================================================================


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.io import ImageReadMode, read_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Alignment_v19 import Alignment_v19
from models.Embedding_v19 import Embedding_v19
from models.Net import Net
from utils.train import seed_everything, toggle_grad

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
ALIGN_SHAPE_CACHE_VERSION_V19 = "v19_rotate_shape_only_v5"


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
            "exps_file": USER_EXPS_FILE_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "max_samples": USER_MAX_SAMPLES_FFHQ,
        },
        "small": {
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "color_root": USER_COLOR_ROOT_SMALL,
            "exps_file": USER_EXPS_FILE_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "max_samples": USER_MAX_SAMPLES_SMALL,
        },
    }
    if USER_DATASET_PROFILE not in profiles:
        raise RuntimeError(f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}.")
    return profiles[USER_DATASET_PROFILE]


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_FACE_ROOT = _DATASET_CFG["face_root"]
ACTIVE_SHAPE_ROOT = _DATASET_CFG["shape_root"]
ACTIVE_COLOR_ROOT = _DATASET_CFG["color_root"]
ACTIVE_EXPS_FILE = _DATASET_CFG["exps_file"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_MAX_SAMPLES = int(_DATASET_CFG["max_samples"])


def normalize_optional_path(path_value: Path | str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    path_str = str(path).strip()
    if path_str in ("", "."):
        return None
    return path


def build_opts(device: torch.device) -> Namespace:
    return Namespace(
        size=1024,
        ckpt=USER_STYLEGAN_CKPT,
        channel_multiplier=2,
        latent=512,
        n_mlp=8,
        device=str(device),
        batch_size=USER_BATCH_SIZE,
        save_all=False,
        save_all_dir=Path("output"),
        mixing=0.95,
        smooth=5,
        rotate_checkpoint=USER_ROTATE_CKPT,
        blending_checkpoint=USER_BLENDING_CKPT,
        pp_checkpoint=USER_PP_CKPT,
        e4e_batch_size=1 if USER_LOW_MEMORY_MODE else USER_BATCH_SIZE,
        use_shape_v19=bool(USER_PRECOMPUTE_WITH_TRAINED_V19),
        shape_adapter_v19_checkpoint=USER_SHAPE_ADAPTER_V19_CKPT,
        shape_adapter_v19_hidden=USER_SHAPE_ADAPTER_HIDDEN_V19,
        shape_v19_use_rotation=bool(USER_SHAPE_V19_USE_ROTATION),
        shape_v19_strength=USER_SHAPE_V19_STRENGTH,
        shape_v19_prior_strength=USER_SHAPE_V19_PRIOR_STRENGTH,
        shape_v19_detail_strength=USER_SHAPE_V19_DETAIL_STRENGTH,
        shape_v19_mask_residual_gain=USER_SHAPE_V19_MASK_RESIDUAL_GAIN,
        shape_v19_boundary_radius=USER_SHAPE_V19_BOUNDARY_RADIUS,
    )


def list_image_stems(root: Path) -> list[str]:
    return sorted(build_image_index(root).keys())


def build_image_index(root: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in sorted(root.rglob(f"*{extension}")):
            image_index.setdefault(path.stem, path)
    return image_index


def load_exps_or_build() -> list[tuple[str, str, str]]:
    exps_file = normalize_optional_path(ACTIVE_EXPS_FILE)
    if exps_file is not None and exps_file.is_file():
        exps = []
        with open(exps_file, "r", encoding="utf-8") as file:
            for line in file:
                items = line.strip().split()
                if len(items) == 3:
                    exps.append((items[0], items[1], items[2]))
        return exps

    rng = random.Random(USER_RANDOM_SEED)
    face_stems = list_image_stems(ACTIVE_FACE_ROOT)
    shape_stems = list_image_stems(ACTIVE_SHAPE_ROOT)
    color_stems = list_image_stems(ACTIVE_COLOR_ROOT) if USER_SAVE_ALIGN_COLOR else []
    color_stem_set = set(color_stems)
    shape_color_shared_stems = sorted(set(shape_stems) & color_stem_set) if USER_SAVE_ALIGN_COLOR else []
    if not face_stems or not shape_stems:
        raise RuntimeError("Cannot auto-build dataset.exps because face_root or shape_root is empty.")
    if USER_SAVE_ALIGN_COLOR and not color_stems:
        raise RuntimeError("Cannot auto-build dataset.exps because color_root is empty while USER_SAVE_ALIGN_COLOR=True.")
    if USER_SAVE_ALIGN_COLOR and USER_AUTO_COLOR_MODE == "shape" and not shape_color_shared_stems:
        raise RuntimeError(
            "USER_AUTO_COLOR_MODE='shape' requires shared stems between shape_root and color_root, "
            f"but none were found under {ACTIVE_SHAPE_ROOT} and {ACTIVE_COLOR_ROOT}."
        )

    exps = []
    for idx in range(ACTIVE_MAX_SAMPLES):
        face_stem = face_stems[idx % len(face_stems)]
        if not USER_SAVE_ALIGN_COLOR:
            shape_stem = rng.choice(shape_stems)
            color_stem = face_stem
        elif USER_AUTO_COLOR_MODE == "shape":
            shape_stem = rng.choice(shape_color_shared_stems)
            color_stem = shape_stem
        elif USER_AUTO_COLOR_MODE == "face":
            shape_stem = rng.choice(shape_stems)
            color_stem = face_stem
            if color_stem not in color_stem_set:
                color_stem = rng.choice(color_stems)
        else:
            shape_stem = rng.choice(shape_stems)
            color_stem = rng.choice(color_stems)
        exps.append((face_stem, shape_stem, color_stem))
    return exps


def save_exps(path: Path, exps: list[tuple[str, str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for face_stem, shape_stem, color_stem in exps:
            file.write(f"{face_stem} {shape_stem} {color_stem}\n")


def load_tensor(path: Path) -> torch.Tensor:
    return read_image(str(path), mode=ImageReadMode.RGB)


def load_image_256(root: Path, stem: str, image_index: dict[str, Path] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    path = resolve_image_path(root, stem, image_index=image_index)
    with Image.open(path) as image:
        image = image.convert("RGB").resize((256, 256), Image.BICUBIC)
        image_256 = T.functional.to_tensor(image).unsqueeze(0)
    image_norm_256 = image_256 * 2.0 - 1.0
    return image_256, image_norm_256


def load_npz_tensor(path: Path, key: str, device: torch.device) -> torch.Tensor:
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain key {key!r}")
        return torch.from_numpy(data[key]).float().to(device)


def resolve_image_path(root: Path, stem: str, image_index: dict[str, Path] | None = None) -> Path:
    if image_index is not None and stem in image_index:
        return image_index[stem]
    for ext in IMAGE_EXTENSIONS:
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    for ext in IMAGE_EXTENSIONS:
        matches = sorted(root.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find image for {stem} under {root}")


def role_fs_path(role: str, stem: str) -> Path:
    return ACTIVE_OUTPUT_DIR / "FS" / f"{role}_{stem}.npz"


def role_mask_path(role: str, stem: str) -> Path:
    return ACTIVE_OUTPUT_DIR / "Mask" / f"{role}_{stem}.npz"


def align_shape_path(face_stem: str, shape_stem: str) -> Path:
    return ACTIVE_OUTPUT_DIR / "AlignShape" / f"face_{face_stem}__alignshape_{shape_stem}.npz"


def align_color_path(face_stem: str, color_stem: str) -> Path:
    return ACTIVE_OUTPUT_DIR / "AlignColor" / f"face_{face_stem}__aligncolor_{color_stem}.npz"


def save_npz(path: Path, **payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def resolve_aligned_reference_image_v19(
    align_shape: dict[str, torch.Tensor],
    shape_cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    aligned_reference = align_shape.get("aligned_reference_image_256")
    if aligned_reference is not None:
        return aligned_reference
    if not getattr(resolve_aligned_reference_image_v19, "_warned_missing_key", False):
        print(
            "[shape_gen_v19] Warning: align_images() did not return "
            "'aligned_reference_image_256'; falling back to the normalized shape reference image."
        )
        resolve_aligned_reference_image_v19._warned_missing_key = True
    return shape_cache["image_norm_256"]


def align_shape_cache_complete(path: Path) -> bool:
    if not path.exists():
        return False
    required_keys = {
        "latent_F_base",
        "latent_F_shape",
        "latent_F_reference",
        "latent_F_base64",
        "latent_F_shape64",
        "latent_F_reference64",
        "target_hair_mask",
    }
    with np.load(path) as data:
        files = set(data.files)
        has_reference_image = "reference_image_256" in files or "aligned_reference_image_256" in files
        cache_version = str(np.asarray(data["shape_v19_cache_version"]).item()) if "shape_v19_cache_version" in files else ""
        cache_use_rotation = bool(int(np.asarray(data["shape_v19_use_rotation"]).item())) if "shape_v19_use_rotation" in files else None
        return (
            required_keys.issubset(files)
            and has_reference_image
            and cache_version == ALIGN_SHAPE_CACHE_VERSION_V19
            and cache_use_rotation == bool(USER_SHAPE_V19_USE_ROTATION)
        )


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_release_memory(step_idx: int):
    if USER_LOW_MEMORY_MODE and USER_CLEAR_CACHE_EVERY > 0 and (step_idx + 1) % USER_CLEAR_CACHE_EVERY == 0:
        release_memory()


def save_role_cache(
    embed: Embedding_v19,
    role: str,
    stem: str,
    root: Path,
    image_index: dict[str, Path],
):
    role_key = f"{role}::{stem}"
    image_tensor = load_tensor(resolve_image_path(root, stem, image_index=image_index))
    role_data = embed.embedding_images({image_tensor: [role_key]})[role_key]
    save_npz(
        role_fs_path(role, stem),
        latent_W=role_data["W"].detach().cpu().numpy(),
        latent_S=role_data["S"].detach().cpu().numpy(),
        latent_F32=role_data["F32"].detach().cpu().numpy(),
        latent_F64=role_data["F64"].detach().cpu().numpy(),
    )
    save_npz(role_mask_path(role, stem), parsing_mask=role_data["mask"].detach().cpu().numpy())
    del image_tensor, role_data


def load_role_cache(
    role: str,
    stem: str,
    root: Path,
    device: torch.device,
    image_index: dict[str, Path],
) -> dict[str, torch.Tensor]:
    fs_path = role_fs_path(role, stem)
    mask_path = role_mask_path(role, stem)
    image_256, image_norm_256 = load_image_256(root, stem, image_index=image_index)
    latent_f32 = load_npz_tensor(fs_path, "latent_F32", device)
    return {
        "W": load_npz_tensor(fs_path, "latent_W", device),
        "S": load_npz_tensor(fs_path, "latent_S", device),
        "F": latent_f32,
        "F32": latent_f32,
        "F64": load_npz_tensor(fs_path, "latent_F64", device),
        "mask": load_npz_tensor(mask_path, "parsing_mask", device),
        "image_256": image_256.to(device),
        "image_norm_256": image_norm_256.to(device),
    }


def main():
    seed_everything(USER_RANDOM_SEED)
    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    opts = build_opts(device)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    net = Net(opts)
    embed = Embedding_v19(opts, net=net).to(device).eval()
    align = Alignment_v19(opts, latent_encoder=embed.get_e4e_embed, net=net).to(device).eval()
    toggle_grad(net.generator, False)
    toggle_grad(embed, False)
    toggle_grad(align, False)

    exps = load_exps_or_build()
    if ACTIVE_MAX_SAMPLES > 0:
        exps = exps[:ACTIVE_MAX_SAMPLES]
    save_exps(ACTIVE_OUTPUT_DIR / "dataset.exps", exps)

    role_entries = set()
    shape_pairs = set()
    color_pairs = set()
    image_indices = {
        ACTIVE_FACE_ROOT: build_image_index(ACTIVE_FACE_ROOT),
        ACTIVE_SHAPE_ROOT: build_image_index(ACTIVE_SHAPE_ROOT),
    }
    if USER_SAVE_ALIGN_COLOR:
        image_indices[ACTIVE_COLOR_ROOT] = build_image_index(ACTIVE_COLOR_ROOT)
    for face_stem, shape_stem, color_stem in exps:
        role_entries.add(("face", face_stem, ACTIVE_FACE_ROOT))
        role_entries.add(("shape", shape_stem, ACTIVE_SHAPE_ROOT))
        shape_pairs.add((face_stem, shape_stem))
        if USER_SAVE_ALIGN_COLOR:
            role_entries.add(("color", color_stem, ACTIVE_COLOR_ROOT))
            color_pairs.add((face_stem, color_stem))

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"roles: {len(role_entries)} | shape pairs: {len(shape_pairs)} | color pairs: {len(color_pairs)}")
    print(f"output dir: {ACTIVE_OUTPUT_DIR}")

    release_memory()

    for idx, (role, stem, root) in enumerate(tqdm(sorted(role_entries), desc="save role latents")):
        fs_path = role_fs_path(role, stem)
        mask_path = role_mask_path(role, stem)
        if USER_SKIP_EXISTING and fs_path.exists() and mask_path.exists():
            continue
        save_role_cache(embed, role, stem, root, image_indices[root])
        maybe_release_memory(idx)

    for idx, (face_stem, shape_stem) in enumerate(tqdm(sorted(shape_pairs), desc="save shape pairs")):
        output_path = align_shape_path(face_stem, shape_stem)
        if USER_SKIP_EXISTING and align_shape_cache_complete(output_path):
            continue
        face_cache = load_role_cache("face", face_stem, ACTIVE_FACE_ROOT, device, image_indices[ACTIVE_FACE_ROOT])
        shape_cache = load_role_cache("shape", shape_stem, ACTIVE_SHAPE_ROOT, device, image_indices[ACTIVE_SHAPE_ROOT])
        pair_embed = {
            "face": face_cache,
            "shape": shape_cache,
        }
        align_shape = align.align_images("face", "shape", pair_embed)
        priors = align_shape["priors"]
        reference_image_256 = shape_cache["image_norm_256"]
        aligned_reference_image_256 = resolve_aligned_reference_image_v19(align_shape, shape_cache)
        save_npz(
            output_path,
            latent_F_base=align_shape.get("latent_F_base", align_shape["latent_F_align"]).detach().cpu().numpy(),
            latent_F_shape=align_shape["latent_F_shape"].detach().cpu().numpy(),
            latent_F_intermediate=align_shape["latent_F_intermediate"].detach().cpu().numpy(),
            latent_F_reference=align_shape["latent_F_reference"].detach().cpu().numpy(),
            latent_F_base64=align_shape.get("latent_F_base64", align_shape["latent_F_intermediate64"]).detach().cpu().numpy(),
            latent_F_shape64=align_shape["latent_F_shape64"].detach().cpu().numpy(),
            latent_F_reference64=align_shape["latent_F_reference64"].detach().cpu().numpy(),
            latent_F64_detail_target=align_shape["latent_F64_detail"].detach().cpu().numpy(),
            reference_image_256=reference_image_256.detach().cpu().numpy(),
            aligned_reference_image_256=aligned_reference_image_256.detach().cpu().numpy(),
            target_hair_mask=align_shape["target_hair_mask"].detach().cpu().numpy(),
            shape_v19_cache_version=np.array(ALIGN_SHAPE_CACHE_VERSION_V19),
            shape_v19_use_rotation=np.array(int(USER_SHAPE_V19_USE_ROTATION), dtype=np.int64),
            **{key: value.detach().cpu().numpy() for key, value in priors.items()},
        )
        del face_cache, shape_cache, pair_embed, align_shape, priors
        maybe_release_memory(idx)

    if USER_SAVE_ALIGN_COLOR:
        for idx, (face_stem, color_stem) in enumerate(tqdm(sorted(color_pairs), desc="save color pairs")):
            output_path = align_color_path(face_stem, color_stem)
            if USER_SKIP_EXISTING and output_path.exists():
                continue
            face_cache = load_role_cache("face", face_stem, ACTIVE_FACE_ROOT, device, image_indices[ACTIVE_FACE_ROOT])
            if face_stem == color_stem and ACTIVE_FACE_ROOT == ACTIVE_COLOR_ROOT:
                color_cache = face_cache
            else:
                color_cache = load_role_cache("color", color_stem, ACTIVE_COLOR_ROOT, device, image_indices[ACTIVE_COLOR_ROOT])
            pair_embed = {
                "face": face_cache,
                "color": color_cache,
            }
            align_color = align.shape_module("face", "color", pair_embed)
            save_npz(
                output_path,
                target_hair_mask=align_color["HM_X"].detach().cpu().numpy(),
                pseudo_target_mask=align_color["pseudo_target_mask"].detach().cpu().numpy(),
                boundary_band=align_color["priors"]["boundary_band"].detach().cpu().numpy(),
            )
            if color_cache is not face_cache:
                del color_cache
            del face_cache, pair_embed, align_color
            maybe_release_memory(idx)

    print(f"done: {ACTIVE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
