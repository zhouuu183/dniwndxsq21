import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# ========================= User Config: edit here only =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

# "both": source + one reference image used for both hairstyle and hair color.
# "full": source + hairstyle reference + hair-color reference.
USER_REFERENCE_MODE = "both"
USER_SAMPLE_SIZE = 3000
USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_SAMPLES = False

USER_FACE_ROOT = Path("celeba-1024")
USER_SHAPE_ROOT = Path("celeba-1024")
USER_COLOR_ROOT = Path("celeba-1024")

USER_BLENDING_CHECKPOINT = Path("output/blending_train_v8/checkpoints/best.pth")
USER_RUN_TAG = ""  # Empty means derive a tag from USER_BLENDING_CHECKPOINT.
USER_OUTPUT_ROOT = Path("output/blending_fid_generate_v8")
USER_MANIFEST_DIR = Path("input/blending_fid_samples_v8")
USER_REBUILD_MANIFEST = False
USER_SKIP_EXISTING_IMAGES = True
USER_BLEND_COLOR_STRENGTH_OVERRIDE = 0.0  # 0 uses checkpoint metadata; old checkpoints fall back to 1.0.
USER_SAVE_PANELS = False

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"

USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = "output/satd_train_v8_3000/checkpoints/satd_for_infer_v8.pth"
USER_SATD_BLEND_V8 = 0.34
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0

USER_EMPTY_CACHE_EVERY = 50
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

from hair_swap_v8 import get_parser_v8
from models.Alignment_v8 import Alignment_v8
from models.Embedding import Embedding
from models.Encoders import ClipBlendingModel
from models.Net import Net
from utils.image_utils import DilateErosion, equal_replacer, list_image_files
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.seed import set_seed


def normalize_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in {"both", "full"}:
        raise RuntimeError(f"Unsupported USER_REFERENCE_MODE={mode!r}. Choose 'both' or 'full'.")
    return mode


def manifest_path_for_mode(mode: str) -> Path:
    return USER_MANIFEST_DIR / f"{mode}_{USER_SAMPLE_SIZE}_seed{USER_RANDOM_SEED}.jsonl"


def checkpoint_tag(path: Path) -> str:
    if USER_RUN_TAG:
        raw = USER_RUN_TAG
    else:
        raw_path = Path(path)
        if raw_path.parent.name == "checkpoints" and raw_path.parent.parent.name:
            raw = f"{raw_path.parent.parent.name}__{raw_path.stem}"
        elif raw_path.parent.name:
            raw = f"{raw_path.parent.name}__{raw_path.stem}"
        else:
            raw = raw_path.stem
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return tag or "checkpoint"


def output_dirs(mode: str) -> tuple[Path, Path, Path]:
    run_dir = USER_OUTPUT_ROOT / mode / checkpoint_tag(USER_BLENDING_CHECKPOINT)
    generated_dir = run_dir / "blending_outputs"
    panels_dir = run_dir / "panels"
    return run_dir, generated_dir, panels_dir


def image_stem(file_name: str) -> str:
    return Path(file_name).stem


def assert_unique_stems(images: list[str], label: str) -> None:
    counts: dict[str, int] = {}
    for item in images:
        stem = image_stem(item)
        counts[stem] = counts.get(stem, 0) + 1
    duplicates = [stem for stem, count in counts.items() if count > 1]
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:10])
        raise RuntimeError(f"{label} has duplicate image stems; loading would be ambiguous: {sample}")


def sample_excluding(rng: random.Random, candidates: list[str], forbidden_stems: set[str]) -> str:
    valid = [item for item in candidates if image_stem(item) not in forbidden_stems]
    if not valid:
        raise RuntimeError("No valid candidates left after excluding duplicate stems.")
    return rng.choice(valid)


def sample_rows(
    face_images: list[str],
    shape_images: list[str],
    color_images: list[str],
    mode: str,
) -> list[dict[str, object]]:
    if not face_images:
        raise RuntimeError(f"No images found under USER_FACE_ROOT={USER_FACE_ROOT}")
    if not shape_images:
        raise RuntimeError(f"No images found under USER_SHAPE_ROOT={USER_SHAPE_ROOT}")
    if not color_images:
        raise RuntimeError(f"No images found under USER_COLOR_ROOT={USER_COLOR_ROOT}")

    if not USER_ALLOW_REUSE_ACROSS_SAMPLES:
        if USER_SAMPLE_SIZE > len(face_images) or USER_SAMPLE_SIZE > len(shape_images):
            raise RuntimeError("Not enough source/shape images for USER_SAMPLE_SIZE without reuse.")
        if mode == "full" and USER_SAMPLE_SIZE > len(color_images):
            raise RuntimeError("Not enough color images for USER_SAMPLE_SIZE without reuse.")

    rng = random.Random(USER_RANDOM_SEED)
    face_pool = face_images.copy()
    shape_pool = shape_images.copy()
    color_pool = color_images.copy()
    face_shape_same_root = USER_FACE_ROOT.resolve() == USER_SHAPE_ROOT.resolve()
    face_color_same_root = USER_FACE_ROOT.resolve() == USER_COLOR_ROOT.resolve()
    shape_color_same_root = USER_SHAPE_ROOT.resolve() == USER_COLOR_ROOT.resolve()
    rows: list[dict[str, object]] = []

    for index in range(1, USER_SAMPLE_SIZE + 1):
        if USER_ALLOW_REUSE_ACROSS_SAMPLES:
            face_file = rng.choice(face_images)
            if mode == "both":
                ref_file = sample_excluding(rng, shape_images, {image_stem(face_file)})
                shape_file = ref_file
                color_file = ref_file
            else:
                shape_file = sample_excluding(rng, shape_images, {image_stem(face_file)})
                color_file = sample_excluding(
                    rng,
                    color_images,
                    {image_stem(face_file), image_stem(shape_file)},
                )
        else:
            if not face_pool or not shape_pool or (mode == "full" and not color_pool):
                raise RuntimeError("Image pool exhausted. Reduce USER_SAMPLE_SIZE or allow reuse.")
            face_file = rng.choice(face_pool)
            if mode == "both":
                ref_file = sample_excluding(rng, shape_pool, {image_stem(face_file)})
                shape_file = ref_file
                color_file = ref_file
            else:
                shape_file = sample_excluding(rng, shape_pool, {image_stem(face_file)})
                color_file = sample_excluding(
                    rng,
                    color_pool,
                    {image_stem(face_file), image_stem(shape_file)},
                )
            face_pool.remove(face_file)
            shape_pool.remove(shape_file)
            if face_shape_same_root and face_file in shape_pool:
                shape_pool.remove(face_file)
            if face_shape_same_root and shape_file in face_pool:
                face_pool.remove(shape_file)
            if mode == "full":
                color_pool.remove(color_file)
                if face_color_same_root and face_file in color_pool:
                    color_pool.remove(face_file)
                if shape_color_same_root and shape_file in color_pool:
                    color_pool.remove(shape_file)
                if face_color_same_root and color_file in face_pool:
                    face_pool.remove(color_file)
                if shape_color_same_root and color_file in shape_pool:
                    shape_pool.remove(color_file)

        rows.append(
            {
                "index": index,
                "mode": mode,
                "output_file": f"{index:06d}.png",
                "source_file": face_file,
                "shape_file": shape_file,
                "color_file": color_file,
                "source_stem": image_stem(face_file),
                "shape_stem": image_stem(shape_file),
                "color_stem": image_stem(color_file),
            }
        )

    return rows


def save_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_manifest(path: Path, mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("mode") != mode:
                raise RuntimeError(f"Manifest {path} contains mode={row.get('mode')!r}, expected {mode!r}.")
            rows.append(row)
    if len(rows) != USER_SAMPLE_SIZE:
        raise RuntimeError(f"Manifest {path} has {len(rows)} rows, expected USER_SAMPLE_SIZE={USER_SAMPLE_SIZE}.")
    return rows


def build_or_load_manifest(mode: str) -> tuple[Path, list[dict[str, object]], bool]:
    path = manifest_path_for_mode(mode)
    if path.exists() and not USER_REBUILD_MANIFEST:
        return path, load_manifest(path, mode), False

    face_images = list_image_files(USER_FACE_ROOT)
    shape_images = list_image_files(USER_SHAPE_ROOT)
    color_images = list_image_files(USER_COLOR_ROOT)
    assert_unique_stems(face_images, "USER_FACE_ROOT")
    assert_unique_stems(shape_images, "USER_SHAPE_ROOT")
    assert_unique_stems(color_images, "USER_COLOR_ROOT")
    rows = sample_rows(face_images, shape_images, color_images, mode)
    save_manifest(path, rows)
    return path, rows, True


def row_color_path(row: dict[str, object]) -> Path:
    if row.get("mode") == "both":
        return USER_SHAPE_ROOT / str(row["color_file"])
    return USER_COLOR_ROOT / str(row["color_file"])


def validate_inputs(rows: list[dict[str, object]]) -> None:
    if not USER_BLENDING_CHECKPOINT.exists():
        raise FileNotFoundError(f"Cannot find USER_BLENDING_CHECKPOINT: {USER_BLENDING_CHECKPOINT}")
    if USER_USE_SATD_V8 and not Path(USER_SATD_CHECKPOINT_V8).exists():
        raise FileNotFoundError(f"Cannot find USER_SATD_CHECKPOINT_V8: {USER_SATD_CHECKPOINT_V8}")

    for row in rows:
        for path in (
            USER_FACE_ROOT / str(row["source_file"]),
            USER_SHAPE_ROOT / str(row["shape_file"]),
            row_color_path(row),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Manifest image missing: {path}")


def load_compatible_state_dict(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    model_state = module.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model_state.update(compatible)
    module.load_state_dict(model_state, strict=False)
    return sorted(compatible.keys())


class BlendingStageV8:
    def __init__(self, model_args):
        self.args = model_args
        self.net = Net(model_args)
        self.embed = Embedding(model_args, net=self.net)
        self.align = Alignment_v8(model_args, self.embed.get_e4e_embed, net=self.net)
        self.dilate_erosion = DilateErosion(dilate_erosion=model_args.smooth, device=model_args.device)

        checkpoint = torch.load(model_args.blending_checkpoint, map_location=model_args.device)
        self.blending_encoder = ClipBlendingModel(checkpoint.get("clip", "ViT-B/32"))
        loaded_keys = load_compatible_state_dict(
            self.blending_encoder,
            checkpoint.get("model_state_dict", checkpoint),
        )
        self.blending_encoder.to(model_args.device).eval()
        if USER_BLEND_COLOR_STRENGTH_OVERRIDE > 0:
            self.blend_color_strength = float(USER_BLEND_COLOR_STRENGTH_OVERRIDE)
        else:
            self.blend_color_strength = float(checkpoint.get("blend_color_strength", 1.0))
        print(
            f"[blending_fid_v8] loaded blending checkpoint with {len(loaded_keys)} compatible tensors "
            f"blend_color_strength={self.blend_color_strength}",
            file=sys.stderr,
        )

    @staticmethod
    def load_image(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return T.functional.to_tensor(image.convert("RGB"))

    @torch.inference_mode()
    def __call__(self, face_path: Path, shape_path: Path, color_path: Path, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            set_seed(seed)

        path_to_images: dict[Path, torch.Tensor] = {}
        images: list[torch.Tensor] = []
        for path in (face_path, shape_path, color_path):
            if path not in path_to_images:
                path_to_images[path] = self.load_image(path)
            images.append(path_to_images[path])
        images = equal_replacer(images)

        images_to_name: dict[torch.Tensor, list[str]] = defaultdict(list)
        for image, name in zip(images, ("face", "shape", "color")):
            images_to_name[image].append(name)

        name_to_embed = self.embed.embedding_images(images_to_name)
        align_shape = self.align.align_images(
            "face",
            "shape",
            name_to_embed,
            use_satd_v8=USER_USE_SATD_V8,
            satd_blend_v8=USER_SATD_BLEND_V8,
            satd_boundary_v8=USER_SATD_BOUNDARY_V8,
            eq8_reference_blend_v8=USER_EQ8_REFERENCE_BLEND_V8,
        )
        if images[1] is not images[2]:
            align_color = self.align.shape_module("face", "color", name_to_embed)
        else:
            align_color = align_shape
        return self.blend_images(align_shape, align_color, name_to_embed)

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed) -> torch.Tensor:
        i_face = name_to_embed["face"]["image_norm_256"]
        i_color = name_to_embed["color"]["image_norm_256"]

        face_mask, _ = filter_parsing_to_primary_subject(name_to_embed["face"]["mask"])
        color_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
        hair_face = torch.where(face_mask == 13, torch.ones_like(face_mask), torch.zeros_like(face_mask)).float()
        hair_color = torch.where(color_mask == 13, torch.ones_like(color_mask), torch.zeros_like(color_mask)).float()
        hair_face_dilate, _ = self.dilate_erosion.mask(hair_face)
        hair_color_dilate, hair_color_erode = self.dilate_erosion.mask(hair_color)

        latent_s_face = name_to_embed["face"]["S"]
        latent_s_color = name_to_embed["color"]["S"]
        latent_f_align = align_shape["latent_F_align"]
        aligned_target_hair = align_shape["HM_X"]

        aligned_target_hair_dilate, _ = self.dilate_erosion.mask(aligned_target_hair)
        target_mask = (1 - hair_face_dilate) * (1 - hair_color_dilate) * (1 - aligned_target_hair_dilate)

        face_tail = latent_s_face[:, 6:]
        blend_tail_raw = self.blending_encoder(
            face_tail,
            latent_s_color[:, 6:],
            i_face * target_mask,
            i_color * hair_color_erode,
        )
        blend_tail = face_tail + self.blend_color_strength * (blend_tail_raw - face_tail)
        latent_s_blend = torch.cat((latent_s_face[:, :6], blend_tail), dim=1)
        i_blend, _ = self.net.generator(
            [latent_s_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_f_align,
        )
        return ((i_blend[0] + 1) / 2).clamp(0, 1)


def build_model() -> BlendingStageV8:
    model_args = get_parser_v8().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    model_args.ckpt = USER_STYLEGAN_CKPT
    model_args.rotate_checkpoint = USER_ROTATE_CKPT
    model_args.blending_checkpoint = str(USER_BLENDING_CHECKPOINT)
    model_args.use_satd_v8 = bool(USER_USE_SATD_V8)
    model_args.satd_checkpoint_v8 = USER_SATD_CHECKPOINT_V8
    model_args.satd_blend_v8 = USER_SATD_BLEND_V8
    model_args.satd_boundary_v8 = USER_SATD_BOUNDARY_V8
    model_args.eq8_reference_blend_v8 = USER_EQ8_REFERENCE_BLEND_V8
    return BlendingStageV8(model_args)


def load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return T.functional.to_tensor(image.convert("RGB"))


def resize_chw(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[-2:]) == size:
        return image
    return F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]


def save_panel(path: Path, source_path: Path, shape_path: Path, color_path: Path, result: torch.Tensor) -> None:
    result = result.detach().cpu().clamp(0, 1)
    size = tuple(result.shape[-2:])
    source = resize_chw(load_rgb_tensor(source_path), size)
    shape = resize_chw(load_rgb_tensor(shape_path), size)
    color = resize_chw(load_rgb_tensor(color_path), size)
    save_image(torch.cat([source, shape, color, result], dim=2), path)


def write_run_config(run_dir: Path, mode: str, manifest_path: Path, generated_dir: Path) -> None:
    config = {
        "reference_mode": mode,
        "sample_size": USER_SAMPLE_SIZE,
        "seed": USER_RANDOM_SEED,
        "face_root": str(USER_FACE_ROOT),
        "shape_root": str(USER_SHAPE_ROOT),
        "color_root": str(USER_COLOR_ROOT),
        "stage": "blending",
        "blending_checkpoint": str(USER_BLENDING_CHECKPOINT),
        "satd_checkpoint_v8": USER_SATD_CHECKPOINT_V8,
        "manifest_path": str(manifest_path),
        "generated_dir": str(generated_dir),
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=True, indent=2)


@torch.inference_mode()
def main() -> None:
    mode = normalize_mode(USER_REFERENCE_MODE)
    set_seed(USER_RANDOM_SEED)

    manifest_path, rows, created_manifest = build_or_load_manifest(mode)
    validate_inputs(rows)

    run_dir, generated_dir, panels_dir = output_dirs(mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_PANELS:
        panels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, run_dir / "samples.jsonl")
    write_run_config(run_dir, mode, manifest_path, generated_dir)

    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    model = build_model()

    generated_count = 0
    skipped_count = 0
    for row in tqdm(rows, desc=f"Generate v8 FID images ({mode})"):
        output_path = generated_dir / str(row["output_file"])
        panel_path = panels_dir / str(row["output_file"])
        if USER_SKIP_EXISTING_IMAGES and output_path.exists():
            skipped_count += 1
            continue

        source_path = USER_FACE_ROOT / str(row["source_file"])
        shape_path = USER_SHAPE_ROOT / str(row["shape_file"])
        color_path = row_color_path(row)

        result = model(
            source_path,
            shape_path,
            color_path,
            seed=USER_RANDOM_SEED,
        )
        save_image(result.detach().cpu().clamp(0, 1), output_path)
        if USER_SAVE_PANELS:
            save_panel(panel_path, source_path, shape_path, color_path, result)

        generated_count += 1
        if device.type == "cuda" and USER_EMPTY_CACHE_EVERY > 0 and generated_count % USER_EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()

    final_count = len(list(generated_dir.glob("*.png")))
    if final_count != len(rows):
        raise RuntimeError(f"Generated directory has {final_count} png files, expected {len(rows)}.")

    print(f"reference mode: {mode}")
    print(f"manifest: {manifest_path} ({'created' if created_manifest else 'reused'})")
    print(f"blending checkpoint: {USER_BLENDING_CHECKPOINT}")
    print(f"generated dir: {generated_dir}")
    print(f"generated this run: {generated_count}")
    print(f"skipped existing: {skipped_count}")
    print(f"total generated images: {final_count}")
    print("Use this directory as a method dataset for scripts/fid_metric.py.")


if __name__ == "__main__":
    main()
