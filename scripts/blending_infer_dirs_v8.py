import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


# ========================= User Config: edit here only =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_SOURCE_DIR = Path("images/FFHQ_long")
USER_SHAPE_DIR = Path("images/FFHQ_short")
USER_COLOR_DIR = Path("images/FFHQ_color")
USER_OUTPUT_DIR = Path("output/blending_infer_v8")

# First run creates this file; later runs reuse it so every method sees the
# exact same source / hairstyle / hair-color triplets.
USER_TRIPLET_MANIFEST = Path("input/blending_infer_triplets_v8/triplets.jsonl")
USER_REBUILD_TRIPLET_MANIFEST = False

USER_BLENDING_CHECKPOINT = Path(
    "output/blending_train_v8_direct_anchor_v2_2/checkpoints/best_balanced.pth"
)
USER_PAIRING_MODE = "by_index"  # "by_index", "random", or "cartesian"
USER_SAMPLE_COUNT = 0  # 0 means all available for by_index/cartesian, or len(source) for random.
USER_OUTPUT_SAMPLE_COUNT = 0  # 0 means save outputs for every triplet in the manifest.
USER_RANDOM_SEED = 3407
USER_RANDOM_ALLOW_REUSE = False
USER_AVOID_SAME_STEM_WITHIN_TRIPLET = True
USER_SKIP_EXISTING_IMAGES = True
USER_SAVE_PANELS = False
USER_SAVE_INPUT_TRIPLETS = True
USER_SAVE_SATD_DESHADOW_IMAGES = True
USER_SAVE_COLOR_BLENDING_IMAGES = True
USER_INPUT_TRIPLET_SIZE = 1024

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"

USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = "output/satd_train_v8_3000/checkpoints/satd_for_infer_v8.pth"
USER_SATD_BLEND_V8 = 0.34
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0
USER_AB_NO_EDIT_THRESHOLD_V8 = 1.5
USER_AB_FULL_EDIT_THRESHOLD_V8 = 15.0
USER_HUE_NO_EDIT_DEG_V8 = 4.0
USER_HUE_FULL_EDIT_DEG_V8 = 30.0
USER_CHROMA_MAG_NO_EDIT_V8 = 2.0
USER_CHROMA_MAG_FULL_EDIT_V8 = 15.0
USER_COLOR_DIST_NO_EDIT_V8 = 2.0
USER_COLOR_DIST_FULL_EDIT_V8 = 15.0
USER_LIGHTNESS_NO_EDIT_THRESHOLD_V8 = 3.0
USER_LIGHTNESS_FULL_EDIT_THRESHOLD_V8 = 15.0
USER_MAX_GLOBAL_L_SHIFT_V8 = 40.0
USER_RELATIVE_LUMA_BINS_V8 = 8
USER_RELATIVE_LUMA_MIN_SCALE_V8 = 3.0
USER_GLOBAL_AB_FALLBACK_MIN_RELIABILITY_V8 = 0.5
USER_MIN_SAFE_REFERENCE_FRACTION_V8 = 0.35
USER_ALPHA_INIT_V8 = 0.70
USER_LAYER_OFFSET_MAX_V8 = 0.15
USER_CORRECTION_CHROMA_BUDGET_RATIO_V8 = 0.15
USER_CORRECTION_LUMA_BUDGET_RATIO_V8 = 0.10
USER_CORRECTION_ORTH_SCALE_V8 = 0.25

USER_EMPTY_CACHE_EVERY = 25
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
from models.Encoders import (
    DIRECT_COLOR_ARCH_V8_4,
    DirectColorBlendAdapterV8,
    load_direct_color_adapter_state_v8,
)
from models.Net import Net
from models.color_condition_v8 import ColorConditionConfigV8, build_color_condition_bundle
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion, equal_replacer, list_image_files
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.seed import set_seed


def normalize_choice(value: str, allowed: set[str], name: str) -> str:
    value = str(value).strip().lower()
    if value not in allowed:
        raise RuntimeError(f"Unsupported {name}={value!r}. Choose one of: {', '.join(sorted(allowed))}.")
    return value


def image_stem(file_name: str) -> str:
    return Path(file_name).stem


def list_images(root: Path, label: str) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"Cannot find {label}: {root}")
    images = list_image_files(root)
    if not images:
        raise RuntimeError(f"No jpg/jpeg/png images found under {label}: {root}")
    return images


def sample_excluding(rng: random.Random, pool: list[str], forbidden_stems: set[str]) -> str:
    candidates = [item for item in pool if image_stem(item) not in forbidden_stems]
    if not candidates:
        raise RuntimeError("No candidate left after excluding same-stem images.")
    return rng.choice(candidates)


def maybe_forbidden(*files: str) -> set[str]:
    if not USER_AVOID_SAME_STEM_WITHIN_TRIPLET:
        return set()
    return {image_stem(file) for file in files}


def build_triplets_by_index(source_files: list[str], shape_files: list[str], color_files: list[str]) -> list[dict[str, str]]:
    count = min(len(source_files), len(shape_files), len(color_files))
    if USER_SAMPLE_COUNT > 0:
        count = min(count, USER_SAMPLE_COUNT)
    triplets = []
    for index in range(count):
        source = source_files[index]
        shape = shape_files[index]
        color = color_files[index]
        if USER_AVOID_SAME_STEM_WITHIN_TRIPLET and len({image_stem(source), image_stem(shape), image_stem(color)}) < 3:
            continue
        triplets.append({"source_file": source, "shape_file": shape, "color_file": color})
    return triplets


def build_triplets_random(source_files: list[str], shape_files: list[str], color_files: list[str]) -> list[dict[str, str]]:
    rng = random.Random(USER_RANDOM_SEED)
    count = USER_SAMPLE_COUNT if USER_SAMPLE_COUNT > 0 else len(source_files)
    if not USER_RANDOM_ALLOW_REUSE and count > min(len(source_files), len(shape_files), len(color_files)):
        raise RuntimeError("USER_SAMPLE_COUNT is too large for random sampling without reuse.")

    source_pool = source_files.copy()
    shape_pool = shape_files.copy()
    color_pool = color_files.copy()
    triplets: list[dict[str, str]] = []

    for _ in range(count):
        if USER_RANDOM_ALLOW_REUSE:
            source = rng.choice(source_files)
            shape = sample_excluding(rng, shape_files, maybe_forbidden(source))
            color = sample_excluding(rng, color_files, maybe_forbidden(source, shape))
        else:
            source = rng.choice(source_pool)
            shape = sample_excluding(rng, shape_pool, maybe_forbidden(source))
            color = sample_excluding(rng, color_pool, maybe_forbidden(source, shape))
            source_pool.remove(source)
            shape_pool.remove(shape)
            color_pool.remove(color)

        triplets.append({"source_file": source, "shape_file": shape, "color_file": color})

    return triplets


def build_triplets_cartesian(source_files: list[str], shape_files: list[str], color_files: list[str]) -> list[dict[str, str]]:
    triplets: list[dict[str, str]] = []
    for source in source_files:
        for shape in shape_files:
            if USER_AVOID_SAME_STEM_WITHIN_TRIPLET and image_stem(shape) == image_stem(source):
                continue
            for color in color_files:
                if USER_AVOID_SAME_STEM_WITHIN_TRIPLET and image_stem(color) in {image_stem(source), image_stem(shape)}:
                    continue
                triplets.append({"source_file": source, "shape_file": shape, "color_file": color})
                if USER_SAMPLE_COUNT > 0 and len(triplets) >= USER_SAMPLE_COUNT:
                    return triplets
    return triplets


def build_triplets() -> list[dict[str, str]]:
    source_files = list_images(USER_SOURCE_DIR, "USER_SOURCE_DIR")
    shape_files = list_images(USER_SHAPE_DIR, "USER_SHAPE_DIR")
    color_files = list_images(USER_COLOR_DIR, "USER_COLOR_DIR")
    mode = normalize_choice(USER_PAIRING_MODE, {"by_index", "random", "cartesian"}, "USER_PAIRING_MODE")

    if mode == "by_index":
        triplets = build_triplets_by_index(source_files, shape_files, color_files)
    elif mode == "random":
        triplets = build_triplets_random(source_files, shape_files, color_files)
    else:
        triplets = build_triplets_cartesian(source_files, shape_files, color_files)

    if not triplets:
        raise RuntimeError("No triplets selected. Check directories and pairing options.")
    return triplets


def validate_triplet_paths(triplets: list[dict[str, str]]) -> None:
    for index, triplet in enumerate(triplets, start=1):
        for root, key, label in (
            (USER_SOURCE_DIR, "source_file", "source"),
            (USER_SHAPE_DIR, "shape_file", "shape"),
            (USER_COLOR_DIR, "color_file", "color"),
        ):
            path = image_path(root, triplet[key])
            if not path.exists():
                raise FileNotFoundError(f"Triplet {index} {label} image missing: {path}")


def load_triplet_manifest(path: Path) -> list[dict[str, str]]:
    triplets: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if all(key in row for key in ("source_file", "shape_file", "color_file")):
                triplet = {
                    "source_file": str(row["source_file"]),
                    "shape_file": str(row["shape_file"]),
                    "color_file": str(row["color_file"]),
                }
            elif all(key in row for key in ("source_path", "shape_path", "color_path")):
                triplet = {
                    "source_file": Path(str(row["source_path"])).name,
                    "shape_file": Path(str(row["shape_path"])).name,
                    "color_file": Path(str(row["color_path"])).name,
                }
            else:
                raise RuntimeError(f"Manifest {path} line {line_number} is missing triplet fields.")
            triplets.append(triplet)
    if not triplets:
        raise RuntimeError(f"Manifest {path} has no triplets.")
    return triplets


def save_triplet_manifest(path: Path, triplets: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for index, triplet in enumerate(triplets, start=1):
            row = {
                "index": index,
                "pairing_mode": USER_PAIRING_MODE,
                "random_seed": USER_RANDOM_SEED,
                "source_dir": str(USER_SOURCE_DIR),
                "shape_dir": str(USER_SHAPE_DIR),
                "color_dir": str(USER_COLOR_DIR),
                "source_file": triplet["source_file"],
                "shape_file": triplet["shape_file"],
                "color_file": triplet["color_file"],
                "source_path": str(image_path(USER_SOURCE_DIR, triplet["source_file"])),
                "shape_path": str(image_path(USER_SHAPE_DIR, triplet["shape_file"])),
                "color_path": str(image_path(USER_COLOR_DIR, triplet["color_file"])),
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_or_load_triplets() -> tuple[list[dict[str, str]], bool]:
    if USER_TRIPLET_MANIFEST.exists() and not USER_REBUILD_TRIPLET_MANIFEST:
        triplets = load_triplet_manifest(USER_TRIPLET_MANIFEST)
        validate_triplet_paths(triplets)
        return triplets, False

    triplets = build_triplets()
    validate_triplet_paths(triplets)
    save_triplet_manifest(USER_TRIPLET_MANIFEST, triplets)
    return triplets, True


def safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "sample"


def output_name(index: int, triplet: dict[str, str]) -> str:
    source = image_stem(triplet["source_file"])
    shape = image_stem(triplet["shape_file"])
    color = image_stem(triplet["color_file"])
    return f"{index:06d}__{safe_tag(source)}__{safe_tag(shape)}__{safe_tag(color)}.png"


def validate_checkpoints() -> None:
    for path, label in (
        (Path(USER_STYLEGAN_CKPT), "USER_STYLEGAN_CKPT"),
        (Path(USER_ROTATE_CKPT), "USER_ROTATE_CKPT"),
        (Path(USER_BLENDING_CHECKPOINT), "USER_BLENDING_CHECKPOINT"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {label}: {path}")
    if USER_USE_SATD_V8 and not Path(USER_SATD_CHECKPOINT_V8).exists():
        raise FileNotFoundError(f"Cannot find USER_SATD_CHECKPOINT_V8: {USER_SATD_CHECKPOINT_V8}")


def make_model_args():
    args = get_parser_v8().parse_args([])
    args.device = USER_DEVICE
    args.save_all = False
    args.ckpt = USER_STYLEGAN_CKPT
    args.rotate_checkpoint = USER_ROTATE_CKPT
    args.blending_checkpoint = str(USER_BLENDING_CHECKPOINT)
    args.use_satd_v8 = bool(USER_USE_SATD_V8)
    args.satd_checkpoint_v8 = USER_SATD_CHECKPOINT_V8
    args.satd_blend_v8 = USER_SATD_BLEND_V8
    args.satd_boundary_v8 = USER_SATD_BOUNDARY_V8
    args.eq8_reference_blend_v8 = USER_EQ8_REFERENCE_BLEND_V8
    args.ab_no_edit_threshold_v8 = USER_AB_NO_EDIT_THRESHOLD_V8
    args.ab_full_edit_threshold_v8 = USER_AB_FULL_EDIT_THRESHOLD_V8
    args.hue_no_edit_deg_v8 = USER_HUE_NO_EDIT_DEG_V8
    args.hue_full_edit_deg_v8 = USER_HUE_FULL_EDIT_DEG_V8
    args.chroma_mag_no_edit_v8 = USER_CHROMA_MAG_NO_EDIT_V8
    args.chroma_mag_full_edit_v8 = USER_CHROMA_MAG_FULL_EDIT_V8
    args.color_dist_no_edit_v8 = USER_COLOR_DIST_NO_EDIT_V8
    args.color_dist_full_edit_v8 = USER_COLOR_DIST_FULL_EDIT_V8
    args.lightness_no_edit_threshold_v8 = USER_LIGHTNESS_NO_EDIT_THRESHOLD_V8
    args.lightness_full_edit_threshold_v8 = USER_LIGHTNESS_FULL_EDIT_THRESHOLD_V8
    args.max_global_l_shift_v8 = USER_MAX_GLOBAL_L_SHIFT_V8
    args.relative_luma_bins_v8 = USER_RELATIVE_LUMA_BINS_V8
    args.relative_luma_min_scale_v8 = USER_RELATIVE_LUMA_MIN_SCALE_V8
    args.global_ab_fallback_min_reliability_v8 = (
        USER_GLOBAL_AB_FALLBACK_MIN_RELIABILITY_V8
    )
    args.min_safe_reference_fraction_v8 = USER_MIN_SAFE_REFERENCE_FRACTION_V8
    args.alpha_init_v8 = USER_ALPHA_INIT_V8
    args.layer_offset_max_v8 = USER_LAYER_OFFSET_MAX_V8
    args.correction_chroma_budget_ratio_v8 = USER_CORRECTION_CHROMA_BUDGET_RATIO_V8
    args.correction_luma_budget_ratio_v8 = USER_CORRECTION_LUMA_BUDGET_RATIO_V8
    args.correction_orth_scale_v8 = USER_CORRECTION_ORTH_SCALE_V8
    return args


class BlendingStageOnlyV8:
    def __init__(self, args):
        self.args = args
        self.net = Net(args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment_v8(args, self.embed.get_e4e_embed, net=self.net)
        self.dilate_erosion = DilateErosion(dilate_erosion=args.smooth, device=args.device)
        self.downsample_256 = BicubicDownSample(factor=4)
        self.color_config = ColorConditionConfigV8(
            ab_no_edit_threshold=args.ab_no_edit_threshold_v8,
            ab_full_edit_threshold=args.ab_full_edit_threshold_v8,
            hue_no_edit_deg=args.hue_no_edit_deg_v8,
            hue_full_edit_deg=args.hue_full_edit_deg_v8,
            chroma_mag_no_edit=args.chroma_mag_no_edit_v8,
            chroma_mag_full_edit=args.chroma_mag_full_edit_v8,
            color_dist_no_edit=args.color_dist_no_edit_v8,
            color_dist_full_edit=args.color_dist_full_edit_v8,
            lightness_no_edit_threshold=args.lightness_no_edit_threshold_v8,
            lightness_full_edit_threshold=args.lightness_full_edit_threshold_v8,
            max_global_l_shift=args.max_global_l_shift_v8,
            relative_luma_bins=args.relative_luma_bins_v8,
            relative_luma_min_scale=args.relative_luma_min_scale_v8,
            global_ab_fallback_min_reliability=(
                args.global_ab_fallback_min_reliability_v8
            ),
            min_safe_fraction=args.min_safe_reference_fraction_v8,
        )

        checkpoint = torch.load(args.blending_checkpoint, map_location=args.device)
        checkpoint_arch = checkpoint.get("arch") if isinstance(checkpoint, dict) else None
        if checkpoint_arch != DIRECT_COLOR_ARCH_V8_4:
            raise RuntimeError(
                f"Refusing incompatible BlendingV8 checkpoint {args.blending_checkpoint}: "
                f"arch={checkpoint_arch!r}, required={DIRECT_COLOR_ARCH_V8_4!r}"
            )
        adapter_config = checkpoint.get("adapter_config", {})
        self.blending_encoder = DirectColorBlendAdapterV8(
            checkpoint.get("clip", "ViT-B/32"),
            alpha_init=adapter_config.get("alpha_init", args.alpha_init_v8),
            layer_offset_max=adapter_config.get(
                "layer_offset_max", args.layer_offset_max_v8
            ),
            correction_chroma_budget_ratio=adapter_config.get(
                "correction_chroma_budget_ratio", args.correction_chroma_budget_ratio_v8
            ),
            correction_luma_budget_ratio=adapter_config.get(
                "correction_luma_budget_ratio", args.correction_luma_budget_ratio_v8
            ),
            correction_orth_scale=adapter_config.get(
                "correction_orth_scale", args.correction_orth_scale_v8
            ),
        )
        report = load_direct_color_adapter_state_v8(
            self.blending_encoder,
            checkpoint["model_state_dict"],
        )
        self.blending_encoder.set_anchor_trainable(False)
        self.blending_encoder.set_correction_trainable(False)
        self.blending_encoder.to(args.device).eval()
        print(
            f"[blending_infer_v8] loaded arch={DIRECT_COLOR_ARCH_V8_4} "
            f"strict adapter tensors={len(report['loaded'])}",
            file=sys.stderr,
        )

    @staticmethod
    def load_image(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return T.functional.to_tensor(image.convert("RGB"))

    @torch.inference_mode()
    def __call__(self, source_path: Path, shape_path: Path, color_path: Path, seed: int | None = None) -> torch.Tensor:
        return self.run_stages(source_path, shape_path, color_path, seed=seed)["color_blending"]

    @torch.inference_mode()
    def run_stages(self, source_path: Path, shape_path: Path, color_path: Path, seed: int | None = None) -> dict[str, torch.Tensor]:
        if seed is not None:
            set_seed(seed)

        path_to_images: dict[Path, torch.Tensor] = {}
        images: list[torch.Tensor] = []
        for path in (source_path, shape_path, color_path):
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
        align_color = self.align.shape_module("face", "color", name_to_embed) if images[1] is not images[2] else align_shape
        return {
            "satd_deshadow": self.satd_deshadow_image(align_shape, name_to_embed),
            "color_blending": self.blend_images(align_shape, align_color, name_to_embed),
        }

    @torch.inference_mode()
    def satd_deshadow_image(self, align_shape, name_to_embed) -> torch.Tensor:
        image, _ = self.net.generator(
            [name_to_embed["face"]["S"]],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=align_shape["latent_F_align"],
        )
        return ((image[0] + 1) / 2).clamp(0, 1)

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed) -> torch.Tensor:
        del align_color
        color_256 = name_to_embed["color"]["image_norm_256"]

        color_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
        color_hair = torch.where(color_mask == 13, torch.ones_like(color_mask), torch.zeros_like(color_mask)).float()
        _, color_hair_erode = self.dilate_erosion.mask(color_hair)

        # Match Blending_v8 and blending_train_v8: shape/SATD owns the
        # generated geometry, while the color image only supplies hair color.
        target_hair = align_shape["HM_X"]
        _, target_hair_erode = self.dilate_erosion.mask(target_hair)

        base_image, _ = self.net.generator(
            [name_to_embed["face"]["S"]],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=align_shape["latent_F_align"],
        )
        bundle = build_color_condition_bundle(
            reference_image=color_256,
            reference_hair_mask=color_hair_erode,
            base_image=self.downsample_256(base_image),
            target_hair_mask=target_hair_erode,
            config=self.color_config,
        )
        blend_tail = self.blending_encoder(
            latent_face=name_to_embed["face"]["S"][:, 6:],
            latent_color=name_to_embed["color"]["S"][:, 6:],
            color_descriptor=bundle["descriptor"],
            chroma_need_gate=bundle["chroma_need_gate"],
            lightness_need_gate=bundle["lightness_need_gate"],
            edit_need_gate=bundle["edit_need_gate"],
            correction_enabled=True,
        )
        blend_s = torch.cat((name_to_embed["face"]["S"][:, :6], blend_tail), dim=1)
        blend_image, _ = self.net.generator(
            [blend_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=align_shape["latent_F_align"],
        )
        return ((blend_image[0] + 1) / 2).clamp(0, 1)


def build_model() -> BlendingStageOnlyV8:
    args = make_model_args()
    return BlendingStageOnlyV8(args)


def image_path(root: Path, file_name: str) -> Path:
    return root / file_name


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


def save_input_triplet(path: Path, source_path: Path, shape_path: Path, color_path: Path) -> None:
    size = (USER_INPUT_TRIPLET_SIZE, USER_INPUT_TRIPLET_SIZE)
    source = resize_chw(load_rgb_tensor(source_path), size)
    shape = resize_chw(load_rgb_tensor(shape_path), size)
    color = resize_chw(load_rgb_tensor(color_path), size)
    save_image(torch.cat([source, shape, color], dim=2), path)


def make_run_records(
    triplets: list[dict[str, str]],
    outputs: list[str],
    satd_dir: Path,
    color_blending_dir: Path,
    panel_dir: Path,
    input_triplet_dir: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, (triplet, output_file) in enumerate(zip(triplets, outputs), start=1):
        record = {
            "index": index,
            "triplet_manifest": str(USER_TRIPLET_MANIFEST),
            "source_file": triplet["source_file"],
            "shape_file": triplet["shape_file"],
            "color_file": triplet["color_file"],
            "source_stem": image_stem(triplet["source_file"]),
            "shape_stem": image_stem(triplet["shape_file"]),
            "color_stem": image_stem(triplet["color_file"]),
            "source_path": str(image_path(USER_SOURCE_DIR, triplet["source_file"])),
            "shape_path": str(image_path(USER_SHAPE_DIR, triplet["shape_file"])),
            "color_path": str(image_path(USER_COLOR_DIR, triplet["color_file"])),
            "output_file": output_file,
            "satd_deshadow_path": str(satd_dir / output_file),
            "color_blending_path": str(color_blending_dir / output_file),
            "panel_path": str(panel_dir / output_file),
            "input_triplet_path": str(input_triplet_dir / output_file),
        }
        records.append(record)
    return records


def save_run_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    csv_path = path.with_suffix(".csv")
    fieldnames = [
        "index",
        "source_stem",
        "shape_stem",
        "color_stem",
        "source_file",
        "shape_file",
        "color_file",
        "output_file",
        "satd_deshadow_path",
        "color_blending_path",
        "input_triplet_path",
        "source_path",
        "shape_path",
        "color_path",
        "panel_path",
        "triplet_manifest",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


@torch.inference_mode()
def main() -> None:
    set_seed(USER_RANDOM_SEED)
    validate_checkpoints()

    triplets, created_triplet_manifest = build_or_load_triplets()
    if USER_OUTPUT_SAMPLE_COUNT > 0:
        triplets = triplets[:USER_OUTPUT_SAMPLE_COUNT]

    satd_dir = USER_OUTPUT_DIR / "satd_deshadow"
    color_blending_dir = USER_OUTPUT_DIR / "color_blending"
    input_triplet_dir = USER_OUTPUT_DIR / "input_triplets"
    panel_dir = USER_OUTPUT_DIR / "panels"
    if USER_SAVE_SATD_DESHADOW_IMAGES:
        satd_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_COLOR_BLENDING_IMAGES:
        color_blending_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_INPUT_TRIPLETS:
        input_triplet_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_PANELS:
        panel_dir.mkdir(parents=True, exist_ok=True)

    model = build_model() if (USER_SAVE_SATD_DESHADOW_IMAGES or USER_SAVE_COLOR_BLENDING_IMAGES) else None
    output_files = [output_name(index, triplet) for index, triplet in enumerate(triplets, start=1)]
    manifest_path = USER_OUTPUT_DIR / "manifest.jsonl"
    records = make_run_records(triplets, output_files, satd_dir, color_blending_dir, panel_dir, input_triplet_dir)
    save_run_manifest(manifest_path, records)

    satd_saved = 0
    satd_skipped = 0
    color_blending_saved = 0
    color_blending_skipped = 0
    input_triplets_saved = 0
    input_triplets_skipped = 0
    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    for index, (triplet, output_file) in enumerate(
        tqdm(list(zip(triplets, output_files)), desc="Generate v8 SATD / color-blending images"),
        start=1,
    ):
        source_path = image_path(USER_SOURCE_DIR, triplet["source_file"])
        shape_path = image_path(USER_SHAPE_DIR, triplet["shape_file"])
        color_path = image_path(USER_COLOR_DIR, triplet["color_file"])

        if USER_SAVE_INPUT_TRIPLETS:
            input_triplet_path = input_triplet_dir / output_file
            if USER_SKIP_EXISTING_IMAGES and input_triplet_path.exists():
                input_triplets_skipped += 1
            else:
                save_input_triplet(input_triplet_path, source_path, shape_path, color_path)
                input_triplets_saved += 1

        satd_path = satd_dir / output_file
        color_blending_path = color_blending_dir / output_file
        need_satd = USER_SAVE_SATD_DESHADOW_IMAGES and not (USER_SKIP_EXISTING_IMAGES and satd_path.exists())
        need_color_blending = USER_SAVE_COLOR_BLENDING_IMAGES and not (
            USER_SKIP_EXISTING_IMAGES and color_blending_path.exists()
        )
        if USER_SAVE_SATD_DESHADOW_IMAGES and not need_satd:
            satd_skipped += 1
        if USER_SAVE_COLOR_BLENDING_IMAGES and not need_color_blending:
            color_blending_skipped += 1
        if not need_satd and not need_color_blending:
            continue

        stage_outputs = model.run_stages(source_path, shape_path, color_path, seed=USER_RANDOM_SEED)
        if need_satd:
            save_image(stage_outputs["satd_deshadow"].detach().cpu().clamp(0, 1), satd_path)
            satd_saved += 1
        if need_color_blending:
            save_image(stage_outputs["color_blending"].detach().cpu().clamp(0, 1), color_blending_path)
            color_blending_saved += 1
        if USER_SAVE_PANELS:
            save_panel(panel_dir / output_file, source_path, shape_path, color_path, stage_outputs["color_blending"])

        generated_total = satd_saved + color_blending_saved
        if device.type == "cuda" and USER_EMPTY_CACHE_EVERY > 0 and generated_total % USER_EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()

    print(f"pairing mode: {USER_PAIRING_MODE}")
    print(f"triplets: {len(triplets)}")
    print(f"triplet manifest: {USER_TRIPLET_MANIFEST} ({'created' if created_triplet_manifest else 'reused'})")
    if USER_SAVE_SATD_DESHADOW_IMAGES:
        print(f"satd deshadow saved this run: {satd_saved}")
        print(f"satd deshadow skipped existing: {satd_skipped}")
        print(f"satd deshadow dir: {satd_dir}")
    if USER_SAVE_COLOR_BLENDING_IMAGES:
        print(f"color blending saved this run: {color_blending_saved}")
        print(f"color blending skipped existing: {color_blending_skipped}")
        print(f"color blending dir: {color_blending_dir}")
    if USER_SAVE_INPUT_TRIPLETS:
        print(f"input triplets saved this run: {input_triplets_saved}")
        print(f"input triplets skipped existing: {input_triplets_skipped}")
        print(f"input triplets dir: {input_triplet_dir}")
    print(f"manifest: {manifest_path}")
    print(f"manifest csv: {manifest_path.with_suffix('.csv')}")
    if USER_SAVE_PANELS:
        print(f"panels dir: {panel_dir}")


if __name__ == "__main__":
    main()
