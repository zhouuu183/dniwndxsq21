import csv
import json
import os
import random
import re
import sys
from pathlib import Path


# ========================= User Config: edit here only =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_SOURCE_DIR = Path("input/source_faces")
USER_SHAPE_DIR = Path("input/reference_shapes")
USER_COLOR_DIR = Path("input/reference_colors")
USER_OUTPUT_DIR = Path("output/blending_infer_v5")

USER_TRIPLET_MANIFEST = Path("input/blending_infer_triplets_v5/triplets.jsonl")
USER_REBUILD_TRIPLET_MANIFEST = False

USER_BLENDING_CHECKPOINT = Path("pretrained_models/Blending/checkpoint.pth")
USER_PP_V5_CKPT = Path(r"C:\Users\yxdzy\Desktop\pp_v5_checkpoints_small\best_14.pth")
USER_ENABLE_CLEANUP_FACE_REFINER = False
USER_EAR_LOW_ALPHA = 0.1
USER_EAR_MASK_INIT_BIAS = -4.0
USER_EARRING_FINE_MASK_FLOOR = 0.18
USER_EARRING_FINE_MASK_DILATE = 5
USER_EARRING_OBJECT_DILATE = 9
USER_EARRING_OBJECT_SUPPORT_DILATE = 13
USER_ENABLE_EARRING_QUERY_RECALL = True
USER_EARRING_QUERY_RECALL_DILATE = 7
USER_EARRING_QUERY_DOWNWARD_SHIFT = 18
USER_EARRING_QUERY_LOWER_LOBE_WEIGHT = 0.20
USER_EARRING_QUERY_CANDIDATE_BOOST = 0.90
USER_EARRING_QUERY_BLOCK_PROTECT = 0.85
USER_PAIRING_MODE = "by_index"  # "by_index", "random", or "cartesian"
USER_SAMPLE_COUNT = 0  # 0 means all available for by_index/cartesian, or len(source) for random.
USER_RANDOM_SEED = 3407
USER_RANDOM_ALLOW_REUSE = False
USER_AVOID_SAME_STEM_WITHIN_TRIPLET = True
USER_SKIP_EXISTING_IMAGES = True
USER_SAVE_PANELS = False
USER_SAVE_DEBUG_MASKS = False
USER_SAVE_INPUT_TRIPLETS = True
USER_INPUT_TRIPLET_COUNT = 0  # 0 means save a triplet image for every selected sample.
USER_INPUT_TRIPLET_SIZE = 1024

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"

USER_USE_SATD_V8 = False
USER_SATD_CHECKPOINT_V8 = "output/satd_train_v8_3000/checkpoints/satd_for_infer_v8.pth"
USER_SATD_BLEND_V8 = 0.34
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0

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

from hair_swap_v5 import HairFastV5, get_parser
from utils.image_utils import list_image_files
from utils.seed import set_seed


def normalize_choice(value: str, allowed: set[str], name: str) -> str:
    value = str(value).strip().lower()
    if value not in allowed:
        raise RuntimeError(f"Unsupported {name}={value!r}. Choose one of: {', '.join(sorted(allowed))}.")
    return value


def image_stem(file_name: str) -> str:
    return Path(file_name).stem


def image_path(root: Path, file_name: str) -> Path:
    return root / file_name


def list_images(root: Path, label: str) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"Cannot find {label}: {root}")
    images = list_image_files(root)
    if not images:
        raise RuntimeError(f"No jpg/jpeg/png images found under {label}: {root}")
    return images


def maybe_forbidden(*files: str) -> set[str]:
    if not USER_AVOID_SAME_STEM_WITHIN_TRIPLET:
        return set()
    return {image_stem(file) for file in files}


def sample_excluding(rng: random.Random, pool: list[str], forbidden_stems: set[str]) -> str:
    candidates = [item for item in pool if image_stem(item) not in forbidden_stems]
    if not candidates:
        raise RuntimeError("No candidate left after excluding same-stem images.")
    return rng.choice(candidates)


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
                triplets.append(
                    {
                        "source_file": str(row["source_file"]),
                        "shape_file": str(row["shape_file"]),
                        "color_file": str(row["color_file"]),
                    }
                )
            elif all(key in row for key in ("source_path", "shape_path", "color_path")):
                triplets.append(
                    {
                        "source_file": Path(str(row["source_path"])).name,
                        "shape_file": Path(str(row["shape_path"])).name,
                        "color_file": Path(str(row["color_path"])).name,
                    }
                )
            else:
                raise RuntimeError(f"Manifest {path} line {line_number} is missing triplet fields.")
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
        (Path(USER_PP_V5_CKPT), "USER_PP_V5_CKPT"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {label}: {path}")
    if USER_USE_SATD_V8 and not Path(USER_SATD_CHECKPOINT_V8).exists():
        raise FileNotFoundError(f"Cannot find USER_SATD_CHECKPOINT_V8: {USER_SATD_CHECKPOINT_V8}")


def make_model_args():
    args = get_parser().parse_args([])
    args.device = USER_DEVICE
    args.save_all = bool(USER_SAVE_DEBUG_MASKS)
    args.save_all_dir = USER_OUTPUT_DIR / "debug"
    args.ckpt = USER_STYLEGAN_CKPT
    args.rotate_checkpoint = USER_ROTATE_CKPT
    args.blending_checkpoint = str(USER_BLENDING_CHECKPOINT)
    args.pp_checkpoint = str(USER_PP_V5_CKPT)
    args.pp_v5_checkpoint = str(USER_PP_V5_CKPT)
    args.enable_cleanup_face_refiner = bool(USER_ENABLE_CLEANUP_FACE_REFINER)
    args.ear_low_alpha = USER_EAR_LOW_ALPHA
    args.ear_mask_init_bias = USER_EAR_MASK_INIT_BIAS
    args.earring_fine_mask_floor = USER_EARRING_FINE_MASK_FLOOR
    args.earring_fine_mask_dilate = USER_EARRING_FINE_MASK_DILATE
    args.earring_object_dilate = USER_EARRING_OBJECT_DILATE
    args.earring_object_support_dilate = USER_EARRING_OBJECT_SUPPORT_DILATE
    args.enable_earring_query_recall = bool(USER_ENABLE_EARRING_QUERY_RECALL)
    args.earring_query_recall_dilate = USER_EARRING_QUERY_RECALL_DILATE
    args.earring_query_downward_shift = USER_EARRING_QUERY_DOWNWARD_SHIFT
    args.earring_query_lower_lobe_weight = USER_EARRING_QUERY_LOWER_LOBE_WEIGHT
    args.earring_query_candidate_boost = USER_EARRING_QUERY_CANDIDATE_BOOST
    args.earring_query_block_protect = USER_EARRING_QUERY_BLOCK_PROTECT
    args.use_satd_v8 = bool(USER_USE_SATD_V8)
    args.satd_checkpoint_v8 = USER_SATD_CHECKPOINT_V8
    args.satd_blend_v8 = USER_SATD_BLEND_V8
    args.satd_boundary_v8 = USER_SATD_BOUNDARY_V8
    args.eq8_reference_blend_v8 = USER_EQ8_REFERENCE_BLEND_V8
    return args


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
    result_dir: Path,
    panel_dir: Path,
    input_triplet_dir: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, (triplet, output_file) in enumerate(zip(triplets, outputs), start=1):
        records.append(
            {
                "index": index,
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
                "output_path": str(result_dir / output_file),
                "panel_path": str(panel_dir / output_file),
                "input_triplet_path": str(input_triplet_dir / output_file),
                "triplet_manifest": str(USER_TRIPLET_MANIFEST),
            }
        )
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
        "output_path",
        "source_path",
        "shape_path",
        "color_path",
        "input_triplet_path",
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
    result_dir = USER_OUTPUT_DIR / "images"
    panel_dir = USER_OUTPUT_DIR / "panels"
    input_triplet_dir = USER_OUTPUT_DIR / "input_triplets"
    result_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_PANELS:
        panel_dir.mkdir(parents=True, exist_ok=True)
    if USER_SAVE_INPUT_TRIPLETS:
        input_triplet_dir.mkdir(parents=True, exist_ok=True)

    model = HairFastV5(make_model_args())
    output_files = [output_name(index, triplet) for index, triplet in enumerate(triplets, start=1)]
    manifest_path = USER_OUTPUT_DIR / "manifest.jsonl"
    records = make_run_records(triplets, output_files, result_dir, panel_dir, input_triplet_dir)
    save_run_manifest(manifest_path, records)

    generated = 0
    skipped = 0
    input_triplets_saved = 0
    device = torch.device(USER_DEVICE if torch.cuda.is_available() else "cpu")
    for index, (triplet, output_file) in enumerate(tqdm(list(zip(triplets, output_files)), desc="Generate v5 images"), start=1):
        output_path = result_dir / output_file
        source_path = image_path(USER_SOURCE_DIR, triplet["source_file"])
        shape_path = image_path(USER_SHAPE_DIR, triplet["shape_file"])
        color_path = image_path(USER_COLOR_DIR, triplet["color_file"])

        if USER_SAVE_INPUT_TRIPLETS and (USER_INPUT_TRIPLET_COUNT <= 0 or index <= USER_INPUT_TRIPLET_COUNT):
            input_triplet_path = input_triplet_dir / output_file
            if not (USER_SKIP_EXISTING_IMAGES and input_triplet_path.exists()):
                save_input_triplet(input_triplet_path, source_path, shape_path, color_path)
                input_triplets_saved += 1

        if USER_SKIP_EXISTING_IMAGES and output_path.exists():
            skipped += 1
            continue

        result = model(
            source_path,
            shape_path,
            color_path,
            seed=USER_RANDOM_SEED,
            use_satd_v8=USER_USE_SATD_V8,
            satd_blend_v8=USER_SATD_BLEND_V8,
            satd_boundary_v8=USER_SATD_BOUNDARY_V8,
            eq8_reference_blend_v8=USER_EQ8_REFERENCE_BLEND_V8,
            exp_name=Path(output_file).stem,
        )
        save_image(result.detach().cpu().clamp(0, 1), output_path)
        if USER_SAVE_PANELS:
            save_panel(panel_dir / output_file, source_path, shape_path, color_path, result)

        generated += 1
        if device.type == "cuda" and USER_EMPTY_CACHE_EVERY > 0 and generated % USER_EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()

    print(f"pairing mode: {USER_PAIRING_MODE}")
    print(f"triplets: {len(triplets)}")
    print(f"triplet manifest: {USER_TRIPLET_MANIFEST} ({'created' if created_triplet_manifest else 'reused'})")
    print(f"generated: {generated}")
    print(f"skipped existing: {skipped}")
    if USER_SAVE_INPUT_TRIPLETS:
        print(f"input triplets saved this run: {input_triplets_saved}")
        print(f"input triplets dir: {input_triplet_dir}")
    print(f"images dir: {result_dir}")
    print(f"manifest: {manifest_path}")
    print(f"manifest csv: {manifest_path.with_suffix('.csv')}")
    if USER_SAVE_PANELS:
        print(f"panels dir: {panel_dir}")


if __name__ == "__main__":
    main()
