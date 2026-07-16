import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

# ========================= 用户配置区域：只改这里 =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_PRIOR_OUTPUT_DIR_FFHQ = Path("images/spsa_priors_v13")
USER_DATASET_OUTPUT_DIR_FFHQ = Path("images/spsa_dataset_v13")
USER_DATASET_SIZE_FFHQ = 3000
USER_REFERENCE_LIST_FFHQ = None

USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/selected_reference_shapes")
USER_SHAPE_ROOT_SMALL_EXTRA = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL_EXTRA_COUNT = 100
USER_PRIOR_OUTPUT_DIR_SMALL = Path("images/spsa_priors_v13_small")
USER_DATASET_OUTPUT_DIR_SMALL = Path("images/spsa_dataset_v13_small")
USER_DATASET_SIZE_SMALL = 300
USER_REFERENCE_LIST_SMALL = None

USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_PAIRS = True
USER_COLOR_FROM_REFERENCE = True
USER_DIRECTION_SIGMA_V13 = 2.0

USER_FIXED_TEST_PAIR_COUNT = 15
USER_FIXED_TEST_PAIRS = [
    # {"source_name": "00001.png", "reference_name": "ref_01.png"},
]
# ========================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

import torch
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR_STR = str(ROOT_DIR)
if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)


def _clear_stale_repo_modules():
    for module_name in ("utils", "datasets", "models"):
        loaded = sys.modules.get(module_name)
        loaded_file = getattr(loaded, "__file__", "") if loaded is not None else ""
        if loaded is not None and loaded_file and not str(loaded_file).startswith(ROOT_DIR_STR):
            del sys.modules[module_name]


def _build_inline_helpers():
    import cv2
    import numpy as np
    import scipy.ndimage as ndi
    from PIL import Image
    from torchvision import transforms as T

    from models.Net import get_segmentation
    from utils.image_utils import list_image_files

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    hair_label = 13
    face_labels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    eyebrow_labels = {4, 5, 6, 7}

    def read_image_tensor(path: str | Path) -> torch.Tensor:
        with Image.open(path) as image:
            return T.functional.to_tensor(image.convert("RGB"))

    def path_to_prior_key(path: str | Path, root: str | Path | None = None) -> str:
        path = Path(path)
        if root is not None:
            try:
                path = path.relative_to(Path(root))
            except ValueError:
                pass
        return "__".join(path.with_suffix("").parts)

    def resolve_image_paths(
        *,
        image_dir: str | Path,
        list_file: str | Path | None = None,
        limit: int | None = None,
    ) -> list[Path]:
        image_dir = Path(image_dir)
        if list_file is None:
            image_paths = [image_dir / file_name for file_name in list_image_files(image_dir)]
        else:
            image_paths = []
            with open(list_file, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    path = Path(line)
                    if not path.is_absolute():
                        path = image_dir / path
                    image_paths.append(path)
        if limit is not None:
            image_paths = image_paths[: max(0, int(limit))]
        return image_paths

    def build_celeba_parsing(image_tensor: torch.Tensor) -> np.ndarray:
        image_512 = T.functional.resize(
            image_tensor,
            [512, 512],
            interpolation=T.InterpolationMode.BILINEAR,
        )
        image_norm = T.functional.normalize(image_512, imagenet_mean, imagenet_std)
        image_batch = image_norm.unsqueeze(0)
        if torch.cuda.is_available():
            image_batch = image_batch.cuda()
        parsing = get_segmentation(image_batch, resize=False)
        return parsing[0, 0].detach().cpu().numpy().astype(np.int64)

    def mask_to_box(mask: np.ndarray) -> np.ndarray:
        coords = np.argwhere(mask > 0.5)
        if coords.size == 0:
            return np.array([0, 0, 1, 1], dtype=np.float32)
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0) + 1
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def expand_box(box: np.ndarray, *, image_hw: tuple[int, int], scale_x: float, scale_y: float) -> np.ndarray:
        image_h, image_w = image_hw
        x1, y1, x2, y2 = box.astype(np.float32)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        new_w = width * scale_x
        new_h = height * scale_y
        x1 = max(0.0, cx - 0.5 * new_w)
        y1 = max(0.0, cy - 0.5 * new_h)
        x2 = min(float(image_w), cx + 0.5 * new_w)
        y2 = min(float(image_h), cy + 0.5 * new_h)
        if x2 <= x1:
            x2 = min(float(image_w), x1 + 1.0)
        if y2 <= y1:
            y2 = min(float(image_h), y1 + 1.0)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def boundary_from_mask(hair_mask: np.ndarray, dilate_iter: int = 1) -> np.ndarray:
        mask_bool = hair_mask > 0.5
        dilated = ndi.binary_dilation(mask_bool, iterations=dilate_iter)
        eroded = ndi.binary_erosion(mask_bool, iterations=dilate_iter)
        return np.logical_and(dilated, np.logical_not(eroded)).astype(np.float32)

    def distance_map_from_mask(hair_mask: np.ndarray) -> np.ndarray:
        mask_bool = hair_mask > 0.5
        distance = ndi.distance_transform_edt(mask_bool).astype(np.float32)
        max_value = float(distance.max())
        if max_value > 1e-6:
            distance /= max_value
        return distance * hair_mask.astype(np.float32)

    def direction_prior_from_image(image_rgb: np.ndarray, hair_mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        image_u8 = np.clip(image_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma, sigmaY=sigma)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)
        angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-6)
        dx = np.cos(angle) * hair_mask
        dy = np.sin(angle) * hair_mask
        norm = np.sqrt(dx * dx + dy * dy)
        norm = np.where(norm > 1e-6, norm, 1.0)
        return np.stack([dx / norm, dy / norm], axis=0).astype(np.float32)

    def bang_roi_from_parsing(hair_mask: np.ndarray, parsing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image_h, image_w = hair_mask.shape
        hair_box = mask_to_box(hair_mask)
        face_mask = np.isin(parsing, list(face_labels)).astype(np.float32)
        face_box = mask_to_box(face_mask) if face_mask.any() else hair_box.copy()
        feature_mask = np.isin(parsing, list(eyebrow_labels))
        if feature_mask.any():
            feature_y = np.argwhere(feature_mask)[:, 0]
            forehead_bottom = min(image_h, int(feature_y.max() + 0.18 * max(1.0, face_box[3] - face_box[1])))
        else:
            forehead_bottom = min(image_h, int(face_box[1] + 0.30 * max(1.0, face_box[3] - face_box[1])))
        x_center = 0.5 * (face_box[0] + face_box[2])
        width = max(1.0, face_box[2] - face_box[0])
        x1 = max(hair_box[0], x_center - 0.62 * width)
        x2 = min(hair_box[2], x_center + 0.62 * width)
        y1 = hair_box[1]
        y2 = max(y1 + 1.0, min(float(hair_box[3]), float(forehead_bottom)))
        bang_box = expand_box(
            np.array([x1, y1, x2, y2], dtype=np.float32),
            image_hw=(image_h, image_w),
            scale_x=1.10,
            scale_y=1.20,
        )
        bang_mask = np.zeros_like(hair_mask, dtype=np.float32)
        bx1, by1, bx2, by2 = bang_box.astype(np.int32)
        bang_mask[by1:by2, bx1:bx2] = hair_mask[by1:by2, bx1:bx2]
        return bang_box, bang_mask

    def tail_roi_from_parsing(hair_mask: np.ndarray, parsing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image_h, image_w = hair_mask.shape
        hair_box = mask_to_box(hair_mask)
        face_mask = np.isin(parsing, list(face_labels))
        face_box = mask_to_box(face_mask.astype(np.float32)) if face_mask.any() else hair_box.copy()
        dilated_face = ndi.binary_dilation(face_mask, iterations=18)
        yy, xx = np.indices(hair_mask.shape)
        lower_band = yy >= int(hair_box[1] + 0.52 * max(1.0, hair_box[3] - hair_box[1]))
        side_band = np.logical_or(xx <= face_box[0], xx >= face_box[2])
        tail_candidate = np.logical_and(
            hair_mask > 0.5,
            np.logical_and(np.logical_not(dilated_face), np.logical_or(lower_band, side_band)),
        )
        if tail_candidate.sum() < 32:
            tail_candidate = np.logical_and(hair_mask > 0.5, lower_band)
        tail_box = expand_box(
            mask_to_box(tail_candidate.astype(np.float32)),
            image_hw=(image_h, image_w),
            scale_x=1.20,
            scale_y=1.20,
        )
        tail_mask = np.zeros_like(hair_mask, dtype=np.float32)
        tx1, ty1, tx2, ty2 = tail_box.astype(np.int32)
        tail_mask[ty1:ty2, tx1:tx2] = hair_mask[ty1:ty2, tx1:tx2]
        return tail_box, tail_mask

    def compute_spsa_prior(image_tensor: torch.Tensor, *, direction_sigma: float = 2.0) -> dict[str, np.ndarray]:
        parsing = build_celeba_parsing(image_tensor)
        hair_mask = (parsing == hair_label).astype(np.float32)
        boundary = boundary_from_mask(hair_mask)
        distance_map = distance_map_from_mask(hair_mask)
        image_512 = T.functional.resize(
            image_tensor,
            [512, 512],
            interpolation=T.InterpolationMode.BILINEAR,
        ).permute(1, 2, 0).detach().cpu().numpy()
        direction = direction_prior_from_image(image_512, hair_mask, sigma=direction_sigma)
        bang_box, bang_mask = bang_roi_from_parsing(hair_mask, parsing)
        tail_box, tail_mask = tail_roi_from_parsing(hair_mask, parsing)
        return {
            "H_tgt": hair_mask[None].astype(np.float32),
            "E_tgt": boundary[None].astype(np.float32),
            "D_tgt": distance_map[None].astype(np.float32),
            "O_tgt": direction.astype(np.float32),
            "bang_box": bang_box.astype(np.float32),
            "tail_box": tail_box.astype(np.float32),
            "bang_mask": bang_mask[None].astype(np.float32),
            "tail_mask": tail_mask[None].astype(np.float32),
        }

    def save_prior_npz(path: str | Path, prior: dict[str, np.ndarray]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **prior)

    def make_manifest_record(
        *,
        sample_id: str,
        source_path: str | Path,
        reference_path: str | Path,
        color_path: str | Path,
        prior_path: str | Path,
    ) -> dict[str, str]:
        return {
            "sample_id": sample_id,
            "source_path": str(Path(source_path)),
            "reference_path": str(Path(reference_path)),
            "color_path": str(Path(color_path)),
            "prior_path": str(Path(prior_path)),
        }

    def write_manifest(records, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=True) + "\n")

    return {
        "compute_spsa_prior": compute_spsa_prior,
        "make_manifest_record": make_manifest_record,
        "path_to_prior_key": path_to_prior_key,
        "read_image_tensor": read_image_tensor,
        "resolve_image_paths": resolve_image_paths,
        "save_prior_npz": save_prior_npz,
        "write_manifest": write_manifest,
    }


def _module_helpers(module):
    return {
        "compute_spsa_prior": module.compute_spsa_prior,
        "make_manifest_record": module.make_manifest_record,
        "path_to_prior_key": module.path_to_prior_key,
        "read_image_tensor": module.read_image_tensor,
        "resolve_image_paths": module.resolve_image_paths,
        "save_prior_npz": module.save_prior_npz,
        "write_manifest": module.write_manifest,
    }


def load_precompute_helpers():
    _clear_stale_repo_modules()

    try:
        from utils import spsa_precompute_v13 as helper_module

        return _module_helpers(helper_module)
    except (ModuleNotFoundError, ImportError):
        pass

    helper_path = ROOT_DIR / "utils" / "spsa_precompute_v13.py"
    if helper_path.is_file():
        try:
            spec = importlib.util.spec_from_file_location("utils.spsa_precompute_v13_local", helper_path)
            helper_module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(helper_module)
            return _module_helpers(helper_module)
        except Exception:
            pass

    return _build_inline_helpers()


HELPERS = load_precompute_helpers()
compute_spsa_prior = HELPERS["compute_spsa_prior"]
make_manifest_record = HELPERS["make_manifest_record"]
path_to_prior_key = HELPERS["path_to_prior_key"]
read_image_tensor = HELPERS["read_image_tensor"]
resolve_image_paths = HELPERS["resolve_image_paths"]
save_prior_npz = HELPERS["save_prior_npz"]
write_manifest = HELPERS["write_manifest"]


def resolve_profile_defaults():
    if USER_DATASET_PROFILE == "ffhq":
        return {
            "dataset_profile": "ffhq",
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "prior_output_dir": USER_PRIOR_OUTPUT_DIR_FFHQ,
            "dataset_output_dir": USER_DATASET_OUTPUT_DIR_FFHQ,
            "dataset_size": USER_DATASET_SIZE_FFHQ,
            "reference_list": USER_REFERENCE_LIST_FFHQ,
        }
    if USER_DATASET_PROFILE == "small":
        return {
            "dataset_profile": "small",
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "shape_root_extra": USER_SHAPE_ROOT_SMALL_EXTRA,
            "shape_root_extra_count": USER_SHAPE_ROOT_SMALL_EXTRA_COUNT,
            "prior_output_dir": USER_PRIOR_OUTPUT_DIR_SMALL,
            "dataset_output_dir": USER_DATASET_OUTPUT_DIR_SMALL,
            "dataset_size": USER_DATASET_SIZE_SMALL,
            "reference_list": USER_REFERENCE_LIST_SMALL,
        }
    raise ValueError(f"Unsupported USER_DATASET_PROFILE: {USER_DATASET_PROFILE}")


PROFILE_DEFAULTS = resolve_profile_defaults()


def build_parser():
    parser = argparse.ArgumentParser(description="SPSA prior precompute v13")
    parser.add_argument("--dataset_profile", type=str, default=PROFILE_DEFAULTS["dataset_profile"])
    parser.add_argument("--face_root", type=Path, default=PROFILE_DEFAULTS["face_root"])
    parser.add_argument("--shape_root", type=Path, default=PROFILE_DEFAULTS["shape_root"])
    parser.add_argument("--shape_root_extra", type=Path, default=PROFILE_DEFAULTS.get("shape_root_extra"))
    parser.add_argument("--shape_root_extra_count", type=int, default=PROFILE_DEFAULTS.get("shape_root_extra_count", 0))
    parser.add_argument("--prior_output_dir", type=Path, default=PROFILE_DEFAULTS["prior_output_dir"])
    parser.add_argument("--dataset_output_dir", type=Path, default=PROFILE_DEFAULTS["dataset_output_dir"])
    parser.add_argument("--dataset_size", type=int, default=PROFILE_DEFAULTS["dataset_size"])
    parser.add_argument("--reference_list", type=Path, default=PROFILE_DEFAULTS["reference_list"])
    parser.add_argument("--device", type=str, default=USER_DEVICE)
    parser.add_argument("--seed", type=int, default=USER_RANDOM_SEED)
    parser.add_argument("--allow_reuse_across_pairs", type=int, default=int(USER_ALLOW_REUSE_ACROSS_PAIRS))
    parser.add_argument("--color_from_reference", type=int, default=int(USER_COLOR_FROM_REFERENCE))
    parser.add_argument("--direction_sigma", type=float, default=USER_DIRECTION_SIGMA_V13)
    parser.add_argument("--fixed_test_pair_count", type=int, default=USER_FIXED_TEST_PAIR_COUNT)
    return parser


def _dedupe_paths_by_name(paths: list[Path]) -> list[Path]:
    deduped = []
    seen_names = set()
    for path in paths:
        key = path.name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        deduped.append(path)
    return deduped


def _reference_root_for_path(reference_path: Path, args) -> Path:
    extra_root = getattr(args, "shape_root_extra", None)
    if extra_root is not None:
        extra_root = Path(extra_root)
        try:
            reference_path.relative_to(extra_root)
            return extra_root
        except ValueError:
            pass
    return Path(args.shape_root)


def select_reference_paths(args) -> list[Path]:
    primary_paths = resolve_image_paths(
        image_dir=args.shape_root,
        list_file=args.reference_list,
        limit=None,
    )
    primary_paths = _dedupe_paths_by_name(primary_paths)

    extra_root = getattr(args, "shape_root_extra", None)
    extra_count = max(0, int(getattr(args, "shape_root_extra_count", 0)))
    if extra_root is None or extra_count <= 0:
        return primary_paths

    extra_paths = resolve_image_paths(
        image_dir=extra_root,
        list_file=None,
        limit=None,
    )
    primary_names = {path.name.lower() for path in primary_paths}
    extra_candidates = [path for path in extra_paths if path.name.lower() not in primary_names]
    rng = random.Random(args.seed + 1701)
    if len(extra_candidates) > extra_count:
        extra_candidates = rng.sample(extra_candidates, extra_count)
    return primary_paths + extra_candidates


def precompute_reference_priors(reference_paths: list[Path], args) -> dict[str, Path]:
    prior_paths: dict[str, Path] = {}
    for ref_path in tqdm(reference_paths, desc="Precompute SPSA priors"):
        prior_root = _reference_root_for_path(ref_path, args)
        root_tag = prior_root.name
        prior_key = f"{root_tag}__{path_to_prior_key(ref_path, root=prior_root)}"
        prior_path = args.prior_output_dir / f"{prior_key}.npz"
        prior_paths[str(ref_path)] = prior_path
        if prior_path.is_file():
            continue
        image_tensor = read_image_tensor(ref_path)
        prior = compute_spsa_prior(image_tensor, direction_sigma=args.direction_sigma)
        save_prior_npz(prior_path, prior)
    return prior_paths


def build_random_pairs(face_paths: list[Path], reference_paths: list[Path], args) -> list[tuple[Path, Path]]:
    rng = random.Random(args.seed)
    if not face_paths or not reference_paths:
        raise ValueError("Face root or shape root is empty.")

    dataset_size = max(1, int(args.dataset_size))
    allow_reuse = bool(args.allow_reuse_across_pairs)

    if allow_reuse:
        return [(rng.choice(face_paths), rng.choice(reference_paths)) for _ in range(dataset_size)]

    max_size = min(dataset_size, len(face_paths), len(reference_paths))
    rng.shuffle(face_paths)
    rng.shuffle(reference_paths)
    return list(zip(face_paths[:max_size], reference_paths[:max_size]))


def build_fixed_pairs(face_paths: list[Path], reference_paths: list[Path], args) -> list[tuple[Path, Path]]:
    if USER_FIXED_TEST_PAIRS:
        resolved = []
        for pair in USER_FIXED_TEST_PAIRS:
            source_path = args.face_root / pair["source_name"]
            reference_path = args.shape_root / pair["reference_name"]
            if not reference_path.is_file() and getattr(args, "shape_root_extra", None) is not None:
                alt_reference_path = Path(args.shape_root_extra) / pair["reference_name"]
                if alt_reference_path.is_file():
                    reference_path = alt_reference_path
            resolved.append((source_path, reference_path))
        return resolved

    rng = random.Random(args.seed + 913)
    pair_count = max(0, int(args.fixed_test_pair_count))
    if pair_count == 0:
        return []
    return [(rng.choice(face_paths), rng.choice(reference_paths)) for _ in range(pair_count)]


def pair_to_record(source_path: Path, reference_path: Path, prior_path: Path, idx: int) -> dict[str, str]:
    sample_id = f"{idx:05d}__{source_path.stem}__{reference_path.stem}"
    color_path = reference_path
    return make_manifest_record(
        sample_id=sample_id,
        source_path=source_path,
        reference_path=reference_path,
        color_path=color_path,
        prior_path=prior_path,
    )


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.prior_output_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_output_dir.mkdir(parents=True, exist_ok=True)

    face_paths = resolve_image_paths(image_dir=args.face_root, list_file=None, limit=None)
    reference_paths = select_reference_paths(args)
    prior_paths = precompute_reference_priors(reference_paths, args)

    dataset_pairs = build_random_pairs(face_paths, reference_paths, args)
    dataset_records = [
        pair_to_record(source_path, reference_path, prior_paths[str(reference_path)], idx)
        for idx, (source_path, reference_path) in enumerate(dataset_pairs)
    ]

    fixed_pairs = build_fixed_pairs(face_paths, reference_paths, args)
    fixed_records = [
        pair_to_record(source_path, reference_path, prior_paths[str(reference_path)], idx)
        for idx, (source_path, reference_path) in enumerate(fixed_pairs)
    ]

    manifest_path = args.dataset_output_dir / "manifest.jsonl"
    fixed_manifest_path = args.dataset_output_dir / "fixed_pairs.jsonl"
    write_manifest(dataset_records, manifest_path)
    write_manifest(fixed_records, fixed_manifest_path)

    meta = {
        "dataset_profile": args.dataset_profile,
        "face_root": str(args.face_root),
        "shape_root": str(args.shape_root),
        "shape_root_extra": str(args.shape_root_extra) if getattr(args, "shape_root_extra", None) else None,
        "shape_root_extra_count": int(getattr(args, "shape_root_extra_count", 0)),
        "prior_output_dir": str(args.prior_output_dir),
        "manifest_path": str(manifest_path),
        "fixed_manifest_path": str(fixed_manifest_path),
        "dataset_size": len(dataset_records),
        "fixed_pair_count": len(fixed_records),
        "reference_pool_size": len(reference_paths),
        "allow_reuse_across_pairs": bool(args.allow_reuse_across_pairs),
        "color_from_reference": bool(args.color_from_reference),
    }
    with open(args.dataset_output_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=True)

    print(
        f"SPSA v13 dataset ready: manifest={manifest_path}, fixed_pairs={fixed_manifest_path}, "
        f"priors={args.prior_output_dir}"
    )


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
