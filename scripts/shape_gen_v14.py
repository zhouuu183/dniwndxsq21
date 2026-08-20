# -*- coding: utf-8 -*-
import json
import os
import random
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.nhr_utils_v14 import resolve_image_paths, save_parsing_cache, write_jsonl


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_SOURCE_ROOT_FULL = Path("images/FFHQ_long")
USER_REFERENCE_ROOT_FULL = Path("images/FFHQ_short")
USER_OUTPUT_DIR_FULL = Path("images/shape_dataset_v14_full")
USER_DATASET_SIZE_FULL = 10000
USER_SOURCE_LIST_FULL = None
USER_REFERENCE_LIST_FULL = None

USER_SOURCE_ROOT_SMALL = Path("images/FFHQ_long")
USER_REFERENCE_ROOT_SMALL = Path("images/FFHQ_short")
USER_OUTPUT_DIR_SMALL = Path("images/shape_dataset_v14_small")
USER_DATASET_SIZE_SMALL = 200
USER_SOURCE_LIST_SMALL = None
USER_REFERENCE_LIST_SMALL = None

USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_PAIRS = True
USER_FIXED_TEST_PAIR_COUNT = 16
USER_FIXED_TEST_PAIRS = [
    # {"source_name": "00001.png", "reference_name": "00021.png"},
]
USER_PRECOMPUTE_PARSING_CACHE = True
# ========================================================================


def resolve_profile_defaults():
    if USER_DATASET_PROFILE == "full":
        return {
            "source_root": USER_SOURCE_ROOT_FULL,
            "reference_root": USER_REFERENCE_ROOT_FULL,
            "output_dir": USER_OUTPUT_DIR_FULL,
            "dataset_size": USER_DATASET_SIZE_FULL,
            "source_list": USER_SOURCE_LIST_FULL,
            "reference_list": USER_REFERENCE_LIST_FULL,
        }
    if USER_DATASET_PROFILE == "small":
        return {
            "source_root": USER_SOURCE_ROOT_SMALL,
            "reference_root": USER_REFERENCE_ROOT_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "dataset_size": USER_DATASET_SIZE_SMALL,
            "source_list": USER_SOURCE_LIST_SMALL,
            "reference_list": USER_REFERENCE_LIST_SMALL,
        }
    raise ValueError(f"Unsupported USER_DATASET_PROFILE: {USER_DATASET_PROFILE}")


def path_to_key(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return "__".join(rel.with_suffix("").parts)


def build_random_pairs(source_paths: list[Path], reference_paths: list[Path], dataset_size: int) -> list[tuple[Path, Path]]:
    rng = random.Random(USER_RANDOM_SEED)
    if not source_paths or not reference_paths:
        raise RuntimeError("source_paths or reference_paths is empty")

    dataset_size = max(1, int(dataset_size))
    if USER_ALLOW_REUSE_ACROSS_PAIRS:
        return [(rng.choice(source_paths), rng.choice(reference_paths)) for _ in range(dataset_size)]

    max_size = min(dataset_size, len(source_paths), len(reference_paths))
    source_paths = source_paths[:]
    reference_paths = reference_paths[:]
    rng.shuffle(source_paths)
    rng.shuffle(reference_paths)
    return list(zip(source_paths[:max_size], reference_paths[:max_size]))


def build_fixed_pairs(source_root: Path, reference_root: Path, source_paths: list[Path], reference_paths: list[Path]) -> list[tuple[Path, Path]]:
    if USER_FIXED_TEST_PAIRS:
        return [(source_root / pair["source_name"], reference_root / pair["reference_name"]) for pair in USER_FIXED_TEST_PAIRS]

    rng = random.Random(USER_RANDOM_SEED + 913)
    pair_count = max(0, int(USER_FIXED_TEST_PAIR_COUNT))
    return [(rng.choice(source_paths), rng.choice(reference_paths)) for _ in range(pair_count)]


def ensure_parsing_cache(paths: list[Path], root: Path, cache_dir: Path) -> dict[str, str]:
    mapping = {}
    for image_path in tqdm(paths, desc=f"Cache parsing: {root.name}"):
        cache_path = cache_dir / f"{path_to_key(image_path, root)}.npy"
        mapping[str(image_path)] = str(cache_path)
        if cache_path.is_file():
            continue
        save_parsing_cache(image_path, cache_path)
    return mapping


def make_record(sample_id: str, source_path: Path, reference_path: Path, source_parsing_path: str, reference_parsing_path: str) -> dict:
    return {
        "sample_id": sample_id,
        "source_path": str(source_path),
        "reference_path": str(reference_path),
        "color_path": str(reference_path),
        "source_parsing_path": source_parsing_path,
        "reference_parsing_path": reference_parsing_path,
    }


def main():
    cfg = resolve_profile_defaults()
    source_root = cfg["source_root"]
    reference_root = cfg["reference_root"]
    output_dir = cfg["output_dir"]
    dataset_size = cfg["dataset_size"]

    source_paths = resolve_image_paths(image_dir=source_root, list_file=cfg["source_list"], limit=None)
    reference_paths = resolve_image_paths(image_dir=reference_root, list_file=cfg["reference_list"], limit=None)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "parsing_cache"
    source_cache = {}
    reference_cache = {}
    if USER_PRECOMPUTE_PARSING_CACHE:
        source_cache = ensure_parsing_cache(source_paths, source_root, cache_dir / "source")
        reference_cache = ensure_parsing_cache(reference_paths, reference_root, cache_dir / "reference")

    dataset_pairs = build_random_pairs(source_paths, reference_paths, dataset_size)
    fixed_pairs = build_fixed_pairs(source_root, reference_root, source_paths, reference_paths)

    dataset_records = []
    for idx, (source_path, reference_path) in enumerate(dataset_pairs):
        dataset_records.append(
            make_record(
                sample_id=f"{idx:05d}__{source_path.stem}__{reference_path.stem}",
                source_path=source_path,
                reference_path=reference_path,
                source_parsing_path=source_cache.get(str(source_path), ""),
                reference_parsing_path=reference_cache.get(str(reference_path), ""),
            )
        )

    fixed_records = []
    for idx, (source_path, reference_path) in enumerate(fixed_pairs):
        fixed_records.append(
            make_record(
                sample_id=f"fixed_{idx:03d}__{source_path.stem}__{reference_path.stem}",
                source_path=source_path,
                reference_path=reference_path,
                source_parsing_path=source_cache.get(str(source_path), ""),
                reference_parsing_path=reference_cache.get(str(reference_path), ""),
            )
        )

    write_jsonl(dataset_records, output_dir / "manifest.jsonl")
    write_jsonl(fixed_records, output_dir / "fixed_pairs.jsonl")

    meta = {
        "dataset_profile": USER_DATASET_PROFILE,
        "source_root": str(source_root),
        "reference_root": str(reference_root),
        "dataset_size": len(dataset_records),
        "fixed_pair_count": len(fixed_records),
        "allow_reuse_across_pairs": bool(USER_ALLOW_REUSE_ACROSS_PAIRS),
        "precompute_parsing_cache": bool(USER_PRECOMPUTE_PARSING_CACHE),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=True)

    print(f"shape dataset v14 ready: {output_dir / 'manifest.jsonl'}")
    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"source root: {source_root}")
    print(f"reference root: {reference_root}")


if __name__ == "__main__":
    main()
