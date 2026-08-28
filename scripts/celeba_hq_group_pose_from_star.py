"""Group a fixed CelebA-HQ manifest by STAR landmark pose difference.

This script intentionally does not guess how to construct STAR from a .pkl
file.  Run STAR's official inference code first and export a JSON cache, then
use this script to score and split the *same* fixed manifest.  The cache may
be either ``{path: [[x, y], ...]}`` or ``{path: {"points": [...], ...}}``.
Coordinates are pixel coordinates by default; set USER_COORDINATE_SPACE to
"normalized" when STAR exported values in [0, 1].
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


# ========================= User config: edit here only ========================
USER_MODE = "full"  # same mode as the fixed input manifest
USER_CELEBA_HQ_DIR = Path("/root/shared-nvme/HairFastGAN/celeba-1024")
USER_INPUT_MANIFEST = Path(
    "input/eval_pairs_v5/celeba_hq_full_seed3407_3000.jsonl"
)
USER_STAR_LANDMARKS_JSON = Path("input/star_landmarks/celeba_hq_full_3000.json")
USER_OUTPUT_ROOT = Path("input/eval_pairs_pose_v1")
USER_EXPECTED_INPUT_ROWS = 3000
USER_COORDINATE_SPACE = "pixel"  # "pixel" or "normalized"

# WFLW/STAR indexing must be checked against the STAR repository you use.
# These are the conventional WFLW eye ranges within the first 76 landmarks.
USER_LEFT_EYE_INDICES = (60, 61, 62, 63, 64, 65, 66, 67)
USER_RIGHT_EYE_INDICES = (68, 69, 70, 71, 72, 73, 74, 75)
USER_LANDMARK_COUNT = 76
USER_GROUP_NAMES = ("easy", "medium", "hard")
USER_SKIP_INVALID_ROWS = True
# ===============================================================================


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def resolve_image(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object.")
            rows.append(value)
    return rows


def load_landmarks(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("landmarks"), dict):
        payload = payload["landmarks"]
    if not isinstance(payload, dict):
        raise RuntimeError(
            "STAR landmark cache must be an object mapping image path to points."
        )
    return payload


def landmark_entry(cache: dict[str, Any], image_root: Path, value: str) -> Any:
    image_path = resolve_image(image_root, value)
    candidates = [value, Path(value).as_posix(), str(image_path), image_path.as_posix()]
    matches = [cache[key] for key in candidates if key in cache]
    if not matches:
        basename_matches = [entry for key, entry in cache.items() if Path(key).name == image_path.name]
        if len(basename_matches) == 1:
            return basename_matches[0]
        raise KeyError(f"No unique STAR landmarks found for {value}")
    return matches[0]


def parse_points(entry: Any, image_path: Path) -> tuple[list[list[float]], tuple[int, int]]:
    width, height = 1024, 1024
    points = entry
    if isinstance(entry, dict):
        points = entry.get("points", entry.get("landmarks"))
        size = entry.get("image_size", entry.get("size"))
        if isinstance(size, (list, tuple)) and len(size) == 2:
            width, height = int(size[0]), int(size[1])
    if points is None or not isinstance(points, (list, tuple)):
        raise ValueError("landmark entry has no points array")
    parsed: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("landmark point is not [x, y]")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("landmark point contains NaN/Inf")
        parsed.append([x, y])
    if image_path.is_file():
        with Image.open(image_path) as image:
            width, height = image.size
    return parsed, (width, height)


def normalized_points(points: list[list[float]], size: tuple[int, int]) -> list[list[float]]:
    if USER_COORDINATE_SPACE == "normalized":
        return points
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {size}")
    return [[point[0] / width, point[1] / height] for point in points]


def pose_score(
    source_points: list[list[float]],
    shape_points: list[list[float]],
) -> tuple[float, float]:
    if len(source_points) < USER_LANDMARK_COUNT or len(shape_points) < USER_LANDMARK_COUNT:
        raise ValueError(
            f"STAR must provide at least {USER_LANDMARK_COUNT} points; "
            f"got {len(source_points)} and {len(shape_points)}"
        )
    source = source_points[:USER_LANDMARK_COUNT]
    shape = shape_points[:USER_LANDMARK_COUNT]
    squared = [
        (source[index][0] - shape[index][0]) ** 2
        + (source[index][1] - shape[index][1]) ** 2
        for index in range(USER_LANDMARK_COUNT)
    ]
    rmse = math.sqrt(sum(squared) / USER_LANDMARK_COUNT)

    def center(indices: tuple[int, ...]) -> tuple[float, float]:
        return (
            sum(source[index][0] for index in indices) / len(indices),
            sum(source[index][1] for index in indices) / len(indices),
        )

    left = center(USER_LEFT_EYE_INDICES)
    right = center(USER_RIGHT_EYE_INDICES)
    iod = math.hypot(left[0] - right[0], left[1] - right[1])
    if iod <= 1e-8:
        raise ValueError("source interocular distance is zero")
    return rmse / iod, iod


def make_group_manifest(
    rows: list[dict[str, Any]],
    transfer_mode: str,
    group: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, original in enumerate(rows, start=1):
            row = dict(original)
            # ``mode`` remains the original pairing contract (full/both).
            # The pose difficulty belongs only in ``pose_group``.
            row["mode"] = transfer_mode
            row["index"] = index
            row["output_file"] = f"{index:06d}.png"
            row["pose_group"] = group
            row["pose_original_index"] = int(original["index"])
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    mode = USER_MODE.strip().lower()
    if mode not in {"both", "full"}:
        raise ValueError("USER_MODE must be 'both' or 'full'.")
    if USER_COORDINATE_SPACE not in {"pixel", "normalized"}:
        raise ValueError("USER_COORDINATE_SPACE must be 'pixel' or 'normalized'.")
    if set(USER_GROUP_NAMES) != {"easy", "medium", "hard"}:
        raise ValueError("USER_GROUP_NAMES must contain easy, medium and hard.")

    manifest_path = resolve_path(USER_INPUT_MANIFEST)
    image_root = USER_CELEBA_HQ_DIR.expanduser().resolve()
    cache_path = resolve_path(USER_STAR_LANDMARKS_JSON)
    rows = load_jsonl(manifest_path)
    if len(rows) != USER_EXPECTED_INPUT_ROWS:
        raise RuntimeError(
            f"Expected {USER_EXPECTED_INPUT_ROWS} manifest rows, found {len(rows)}."
        )
    cache = load_landmarks(cache_path)

    scored: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        try:
            source_name = str(row.get("source_file", row.get("source_relpath", "")))
            shape_name = str(row.get("shape_file", row.get("shape_relpath", "")))
            source_path = resolve_image(image_root, source_name)
            shape_path = resolve_image(image_root, shape_name)
            source_entry = landmark_entry(cache, image_root, source_name)
            shape_entry = landmark_entry(cache, image_root, shape_name)
            source_points, source_size = parse_points(source_entry, source_path)
            shape_points, shape_size = parse_points(shape_entry, shape_path)
            source_points = normalized_points(source_points, source_size)
            shape_points = normalized_points(shape_points, shape_size)
            score, iod = pose_score(source_points, shape_points)
            scored.append(
                {
                    "row": dict(row),
                    "pose_score": score,
                    "source_iod": iod,
                }
            )
        except Exception as error:  # noqa: BLE001 - record bad STAR rows
            if not USER_SKIP_INVALID_ROWS:
                raise RuntimeError(f"Cannot score manifest row {row.get('index')}: {error}") from error
            skipped.append({"index": row.get("index"), "reason": str(error), "row": row})

    if len(scored) < 3:
        raise RuntimeError(f"Only {len(scored)} valid rows remain; need at least 3.")
    scored.sort(key=lambda item: (float(item["pose_score"]), int(item["row"]["index"])))
    base, remainder = divmod(len(scored), 3)
    offset = 0
    group_records: dict[str, list[dict[str, Any]]] = {}
    for group_index, group in enumerate(USER_GROUP_NAMES):
        size = base + (1 if group_index < remainder else 0)
        records = scored[offset : offset + size]
        offset += size
        group_records[group] = records
        for rank, item in enumerate(records, start=1):
            item["group"] = group
            item["group_rank"] = rank

    output_root = resolve_path(USER_OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    score_path = output_root / f"pose_scores_{mode}_seed3407.jsonl"
    with score_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in scored:
            record = dict(item["row"])
            record.update(
                {
                    "pose_score": float(item["pose_score"]),
                    "source_iod": float(item["source_iod"]),
                    "pose_group": item["group"],
                    "pose_group_rank": item["group_rank"],
                }
            )
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    for group, records in group_records.items():
        group_rows = [item["row"] for item in records]
        make_group_manifest(
            group_rows,
            mode,
            group,
            output_root / f"celeba_hq_{mode}_seed3407_pose_{group}.jsonl",
        )

    metadata = {
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256(manifest_path),
        "star_landmarks_json": str(cache_path),
        "star_landmarks_sha256": sha256(cache_path),
        "mode": mode,
        "coordinate_space": USER_COORDINATE_SPACE,
        "landmark_count_used": USER_LANDMARK_COUNT,
        "left_eye_indices": list(USER_LEFT_EYE_INDICES),
        "right_eye_indices": list(USER_RIGHT_EYE_INDICES),
        "valid_rows": len(scored),
        "skipped_rows": len(skipped),
        "group_counts": {group: len(records) for group, records in group_records.items()},
        "groups": {
            group: {
                "min_score": float(records[0]["pose_score"]),
                "max_score": float(records[-1]["pose_score"]),
            }
            for group, records in group_records.items()
        },
    }
    (output_root / f"pose_groups_{mode}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    if skipped:
        (output_root / f"pose_skipped_{mode}.json").write_text(
            json.dumps(skipped, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )

    print(f"Input rows: {len(rows)}; valid STAR rows: {len(scored)}; skipped: {len(skipped)}")
    for group, records in group_records.items():
        print(
            f"{group}: {len(records)} rows, score range "
            f"{records[0]['pose_score']:.6f}..{records[-1]['pose_score']:.6f}"
        )
    print(f"Output directory: {output_root}")


if __name__ == "__main__":
    main()
