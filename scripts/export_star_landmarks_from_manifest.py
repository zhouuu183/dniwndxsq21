"""Batch-export STAR landmarks for a fixed HairFast evaluation manifest.

This follows STAR/demo.py exactly: dlib 68-point detection chooses the crop,
then STAR Alignment.analyze returns the landmark coordinates in original-image
coordinates.  It extracts each unique source/shape image once and writes a
JSON cache consumed by celeba_hq_group_pose_from_star.py.

Run from the HairFast repository root.  STAR dependencies (including gradio)
must be installed in the active environment because STAR/demo.py imports them.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import dlib
import numpy as np
from tqdm.auto import tqdm


# ========================= User config: edit here only ========================
USER_STAR_ROOT = Path("/data/coding/hairfast_ppmodify/STAR")
USER_STAR_WEIGHT = Path(
    "/data/coding/hairfast_ppmodify/pretrained_models/STAR/"
    "WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl"
)
USER_DLIB_PREDICTOR = Path(
    "/data/coding/HairFastGAN/HairFastGAN-main/pretrained_models/"
    "ShapeAdaptor/shape_predictor_68_face_landmarks.dat"
)
USER_MANIFEST = Path(
    "/data/coding/hairfast_ppmodify/input/eval_pairs_v5/"
    "celeba_hq_both_seed3407_3000.jsonl"
)
USER_IMAGE_ROOT = Path("/root/shared-nvme/HairFastGAN/celeba-1024/celeba-1024")
USER_OUTPUT_JSON = Path(
    "/data/coding/hairfast_ppmodify/input/star_landmarks/"
    "celeba_hq_both_3000.json"
)
USER_DEVICE_IDS = "0"  # use "-1" only if STAR and its dependencies support CPU
USER_SKIP_INVALID_IMAGES = True
# ==============================================================================


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Manifest row {line_number} is not an object.")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    return rows


def unique_manifest_images(rows: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in ("source_file", "source_relpath", "shape_file", "shape_relpath"):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                continue
            if value not in seen:
                seen.add(value)
                values.append(value)
            break
        # The loop above intentionally handles source aliases first.  Add the
        # shape role separately so both roles are always included.
        for key in ("shape_file", "shape_relpath"):
            value = row.get(key)
            if isinstance(value, str) and value and value not in seen:
                seen.add(value)
                values.append(value)
            if isinstance(value, str) and value:
                break
    return values


def image_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def largest_detection(detections: list[Any]) -> Any:
    return max(detections, key=lambda item: int(item.width()) * int(item.height()))


def extract_landmarks(
    image_path_value: str,
    image_root: Path,
    alignment: Any,
    detector: Any,
    predictor: Any,
) -> dict[str, Any]:
    path = image_path(image_root, image_path_value)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image: {path}")
    detections = detector(image, 1)
    if len(detections) == 0:
        raise RuntimeError(f"dlib found no face: {path}")
    detection = largest_detection(detections)
    face = predictor(image, detection)
    shape = np.asarray([(face.part(index).x, face.part(index).y) for index in range(68)])
    x1, x2 = shape[:, 0].min(), shape[:, 0].max()
    y1, y2 = shape[:, 1].min(), shape[:, 1].max()
    scale = min(x2 - x1, y2 - y1) / 200.0 * 1.05
    if scale <= 0:
        raise RuntimeError(f"Invalid dlib face box: {path}")
    center_w = float((x2 + x1) / 2.0)
    center_h = float((y2 + y1) / 2.0)
    points = alignment.analyze(
        image,
        float(scale),
        center_w,
        center_h,
    )
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] < 76:
        raise RuntimeError(f"STAR returned invalid landmarks {points.shape}: {path}")
    height, width = image.shape[:2]
    return {
        "points": points[:, :2].tolist(),
        "image_size": [int(width), int(height)],
    }


def main() -> None:
    star_root = USER_STAR_ROOT.expanduser().resolve()
    weight_path = USER_STAR_WEIGHT.expanduser().resolve()
    predictor_path = USER_DLIB_PREDICTOR.expanduser().resolve()
    manifest_path = USER_MANIFEST.expanduser().resolve()
    image_root = USER_IMAGE_ROOT.expanduser().resolve()
    output_path = USER_OUTPUT_JSON.expanduser().resolve()
    for path, label in (
        (star_root / "demo.py", "STAR demo.py"),
        (weight_path, "STAR weight"),
        (predictor_path, "dlib 68-point predictor"),
        (manifest_path, "fixed manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    rows = load_manifest(manifest_path)
    image_values = unique_manifest_images(rows)
    if not image_values:
        raise RuntimeError("No source/shape images were found in the manifest.")

    original_cwd = Path.cwd()
    os.chdir(star_root)
    sys.path.insert(0, str(star_root))
    try:
        # Import the exact preprocessing/model implementation from STAR/demo.py.
        # demo.py imports gradio for its optional web UI, but Alignment itself
        # does not use gradio.  A stub keeps this batch exporter lightweight.
        if "gradio" not in sys.modules:
            sys.modules["gradio"] = types.ModuleType("gradio")
        from demo import Alignment  # type: ignore[import-not-found]

        args = type("Args", (), {"config_name": "alignment"})()
        device_ids = [int(value.strip()) for value in USER_DEVICE_IDS.split(",")]
        alignment = Alignment(
            args,
            str(weight_path),
            dl_framework="pytorch",
            device_ids=device_ids,
        )
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(predictor_path))

        cache: dict[str, Any] = {}
        skipped: list[dict[str, str]] = []
        for value in tqdm(image_values, desc="STAR landmarks"):
            try:
                cache[value] = extract_landmarks(
                    value,
                    image_root,
                    alignment,
                    detector,
                    predictor,
                )
            except Exception as error:  # noqa: BLE001 - preserve bad input
                if not USER_SKIP_INVALID_IMAGES:
                    raise
                skipped.append({"image": value, "reason": str(error)})

        if not cache:
            raise RuntimeError("STAR failed on every manifest image.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "landmarks": cache,
            "metadata": {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "image_root": str(image_root),
                "star_root": str(star_root),
                "star_weight": str(weight_path),
                "star_weight_sha256": sha256(weight_path),
                "predictor": str(predictor_path),
                "image_count_requested": len(image_values),
                "image_count_valid": len(cache),
                "image_count_skipped": len(skipped),
                "skipped": skipped,
                "coordinate_space": "pixel",
                "landmarks_returned": 98,
            },
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote STAR cache: {output_path}")
        print(f"Valid images: {len(cache)}; skipped: {len(skipped)}")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
