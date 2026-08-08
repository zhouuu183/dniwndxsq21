from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ear_modules_v5 import EAR_NEGATIVE, THO_LER_DATASET_VERSION, THO_LER_METHOD, validate_dataset_part
from models.postprocess_v5 import load_checkpoint_compat


ANOMALY_DIRECTORIES = (
    "parser_missed_but_shape_found", "non_metal_candidate", "tiny_earring",
    "large_fragmented_earring", "accepted_deleted_by_negative",
    "high_missingness_low_gate", "candidate_color_drift", "negative_false_case", "preview",
)


def build_parser():
    parser = argparse.ArgumentParser(description="Audit a THO-LER V3-C dataset before training")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-previews-per-category", type=int, default=100)
    return parser


def load_rgb(path):
    with Image.open(path) as image:
        return T.functional.to_tensor(image.convert("RGB"))


def resized(value, size=(160, 160)):
    value = torch.as_tensor(value).float()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim == 3:
        value = value.unsqueeze(0)
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)[0]


def mask_rgb(value):
    value = resized(value)[:1].clamp(0, 1)
    return value.repeat(3, 1, 1)


def area(item, key):
    return float(torch.as_tensor(item[key]).float().sum())


def mean_value(item, key):
    return float(torch.as_tensor(item[key]).float().mean())


def anomaly_categories(item):
    categories = []
    parser_area = area(item, "parser_candidate_1024")
    shape_area = area(item, "shape_weak_1024")
    material_area = area(item, "material_weak_1024")
    if parser_area <= 0 and shape_area > 0:
        categories.append("parser_missed_but_shape_found")
    if shape_area > 0 and material_area <= 0:
        categories.append("non_metal_candidate")
    records = item.get("component_records_v3c", [])
    if any(record.get("scale_class") == "tiny" and record.get("status") in ("trusted", "candidate") for record in records):
        categories.append("tiny_earring")
    if max(item.get("fragments_per_group", [0]) or [0]) >= 4:
        categories.append("large_fragmented_earring")
    if float(item["hard_negative_deletion_ratio"]) > 0.15:
        categories.append("accepted_deleted_by_negative")
    if float(item["component_missingness"]) >= 0.50 and mean_value(item, "adaptive_low_delta_scale") < 0.08:
        categories.append("high_missingness_low_gate")
    candidate = torch.as_tensor(item["candidate_object_mask_before_negative_1024"]).float()
    source_chroma = torch.as_tensor(item["source_chroma_1024"]).float()
    raw_chroma = torch.as_tensor(item["raw_pp_chroma_1024"]).float()
    if float(candidate.sum()) > 0:
        candidate_color_drift = float(((source_chroma - raw_chroma).abs() * candidate).sum() / candidate.sum().clamp_min(1))
        if candidate_color_drift > 0.08:
            categories.append("candidate_color_drift")
    if int(item["earring_status"]) == EAR_NEGATIVE and area(item, "combined_weak_candidate_1024") > 0:
        categories.append("negative_false_case")
    return categories


PREVIEW_MASKS = (
    "parser_candidate_1024", "material_strong_1024", "shape_strong_1024",
    "local_scale_candidate_1024", "combined_weak_candidate_1024", "raw_component_mask_1024",
    "object_group_mask_1024", "accepted_object_mask_before_negative_1024",
    "raw_hard_negative_mask_1024", "accepted_object_guard_1024",
    "effective_hard_negative_mask_1024", "accepted_object_mask_after_negative_1024",
    "trusted_material_core_1024", "candidate_material_core_1024", "thin_contour_1024",
    "solid_interior_1024", "hollow_interior_1024", "target_hair_alpha_1024",
    "visible_candidate_support_1024", "source_raw_difference_1024", "component_missingness_map_1024",
)


def save_preview(item, path):
    images = [resized(load_rgb(item[key]))[:3].clamp(0, 1) for key in ("source_path", "target_path", "raw_pp_path")]
    images.extend(mask_rgb(item[key]) for key in PREVIEW_MASKS)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat(images, dim=-1), str(path))


def inspect_item(item, sample_index):
    records = item.get("component_records_v3c", [])
    scales = Counter(record.get("scale_class", "unknown") for record in records)
    return {
        "sample_index": sample_index,
        "source_path": item["source_path"],
        "status": int(item["earring_status"]),
        "bucket": item["sample_bucket"],
        "bridge_mode": item.get("bridge_mode", "unknown"),
        "morphology": item["morphology_category"],
        "material_positive": int(area(item, "material_weak_1024") > 0),
        "shape_positive": int(area(item, "shape_weak_1024") > 0),
        "local_only_positive": int(
            area(item, "local_scale_candidate_1024") > 0
            and area(item, "parser_candidate_1024") + area(item, "material_weak_1024") + area(item, "shape_weak_1024") <= 0
        ),
        "tiny": scales["tiny"], "small": scales["small"],
        "medium": scales["medium"], "large": scales["large"],
        "object_group_count": int(item["object_group_count"]),
        "fragment_count": int(item["fragment_count"]),
        "fragments_per_group": json.dumps(item["fragments_per_group"]),
        "accepted_area": area(item, "accepted_object_mask_before_negative_1024"),
        "raw_hard_negative_area": area(item, "raw_hard_negative_mask_1024"),
        "effective_hard_negative_area": area(item, "effective_hard_negative_mask_1024"),
        "deletion_ratio": float(item["hard_negative_deletion_ratio"]),
        "effective_deletion_ratio": float(item.get("effective_hard_negative_deletion_ratio", 0.0)),
        "trusted_raw_overlap_ratio": float(item.get("trusted_raw_hard_negative_overlap_ratio", 0.0)),
        "candidate_raw_overlap_ratio": float(item.get("candidate_raw_hard_negative_overlap_ratio", 0.0)),
        "trusted_area": area(item, "trusted_material_core_1024"),
        "candidate_area": area(item, "candidate_material_core_1024"),
        "thin_area": area(item, "thin_contour_1024"),
        "missingness": float(item["component_missingness"]),
        "source_raw_chroma_difference": float(
            (torch.as_tensor(item["source_chroma_1024"]).float()
             - torch.as_tensor(item["raw_pp_chroma_1024"]).float()).abs().mean()
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ANOMALY_DIRECTORIES:
        (args.output / name).mkdir(parents=True, exist_ok=True)
    part_files = sorted((args.dataset / "parts").glob("pp_part_*.dataset"))
    if not part_files:
        raise FileNotFoundError("No V3-C dataset parts under %s" % (args.dataset / "parts"))
    rows, category_counts, category_previews = [], Counter(), defaultdict(int)
    sample_index = 0
    for part_file in part_files:
        part = load_checkpoint_compat(str(part_file), map_location="cpu")
        if part.get("method") != THO_LER_METHOD:
            raise ValueError("Dataset part is not %s: %s" % (THO_LER_METHOD, part_file))
        for item in validate_dataset_part(part, part_file):
            row = inspect_item(item, sample_index)
            rows.append(row)
            categories = anomaly_categories(item)
            for category in categories:
                category_counts[category] += 1
                if category_previews[category] < args.max_previews_per_category:
                    save_preview(item, args.output / category / ("sample_%06d.png" % sample_index))
                    category_previews[category] += 1
            if category_previews["preview"] < args.max_previews_per_category:
                save_preview(item, args.output / "preview" / ("sample_%06d.png" % sample_index))
                category_previews["preview"] += 1
            sample_index += 1
    if not rows:
        raise RuntimeError("V3-C dataset contains no samples to audit.")
    write_csv(args.output / "samples.csv", rows)
    material_positive = sum(row["material_positive"] for row in rows)
    shape_positive = sum(row["shape_positive"] for row in rows)
    local_only_positive = sum(row["local_only_positive"] for row in rows)
    gate_a_checks = {
        "minimum_100_samples": len(rows) >= 100,
        "strict_object_graph": all(row["bridge_mode"] == "strict" for row in rows),
        "shape_path_has_candidates": shape_positive > 0,
        "tiny_or_small_has_candidates": sum(row["tiny"] + row["small"] for row in rows) > 0,
        "accepted_effective_deletion_zero": max(row["effective_deletion_ratio"] for row in rows) <= 1e-6,
    }
    summary = {
        "method": THO_LER_METHOD,
        "dataset_version": THO_LER_DATASET_VERSION,
        "sample_count": len(rows),
        "status": dict(Counter(row["status"] for row in rows)),
        "buckets": dict(Counter(row["bucket"] for row in rows)),
        "morphology": dict(Counter(row["morphology"] for row in rows)),
        "material_path_positive": material_positive,
        "shape_path_positive": shape_positive,
        "local_scale_only_positive": local_only_positive,
        "scale_counts": {scale: sum(row[scale] for row in rows) for scale in ("tiny", "small", "medium", "large")},
        "mean_object_group_count": sum(row["object_group_count"] for row in rows) / max(1, len(rows)),
        "mean_fragment_count": sum(row["fragment_count"] for row in rows) / max(1, len(rows)),
        "mean_deletion_ratio": sum(row["deletion_ratio"] for row in rows) / max(1, len(rows)),
        "max_effective_deletion_ratio": max(row["effective_deletion_ratio"] for row in rows),
        "mean_missingness": sum(row["missingness"] for row in rows) / max(1, len(rows)),
        "anomalies": dict(category_counts),
        "gate_a_checks": gate_a_checks,
        "gate_a_passed": all(gate_a_checks.values()),
    }
    (args.output / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(build_parser().parse_args())
