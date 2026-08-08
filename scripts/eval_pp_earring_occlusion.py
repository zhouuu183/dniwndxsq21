from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ear_modules_v5 import (
    EAR_NEGATIVE, MorphologyAwareDualBranchRenderer, THO_LER_DATASET_VERSION,
    THO_LER_METHOD, high_pass_filter, rgb_to_ycbcr, validate_dataset_part,
)
from models.postprocess_v5 import load_checkpoint_compat
from scripts.pp_gen_v5 import DATASET_MASK_KEYS, DATASET_TENSOR_KEYS


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate THO-LER V3-C recall, color and pollution")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--gate-mode", choices=("accepted_support", "learned", "morphology", "rollback"), default="rollback")
    parser.add_argument("--preview-count", type=int, default=40)
    return parser


def load_rgb(path, device):
    with Image.open(path) as image:
        value = T.functional.to_tensor(image.convert("RGB"))
    return value.unsqueeze(0).to(device)


def load_oracle_mask(oracle_dir, item, device):
    if oracle_dir is None:
        return None
    candidates = (
        oracle_dir / (Path(item["source_path"]).stem + ".png"),
        oracle_dir / (Path(item["raw_pp_path"]).stem + ".png"),
    )
    for path in candidates:
        if path.is_file():
            with Image.open(path) as image:
                mask = T.functional.to_tensor(image.convert("L"))[:1]
            return (mask.unsqueeze(0).to(device) >= 0.5).float()
    return None


def masked_mean(value, mask):
    mask = mask.float()
    if mask.shape[-2:] != value.shape[-2:]:
        mask = F.interpolate(mask, size=value.shape[-2:], mode="nearest")
    if mask.size(1) != value.size(1):
        mask = mask.expand(-1, value.size(1), -1, -1)
    return float((value * mask).sum() / mask.sum().clamp_min(1e-6))


def coverage(prediction, support):
    support = support.float()
    return float((prediction.float() * support).sum() / support.sum().clamp_min(1e-6))


def item_data(item, device):
    data = {key: torch.as_tensor(item[key]).float().unsqueeze(0).to(device) for key in DATASET_MASK_KEYS}
    data.update({key: torch.as_tensor(item[key]).float().unsqueeze(0).to(device) for key in DATASET_TENSOR_KEYS})
    for key in ("left_bbox", "right_bbox"):
        data[key] = torch.as_tensor(item[key]).long().unsqueeze(0).to(device)
    for key in (
        "earring_status", "earring_presence_score", "component_missingness",
        "component_restore_confidence", "component_source_chroma_confidence",
        "hard_negative_deletion_ratio",
    ):
        data[key] = torch.as_tensor(item[key]).reshape(1).to(device)
    return data


def evaluate_item(model, item, device, gate_mode, oracle_mask=None):
    source = load_rgb(item["source_path"], device)
    target = load_rgb(item["target_path"], device)
    raw_pp = load_rgb(item["raw_pp_path"], device)
    if source.shape[-2:] != raw_pp.shape[-2:]:
        source = F.interpolate(source, size=raw_pp.shape[-2:], mode="bilinear", align_corners=False)
    if target.shape[-2:] != raw_pp.shape[-2:]:
        target = F.interpolate(target, size=raw_pp.shape[-2:], mode="bilinear", align_corners=False)
    data = item_data(item, device)
    with torch.no_grad():
        final, aux = model(raw_pp, source, data, gate_mode=gate_mode, detail_enabled=True)
    aux = {**data, **aux}
    trusted = data["visible_trusted_core_1024"]
    candidate = data["visible_candidate_core_1024"]
    thin = data["visible_thin_contour_1024"]
    solid = data["visible_solid_interior_1024"]
    support = torch.maximum(data["trusted_support_1024"], data["candidate_support_1024"])
    hard_negative = data["effective_hard_negative_mask_1024"]
    hair_core = data["target_hair_core_1024"]
    hollow = data["visible_hollow_interior_1024"]
    change = (final - raw_pp).abs()
    source_chroma = rgb_to_ycbcr(source)[:, 1:3]
    target_chroma = rgb_to_ycbcr(target)[:, 1:3]
    final_chroma = rgb_to_ycbcr(final)[:, 1:3]
    color_mask = data["color_supervision_mask_1024"]
    source_color_distance = masked_mean((final_chroma - source_chroma).abs(), color_mask)
    target_color_distance = masked_mean((final_chroma - target_chroma).abs(), color_mask)
    row = {
        "source_path": item["source_path"],
        "status": int(item["earring_status"]),
        "bucket": item["sample_bucket"],
        "oracle_available": bool(oracle_mask is not None),
        "support_recall": 0.0,
        "trusted_recall": 0.0,
        "candidate_recall": 0.0,
        "small_object_recall": 0.0,
        "large_object_completeness": 0.0,
        "fragment_coverage": coverage(data["accepted_fragment_mask_1024"], data["raw_component_mask_1024"]),
        "trusted_gate_coverage": coverage(aux["trusted_gate"], trusted),
        "candidate_gate_coverage": coverage(aux["candidate_gate"], candidate),
        "thin_gate_coverage": coverage(aux["thin_gate"], thin),
        "high_missingness_gate_coverage": coverage(
            aux["residual_gate"], (data["component_missingness_map_1024"] >= 0.40).float() * support
        ),
        "visible_rgb_error": masked_mean((final - source).abs(), torch.maximum(trusted, candidate)),
        "high_frequency_detail_error": masked_mean(
            (high_pass_filter(final, 5, 1.0) - high_pass_filter(source, 5, 1.0)).abs(),
            torch.maximum(trusted, candidate),
        ),
        "thin_skeleton_detail_error": masked_mean(
            (high_pass_filter(final, 5, 1.0) - high_pass_filter(source, 5, 1.0)).abs(), thin
        ),
        "solid_interior_error": masked_mean((final - source).abs(), solid),
        "ycbcr_chroma_error": source_color_distance,
        "object_mean_chroma_error": source_color_distance,
        "source_chroma_distance": source_color_distance,
        "target_chroma_distance": target_color_distance,
        "target_hair_color_leakage": max(0.0, source_color_distance - target_color_distance),
        "hard_negative_pollution": masked_mean(change, hard_negative),
        "outside_local_change": masked_mean(change, 1.0 - support),
        "hair_core_change": masked_mean(change, hair_core),
        "hollow_change": masked_mean(change, hollow),
        "negative_hallucination": masked_mean(
            change, torch.ones_like(support) if int(item["earring_status"]) == EAR_NEGATIVE else torch.zeros_like(support)
        ),
        "accepted_deletion_ratio": float(item["hard_negative_deletion_ratio"]),
        "effective_deletion_ratio": float(item.get("effective_hard_negative_deletion_ratio", 0.0)),
        "missingness": float(item["component_missingness"]),
        "adaptive_low_scale": float(data["adaptive_low_delta_scale"].max()),
        "adaptive_detail_scale": float(data["adaptive_detail_delta_scale"].max()),
        "adaptive_chroma_scale": float(data["adaptive_chroma_delta_scale"].max()),
    }
    if oracle_mask is not None:
        if oracle_mask.shape[-2:] != support.shape[-2:]:
            oracle_mask = F.interpolate(oracle_mask, size=support.shape[-2:], mode="nearest")
        row["support_recall"] = coverage(support, oracle_mask)
        row["trusted_recall"] = coverage(trusted, oracle_mask)
        row["candidate_recall"] = coverage(torch.maximum(trusted, candidate), oracle_mask)
        scales = {record.get("scale_class") for record in item.get("component_records_v3c", [])}
        if scales & {"tiny", "small"}:
            row["small_object_recall"] = row["candidate_recall"]
        if "large" in scales:
            row["large_object_completeness"] = row["support_recall"]
    return row, (source, raw_pp, final, aux["residual_gate"], change)


def aggregate(rows):
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != "status"]
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in numeric}


def load_renderer(checkpoint_path, device):
    checkpoint = load_checkpoint_compat(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("method") != THO_LER_METHOD or checkpoint.get("dataset_version") != THO_LER_DATASET_VERSION:
        raise ValueError("Evaluation requires a THO-LER V3-C checkpoint.")
    config = checkpoint.get("config", {}) or {}
    model = MorphologyAwareDualBranchRenderer(
        crop_size=int(config.get("local_crop_size", 384)),
        base_channels=int(config.get("local_base_channels", 24)),
        large_crop_size=int(config.get("large_crop_size", 448)),
        gate_mode="rollback",
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(checkpoint["renderer_state_dict"], strict=True)
    return model.eval(), checkpoint


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_renderer(args.checkpoint, device)
    rows, preview_count = [], 0
    for part_path in sorted((args.dataset / "parts").glob("pp_part_*.dataset")):
        part = load_checkpoint_compat(str(part_path), map_location="cpu")
        for item in validate_dataset_part(part, part_path):
            oracle = load_oracle_mask(args.oracle_dir, item, device)
            row, preview = evaluate_item(model, item, device, args.gate_mode, oracle)
            rows.append(row)
            if preview_count < args.preview_count:
                source, raw_pp, final, gate, change = preview
                masks = gate.repeat(1, 3, 1, 1)
                save_image(torch.cat((source, raw_pp, final, masks, change), dim=-1), str(args.output / ("preview_%04d.png" % preview_count)))
                preview_count += 1
            if args.max_samples and len(rows) >= args.max_samples:
                break
        if args.max_samples and len(rows) >= args.max_samples:
            break
    if not rows:
        raise RuntimeError("No V3-C evaluation samples found.")
    metrics = aggregate(rows)
    oracle_rows = [row for row in rows if row["oracle_available"]]
    if oracle_rows:
        oracle_metrics = aggregate(oracle_rows)
        for key in ("support_recall", "trusted_recall", "candidate_recall", "small_object_recall", "large_object_completeness"):
            metrics[key] = oracle_metrics[key]
    summary = {
        "method": THO_LER_METHOD,
        "dataset_version": THO_LER_DATASET_VERSION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_overall_score": checkpoint.get("overall_score"),
        "sample_count": len(rows),
        "oracle_sample_count": len(oracle_rows),
        "metrics": metrics,
    }
    write_csv(args.output / "samples.csv", rows)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(build_parser().parse_args())
