from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ear_modules_v5 import (
    THO_LER_DATASET_VERSION,
    THO_LER_METHOD,
    EarringVisibilityDecomposerV3A,
    OracleMaskAdapter,
    validate_dataset_part,
)
from models.postprocess_v5 import load_checkpoint_compat
from scripts.eval_pp_earring_occlusion import (
    aggregate,
    evaluate_item,
    item_data,
    load_oracle_mask,
    load_renderer,
)


EXPERIMENTS = (
    ("O0_auto_support_learned_gate", False, "learned", True),
    ("O1_oracle_support_learned_gate", True, "learned", True),
    ("O2_oracle_support_forced_gate", True, "visible_support", True),
    ("O3_oracle_forced_context_only", True, "visible_support", False),
    ("O4_oracle_forced_dual_branch", True, "visible_support", True),
)


def build_parser():
    parser = argparse.ArgumentParser(description="Run THO-LER V3-A O0-O4 Oracle Mask experiments")
    parser.add_argument("--dataset", type=Path, default=Path("images/pp_dataset_tho_ler_v3a"))
    parser.add_argument("--oracle-dir", "--oracle_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", "--max_samples", type=int, default=0)
    return parser


def oracle_data(adapter, visibility, oracle_mask, data):
    adapted = adapter(oracle_mask, data)
    adapted.update(visibility(
        adapted["earring_support_1024"], adapted["earring_core_1024"],
        adapted["topology_skeleton_1024"], adapted["target_hair_alpha_1024"],
        adapted["target_hair_core_1024"],
    ))
    return adapted


def main(args):
    checkpoint = load_checkpoint_compat(str(args.checkpoint), map_location="cpu")
    if checkpoint.get("method") != THO_LER_METHOD or checkpoint.get("dataset_version") != THO_LER_DATASET_VERSION:
        raise ValueError("Oracle experiments require a THO-LER V3-A checkpoint.")
    device = torch.device(args.device)
    renderer = load_renderer(checkpoint, device)
    adapter = OracleMaskAdapter().to(device)
    visibility = EarringVisibilityDecomposerV3A().to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    preview_dirs = {}
    for name, _, _, _ in EXPERIMENTS:
        preview_dirs[name] = args.output / name
        preview_dirs[name].mkdir(parents=True, exist_ok=True)
    rows = []
    oracle_count = 0
    for part_path in sorted((args.dataset / "parts").glob("pp_part_*.dataset")):
        part = load_checkpoint_compat(str(part_path), map_location="cpu")
        for item in validate_dataset_part(part, part_path):
            oracle_mask = load_oracle_mask(args.oracle_dir, item, device)
            if oracle_mask is None:
                continue
            automatic = item_data(item, device)
            oracle_override = oracle_data(adapter, visibility, oracle_mask, automatic)
            for name, use_oracle, gate_mode, detail_enabled in EXPERIMENTS:
                data = oracle_override if use_oracle else automatic
                metrics, preview = evaluate_item(
                    renderer, item, device, gate_mode=gate_mode, oracle_mask=oracle_mask,
                    detail_enabled=detail_enabled, data_override=data,
                )
                metrics["experiment"] = name
                metrics["oracle_support"] = bool(use_oracle)
                rows.append(metrics)
                save_image(preview, str(preview_dirs[name] / ("sample_%03d.png" % oracle_count)))
            oracle_count += 1
            if args.max_samples > 0 and oracle_count >= args.max_samples:
                break
        if args.max_samples > 0 and oracle_count >= args.max_samples:
            break
    if oracle_count == 0:
        raise RuntimeError(
            "No Oracle masks matched dataset source/raw_pp stems under %s." % args.oracle_dir
        )
    with (args.output / "oracle_per_sample_metrics.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=sorted(set(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    by_experiment = {
        name: aggregate([row for row in rows if row["experiment"] == name])
        for name, _, _, _ in EXPERIMENTS
    }
    report = {
        "method": THO_LER_METHOD,
        "dataset_version": THO_LER_DATASET_VERSION,
        "checkpoint": str(args.checkpoint),
        "oracle_sample_count": oracle_count,
        "formal_minimum_met": oracle_count >= 20,
        "experiments": by_experiment,
        "interpretation_rules": {
            "O1_gt_O0": "support recall is the primary bottleneck",
            "O2_gt_O1": "learned gate is the primary bottleneck",
            "O4_gt_O3": "full-resolution detail branch improves the context-only renderer",
        },
    }
    (args.output / "oracle_experiment_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(build_parser().parse_args())
