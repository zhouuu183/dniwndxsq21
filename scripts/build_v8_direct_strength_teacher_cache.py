from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.blending_train_v8 as train_v8
from models.SG_IDCT_v16 import lab_to_rgb, rgb_to_lab
from models.direct_strength_teacher_v8 import (
    DIRECT_STRENGTH_TEACHER_ARCH_V8_4,
    alpha_score_field,
    compute_teacher_score,
    select_teacher_strength,
    summarize_teacher_records,
    validate_teacher_distribution,
)
from models.color_condition_v8 import compute_reference_fidelity_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the V8.4 direct-strength teacher cache")
    parser.add_argument(
        "--output",
        type=Path,
        default=train_v8.ACTIVE_DATASET_DIR / train_v8.USER_TEACHER_CACHE_NAME,
    )
    parser.add_argument("--force", action="store_true")
    return parser


def gather_candidate(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return value.gather(1, index[:, None]).squeeze(1)


def luma_shift_preview(prepared: dict[str, object], index: int, shift: float) -> torch.Tensor:
    pseudo_lab = prepared["pseudo_lab"][index : index + 1].clone()
    mask = prepared["color_transfer_mask"][index : index + 1]
    base_lab = rgb_to_lab(prepared["base_i"][index : index + 1])
    pseudo_lab[:, 0:1] = (
        base_lab[:, 0:1] + mask * float(shift)
    ).clamp(0, 100)
    return lab_to_rgb(pseudo_lab).clamp(0, 1) * 2.0 - 1.0


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Teacher cache already exists: {args.output}; pass --force to rebuild")

    train_v8.set_seed(train_v8.USER_RANDOM_SEED)
    triplets = train_v8.read_triplets(train_v8.ACTIVE_DATASET_DIR)
    if not triplets:
        raise RuntimeError(
            f"No 3-column experiments found in {train_v8.ACTIVE_DATASET_DIR / 'dataset.exps'}"
        )
    train_v8.ensure_dataset_cache_v8(triplets)
    device = torch.device(train_v8.USER_DEVICE if torch.cuda.is_available() else "cpu")
    helper = train_v8.MaskPrepHelper(device)
    dataset = train_v8.BlendingDatasetV8(
        triplets,
        train_v8.ACTIVE_DATASET_DIR,
        train_v8.ACTIVE_FACE_ROOT,
        train_v8.ACTIVE_COLOR_ROOT,
        teacher_records=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_v8.USER_BATCH_SIZE,
        shuffle=False,
        num_workers=train_v8.USER_NUM_WORKERS,
        pin_memory=train_v8.USER_PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
    )

    dummy_model = nn.Linear(1, 1).to(device)
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.0)
    trainer = train_v8.BlendingTrainerV8(
        dummy_model, dummy_optimizer, [], [], helper
    )
    candidates = tuple(float(value) for value in train_v8.USER_TEACHER_ALPHA_CANDIDATES)
    records: dict[str, dict[str, float | bool | list[float]]] = {}
    diagnostic_dir = args.output.parent / "teacher_direct_strength_v8_4_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="V8.4 direct-strength teacher sweep"):
        prepared = trainer.prepare_batch(batch)
        if prepared is None:
            continue
        score_columns = []
        color_score_columns = []
        ab_columns = []
        hue_columns = []
        chroma_columns = []
        candidate_images = []
        for alpha in candidates:
            source_tail = prepared["align_s"][:, 6:]
            color_tail = prepared["color_s"][:, 6:]
            candidate_tail = source_tail + alpha * (color_tail - source_tail)
            candidate_image = trainer.render_blend_tail(prepared, candidate_tail)
            candidate_images.append(candidate_image)
            candidate_lab = rgb_to_lab(candidate_image)
            ref_metrics = compute_reference_fidelity_metrics(
                candidate_lab,
                prepared["color_transfer_mask"],
                prepared["ref_stats"],
                trainer.color_config,
            )
            face_keep_region = (
                prepared["face_keep_mask"] * prepared["satd_protect_mask"]
            ).clamp(0, 1)
            face_keep_error = trainer.masked_l1_per_sample(
                candidate_image, prepared["base_i"], face_keep_region
            )
            luma_excess = candidate_lab[:, 0:1] - prepared["pseudo_lab"][:, 0:1]
            luma_artifact = trainer.masked_mean_per_sample(
                torch.relu(luma_excess - train_v8.USER_LUMA_EXCESS_MARGIN),
                prepared["color_transfer_mask"],
            )
            score, color_score = compute_teacher_score(
                ref_metrics["mean_ab_error"],
                ref_metrics["hue_error"],
                ref_metrics["chroma_error"],
                face_keep_error,
                luma_artifact,
            )
            score_columns.append(score)
            color_score_columns.append(color_score)
            ab_columns.append(ref_metrics["mean_ab_error"])
            hue_columns.append(ref_metrics["hue_error"])
            chroma_columns.append(ref_metrics["chroma_error"])

        score_matrix = torch.stack(score_columns, dim=1)
        selection = select_teacher_strength(
            candidates, score_matrix, margin_scale=train_v8.USER_TEACHER_MARGIN_SCALE
        )
        best_index = selection["teacher_index"]
        color_matrix = torch.stack(color_score_columns, dim=1)
        ab_matrix = torch.stack(ab_columns, dim=1)
        hue_matrix = torch.stack(hue_columns, dim=1)
        chroma_matrix = torch.stack(chroma_columns, dim=1)

        for index, sample_id in enumerate(prepared["sample_id"]):
            record: dict[str, float | bool | list[float]] = {
                "teacher_alpha": float(selection["teacher_alpha"][index].item()),
                "teacher_score": float(selection["teacher_score"][index].item()),
                "teacher_color_score": float(
                    gather_candidate(color_matrix, best_index)[index].item()
                ),
                "teacher_confidence": float(selection["teacher_confidence"][index].item()),
                "teacher_margin": float(selection["teacher_margin"][index].item()),
                "teacher_ref_ab_error": float(
                    gather_candidate(ab_matrix, best_index)[index].item()
                ),
                "teacher_ref_hue_error": float(
                    gather_candidate(hue_matrix, best_index)[index].item()
                ),
                "teacher_ref_chroma_error": float(
                    gather_candidate(chroma_matrix, best_index)[index].item()
                ),
                "high_color": bool(prepared["chroma_need_gate"][index].item() >= 0.90),
                "white_or_light_hair": bool(
                    prepared["ref_stats"]["l_median"][index].item() >= 70.0
                ),
            }
            for candidate_index, alpha in enumerate(candidates):
                record[alpha_score_field(alpha)] = float(
                    score_matrix[index, candidate_index].item()
                )
            records[sample_id] = record

            global_index = len(records) - 1
            if global_index in train_v8.USER_FIXED_REGRESSION_INDICES:
                train_v8.save_preview(
                    diagnostic_dir / f"sample_{global_index:03d}_teacher_alpha_sweep.png",
                    [
                        prepared["base_i"][index : index + 1],
                        prepared["color_i"][index : index + 1],
                        prepared["pseudo_rgb"][index : index + 1],
                        *[image[index : index + 1] for image in candidate_images],
                    ],
                )
                train_v8.save_preview(
                    diagnostic_dir / f"sample_{global_index:03d}_lightness_sweep.png",
                    [
                        prepared["base_i"][index : index + 1],
                        prepared["color_i"][index : index + 1],
                        *[
                            luma_shift_preview(prepared, index, shift)
                            for shift in (10.0, 20.0, 30.0, 40.0)
                        ],
                    ],
                )

    if not records:
        raise RuntimeError("Teacher sweep produced no valid records")
    summary = summarize_teacher_records(records)
    payload = {
        "arch": DIRECT_STRENGTH_TEACHER_ARCH_V8_4,
        "alpha_candidates": list(candidates),
        "records": records,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary_path = args.output.parent / "teacher_strength_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"teacher cache: {args.output}")
    print(f"teacher summary: {summary_path}")
    validate_teacher_distribution(records)


if __name__ == "__main__":
    main()
