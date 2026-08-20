import argparse
import json
import os
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.blending_train_v8 as train_v8
from models.Encoders import DIRECT_COLOR_ARCH_V8_4, load_direct_color_adapter_state_v8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one direct-color-anchor V8.4 regression sample")
    parser.add_argument("--val-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=train_v8.ACTIVE_DATASET_DIR)
    parser.add_argument("--face-root", type=Path, default=train_v8.ACTIVE_FACE_ROOT)
    parser.add_argument("--color-root", type=Path, default=train_v8.ACTIVE_COLOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("output/v8_direct_color_diagnostic"))
    return parser


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().mean().item())


def create_untrained_model(device: torch.device):
    return train_v8.BlendingModel(
        train_v8.USER_CLIP_MODEL,
        alpha_init=train_v8.USER_ALPHA_INIT,
        layer_offset_max=train_v8.USER_LAYER_OFFSET_MAX,
        correction_chroma_budget_ratio=train_v8.USER_CORRECTION_CHROMA_BUDGET_RATIO,
        correction_luma_budget_ratio=train_v8.USER_CORRECTION_LUMA_BUDGET_RATIO,
        correction_orth_scale=train_v8.USER_CORRECTION_ORTH_SCALE,
    ).to(device).eval()


def load_v4_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_arch = checkpoint.get("arch") if isinstance(checkpoint, dict) else None
    if checkpoint_arch != DIRECT_COLOR_ARCH_V8_4:
        raise RuntimeError(
            f"Refusing incompatible checkpoint {checkpoint_path}: "
            f"arch={checkpoint_arch!r}, required={DIRECT_COLOR_ARCH_V8_4!r}"
        )
    adapter_config = checkpoint.get("adapter_config", {})
    model.alpha_init = float(adapter_config.get("alpha_init", model.alpha_init))
    model.layer_offset_max = float(adapter_config.get("layer_offset_max", model.layer_offset_max))
    for key in (
        "correction_chroma_budget_ratio",
        "correction_luma_budget_ratio",
        "correction_orth_scale",
    ):
        setattr(model, key, float(adapter_config.get(key, getattr(model, key))))
    report = load_direct_color_adapter_state_v8(model, checkpoint["model_state_dict"])
    model.set_anchor_trainable(False)
    model.set_correction_trainable(False)
    print(
        f"[v8_direct_diagnostic] loaded arch={DIRECT_COLOR_ARCH_V8_4} "
        f"strict adapter tensors={len(report['loaded'])}",
        file=sys.stderr,
    )


def create_trainer(model: torch.nn.Module, device: torch.device):
    helper = train_v8.MaskPrepHelper(device)
    model.set_anchor_trainable(True)
    model.set_correction_trainable(False)
    optimizer = torch.optim.Adam(list(model.anchor_parameters()), lr=train_v8.USER_LR_STAGE_A)
    trainer = train_v8.BlendingTrainerV8(model, optimizer, [], [], helper)
    return trainer, helper


def prepare_validation_case(
    trainer: train_v8.BlendingTrainerV8,
    val_index: int,
    dataset_dir: Path,
    face_root: Path,
    color_root: Path,
):
    triplets = train_v8.read_triplets(dataset_dir)
    if len(triplets) <= train_v8.ACTIVE_VAL_SIZE:
        raise RuntimeError(
            f"Need more than {train_v8.ACTIVE_VAL_SIZE} triplets for the configured validation split"
        )
    _, val_triplets = train_test_split(
        triplets,
        test_size=train_v8.ACTIVE_VAL_SIZE,
        random_state=train_v8.USER_RANDOM_SEED,
    )
    if not 0 <= val_index < len(val_triplets):
        raise IndexError(f"val-index must be in [0,{len(val_triplets) - 1}]")
    triplet = val_triplets[val_index]
    item = train_v8.prepare_item(triplet, dataset_dir, face_root, color_root)
    if item is None:
        raise RuntimeError(f"Could not load cached data for triplet={triplet}")
    batch = tuple(tensor.unsqueeze(0) for tensor in item) + (
        [train_v8.triplet_cache_key(triplet)],
        torch.tensor([float("nan")]),
        torch.tensor([0.0]),
    )
    prepared = trainer.prepare_batch(batch)
    if prepared is None:
        raise RuntimeError(f"No valid target/reference hair pixels for triplet={triplet}")
    return triplet, prepared


def render_tail(trainer, prepared, blend_tail):
    return trainer.render_blend_tail(prepared, blend_tail)


@torch.no_grad()
def run_case(
    *,
    model: torch.nn.Module,
    trainer: train_v8.BlendingTrainerV8,
    val_index: int,
    dataset_dir: Path,
    face_root: Path,
    color_root: Path,
    output_dir: Path,
    correction_enabled: bool,
    run_label: str,
):
    triplet, prepared = prepare_validation_case(
        trainer,
        val_index,
        dataset_dir,
        face_root,
        color_root,
    )
    blend_tail, encoder_aux = trainer.run_adapter(
        prepared,
        correction_enabled=correction_enabled,
    )
    full_direct_tail, _ = trainer.run_adapter(
        prepared,
        correction_enabled=False,
        layer_mix_override=1.0,
    )
    fixed_direct_tail, _ = trainer.run_adapter(
        prepared,
        correction_enabled=False,
        layer_mix_override=train_v8.USER_DIAGNOSTIC_ALPHA,
    )
    anchor_tail, anchor_aux = trainer.run_adapter(prepared, correction_enabled=False)
    generated_256 = render_tail(trainer, prepared, blend_tail)
    full_direct_256 = render_tail(trainer, prepared, full_direct_tail)
    fixed_direct_256 = render_tail(trainer, prepared, fixed_direct_tail)
    anchor_256 = render_tail(trainer, prepared, anchor_tail)

    generated_lab = train_v8.rgb_to_lab(generated_256)
    pseudo_lab = prepared["pseudo_lab"]
    hair_mask = prepared["color_transfer_mask"]
    luma_excess = generated_lab[:, 0:1] - pseudo_lab[:, 0:1]
    gen_ab = generated_lab[:, 1:3]
    pseudo_ab = pseudo_lab[:, 1:3]
    hue_cosine = (
        (gen_ab * pseudo_ab).sum(dim=1, keepdim=True)
        / (
            torch.linalg.vector_norm(gen_ab, dim=1, keepdim=True)
            * torch.linalg.vector_norm(pseudo_ab, dim=1, keepdim=True)
        ).clamp_min(1e-4)
    ).clamp(-1, 1)

    metrics = {
        "run_label": run_label,
        "correction_enabled": correction_enabled,
        "val_index": val_index,
        "triplet": list(triplet),
        "ref_base_ab_distance": scalar(prepared["condition_metrics"]["ref_base_ab_distance"]),
        "delta_L_global": scalar(prepared["condition_metrics"]["delta_l_global"]),
        "chroma_need_gate": scalar(prepared["chroma_need_gate"]),
        "lightness_need_gate": scalar(prepared["lightness_need_gate"]),
        "edit_need_gate": scalar(prepared["edit_need_gate"]),
        "safe_fraction": scalar(prepared["condition_metrics"]["safe_fraction"]),
        "rejected_fraction": scalar(prepared["condition_metrics"]["rejected_fraction"]),
        "result_to_pseudo_ab_l2": scalar(trainer.masked_mean_value(
            torch.linalg.vector_norm(gen_ab - pseudo_ab, dim=1, keepdim=True), hair_mask
        )),
        "result_hue_error": scalar(trainer.masked_mean_value(
            torch.rad2deg(torch.acos(hue_cosine)), hair_mask
        )),
        "mean_L_excess": scalar(trainer.masked_mean_value(torch.relu(luma_excess), hair_mask)),
        "q95_L_excess": scalar(trainer.masked_q95(luma_excess, hair_mask)),
        "frac_L_excess_gt8": scalar(trainer.masked_fraction_above(luma_excess, hair_mask, 8.0)),
        "frac_L_excess_gt12": scalar(trainer.masked_fraction_above(luma_excess, hair_mask, 12.0)),
        "frac_L_excess_gt16": scalar(trainer.masked_fraction_above(luma_excess, hair_mask, 16.0)),
    }
    for key in (
        "direct_delta_norm",
        "direct_component_norm",
        "direct_mix_fraction",
        "layer_mix_mean",
        "layer_mix_min",
        "layer_mix_max",
        "correction_raw_norm",
        "correction_norm",
        "correction_budget",
        "correction_budget_scale",
        "correction_to_direct_ratio",
        "total_delta_norm",
        "total_to_direct_ratio",
        "direct_parallel_correction_coeff",
        "negative_parallel_fraction",
        "correction_chroma_budget",
        "correction_luma_budget",
    ):
        metrics[key] = scalar(encoder_aux[key])

    case_dir = output_dir / f"val_{val_index:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    excess_preview = ((torch.relu(luma_excess) / 20.0).clamp(0, 1) * 2.0 - 1.0).repeat(1, 3, 1, 1)
    train_v8.save_preview(
        case_dir / "diagnostic.png",
        [
            prepared["face_i"],
            prepared["color_i"],
            prepared["base_i"],
            generated_256,
            train_v8.mask_to_preview(prepared["color_transfer_mask"]),
            train_v8.mask_to_preview(prepared["reference_hair_mask"]),
            train_v8.mask_to_preview(prepared["safe_ref_mask"]),
            train_v8.mask_to_preview(prepared["rejected_highlight_mask"]),
            prepared["pseudo_rgb"],
            excess_preview,
        ],
    )
    train_v8.save_preview(
        case_dir / "direct_diagnostic.png",
        [
            prepared["base_i"],
            prepared["color_i"],
            full_direct_256,
            fixed_direct_256,
            anchor_256,
            generated_256,
            prepared["pseudo_rgb"],
        ],
    )
    with open(case_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics


@torch.no_grad()
def main():
    args = build_parser().parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Cannot find checkpoint: {args.checkpoint}")
    device = torch.device(train_v8.USER_DEVICE if torch.cuda.is_available() else "cpu")
    model = create_untrained_model(device)
    load_v4_checkpoint(model, args.checkpoint, device)
    trainer, _ = create_trainer(model, device)
    metrics = run_case(
        model=model,
        trainer=trainer,
        val_index=args.val_index,
        dataset_dir=args.dataset_dir,
        face_root=args.face_root,
        color_root=args.color_root,
        output_dir=args.output_dir,
        correction_enabled=True,
        run_label=f"checkpoint:{args.checkpoint}",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
