import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.blending_train_v8 as train_v8
import scripts.validate_v8_color_direction_diagnostic as diagnostic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the untrained V8.4 direct color path")
    parser.add_argument("--val-indices", type=int, nargs="+", default=[4, 20])
    parser.add_argument("--dataset-dir", type=Path, default=train_v8.ACTIVE_DATASET_DIR)
    parser.add_argument("--face-root", type=Path, default=train_v8.ACTIVE_FACE_ROOT)
    parser.add_argument("--color-root", type=Path, default=train_v8.ACTIVE_COLOR_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=train_v8.ACTIVE_OUTPUT_DIR / "val_images" / "epoch_000_pretrain",
    )
    return parser


@torch.no_grad()
def main():
    args = build_parser().parse_args()
    train_v8.set_seed(train_v8.USER_RANDOM_SEED)
    device = torch.device(train_v8.USER_DEVICE if torch.cuda.is_available() else "cpu")
    model = diagnostic.create_untrained_model(device)
    model.set_correction_trainable(False)
    trainer, _ = diagnostic.create_trainer(model, device)
    records = []
    for val_index in args.val_indices:
        records.append(diagnostic.run_case(
            model=model,
            trainer=trainer,
            val_index=val_index,
            dataset_dir=args.dataset_dir,
            face_root=args.face_root,
            color_root=args.color_root,
            output_dir=args.output_dir,
            correction_enabled=False,
            run_label="untrained_direct_color_anchor_v8_4",
        ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "pretrain_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
