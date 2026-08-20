"""Static contract checks for the deterministic V2.24 training path."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.append(str(ROOT))

from models.boundary_masks_v824 import build_boundary_masks_v824


def main() -> None:
    train_source = (ROOT / "scripts" / "blending_train_v8.py").read_text(encoding="utf-8")
    projector_source = (ROOT / "models" / "selective_color_projector_v824.py").read_text(encoding="utf-8")
    inference_source = (ROOT / "models" / "Blending_v8.py").read_text(encoding="utf-8")
    assert "FULL_COLOR_ARCH_V8_7" in train_source
    assert "build_boundary_masks_v824" in train_source
    assert "hair_edge * hair_support" not in projector_source
    assert "hair_support * candidate_rgb" not in projector_source
    assert "BoundaryStableFullColorProjectorV824" in inference_source
    assert "build_boundary_masks_v824" in inference_source
    assert "boundary_single_alpha" in inference_source
    assert "global_alpha_trainable" not in projector_source

    import torch

    raw = torch.ones(1, 1, 4, 4)
    eroded = torch.zeros_like(raw)
    train_masks = build_boundary_masks_v824(
        target_hair_mask=raw, target_hair_eroded=eroded
    )
    inference_masks = build_boundary_masks_v824(
        target_hair_mask=raw, target_hair_eroded=eroded
    )
    assert max(
        (train_masks[key] - inference_masks[key]).abs().max().item()
        for key in train_masks
    ) <= 1e-6
    print("test_v224_training_contract: PASS")


if __name__ == "__main__":
    main()
