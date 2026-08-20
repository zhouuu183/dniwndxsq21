from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    train = (ROOT / "scripts" / "blending_train_v8.py").read_text(encoding="utf-8")
    inference = (ROOT / "models" / "Blending_v8.py").read_text(encoding="utf-8")
    projector = (ROOT / "models" / "selective_color_projector_v825.py").read_text(encoding="utf-8")
    assert "FULL_COLOR_ARCH_V8_8" in train and "run_v225_pretrain_diagnostic" in train
    assert "target_ref_ab" in train and "target_ref_ab" in inference
    assert "reference_conditioned_boundary_target_v825" in projector
    assert "torch.maximum(base_l, pseudo_lab" not in projector
    assert "hair_support * candidate_rgb" not in projector
    assert "ReferenceConditionedBoundaryProjectorV825" in inference
    print("test_v225_training_contract: PASS")


if __name__ == "__main__":
    main()
