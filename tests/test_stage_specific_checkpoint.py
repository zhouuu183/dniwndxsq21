import argparse

import torch

from models.ear_modules_v5 import MorphologyAwareDualBranchRenderer
from models.postprocess_v5 import load_checkpoint_compat
from scripts.pp_train_v5 import NullLogger, STAGE_CHECKPOINT_NAMES, Trainer


def test_each_stage_checkpoint_has_independent_name_and_contract(tmp_path):
    model = MorphologyAwareDualBranchRenderer(crop_size=16, large_crop_size=16, base_channels=4)
    args = argparse.Namespace(output=tmp_path)
    logger = NullLogger(tmp_path)
    logger.start_logging()
    trainer = Trainer(args, model, [], [], [], logger, "hash")
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics = {
        "trusted_restore_error": 0.2, "candidate_restore_error": 0.2,
        "thin_detail_error": 0.2, "source_chroma_error": 0.2,
        "support_completeness": 0.8, "hard_negative_change": 0.0,
        "outside_change": 0.0, "hair_core_change": 0.0, "negative_hallucination": 0.0,
    }
    for stage in (0, 1):
        trainer.stage_best_metrics[stage] = {"stage_selection_score": float(stage)}
        trainer.save_checkpoint(STAGE_CHECKPOINT_NAMES[stage], stage, 1, metrics)
    first = load_checkpoint_compat(str(tmp_path / "best_stage0_restore.pth"), map_location="cpu")
    second = load_checkpoint_compat(str(tmp_path / "best_stage1_candidate.pth"), map_location="cpu")
    assert first["stage_index"] == 0 and second["stage_index"] == 1
    assert "stage_best_metrics" in second and "overall_score" in second
