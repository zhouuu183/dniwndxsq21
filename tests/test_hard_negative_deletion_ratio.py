import torch

from models.ear_modules_v5 import build_accepted_object_protection


def test_raw_and_effective_deletion_ratios_are_separate():
    accepted = torch.ones(1, 1, 4, 4)
    raw_negative = torch.zeros_like(accepted)
    raw_negative[:, :, :2] = 1
    result = build_accepted_object_protection(accepted, raw_negative)
    assert abs(float(result["hard_negative_deletion_ratio"][0]) - 0.5) < 1e-6
    assert float(result["effective_hard_negative_deletion_ratio"][0]) == 0.0

