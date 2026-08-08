import torch

from models.ear_modules_v5 import build_accepted_object_protection, build_hierarchical_residual_gate


def test_ordinary_negative_cannot_delete_accepted_object():
    accepted = torch.zeros(1, 1, 16, 16)
    accepted[:, :, 6:10, 6:10] = 1
    raw_negative = accepted.clone()
    result = build_accepted_object_protection(accepted, raw_negative)
    assert torch.equal(result["accepted_object_mask_after_negative_1024"], accepted)
    assert float(result["effective_hard_negative_deletion_ratio"][0]) == 0.0


def test_hair_core_and_hollow_remain_hard_protections():
    ones = torch.ones(1, 1, 4, 4)
    zeros = torch.zeros_like(ones)
    hair_blocked = build_hierarchical_residual_gate(
        "accepted_support", ones, zeros, zeros, ones, ones, ones, ones, zeros, zeros
    )
    hollow_blocked = build_hierarchical_residual_gate(
        "accepted_support", ones, zeros, zeros, ones, ones, ones, zeros, zeros, ones
    )
    assert float(hair_blocked["residual_gate"].sum()) == 0.0
    assert float(hollow_blocked["residual_gate"].sum()) == 0.0
