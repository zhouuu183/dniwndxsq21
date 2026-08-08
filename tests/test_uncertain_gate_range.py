import torch

from models.ear_modules_v5 import EAR_NEGATIVE, EAR_POSITIVE, EAR_UNCERTAIN, build_status_gate


def test_uncertain_gate_uses_configured_range():
    status = torch.tensor([EAR_NEGATIVE, EAR_UNCERTAIN, EAR_UNCERTAIN, EAR_POSITIVE])
    gate = build_status_gate(status, torch.tensor([1.0, 0.0, 1.0, 0.0])).flatten()
    assert torch.allclose(gate, torch.tensor([0.0, 0.35, 0.85, 1.0]))

