import torch
import torch.nn as nn

from models.ear_modules_v5 import (
    EAR_NEGATIVE,
    EAR_POSITIVE,
    EAR_UNCERTAIN,
    PollutionSafeEarringEstimator,
    resolve_object_level_status,
)


def test_status_comes_from_accepted_object_quality():
    assert resolve_object_level_status(1, 0, 0.1) == EAR_POSITIVE
    assert resolve_object_level_status(0, 1, 0.8) == EAR_POSITIVE
    assert resolve_object_level_status(0, 1, 0.3) == EAR_UNCERTAIN
    assert resolve_object_level_status(0, 0, 1.0) == EAR_NEGATIVE


class _StaticModule(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, *args, **kwargs):
        return dict(self.output)


def test_online_estimator_preserves_component_object_graph_outputs():
    reference = torch.zeros(1, 1, 8, 8)
    raw_component = torch.ones_like(reference)
    object_group = torch.full_like(reference, 0.5)
    estimator = PollutionSafeEarringEstimator()
    estimator.roi_builder = _StaticModule({"parser_seed_1024": reference})
    estimator.proposal_builder = _StaticModule({
        "parser_seed_1024": reference,
        "proposal_probability_1024": reference,
    })
    estimator.refiner = _StaticModule({
        "raw_component_mask_1024": raw_component,
        "object_group_mask_1024": object_group,
        "trusted_material_core_1024": reference,
        "candidate_material_core_1024": reference,
        "raw_hard_negative_mask_1024": reference,
        "trusted_object_count": torch.tensor([0]),
        "candidate_object_count": torch.tensor([0]),
        "component_quality": torch.tensor([0.0]),
    })

    output = estimator(torch.zeros(1, 3, 8, 8), reference)

    assert output["raw_component_mask_1024"] is raw_component
    assert output["object_group_mask_1024"] is object_group
