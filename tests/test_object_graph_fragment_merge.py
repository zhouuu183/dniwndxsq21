import torch

from models.ear_modules_v5 import ComponentObjectGraph
from tests.v3c_test_utils import proposal_data, roi_data


def test_six_fragments_merge_into_one_object_group():
    data = proposal_data(64)
    xs = [8, 12, 16, 20, 24, 28]
    data["proposal_probability_1024"][0, 0, 20, 8:29] = 1
    for x in xs:
        data["combined_weak_candidate_1024"][0, 0, 20, x] = 1
        data["shape_evidence_1024"][0, 0, 20, x] = 1
        data["shape_strong_1024"][0, 0, 20, x] = 1
    data["earlobe_anchor_1024"].fill_(1)
    result = ComponentObjectGraph()(data, roi_data(64), "strict")
    assert int(result["object_group_count"][0]) == 1
    assert result["fragments_per_group"][0] == [6]
    assert int(result["fragment_count"][0]) == 6

