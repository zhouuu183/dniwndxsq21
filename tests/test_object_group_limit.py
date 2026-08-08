from models.ear_modules_v5 import ComponentObjectGraph
from tests.v3c_test_utils import proposal_data, roi_data


def test_limit_applies_to_object_groups_not_fragments():
    data = proposal_data(96)
    for y, x in ((10, 8), (30, 8), (50, 8), (70, 8)):
        data["combined_weak_candidate_1024"][0, 0, y:y + 2, x:x + 2] = 1
        data["proposal_probability_1024"][0, 0, y:y + 2, x:x + 2] = 1
        data["shape_evidence_1024"][0, 0, y:y + 2, x:x + 2] = 1
        data["shape_strong_1024"][0, 0, y:y + 2, x:x + 2] = 1
        data["earlobe_anchor_1024"][0, 0, y:y + 2, x:x + 2] = 1
    result = ComponentObjectGraph()(data, roi_data(96), "strict")
    assert int(result["object_group_count"][0]) == 3
    assert int(result["fragment_count"][0]) == 4

