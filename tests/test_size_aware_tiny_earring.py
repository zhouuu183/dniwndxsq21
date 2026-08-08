from models.ear_modules_v5 import ComponentObjectGraph
from tests.v3c_test_utils import proposal_data


def test_one_pixel_tiny_earring_survives_with_one_strong_cue_and_anchor():
    data = proposal_data(24)
    data["proposal_probability_1024"][0, 0, 10, 10] = 1
    data["shape_evidence_1024"][0, 0, 10, 10] = 1
    data["shape_strong_1024"][0, 0, 10, 10] = 1
    data["earlobe_anchor_1024"][0, 0, 10, 10] = 1
    record = ComponentObjectGraph()._record(0, [(10, 10)], "left", data)
    assert record["scale_class"] == "tiny"
    assert record["status"] in ("trusted", "candidate")

