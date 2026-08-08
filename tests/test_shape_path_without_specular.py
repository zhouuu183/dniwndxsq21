from models.ear_modules_v5 import ComponentObjectGraph
from tests.v3c_test_utils import proposal_data


def test_shape_path_accepts_low_chroma_non_specular_object():
    data = proposal_data(32)
    coordinates = [(8, 8), (8, 9), (9, 9), (10, 9)]
    ys, xs = zip(*coordinates)
    data["proposal_probability_1024"][0, 0, ys, xs] = 0.95
    data["shape_evidence_1024"][0, 0, ys, xs] = 0.95
    data["curve_evidence_1024"][0, 0, ys, xs] = 0.95
    data["shape_strong_1024"][0, 0, ys, xs] = 1
    data["earlobe_anchor_1024"][0, 0, ys, xs] = 1
    record = ComponentObjectGraph()._record(0, coordinates, "left", data)
    assert record["material_score"] == 0.0
    assert "shape" in record["candidate_paths"]
    assert record["status"] in ("trusted", "candidate")

