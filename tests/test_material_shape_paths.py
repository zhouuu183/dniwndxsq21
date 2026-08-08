import torch

from models.ear_modules_v5 import ComponentObjectGraph
from tests.v3c_test_utils import proposal_data


def test_material_and_shape_are_independent_candidate_paths():
    graph = ComponentObjectGraph()
    coordinates = [(10, 10), (10, 11), (11, 10), (11, 11)]
    for path in ("material", "shape"):
        data = proposal_data(32)
        ys, xs = zip(*coordinates)
        data["proposal_probability_1024"][0, 0, ys, xs] = 1
        data["appearance_evidence_1024"][0, 0, ys, xs] = 0.8
        data["earlobe_anchor_1024"][0, 0, ys, xs] = 1
        data[path + "_evidence_1024"][0, 0, ys, xs] = 0.9
        data[path + "_strong_1024"][0, 0, ys, xs] = 1
        record = graph._record(0, coordinates, "left", data)
        assert path in record["candidate_paths"]
        assert record["status"] in ("trusted", "candidate")

