import pytest

from models.ear_modules_v5 import THO_LER_DATASET_VERSION, THO_LER_METHOD, validate_dataset_part
from tests.v3c_test_utils import dataset_item


def test_v3c_contract_accepts_v3c_and_rejects_v3b():
    part = {"version": THO_LER_DATASET_VERSION, "method": THO_LER_METHOD, "items": [dataset_item()]}
    assert len(validate_dataset_part(part)) == 1
    with pytest.raises(ValueError):
        validate_dataset_part({**part, "version": "tho_ler_v3b_morphology_pollution_safe"})


def test_v3c_contract_rejects_missing_required_field():
    item = dataset_item()
    del item["effective_hard_negative_mask_1024"]
    with pytest.raises(KeyError):
        validate_dataset_part({"version": THO_LER_DATASET_VERSION, "method": THO_LER_METHOD, "items": [item]})
