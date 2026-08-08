from scripts.pp_train_v5 import sample_bucket


def test_positive_with_distractor_is_not_pure_hard_negative():
    labels = {
        "has_positive": True, "has_candidate": True, "has_hard_negative": True,
        "is_high_quality_thin": False, "is_solid_hollow": False,
    }
    assert sample_bucket(labels) == "positive_with_distractor"

