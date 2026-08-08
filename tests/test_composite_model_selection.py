from scripts.pp_train_v5 import composite_model_score


def test_composite_score_rewards_recall_without_ignoring_pollution():
    good = {
        "trusted_restore_error": 0.1, "candidate_restore_error": 0.1,
        "thin_detail_error": 0.1, "source_chroma_error": 0.1,
        "support_completeness": 0.9, "hard_negative_change": 0.01,
        "outside_change": 0.0, "hair_core_change": 0.0, "negative_hallucination": 0.0,
    }
    polluted = dict(good, hard_negative_change=0.8, outside_change=0.5)
    no_recall = dict(good, trusted_restore_error=0.9, candidate_restore_error=0.9, support_completeness=0.0)
    assert composite_model_score(good) > composite_model_score(polluted)
    assert composite_model_score(good) > composite_model_score(no_recall)

