import numpy as np

from runtime.prediction_surprise import PendingPrediction, SurpriseAligner, latent_distances


def test_prediction_is_compared_only_at_aligned_checkpoint():
    aligner = SurpriseAligner()
    prediction = PendingPrediction("r", 16, np.zeros((2, 2)))
    aligner.add(prediction)
    assert aligner.pop_aligned(1) == []
    assert aligner.pop_aligned(16) == [prediction]


def test_latent_distances_zero_for_identical_values():
    metrics = latent_distances(np.ones((2, 3)), np.ones((2, 3)))
    assert metrics["latent_l1"] == 0.0
    assert metrics["latent_l2"] == 0.0
    assert abs(metrics["cosine_distance"]) < 1e-6

