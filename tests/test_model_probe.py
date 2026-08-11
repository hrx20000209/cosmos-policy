import pytest
import torch

from cosmos_policy.runtime.model_probe import (
    SpatialTokenReducer,
    TokenProbeConfig,
    patch_hidden_slots,
    replace_latent_slots,
    summarize_pooled_features,
)


def test_spatial_token_reducer_keeps_only_slot_vectors():
    hidden = torch.arange(2 * 4 * 3 * 3 * 5, dtype=torch.float32).reshape(2, 4, 3, 3, 5)
    reducer = SpatialTokenReducer(TokenProbeConfig(slot_indices=(1, 3)))

    pooled = reducer(hidden, block_id=7)

    assert pooled.shape == (2, 2, 5)
    torch.testing.assert_close(pooled, hidden.mean(dim=(2, 3)).index_select(1, torch.tensor([1, 3])))
    assert pooled.requires_grad is False


def test_spatial_token_reducer_rejects_bad_slot():
    hidden = torch.zeros(1, 2, 1, 1, 3)
    with pytest.raises(IndexError):
        SpatialTokenReducer(TokenProbeConfig(slot_indices=(2,)))(hidden, block_id=0)


def test_feature_summary_is_compact_and_tracks_block_order():
    features = [torch.ones(1, 3, 4), torch.ones(1, 3, 4) * 2]

    summary = summarize_pooled_features(features, block_ids=(2, 6))

    assert summary["block_ids"] == [2, 6]
    assert summary["slot_norm"].shape == (1, 2, 3)
    assert summary["cosine_distance"].shape == (1, 2, 3)
    torch.testing.assert_close(summary["cosine_distance"], torch.zeros(1, 2, 3))


def test_replace_latent_slots_changes_only_requested_temporal_slots():
    base = torch.zeros(1, 2, 9, 1, 1)
    source = torch.arange(18, dtype=torch.float32).reshape(1, 2, 9, 1, 1)

    result = replace_latent_slots(base, source, {2: 6, 3: 7})

    assert torch.equal(result[:, :, 2], source[:, :, 6])
    assert torch.equal(result[:, :, 3], source[:, :, 7])
    assert torch.count_nonzero(result[:, :, [0, 1, 4, 5, 6, 7, 8]]) == 0
    assert torch.count_nonzero(base) == 0


def test_patch_hidden_slots_builds_one_oracle_branch_per_group():
    speculative = torch.zeros(1, 9, 2, 2, 3)
    fresh = torch.ones_like(speculative)

    patched = patch_hidden_slots(speculative, fresh, [(2, 3), (1,), (4,)])

    assert patched.shape == (3, 9, 2, 2, 3)
    assert torch.all(patched[0, 2:4] == 1)
    assert torch.count_nonzero(patched[0, [0, 1, 4, 5, 6, 7, 8]]) == 0
    assert torch.all(patched[1, 1] == 1)
    assert torch.all(patched[2, 4] == 1)
