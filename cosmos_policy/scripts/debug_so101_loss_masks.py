"""在 CPU 上逐项验证 SO101 三种显式 loss mask。"""

from __future__ import annotations

import torch

from cosmos_policy.models.policy_text2world_model import (
    build_so101_loss_mask,
    normalize_so101_masked_edm_loss,
)


def main() -> None:
    indices = {
        "action_indices": torch.tensor([2, 2]),
        "future_proprio_indices": torch.tensor([3, 3]),
        "future_wrist_image_indices": torch.tensor([4, 4]),
        "future_wrist_image2_indices": torch.tensor([5, -1]),
        "future_image_indices": torch.tensor([6, 6]),
        "future_image2_indices": torch.tensor([7, 7]),
        "value_indices": torch.tensor([8, 8]),
    }
    expected = {
        "action_only": ({2}, {2}),
        "future_state_only": ({3, 4, 5, 6, 7}, {3, 4, 6, 7}),
        "joint_action_future_state": ({2, 3, 4, 5, 6, 7}, {2, 3, 4, 6, 7}),
    }
    for mode, expected_per_sample in expected.items():
        mask = build_so101_loss_mask(2, 11, mode, False, **indices)
        actual = tuple(set(torch.nonzero(row, as_tuple=False).flatten().tolist()) for row in mask)
        assert actual == expected_per_sample, (mode, actual, expected_per_sample)
        print(f"{mode}: sample0={sorted(actual[0])}, sample1={sorted(actual[1])}")

    with_value = build_so101_loss_mask(2, 11, "action_only", True, **indices)
    assert torch.equal(torch.nonzero(with_value[0]).flatten(), torch.tensor([2, 8]))
    print("include_value=True: value slot 8 验证通过")

    loss = torch.arange(1, 1 + 2 * 3 * 11 * 2 * 2, dtype=torch.float32).reshape(2, 3, 11, 2, 2)
    mask = build_so101_loss_mask(2, 11, "action_only", False, **indices)
    normalized = normalize_so101_masked_edm_loss(loss, mask)
    selected = torch.cat([loss[0, :, 2].flatten(), loss[1, :, 2].flatten()]).mean()
    assert torch.allclose(normalized, selected), (normalized, selected)
    print(f"normalize_so101_masked_loss=True: selected-slot mean={normalized.item():.6f} 验证通过")


if __name__ == "__main__":
    main()
