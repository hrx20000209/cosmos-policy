"""Small, inference-only probes for Cosmos policy internals.

The policy DiT exposes intermediate block features as ``[B, T, H, W, D]``.
Keeping that tensor for several blocks is prohibitively expensive on the robot,
so this module reduces it on-device to one spatially pooled vector per latent
slot.  The probe is deliberately independent of the policy model and is
disabled unless a caller explicitly installs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

LIBERO_SLOT_NAMES = (
    "leading_placeholder",
    "current_proprio",
    "current_wrist",
    "current_primary",
    "action",
    "future_proprio",
    "future_wrist",
    "future_primary",
    "value_excluded",
)

LIBERO_PATCH_GROUPS: Mapping[str, tuple[int, ...]] = {
    "current_visual": (2, 3),
    "visual_plus_proprio": (1, 2, 3),
    "action_only": (4,),
    "visual_plus_action": (2, 3, 4),
    # Slot 0 is a structural temporal-VAE placeholder.  Including it makes
    # this the literal all-current-observation patch and also tests whether it
    # is behaviorally equivalent to visual_plus_proprio on LIBERO.
    "all_current_observation": (0, 1, 2, 3),
}


@dataclass(frozen=True)
class TokenProbeConfig:
    """Which latent slots should be retained by a :class:`SpatialTokenReducer`."""

    slot_indices: tuple[int, ...] | None = None

    @classmethod
    def all_slots(cls) -> "TokenProbeConfig":
        return cls(slot_indices=None)


class SpatialTokenReducer:
    """Pool spatial patch tokens while they are still on the accelerator."""

    def __init__(self, config: TokenProbeConfig | None = None):
        self.config = config or TokenProbeConfig.all_slots()

    def __call__(self, hidden: torch.Tensor, block_id: int) -> torch.Tensor:
        del block_id  # The block id is carried by the list position in the model output.
        if hidden.ndim != 5:
            raise ValueError(f"expected [B,T,H,W,D] hidden state, got {tuple(hidden.shape)}")
        pooled = hidden.detach().float().mean(dim=(2, 3))
        if self.config.slot_indices is None:
            return pooled
        slots = torch.as_tensor(self.config.slot_indices, device=pooled.device)
        if torch.any(slots < 0) or torch.any(slots >= pooled.shape[1]):
            raise IndexError(f"slot index out of range for T={pooled.shape[1]}: {self.config.slot_indices}")
        return pooled.index_select(1, slots)


class FullHiddenCapture:
    """Copy a full block activation to CPU for an offline oracle intervention."""

    def __call__(self, hidden: torch.Tensor, block_id: int) -> torch.Tensor:
        del block_id
        if hidden.ndim != 5:
            raise ValueError(f"expected [B,T,H,W,D] hidden state, got {tuple(hidden.shape)}")
        if hidden.shape[0] != 1:
            raise ValueError(f"full hidden capture expects batch size one, got {hidden.shape[0]}")
        return hidden.detach().to(device="cpu", copy=True)


def replace_latent_slots(
    base: torch.Tensor,
    source: torch.Tensor,
    replacements: Mapping[int, int],
) -> torch.Tensor:
    """Return ``base`` with selected temporal slots copied from ``source``.

    The helper keeps the untouched conditional slots identical, which is
    important for causal input interventions: a predicted-wrist condition
    must not accidentally replace the proprio or primary-camera condition.
    """

    if base.ndim != 5 or source.ndim != 5:
        raise ValueError("expected latent tensors shaped [B,C,T,H,W]")
    if base.shape != source.shape:
        raise ValueError(f"latent shape mismatch: {tuple(base.shape)} != {tuple(source.shape)}")
    result = base.detach().clone()
    temporal_size = result.shape[2]
    for destination, origin in replacements.items():
        if not 0 <= destination < temporal_size or not 0 <= origin < temporal_size:
            raise IndexError(f"latent slot mapping {destination}<-{origin} is invalid for T={temporal_size}")
        result[:, :, destination] = source[:, :, origin]
    return result


def patch_hidden_slots(
    speculative: torch.Tensor,
    fresh: torch.Tensor,
    slot_groups: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Build a batch of oracle activation patches from one paired state.

    ``fresh`` and ``speculative`` are full DiT hidden states with shape
    ``[1,T,H,W,D]``.  The returned batch has one item per slot group and is
    suitable for running the remaining DiT blocks in parallel.
    """

    if speculative.shape != fresh.shape or speculative.ndim != 5:
        raise ValueError(
            f"expected matching [1,T,H,W,D] tensors, got {tuple(speculative.shape)} and {tuple(fresh.shape)}"
        )
    if speculative.shape[0] != 1:
        raise ValueError(f"patch_hidden_slots expects one paired state, got batch={speculative.shape[0]}")
    patched = speculative.expand(len(slot_groups), *speculative.shape[1:]).clone()
    for batch_index, slots in enumerate(slot_groups):
        for slot in slots:
            if not 0 <= int(slot) < speculative.shape[1]:
                raise IndexError(f"hidden slot {slot} is invalid for T={speculative.shape[1]}")
            patched[batch_index, int(slot)] = fresh[0, int(slot)]
    return patched


def summarize_pooled_features(
    features: Sequence[torch.Tensor],
    block_ids: Iterable[int] | None = None,
) -> dict[str, torch.Tensor | list[int]]:
    """Return compact norms and adjacent-block cosine distances.

    ``features`` must contain pooled tensors shaped ``[B, T, D]``.  The result
    stays on the source device so a runtime scheduler can consume it without a
    synchronizing device-to-host copy.  It is intended for diagnostics, not
    as a learned policy input.
    """

    if not features:
        return {"block_ids": [], "slot_norm": torch.empty(0), "cosine_distance": torch.empty(0)}
    if any(feature.ndim != 3 for feature in features):
        shapes = [tuple(feature.shape) for feature in features]
        raise ValueError(f"expected pooled [B,T,D] features, got {shapes}")

    stacked = torch.stack([feature.detach().float() for feature in features], dim=1)
    norms = torch.linalg.vector_norm(stacked, dim=-1)
    if stacked.shape[1] == 1:
        cosine_distance = torch.zeros_like(norms)
    else:
        current = torch.nn.functional.normalize(stacked[:, 1:], dim=-1)
        previous = torch.nn.functional.normalize(stacked[:, :-1], dim=-1)
        adjacent = 1.0 - (current * previous).sum(dim=-1)
        cosine_distance = torch.cat([torch.zeros_like(adjacent[:, :1]), adjacent], dim=1)

    ids = list(block_ids) if block_ids is not None else list(range(len(features)))
    if len(ids) != len(features):
        raise ValueError(f"block_ids length {len(ids)} does not match features length {len(features)}")
    return {"block_ids": ids, "slot_norm": norms, "cosine_distance": cosine_distance}
