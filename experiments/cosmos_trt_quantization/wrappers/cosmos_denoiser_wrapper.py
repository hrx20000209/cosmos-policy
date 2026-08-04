"""Fixed-signature wrapper around the real Cosmos Policy ``model.net``.

This deliberately wraps one DiT denoising forward only.  It excludes LIBERO,
tokenization, VAE, scheduler updates, action reconstruction, and file I/O.
The signature is based on ``MinimalV1LVGDiT.forward`` and inputs captured from
the verified policy path, rather than an invented generic diffusion API.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from cosmos_policy._src.predict2.conditioner import DataType


class CosmosDenoiserWrapper(nn.Module):
    """Single fixed-shape Cosmos DiT denoising step for LIBERO inference."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.net(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            data_type=DataType.VIDEO,
            intermediate_feature_ids=None,
            img_context_emb=None,
        )


def fixture_args(fixture: dict, device: str | torch.device = "cuda") -> tuple[torch.Tensor, ...]:
    """Return wrapper positional arguments from a captured tensor-only fixture."""
    keys = [
        "x_B_C_T_H_W",
        "timesteps_B_T",
        "crossattn_emb",
        "condition_video_input_mask_B_C_T_H_W",
    ]
    args = [fixture[key].to(device=device) for key in keys]
    for optional in ("fps", "padding_mask"):
        value = fixture.get(optional)
        if isinstance(value, torch.Tensor):
            args.append(value.to(device=device))
        else:
            args.append(None)
    return tuple(args)


def tensor_metadata(fixture: dict) -> dict:
    """JSON-serializable shape/dtype audit."""
    result = {}
    for key, value in fixture.items():
        if isinstance(value, torch.Tensor):
            result[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": value.numel(),
                "finite": bool(torch.isfinite(value.float()).all()),
            }
    return result

