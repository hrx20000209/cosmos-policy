#!/usr/bin/env python3
"""Capture the first real DiT call made by deterministic Cosmos LIBERO inference."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
import ctypes

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cosmos_trt_quantization"
OLD_SCRIPTS = ROOT / "experiments/liberoplus_quantization/scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OLD_SCRIPTS))
sys.path.insert(0, str(EXP))

from quant_microbench import CKPT_DIR, Cfg, get_fixed_observation  # noqa: E402
from wrappers.cosmos_denoiser_wrapper import (  # noqa: E402
    CosmosDenoiserWrapper,
    fixture_args,
    tensor_metadata,
)

# LIBERO-Plus imports Wand after CUDA/TransformerEngine.  Preload its private
# userspace ImageMagick build before those libraries can alter loader order.
_magick_wand = "/data/rxhuang/envs/imagemagick/lib/libMagickWand.so"
if Path(_magick_wand).exists():
    ctypes.CDLL(_magick_wand, mode=ctypes.RTLD_GLOBAL)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(af, bf, dim=0))


def main() -> None:
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        load_dataset_stats,
    )
    from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size

    out_path = EXP / "calibration/fixed_denoiser_inputs.pt"
    summary_path = EXP / "summaries/denoiser_wrapper_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(195)
    np.random.seed(195)
    cfg = Cfg(num_denoising_steps_action=5)
    dataset_stats = load_dataset_stats(f"{CKPT_DIR}/libero_dataset_statistics.json")
    observation, task_label = get_fixed_observation(cfg, get_image_resize_size(cfg.model_family))
    with open(f"{CKPT_DIR}/libero_t5_embeddings.pkl", "rb") as handle:
        cached_embeddings = pickle.load(handle)
    prefix_matches = [key for key in cached_embeddings if task_label.startswith(key)]
    embedding_key = task_label if task_label in cached_embeddings else (
        max(prefix_matches, key=len) if prefix_matches else None
    )
    if embedding_key is None:
        raise KeyError(f"No cached base instruction matches {task_label!r}")
    task_embedding = cached_embeddings[embedding_key]
    if isinstance(task_embedding, torch.Tensor):
        task_embedding = task_embedding.detach().float().cpu().numpy()

    model, _ = get_model(cfg)
    model.eval()
    captured: dict[str, object] = {}
    selected = {
        "x_B_C_T_H_W",
        "timesteps_B_T",
        "crossattn_emb",
        "condition_video_input_mask_B_C_T_H_W",
        "fps",
        "padding_mask",
    }

    def pre_hook(_module, args, kwargs):
        if captured:
            return
        if args:
            raise RuntimeError("Unexpected positional MinimalV1LVGDiT inputs; wrapper assumptions invalid")
        for key in selected:
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                captured[key] = value.detach().cpu().clone()
            elif value is not None:
                captured[f"{key}__non_tensor_repr"] = repr(value)

    handle = model.net.register_forward_pre_hook(pre_hook, with_kwargs=True)
    policy_output = get_action(
        cfg,
        model,
        dataset_stats,
        observation,
        task_embedding,
        seed=195,
        num_denoising_steps_action=5,
        generate_future_state_and_value_in_parallel=False,
        batch_size=1,
    )
    handle.remove()
    torch.cuda.synchronize()
    required = {
        "x_B_C_T_H_W",
        "timesteps_B_T",
        "crossattn_emb",
        "condition_video_input_mask_B_C_T_H_W",
    }
    missing = sorted(required - captured.keys())
    if missing:
        raise RuntimeError(f"Did not capture required network inputs: {missing}; got {sorted(captured)}")

    fixture = {key: value for key, value in captured.items() if isinstance(value, torch.Tensor)}
    torch.save(fixture, out_path)
    args = fixture_args(fixture)
    wrapper = CosmosDenoiserWrapper(model.net).eval()
    with torch.inference_mode():
        direct = model.net(
            x_B_C_T_H_W=args[0],
            timesteps_B_T=args[1],
            crossattn_emb=args[2],
            condition_video_input_mask_B_C_T_H_W=args[3],
            fps=args[4],
            padding_mask=args[5],
        )
        wrapped = wrapper(*args)
        repeated = wrapper(*args)
    torch.cuda.synchronize()
    diff = wrapped.float() - direct.float()
    repeat_diff = repeated.float() - wrapped.float()
    action = policy_output["actions"] if isinstance(policy_output, dict) else policy_output
    result = {
        "fixture": str(out_path),
        "task_label": task_label,
        "cached_embedding_key": embedding_key,
        "seed": 195,
        "batch_size": 1,
        "chunk_size": cfg.chunk_size,
        "denoising_steps": 5,
        "inputs": tensor_metadata(fixture),
        "output": {
            "shape": list(wrapped.shape),
            "dtype": str(wrapped.dtype),
            "finite": bool(torch.isfinite(wrapped.float()).all()),
            "cosine_vs_direct": cosine(wrapped, direct),
            "max_abs_vs_direct": float(diff.abs().max()),
            "l2_vs_direct": float(torch.linalg.vector_norm(diff)),
            "repeat_max_abs": float(repeat_diff.abs().max()),
        },
        "policy_action_shape": list(np.asarray(action).shape),
    }
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["output"]["cosine_vs_direct"] <= 0.99999:
        raise AssertionError("Wrapper cosine gate failed")
    if not result["output"]["finite"] or result["output"]["repeat_max_abs"] != 0:
        raise AssertionError("Wrapper finite/repeatability gate failed")


if __name__ == "__main__":
    main()
