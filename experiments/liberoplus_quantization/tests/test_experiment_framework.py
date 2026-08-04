from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))

import execution_metrics as metrics
import quant_lib
from experiment_config import load_config
from liberoplus_utils import resolve_instruction


def test_config_inheritance_and_dynamic_override() -> None:
    cfg = load_config(EXPERIMENT / "configs/dynamic_replan_denoising.yaml")
    assert cfg.chunk_size == 16
    assert cfg.dynamic_replan.enabled
    assert cfg.dynamic_replan.dynamic_denoising
    assert cfg.dynamic_replan.candidate_open_loop_steps == [4, 8, 16]


def test_non_language_suffix_uses_longest_cached_prefix() -> None:
    instruction, method = resolve_instruction(
        "pick up the bowl and place it on the plate table 14",
        "Background Textures",
        ["pick up the bowl", "pick up the bowl and place it on the plate"],
    )
    assert instruction == "pick up the bowl and place it on the plate"
    assert method == "canonical_prefix"


def test_language_perturbation_never_falls_back() -> None:
    with pytest.raises(KeyError, match="genuine language perturbation"):
        resolve_instruction(
            "deposit the dark vessel on the flat dish",
            "Language Instructions",
            ["put the black bowl on the plate"],
        )


def test_chunk_metrics_use_generated_denominator() -> None:
    chunk = metrics.ChunkMetrics(
        chunk_index=0,
        generated_step=4,
        generated_time_s=1.0,
        generated_actions=16,
        planned_open_loop_steps=4,
        denoising_steps=5,
    )
    chunk.execute(current_step=4, current_time_s=1.1)
    chunk.execute(current_step=5, current_time_s=1.2)
    summary = metrics.summarize_chunks([chunk])
    assert summary["executed_actions"] == 2
    assert summary["discarded_actions"] == 14
    assert summary["stale_action_ratio"] == pytest.approx(14 / 16)
    assert summary["observation_staleness_steps_mean"] == 0.5


def test_action_boundary_metrics() -> None:
    chunk = metrics.ChunkMetrics(1, 0, 0.0, 16, 8, 5)
    metrics.set_boundary_metrics(chunk, np.zeros(3), np.array([1.0, -2.0, 2.0]))
    assert chunk.boundary_l1 == 5.0
    assert chunk.boundary_l2 == 3.0
    assert chunk.boundary_per_dimension == [1.0, 2.0, 2.0]


class DummyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(256, 256, bias=False)
        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(256, 512, bias=False)


class DummyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DummyBlock()])
        self.x_embedder = nn.Module()
        self.x_embedder.proj = nn.Sequential(nn.Identity(), nn.Linear(72, 2048, bias=False))
        self.final_layer = nn.Module()
        self.final_layer.wrapped = nn.Module()
        self.final_layer.wrapped.linear = nn.Linear(2048, 64, bias=False)


def test_quantization_scopes_are_generic() -> None:
    modules = dict(DummyNet().named_modules())
    assert quant_lib.is_candidate(
        "blocks.0.self_attn.q_proj", modules["blocks.0.self_attn.q_proj"], "attention"
    )
    assert not quant_lib.is_candidate(
        "blocks.0.mlp.layer1", modules["blocks.0.mlp.layer1"], "attention"
    )
    assert quant_lib.is_candidate(
        "x_embedder.proj.1", modules["x_embedder.proj.1"], "vision_input_proxy"
    )
    assert quant_lib.is_candidate(
        "final_layer.wrapped.linear",
        modules["final_layer.wrapped.linear"],
        "shared_output_proxy",
    )


def test_native_precision_updates_wrapper_and_net() -> None:
    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(4, 4)
            self.precision = torch.bfloat16
            self.tensor_kwargs = {"device": "cpu", "dtype": torch.bfloat16}
            self.config = type("Config", (), {"precision": "bfloat16"})()

    model = Wrapper()
    quant_lib.configure_model_precision(model, "fp16")
    assert model.precision == torch.float16
    assert model.tensor_kwargs["dtype"] == torch.float16
    assert model.config.precision == "float16"
    assert model.net.weight.dtype == torch.float16
