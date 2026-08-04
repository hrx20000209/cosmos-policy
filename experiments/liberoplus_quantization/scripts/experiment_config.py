"""Typed YAML configuration for reproducible LIBERO-Plus evaluations."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DynamicReplanConfig:
    enabled: bool = False
    candidate_open_loop_steps: list[int] = field(default_factory=lambda: [4, 8, 16])
    visual_change_low: float = 0.015
    visual_change_high: float = 0.050
    action_magnitude_high: float = 0.35
    action_jerk_high: float = 0.25
    no_progress_steps: int = 8
    min_open_loop_steps: int = 4
    dynamic_denoising: bool = False
    denoising_steps_low_risk: int = 4
    denoising_steps_high_risk: int = 8


@dataclass
class EvalConfig:
    experiment_name: str
    checkpoint: str
    dataset_stats_path: str
    t5_embeddings_path: str
    task_list: str
    output_tag: str
    notes: list[str] = field(default_factory=list)
    precision: str = "bf16"
    quantization_backend: str = "none"
    quantization_mode: str = "bf16"
    quantization_scope: str = "backbone"
    quantized_modules: list[str] = field(default_factory=list)
    excluded_modules: list[str] = field(default_factory=list)
    group_size: int = 128
    real_quantization_required: bool = False
    task_names: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [195])
    episodes_per_task: int = 1
    initial_state_index: int = 0
    chunk_size: int = 16
    num_open_loop_steps: int = 16
    denoising_steps: int = 5
    device: str = "cuda:0"
    episode_timeout_seconds: float = 300.0
    max_environment_steps: int | None = None
    warmup_policy_calls: int = 1
    skip_completed: bool = True
    fail_on_missing_t5: bool = True
    deterministic: bool = True
    flip_images: bool = True
    use_jpeg_compression: bool = True
    use_wrist_image: bool = True
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    collect_power: bool = True
    dynamic_replan: DynamicReplanConfig = field(default_factory=DynamicReplanConfig)

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 1 <= self.num_open_loop_steps <= self.chunk_size:
            raise ValueError("num_open_loop_steps must be in [1, chunk_size]")
        if self.denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.episodes_per_task <= 0:
            raise ValueError("episodes_per_task must be positive")
        if self.dynamic_replan.enabled:
            candidates = self.dynamic_replan.candidate_open_loop_steps
            if not candidates or any(step < 1 or step > self.chunk_size for step in candidates):
                raise ValueError("dynamic open-loop candidates must be in [1, chunk_size]")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _expand(value: Any, base_dir: Path) -> Any:
    if isinstance(value, str):
        value = os.path.expandvars(os.path.expanduser(value))
        if value.startswith("./") or value.startswith("../"):
            return str((base_dir / value).resolve())
    if isinstance(value, list):
        return [_expand(item, base_dir) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, base_dir) for key, item in value.items()}
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f"cyclic config inheritance at {path}")
    seen.add(path)
    with path.open() as stream:
        raw = yaml.safe_load(stream) or {}
    base_name = raw.pop("base", None)
    if base_name is not None:
        base_path = (path.parent / base_name).resolve()
        raw = _deep_merge(_load_yaml(base_path, seen), raw)
    seen.remove(path)
    return raw


def load_config(path: str | os.PathLike[str]) -> EvalConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path, set())
    raw = _expand(raw, config_path.parent)
    dynamic = DynamicReplanConfig(**raw.pop("dynamic_replan", {}))
    cfg = EvalConfig(dynamic_replan=dynamic, **raw)
    cfg.validate()
    return cfg
