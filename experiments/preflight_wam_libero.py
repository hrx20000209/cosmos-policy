#!/usr/bin/env python3
"""Fail-fast validation for real Cosmos/LingBot-VA LIBERO runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.libero_harness import (  # noqa: E402
    RealLiberoEnvironment,
    apply_overrides,
    build_adapter,
    configure_repository_paths,
    load_yaml,
    repository_state,
)


class Checks:
    def __init__(self):
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: Any) -> None:
        self.items.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.items)


def _resolve_file(checks: Checks, name: str, raw_path: str | None, minimum_bytes: int = 1) -> Path | None:
    if not raw_path:
        checks.add(name, False, "path is not configured")
        return None
    path = Path(raw_path).expanduser().resolve()
    passed = path.is_file() and path.stat().st_size >= minimum_bytes
    detail = {"path": str(path), "size_bytes": path.stat().st_size if path.is_file() else None}
    checks.add(name, passed, detail)
    return path if passed else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_cosmos_assets(checks: Checks, config: dict[str, Any]) -> None:
    model = config["model"]
    checkpoint = _resolve_file(checks, "checkpoint", model.get("checkpoint"), 1_000_000_000)
    stats_path = _resolve_file(checks, "dataset_statistics", model.get("dataset_stats_path"))
    embeddings_path = _resolve_file(checks, "t5_embeddings", model.get("t5_embeddings_path"), 1_000_000)

    if checkpoint is not None:
        checks.add("checkpoint_suffix", checkpoint.suffix == ".pt", checkpoint.suffix)
        expected = model.get("checkpoint_sha256")
        actual = _sha256(checkpoint)
        checks.add("checkpoint_sha256", not expected or actual == expected, {"expected": expected, "actual": actual})
    if stats_path is not None:
        expected = model.get("dataset_stats_sha256")
        actual = _sha256(stats_path)
        checks.add(
            "dataset_statistics_sha256",
            not expected or actual == expected,
            {"expected": expected, "actual": actual},
        )
        with stats_path.open(encoding="utf-8") as handle:
            stats = json.load(handle)
        expected = {
            "actions_min",
            "actions_max",
            "actions_mean",
            "actions_std",
            "proprio_min",
            "proprio_max",
            "proprio_mean",
            "proprio_std",
        }
        dimensions_ok = (
            len(stats.get("actions_min", [])) == 7
            and len(stats.get("proprio_min", [])) == 9
        )
        checks.add(
            "dataset_statistics_schema",
            expected.issubset(stats) and dimensions_ok,
            {"keys": sorted(stats), "action_dim": 7, "proprio_dim": 9},
        )
    if embeddings_path is not None:
        expected = model.get("t5_embeddings_sha256")
        actual = _sha256(embeddings_path)
        checks.add(
            "t5_embeddings_sha256",
            not expected or actual == expected,
            {"expected": expected, "actual": actual},
        )
        # This is a trusted checkpoint artifact. Loading it here catches
        # truncated files before the expensive model construction begins.
        with embeddings_path.open("rb") as handle:
            embeddings = pickle.load(handle)
        shapes = sorted({tuple(value.shape) for value in embeddings.values() if hasattr(value, "shape")})
        checks.add(
            "t5_embeddings_schema",
            isinstance(embeddings, dict) and len(embeddings) >= 40 and shapes == [(1, 512, 1024)],
            {"entries": len(embeddings), "shapes": shapes},
        )


def _check_lingbot_assets(checks: Checks, config: dict[str, Any]) -> None:
    raw_path = config["model"].get("checkpoint")
    path = Path(raw_path).expanduser().resolve() if raw_path else None
    passed = bool(path and path.is_dir() and any(path.rglob("*.safetensors")))
    checks.add("checkpoint", passed, str(path) if path else "path is not configured")


def _check_repositories(checks: Checks, config: dict[str, Any]) -> None:
    for name, raw_path in config.get("repositories", {}).items():
        path = Path(raw_path).expanduser().resolve()
        state = repository_state(path)
        checks.add(
            f"repository:{name}",
            path.is_dir() and not str(state["commit_sha"]).startswith("unavailable:"),
            state,
        )

    libero_module = importlib.import_module("libero.libero")
    expected = Path(config["repositories"]["libero"]).expanduser().resolve()
    actual = Path(libero_module.__file__).resolve()
    checks.add("libero_import_identity", expected in actual.parents, {"expected": str(expected), "actual": str(actual)})


def _check_cuda(checks: Checks) -> bool:
    import torch

    available = torch.cuda.is_available()
    detail: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "available": available,
        "device_count": torch.cuda.device_count(),
    }
    if available:
        detail["device_0"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        detail["memory_free_bytes"] = free
        detail["memory_total_bytes"] = total
    checks.add("cuda", available, detail)
    return available


def _check_sampler_steps(checks: Checks) -> None:
    import torch

    from cosmos_policy.modules.cosmos_sampler import CosmosPolicySampler

    observed: dict[int, int] = {}
    for requested in (1, 2, 3, 4, 5, 8):
        calls = 0

        def denoiser(noisy, sigma):
            nonlocal calls
            calls += 1
            return torch.zeros_like(noisy)

        CosmosPolicySampler()(denoiser, torch.ones(1, 2), num_steps=requested, sigma_min=4.0, sigma_max=80.0)
        observed[requested] = calls
    checks.add("cosmos_denoiser_step_contract", all(key == value for key, value in observed.items()), observed)


def _check_environment(checks: Checks, config: dict[str, Any], task_id: int) -> str | None:
    evaluation = config["evaluation"]
    try:
        env = RealLiberoEnvironment(
            evaluation["task_suite"],
            task_id,
            int(evaluation.get("resolution", 128)),
        )
        try:
            observation = env.reset(0)
            _, _, done, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            resolution = int(evaluation.get("resolution", 128))
            shapes = {
                "agentview": list(observation["agentview_image"].shape),
                "wrist": list(observation["robot0_eye_in_hand_image"].shape),
            }
            passed = shapes == {
                "agentview": [resolution, resolution, 3],
                "wrist": [resolution, resolution, 3],
            }
            checks.add(
                "libero_reset_render_step",
                passed,
                {
                    "task": env.description,
                    "init_states": len(env.initial_states),
                    "shapes": shapes,
                    "done": bool(done),
                },
            )
            return env.description
        finally:
            env.close()
    except Exception as error:
        checks.add("libero_reset_render_step", False, f"{type(error).__name__}: {error}")
        return None


def _get_task_description(config: dict[str, Any], task_id: int) -> str:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[config["evaluation"]["task_suite"]]()
    return suite.get_task(task_id).language


def _load_model(
    checks: Checks,
    model_name: str,
    config: dict[str, Any],
    task: str | None,
    run_inference: bool = False,
) -> None:
    if task is None:
        checks.add("model_load", False, "environment task description is unavailable")
        return
    adapter = build_adapter(model_name, config, mock=False)
    try:
        adapter.reset(task, 195)
        module = getattr(adapter, "model", None)
        parameters = sum(parameter.numel() for parameter in module.parameters()) if module is not None else None
        checks.add("model_load", module is not None, {"parameter_count": parameters})
        if run_inference:
            import time

            import torch

            from runtime.async_pipeline import InferenceRequest
            from runtime.observation_buffer import Observation
            from runtime.runtime_metrics import monotonic_ns

            resolution = int(config["model"].get("image_resolution", 224))
            image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            if model_name == "cosmos":
                with Path(config["model"]["dataset_stats_path"]).open(encoding="utf-8") as handle:
                    proprio = np.asarray(json.load(handle)["proprio_mean"], dtype=np.float32)
                steps = 1
            else:
                proprio = np.zeros(9, dtype=np.float32)
                steps = int(config["denoising"].get("minimum_steps", 5))
            observation = Observation(monotonic_ns(), image, image.copy(), proprio)
            request = InferenceRequest.create(observation.timestamp_ns, "preflight", 0)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            output = adapter.infer(observation, request, steps, [observation])
            torch.cuda.synchronize()
            elapsed_seconds = time.monotonic() - started
            expected_shape = (
                int(config["model"].get("action_horizon", 16)),
                int(config["model"].get("action_dim", 7)),
            )
            passed = (
                output.actions.shape == expected_shape
                and np.isfinite(output.actions).all()
                and output.denoiser_forward_count >= steps
            )
            checks.add(
                "synthetic_model_inference",
                passed,
                {
                    "synthetic_observation": True,
                    "requested_steps": steps,
                    "reported_denoiser_forwards": output.denoiser_forward_count,
                    "action_shape": list(output.actions.shape),
                    "action_min": float(output.actions.min()),
                    "action_max": float(output.actions.max()),
                    "elapsed_seconds": elapsed_seconds,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                    "stage_metrics_ms": output.stage_metrics_ms,
                },
            )
    except Exception as error:
        check_name = "synthetic_model_inference" if run_inference and any(
            item["name"] == "model_load" and item["passed"] for item in checks.items
        ) else "model_load"
        checks.add(check_name, False, f"{type(error).__name__}: {error}")
    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cosmos", "lingbot_va"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--run-inference", action="store_true", help="run one synthetic observation through the model")
    parser.add_argument("--output", help="optional path for a machine-readable JSON report")
    args = parser.parse_args()

    config = apply_overrides(load_yaml(args.config), args.overrides)
    config["model"]["name"] = args.model
    configure_repository_paths(config)
    checks = Checks()

    _check_repositories(checks, config)
    if args.model == "cosmos":
        _check_cosmos_assets(checks, config)
        _check_sampler_steps(checks)
    else:
        _check_lingbot_assets(checks, config)
    cuda_available = _check_cuda(checks)
    task = _check_environment(checks, config, args.task_id)
    if task is None:
        task = _get_task_description(config, args.task_id)
    if args.load_model or args.run_inference:
        if cuda_available:
            _load_model(checks, args.model, config, task, run_inference=args.run_inference)
        else:
            checks.add("model_load", False, "skipped because CUDA is unavailable")

    result = {"passed": checks.passed, "model": args.model, "checks": checks.items}
    rendered = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    )
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if checks.passed else 1)


if __name__ == "__main__":
    main()
