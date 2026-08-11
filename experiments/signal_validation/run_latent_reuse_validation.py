"""Run Cosmos future-latent reuse validation on aligned LIBERO states.

The experiment compares five calls on exactly the same target state and seed:

* fresh: target RGB -> VAE -> DiT
* cache: t visual latent -> DiT, target proprio
* predicted_both: predicted wrist + primary content -> current slots -> DiT
* predicted_wrist: predicted wrist, fresh primary -> DiT
* predicted_primary: fresh wrist, predicted primary -> DiT

No value prediction is read and no intermediate-feature probe is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.signal_validation.collect_latent_reuse_trace import build_cfg

DEFAULT_CHECKPOINT = "/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt"
DEFAULT_STATS = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json"
DEFAULT_T5 = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl"
DEFAULT_TRACE = "reports/artifacts/libero_latent_reuse_trace.json"
DEFAULT_SIGNAL = "/data/rxhuang/wam_signal_validation/libero6_original_steps1.json"
CAMERA_NAMES = ("wrist", "primary")
VARIANTS = ("cache", "predicted_both", "predicted_wrist", "predicted_primary")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prepare_full_video(cfg, current_wrist: np.ndarray, current_primary: np.ndarray) -> torch.Tensor:
    """Build the exact 33-pixel-frame LIBERO sequence used by get_action."""
    from cosmos_policy.experiments.robot.cosmos_utils import prepare_images_for_model

    blank_source = np.zeros_like(current_primary)
    processed = np.asarray(
        prepare_images_for_model(
            [blank_source, current_wrist, current_primary, current_wrist, current_primary],
            cfg,
        ),
        dtype=np.uint8,
    )
    blank, wrist, primary, future_wrist, future_primary = processed
    sequence = np.concatenate(
        [
            blank[None],
            np.repeat(blank[None], 4, axis=0),
            np.repeat(wrist[None], 4, axis=0),
            np.repeat(primary[None], 4, axis=0),
            np.repeat(blank[None], 4, axis=0),
            np.repeat(blank[None], 4, axis=0),
            np.repeat(future_wrist[None], 4, axis=0),
            np.repeat(future_primary[None], 4, axis=0),
            np.repeat(blank[None], 4, axis=0),
        ],
        axis=0,
    )
    video = torch.from_numpy(np.transpose(sequence[None], (0, 4, 1, 2, 3))).cuda()
    return video.to(dtype=torch.bfloat16) / 127.5 - 1.0


def encode_full_latents(model, cfg, arrays: dict[str, np.ndarray], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode current and target states while preserving full temporal context."""
    current_videos = []
    target_videos = []
    for wrist, primary in zip(arrays["current_wrist"], arrays["current_primary"]):
        current_videos.append(prepare_full_video(cfg, wrist, primary))
    for wrist, primary in zip(arrays["target_wrist"], arrays["target_primary"]):
        target_videos.append(prepare_full_video(cfg, wrist, primary))

    current_latents: list[np.ndarray] = []
    target_latents: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(current_videos), batch_size):
            current_batch = torch.cat(current_videos[start : start + batch_size], dim=0)
            target_batch = torch.cat(target_videos[start : start + batch_size], dim=0)
            current_latents.append(model.encode(current_batch).float().cpu().numpy())
            target_latents.append(model.encode(target_batch).float().cpu().numpy())
            del current_batch, target_batch
    return np.concatenate(current_latents, axis=0), np.concatenate(target_latents, axis=0)


def inject_visual_content(base: np.ndarray, wrist: np.ndarray | None, primary: np.ndarray | None) -> np.ndarray:
    """Replace content in both current and duplicate future visual slots.

    The semantic content is copied into slot 2/3 (the conditioned current
    positions) and slots 6/7 (the current-image placeholder positions).  The
    tensor itself remains at its original slot indices, so DiT positional
    information is not copied from future slots into current slots.
    """
    result = base.copy()
    if wrist is not None:
        result[:, 2] = wrist
        result[:, 6] = wrist
    if primary is not None:
        result[:, 3] = primary
        result[:, 7] = primary
    return result


def timed_fresh_call(cfg, model, stats, observation: dict, task: str, seed: int) -> tuple[np.ndarray, dict]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    sink: dict[str, float] = {}
    cfg._inference_metrics_sink = sink
    original_encode = model.encode

    def timed_encode(state: torch.Tensor) -> torch.Tensor:
        _sync()
        start = time.perf_counter_ns()
        output = original_encode(state)
        _sync()
        sink["vae_encoding_ms"] = (time.perf_counter_ns() - start) / 1e6
        return output

    model.encode = timed_encode
    try:
        start_ns = time.perf_counter_ns()
        result = get_action(
            cfg,
            model,
            stats,
            observation,
            task,
            seed=seed,
            num_denoising_steps_action=1,
            generate_future_state_and_value_in_parallel=False,
            decode_future_state=False,
        )
        _sync()
        total_ms = (time.perf_counter_ns() - start_ns) / 1e6
    finally:
        if "encode" in model.__dict__:
            del model.__dict__["encode"]
        cfg._inference_metrics_sink = None
    actions = np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)
    return actions, {"total_wall_ms": total_ms, **{key: float(value) for key, value in sink.items()}}


def latent_call(
    cfg,
    model,
    stats,
    observation: dict,
    task: str,
    seed: int,
    latent: np.ndarray,
) -> tuple[np.ndarray, dict]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    sink: dict[str, float] = {}
    cfg._inference_metrics_sink = sink
    latent_tensor = torch.from_numpy(latent[None]).cuda()
    start_ns = time.perf_counter_ns()
    try:
        result = get_action(
            cfg,
            model,
            stats,
            observation,
            task,
            seed=seed,
            num_denoising_steps_action=1,
            generate_future_state_and_value_in_parallel=False,
            decode_future_state=False,
            skip_vae_encoding=True,
            previous_generated_latent=latent_tensor,
            skip_camera_preprocessing=True,
        )
        _sync()
        total_ms = (time.perf_counter_ns() - start_ns) / 1e6
    finally:
        cfg._inference_metrics_sink = None
        del latent_tensor
    actions = np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)
    return actions, {
        "total_wall_ms": total_ms,
        "vae_encoding_ms": 0.0,
        "camera_preprocessing_bypassed": True,
        **{key: float(value) for key, value in sink.items()},
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(np.dot(a_flat, b_flat) / denominator) if denominator > 1e-12 else 0.0


def action_distance(actions: np.ndarray, fresh: np.ndarray) -> dict:
    diff = actions.astype(np.float64) - fresh.astype(np.float64)
    per_step_l2 = np.linalg.norm(diff, axis=1)
    fresh_step_norm = np.linalg.norm(fresh.astype(np.float64), axis=1)
    variant_step_norm = np.linalg.norm(actions.astype(np.float64), axis=1)
    trajectory = np.cumsum(diff, axis=0)
    return {
        "mean_per_step_l2": float(np.mean(per_step_l2)),
        "first_action_l2": float(per_step_l2[0]),
        "direction_cosine": cosine(actions, fresh),
        "action_magnitude_difference": float(np.mean(np.abs(variant_step_norm - fresh_step_norm))),
        "per_dim_mean_abs": np.mean(np.abs(diff), axis=0).astype(float).tolist(),
        "gripper_disagreement": float(np.mean(np.sign(actions[:, 6]) != np.sign(fresh[:, 6]))),
        "eef_action_mean_l2": float(np.mean(np.linalg.norm(diff[:, :6], axis=1))),
        "chunk_trajectory_mean_l2": float(np.mean(np.linalg.norm(trajectory, axis=1))),
        "chunk_trajectory_endpoint_l2": float(np.linalg.norm(trajectory[-1])),
        "prefix_mean_per_step_l2": {
            str(h): float(np.mean(per_step_l2[:h])) for h in (1, 4, 8, 16)
        },
        "prefix_trajectory_endpoint_l2": {
            str(h): float(np.linalg.norm(trajectory[h - 1])) for h in (1, 4, 8, 16)
        },
    }


def mean_quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def correlation(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {"pearson": pearson(a, b), "spearman": pearson(rankdata(a), rankdata(b))}


def pca_summary(predicted: np.ndarray, real: np.ndarray, components: int = 3) -> dict:
    result = {}
    for name, values in (("predicted", predicted), ("real", real)):
        matrix = values.reshape(len(values), -1).astype(np.float64)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        variance = singular_values**2
        total = float(np.sum(variance))
        result[name] = {
            "explained_variance_ratio_top3": (
                (variance[:components] / total).astype(float).tolist() if total > 0 else []
            ),
            "total_centered_variance": total / max(1, len(values) - 1),
        }
    return result


def distribution_metrics(predicted: np.ndarray, real: np.ndarray) -> dict:
    predicted = predicted.astype(np.float64)
    real = real.astype(np.float64)
    pred_norm = np.linalg.norm(predicted.reshape(len(predicted), -1), axis=1)
    real_norm = np.linalg.norm(real.reshape(len(real), -1), axis=1)
    delta = predicted - real
    return {
        "predicted_norm": mean_quantiles(pred_norm),
        "real_norm": mean_quantiles(real_norm),
        "norm_ratio_predicted_over_real_mean": float(np.mean(pred_norm / np.maximum(real_norm, 1e-12))),
        "predicted_elementwise_mean": float(np.mean(predicted)),
        "real_elementwise_mean": float(np.mean(real)),
        "elementwise_mean_bias_l2": float(np.linalg.norm(np.mean(predicted - real, axis=0))),
        "elementwise_std_ratio_predicted_over_real_mean": float(
            np.mean(np.std(predicted, axis=0) / np.maximum(np.std(real, axis=0), 1e-8))
        ),
        "cosine": mean_quantiles([cosine(p, r) for p, r in zip(predicted, real)]),
        "l1": mean_quantiles(np.mean(np.abs(delta.reshape(len(delta), -1)), axis=1)),
        "l2": mean_quantiles(np.linalg.norm(delta.reshape(len(delta), -1), axis=1)),
        "pca": pca_summary(predicted, real),
    }


def build_visual_gain_lookup(path: str) -> dict[tuple[int, int], dict[str, float]]:
    if not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for row in data.get("requests", []):
        key = (int(row["task_id"]), int(row["control_step"]))
        result[key] = {}
        for camera, field in (("wrist", "visual_future_wrist"), ("primary", "visual_future_primary")):
            values = [item for item in row.get(field, []) if int(item.get("horizon_steps", -1)) == 16]
            if values:
                result[key][camera] = float(values[0]["gain_e_const_minus_e_pred"])
    return result


def summarize_action_rows(rows: list[dict]) -> dict:
    summary = {}
    for key in ("mean_per_step_l2", "first_action_l2", "direction_cosine", "action_magnitude_difference", "gripper_disagreement", "eef_action_mean_l2", "chunk_trajectory_mean_l2", "chunk_trajectory_endpoint_l2"):
        summary[key] = mean_quantiles([row[key] for row in rows])
    summary["per_dim_mean_abs_mean"] = np.mean(
        np.asarray([row["per_dim_mean_abs"] for row in rows], dtype=np.float64), axis=0
    ).astype(float).tolist()
    summary["prefix_mean_per_step_l2"] = {
        str(h): mean_quantiles([row["prefix_mean_per_step_l2"][str(h)] for row in rows])
        for h in (1, 4, 8, 16)
    }
    summary["prefix_trajectory_endpoint_l2"] = {
        str(h): mean_quantiles([row["prefix_trajectory_endpoint_l2"][str(h)] for row in rows])
        for h in (1, 4, 8, 16)
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_T5)
    parser.add_argument("--trace", default=DEFAULT_TRACE)
    parser.add_argument("--signal-validation", default=DEFAULT_SIGNAL)
    parser.add_argument("--vae-batch-size", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    if "so101" in checkpoint.lower() or "finet" in checkpoint.lower():
        raise ValueError(f"benchmark validation refuses a finetuned/SO101 checkpoint: {checkpoint}")
    trace_path = Path(args.trace)
    metadata = json.loads(trace_path.read_text(encoding="utf-8"))
    arrays_path = Path(metadata["array_file"])
    if not arrays_path.is_absolute() and not arrays_path.is_file():
        arrays_path = trace_path.parent / arrays_path
    loaded = np.load(arrays_path)
    arrays = {key: loaded[key] for key in loaded.files}
    records = metadata["requests"]
    if len(records) != 75:
        raise ValueError(f"expected 75 aligned requests, got {len(records)}")
    if metadata["checkpoint_sha256"] != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
        raise ValueError("trace checkpoint hash does not match requested checkpoint")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    init_t5_text_embeddings_cache(args.t5_embeddings)
    stats = load_dataset_stats(args.dataset_stats)
    cfg = build_cfg(checkpoint)
    model, _ = get_model(cfg)
    model.eval()
    model.intermediate_feature_ids = None
    model.intermediate_feature_reducer = None

    latent_cache_path = Path(args.output).with_suffix(".full_latents.npz")
    if latent_cache_path.is_file():
        cached_latents = np.load(latent_cache_path)
        current_full = cached_latents["current_full"]
        target_full = cached_latents["target_full"]
        print(f"[reuse] loaded cached full latents shape={current_full.shape}", flush=True)
    else:
        print(f"[reuse] encoding {len(records)} current and target full sequences", flush=True)
        current_full, target_full = encode_full_latents(model, cfg, arrays, args.vae_batch_size)
        np.savez_compressed(latent_cache_path, current_full=current_full, target_full=target_full)
        print(f"[reuse] encoded full latents shape={current_full.shape}", flush=True)

    visual_predictions = {
        "wrist": arrays["predicted_wrist"],
        "primary": arrays["predicted_primary"],
    }
    visual_real = {
        "wrist": target_full[:, :, 2],
        "primary": target_full[:, :, 3],
    }
    distributions = {camera: distribution_metrics(visual_predictions[camera], visual_real[camera]) for camera in CAMERA_NAMES}

    actions_by_variant: dict[str, list[np.ndarray]] = {"fresh": [], **{variant: [] for variant in VARIANTS}}
    latency_by_variant: dict[str, list[dict]] = {"fresh": []}
    distance_by_variant: dict[str, list[dict]] = {variant: [] for variant in VARIANTS}
    visual_gains = build_visual_gain_lookup(args.signal_validation)
    sample_rows: list[dict] = []

    for index, record in enumerate(records):
        observation = {
            "primary_image": arrays["target_primary"][index],
            "wrist_image": arrays["target_wrist"][index],
            "proprio": arrays["target_proprio"][index],
        }
        fresh_actions, fresh_latency = timed_fresh_call(
            cfg, model, stats, observation, record["task"], int(record["seed"])
        )
        actions_by_variant["fresh"].append(fresh_actions)
        latency_by_variant["fresh"].append(fresh_latency)

        base = target_full[index]
        cache = current_full[index]
        pred_wrist = visual_predictions["wrist"][index]
        pred_primary = visual_predictions["primary"][index]
        overrides = {
            "cache": cache,
            "predicted_both": inject_visual_content(base, pred_wrist, pred_primary),
            "predicted_wrist": inject_visual_content(base, pred_wrist, None),
            "predicted_primary": inject_visual_content(base, None, pred_primary),
        }
        row = {
            "task_id": int(record["task_id"]),
            "control_step": int(record["control_step"]),
            "stage": record.get("stage"),
            "e_proprio_l1": float(np.mean(np.abs(arrays["predicted_proprio"][index] - arrays["target_proprio"][index]))),
            "visual_gain_wrist": visual_gains.get((int(record["task_id"]), int(record["control_step"])), {}).get("wrist"),
            "visual_gain_primary": visual_gains.get((int(record["task_id"]), int(record["control_step"])), {}).get("primary"),
        }
        for variant, override in overrides.items():
            actions, latency = latent_call(
                cfg,
                model,
                stats,
                observation,
                record["task"],
                int(record["seed"]),
                override,
            )
            actions_by_variant[variant].append(actions)
            latency_by_variant.setdefault(variant, []).append(latency)
            distance_by_variant[variant].append(action_distance(actions, fresh_actions))
            row[variant] = distance_by_variant[variant][-1]
        sample_rows.append(row)
        if (index + 1) % 5 == 0:
            print(f"[reuse] evaluated {index + 1}/{len(records)}", flush=True)

    action_summary = {
        variant: summarize_action_rows(rows) for variant, rows in distance_by_variant.items()
    }
    task_summary = {}
    for task_id in sorted({int(row["task_id"]) for row in sample_rows}):
        task_summary[str(task_id)] = {
            variant: summarize_action_rows(
                [row[variant] for row in sample_rows if int(row["task_id"]) == task_id]
            )
            for variant in VARIANTS
        }

    cache_d = np.asarray([row["cache"]["mean_per_step_l2"] for row in sample_rows])
    pred_both_d = np.asarray([row["predicted_both"]["mean_per_step_l2"] for row in sample_rows])
    pred_wrist_d = np.asarray([row["predicted_wrist"]["mean_per_step_l2"] for row in sample_rows])
    pred_primary_d = np.asarray([row["predicted_primary"]["mean_per_step_l2"] for row in sample_rows])
    e_proprio = np.asarray([row["e_proprio_l1"] for row in sample_rows])

    proprio_relation = {
        "target_error": "mean absolute proprio error between predicted future slot and real t+16 proprio",
        "reuse_action_error": "mean per-step action L2 between predicted_both and fresh",
        **correlation(e_proprio, pred_both_d),
        "by_task": {},
    }
    for task_id in sorted({int(row["task_id"]) for row in sample_rows}):
        mask = np.asarray([int(row["task_id"]) == task_id for row in sample_rows])
        proprio_relation["by_task"][str(task_id)] = {
            "n": int(np.sum(mask)),
            **correlation(e_proprio[mask], pred_both_d[mask]),
        }
    q25, q75 = np.quantile(e_proprio, [0.25, 0.75])
    bins = {
        "bottom_25pct": e_proprio <= q25,
        "middle_50pct": (e_proprio > q25) & (e_proprio < q75),
        "top_25pct": e_proprio >= q75,
    }
    proprio_relation["quantile_groups"] = {}
    for name, mask in bins.items():
        improvement = cache_d[mask] - pred_both_d[mask]
        proprio_relation["quantile_groups"][name] = {
            "n": int(np.sum(mask)),
            "e_proprio_l1": mean_quantiles(e_proprio[mask]),
            "reuse_action_error_median": float(np.median(pred_both_d[mask])),
            "reuse_action_error_p90": float(np.quantile(pred_both_d[mask], 0.90)),
            "fraction_predicted_better_than_cache": float(np.mean(improvement > 0)),
        }

    action_improvement = {
        "predicted_both": (cache_d - pred_both_d).astype(float).tolist(),
        "predicted_wrist": (cache_d - pred_wrist_d).astype(float).tolist(),
        "predicted_primary": (cache_d - pred_primary_d).astype(float).tolist(),
    }
    gain_action_relation = {}
    for camera, variant, values in (
        ("wrist", "predicted_wrist", pred_wrist_d),
        ("primary", "predicted_primary", pred_primary_d),
    ):
        rows = [row for row in sample_rows if row[f"visual_gain_{camera}"] is not None]
        gains = np.asarray([row[f"visual_gain_{camera}"] for row in rows], dtype=np.float64)
        improvements = np.asarray([
            row["cache"]["mean_per_step_l2"] - row[variant]["mean_per_step_l2"] for row in rows
        ])
        gain_action_relation[camera] = {
            "n": len(rows),
            "visual_gain_definition": "constant-current latent L1 minus predicted latent L1 at horizon 16",
            "action_improvement_definition": "D_cache - D_predicted_view",
            **correlation(gains, improvements),
            "visual_gain_summary": mean_quantiles(gains),
            "action_improvement_summary": mean_quantiles(improvements),
        }

    latency_summary = {}
    for variant, rows in latency_by_variant.items():
        latency_summary[variant] = {
            key: mean_quantiles([row.get(key, 0.0) for row in rows])
            for key in ("total_wall_ms", "preprocess_and_h2d_ms", "vae_encoding_ms", "generation_wall_ms", "postprocess_ms")
        }
        latency_summary[variant]["vae_encode_count"] = int(sum(row.get("vae_encoding_ms", 0.0) > 0 for row in rows))
    latency_summary["fresh_minus_reuse_total_mean_ms"] = float(
        latency_summary["fresh"]["total_wall_ms"]["mean"]
        - latency_summary["cache"]["total_wall_ms"]["mean"]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "n_aligned": len(records),
        "denoising_steps": 1,
        "action_horizon": 16,
        "same_seed": True,
        "value_used": False,
        "intermediate_probe_used": False,
        "slot_layout": metadata["slot_layout"],
        "latent_reuse_method": {
            "fresh": "target RGB full 33-frame sequence encoded by model.encode",
            "cache": "full t RGB sequence latent passed as previous_generated_latent",
            "predicted_both": "target full latent with generated slot 6/7 content copied to current slot 2/3 and placeholder 6/7",
            "predicted_wrist": "predicted wrist content plus fresh target primary content",
            "predicted_primary": "fresh target wrist content plus predicted primary content",
            "conditioning_positions": "current slot indices 2/3; future slot positional information is not copied",
        },
        "action_metrics_definition": {
            "D": "mean per-step action L2 over the 16x7 chunk",
            "direction_cosine": "cosine over the flattened 16x7 action chunk",
            "eef_action": "first six LIBERO action dimensions",
            "chunk_trajectory": "cumulative sum of action difference, reported as mean and endpoint L2",
        },
        "action_summary": action_summary,
        "task_summary": task_summary,
        "action_improvement": action_improvement,
        "distribution_compatibility": distributions,
        "proprio_relation": proprio_relation,
        "visual_gain_vs_action_improvement": gain_action_relation,
        "latency": latency_summary,
        "samples": sample_rows,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n_aligned": len(records), "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
