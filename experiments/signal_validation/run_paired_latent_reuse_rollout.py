"""Run paired 16-step causal rollouts from restored LIBERO simulator states."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation
from experiments.signal_validation.collect_latent_reuse_trace import build_cfg
from experiments.signal_validation.run_latent_reuse_validation import (
    inject_visual_content,
    latent_call,
    timed_fresh_call,
)

DEFAULT_CHECKPOINT = "/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt"
DEFAULT_STATS = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json"
DEFAULT_T5 = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl"
DEFAULT_TRACE = "reports/artifacts/libero_latent_reuse_trace.json"
DEFAULT_VALIDATION = "reports/artifacts/libero_latent_reuse_validation.json"
DEFAULT_FULL_LATENTS = "reports/artifacts/libero_latent_reuse_validation.full_latents.npz"


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = values.min(), values.max()
    return (values - low) / (high - low) if high > low else np.zeros_like(values)


def select_states(records: list[dict], arrays: dict[str, np.ndarray], validation: dict, per_task: int = 3) -> list[int]:
    rows = validation["samples"]
    visual_wrist = np.asarray([row["visual_gain_wrist"] for row in rows], dtype=np.float64)
    visual_primary = np.asarray([row["visual_gain_primary"] for row in rows], dtype=np.float64)
    e_proprio = np.mean(np.abs(arrays["predicted_proprio"] - arrays["target_proprio"]), axis=1)
    action_magnitude = np.linalg.norm(arrays["action"].reshape(len(records), -1), axis=1)
    combined = normalize(action_magnitude) + normalize(visual_wrist) + normalize(visual_primary) + normalize(e_proprio)
    selected = []
    for task_id in sorted({int(row["task_id"]) for row in records}):
        candidates = [i for i, row in enumerate(records) if int(row["task_id"]) == task_id]
        ordered = sorted(candidates, key=lambda i: combined[i])
        positions = np.linspace(0, len(ordered) - 1, per_task).round().astype(int)
        selected.extend(ordered[int(position)] for position in positions)
    return sorted(set(selected))


def names_and_positions(env) -> tuple[list[str], np.ndarray]:
    model = env.env.sim.model
    names = []
    for index in range(model.nbody):
        name = model.body_id2name(index)
        names.append(name.decode() if isinstance(name, bytes) else str(name))
    return names, env.env.sim.data.body_xpos.copy()


def object_indices(names: list[str]) -> list[int]:
    excluded = ("world", "robot", "mount", "base", "link", "gripper", "hand", "forearm", "wrist", "table", "floor", "wall", "camera", "eef")
    indices = [i for i, name in enumerate(names) if not any(token in name.lower() for token in excluded)]
    return indices or list(range(len(names)))


def capture_snapshot(env, raw: dict) -> dict:
    names, body_xpos = names_and_positions(env)
    return {
        "eef": extract_observation(raw, flip_vertical=True).proprio[2:5].astype(np.float64),
        "gripper_qpos": extract_observation(raw, flip_vertical=True).proprio[:2].astype(np.float64),
        "body_names": names,
        "body_xpos": body_xpos,
        "object_indices": object_indices(names),
        "ncon": int(env.env.sim.data.ncon),
        "qpos": env.env.sim.data.qpos.copy(),
    }


def restore_state(env, state: np.ndarray) -> tuple[dict, float]:
    raw = env.env.regenerate_obs_from_state(state)
    snapshot = capture_snapshot(env, raw)
    restore_error = float(np.max(np.abs(snapshot["qpos"] - env._restore_reference_qpos)))
    return {"raw": raw, "snapshot": snapshot}, restore_error


def rollout_branch(env, state: np.ndarray, actions: np.ndarray, reference_qpos: np.ndarray) -> dict:
    env._restore_reference_qpos = reference_qpos
    restored, restore_error = restore_state(env, state)
    raw = restored["raw"]
    initial = restored["snapshot"]
    eef_path = []
    gripper_path = []
    ncon_path = []
    done = False
    for action in actions:
        raw, _, done, _ = env.step(np.asarray(action, dtype=np.float32))
        observation = extract_observation(raw, flip_vertical=True)
        eef_path.append(observation.proprio[2:5].astype(np.float64))
        gripper_path.append(observation.proprio[:2].astype(np.float64))
        ncon_path.append(int(env.env.sim.data.ncon))
    final = capture_snapshot(env, raw)
    object_delta = final["body_xpos"] - initial["body_xpos"]
    object_delta_norm = np.linalg.norm(object_delta[initial["object_indices"]], axis=1)
    return {
        "restore_qpos_max_abs": restore_error,
        "eef_path": np.asarray(eef_path).astype(float).tolist(),
        "gripper_path": np.asarray(gripper_path).astype(float).tolist(),
        "ncon_path": ncon_path,
        "eef_endpoint_l2_from_start": float(np.linalg.norm(np.asarray(eef_path)[-1] - initial["eef"])),
        "object_displacement_max_l2": float(np.max(object_delta_norm)) if len(object_delta_norm) else 0.0,
        "object_displacement_mean_l2": float(np.mean(object_delta_norm)) if len(object_delta_norm) else 0.0,
        "contact_count_mean": float(np.mean(ncon_path)) if ncon_path else float(initial["ncon"]),
        "contact_count_final": int(final["ncon"]),
        "gripper_final": final["gripper_qpos"].astype(float).tolist(),
        "success_oracle": bool(env.env.check_success()),
        "done": bool(done),
    }


def replay_to_target(env, rows: list[dict], actions: np.ndarray, target_index: int, settle_steps: int) -> tuple[np.ndarray, dict, float]:
    raw = env.reset(0)
    settle = np.zeros(7, dtype=np.float32)
    settle[-1] = -1.0
    for _ in range(settle_steps):
        raw, _, _, _ = env.step(settle)
    target_state = None
    target_raw = None
    for row_index, row in rows:
        raw, _, _, _ = env.step(actions[row_index].tolist())
        if row_index == target_index:
            target_state = env.env.get_sim_state().copy()
            target_raw = raw
            break
    if target_state is None or target_raw is None:
        raise RuntimeError(f"could not replay task to target index {target_index}")
    target_observation = extract_observation(target_raw, flip_vertical=True)
    trace_target = target_observation
    expected = row_index
    image_error = max(
        float(np.max(np.abs(trace_target.primary_image.astype(np.int16) - arrays_global["target_primary"][expected].astype(np.int16)))),
        float(np.max(np.abs(trace_target.wrist_image.astype(np.int16) - arrays_global["target_wrist"][expected].astype(np.int16)))),
    )
    proprio_error = float(np.linalg.norm(trace_target.proprio - arrays_global["target_proprio"][expected]))
    return target_state, {"image_max_abs": image_error, "proprio_l2": proprio_error}, float(np.max(np.abs(env.env.sim.data.qpos - env.env.sim.data.qpos)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_T5)
    parser.add_argument("--trace", default=DEFAULT_TRACE)
    parser.add_argument("--validation", default=DEFAULT_VALIDATION)
    parser.add_argument("--full-latents", default=DEFAULT_FULL_LATENTS)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    if "so101" in checkpoint.lower() or "finet" in checkpoint.lower():
        raise ValueError(f"benchmark validation refuses a finetuned/SO101 checkpoint: {checkpoint}")
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    validation = json.loads(Path(args.validation).read_text(encoding="utf-8"))
    arrays_path = Path(trace["array_file"])
    if not arrays_path.is_file():
        arrays_path = Path(args.trace).parent / arrays_path
    loaded = np.load(arrays_path)
    arrays = {key: loaded[key] for key in loaded.files}
    global arrays_global
    arrays_global = arrays
    full = np.load(args.full_latents)
    current_full, target_full = full["current_full"], full["target_full"]
    records = trace["requests"]
    selected = select_states(records, arrays, validation)
    if len(selected) != 18:
        raise ValueError(f"expected 18 selected states, got {len(selected)}")

    configure_repository_paths({"repositories": {"libero": "/home/rxhuang/Projects/LIBERO", "cosmos": str(REPO_ROOT)}})
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

    selected_set = set(selected)
    by_task: dict[int, list[tuple[int, dict]]] = {}
    for index, row in enumerate(records):
        if index in selected_set:
            by_task.setdefault(int(row["task_id"]), []).append((index, row))

    all_results = []
    for task_id, task_rows in sorted(by_task.items()):
        env = RealLiberoEnvironment("libero_10", task_id, 256)
        try:
            replay_rows = [(i, records[i]) for i in range(len(records)) if int(records[i]["task_id"]) == task_id]
            raw = env.reset(0)
            settle = np.zeros(7, dtype=np.float32)
            settle[-1] = -1.0
            for _ in range(args.settle_steps):
                raw, _, _, _ = env.step(settle)
            for row_index, row in replay_rows:
                for baseline_action in arrays["action"][row_index]:
                    raw, _, _, _ = env.step(baseline_action)
                if row_index not in selected_set:
                    continue
                target_state = env.env.get_sim_state().copy()
                target_observation = extract_observation(raw, flip_vertical=True)
                image_error = max(
                    float(np.max(np.abs(target_observation.primary_image.astype(np.int16) - arrays["target_primary"][row_index].astype(np.int16)))),
                    float(np.max(np.abs(target_observation.wrist_image.astype(np.int16) - arrays["target_wrist"][row_index].astype(np.int16)))),
                )
                proprio_error = float(np.linalg.norm(target_observation.proprio - arrays["target_proprio"][row_index]))
                reference_qpos = env.env.sim.data.qpos.copy()
                base = target_full[row_index]
                pred = inject_visual_content(base, arrays["predicted_wrist"][row_index], arrays["predicted_primary"][row_index])
                observation = {
                    "primary_image": arrays["target_primary"][row_index],
                    "wrist_image": arrays["target_wrist"][row_index],
                    "proprio": arrays["target_proprio"][row_index],
                }
                fresh_actions, _ = timed_fresh_call(cfg, model, stats, observation, row["task"], int(row["seed"]))
                cache_actions, _ = latent_call(cfg, model, stats, observation, row["task"], int(row["seed"]), current_full[row_index])
                pred_actions, _ = latent_call(cfg, model, stats, observation, row["task"], int(row["seed"]), pred)
                branches = {
                    "fresh": fresh_actions,
                    "predicted": pred_actions,
                    "cache": cache_actions,
                }
                branch_results = {}
                for branch_name, branch_actions in branches.items():
                    branch_results[branch_name] = rollout_branch(env, target_state, branch_actions, reference_qpos)
                fresh_eef = np.asarray(branch_results["fresh"]["eef_path"], dtype=np.float64)
                fresh_gripper = np.asarray(branch_results["fresh"]["gripper_path"], dtype=np.float64)
                for branch_name in ("predicted", "cache"):
                    eef = np.asarray(branch_results[branch_name]["eef_path"], dtype=np.float64)
                    gripper = np.asarray(branch_results[branch_name]["gripper_path"], dtype=np.float64)
                    branch_results[branch_name]["eef_path_mean_l2_to_fresh"] = float(np.mean(np.linalg.norm(eef - fresh_eef, axis=1)))
                    branch_results[branch_name]["eef_endpoint_l2_to_fresh"] = float(np.linalg.norm(eef[-1] - fresh_eef[-1]))
                    branch_results[branch_name]["gripper_path_mean_l2_to_fresh"] = float(np.mean(np.linalg.norm(gripper - fresh_gripper, axis=1)))
                    branch_results[branch_name]["action_gripper_sign_disagreement_to_fresh"] = float(
                        np.mean(np.sign(branches[branch_name][:, 6]) != np.sign(fresh_actions[:, 6]))
                    )
                all_results.append(
                    {
                        "sample_index": row_index,
                        "task_id": task_id,
                        "control_step": int(row["control_step"]),
                        "stage": row.get("stage"),
                        "replay_target_validation": {"image_max_abs": image_error, "proprio_l2": proprio_error},
                        "branches": branch_results,
                    }
                )
                # The next baseline chunk must start from the baseline replay
                # state, not from whichever branch ran last.
                env.env.regenerate_obs_from_state(target_state)
        finally:
            env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "n_selected": len(all_results),
        "horizon_steps": 16,
        "state_restore": "LIBERO sim.get_state().flatten() / regenerate_obs_from_state",
        "value_used": False,
        "intermediate_probe_used": False,
        "selection": "3 deterministic rank-spread states per task over action magnitude, wrist/primary visual gain, and proprio error",
        "results": all_results,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n_selected": len(all_results), "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
