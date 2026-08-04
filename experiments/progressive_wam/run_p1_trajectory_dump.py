"""P1: denoising-trajectory audit for Cosmos Policy on LIBERO.

Runs real LIBERO episodes under the **official** 5-step action schedule, and at
every policy request records the predicted-clean action after each of the 5 full
denoiser forwards.  At a configurable stride it additionally runs
``standalone_k_step`` generations (k = 1..N) from the identical observation, seed
and scheduler state, so the report can separate:

  * a one-step *sampler configuration* (standalone_one_step), from
  * the first *checkpoint* of the full schedule, from
  * a noisy solver state (never executed).

The episode itself is always driven by the official 5-step actions, so the state
distribution being audited is the real baseline distribution.

The MuJoCo state is snapshotted at every request, which is what lets P2 branch
from exactly these states without re-running the policy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.libero_harness import (  # noqa: E402
    RealLiberoEnvironment,
    configure_repository_paths,
    extract_observation,
)
from experiments.progressive_wam.cosmos_hook import CosmosCheckpointCapture  # noqa: E402
from experiments.progressive_wam.provenance import run_provenance, write_json  # noqa: E402
from experiments.progressive_wam.task_stage import label_episode  # noqa: E402


DEFAULT_CHECKPOINT = "/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt"
DEFAULT_STATS = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json"
DEFAULT_T5 = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl"
KNOWN_CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def build_cfg(args: argparse.Namespace) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        suite="libero",
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=args.checkpoint,
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True,
        num_third_person_images=1,
        use_wrist_image=True,
        num_wrist_images=1,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        use_variance_scale=False,
        use_jpeg_compression=True,
        trained_with_image_aug=True,
        chunk_size=args.action_horizon,
        action_dim=7,
    )


def load_model(cfg: Any, args: argparse.Namespace):
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    init_t5_text_embeddings_cache(args.t5_embeddings)
    dataset_stats = load_dataset_stats(args.dataset_stats)
    model, train_config = get_model(cfg)
    train_horizon = int(train_config.dataloader_train.dataset.chunk_size)
    if train_horizon != args.action_horizon:
        raise RuntimeError(f"checkpoint action horizon is {train_horizon}, configured {args.action_horizon}")
    return model, dataset_stats


def actions_from_result(result: dict) -> np.ndarray:
    return np.asarray(result["actions"], dtype=np.float32).reshape(-1, 7)


def run_episode(
    *,
    env: RealLiberoEnvironment,
    model: Any,
    cfg: Any,
    dataset_stats: dict,
    capture: CosmosCheckpointCapture,
    task_suite: str,
    task_id: int,
    episode_index: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    episode_id = f"{task_suite}:{task_id}:{episode_index}:{seed}"
    raw = env.reset(episode_index)
    settle = np.zeros(7, dtype=np.float32)
    settle[-1] = args.settle_gripper_action
    for _ in range(args.settle_steps):
        raw, _, _, _ = env.step(settle)

    max_steps = args.max_steps or MAX_STEPS.get(task_suite, 520)
    control_step = 0
    request_index = 0
    success = False
    requests: list[dict[str, Any]] = []
    executed_actions: list[np.ndarray] = []
    executed_proprio: list[np.ndarray] = []
    timing_rows: list[dict[str, Any]] = []

    while control_step < max_steps and not success:
        observation = extract_observation(raw, flip_vertical=True)
        obs_dict = {
            "primary_image": observation.primary_image,
            "wrist_image": observation.wrist_image,
            "proprio": observation.proprio,
        }
        sim_state = np.asarray(env.env.get_sim_state(), dtype=np.float64)
        request_id = f"{episode_id}:req{request_index}"

        wall_start = time.perf_counter()
        with capture.request(request_id, wall_start) as checkpoints:
            result = get_action(
                cfg,
                model,
                dataset_stats,
                obs_dict,
                env.description,
                seed=seed,
                num_denoising_steps_action=args.full_steps,
                generate_future_state_and_value_in_parallel=True,
                decode_future_state=False,
            )
        full_latency_ms = (time.perf_counter() - wall_start) * 1e3
        capture.assert_indices_match(result["latent_indices"])

        final_actions = actions_from_result(result)
        if len(checkpoints) != args.full_steps:
            raise RuntimeError(
                f"expected {args.full_steps} checkpoints for a {args.full_steps}-step schedule, "
                f"captured {len(checkpoints)}"
            )

        # The last checkpoint IS the sampler output; verifying it against the
        # official extraction path proves the hook reads the same tensor the
        # policy executes rather than a look-alike.
        hook_final = capture.unnormalize_action(checkpoints[-1].predicted_clean_action.numpy())
        final_agreement = float(np.max(np.abs(hook_final - final_actions)))

        checkpoint_actions = np.stack(
            [capture.unnormalize_action(cp.predicted_clean_action.numpy()) for cp in checkpoints]
        )
        record: dict[str, Any] = {
            "request_id": request_id,
            "episode_id": episode_id,
            "task_suite": task_suite,
            "task_id": task_id,
            "episode_index": episode_index,
            "seed": seed,
            "control_step": control_step,
            "request_index": request_index,
            "sim_state": sim_state,
            "proprio": np.asarray(observation.proprio, dtype=np.float32),
            "final_actions": final_actions,
            "checkpoint_actions": checkpoint_actions,
            "sigmas": np.asarray([cp.sigma for cp in checkpoints], dtype=np.float64),
            "values": np.asarray(
                [float(cp.value_prediction.item()) if cp.value_prediction is not None else np.nan for cp in checkpoints]
            ),
            "final_agreement_max_abs": final_agreement,
            "policy_latency_ms": full_latency_ms,
            "gpu_elapsed_ms": [cp.extra.get("gpu_elapsed_ms_from_request_start") for cp in checkpoints],
        }
        if args.capture_future_latent:
            record["future_latents"] = torch.stack(
                [cp.predicted_future_latent for cp in checkpoints]
            ).numpy()

        # ---- standalone_k_step probes on the same observation and seed -------
        if args.standalone_stride > 0 and request_index % args.standalone_stride == 0:
            standalone: dict[int, np.ndarray] = {}
            standalone_latency: dict[int, float] = {}
            for k in args.standalone_steps:
                probe_start = time.perf_counter()
                with capture.request(f"{request_id}:standalone{k}", probe_start) as probe_cps:
                    probe = get_action(
                        cfg,
                        model,
                        dataset_stats,
                        obs_dict,
                        env.description,
                        seed=seed,
                        num_denoising_steps_action=k,
                        generate_future_state_and_value_in_parallel=True,
                        decode_future_state=False,
                    )
                standalone_latency[k] = (time.perf_counter() - probe_start) * 1e3
                standalone[k] = actions_from_result(probe)
                if len(probe_cps) != k:
                    raise RuntimeError(f"standalone_{k}_step produced {len(probe_cps)} checkpoints")
            record["standalone_actions"] = {int(k): v for k, v in standalone.items()}
            record["standalone_latency_ms"] = {int(k): v for k, v in standalone_latency.items()}

        requests.append(record)
        timing_rows.append(
            {
                "request_id": request_id,
                "control_step": control_step,
                "policy_latency_ms": full_latency_ms,
                "gpu_elapsed_ms": record["gpu_elapsed_ms"],
                "sigmas": record["sigmas"].tolist(),
                "final_agreement_max_abs": final_agreement,
            }
        )

        prefix = min(args.execute_horizon, final_actions.shape[0])
        for step_index in range(prefix):
            action = final_actions[step_index]
            executed_actions.append(action.copy())
            executed_proprio.append(np.asarray(extract_observation(raw, True).proprio, dtype=np.float32))
            raw, _, done, _ = env.step(action)
            control_step += 1
            if done:
                success = bool(env.env.check_success()) if hasattr(env.env, "check_success") else True
                break
            if control_step >= max_steps:
                break
        request_index += 1

    stages = label_episode(np.stack(executed_proprio), np.stack(executed_actions)) if executed_actions else []
    return {
        "episode_id": episode_id,
        "task_suite": task_suite,
        "task_id": task_id,
        "task_description": env.description,
        "episode_index": episode_index,
        "seed": seed,
        "success": bool(success),
        "control_steps": control_step,
        "requests": requests,
        "timing": timing_rows,
        "stage_labels": stages,
        "executed_actions": np.stack(executed_actions) if executed_actions else np.zeros((0, 7)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/data/rxhuang/wam_progressive_outputs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_T5)
    parser.add_argument("--task-suites", nargs="+", default=["libero_10"])
    parser.add_argument("--task-ids", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[195, 196, 197])
    parser.add_argument("--full-steps", type=int, default=5, help="official action denoising steps")
    parser.add_argument("--standalone-steps", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--standalone-stride",
        type=int,
        default=4,
        help="run standalone_k probes every N requests; 0 disables them",
    )
    parser.add_argument("--execute-horizon", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--settle-gripper-action", type=float, default=-1.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--capture-future-latent", action="store_true", default=True)
    parser.add_argument("--no-capture-future-latent", dest="capture_future_latent", action="store_false")
    parser.add_argument("--libero-repo", default="/home/rxhuang/Projects/LIBERO")
    args = parser.parse_args()

    # This machine also has an editable LIBERO-plus install; without pinning the
    # path, `import libero` silently resolves to it.
    configure_repository_paths({"repositories": {"libero": args.libero_repo, "cosmos": str(REPO_ROOT)}})

    run_id = args.run_id or f"p1-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.output_root) / "trajectories" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    model, dataset_stats = load_model(cfg, args)
    capture = CosmosCheckpointCapture(
        model,
        cfg,
        dataset_stats,
        capture_future_latent=args.capture_future_latent,
    )
    capture.install()

    provenance = run_provenance(
        repos={
            "cosmos": REPO_ROOT,
            "libero": "/home/rxhuang/Projects/LIBERO",
            "lingbot_va": "/home/rxhuang/Projects/lingbot-va",
        },
        checkpoint=args.checkpoint,
        config=vars(args),
        checkpoint_sha256=KNOWN_CHECKPOINT_SHA256,
    )
    provenance["dataset_stats"] = {k: list(map(float, v)) for k, v in dataset_stats.items()}
    provenance["run_id"] = run_id

    episodes: list[dict[str, Any]] = []
    timing_path = out_dir / "timing.jsonl"
    with timing_path.open("w") as timing_file:
        for task_suite in args.task_suites:
            for task_id in args.task_ids:
                env = RealLiberoEnvironment(task_suite, task_id, args.resolution)
                try:
                    for episode_index, seed in enumerate(args.seeds):
                        print(f"[p1] {task_suite} task={task_id} episode={episode_index} seed={seed}", flush=True)
                        episode = run_episode(
                            env=env,
                            model=model,
                            cfg=cfg,
                            dataset_stats=dataset_stats,
                            capture=capture,
                            task_suite=task_suite,
                            task_id=task_id,
                            episode_index=episode_index,
                            seed=seed,
                            args=args,
                        )
                        for row in episode["timing"]:
                            row["episode_id"] = episode["episode_id"]
                            timing_file.write(json.dumps(row) + "\n")
                        timing_file.flush()
                        episodes.append(episode)
                        print(
                            f"[p1]   success={episode['success']} steps={episode['control_steps']} "
                            f"requests={len(episode['requests'])}",
                            flush=True,
                        )
                finally:
                    env.close()

    torch.save(episodes, out_dir / "checkpoints.pt")
    np.save(
        out_dir / "actions.npy",
        np.concatenate([ep["executed_actions"] for ep in episodes]) if episodes else np.zeros((0, 7)),
    )
    provenance["summary"] = {
        "episodes": len(episodes),
        "successes": sum(int(ep["success"]) for ep in episodes),
        "total_requests": sum(len(ep["requests"]) for ep in episodes),
        "max_final_agreement_abs": max(
            (r["final_agreement_max_abs"] for ep in episodes for r in ep["requests"]), default=0.0
        ),
    }
    write_json(out_dir / "metadata.json", provenance)
    print(json.dumps(provenance["summary"], indent=2))
    print(f"[p1] wrote {out_dir}")


if __name__ == "__main__":
    main()
