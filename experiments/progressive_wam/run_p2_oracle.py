"""P2: simulator branched-rollout oracle.

For every recorded request we restore the exact MuJoCo state captured in P1 and
execute two branches from it:

  branch A: the prefix of the **intermediate** checkpoint-j action chunk
  branch B: the prefix of the **full-schedule** action chunk

Everything else -- initial state, object placement, controller, number of steps --
is identical, so the difference is attributable to the denoising checkpoint alone.

Two tiers:

* ``tier1`` (default, no model needed): state divergence + safety checks after the
  prefix.  Answers "does the early action put the robot somewhere materially
  different, or somewhere unsafe".
* ``tier2`` (subsampled, needs the policy): after the prefix, continue the real
  official-schedule policy to the end of the episode on both branches and compare
  task success.  This is the only measurement that can speak about success, and it
  is expensive, so it runs on a small subsample.

This experiment is the feasibility *upper bound*: it uses privileged simulator
access that an online system will not have.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

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
from experiments.progressive_wam.metrics import derivative_stats  # noqa: E402
from experiments.progressive_wam.provenance import run_provenance, write_json  # noqa: E402


@dataclass
class OracleThresholds:
    """Every threshold is configurable; ``analyze_p2.sensitivity`` sweeps them.

    The two safety caps are calibrated against what the **official 5-step policy
    itself** commands over the 3720 executed actions of the P1 run, so "unsafe"
    means "more violent than anything the baseline ever did", not an arbitrary
    number::

        |d_xyz|  mean 0.647  p50 0.727  p95 1.076  p99 1.177  max 1.313
        jerk     mean 0.062  p50 0.037  p95 0.188  p99 0.482  max 1.232

    An earlier version used 0.12 / 0.30, which sit below the baseline median and
    p97 respectively and would have labelled almost every branch unsafe.
    """

    eef_position_error: float = 0.02  # metres; ~25% of an 8-step prefix displacement
    eef_rotation_error: float = 0.15  # quaternion distance
    object_position_error: float = 0.01  # metres
    gripper_width_error: float = 0.01  # finger-joint sum; open ~0.08, closed ~0.02
    max_velocity: float = 1.35  # just above the baseline max of 1.313
    max_jerk: float = 1.30  # just above the baseline max of 1.232
    allow_gripper_mismatch: bool = False
    progress_drop: float = 0.0  # tier2 only: allowed drop in success indicator


def quat_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - abs(np.dot(a, b)))


def _robot_body_mask(sim) -> np.ndarray:
    names = list(sim.model.body_names)
    return np.array(
        [not (n.startswith("robot0") or n.startswith("gripper0") or n in ("world", "table")) for n in names],
        dtype=bool,
    )


def probe_state(env: RealLiberoEnvironment) -> dict[str, Any]:
    """Everything we compare between branches, read straight from MuJoCo."""
    sim = env.env.sim
    mask = _robot_body_mask(sim)
    obs = env.env.env._get_observations() if hasattr(env.env, "env") else None
    joint_violation = 0
    try:
        qpos = np.asarray(sim.data.qpos, dtype=np.float64)
        ranges = np.asarray(sim.model.jnt_range, dtype=np.float64)
        limited = np.asarray(sim.model.jnt_limited, dtype=bool)
        addr = np.asarray(sim.model.jnt_qposadr, dtype=int)
        for j in np.where(limited)[0]:
            value = qpos[addr[j]]
            if value < ranges[j, 0] - 1e-6 or value > ranges[j, 1] + 1e-6:
                joint_violation += 1
    except Exception:  # pragma: no cover - model layout dependent
        joint_violation = -1
    return {
        "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64) if obs else None,
        "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float64) if obs else None,
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64) if obs else None,
        "object_xpos": np.asarray(sim.data.body_xpos, dtype=np.float64)[mask].copy(),
        "object_xquat": np.asarray(sim.data.body_xquat, dtype=np.float64)[mask].copy(),
        "ncon": int(sim.data.ncon),
        "joint_limit_violations": joint_violation,
        "success": bool(env.env.check_success()),
        "agentview": np.asarray(obs["agentview_image"], dtype=np.float32) if obs else None,
        "wrist": np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.float32) if obs else None,
    }


def compare_states(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    out["eef_position_error"] = float(np.linalg.norm(a["eef_pos"] - b["eef_pos"]))
    out["eef_rotation_error"] = quat_distance(a["eef_quat"], b["eef_quat"])
    out["gripper_width_error"] = float(np.abs(np.abs(a["gripper_qpos"]).sum() - np.abs(b["gripper_qpos"]).sum()))
    obj_delta = np.linalg.norm(a["object_xpos"] - b["object_xpos"], axis=1)
    out["object_position_error_max"] = float(obj_delta.max()) if obj_delta.size else 0.0
    out["object_position_error_mean"] = float(obj_delta.mean()) if obj_delta.size else 0.0
    out["object_rotation_error_max"] = float(
        max((quat_distance(x, y) for x, y in zip(a["object_xquat"], b["object_xquat"])), default=0.0)
    )
    out["contact_count_delta"] = float(a["ncon"] - b["ncon"])
    out["joint_limit_violations"] = float(a["joint_limit_violations"])
    # The reference count is kept so a nonzero branch count can be read as
    # "worse than the official policy" rather than "LIBERO reports limits at all".
    out["reference_joint_limit_violations"] = float(b["joint_limit_violations"])
    out["success_delta"] = float(int(a["success"]) - int(b["success"]))
    # Pixel-space image distance. Named honestly: this is NOT a VAE latent
    # distance; computing one would require building the full Cosmos conditioning
    # batch, which P2 deliberately avoids so tier1 can run without the model.
    if a["agentview"] is not None and b["agentview"] is not None:
        out["visual_pixel_l2_agentview"] = float(
            np.sqrt(np.mean((a["agentview"] - b["agentview"]) ** 2))
        )
        out["visual_pixel_l2_wrist"] = float(np.sqrt(np.mean((a["wrist"] - b["wrist"]) ** 2)))
    return out


def restore(env: RealLiberoEnvironment, sim_state: np.ndarray) -> None:
    """Restore a MuJoCo snapshot *and* every piece of state that rides alongside it.

    Three things have to be rewound, and missing any one of them silently
    corrupts the comparison:

    1. **Physics** -- ``regenerate_obs_from_state`` handles qpos/qvel.
    2. **Episode bookkeeping** -- robosuite keeps its own ``done`` flag and
       ``timestep`` counter. Once a branch terminates (horizon or success) the
       next ``step`` raises "executing action in terminated episode".
    3. **Controller state** -- the OSC controller carries ``goal_pos``/``goal_ori``
       from whichever branch ran last. Without resetting it, branches drift
       further from the reference the later they run, and the measured error
       grows with branch *order* rather than with denoising checkpoint. That
       artefact was visible as a monotone rise in ``eef_position_error`` across
       checkpoints even though the last checkpoint executes the reference action.
       ``update(force=True)`` first, because ``reset_goal`` reads the cached
       ``ee_pos``/``ee_ori_mat`` that ``update`` refreshes from the restored sim.
    """
    env.env.regenerate_obs_from_state(sim_state)
    sim_env = env.env.env
    sim_env.done = False
    sim_env.timestep = 0
    # MuJoCo seeds its constraint solver with the previous step's acceleration,
    # and `sim.get_state()` does not carry qacc_warmstart. Leaving it in place
    # makes each branch start from whatever the previous branch ended with, which
    # perturbs the solution at the 1e-4 m level -- the same order as the effect
    # being measured. Zeroing it gives every branch the identical solver seed.
    try:
        sim_env.sim.data.qacc_warmstart[:] = 0.0
        sim_env.sim.forward()
    except AttributeError:  # pragma: no cover - mujoco binding dependent
        pass
    for robot in getattr(sim_env, "robots", []):
        controller = getattr(robot, "controller", None)
        if controller is None:
            continue
        if hasattr(controller, "update"):
            controller.update(force=True)
        if hasattr(controller, "reset_goal"):
            controller.reset_goal()


def execute_prefix(env: RealLiberoEnvironment, sim_state: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    restore(env, sim_state)
    done = False
    for action in actions:
        _, _, done, _ = env.env.step(action.tolist())
        if done:
            break
    state = probe_state(env)
    state["terminated_early"] = bool(done)
    return state


def oracle_label(
    comparison: dict[str, float],
    intermediate_chunk: np.ndarray,
    prefix_length: int,
    thresholds: OracleThresholds,
    gripper_mismatch: bool,
) -> dict[str, Any]:
    derivs = derivative_stats(intermediate_chunk[:prefix_length])
    reasons: list[str] = []
    if comparison["eef_position_error"] >= thresholds.eef_position_error:
        reasons.append("eef_position")
    if comparison["eef_rotation_error"] >= thresholds.eef_rotation_error:
        reasons.append("eef_rotation")
    if comparison["object_position_error_max"] >= thresholds.object_position_error:
        reasons.append("object_moved")
    if comparison["gripper_width_error"] >= thresholds.gripper_width_error:
        reasons.append("gripper_width")
    if comparison["joint_limit_violations"] > 0:
        reasons.append("joint_limit")
    if derivs["jerk_max"] >= thresholds.max_jerk:
        reasons.append("jerk")
    if derivs["velocity_max"] >= thresholds.max_velocity:
        reasons.append("velocity")
    if gripper_mismatch and not thresholds.allow_gripper_mismatch:
        reasons.append("gripper_transition_mismatch")
    safety_valid = not {"joint_limit", "jerk", "velocity"} & set(reasons)
    return {
        "oracle_reliable": len(reasons) == 0,
        "safety_valid": safety_valid,
        "failure_reasons": reasons,
        **{f"prefix_{k}": v for k, v in derivs.items()},
    }


def gripper_transition_mismatch(intermediate: np.ndarray, final: np.ndarray, prefix: int) -> bool:
    a = intermediate[:prefix, 6] > 0
    b = final[:prefix, 6] > 0
    return bool(np.any(a != b))


def iter_requests(episodes: list[dict], stride: int, max_per_episode: int) -> Iterable[tuple[dict, dict]]:
    for episode in episodes:
        picked = 0
        for index, request in enumerate(episode["requests"]):
            if stride > 1 and index % stride != 0:
                continue
            if max_per_episode and picked >= max_per_episode:
                break
            picked += 1
            yield episode, request


def run_tier1(
    episodes: list[dict],
    args: argparse.Namespace,
    thresholds: OracleThresholds,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_cache: dict[tuple[str, int], RealLiberoEnvironment] = {}
    try:
        for episode, request in iter_requests(episodes, args.request_stride, args.max_requests_per_episode):
            key = (episode["task_suite"], episode["task_id"])
            if key not in env_cache:
                env_cache[key] = RealLiberoEnvironment(key[0], key[1], args.resolution)
            env = env_cache[key]
            sim_state = np.asarray(request["sim_state"], dtype=np.float64)
            final_chunk = np.asarray(request["final_actions"], dtype=np.float32)
            checkpoint_chunks = np.asarray(request["checkpoint_actions"], dtype=np.float32)

            for prefix in args.prefix_lengths:
                if prefix > final_chunk.shape[0]:
                    continue
                reference = execute_prefix(env, sim_state, final_chunk[:prefix])
                # checkpoint 0 is an identical-action control: it replays the very
                # same chunk as the reference. Any error it shows is the
                # simulator's own reproduction noise, and every checkpoint below
                # must be read against that floor rather than against zero.
                branches: list[tuple[int, np.ndarray, float]] = [(0, final_chunk, float("nan"))]
                branches += [
                    (j + 1, checkpoint_chunks[j], float(request["sigmas"][j]))
                    for j in range(checkpoint_chunks.shape[0])
                ]
                # `restore` is not perfectly idempotent: each successive branch
                # drifts a further ~2.3e-4 m from the first, so running the
                # checkpoints in order 1..N makes late checkpoints look worse for
                # a reason that has nothing to do with denoising. Shuffling the
                # execution order turns that systematic bias into noise that every
                # checkpoint (and the control) bears equally. The seed is derived
                # from the request id so the shuffle is reproducible.
                rng = np.random.default_rng(abs(hash((request["request_id"], prefix))) % (2**32))
                rng.shuffle(branches)
                for checkpoint_index, chunk, sigma in branches:
                    branch = execute_prefix(env, sim_state, chunk[:prefix])
                    comparison = compare_states(branch, reference)
                    mismatch = gripper_transition_mismatch(chunk, final_chunk, prefix)
                    label = oracle_label(comparison, chunk, prefix, thresholds, mismatch)
                    rows.append(
                        {
                            "request_id": request["request_id"],
                            "episode_id": episode["episode_id"],
                            "task_suite": episode["task_suite"],
                            "task_id": episode["task_id"],
                            "seed": episode["seed"],
                            "control_step": request["control_step"],
                            "checkpoint": checkpoint_index,
                            "sigma": sigma,
                            "prefix_length": prefix,
                            "gripper_transition_mismatch": mismatch,
                            **comparison,
                            **label,
                        }
                    )
            if len(rows) % 200 < len(args.prefix_lengths) * (checkpoint_chunks.shape[0] + 1):
                print(f"[p2-tier1] {len(rows)} branch comparisons done", flush=True)
    finally:
        for env in env_cache.values():
            env.close()

    # Determinism self-check. The control branch replays the reference chunk, so
    # a non-negligible error here means the restore is still incomplete and the
    # oracle labels are measuring simulator noise instead of checkpoint quality.
    control = [r for r in rows if r["checkpoint"] == 0]
    if control:
        errors = np.array([r["eef_position_error"] for r in control])
        print(
            f"[p2-tier1] determinism control (n={len(control)}): "
            f"eef error median={np.median(errors):.3e} p95={np.percentile(errors, 95):.3e} "
            f"max={errors.max():.3e}",
            flush=True,
        )
        if np.median(errors) > 1e-6:
            print(
                "[p2-tier1] WARNING: identical-action branches do not reproduce; "
                "treat every reliability number as noise-floor limited",
                flush=True,
            )
    return rows


def run_tier2(
    episodes: list[dict],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Continue the real policy after the prefix and compare final task success."""
    from cosmos_policy.experiments.robot.cosmos_utils import get_action
    from experiments.progressive_wam.run_p1_trajectory_dump import build_cfg, load_model

    cfg = build_cfg(args)
    model, dataset_stats = load_model(cfg, args)

    rows: list[dict[str, Any]] = []
    env_cache: dict[tuple[str, int], RealLiberoEnvironment] = {}

    def continue_episode(env, seed: int, remaining_steps: int) -> tuple[bool, int]:
        # The prefix itself may already have solved the task; continuing from a
        # terminated episode would raise instead of scoring a success.
        if env.env.check_success():
            return True, 0
        env.env.env.done = False
        steps = 0
        # ControlEnv does not forward _get_observations; the robosuite env does.
        raw = env.env.env._get_observations()
        while steps < remaining_steps:
            observation = extract_observation(raw, flip_vertical=True)
            result = get_action(
                cfg,
                model,
                dataset_stats,
                {
                    "primary_image": observation.primary_image,
                    "wrist_image": observation.wrist_image,
                    "proprio": observation.proprio,
                },
                env.description,
                seed=seed,
                num_denoising_steps_action=args.full_steps,
                generate_future_state_and_value_in_parallel=True,
                decode_future_state=False,
            )
            chunk = np.asarray(result["actions"], dtype=np.float32).reshape(-1, 7)
            for action in chunk[: args.execute_horizon]:
                raw, _, done, _ = env.env.step(action.tolist())
                steps += 1
                if done or steps >= remaining_steps:
                    break
            if env.env.check_success():
                return True, steps
        return bool(env.env.check_success()), steps

    try:
        for episode, request in iter_requests(episodes, args.tier2_request_stride, args.tier2_max_requests):
            key = (episode["task_suite"], episode["task_id"])
            if key not in env_cache:
                env_cache[key] = RealLiberoEnvironment(key[0], key[1], args.resolution)
            env = env_cache[key]
            sim_state = np.asarray(request["sim_state"], dtype=np.float64)
            final_chunk = np.asarray(request["final_actions"], dtype=np.float32)
            checkpoint_chunks = np.asarray(request["checkpoint_actions"], dtype=np.float32)
            budget = max(0, args.tier2_max_steps - request["control_step"])
            if budget < args.execute_horizon:
                continue

            for prefix in args.tier2_prefix_lengths:
                execute_prefix(env, sim_state, final_chunk[:prefix])
                ref_success, ref_steps = continue_episode(env, episode["seed"], budget - prefix)
                for j in args.tier2_checkpoints:
                    if j > checkpoint_chunks.shape[0]:
                        continue
                    execute_prefix(env, sim_state, checkpoint_chunks[j - 1][:prefix])
                    success, steps = continue_episode(env, episode["seed"], budget - prefix)
                    rows.append(
                        {
                            "request_id": request["request_id"],
                            "episode_id": episode["episode_id"],
                            "task_suite": episode["task_suite"],
                            "task_id": episode["task_id"],
                            "control_step": request["control_step"],
                            "checkpoint": j,
                            "prefix_length": prefix,
                            "reference_success": ref_success,
                            "reference_steps": ref_steps,
                            "branch_success": success,
                            "branch_steps": steps,
                            "success_delta": int(success) - int(ref_success),
                        }
                    )
                    print(f"[p2-tier2] {rows[-1]}", flush=True)
    finally:
        for env in env_cache.values():
            env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", required=True, help="P1 output directory containing checkpoints.pt")
    parser.add_argument("--output-root", default="/data/rxhuang/wam_progressive_outputs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tier", choices=["tier1", "tier2", "both"], default="tier1")
    parser.add_argument("--prefix-lengths", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--request-stride", type=int, default=1)
    parser.add_argument("--max-requests-per-episode", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    # tier2 only
    parser.add_argument("--tier2-request-stride", type=int, default=6)
    parser.add_argument("--tier2-max-requests", type=int, default=2)
    parser.add_argument("--tier2-prefix-lengths", nargs="+", type=int, default=[8])
    parser.add_argument("--tier2-checkpoints", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--tier2-max-steps", type=int, default=520)
    parser.add_argument("--full-steps", type=int, default=5)
    parser.add_argument("--execute-horizon", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--checkpoint", default="/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
    parser.add_argument("--dataset-stats", default="/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json")
    parser.add_argument("--t5-embeddings", default="/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl")
    parser.add_argument("--libero-repo", default="/home/rxhuang/Projects/LIBERO")
    args = parser.parse_args()

    # Pin LIBERO ahead of the editable LIBERO-plus install on this machine.
    configure_repository_paths({"repositories": {"libero": args.libero_repo, "cosmos": str(REPO_ROOT)}})

    trajectory_dir = Path(args.trajectory_dir)
    episodes = torch.load(trajectory_dir / "checkpoints.pt", weights_only=False)
    run_id = args.run_id or f"p2-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.output_root) / "oracle" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = OracleThresholds()
    provenance = run_provenance(
        repos={"cosmos": REPO_ROOT, "libero": "/home/rxhuang/Projects/LIBERO"},
        checkpoint=args.checkpoint if args.tier in ("tier2", "both") else None,
        config={**vars(args), "thresholds": asdict(thresholds)},
    )
    provenance["run_id"] = run_id
    provenance["source_trajectory_dir"] = str(trajectory_dir)

    if args.tier in ("tier1", "both"):
        rows = run_tier1(episodes, args, thresholds)
        with (out_dir / "tier1_branches.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=float) + "\n")
        provenance["tier1_rows"] = len(rows)
        print(f"[p2] tier1 wrote {len(rows)} rows")

    if args.tier in ("tier2", "both"):
        rows = run_tier2(episodes, args)
        with (out_dir / "tier2_success.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=float) + "\n")
        provenance["tier2_rows"] = len(rows)
        print(f"[p2] tier2 wrote {len(rows)} rows")

    write_json(out_dir / "metadata.json", provenance)
    print(f"[p2] wrote {out_dir}")


if __name__ == "__main__":
    main()
