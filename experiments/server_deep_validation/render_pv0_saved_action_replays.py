#!/usr/bin/env python3
"""Render offline diagnostic videos by replaying saved closed-loop actions.

This utility is intentionally outside the policy path.  It restores each
manifested physical initial state, executes an already-saved float32 action
sequence, and records the same post-flip agent/wrist camera views that the
benchmark worker saw.  It never loads Cosmos, computes a value, or injects
simulator state into a policy.

Use it for the fixed Fresh-success/PV0-failure cases after the full-scale
benchmark.  Replay completion/success is audited against the raw episode
artifact; it is a visual diagnostic, not a new online experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import (
    ManifestLiberoEnvironment,
    install_libero_checkout,
    load_jsonl,
)
from experiments.libero_harness import EpisodeVideoWriter, extract_observation, load_yaml


MODES = ("fresh", "predicted_reuse", "native_persistent")
ORIGINAL_CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_actions(actions: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def route_artifact(phase_dir: Path, mode: str, episode_key: str, seed_offset: int) -> Path:
    path = phase_dir / "episodes" / f"{mode}_{episode_key}_seedoff{seed_offset}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_artifact(path: Path, mode: str) -> tuple[dict[str, Any], np.ndarray]:
    payload = load_json(path)
    if payload.get("experiment") != "PV0_closed_loop_episode" or payload.get("status") != "PASS":
        raise ValueError(f"{path}: not a PASS PV0 closed-loop artifact")
    if payload.get("mode") != mode:
        raise ValueError(f"{path}: route mode mismatch")
    if payload.get("checkpoint_sha256") != ORIGINAL_CHECKPOINT_SHA256:
        raise ValueError(f"{path}: unexpected checkpoint")
    contract = payload.get("route_contract") or {}
    if int(contract.get("denoising_steps", -1)) != 1:
        raise ValueError(f"{path}: not a one-denoise artifact")
    for key in (
        "value_used",
        "privileged_runtime_state_input",
        "adaptive_scheduler_used",
        "threshold_used",
        "hidden_activation_patch_used",
        "fresh_prefix_oracle_used",
    ):
        if bool(payload.get(key)) or bool(contract.get(key)):
            raise ValueError(f"{path}: frozen route contract violates {key}=False")
    record = payload.get("record") or {}
    action_path = Path(str(record.get("executed_actions_path") or payload.get("action_path") or ""))
    if not action_path.is_file():
        raise FileNotFoundError(f"{path}: missing saved actions {action_path}")
    actions = np.load(action_path, allow_pickle=False)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"{action_path}: expected N x 7 actions, got {actions.shape}")
    if len(actions) != int(record.get("episode_steps", -1)):
        raise ValueError(f"{path}: saved action count differs from episode_steps")
    if sha256_actions(actions) != record.get("executed_action_sha256"):
        raise ValueError(f"{path}: saved action SHA mismatch")
    return payload, np.ascontiguousarray(actions, dtype=np.float32)


def replay(
    row: dict[str, Any],
    actions: np.ndarray,
    *,
    settle_steps: int,
    settle_gripper_action: float,
    resolution: int,
    fps: int,
    video_path: Path,
) -> dict[str, Any]:
    """Execute only the saved actions; never construct or invoke a policy."""

    environment: ManifestLiberoEnvironment | None = None
    writer: EpisodeVideoWriter | None = None
    done = False
    total_reward = 0.0
    replayed_steps = 0
    terminal_step: int | None = None
    try:
        install_libero_checkout(Path(row["libero_repo"]), Path(row["libero_config_path"]))
        environment = ManifestLiberoEnvironment(row, resolution, None)
        raw = environment.reset()
        settle_action = np.zeros(7, dtype=np.float32)
        settle_action[-1] = float(settle_gripper_action)
        for _ in range(settle_steps):
            raw, _, _, _ = environment.step(settle_action)
        writer = EpisodeVideoWriter(video_path, fps)
        for index, action in enumerate(actions):
            writer.append(extract_observation(raw, flip_vertical=True))
            raw, reward, done, _ = environment.step(action)
            total_reward += float(reward)
            replayed_steps += 1
            if done:
                terminal_step = index + 1
                writer.append(extract_observation(raw, flip_vertical=True))
                break
        if not done:
            writer.append(extract_observation(raw, flip_vertical=True))
        return {
            "replayed_steps": replayed_steps,
            "saved_action_count": int(len(actions)),
            "terminal_done": bool(done),
            "terminal_step": terminal_step,
            "total_reward": float(total_reward),
        }
    finally:
        if writer is not None:
            writer.close()
        if environment is not None:
            environment.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, default=Path("reports/pv0_overnight/phase_b_600"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("reports/server_deep_validation/manifests/full_scale_40_task.jsonl")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/server_deep_validation/server_sweep.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-key", action="append", required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Explicit EGL settings are respected when supplied by the launcher.  The
    # default provides a safe server-side rendering setup for an otherwise
    # clean GPU and does not affect the already-completed benchmark artifacts.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    rows = {row["episode_key"]: row for row in load_jsonl(args.manifest)}
    base = load_yaml(args.config)
    evaluation = base["evaluation"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "replay_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {summary_path}; pass --overwrite after review")
    candidates: list[dict[str, Any]] = []
    for episode_key in args.episode_key:
        row = rows.get(episode_key)
        if row is None:
            raise ValueError(f"episode key is absent from manifest: {episode_key}")
        for mode in MODES:
            artifact_path = route_artifact(args.phase_dir, mode, episode_key, args.seed_offset)
            artifact, actions = validate_artifact(artifact_path, mode)
            expected_success = bool((artifact.get("record") or {}).get("success"))
            video_path = output_dir / f"{mode}_{episode_key}_seedoff{args.seed_offset}.mp4"
            if video_path.exists() and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {video_path}; pass --overwrite after review")
            replay_result = replay(
                row,
                actions,
                settle_steps=int(evaluation["settle_steps"]),
                settle_gripper_action=float(evaluation["settle_gripper_action"]),
                resolution=int(evaluation["resolution"]),
                fps=int(evaluation["video_fps"]),
                video_path=video_path,
            )
            if replay_result["replayed_steps"] != int(len(actions)):
                raise RuntimeError(f"{video_path}: replay terminated before its saved action trace ended")
            if bool(replay_result["terminal_done"]) != expected_success:
                raise RuntimeError(
                    f"{video_path}: replay outcome mismatch (recorded={expected_success}, replay={replay_result['terminal_done']})"
                )
            candidates.append(
                {
                    "episode_key": episode_key,
                    "task_uid": row["task_uid"],
                    "task_name": row["task_name"],
                    "split": row["split"],
                    "instruction": row["instruction"],
                    "mode": mode,
                    "recorded_success": expected_success,
                    "recorded_termination_reason": (artifact.get("record") or {}).get("termination_reason"),
                    "action_sha256": sha256_actions(actions),
                    "source_episode_artifact": str(artifact_path),
                    "video_path": str(video_path),
                    **replay_result,
                }
            )
    payload = {
        "schema_version": 1,
        "experiment": "PV0_offline_saved_action_replay",
        "status": "PASS",
        "purpose": "post-hoc visual failure diagnosis only",
        "runtime_policy_used": False,
        "model_loaded": False,
        "cosmos_value_used": False,
        "privileged_state_input_to_policy": False,
        "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
        "denoising_steps_in_source_artifacts": 1,
        "render_contract": {
            "initial_state_source": "manifested LIBERO initial state",
            "actions_source": "SHA-verified saved float32 closed-loop action traces",
            "settle_steps": int(evaluation["settle_steps"]),
            "camera_views": "post-flip agentview and wrist side-by-side",
            "fps": int(evaluation["video_fps"]),
        },
        "replays": candidates,
    }
    atomic_write(summary_path, payload)
    print(json.dumps({"status": "PASS", "replays": len(candidates), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
