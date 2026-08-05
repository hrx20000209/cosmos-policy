"""Async-style dataset replay for the SO101 Cosmos deployment.

Feeds one Three_Cubes_1 episode through *the actual deployment server object* -- same
config, same chunk_size, same safety filters -- simulating the async execution pattern
(predict a 30-step chunk, execute the first `actions_per_chunk` steps, replan from a fresh
observation), and plots predicted vs ground-truth action vs observation.state.

Observations are teacher-forced from the dataset at each replanning boundary (i.e. the arm
is assumed to track ground truth perfectly). That deliberately isolates *model quality*
from execution/tracking error -- if the prediction is flat here, no amount of safety-limit
tuning will make the real arm move.

Two figures are produced:
  1. <out>.png            -- prediction vs GT vs observation.state (reference-style)
  2. <out>_clamped.png    -- raw prediction vs the same prediction after the server's
                             safety clamps, which answers "are the clamps what is
                             stopping the arm?"

Usage (from the repo root, cosmos env):
    PYTHONPATH=$PWD:/home/hrx/Projects/lerobot/src python \
        cosmos_policy/scripts/replay_so101_async_action_curve.py --episode 0
"""

from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from cosmos_policy.experiments.robot.cosmos_utils import get_action
from cosmos_policy.experiments.robot.so101_async_deploy import (
    SO101_ACTION_NAMES,
    SO101CosmosAsyncPolicyServer,
    SO101CosmosAsyncServerConfig,
)

CAMERA_KEYS = {"front": "observation.images.front",
               "right": "observation.images.right",
               "wrist": "observation.images.wrist"}

# Colour = joint identity, line style = role. Matches the existing VLA-JEPA replay figures
# so the two can be read side by side.
JOINT_COLORS = {
    "shoulder_pan.pos": "#ff7f0e", "shoulder_lift.pos": "#1f77b4", "elbow_flex.pos": "#2ca02c",
    "wrist_flex.pos": "#ff7f0e", "wrist_roll.pos": "#1f77b4", "gripper.pos": "#2ca02c",
}
GROUPS = [["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos"],
          ["wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="hrx2000/Three_Cubes_1")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--actions_per_chunk", type=int, default=None,
                   help="replan stride; defaults to the server config value")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--closed_loop", action="store_true",
                   help="feed back the model's own last executed action as proprio instead "
                        "of the dataset state; shows whether commanded motion actually "
                        "accumulates to the goal (camera frames still come from the dataset)")
    p.add_argument("--denoise", type=int, default=None)
    p.add_argument("--max_delta", type=float, default=None,
                   help="override max_delta_from_observation (body + gripper)")
    p.add_argument("--max_step", type=float, default=None,
                   help="override max_step_delta (body + gripper)")
    p.add_argument("--max_relative_target", type=float, default=None,
                   help="simulate the CLIENT-side clamp (lerobot ensure_safe_goal_position): "
                        "every commanded step is capped to present_pos +/- this value. The "
                        "robot client applies this on top of the server clamps; without it a "
                        "replay is more permissive than the real arm.")
    p.add_argument("--out", default="/home/hrx/Projects/cosmos-policy/outputs/so101_async_replay")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = SO101CosmosAsyncServerConfig()
    if args.max_delta is not None:
        cfg.max_delta_from_observation = args.max_delta
        cfg.max_gripper_delta_from_observation = args.max_delta
    if args.max_step is not None:
        cfg.max_step_delta = args.max_step
        cfg.max_gripper_step_delta = args.max_step
    print(f"clamps: delta_from_obs={cfg.max_delta_from_observation} "
          f"gripper_delta={cfg.max_gripper_delta_from_observation} "
          f"step={cfg.max_step_delta} gripper_step={cfg.max_gripper_step_delta}")
    stride = args.actions_per_chunk or cfg.actions_per_chunk
    print(f"instantiating deployment server object (loads checkpoint {cfg.ckpt_path})")
    server = SO101CosmosAsyncPolicyServer(cfg)
    chunk_size = server.cosmos_cfg.chunk_size
    print(f"chunk_size={chunk_size}  replan stride={stride}  denoise={cfg.num_denoising_steps_action}")

    ds = LeRobotDataset(args.repo_id)
    # lerobot 0.5.2 exposes episode frame ranges on meta.episodes, not episode_data_index.
    eps = ds.meta.episodes
    ep_from = int(eps["dataset_from_index"][args.episode])
    ep_to = int(eps["dataset_to_index"][args.episode])
    n = ep_to - ep_from
    if args.max_frames:
        n = min(n, args.max_frames)
    print(f"episode {args.episode}: frames {ep_from}..{ep_from + n} ({n} frames, {n / ds.fps:.1f}s)")

    gt_action = np.full((n, 6), np.nan, dtype=np.float32)
    obs_state = np.full((n, 6), np.nan, dtype=np.float32)
    pred_raw = np.full((n, 6), np.nan, dtype=np.float32)
    pred_clamped = np.full((n, 6), np.nan, dtype=np.float32)

    for i in range(n):
        s = ds[ep_from + i]
        gt_action[i] = np.asarray(s["action"], dtype=np.float32)
        obs_state[i] = np.asarray(s["observation.state"], dtype=np.float32)

    task = ds[ep_from].get("task") or ""
    print(f"task: {task!r}")

    boundaries = []
    infer_ms = []
    # In closed loop the arm's state is whatever the policy itself last commanded.
    fed_state = obs_state[0].copy()
    for t in range(0, n, stride):
        sample = ds[ep_from + t]
        raw = {cam: sample[key] for cam, key in CAMERA_KEYS.items()}
        proprio_t = fed_state if args.closed_loop else obs_state[t]
        for j, name in enumerate(SO101_ACTION_NAMES):
            raw[name] = float(proprio_t[j])
        cosmos_obs = server._build_cosmos_observation(raw)

        t0 = time.perf_counter()
        result = get_action(
            server.cosmos_cfg, server.model, server.dataset_stats, cosmos_obs, task,
            seed=cfg.seed + t, randomize_seed=cfg.randomize_seed,
            num_denoising_steps_action=args.denoise or cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )
        infer_ms.append((time.perf_counter() - t0) * 1000)

        chunk = np.asarray(result["actions"], dtype=np.float32)
        if chunk.shape != (chunk_size, 6):
            raise RuntimeError(f"unexpected chunk shape {chunk.shape}")
        clamped = server._apply_safety_filters(chunk, cosmos_obs["proprio"])

        k = min(stride, n - t)

        # Client-side clamp (lerobot robots/utils.py:ensure_safe_goal_position). Applied
        # per commanded step against the CURRENT position, which in closed loop is whatever
        # the arm last held -- so it must be simulated sequentially, not vectorised.
        if args.max_relative_target is not None:
            cap = float(args.max_relative_target)
            present = proprio_t.astype(np.float32).copy()
            executed = np.empty_like(clamped[:k])
            for i in range(k):
                executed[i] = present + np.clip(clamped[i] - present, -cap, cap)
                present = executed[i]
            clamped = np.concatenate([executed, clamped[k:]], axis=0)

        pred_raw[t:t + k] = chunk[:k]
        pred_clamped[t:t + k] = clamped[:k]
        boundaries.append(t)
        # What the arm would actually be holding after this chunk executes.
        fed_state = (clamped if not args.closed_loop else clamped)[k - 1].copy()
        lag = np.abs(fed_state - gt_action[min(t + k - 1, n - 1)]).max()
        print(f"  t={t:4d}  infer={infer_ms[-1]:6.0f} ms  "
              f"raw|Δ from proprio|max={np.abs(chunk[:k] - proprio_t).max():6.2f}  "
              f"clamped max={np.abs(clamped[:k] - proprio_t).max():6.2f}"
              + (f"  |lag vs GT|max={lag:6.2f}" if args.closed_loop else ""))

    time_s = np.arange(n) / ds.fps
    _report(gt_action, obs_state, pred_raw, pred_clamped, boundaries, stride, infer_ms)
    _plot_reference_style(time_s, gt_action, obs_state, pred_raw, boundaries, args, ds.fps)
    _plot_clamp_effect(time_s, gt_action, pred_raw, pred_clamped, boundaries, args)


def _report(gt, state, raw, clamped, boundaries, stride, infer_ms):
    print("\n=== numbers ===")
    print(f"inference: mean {np.mean(infer_ms):.0f} ms over {len(infer_ms)} replans")

    print("\nper-joint MAE (deg) and motion range:")
    print(f"{'joint':20s} {'MAE raw':>9s} {'MAE clamp':>10s} {'GT range':>9s} "
          f"{'pred range':>11s} {'clamp range':>12s}")
    for j, name in enumerate(SO101_ACTION_NAMES):
        print(f"{name:20s} {np.nanmean(np.abs(raw[:, j] - gt[:, j])):9.2f} "
              f"{np.nanmean(np.abs(clamped[:, j] - gt[:, j])):10.2f} "
              f"{np.ptp(gt[:, j]):9.2f} {np.ptp(raw[:, j]):11.2f} {np.ptp(clamped[:, j]):12.2f}")

    # How much of the motion is the model itself producing, before any clamping?
    print("\nwithin-chunk motion produced by the RAW model (deg, max over chunk):")
    spans = []
    for b in boundaries:
        seg = raw[b:b + stride]
        if len(seg):
            spans.append(np.ptp(seg, axis=0))
    spans = np.asarray(spans)
    for j, name in enumerate(SO101_ACTION_NAMES):
        print(f"  {name:20s} mean {spans[:, j].mean():6.2f}   max {spans[:, j].max():6.2f}")

    # Is the safety clamp actually binding?
    diff = np.abs(clamped - raw)
    touched = diff > 1e-4
    print("\nsafety-clamp effect (clamped vs raw):")
    print(f"  fraction of commanded values altered by clamps: {touched.mean() * 100:.1f}%")
    print(f"  mean |clamped - raw| where altered: "
          f"{diff[touched].mean() if touched.any() else 0.0:.3f} deg")
    for j, name in enumerate(SO101_ACTION_NAMES):
        print(f"  {name:20s} altered {touched[:, j].mean() * 100:5.1f}%   "
              f"max |Δ| {diff[:, j].max():6.2f}")


def _decorate(ax, boundaries, fps, title):
    for b in boundaries:
        ax.axvline(b / fps, color="red", alpha=0.18, lw=0.8, zorder=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint position")
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.6)


def _plot_reference_style(time_s, gt, state, raw, boundaries, args, fps):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.8))
    for ax, group in zip(axes, GROUPS):
        for name in group:
            j = SO101_ACTION_NAMES.index(name)
            c = JOINT_COLORS[name]
            ax.plot(time_s, raw[:, j], color=c, lw=2.0, label=f"{name} pred")
            ax.plot(time_s, gt[:, j], color=c, lw=1.6, ls="--", label=f"{name} GT action")
            ax.plot(time_s, state[:, j], color=c, lw=1.2, ls=":", label=f"{name} observation.state")
        _decorate(ax, boundaries, fps, ", ".join(group))
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Cosmos Policy step-5000 async replay: prediction vs GT action vs "
                 "observation.state\nred lines = replanning boundaries "
                 f"(stride={args.actions_per_chunk or 'default'}, episode {args.episode})")
    fig.tight_layout()
    out = f"{args.out}_ep{args.episode}{'_closedloop' if args.closed_loop else ''}.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


def _plot_clamp_effect(time_s, gt, raw, clamped, boundaries, args):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.8))
    for ax, group in zip(axes, GROUPS):
        for name in group:
            j = SO101_ACTION_NAMES.index(name)
            c = JOINT_COLORS[name]
            ax.plot(time_s, raw[:, j], color=c, lw=2.0, label=f"{name} pred (raw)")
            ax.plot(time_s, clamped[:, j], color=c, lw=1.5, ls="-.",
                    label=f"{name} pred (safety-clamped)")
            ax.plot(time_s, gt[:, j], color=c, lw=1.0, ls="--", alpha=0.45,
                    label=f"{name} GT action")
        _decorate(ax, boundaries, 30, ", ".join(group))
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Does the safety clamp stop the arm? raw model output vs clamped output\n"
                 "if raw and clamped overlap, the clamps are NOT what is limiting motion")
    fig.tight_layout()
    out = f"{args.out}_ep{args.episode}_clamped.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
