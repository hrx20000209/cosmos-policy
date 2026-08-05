"""Does the SO101 policy actually use the cameras, or is it echoing its proprio input?

For a fixed set of observations we hold the three camera views constant and perturb only
the proprio vector, then measure how much the predicted action chunk moves. The mirror test
holds proprio constant and swaps in images from a different point in the episode.

Interpretation
--------------
  * action follows proprio, barely reacts to images -> proprio shortcut ("copycat"): the
    policy has learned action ~= current state + small delta. It scores well under
    teacher-forced evaluation and cannot drive the task on hardware.
  * action reacts to images, is stable under proprio noise -> the policy is visually
    grounded and the closed-loop failure is elsewhere.

Read-only: never touches the robot.
"""

from __future__ import annotations

import argparse

import numpy as np

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="hrx2000/Three_Cubes_1")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", type=int, nargs="*", default=[60, 150, 240, 330, 420])
    p.add_argument("--noise", type=float, nargs="*", default=[2.0, 5.0, 10.0, 20.0])
    p.add_argument("--denoise", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SO101CosmosAsyncServerConfig()
    nd = args.denoise or cfg.num_denoising_steps_action
    server = SO101CosmosAsyncPolicyServer(cfg)
    ds = LeRobotDataset(args.repo_id)
    ep_from = int(ds.meta.episodes["dataset_from_index"][args.episode])
    task = ds[ep_from].get("task") or ""
    rng = np.random.default_rng(0)

    def predict(images, proprio, seed):
        raw = dict(images)
        for j, nm in enumerate(SO101_ACTION_NAMES):
            raw[nm] = float(proprio[j])
        obs = server._build_cosmos_observation(raw)
        r = get_action(server.cosmos_cfg, server.model, server.dataset_stats, obs, task,
                       seed=seed, randomize_seed=False, num_denoising_steps_action=nd,
                       generate_future_state_and_value_in_parallel=False)
        return np.asarray(r["actions"], dtype=np.float32)

    print(f"\nproprio ablation | episode {args.episode} | frames {args.frames} | denoise {nd}\n")

    # --- Baseline + seed sensitivity (the noise floor every other number is judged against)
    base = {}
    seed_delta = []
    for f in args.frames:
        s = ds[ep_from + f]
        imgs = {c: s[k] for c, k in CAMERA_KEYS.items()}
        prop = np.asarray(s["observation.state"], dtype=np.float32)
        a1 = predict(imgs, prop, cfg.seed)
        a2 = predict(imgs, prop, cfg.seed + 977)
        base[f] = (imgs, prop, a1)
        seed_delta.append(np.abs(a1 - a2).mean())
    floor = float(np.mean(seed_delta))
    print(f"seed-only noise floor (same inputs, different seed): {floor:.3f} deg mean |Δ|\n")

    # --- Perturb proprio, keep images fixed
    print("A. proprio perturbed, images held FIXED")
    print(f"{'noise (deg)':>12} {'mean |Δaction|':>15} {'vs floor':>10} {'Δ/noise':>9}")
    prop_sens = {}
    for sigma in args.noise:
        deltas = []
        for f in args.frames:
            imgs, prop, a1 = base[f]
            pert = prop + rng.normal(0, sigma, size=prop.shape).astype(np.float32)
            a2 = predict(imgs, pert, cfg.seed)
            deltas.append(np.abs(a1 - a2).mean())
        d = float(np.mean(deltas))
        prop_sens[sigma] = d
        print(f"{sigma:12.1f} {d:15.3f} {d / floor:9.1f}x {d / sigma:9.3f}")

    # --- Keep proprio, swap images to a different phase of the episode
    print("\nB. images swapped to another frame, proprio held FIXED")
    print(f"{'frame -> src':>16} {'mean |Δaction|':>15} {'vs floor':>10}")
    img_deltas = []
    for f in args.frames:
        imgs, prop, a1 = base[f]
        src = args.frames[(args.frames.index(f) + 2) % len(args.frames)]
        s2 = ds[ep_from + src]
        other = {c: s2[k] for c, k in CAMERA_KEYS.items()}
        a2 = predict(other, prop, cfg.seed)
        d = float(np.abs(a1 - a2).mean())
        img_deltas.append(d)
        print(f"{f:6d} -> {src:<6d} {d:15.3f} {d / floor:9.1f}x")
    img_sens = float(np.mean(img_deltas))

    # --- Verdict
    prop10 = prop_sens.get(10.0, list(prop_sens.values())[-1])
    print("\n=== summary ===")
    print(f"seed noise floor              : {floor:7.3f} deg")
    print(f"sensitivity to 10 deg proprio : {prop10:7.3f} deg  ({prop10 / floor:5.1f}x floor)")
    print(f"sensitivity to swapped images : {img_sens:7.3f} deg  ({img_sens / floor:5.1f}x floor)")
    ratio = prop10 / img_sens if img_sens > 1e-6 else float("inf")
    print(f"\nproprio : image influence ratio = {ratio:.2f}")
    if ratio > 2.0:
        print("=> The action is driven mostly by PROPRIO. This is the copycat/shortcut")
        print("   signature: on hardware the policy tracks wherever the arm already is")
        print("   instead of driving toward the goal. No deployment parameter fixes this.")
    elif ratio < 0.5:
        print("=> The action is driven mostly by the IMAGES. The policy is visually")
        print("   grounded, so the closed-loop failure lies in execution, not conditioning.")
    else:
        print("=> Mixed influence; neither input dominates.")


if __name__ == "__main__":
    main()
