#!/usr/bin/env python3
"""Phase 1 minimal LIBERO env test: load task -> reset -> obs -> no-op steps -> close.
Validates the whole sim/render stack BEFORE loading the 2B model.
Run: MUJOCO_GL=egl .venv/bin/python min_env_test.py [--suite libero_10] [--task 0]
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--outdir", default="experiments/liberoplus_quantization/logs")
    args = ap.parse_args()

    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import (
        get_libero_env, get_libero_image, get_libero_wrist_image, get_libero_dummy_action)

    bm = benchmark.get_benchmark_dict()[args.suite]()
    n = bm.n_tasks
    print(f"[suite {args.suite}] n_tasks={n}")
    task = bm.get_task(args.task)
    print(f"[task {args.task}] name={task.name}")
    print(f"          language={task.language!r}")

    env, desc = get_libero_env(task, "cosmos", resolution=256)
    init_states = bm.get_task_init_states(args.task)
    print(f"init_states shape={np.asarray(init_states).shape}")

    obs = env.reset()
    obs = env.set_init_state(init_states[0])
    print("obs keys:", sorted(k for k in obs.keys())[:12], "...")

    agent = get_libero_image(obs)
    wrist = get_libero_wrist_image(obs)
    proprio_keys = [k for k in obs if k.startswith("robot0") and "image" not in k]
    print(f"agentview img: shape={agent.shape} dtype={agent.dtype} range=[{agent.min()},{agent.max()}]")
    print(f"wrist img:     shape={wrist.shape} dtype={wrist.dtype} range=[{wrist.min()},{wrist.max()}]")
    ee = obs.get("robot0_eef_pos")
    gq = obs.get("robot0_gripper_qpos")
    jq = obs.get("robot0_joint_pos")
    print(f"proprio keys (sample): {proprio_keys[:8]}")
    print(f"  robot0_eef_pos={np.round(ee,4) if ee is not None else None}")
    print(f"  robot0_gripper_qpos={np.round(gq,4) if gq is not None else None}")
    print(f"  robot0_joint_pos={np.round(jq,4) if jq is not None else None}")

    noop = get_libero_dummy_action("cosmos")
    print(f"no-op action = {noop}")
    for t in range(args.steps):
        obs, reward, done, info = env.step(noop)
    print(f"stepped {args.steps} no-op actions OK. last reward={reward} done={done}")

    # save a frame to confirm rendering
    try:
        from PIL import Image
        os.makedirs(args.outdir, exist_ok=True)
        p = os.path.join(args.outdir, f"min_env_{args.suite}_task{args.task}.png")
        Image.fromarray(np.flipud(agent).astype(np.uint8)).save(p)
        print(f"saved frame -> {p}")
    except Exception as e:
        print("frame save skipped:", e)

    env.close()
    print("ENV_TEST_OK")


if __name__ == "__main__":
    sys.exit(main())
