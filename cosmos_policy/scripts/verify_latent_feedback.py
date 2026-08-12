#!/usr/bin/env python
"""Measure what feeding predicted future latents back in costs, in degrees.

The question the design cannot answer on its own is exposure bias: the model
conditions on encoded *real* frames during training, and latent feedback hands
it its own output instead. So this replays a stretch of real recorded
observations twice --

    reference : every step encodes the real cameras
    feedback  : step 1 encodes for real, steps 2..N are served from the
                previous step's predicted future slots and never touch the VAE

-- with identical proprio, identical seed, and identical everything else, then
reports how far the action chunk has drifted after k consecutive fed-back steps.

That curve is the whole design input for a schedule: it says how many steps of
imagination the policy tolerates before a real observation has to be spent.

This offline comparison is valid in a way the APEX replay was not. Nothing here
depends on how the arm responds -- it is the same model given two different
inputs, so the difference measured is the entire effect.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_observations(run_dir: Path, start: float, count: int, stride: int):
    """Pull (images, proprio) tuples from a recorded profiling run."""
    run_dir = Path(run_dir)
    acts = [json.loads(l) for l in open(next(run_dir.glob("action_trace_*.jsonl")))]
    obs_t = [json.loads(l) for l in open(next(run_dir.glob("observation_trace_*.jsonl")))]
    t0 = min(o["wall_time"] for o in obs_t)

    shot_dir = next(run_dir.glob("screenshots_*"))
    pat = re.compile(r"t(\d+)_(\d+\.\d+)_(\w+)\.jpg")
    frames: dict[float, dict[str, Path]] = {}
    for p in shot_dir.glob("*.jpg"):
        m = pat.fullmatch(p.name)
        if m and not p.name.endswith("_pred.jpg"):
            frames.setdefault(float(m.group(2)), {})[m.group(3)] = p
    times = sorted(t for t, v in frames.items() if {"front", "right", "wrist"} <= set(v))
    if not times:
        raise SystemExit(f"no complete camera triples under {shot_dir}")

    import cv2

    a_t = np.array([a["exec_start"] for a in acts])
    joints = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

    sel = [t for t in times if t - t0 >= start][:: max(1, stride)][:count]
    out = []
    for t in sel:
        imgs = {c: cv2.cvtColor(cv2.imread(str(frames[t][c])), cv2.COLOR_BGR2RGB) for c in ("front", "right", "wrist")}
        i = int(np.argmin(np.abs(a_t - t)))
        proprio = np.array([acts[i]["present_position"][j] for j in joints], dtype=np.float32)
        out.append({"rel_t": t - t0, "images": imgs, "proprio": proprio})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="A profiling run directory with screenshots + traces")
    ap.add_argument("--ckpt", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step20000")
    ap.add_argument("--start", type=float, default=4.0, help="Seconds into the run to start from")
    ap.add_argument("--count", type=int, default=10, help="Consecutive observations to replay")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--denoise-steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="go to red cube. take the red cube. go to box. put the red cube in box.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "robot"))
    from cosmos_utils import get_action  # noqa: E402
    from latent_feedback import LatentFeedback  # noqa: E402

    from so101_async_deploy_three_cubes_k16 import (  # noqa: E402
        SO101CosmosAsyncServerConfig,
        SO101CosmosAsyncPolicyServer,
    )

    obs_seq = load_observations(Path(args.run_dir), args.start, args.count, args.stride)
    print(f"{len(obs_seq)} observations, t = {obs_seq[0]['rel_t']:.1f}..{obs_seq[-1]['rel_t']:.1f}s")

    # __init__ loads the model and does not bind a port, so it is reusable here.
    cfg = SO101CosmosAsyncServerConfig(
        ckpt_path=args.ckpt,
        num_denoising_steps_action=args.denoise_steps,
        truncate_vae_encode=False,
        profile_stages=False,
    )
    server = SO101CosmosAsyncPolicyServer(cfg)
    model, cosmos_cfg, stats = server.model, server.cosmos_cfg, server.dataset_stats

    def infer(o):
        co = {
            "primary_image": o["images"]["front"],
            "left_wrist_image": o["images"]["right"],
            "right_wrist_image": o["images"]["wrist"],
            "proprio": o["proprio"],
        }
        t = time.perf_counter()
        r = get_action(cosmos_cfg, model, stats, co, args.task, seed=args.seed,
                       num_denoising_steps_action=args.denoise_steps,
                       generate_future_state_and_value_in_parallel=False)
        return np.asarray(r["actions"], dtype=np.float32), (time.perf_counter() - t) * 1000

    print("\n--- reference: every step encodes the real cameras ---")
    ref, ref_ms = [], []
    for o in obs_seq:
        a, ms = infer(o)
        ref.append(a); ref_ms.append(ms)
        print(f"  t={o['rel_t']:5.1f}s  {ms:6.1f} ms")

    lf = LatentFeedback(model)
    print("\n--- feedback:", lf.install(), "---")
    fb, fb_ms, served = [], [], []
    for k, o in enumerate(obs_seq):
        if k > 0:
            armed = lf.arm()
        else:
            lf.disarm(); armed = False
        a, ms = infer(o)
        lf.disarm()
        fb.append(a); fb_ms.append(ms); served.append(armed)
        d = np.abs(a - ref[k])
        print(f"  t={o['rel_t']:5.1f}s  {ms:6.1f} ms  fed_back={armed}  "
              f"|Δaction| mean {d.mean():6.3f}  max {d.max():6.3f} deg")

    ref_ms, fb_ms = np.array(ref_ms), np.array(fb_ms)
    n_fb = sum(served)
    print("\n=== summary ===")
    print(f"  reference        median {np.median(ref_ms):6.1f} ms")
    if n_fb:
        print(f"  fed-back steps   median {np.median(fb_ms[np.array(served)]):6.1f} ms  "
              f"({100 * (1 - np.median(fb_ms[np.array(served)]) / np.median(ref_ms)):.0f}% faster)")
    print("\n  consecutive fed-back steps -> action deviation from the real-encode reference")
    for k in range(1, len(obs_seq)):
        d = np.abs(fb[k] - ref[k])
        print(f"    k={k:2d}   mean {d.mean():7.3f} deg   max {d.max():7.3f} deg")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "rel_t": [o["rel_t"] for o in obs_seq],
            "ref_ms": ref_ms.tolist(), "fb_ms": fb_ms.tolist(), "served": served,
            "mean_dev": [float(np.abs(fb[k] - ref[k]).mean()) for k in range(len(obs_seq))],
            "max_dev": [float(np.abs(fb[k] - ref[k]).max()) for k in range(len(obs_seq))],
        }, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
