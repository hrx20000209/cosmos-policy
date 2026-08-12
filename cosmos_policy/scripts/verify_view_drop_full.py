#!/usr/bin/env python
"""Full-episode action curves for every view that can be dropped.

The earlier ablation compared a single chunk, which cannot show whether a
deviation is a transient or a systematic bias that accumulates over a task. This
replays a whole recorded episode -- one inference per sampled observation, with
identical proprio and seed -- for the three-view baseline and for each choice of
dropped conditioning slot, and plots the resulting action trajectory end to end.

It also reports, per choice, how much the *surviving* slots are altered. The
tokenizer is causal in time, so removing slot k's pixel frames changes the
receptive field of every slot after it: dropping slot 2 disturbs 3 and 4,
dropping 3 disturbs only 4, and dropping 4 -- the last conditioning slot --
disturbs nothing and is the only ablation whose result is attributable to the
missing view rather than to the views it damaged on the way out.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ROLE = {0: "blank", 1: "proprio", 2: "左腕(=right相机)", 3: "右腕(=wrist相机)", 4: "主视角(=front)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step20000/model")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--count", type=int, default=26)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--drop-slots", default="2,3,4")
    ap.add_argument("--task", default="go to red cube. take the red cube. go to box. put the red cube in box.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    here = Path(__file__).resolve().parents[1] / "experiments" / "robot"
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from cosmos_utils import get_action  # noqa: E402
    from so101_async_deploy_three_cubes_k16 import (  # noqa: E402
        SO101CosmosAsyncServerConfig, SO101CosmosAsyncPolicyServer,
    )
    from verify_latent_feedback import load_observations  # noqa: E402

    obs_seq = load_observations(Path(args.run_dir), args.start, args.count, args.stride)
    print(f"{len(obs_seq)} observations, t = {obs_seq[0]['rel_t']:.1f}..{obs_seq[-1]['rel_t']:.1f}s", flush=True)

    server = SO101CosmosAsyncPolicyServer(SO101CosmosAsyncServerConfig(
        ckpt_path=args.ckpt, num_denoising_steps_action=1,
        truncate_vae_encode=False, profile_stages=False,
    ))
    model, cfg, stats = server.model, server.cosmos_cfg, server.dataset_stats
    tok = model.tokenizer
    pristine = tok.encode                      # never wrapped; every mode rebuilds from this
    n_cond = int(model.config.min_num_conditional_frames)
    state_t = int(model.config.state_t)
    total = int(tok.get_pixel_num_frames(state_t))
    prefix = int(tok.get_pixel_num_frames(n_cond))

    def make_encode(drop_slot):
        """Truncate to the conditioning prefix, optionally dropping one slot."""
        lo = 0 if drop_slot == 0 else int(tok.get_pixel_num_frames(drop_slot))
        hi = 1 if drop_slot == 0 else int(tok.get_pixel_num_frames(drop_slot + 1))

        def enc(x, *a, **kw):
            if not (torch.is_tensor(x) and x.dim() == 5 and x.shape[2] == total):
                return pristine(x, *a, **kw)
            if drop_slot < 0:
                keep = x[:, :, :prefix].contiguous()
            else:
                keep = torch.cat([x[:, :, :lo], x[:, :, hi:prefix]], dim=2).contiguous()
            out = pristine(keep, *a, **kw)
            lat = out[0] if isinstance(out, (tuple, list)) else out
            if drop_slot >= 0:
                gap = torch.zeros_like(lat[:, :, :1])
                lat = torch.cat([lat[:, :, :drop_slot], gap, lat[:, :, drop_slot:]], dim=2)
            miss = state_t - lat.shape[2]
            if miss > 0:
                lat = torch.cat([lat, torch.zeros(
                    lat.shape[0], lat.shape[1], miss, lat.shape[3], lat.shape[4],
                    device=lat.device, dtype=lat.dtype)], dim=2)
            return (lat, *out[1:]) if isinstance(out, (tuple, list)) else lat
        return enc

    def infer(o):
        co = {"primary_image": o["images"]["front"],
              "left_wrist_image": o["images"]["right"],
              "right_wrist_image": o["images"]["wrist"],
              "proprio": o["proprio"]}
        t = time.perf_counter()
        r = get_action(cfg, model, stats, co, args.task, seed=0,
                       num_denoising_steps_action=1,
                       generate_future_state_and_value_in_parallel=False)
        return np.asarray(r["actions"], dtype=np.float32), (time.perf_counter() - t) * 1000

    modes = [-1] + [int(v) for v in args.drop_slots.split(",") if v.strip()]
    results, timings = {}, {}
    for ds in modes:
        tok.encode = make_encode(ds)
        name = "3视角基准" if ds < 0 else f"删槽位{ds} {ROLE.get(ds,'')}"
        acts, ms = [], []
        for o in obs_seq:
            a, t = infer(o)
            acts.append(a); ms.append(t)
        results[ds] = np.array(acts)      # (T, chunk, 6)
        timings[ds] = float(np.median(ms[1:] or ms))
        print(f"  {name:26s} 中位 {timings[ds]:7.1f} ms", flush=True)
    tok.encode = pristine

    # --- how much the surviving slots move
    print("\n=== 删除后其余条件槽位的 latent 变化 ===", flush=True)
    grabbed = {}

    def capture(x, *a, **kw):
        if torch.is_tensor(x) and x.dim() == 5 and "x" not in grabbed:
            grabbed["x"] = x.detach().clone()
        return pristine(x, *a, **kw)

    tok.encode = capture
    infer(obs_seq[len(obs_seq) // 2])
    tok.encode = pristine
    x = grabbed["x"]
    corruption = {}
    with torch.inference_mode():
        full = pristine(x[:, :, :prefix].contiguous())
        full = (full[0] if isinstance(full, (tuple, list)) else full).float()
        scale = full[:, :, :n_cond].abs().mean().item()
        for ds in modes:
            if ds < 0:
                continue
            lo = 0 if ds == 0 else int(tok.get_pixel_num_frames(ds))
            hi = 1 if ds == 0 else int(tok.get_pixel_num_frames(ds + 1))
            keep = torch.cat([x[:, :, :lo], x[:, :, hi:prefix]], dim=2).contiguous()
            d = pristine(keep)
            d = (d[0] if isinstance(d, (tuple, list)) else d).float()
            rows = []
            for s in range(n_cond):
                if s == ds:
                    continue
                src = s if s < ds else s - 1
                rel = (full[:, :, s] - d[:, :, src]).abs().mean().item() / scale
                rows.append((s, rel))
            corruption[ds] = rows
            worst = max(r for _, r in rows)
            print(f"  删槽位 {ds}: " + "  ".join(f"槽{s}={100*r:5.1f}%" for s, r in rows)
                  + f"   最大 {100*worst:.1f}%", flush=True)

    payload = {
        "rel_t": [o["rel_t"] for o in obs_seq],
        "timings_ms": {str(k): v for k, v in timings.items()},
        "corruption": {str(k): v for k, v in corruption.items()},
        "actions": {str(k): v.tolist() for k, v in results.items()},
    }
    out = args.out or str(Path(args.run_dir) / "view_drop_full.json")
    Path(out).write_text(json.dumps(payload))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
