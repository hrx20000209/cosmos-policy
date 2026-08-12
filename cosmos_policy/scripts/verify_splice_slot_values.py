#!/usr/bin/env python
"""Does dropping a middle view's frames change the *surviving* slots' latents?

``truncated_encode.install(drop_slot=k)`` removes slot k's pixel frames before
encoding and re-inserts a zero latent at index k afterwards, so the output
ordering is exactly the original. The open question is the values: the Wan2.1
tokenizer is causal in time, so the receptive field of every slot after k used
to contain slot k's frames and no longer does.

This encodes one real policy input both ways and compares the slots that
survive. If the deltas are at the numerical-noise level, splicing a middle view
is clean and the hardware failure has to be explained by the missing view
itself. If they are not, the ablation confounds "view removed" with "later
views altered" and cannot attribute anything.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step20000/model")
    ap.add_argument("--drop-slot", type=int, default=2)
    ap.add_argument("--start", type=float, default=6.0)
    ap.add_argument("--task", default="go to red cube. take the red cube. go to box. put the red cube in box.")
    args = ap.parse_args()

    here = Path(__file__).resolve().parents[1] / "experiments" / "robot"
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from cosmos_utils import get_action  # noqa: E402
    from so101_async_deploy_three_cubes_k16 import (  # noqa: E402
        SO101CosmosAsyncServerConfig, SO101CosmosAsyncPolicyServer,
    )
    from verify_latent_feedback import load_observations  # noqa: E402

    obs = load_observations(Path(args.run_dir), args.start, 1, 1)[0]
    server = SO101CosmosAsyncPolicyServer(SO101CosmosAsyncServerConfig(
        ckpt_path=args.ckpt, num_denoising_steps_action=1,
        truncate_vae_encode=False, profile_stages=False,
    ))
    model, cfg, stats = server.model, server.cosmos_cfg, server.dataset_stats
    tok = model.tokenizer
    orig_encode = tok.encode

    # Capture the exact pixel tensor the policy feeds the tokenizer.
    grabbed = {}

    def capture(x, *a, **kw):
        if torch.is_tensor(x) and x.dim() == 5 and "x" not in grabbed:
            grabbed["x"] = x.detach().clone()
        return orig_encode(x, *a, **kw)

    tok.encode = capture
    get_action(cfg, model, stats, {
        "primary_image": obs["images"]["front"],
        "left_wrist_image": obs["images"]["right"],
        "right_wrist_image": obs["images"]["wrist"],
        "proprio": obs["proprio"],
    }, args.task, seed=0, num_denoising_steps_action=1,
        generate_future_state_and_value_in_parallel=False)
    tok.encode = orig_encode

    x = grabbed["x"]
    n_cond = int(model.config.min_num_conditional_frames)
    prefix = int(tok.get_pixel_num_frames(n_cond))
    k = args.drop_slot
    lo = 0 if k == 0 else int(tok.get_pixel_num_frames(k))
    hi = 1 if k == 0 else int(tok.get_pixel_num_frames(k + 1))
    print(f"\npixel tensor {tuple(x.shape)}, conditioning prefix {prefix} frames, "
          f"slot {k} spans frames [{lo}, {hi})")

    with torch.inference_mode():
        full = orig_encode(x[:, :, :prefix].contiguous())
        full = full[0] if isinstance(full, (tuple, list)) else full
        keep = torch.cat([x[:, :, :lo], x[:, :, hi:prefix]], dim=2).contiguous()
        drop = orig_encode(keep)
        drop = drop[0] if isinstance(drop, (tuple, list)) else drop

    print(f"full encode -> {tuple(full.shape)}   dropped encode -> {tuple(drop.shape)}")
    f = full.float()
    d = drop.float()
    scale = f[:, :, :n_cond].abs().mean().item()
    print(f"\nreference scale: mean |latent| over the conditioning slots = {scale:.4f}\n")
    print(f"{'slot':>5s} {'role':<22s} {'max|Δ|':>10s} {'mean|Δ|':>10s} {'rel. to scale':>14s}")
    roles = {0: "blank", 1: "proprio", 2: "left wrist (=right cam)",
             3: "right wrist (=wrist)", 4: "primary (=front)"}
    for s in range(n_cond):
        if s == k:
            print(f"{s:5d} {roles.get(s,''):<22s} {'-- dropped, zero-filled --':>36s}")
            continue
        src = s if s < k else s - 1          # index of this slot in the shortened encode
        delta = (f[:, :, s] - d[:, :, src]).abs()
        print(f"{s:5d} {roles.get(s,''):<22s} {delta.max().item():10.5f} "
              f"{delta.mean().item():10.5f} {delta.mean().item()/scale:13.1%}")


if __name__ == "__main__":
    main()
