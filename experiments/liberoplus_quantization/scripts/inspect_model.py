#!/usr/bin/env python3
"""Load Cosmos Policy and dump the module tree of model.net (the DiT backbone):
per top-level submodule param counts, dtypes, and Linear inventory. Feeds
report/model_quantization_map.md and the quantization filter design.

Run (needs base model cached + HF token in $HF_HOME/token):
  source env.sh; CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
    .venv/bin/python experiments/liberoplus_quantization/scripts/inspect_model.py
"""
import argparse
import collections
import json
import os

import torch
import torch.nn as nn


def build_cfg(ckpt):
    from dataclasses import dataclass

    @dataclass
    class C:
        suite = "libero"
        model_family = "cosmos"
        config = "cosmos_predict2_2b_480p_libero__inference_only"
        ckpt_path = ckpt
        planning_model_config_name = ""
        planning_model_ckpt_path = ""
        config_file = "cosmos_policy/config/config.py"
    return C()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
    ap.add_argument("--out", default="experiments/liberoplus_quantization/report/model_structure.json")
    args = ap.parse_args()

    from cosmos_policy.experiments.robot.cosmos_utils import get_model

    cfg = build_cfg(args.ckpt)
    model, config = get_model(cfg)
    print("=== model class:", type(model).__name__)
    # find the DiT
    net = getattr(model, "net", None)
    print("=== model.net class:", type(net).__name__ if net is not None else None)

    def count(m):
        return sum(p.numel() for p in m.parameters())

    total = count(model)
    net_total = count(net) if net is not None else 0
    print(f"total params: {total/1e9:.3f} B ; net params: {net_total/1e9:.3f} B ({net_total/max(total,1)*100:.1f}%)")

    # top-level children of model
    print("\n=== model top-level children (param B) ===")
    top = {}
    for name, child in model.named_children():
        c = count(child)
        top[name] = c
        print(f"  {name:24s} {c/1e9:8.4f} B  {type(child).__name__}")

    # net top-level children
    net_children = {}
    if net is not None:
        print("\n=== model.net children (param B) ===")
        for name, child in net.named_children():
            c = count(child)
            net_children[name] = c
            print(f"  {name:28s} {c/1e9:8.4f} B  {type(child).__name__}")

    # Linear inventory in net: group by the module-name prefix (strip block index)
    import re
    lin = collections.Counter()
    lin_params = collections.Counter()
    dtypes = collections.Counter()
    n_linear = 0
    example = {}
    if net is not None:
        for name, mod in net.named_modules():
            for p in mod.parameters(recurse=False):
                dtypes[str(p.dtype)] += p.numel()
            if isinstance(mod, nn.Linear):
                n_linear += 1
                key = re.sub(r"\.\d+\.", ".<i>.", name)
                lin[key] += 1
                lin_params[key] += mod.weight.numel()
                if key not in example:
                    example[key] = f"{tuple(mod.weight.shape)}"
    print(f"\n=== net Linear layers: {n_linear} total ===")
    for k, v in sorted(lin_params.items(), key=lambda x: -x[1]):
        print(f"  {lin[k]:4d}x  {lin_params[k]/1e6:9.2f} M  {k}   e.g.{example[k]}")

    print("\n=== net param dtypes ===")
    for d, n in dtypes.items():
        print(f"  {d}: {n/1e6:.1f} M")

    out = {
        "model_class": type(model).__name__,
        "net_class": type(net).__name__ if net is not None else None,
        "total_params": total, "net_params": net_total,
        "model_top": top, "net_children": net_children,
        "linear_groups": {k: {"count": lin[k], "params": lin_params[k], "shape": example[k]} for k in lin},
        "n_linear": n_linear,
        "net_dtypes": dict(dtypes),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=str)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
