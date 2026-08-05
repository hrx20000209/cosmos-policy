"""How much does the Wan2.1 VAE encode cost as a function of sequence length?

The deployed SO101 config feeds 41 raw frames (11 latent slots x temporal
compression 4, plus the leading blank) to serve 3 real camera views.  Dropping a
view would shorten the sequence -- but ``state_t`` is baked into the model config
and the DiT was trained against that latent layout, so a shorter sequence is
rejected at inference (`Input video length doesn't match expected length
specified by state_t`).  Reducing views is therefore a *retraining* decision.

This benchmark prices that decision: it times the tokenizer encode alone across
sequence lengths, so the saving from a hypothetical 2-view (33 frame) or 1-view
(25 frame) checkpoint is known before anyone spends GPU-weeks on it.

Usage (cosmos env, repo root):
    PYTHONPATH=$PWD python cosmos_policy/scripts/bench_vae_frames.py
"""

import argparse
import json
import statistics
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step13000/model/model")
    ap.add_argument("--config", default="cosmos_predict2_2b_three_cubes_full_ft")
    ap.add_argument("--config-file", default="configs/eval_config.py")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step13000/profiling/vae_frame_scaling.json")
    args = ap.parse_args()

    from types import SimpleNamespace

    from cosmos_policy.experiments.robot.cosmos_utils import get_model

    cfg = SimpleNamespace(config=args.config, ckpt_path=args.ckpt, config_file=args.config_file)
    model, _ = get_model(cfg)
    tok = model.tokenizer
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 4 frames per latent slot + 1 leading blank.  11 slots = 3 views (trained),
    # 9 = 2 views, 7 = 1 view.
    cases = [(41, "11 slots / 3 views (当前训练配置)"),
             (33, "9 slots / 2 views"),
             (25, "7 slots / 1 view"),
             (17, "5 slots"),
             (9, "3 slots")]

    results = []
    for n_frames, label in cases:
        x = torch.randint(0, 255, (1, 3, n_frames, args.size, args.size), device=dev)
        x = x.to(dtype) / 127.5 - 1.0
        try:
            for _ in range(args.warmup):
                with torch.no_grad():
                    tok.encode(x)
            sync()
            ts = []
            for _ in range(args.iters):
                sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    tok.encode(x)
                sync()
                ts.append((time.perf_counter() - t0) * 1000)
            results.append({"frames": n_frames, "label": label, "p50_ms": statistics.median(ts),
                            "min_ms": min(ts), "max_ms": max(ts)})
        except Exception as exc:  # noqa: BLE001
            results.append({"frames": n_frames, "label": label, "error": repr(exc)[:200]})
        del x
        torch.cuda.empty_cache()

    base = next((r for r in results if r.get("frames") == 41 and "p50_ms" in r), None)
    print(f"\n{'frames':>7}  {'p50 ms':>9}  {'vs 41帧':>9}   说明")
    for r in results:
        if "p50_ms" not in r:
            print(f"{r['frames']:>7}  {'ERROR':>9}             {r['label']}  {r.get('error','')[:60]}")
            continue
        rel = f"{(r['p50_ms'] / base['p50_ms'] - 1) * 100:+.0f}%" if base else "-"
        print(f"{r['frames']:>7}  {r['p50_ms']:>9.1f}  {rel:>9}   {r['label']}")
    if base:
        print(f"\n每帧边际成本 ~ {base['p50_ms'] / 41:.1f} ms/frame")

    with open(args.out, "w") as f:
        json.dump({"size": args.size, "iters": args.iters, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
