"""Can we encode only the conditioning slots and pad the rest in latent space?

The SO101 layout feeds 41 raw frames -> 11 latent slots, but the experiment
config conditions on only the first 5 of them:

    min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, left wrist, right wrist, primary)

Slots 5..10 (action, future proprio, future wrist x2, future primary, value)
are generated from noise, so whatever the VAE produced for them is discarded.

And the Wan2.1 tokenizer is causal (CausalConv3d): latent[k] depends only on
frames <= k. So encoding just the first 17 frames (1 + 4*4) must yield
bit-identical latents for slots 0..4, and slots 5..10 can be filled with
anything.

This script tests exactly that, two ways:
  1. numerically -- are the first 5 latent slots identical between a full
     41-frame encode and a truncated 17-frame encode?
  2. end-to-end -- does get_action() return the same action chunk when the
     tokenizer is patched to encode only the prefix and zero-pad the rest?

If both hold, the encode cost drops from 41 frames to 17.
"""

import argparse
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch

PREFIX_FRAMES = 17  # 1 blank + 4 conditioning slots x temporal compression 4
TOTAL_FRAMES = 41


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step13000/model/model")
    ap.add_argument("--config", default="cosmos_predict2_2b_three_cubes_full_ft")
    ap.add_argument("--config-file", default="configs/eval_config.py")
    ap.add_argument("--stats", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy/processed_data/dataset_statistics.json")
    ap.add_argument("--t5", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy/processed_data/t5_text_embeddings.pkl")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--task", default="go to red cube. take the red cube. go to box. put the red cube in box.")
    ap.add_argument(
        "--real-obs-dir",
        default=None,
        help="Screenshot dir from a profiled run; uses real front/right/wrist frames instead of noise.",
    )
    ap.add_argument("--real-obs-count", type=int, default=5, help="How many real observations to check.")
    args = ap.parse_args()

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    stats = load_dataset_stats(args.stats)
    init_t5_text_embeddings_cache(args.t5)

    cfg = SimpleNamespace(
        suite="aloha", config=args.config, ckpt_path=args.ckpt, config_file=args.config_file,
        use_third_person_image=True, num_third_person_images=1,
        use_wrist_image=True, num_wrist_images=2,
        use_proprio=True, normalize_proprio=True, unnormalize_actions=True,
        dataset_stats_path=args.stats, t5_text_embeddings_path=args.t5,
        trained_with_image_aug=True, chunk_size=16, num_open_loop_steps=16,
        ar_future_prediction=False, ar_value_prediction=False, ar_qvalue_prediction=False,
        use_jpeg_compression=False, flip_images=False,
        num_denoising_steps_action=1, num_denoising_steps_future_state=1, num_denoising_steps_value=1,
        deterministic=True, seed=195, use_variance_scale=False, action_dim=6,
    )
    model, _ = get_model(cfg)
    tok = model.tokenizer

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---------- test 1: are the conditioning latents identical? ----------
    print("\n=== 测试 1：前 5 个槽位的 latent 是否逐比特相同 ===")
    torch.manual_seed(0)
    x_full = (torch.randint(0, 255, (1, 3, TOTAL_FRAMES, 224, 224), device=dev).to(dtype) / 127.5 - 1.0)
    with torch.no_grad():
        lat_full = tok.encode(x_full)
        lat_pref = tok.encode(x_full[:, :, :PREFIX_FRAMES].contiguous())
    lf = lat_full[0] if isinstance(lat_full, (tuple, list)) else lat_full
    lp = lat_pref[0] if isinstance(lat_pref, (tuple, list)) else lat_pref
    print(f"  full  encode -> {tuple(lf.shape)}")
    print(f"  prefix encode -> {tuple(lp.shape)}")
    n = lp.shape[2]
    diff = (lf[:, :, :n] - lp).abs()
    print(f"  前 {n} 个槽位 max|diff| = {diff.max().item():.3e}   mean = {diff.mean().item():.3e}")
    print(f"  {'✅ 逐比特相同（因果性成立）' if diff.max().item() == 0 else '⚠️ 存在差异'}")

    # ---------- test 2: does the action change? ----------
    print("\n=== 测试 2：截断编码 + latent 补零，动作是否改变 ===")
    rng = np.random.default_rng(0)
    observations = []
    if args.real_obs_dir:
        # Real camera frames exercise the encoder on the actual input
        # distribution; noise could hide a difference that real images show.
        from pathlib import Path

        from PIL import Image

        shot_dir = Path(args.real_obs_dir)
        stems = sorted({p.name.rsplit("_", 1)[0] for p in shot_dir.glob("*_front.jpg")})
        for stem in stems[:: max(1, len(stems) // args.real_obs_count)][: args.real_obs_count]:
            try:
                observations.append({
                    "primary_image": np.asarray(Image.open(shot_dir / f"{stem}_front.jpg").convert("RGB")),
                    "left_wrist_image": np.asarray(Image.open(shot_dir / f"{stem}_right.jpg").convert("RGB")),
                    "right_wrist_image": np.asarray(Image.open(shot_dir / f"{stem}_wrist.jpg").convert("RGB")),
                    "proprio": np.array([0.5, -100.0, 90.0, 75.0, -2.0, 2.0], dtype=np.float32),
                })
            except FileNotFoundError:
                continue
        print(f"  使用 {len(observations)} 组真实观测 ({shot_dir.name})")
    if not observations:
        observations = [{
            "primary_image": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "left_wrist_image": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "right_wrist_image": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "proprio": np.array([0.5, -100.0, 90.0, 75.0, -2.0, 2.0], dtype=np.float32),
        }]
        print("  使用合成随机图像")
    obs = observations[0]

    def run():
        return np.asarray(
            get_action(cfg, model, stats, obs, args.task, seed=195, randomize_seed=False,
                       num_denoising_steps_action=1,
                       generate_future_state_and_value_in_parallel=False)["actions"],
            dtype=np.float32,
        )

    baseline = run()
    sync()
    t_full = []
    for _ in range(args.iters):
        sync(); t0 = time.perf_counter(); run(); sync()
        t_full.append((time.perf_counter() - t0) * 1000)

    orig_encode = tok.encode
    calls = {"n": 0}

    def truncated_encode(x, *a, **kw):
        # Only patch the full 41-frame policy input, nothing else.
        if not (torch.is_tensor(x) and x.dim() == 5 and x.shape[2] == TOTAL_FRAMES):
            return orig_encode(x, *a, **kw)
        calls["n"] += 1
        out = orig_encode(x[:, :, :PREFIX_FRAMES].contiguous(), *a, **kw)
        was_tuple = isinstance(out, (tuple, list))
        lat = out[0] if was_tuple else out
        full_t = lf.shape[2]
        pad = torch.zeros(
            lat.shape[0], lat.shape[1], full_t - lat.shape[2], lat.shape[3], lat.shape[4],
            device=lat.device, dtype=lat.dtype,
        )
        lat = torch.cat([lat, pad], dim=2)
        return (lat, *out[1:]) if was_tuple else lat

    tok.encode = truncated_encode
    try:
        truncated = run()
        sync()
        t_trunc = []
        for _ in range(args.iters):
            sync(); t0 = time.perf_counter(); run(); sync()
            t_trunc.append((time.perf_counter() - t0) * 1000)
    finally:
        tok.encode = orig_encode

    d = np.abs(baseline - truncated)
    print(f"  截断编码被调用 {calls['n']} 次")
    print(f"  动作 chunk 形状 {baseline.shape}")
    print(f"  max|Δaction| = {d.max():.6f} deg    mean = {d.mean():.6f} deg")
    print(f"  首步动作 baseline  = {np.array2string(baseline[0], precision=3)}")
    print(f"  首步动作 truncated = {np.array2string(truncated[0], precision=3)}")
    verdict = "✅ 完全一致" if d.max() < 1e-4 else ("⚠️ 有微小差异" if d.max() < 0.5 else "❌ 动作改变")
    print(f"  {verdict}")

    # Sweep the remaining observations so the verdict is not one lucky frame.
    if len(observations) > 1:
        print(f"\n  --- 逐观测复核 (n={len(observations)}) ---")
        print(f"  {'#':>3}{'max|Δ|':>12}{'mean|Δ|':>12}")
        worst = 0.0
        for i, o in enumerate(observations):
            obs = o
            b = run()
            tok.encode = truncated_encode
            try:
                t = run()
            finally:
                tok.encode = orig_encode
            dd = np.abs(b - t)
            worst = max(worst, dd.max())
            print(f"  {i:>3}{dd.max():>12.6f}{dd.mean():>12.6f}")
        print(f"  全部真实观测 max|Δaction| = {worst:.6f} deg")

    print(f"\n=== 端到端耗时 (denoise=1, n={args.iters}) ===")
    print(f"  完整 41 帧编码 : p50 = {statistics.median(t_full):7.1f} ms")
    print(f"  截断 17 帧编码 : p50 = {statistics.median(t_trunc):7.1f} ms")
    saved = statistics.median(t_full) - statistics.median(t_trunc)
    print(f"  节省           : {saved:7.1f} ms  ({saved / statistics.median(t_full) * 100:.0f}%)")


if __name__ == "__main__":
    main()
