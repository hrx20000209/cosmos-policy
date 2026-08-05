"""Two things the pixel-space future-state experiment left open.

1. Future proprio straight out of the latent.
   Decoding the predicted future *frames* costs ~994 ms, which rules it out of
   the control loop. But the future robot state does not need decoding: it is
   written into latent slot 6 the same way the action chunk is written into slot
   5, and ``extract_action_chunk_from_latent_sequence`` already inverts exactly
   that encoding. If it can be read for free, it is a candidate anchor for
   extending the action horizon beyond the 16-step chunk.

   The question that decides whether it is useful: does it say anything the last
   step of the action chunk does not already say?

2. A 25-frame truncation.
   ``truncated_encode`` cuts to 17 frames (slots 0-4) and is incompatible with
   future-state decoding, which needs slots 5 and 6 intact. Cutting to 25 frames
   (slots 0-6) should preserve the future decode while still skipping the four
   future-image slots -- keeping part of the saving instead of none.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

TOTAL_FRAMES = 41


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step20000/model")
    ap.add_argument("--config", default="cosmos_predict2_2b_three_cubes_full_ft")
    ap.add_argument("--config-file", default="configs/eval_config.py")
    ap.add_argument("--stats", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy/processed_data/dataset_statistics.json")
    ap.add_argument("--t5", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy/processed_data/t5_text_embeddings.pkl")
    ap.add_argument("--run-dir", required=True, help="A profiled run: supplies real observations and the measured future")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--task", default="go to red cube. take the red cube. go to box. put the red cube in box.")
    args = ap.parse_args()

    from PIL import Image

    from cosmos_policy.experiments.robot.cosmos_utils import (
        extract_action_chunk_from_latent_sequence, get_action, get_model,
        init_t5_text_embeddings_cache, load_dataset_stats,
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
    orig_encode = tok.encode
    state_t = int(model.config.state_t)
    p_min = np.asarray(stats["proprio_min"], dtype=np.float64)
    p_max = np.asarray(stats["proprio_max"], dtype=np.float64)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ---- real observations, paired with the joint positions actually measured later ----
    run_dir = Path(args.run_dir)
    shot_dir = next(run_dir.glob("screenshots_*"))
    acts = [json.loads(l) for l in open(next(run_dir.glob("action_trace_*.jsonl")))]
    JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    a_t = np.array([a["exec_start"] for a in acts])
    a_pos = np.array([[a["present_position"][j] for j in JOINTS] for a in acts])

    stems = sorted({p.name.rsplit("_", 1)[0] for p in shot_dir.glob("*_front.jpg")})
    step = max(1, len(stems) // args.n)
    samples = []
    for stem in stems[::step][: args.n]:
        w = float(stem.split("_")[1])
        try:
            imgs = {k: np.asarray(Image.open(shot_dir / f"{stem}_{c}.jpg").convert("RGB"))
                    for k, c in (("primary_image", "front"), ("left_wrist_image", "right"), ("right_wrist_image", "wrist"))}
        except FileNotFoundError:
            continue
        i0 = int(np.argmin(np.abs(a_t - w)))
        samples.append({"wall": w, "obs": {**imgs, "proprio": a_pos[i0].astype(np.float32)}, "i0": i0})
    print(f"取到 {len(samples)} 组真实观测")

    # ---------------- 1. future proprio from the latent ----------------
    print("\n=== 1. 从 latent 槽位直读 future proprio ===")
    rows = []
    for s in samples:
        res = get_action(cfg, model, stats, s["obs"], args.task, seed=195, randomize_seed=False,
                         num_denoising_steps_action=1, generate_future_state_and_value_in_parallel=False)
        lat = res["generated_latent"]
        idx = res["latent_indices"]["future_proprio_latent_idx"]
        if not torch.is_tensor(idx):
            idx = torch.tensor([idx], device=lat.device)
        # Same inversion the action chunk uses, with a (1, proprio_dim) target.
        fp = extract_action_chunk_from_latent_sequence(lat, (1, 6), idx).to(torch.float32).cpu().numpy()[0, 0]
        fp = 0.5 * (fp + 1) * (p_max - p_min) + p_min       # undo rescale_proprio
        chunk = np.asarray(res["actions"], dtype=np.float32)
        rows.append({"wall": s["wall"], "i0": s["i0"], "future_proprio": fp.tolist(),
                     "chunk_last": chunk[-1].tolist(), "proprio_now": s["obs"]["proprio"].tolist()})

    fp = np.array([r["future_proprio"] for r in rows])
    last = np.array([r["chunk_last"] for r in rows])
    now = np.array([r["proprio_now"] for r in rows])
    print(f"  预测 future proprio 与 chunk 末步 的差:  mean|Δ|={np.abs(fp - last).mean():.3f}  "
          f"max={np.abs(fp - last).max():.3f} deg")
    print(f"  预测 future proprio 与 当前 proprio 的差: mean|Δ|={np.abs(fp - now).mean():.3f} deg")
    print("  -> 若前者远小于后者，说明它只是复述 chunk 末步，对扩展视界没有新信息")

    print(f"\n  与真实测得的未来位姿对比（沿时间轴扫描视界）:")
    print(f"  {'horizon(s)':>11}{'future_proprio':>16}{'chunk末步':>12}{'持恒(当前)':>12}")
    for h in (0.5, 1.0, 2.0, 3.0, 4.0):
        e_fp, e_last, e_now = [], [], []
        for k, r in enumerate(rows):
            tgt = r["wall"] + h
            if tgt > a_t[-1]:
                continue
            i = int(np.argmin(np.abs(a_t - tgt)))
            if abs(a_t[i] - tgt) > 0.5:
                continue
            truth = a_pos[i]
            e_fp.append(np.abs(fp[k] - truth).mean())
            e_last.append(np.abs(last[k] - truth).mean())
            e_now.append(np.abs(now[k] - truth).mean())
        if e_fp:
            print(f"  {h:>11.1f}{np.mean(e_fp):>16.2f}{np.mean(e_last):>12.2f}{np.mean(e_now):>12.2f}")
    print("  (数值为平均绝对误差 deg，越小越准)")

    # ---------------- 2. 25-frame truncation ----------------
    print("\n=== 2. 截断到 25 帧（保留 future 解码所需的槽位 5、6）===")

    def make_trunc(n_frames):
        def enc(x, *a, **kw):
            if not (torch.is_tensor(x) and x.dim() == 5 and x.shape[2] == TOTAL_FRAMES):
                return orig_encode(x, *a, **kw)
            out = orig_encode(x[:, :, :n_frames].contiguous(), *a, **kw)
            was = isinstance(out, (tuple, list))
            lat = out[0] if was else out
            miss = state_t - lat.shape[2]
            if miss > 0:
                pad = torch.zeros(lat.shape[0], lat.shape[1], miss, lat.shape[3], lat.shape[4],
                                  device=lat.device, dtype=lat.dtype)
                lat = torch.cat([lat, pad], dim=2)
            return (lat, *out[1:]) if was else lat
        return enc

    obs0 = samples[0]["obs"]

    def run_full(future):
        return get_action(cfg, model, stats, obs0, args.task, seed=195, randomize_seed=False,
                          num_denoising_steps_action=1,
                          generate_future_state_and_value_in_parallel=future)

    base = run_full(True)
    base_act = np.asarray(base["actions"], dtype=np.float32)
    base_img = {k: (v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
                for k, v in (base.get("future_image_predictions") or {}).items()}

    results = {}
    for n_frames, label in ((41, "完整"), (25, "截断25"), (17, "截断17")):
        tok.encode = orig_encode if n_frames == TOTAL_FRAMES else make_trunc(n_frames)
        try:
            r = run_full(True)
            act = np.asarray(r["actions"], dtype=np.float32)
            imgs = {k: (v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
                    for k, v in (r.get("future_image_predictions") or {}).items()}
            img_err = float(np.mean([np.abs(imgs[k] - base_img[k]).mean() for k in base_img if k in imgs])) if base_img else float("nan")
            sync()
            ts = []
            for _ in range(args.iters):
                sync(); t0 = time.perf_counter(); run_full(True); sync()
                ts.append((time.perf_counter() - t0) * 1000)
            results[label] = {"frames": n_frames, "p50_ms": statistics.median(ts),
                              "action_max_delta": float(np.abs(act - base_act).max()),
                              "future_image_mean_abs_delta": img_err}
        finally:
            tok.encode = orig_encode

    print(f"  {'方案':<10}{'帧数':>6}{'p50 ms':>10}{'动作max|Δ|':>13}{'future图 mean|Δ|':>18}")
    for k, v in results.items():
        print(f"  {k:<10}{v['frames']:>6}{v['p50_ms']:>10.1f}{v['action_max_delta']:>13.3f}"
              f"{v['future_image_mean_abs_delta']:>18.4f}")
    print("  (相对'完整'的偏差；future图偏差接近 0 表示该截断仍能正确解码未来帧)")

    if args.out:
        Path(args.out).write_text(json.dumps({"future_proprio_rows": rows, "truncation": results},
                                             indent=2, ensure_ascii=False))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
