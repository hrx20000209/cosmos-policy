"""Drop a camera view by splicing its latent slot instead of encoding it.

The pixel-space route is blocked: ``state_t`` is fixed by the training config, so
a shorter video is rejected, and zero-padding it back to 41 frames saves nothing
because the VAE still encodes all of them.

The latent-space route is not blocked.  ``truncated_encode`` already proves we
can hand the model a latent tensor we assembled ourselves.  So: encode only the
frames of the views we keep, then splice something into the dropped view's slot
(zeros, or a copy of another view's latent) before the DiT sees it.

Slot layout for this SO101 config, after truncation to the conditioning prefix:

    slot 0  blank         frame  0
    slot 1  proprio       frames 1-4
    slot 2  left_wrist    frames 5-8     <- "right" camera
    slot 3  right_wrist   frames 9-12    <- "wrist" camera
    slot 4  primary       frames 13-16   <- "front" camera

Caveat this measures: the tokenizer is causal, so removing slot 2's frames
shifts slots 3 and 4 earlier and changes *their* latents too.  The spliced
result is therefore not the latent the model was trained on, and the size of
that mismatch is exactly what this script quantifies.

Note the asymmetry: dropping the *last* conditioning slot (primary) is a clean
truncation that leaves the others bit-identical, while dropping a middle slot
perturbs everything after it.
"""

import argparse
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch

TOTAL_FRAMES = 41
PREFIX_FRAMES = 17
SLOT_FRAMES = {  # slot -> (start, end) in the 17-frame prefix
    0: (0, 1),
    1: (1, 5),
    2: (5, 9),
    3: (9, 13),
    4: (13, 17),
}
SLOT_NAME = {0: "blank", 1: "proprio", 2: "right(左腕槽)", 3: "wrist(右腕槽)", 4: "front(主视角)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step20000/model")
    ap.add_argument("--config", default="cosmos_predict2_2b_three_cubes_full_ft")
    ap.add_argument("--config-file", default="configs/eval_config.py")
    ap.add_argument("--stats", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy/processed_data/dataset_statistics.json")
    ap.add_argument("--t5", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy/processed_data/t5_text_embeddings.pkl")
    ap.add_argument("--drop-slot", type=int, default=2, help="Which conditioning slot to stop encoding")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--real-obs-dir", default=None)
    ap.add_argument("--obs-count", type=int, default=4)
    ap.add_argument("--task", default="go to red cube. take the red cube. go to box. put the red cube in box.")
    args = ap.parse_args()

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
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

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ---------- observations ----------
    observations = []
    if args.real_obs_dir:
        from pathlib import Path
        from PIL import Image
        sd = Path(args.real_obs_dir)
        stems = sorted({p.name.rsplit("_", 1)[0] for p in sd.glob("*_front.jpg")})
        step = max(1, len(stems) // args.obs_count)
        for stem in stems[::step][: args.obs_count]:
            try:
                observations.append({
                    "primary_image": np.asarray(Image.open(sd / f"{stem}_front.jpg").convert("RGB")),
                    "left_wrist_image": np.asarray(Image.open(sd / f"{stem}_right.jpg").convert("RGB")),
                    "right_wrist_image": np.asarray(Image.open(sd / f"{stem}_wrist.jpg").convert("RGB")),
                    "proprio": np.array([0.5, -100.0, 90.0, 75.0, -2.0, 2.0], dtype=np.float32),
                })
            except FileNotFoundError:
                pass
    if not observations:
        rng = np.random.default_rng(0)
        observations = [{
            "primary_image": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "left_wrist_image": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "right_wrist_image": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "proprio": np.array([0.5, -100.0, 90.0, 75.0, -2.0, 2.0], dtype=np.float32),
        }]
    print(f"观测数: {len(observations)}  丢弃槽位: {args.drop_slot} = {SLOT_NAME[args.drop_slot]}")

    def pad_to_state_t(lat):
        missing = state_t - lat.shape[2]
        if missing <= 0:
            return lat
        pad = torch.zeros(lat.shape[0], lat.shape[1], missing, lat.shape[3], lat.shape[4],
                          device=lat.device, dtype=lat.dtype)
        return torch.cat([lat, pad], dim=2)

    def make_encoder(mode):
        """mode: 'baseline' | 'zero' | 'copy_front' | 'copy_wrist'"""
        lo, hi = SLOT_FRAMES[args.drop_slot]

        def enc(x, *a, **kw):
            if not (torch.is_tensor(x) and x.dim() == 5 and x.shape[2] == TOTAL_FRAMES):
                return orig_encode(x, *a, **kw)
            if mode == "baseline":
                out = orig_encode(x[:, :, :PREFIX_FRAMES].contiguous(), *a, **kw)
                lat = out[0] if isinstance(out, (tuple, list)) else out
                return pad_to_state_t(lat)
            # Encode the prefix with the dropped slot's frames removed.
            keep = torch.cat([x[:, :, :lo], x[:, :, hi:PREFIX_FRAMES]], dim=2).contiguous()
            out = orig_encode(keep, *a, **kw)
            lat = out[0] if isinstance(out, (tuple, list)) else out  # (B,C,4,h,w)
            if mode == "zero":
                fill = torch.zeros_like(lat[:, :, :1])
            elif mode == "copy_front":
                fill = lat[:, :, -1:].clone()      # primary is the last encoded slot
            elif mode == "copy_wrist":
                fill = lat[:, :, -2:-1].clone()    # right_wrist sits just before primary
            else:
                raise ValueError(mode)
            lat = torch.cat([lat[:, :, : args.drop_slot], fill, lat[:, :, args.drop_slot :]], dim=2)
            return pad_to_state_t(lat)

        return enc

    def run(obs):
        return np.asarray(
            get_action(cfg, model, stats, obs, args.task, seed=195, randomize_seed=False,
                       num_denoising_steps_action=1,
                       generate_future_state_and_value_in_parallel=False)["actions"],
            dtype=np.float32,
        )

    modes = ["baseline", "zero", "copy_front", "copy_wrist"]
    per_mode_actions = {}
    for m in modes:
        tok.encode = make_encoder(m)
        try:
            per_mode_actions[m] = [run(o) for o in observations]
        finally:
            tok.encode = orig_encode

    base = per_mode_actions["baseline"]
    print(f"\n=== 动作偏差 vs 完整 3 视角 (n={len(observations)} 观测, 单位 deg) ===")
    print(f"{'方案':<16}{'max|Δ|':>10}{'mean|Δ|':>10}{'gripper max|Δ|':>16}")
    for m in modes[1:]:
        ds = [np.abs(b - t) for b, t in zip(base, per_mode_actions[m])]
        gd = [np.abs(b[:, 5] - t[:, 5]) for b, t in zip(base, per_mode_actions[m])]
        label = {"zero": "补零", "copy_front": "复制 front", "copy_wrist": "复制 wrist"}[m]
        print(f"{label:<16}{max(d.max() for d in ds):>10.3f}{np.mean([d.mean() for d in ds]):>10.3f}"
              f"{max(d.max() for d in gd):>16.3f}")

    print(f"\n=== 端到端耗时 (denoise=1, n={args.iters}) ===")
    obs0 = observations[0]
    for m in modes:
        tok.encode = make_encoder(m)
        try:
            run(obs0); sync()
            ts = []
            for _ in range(args.iters):
                sync(); t0 = time.perf_counter(); run(obs0); sync()
                ts.append((time.perf_counter() - t0) * 1000)
        finally:
            tok.encode = orig_encode
        frames = PREFIX_FRAMES if m == "baseline" else PREFIX_FRAMES - 4
        print(f"  {m:<12} 编码 {frames:>2} 帧   p50 = {statistics.median(ts):7.1f} ms")


if __name__ == "__main__":
    main()
