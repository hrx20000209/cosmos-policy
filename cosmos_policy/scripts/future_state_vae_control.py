"""A fair test of the world model's future frames: compare inside the VAE domain.

Comparing a decoded prediction against a raw camera frame is not a test of
prediction quality. The decoded frame has already lost whatever the VAE cannot
represent, and it is framed the way ``prepare_images_for_model`` frames it, so
even a perfect prediction scores far below a raw-pixel persistence baseline.
That is exactly what a first pass showed: predictions sat at SSIM ~0.70 while
raw persistence sat at ~0.94, and the prediction curve was flat in horizon.

This puts everything in the same domain:

    ceiling      SSIM(roundtrip(x_t), model_input(x_t))
                 what the VAE alone costs -- the best any prediction can score.
    prediction   SSIM(pred_t, roundtrip(x_{t+h}))
    persistence  SSIM(roundtrip(x_t), roundtrip(x_{t+h}))

If prediction beats persistence in this domain, the world model is contributing
something beyond "nothing changes"; if it only approaches the ceiling on the
static parts, it is reconstructing rather than predicting.
"""

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

# scikit-image is not installed in the cosmos env, and this is the only thing it
# would be pulled in for. Standard SSIM (Wang et al. 2004) with the usual 7x7
# uniform window, averaged over channels -- matches
# skimage.metrics.structural_similarity(..., gaussian_weights=False).
_C1, _C2 = (0.01 ** 2), (0.03 ** 2)


def _boxfilter(x, r=3):
    """Mean over a (2r+1)^2 window via a separable cumulative sum."""
    pad = np.pad(x, ((r + 1, r), (r + 1, r)), mode="edge")
    c = pad.cumsum(0)
    c = c[2 * r + 1 :] - c[: -(2 * r + 1)]
    c = c.cumsum(1)
    c = c[:, 2 * r + 1 :] - c[:, : -(2 * r + 1)]
    return c / ((2 * r + 1) ** 2)


def ssim(a, b, channel_axis=2, data_range=1.0, **_kw):
    vals = []
    for ch in range(a.shape[channel_axis]):
        x, y = a[..., ch].astype(np.float64), b[..., ch].astype(np.float64)
        mx, my = _boxfilter(x), _boxfilter(y)
        sxx = _boxfilter(x * x) - mx * mx
        syy = _boxfilter(y * y) - my * my
        sxy = _boxfilter(x * y) - mx * my
        c1, c2 = _C1 * data_range ** 2, _C2 * data_range ** 2
        num = (2 * mx * my + c1) * (2 * sxy + c2)
        den = (mx ** 2 + my ** 2 + c1) * (sxx + syy + c2)
        vals.append(np.mean(num / den))
    return float(np.mean(vals))

PRED_RE = re.compile(r"t(\d+)_(\d+\.\d+)_(\w+)_pred\.jpg")
ACT_RE = re.compile(r"t(\d+)_(\d+\.\d+)_(\w+)\.jpg")
TOTAL_FRAMES = 41


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step20000/model")
    ap.add_argument("--config", default="cosmos_predict2_2b_three_cubes_full_ft")
    ap.add_argument("--config-file", default="configs/eval_config.py")
    ap.add_argument("--future-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--horizons", default="0,0.5,1,2,3,4")
    args = ap.parse_args()

    from cosmos_policy.experiments.robot.cosmos_utils import get_model, prepare_images_for_model

    cfg = SimpleNamespace(config=args.config, ckpt_path=args.ckpt, config_file=args.config_file,
                          trained_with_image_aug=True, use_jpeg_compression=False, flip_images=False)
    model, _ = get_model(cfg)
    tok = model.tokenizer
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    run_dir, fut_dir = Path(args.run_dir), Path(args.future_dir)
    shot_dir = next(run_dir.glob("screenshots_*"))
    horizons = [float(h) for h in args.horizons.split(",")]

    def model_view(path):
        """The exact 224x224 the model is fed, via the real preprocessing."""
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        out = prepare_images_for_model([img], cfg)
        a = np.asarray(out)[0]
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = np.moveaxis(a, 0, -1)
        return a.astype(np.float64) / 255.0 if a.dtype == np.uint8 else np.clip(a, 0, 1).astype(np.float64)

    @torch.no_grad()
    def roundtrip(img01):
        """Encode then decode a single frame through the tokenizer."""
        x = torch.from_numpy(img01).permute(2, 0, 1)[None, :, None]  # B,C,1,H,W
        x = x.repeat(1, 1, TOTAL_FRAMES, 1, 1).to(dev, dtype) * 2 - 1
        lat = tok.encode(x)
        lat = lat[0] if isinstance(lat, (tuple, list)) else lat
        dec = tok.decode(lat)
        dec = dec[0] if isinstance(dec, (tuple, list)) else dec
        y = dec[0, :, -1].float().cpu().numpy()  # last frame
        y = np.moveaxis(y, 0, -1)
        return np.clip((y + 1) / 2, 0, 1).astype(np.float64)

    results = {}
    for cam in ("front", "right", "wrist"):
        acts = sorted((float(ACT_RE.fullmatch(p.name).group(2)), p)
                      for p in shot_dir.glob(f"*_{cam}.jpg") if ACT_RE.fullmatch(p.name))
        preds = sorted((float(PRED_RE.fullmatch(p.name).group(2)), p)
                       for p in fut_dir.glob(f"*_{cam}_pred.jpg") if PRED_RE.fullmatch(p.name))
        if not acts or not preds:
            continue
        times = np.array([t for t, _ in acts])
        step = max(1, len(preds) // args.n)
        preds = preds[::step][: args.n]

        rt_cache = {}
        def rt(idx):
            if idx not in rt_cache:
                rt_cache[idx] = roundtrip(model_view(acts[idx][1]))
            return rt_cache[idx]

        ceil, per_h = [], {h: {"pred": [], "pers": []} for h in horizons}
        for w, pp in preds:
            i0 = int(np.argmin(np.abs(times - w)))
            mv0, rt0 = model_view(acts[i0][1]), rt(i0)
            ceil.append(ssim(rt0, mv0, channel_axis=2, data_range=1.0))
            pred = np.asarray(Image.open(pp).convert("RGB"), dtype=np.float64) / 255.0
            if pred.shape[:2] != rt0.shape[:2]:
                pred = np.asarray(Image.open(pp).convert("RGB").resize(rt0.shape[1::-1]), dtype=np.float64) / 255.0
            for h in horizons:
                tgt = w + h
                if tgt > times[-1]:
                    continue
                i = int(np.argmin(np.abs(times - tgt)))
                if abs(times[i] - tgt) > 0.6:
                    continue
                rth = rt(i)
                per_h[h]["pred"].append(ssim(pred, rth, channel_axis=2, data_range=1.0))
                per_h[h]["pers"].append(ssim(rt0, rth, channel_axis=2, data_range=1.0))

        print(f"\n=== {cam} (n={len(preds)}) ===")
        print(f"  VAE 天花板 SSIM(往返, 模型输入) = {np.mean(ceil):.4f}   <- 任何预测的上限")
        print(f"  {'horizon':>8}{'预测':>10}{'持恒':>10}{'预测-持恒':>12}{'n':>5}")
        rec = {"vae_ceiling": float(np.mean(ceil)), "horizons": {}}
        for h in horizons:
            p, q = per_h[h]["pred"], per_h[h]["pers"]
            if not p:
                continue
            print(f"  {h:>8.2f}{np.mean(p):>10.4f}{np.mean(q):>10.4f}{np.mean(p) - np.mean(q):>+12.4f}{len(p):>5}")
            rec["horizons"][h] = {"pred": float(np.mean(p)), "persistence": float(np.mean(q)), "n": len(p)}
        results[cam] = rec

    out = Path(args.out or (run_dir / "analysis" / "future_state_vae_control.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
