"""阶段 3.1：从 metrics.jsonl 自动重绘 loss 曲线（每次 eval 后覆盖保存到固定路径）。

产物（默认写到 <run_dir>/plots/）：
  - loss_train.png : train video / action / 总 loss，原始值 + EMA 平滑叠加，log y。
  - loss_det_val.png : 确定性验证 loss（video / action 分开），log y。
x 轴同时标 step 和（若给了 --samples-per-epoch）epoch。

metrics.jsonl 由仓库 WandbCallback（split="train"/"val"）与本项目
DeterministicValLossCallback（split="det_val"）追加写入。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe–Ito
C_TOTAL = "#000000"
C_ACTION = "#D55E00"
C_VIDEO = "#0072B2"
C_WRIST = "#56B4E9"


def _load(metrics_path: Path, split: str) -> list[dict]:
    rows = []
    if not metrics_path.is_file():
        return rows
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("split") == split and "iteration" in rec:
            rows.append(rec)
    rows.sort(key=lambda r: r["iteration"])
    return rows


def _series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            xs.append(r["iteration"])
            ys.append(fv)
    return np.asarray(xs, float), np.asarray(ys, float)


def _ema(y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    if len(y) == 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def _add_epoch_axis(ax, samples_per_epoch: float | None) -> None:
    if not samples_per_epoch or samples_per_epoch <= 0:
        return
    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda s: s / samples_per_epoch, lambda e: e * samples_per_epoch),
    )
    sec.set_xlabel("epoch")


def _plot_train(rows: list[dict], out: Path, samples_per_epoch: float | None) -> bool:
    specs = [
        ("total loss (EDM)", "train/loss", C_TOTAL),
        ("action mse (raw)", "train/demo_sample_action_mse_loss", C_ACTION),
        ("future-image mse (raw)", "train/demo_sample_future_image_mse_loss", C_VIDEO),
        ("future-wrist-image mse (raw)", "train/demo_sample_future_wrist_image_mse_loss", C_WRIST),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    any_data = False
    for label, key, color in specs:
        x, y = _series(rows, key)
        if len(x) == 0:
            continue
        any_data = True
        ax.plot(x, y, color=color, alpha=0.28, lw=1.0)
        ax.plot(x, _ema(y), color=color, lw=2.0, label=label)
    if not any_data:
        plt.close(fig)
        return False
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Training loss — raw (faint) + EMA (bold)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    _add_epoch_axis(ax, samples_per_epoch)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def _plot_det_val(rows: list[dict], out: Path, samples_per_epoch: float | None) -> bool:
    specs = [
        ("action mse", "det_val/action_mse", C_ACTION),
        ("future-image mse (video)", "det_val/future_image_mse", C_VIDEO),
        ("future-wrist-image mse", "det_val/future_wrist_image_mse", C_WRIST),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    any_data = False
    # 左：action，右：video（分开，尺度不同）
    for (label, key, color), which in zip(specs, ("action", "video", "video")):
        x, y = _series(rows, key)
        if len(x) == 0:
            continue
        any_data = True
        ax = axes[0] if which == "action" else axes[1]
        ax.plot(x, y, color=color, lw=2.0, marker="o", ms=3, label=label)
    if not any_data:
        plt.close(fig)
        return False
    axes[0].set_title("Deterministic val — action")
    axes[1].set_title("Deterministic val — video (future images)")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("EDM mse (log scale)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=9)
        _add_epoch_axis(ax, samples_per_epoch)
    fig.suptitle("Deterministic validation loss (fixed samples / fixed sigma grid / fixed noise)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path, help="含 metrics.jsonl 的 run 目录")
    parser.add_argument("--metrics", type=Path, default=None, help="覆盖 metrics.jsonl 路径")
    parser.add_argument("--out_dir", type=Path, default=None, help="图输出目录，默认 <run_dir>/plots")
    parser.add_argument("--samples-per-epoch", type=float, default=None,
                        help="训练集有效样本数（用于叠加 epoch 轴），可选")
    args = parser.parse_args()

    metrics_path = args.metrics or (args.run_dir / "metrics.jsonl")
    out_dir = args.out_dir or (args.run_dir / "plots")

    train_rows = _load(metrics_path, "train")
    det_rows = _load(metrics_path, "det_val")
    a = _plot_train(train_rows, out_dir / "loss_train.png", args.samples_per_epoch)
    b = _plot_det_val(det_rows, out_dir / "loss_det_val.png", args.samples_per_epoch)
    print(f"train rows={len(train_rows)} det_val rows={len(det_rows)}")
    print(f"loss_train.png: {'saved' if a else 'no data'} | loss_det_val.png: {'saved' if b else 'no data'}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
