"""P1 analysis: how fast does the action chunk converge across denoising checkpoints?

Consumes a ``run_p1_trajectory_dump.py`` output directory and produces the
per-(checkpoint, horizon) convergence tables, the near-vs-far comparison, the
task-stage breakdown, and the standalone-vs-checkpoint contrast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import sys  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.progressive_wam import metrics as M  # noqa: E402
from experiments.progressive_wam.task_stage import stage_of_step  # noqa: E402


HORIZONS = (1, 2, 4, 8, 16)


def bootstrap_ci(values: np.ndarray, iterations: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(iterations, values.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def build_frame(episodes: list[dict], dataset_stats: dict) -> pd.DataFrame:
    scale = M.per_dim_scale(dataset_stats, action_dim=7)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        stages = episode.get("stage_labels", [])
        for request in episode["requests"]:
            final = np.asarray(request["final_actions"], dtype=np.float64)
            chunks = np.asarray(request["checkpoint_actions"], dtype=np.float64)
            stage = stage_of_step(stages, int(request["control_step"]))
            for j in range(chunks.shape[0]):
                base = {
                    "episode_id": episode["episode_id"],
                    "task_suite": episode["task_suite"],
                    "task_id": episode["task_id"],
                    "seed": episode["seed"],
                    "success": episode["success"],
                    "request_id": request["request_id"],
                    "control_step": int(request["control_step"]),
                    "stage": stage,
                    "checkpoint": j + 1,
                    "sigma": float(request["sigmas"][j]),
                    "value": float(request["values"][j]) if np.isfinite(request["values"][j]) else np.nan,
                }
                bundle = M.checkpoint_metrics(chunks[j], final, scale, horizons=HORIZONS)
                for h in HORIZONS:
                    rows.append(
                        {
                            **base,
                            "horizon": h,
                            "normalized_l1": bundle[f"l1_h{h}"],
                            "normalized_l2": bundle[f"l2_h{h}"],
                            "cosine": bundle[f"cosine_h{h}"],
                            "sign_agreement": bundle[f"sign_h{h}"],
                            "gripper_agreement": bundle[f"gripper_h{h}"],
                            "weighted_error": bundle[f"weighted_error_h{h}"],
                            "endpoint_translation": bundle[f"endpoint_translation_h{h}"],
                            "endpoint_rotation": bundle[f"endpoint_rotation_h{h}"],
                            "intermediate_jerk_max": bundle["intermediate_jerk_max"],
                            "final_jerk_max": bundle["final_jerk_max"],
                            "intermediate_velocity_max": bundle["intermediate_velocity_max"],
                        }
                    )
    return pd.DataFrame(rows)


def build_per_token_frame(episodes: list[dict], dataset_stats: dict) -> pd.DataFrame:
    """Error of each individual action token k, not of the prefix ``[0:k]``.

    The prefix view mixes near and far tokens together, so it cannot answer
    "do near-term tokens converge earlier". This one can.
    """
    scale = M.per_dim_scale(dataset_stats, action_dim=7)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        stages = episode.get("stage_labels", [])
        for request in episode["requests"]:
            final = np.asarray(request["final_actions"], dtype=np.float64)
            chunks = np.asarray(request["checkpoint_actions"], dtype=np.float64)
            stage = stage_of_step(stages, int(request["control_step"]))
            for j in range(chunks.shape[0]):
                per_token = np.mean(np.abs(chunks[j] - final) / (scale[None, :] + 1e-8), axis=1)
                for k, value in enumerate(per_token):
                    rows.append(
                        {
                            "request_id": request["request_id"],
                            "task_suite": episode["task_suite"],
                            "stage": stage,
                            "checkpoint": j + 1,
                            "token": k,
                            "normalized_l1": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def build_future_frame(episodes: list[dict]) -> pd.DataFrame:
    """Convergence of the predicted future latent, on the same axis as the action.

    Both come from the *same* denoiser forward (Cosmos denoises action, future
    state and value jointly), so this compares like with like: at checkpoint j,
    is the imagined future settled to the same degree as the action?

    Errors are normalised by the final latent's own RMS so the action curve and
    the future curve are on a comparable scale despite living in different spaces.
    """
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        stages = episode.get("stage_labels", [])
        for request in episode["requests"]:
            future = request.get("future_latents")
            if future is None:
                continue
            arr = np.asarray(future, dtype=np.float32)  # (checkpoints, slots, C, H, W)
            final = arr[-1]
            denom = float(np.sqrt(np.mean(final.astype(np.float64) ** 2))) + 1e-8
            stage = stage_of_step(stages, int(request["control_step"]))
            for j in range(arr.shape[0]):
                delta = arr[j].astype(np.float64) - final.astype(np.float64)
                rows.append(
                    {
                        "request_id": request["request_id"],
                        "task_suite": episode["task_suite"],
                        "stage": stage,
                        "checkpoint": j + 1,
                        "future_rel_l1": float(np.mean(np.abs(delta)) / denom),
                        "future_rel_l2": float(np.sqrt(np.mean(delta**2)) / denom),
                    }
                )
    return pd.DataFrame(rows)


def plot_action_vs_future(frame: pd.DataFrame, future: pd.DataFrame, out_path: Path) -> None:
    if future.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    action = frame[frame.horizon == 16].groupby("checkpoint")["normalized_l2"].mean()
    action_rel = action / (action.iloc[0] if action.iloc[0] > 0 else 1.0)
    fut = future.groupby("checkpoint")["future_rel_l2"].mean()
    fut_rel = fut / (fut.iloc[0] if fut.iloc[0] > 0 else 1.0)
    ax.plot(action_rel.index, action_rel.to_numpy(), "-o", label="action chunk")
    ax.plot(fut_rel.index, fut_rel.to_numpy(), "-s", label="future latent (proprio+wrist+primary)")
    ax.set_xlabel("denoising checkpoint j")
    ax.set_ylabel("error relative to its own j=1 error")
    ax.set_title("Action vs imagined future: which settles first?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_token(frame: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    for checkpoint in sorted(frame.checkpoint.unique()):
        subset = frame[frame.checkpoint == checkpoint].groupby("token")["normalized_l1"].mean()
        if subset.max() <= 0:
            continue  # the final checkpoint is the reference: identically zero
        ax.plot(subset.index, subset.to_numpy(), "-o", label=f"j={checkpoint}")
    ax.set_xlabel("action token index k (0 = next action executed)")
    ax.set_ylabel("normalized L1 to final action")
    ax.set_title("Per-token convergence: do near-term tokens settle earlier?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_standalone_frame(episodes: list[dict], dataset_stats: dict) -> pd.DataFrame:
    """standalone_k_step sampler configuration vs the k-th checkpoint of the full run."""
    scale = M.per_dim_scale(dataset_stats, action_dim=7)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for request in episode["requests"]:
            standalone = request.get("standalone_actions")
            if not standalone:
                continue
            final = np.asarray(request["final_actions"], dtype=np.float64)
            chunks = np.asarray(request["checkpoint_actions"], dtype=np.float64)
            for k, actions in standalone.items():
                k = int(k)
                actions = np.asarray(actions, dtype=np.float64)
                rows.append(
                    {
                        "episode_id": episode["episode_id"],
                        "request_id": request["request_id"],
                        "k": k,
                        # standalone_k vs the k-th checkpoint of the full schedule
                        "standalone_vs_checkpoint_l1": M.normalized_l1(actions, chunks[k - 1], scale)
                        if k <= chunks.shape[0]
                        else np.nan,
                        "standalone_vs_checkpoint_max_abs": float(np.max(np.abs(actions - chunks[k - 1])))
                        if k <= chunks.shape[0]
                        else np.nan,
                        # both against the official final action
                        "standalone_vs_final_l1": M.normalized_l1(actions, final, scale),
                        "checkpoint_vs_final_l1": M.normalized_l1(chunks[k - 1], final, scale)
                        if k <= chunks.shape[0]
                        else np.nan,
                        "standalone_vs_final_cosine": M.cosine_similarity(actions, final),
                        "latency_ms": float(request["standalone_latency_ms"][k])
                        if request.get("standalone_latency_ms")
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, by: list[str], columns: list[str]) -> pd.DataFrame:
    out = []
    for key, group in frame.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n"] = len(group)
        for column in columns:
            values = group[column].to_numpy(dtype=np.float64)
            row[f"{column}_mean"] = float(np.nanmean(values)) if values.size else np.nan
            row[f"{column}_std"] = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0
            lo, hi = bootstrap_ci(values)
            row[f"{column}_ci_lo"] = lo
            row[f"{column}_ci_hi"] = hi
        out.append(row)
    return pd.DataFrame(out)


def plot_heatmap(frame: pd.DataFrame, out_path: Path) -> None:
    pivot = frame.pivot_table(index="checkpoint", columns="horizon", values="normalized_l1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    ax.set_xlabel("prefix horizon h (action tokens)")
    ax.set_ylabel("denoising checkpoint j")
    ax.set_title("Normalized L1 to final action\n(lower = already converged)")
    for yi in range(pivot.shape[0]):
        for xi in range(pivot.shape[1]):
            value = pivot.to_numpy()[yi, xi]
            ax.text(xi, yi, f"{value:.3f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="normalized L1")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_near_vs_far(frame: pd.DataFrame, out_path: Path) -> None:
    final_checkpoint = int(frame.checkpoint.max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    for horizon, style in ((1, "-o"), (2, "-s"), (4, "-^"), (8, "-v"), (16, "-d")):
        rows = frame[frame.horizon == horizon]
        # j = final is the reference and is identically zero; excluded from the log axis.
        subset = rows[rows.checkpoint < final_checkpoint].groupby("checkpoint")["normalized_l1"].mean()
        axes[0].plot(subset.index, subset.to_numpy(), style, label=f"h={horizon}")
        subset_c = frame[frame.horizon == horizon].groupby("checkpoint")["cosine"].mean()
        axes[1].plot(subset_c.index, subset_c.to_numpy(), style, label=f"h={horizon}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("denoising checkpoint j")
    axes[0].set_ylabel("normalized L1 to final")
    axes[0].set_title("Near-term prefixes converge earlier?")
    axes[1].set_xlabel("denoising checkpoint j")
    axes[1].set_ylabel("cosine to final")
    axes[1].set_title("Cosine similarity by prefix horizon")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_by_stage(frame: pd.DataFrame, out_path: Path) -> None:
    stages = [s for s in frame.stage.unique() if isinstance(s, str)]
    # The final checkpoint is the reference (error identically zero); including it
    # on a log axis compresses every real curve into the top decade.
    final_checkpoint = int(frame.checkpoint.max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=300)
    for stage in sorted(stages):
        rows = frame[(frame.stage == stage) & (frame.horizon == 4)]
        requests = rows.request_id.nunique()
        subset = rows[rows.checkpoint < final_checkpoint].groupby("checkpoint")["normalized_l1"].mean()
        axes[0].plot(subset.index, subset.to_numpy(), "-o", label=f"{stage} (n={requests})")
        subset_g = rows.groupby("checkpoint")["gripper_agreement"].mean()
        axes[1].plot(subset_g.index, subset_g.to_numpy(), "-o", label=stage)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("denoising checkpoint j")
    axes[0].set_ylabel("normalized L1 (h=4)")
    axes[0].set_title(f"Convergence by task stage (prefix h=4, j<{final_checkpoint})")
    axes[1].set_xlabel("denoising checkpoint j")
    axes[1].set_ylabel("gripper agreement (h=4)")
    axes[1].set_title("Gripper command agreement by stage")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_report(
    out_path: Path,
    run_dir: Path,
    metadata: dict,
    by_checkpoint: pd.DataFrame,
    by_stage: pd.DataFrame,
    standalone: pd.DataFrame,
    frame: pd.DataFrame,
    per_token: pd.DataFrame,
    future: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# P1 Denoising Trajectory Audit (Cosmos Policy / LIBERO)\n")
    lines.append(f"来源 run：`{run_dir}`\n")
    summary = metadata.get("summary", {})
    lines.append(
        f"- episodes：{summary.get('episodes')}，成功 {summary.get('successes')}\n"
        f"- 总 requests：{summary.get('total_requests')}\n"
        f"- hook 最终 checkpoint 与官方输出的最大绝对差："
        f"`{summary.get('max_final_agreement_abs'):.3e}`（0 表示逐位相同）\n"
    )
    lines.append("\n## 1. 每个 checkpoint × prefix horizon 的收敛\n")
    lines.append("`normalized_l1` 以 dataset action std 归一化，`checkpoint=5` 是官方最终输出（误差恒为 0）。\n")
    pivot = frame.pivot_table(index="checkpoint", columns="horizon", values="normalized_l1", aggfunc="mean")
    lines.append("\n| checkpoint | " + " | ".join(f"h={c}" for c in pivot.columns) + " |")
    lines.append("|---:|" + "---:|" * len(pivot.columns))
    for idx, row in pivot.iterrows():
        lines.append(f"| {idx} | " + " | ".join(f"{v:.4f}" for v in row) + " |")

    lines.append("\n\n## 2. 逐 checkpoint 汇总（h=4，含 bootstrap 95% CI）\n")
    sub = by_checkpoint[by_checkpoint.horizon == 4]
    lines.append("| j | sigma | n | L1 mean | 95% CI | cosine mean | sign agree | gripper agree |")
    lines.append("|---:|---:|---:|---:|---|---:|---:|---:|")
    for _, row in sub.iterrows():
        lines.append(
            f"| {int(row['checkpoint'])} | {row['sigma_mean']:.2f} | {int(row['n'])} | "
            f"{row['normalized_l1_mean']:.4f} | [{row['normalized_l1_ci_lo']:.4f}, {row['normalized_l1_ci_hi']:.4f}] | "
            f"{row['cosine_mean']:.6f} | {row['sign_agreement_mean']:.4f} | {row['gripper_agreement_mean']:.4f} |"
        )

    lines.append("\n\n## 3. 近期 vs 远期 token\n")
    near = frame[frame.horizon == 1].groupby("checkpoint")["normalized_l1"].mean()
    far = frame[frame.horizon == 16].groupby("checkpoint")["normalized_l1"].mean()
    lines.append("| j | h=1 L1 | h=16 L1 | ratio (near/far) |")
    lines.append("|---:|---:|---:|---:|")
    for j in near.index:
        # The final checkpoint is the reference, so both errors are float noise
        # there; a ratio of noise/noise is meaningless and is reported as n/a.
        ratio = near[j] / far[j] if far[j] > 1e-9 else float("nan")
        ratio_text = "n/a" if not np.isfinite(ratio) else f"{ratio:.3f}"
        lines.append(f"| {j} | {near[j]:.4f} | {far[j]:.4f} | {ratio_text} |")
    lines.append(
        "\nratio < 1 表示近期 token 比远期 token 更早收敛，这是「progressive prefix commitment」"
        "假设的核心可证伪点。注意 h=1 只含第 0 个 token，h=16 是全部 16 个 token 的平均，"
        "所以 ratio > 1 意味着最先执行的那个 action 反而比 chunk 平均更不收敛。\n"
    )

    lines.append("\n### 3b. 逐 token 收敛（直接证据）\n")
    pivot_token = per_token.pivot_table(index="checkpoint", columns="token", values="normalized_l1", aggfunc="mean")
    show_tokens = [t for t in pivot_token.columns if t in (0, 1, 2, 3, 7, 11, 15)]
    lines.append(
        "| j | " + " | ".join(f"k={t}" for t in show_tokens) + " | near k0-3 | mid k6-9 | far k12-15 | near/far |"
    )
    lines.append("|---:|" + "---:|" * (len(show_tokens) + 4))
    for idx, row in pivot_token.iterrows():
        values = row.to_numpy()
        near = float(values[0:4].mean())
        mid = float(values[6:10].mean())
        far = float(values[12:16].mean())
        # A single slope is a bad summary when the curve is U-shaped in k, so the
        # three band means are reported instead.
        ratio_text = "n/a" if far <= 1e-9 else f"{near / far:.3f}"
        lines.append(
            f"| {idx} | "
            + " | ".join(f"{row[t]:.4f}" for t in show_tokens)
            + f" | {near:.4f} | {mid:.4f} | {far:.4f} | {ratio_text} |"
        )
    lines.append(
        "\n`near/far < 1` 才支持「近期 token 更早收敛」。若 `mid` 明显低于 `near` 和 `far`，"
        "说明误差沿 token 索引是 U 形而不是单调的，此时任何单一斜率都是误导性的摘要。\n"
    )

    lines.append("\n### 3c. action vs 想象的未来（同一次 forward）\n")
    if future.empty:
        lines.append("本 run 未采集 future latent（`--no-capture-future-latent`）。\n")
    else:
        action_curve = frame[frame.horizon == 16].groupby("checkpoint")["normalized_l2"].mean()
        future_curve = future.groupby("checkpoint")["future_rel_l2"].mean()
        a0 = action_curve.iloc[0] if action_curve.iloc[0] > 0 else 1.0
        f0 = future_curve.iloc[0] if future_curve.iloc[0] > 0 else 1.0
        lines.append("| j | action L2 | action 相对 j=1 | future 相对 L2 | future 相对 j=1 |")
        lines.append("|---:|---:|---:|---:|---:|")
        for j in action_curve.index:
            if j not in future_curve.index:
                continue
            lines.append(
                f"| {j} | {action_curve[j]:.5f} | {action_curve[j] / a0:.3f} | "
                f"{future_curve[j]:.5f} | {future_curve[j] / f0:.3f} |"
            )
        lines.append(
            "\nCosmos 的 action、future state 和 value 由 **同一次 DiT forward** 联合去噪，"
            "所以两条曲线可以直接比较。若 future 的相对曲线下降更慢，说明「先拿 action、"
            "让未来继续去噪」在结构上是可能的；若两者同步下降，则 action 并没有比未来更早稳定。\n"
        )

    lines.append("\n## 4. 按任务阶段\n")
    stage_sub = by_stage[by_stage.horizon == 4]
    lines.append("| stage | j | n | L1 mean | cosine | gripper agree |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in stage_sub.sort_values(["stage", "checkpoint"]).iterrows():
        lines.append(
            f"| {row['stage']} | {int(row['checkpoint'])} | {int(row['n'])} | "
            f"{row['normalized_l1_mean']:.4f} | {row['cosine_mean']:.6f} | {row['gripper_agreement_mean']:.4f} |"
        )

    lines.append("\n\n## 5. standalone_k_step vs 完整 schedule 的第 k 个 checkpoint\n")
    if standalone.empty:
        lines.append("本 run 未采集 standalone 探针（`--standalone-stride 0`）。\n")
    else:
        agg = standalone.groupby("k").mean(numeric_only=True)
        lines.append("| k | standalone vs checkpoint_k (max abs) | standalone vs final L1 | checkpoint_k vs final L1 | standalone latency ms |")
        lines.append("|---:|---:|---:|---:|---:|")
        for k, row in agg.iterrows():
            lines.append(
                f"| {int(k)} | {row['standalone_vs_checkpoint_max_abs']:.3e} | "
                f"{row['standalone_vs_final_l1']:.4f} | {row['checkpoint_vs_final_l1']:.4f} | "
                f"{row['latency_ms']:.1f} |"
            )
        lines.append(
            "\n`k=1` 的 max abs 应当为 0：EDM σ 网格的起点恒为 σ_max，"
            "所以 standalone_one_step 与任意 schedule 的第 1 个 predicted-clean 是同一次前向。"
            "`k>1` 出现差异是预期的，因为两者的中间 σ 网格不同。\n"
        )

    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", required=True, nargs="+")
    parser.add_argument("--label", default=None, help="name for the merged analysis directory")
    parser.add_argument("--output-root", default="/data/rxhuang/wam_progressive_outputs")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.trajectory_dir]
    episodes: list[dict] = []
    metadata: dict = {}
    for run_dir in run_dirs:
        episodes.extend(torch.load(run_dir / "checkpoints.pt", weights_only=False))
        part = json.loads((run_dir / "metadata.json").read_text())
        if not metadata:
            metadata = part
        else:
            # Merged runs differ only in the suite/task split; the summary must
            # reflect the union, not whichever part was loaded first.
            for key in ("episodes", "successes", "total_requests"):
                metadata["summary"][key] += part["summary"][key]
            metadata["summary"]["max_final_agreement_abs"] = max(
                metadata["summary"]["max_final_agreement_abs"], part["summary"]["max_final_agreement_abs"]
            )
    run_dir = run_dirs[0] if len(run_dirs) == 1 else Path(args.label or "merged")
    dataset_stats = {k: np.asarray(v) for k, v in metadata["dataset_stats"].items()}

    frame = build_frame(episodes, dataset_stats)
    per_token = build_per_token_frame(episodes, dataset_stats)
    future = build_future_frame(episodes)
    standalone = build_standalone_frame(episodes, dataset_stats)

    columns = [
        "normalized_l1",
        "normalized_l2",
        "cosine",
        "sign_agreement",
        "gripper_agreement",
        "weighted_error",
        "endpoint_translation",
        "endpoint_rotation",
        "sigma",
    ]
    by_checkpoint = summarise(frame, ["checkpoint", "horizon"], columns)
    by_stage = summarise(frame, ["stage", "checkpoint", "horizon"], columns)

    out_root = Path(args.output_root)
    plots = out_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    tables = out_root / "audits" / run_dir.name
    tables.mkdir(parents=True, exist_ok=True)

    frame.to_csv(tables / "p1_checkpoint_metrics.csv", index=False)
    per_token.to_csv(tables / "p1_per_token.csv", index=False)
    by_checkpoint.to_csv(tables / "p1_by_checkpoint.csv", index=False)
    by_stage.to_csv(tables / "p1_by_stage.csv", index=False)
    if not standalone.empty:
        standalone.to_csv(tables / "p1_standalone_vs_checkpoint.csv", index=False)

    plot_heatmap(frame, plots / "action_error_step_horizon_heatmap.png")
    plot_near_vs_far(frame, plots / "near_vs_far_action_convergence.png")
    plot_by_stage(frame, plots / "action_convergence_by_task_stage.png")
    plot_per_token(per_token, plots / "per_token_convergence.png")
    plot_action_vs_future(frame, future, plots / "action_vs_future_convergence.png")
    if not future.empty:
        future.to_csv(tables / "p1_future_latent.csv", index=False)

    report_path = Path(args.report) if args.report else REPO_ROOT / "reports" / "trajectory_audit.md"
    write_report(report_path, run_dir, metadata, by_checkpoint, by_stage, standalone, frame, per_token, future)
    print(f"[p1-analysis] tables -> {tables}")
    print(f"[p1-analysis] plots  -> {plots}")
    print(f"[p1-analysis] report -> {report_path}")


if __name__ == "__main__":
    main()
