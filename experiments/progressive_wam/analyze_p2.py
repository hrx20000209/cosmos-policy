"""P2 analysis: earliest reliable denoising step and oracle opportunity.

Computes ``j*(h) = min{ j : oracle_reliable(j, h) }`` per request, its CDF, the
coverage per prefix length, the task-stage breakdown, and a sensitivity analysis
over the oracle thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.progressive_wam.run_p2_oracle import OracleThresholds  # noqa: E402
from experiments.progressive_wam.task_stage import stage_of_step  # noqa: E402


NO_RELIABLE_STEP = np.inf


def load_tier1(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def attach_stage(frame: pd.DataFrame, trajectory_dirs: list[Path]) -> pd.DataFrame:
    stages: dict[str, list[str]] = {}
    for trajectory_dir in trajectory_dirs:
        for ep in torch.load(trajectory_dir / "checkpoints.pt", weights_only=False):
            stages[ep["episode_id"]] = ep.get("stage_labels", [])
    frame = frame.copy()
    frame["stage"] = [
        stage_of_step(stages.get(row.episode_id, []), int(row.control_step)) for row in frame.itertuples()
    ]
    return frame


def relabel(frame: pd.DataFrame, thresholds: OracleThresholds) -> pd.Series:
    """Recompute ``oracle_reliable`` from the stored raw measurements.

    Storing the measurements rather than only the boolean is what makes the
    sensitivity analysis possible without re-running the simulator.
    """
    # Joint limits are scored *against the reference*, not against zero: the
    # official policy itself sits on a limit in ~2.3% of these states, and both
    # branches inherit it. Charging the branch for a violation the reference also
    # has would report a failure that early exit did not cause.
    reference_violations = frame.get("reference_joint_limit_violations")
    if reference_violations is None:
        reference_violations = 0
    ok = (
        (frame["eef_position_error"] < thresholds.eef_position_error)
        & (frame["eef_rotation_error"] < thresholds.eef_rotation_error)
        & (frame["object_position_error_max"] < thresholds.object_position_error)
        & (frame["gripper_width_error"] < thresholds.gripper_width_error)
        & (frame["joint_limit_violations"] <= reference_violations)
        & (frame["prefix_jerk_max"] < thresholds.max_jerk)
        & (frame["prefix_velocity_max"] < thresholds.max_velocity)
    )
    if not thresholds.allow_gripper_mismatch:
        ok = ok & (~frame["gripper_transition_mismatch"].astype(bool))
    return ok


def earliest_reliable(frame: pd.DataFrame, reliable_column: str = "oracle_reliable") -> pd.DataFrame:
    """j*(h) per (request, prefix). ``inf`` when no checkpoint qualifies.

    ``checkpoint == 0`` is the identical-action determinism control, not a real
    denoising checkpoint, so it never counts as an early exit.
    """
    out: list[dict[str, Any]] = []
    frame = frame[frame["checkpoint"] >= 1]
    for (request_id, prefix), group in frame.groupby(["request_id", "prefix_length"]):
        group = group.sort_values("checkpoint")
        reliable = group[group[reliable_column].astype(bool)]
        j_star = float(reliable["checkpoint"].min()) if len(reliable) else NO_RELIABLE_STEP
        first = group.iloc[0]
        out.append(
            {
                "request_id": request_id,
                "episode_id": first["episode_id"],
                "task_suite": first["task_suite"],
                "task_id": first["task_id"],
                "control_step": first["control_step"],
                "stage": first.get("stage", "unknown"),
                "prefix_length": prefix,
                "j_star": j_star,
                "n_checkpoints": int(group["checkpoint"].max()),
            }
        )
    return pd.DataFrame(out)


def plot_cdf(js: pd.DataFrame, out_path: Path, n_full: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for prefix in sorted(js.prefix_length.unique()):
        subset = js[js.prefix_length == prefix]["j_star"].to_numpy()
        grid = np.arange(1, n_full + 1)
        cdf = [(subset <= j).mean() for j in grid]
        ax.plot(grid, cdf, "-o", label=f"prefix h={prefix}")
    ax.set_xlabel("denoising checkpoint j")
    ax.set_ylabel("fraction of states with j*(h) <= j")
    ax.set_ylim(0, 1.02)
    ax.set_title("Earliest reliable denoising step (oracle)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_coverage(frame: pd.DataFrame, out_path: Path) -> None:
    pivot = frame.pivot_table(
        index="checkpoint", columns="prefix_length", values="oracle_reliable", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for prefix in pivot.columns:
        ax.plot(pivot.index, pivot[prefix].to_numpy(), "-o", label=f"h={prefix}")
    if 0 in pivot.index:
        floor = float(pivot.loc[0].mean())
        ax.axhline(floor, ls="--", c="k", lw=1, label=f"determinism control ({floor:.3f})")
    ax.set_xlabel("denoising checkpoint j")
    ax.set_ylabel("fraction of branches labelled reliable")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Oracle coverage per checkpoint and prefix length")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_by_stage(js: pd.DataFrame, out_path: Path, n_full: int) -> None:
    stages = sorted(s for s in js.stage.unique() if isinstance(s, str))
    prefixes = sorted(js.prefix_length.unique())
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    width = 0.8 / max(1, len(prefixes))
    x = np.arange(len(stages))
    for i, prefix in enumerate(prefixes):
        values = []
        for stage in stages:
            subset = js[(js.stage == stage) & (js.prefix_length == prefix)]["j_star"].to_numpy()
            finite = subset[np.isfinite(subset)]
            # Un-reliable states are charged the full schedule rather than dropped,
            # so a stage that is never safe cannot look cheap.
            charged = np.where(np.isfinite(subset), subset, n_full)
            values.append(charged.mean() if charged.size else np.nan)
        ax.bar(x + i * width, values, width, label=f"h={prefix}")
    ax.set_xticks(x + 0.4 - width / 2, stages, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"mean j*(h) (unreliable charged as {n_full})")
    ax.set_title("Oracle opportunity by task stage")
    ax.axhline(n_full, ls="--", c="k", lw=1, label="full schedule")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def sensitivity(frame: pd.DataFrame, base: OracleThresholds, n_full: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sweeps = {
        "eef_position_error": [0.005, 0.01, 0.02, 0.04, 0.08],
        "object_position_error": [0.002, 0.005, 0.01, 0.02, 0.05],
        "gripper_width_error": [0.002, 0.005, 0.01, 0.02, 0.05],
        # Safety caps swept down through the baseline distribution: 0.48 is the
        # baseline jerk p99, 1.30 is just above its max.
        "max_jerk": [0.20, 0.48, 0.80, 1.30],
        "max_velocity": [0.73, 1.08, 1.18, 1.35],
    }
    for field, values in sweeps.items():
        for value in values:
            thresholds = OracleThresholds(**{**base.__dict__, field: value})
            labels = relabel(frame, thresholds)
            local = frame.assign(oracle_reliable_sweep=labels)
            js = earliest_reliable(local, "oracle_reliable_sweep")
            for prefix in sorted(js.prefix_length.unique()):
                subset = js[js.prefix_length == prefix]["j_star"].to_numpy()
                charged = np.where(np.isfinite(subset), subset, n_full)
                rows.append(
                    {
                        "threshold": field,
                        "value": value,
                        "prefix_length": prefix,
                        "reliable_fraction_any": float(np.isfinite(subset).mean()),
                        "mean_j_star_charged": float(charged.mean()),
                        "nfe_saved_mean": float(n_full - charged.mean()),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    out_path: Path,
    tier1_path: Path,
    frame: pd.DataFrame,
    js: pd.DataFrame,
    sens: pd.DataFrame,
    tier2: pd.DataFrame | None,
    thresholds: OracleThresholds,
    n_full: int,
) -> None:
    lines = ["# P2 Oracle Opportunity (Simulator Branched Rollout)\n"]
    lines.append(f"来源：`{tier1_path}`\n")
    lines.append(
        f"- 分支比较条数：{len(frame)}\n"
        f"- 覆盖 requests：{frame.request_id.nunique()}，episodes：{frame.episode_id.nunique()}\n"
        f"- 完整 schedule 步数 N = {n_full}\n"
    )
    lines.append("\n阈值（全部可配置，见下方 sensitivity）：\n\n```json\n")
    lines.append(json.dumps(thresholds.__dict__, indent=2))
    lines.append("\n```\n")

    control = frame[frame["checkpoint"] == 0]
    lines.append("\n## 0. 确定性对照（checkpoint 0 = 重放 reference 自己的 chunk）\n")
    if control.empty:
        lines.append(
            "本 run 没有对照分支（早于该功能的运行）。此时无法区分「checkpoint 质量差异」"
            "与「模拟器复现噪声」，所有可靠比例只能作为上界读。\n"
        )
    else:
        lines.append("| 量 | median | p95 | max |")
        lines.append("|---|---:|---:|---:|")
        for column in ("eef_position_error", "object_position_error_max", "gripper_width_error"):
            values = control[column].to_numpy()
            lines.append(
                f"| {column} | {np.median(values):.3e} | {np.percentile(values, 95):.3e} | {values.max():.3e} |"
            )
        lines.append(
            f"\n对照分支被判为不可靠的比例：**{1 - control['oracle_reliable'].mean():.4f}**"
            f"（n={len(control)}）。这就是本实验的 **噪声地板**：任何 checkpoint 的不可靠率"
            "都必须与它比较，而不是与 0 比较。\n"
        )

    lines.append("\n## 1. 每个 checkpoint × prefix 的 oracle 可靠比例\n")
    pivot = frame.pivot_table(index="checkpoint", columns="prefix_length", values="oracle_reliable", aggfunc="mean")
    lines.append("| j | " + " | ".join(f"h={c}" for c in pivot.columns) + " |")
    lines.append("|---:|" + "---:|" * len(pivot.columns))
    for idx, row in pivot.iterrows():
        label = "0 (control)" if idx == 0 else str(idx)
        lines.append(f"| {label} | " + " | ".join(f"{v:.3f}" for v in row) + " |")

    lines.append("\n\n## 2. 最早可靠步 j*(h)\n")
    lines.append("| h | 有可靠步的比例 | mean j* (不可靠按 N 计) | median j* | 可节省 NFE（均值） |")
    lines.append("|---:|---:|---:|---:|---:|")
    for prefix in sorted(js.prefix_length.unique()):
        subset = js[js.prefix_length == prefix]["j_star"].to_numpy()
        charged = np.where(np.isfinite(subset), subset, n_full)
        lines.append(
            f"| {prefix} | {np.isfinite(subset).mean():.3f} | {charged.mean():.3f} | "
            f"{np.median(charged):.1f} | {n_full - charged.mean():.3f} |"
        )

    lines.append("\n\n## 3. 按任务阶段\n")
    lines.append("| stage | h | n | 有可靠步比例 | mean j* (charged) |")
    lines.append("|---|---:|---:|---:|---:|")
    for (stage, prefix), group in js.groupby(["stage", "prefix_length"]):
        subset = group["j_star"].to_numpy()
        charged = np.where(np.isfinite(subset), subset, n_full)
        lines.append(
            f"| {stage} | {prefix} | {len(group)} | {np.isfinite(subset).mean():.3f} | {charged.mean():.3f} |"
        )

    lines.append("\n\n## 4. 失败原因分布\n")
    lines.append(
        "下表是 **运行时** 记录的原因（`joint_limit` 按绝对违反数判定）。"
        "本报告的主表已改用比较式判定：只有当分支的违反数 **超过 reference** 时才算失败，"
        "因为官方策略自身在约 2.3% 的状态下就贴着关节限位，两个分支都会继承。\n"
    )
    reasons: dict[str, int] = {}
    for entry in frame["failure_reasons"]:
        for reason in entry if isinstance(entry, list) else []:
            reasons[reason] = reasons.get(reason, 0) + 1
    lines.append("| reason | count | 占全部分支比例 |")
    lines.append("|---|---:|---:|")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {count} | {count / max(1, len(frame)):.3f} |")
    if "oracle_reliable_as_run" in frame.columns:
        lines.append(
            f"\n运行时判定的可靠率 **{frame['oracle_reliable_as_run'].mean():.4f}**，"
            f"比较式修正后 **{frame['oracle_reliable'].mean():.4f}**。\n"
        )

    lines.append("\n\n## 5. 阈值敏感性\n")
    lines.append("| threshold | value | h | 有可靠步比例 | mean j* | 节省 NFE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in sens.iterrows():
        lines.append(
            f"| {row['threshold']} | {row['value']} | {int(row['prefix_length'])} | "
            f"{row['reliable_fraction_any']:.3f} | {row['mean_j_star_charged']:.3f} | {row['nfe_saved_mean']:.3f} |"
        )

    lines.append("\n\n## 6. Tier-2 成功率对照\n")
    if tier2 is None or tier2.empty:
        lines.append("未运行 tier2（`--tier tier1`）。因此本报告 **不能** 回答成功率问题，只能回答状态偏离与安全。\n")
    else:
        lines.append("| j | h | n | reference success | branch success | success delta |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for (j, prefix), group in tier2.groupby(["checkpoint", "prefix_length"]):
            lines.append(
                f"| {int(j)} | {int(prefix)} | {len(group)} | {group['reference_success'].mean():.3f} | "
                f"{group['branch_success'].mean():.3f} | {group['success_delta'].mean():+.3f} |"
            )

    lines.append(
        "\n\n## 7. 口径说明\n\n"
        "- 这是 **feasibility upper bound**：oracle 使用了 simulator 的特权状态访问，"
        "在线系统拿不到这些量。\n"
        "- `visual_pixel_l2_*` 是像素空间距离，不是 VAE latent 距离，名称已如实标注。\n"
        "- 「不可靠」的 request 在均值中按完整 schedule N 计费，不是丢弃，"
        "所以节省的 NFE 不会被幸存者偏差抬高。\n"
        "- checkpoint = N 的动作与 reference 只差 ~1e-7，但它的可靠率**不是 1**，"
        "与 checkpoint 0 对照一致 —— 因为 `regenerate_obs_from_state` 不完全幂等，"
        "分支执行次序本身会带来约 1e-3 m 的 eef 漂移。分支顺序已按 request-id 派生的固定种子随机化，"
        "使这项漂移对所有 checkpoint 期望相同；checkpoint 0 用来量化它。\n"
        "- 关节限位按「是否超过 reference 的违反数」判定，不按绝对值，"
        "因为官方策略自身在约 2.3% 的状态下就贴着限位。\n"
    )
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", required=True, nargs="+")
    parser.add_argument("--trajectory-dir", required=True, nargs="+")
    parser.add_argument("--label", default=None, help="name for the merged analysis directory")
    parser.add_argument("--output-root", default="/data/rxhuang/wam_progressive_outputs")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    oracle_dirs = [Path(p) for p in args.oracle_dir]
    tier1_paths = [d / "tier1_branches.jsonl" for d in oracle_dirs if (d / "tier1_branches.jsonl").is_file()]
    if not tier1_paths:
        raise FileNotFoundError(f"no tier1_branches.jsonl under {oracle_dirs}")
    frame = pd.concat([load_tier1(p) for p in tier1_paths], ignore_index=True)
    frame = attach_stage(frame, [Path(p) for p in args.trajectory_dir])
    n_full = int(frame["checkpoint"].max())

    metadata = json.loads((tier1_paths[0].parent / "metadata.json").read_text())
    thresholds = OracleThresholds(**metadata["config"]["thresholds"])
    # Recompute the label from the stored raw measurements so the joint-limit
    # correction in relabel() applies to the headline numbers too, not only to
    # the sensitivity sweep.
    frame["oracle_reliable_as_run"] = frame["oracle_reliable"]
    frame["oracle_reliable"] = relabel(frame, thresholds)
    oracle_dir = oracle_dirs[0] if len(oracle_dirs) == 1 else Path(args.label or "merged")
    tier1_path = tier1_paths[0] if len(tier1_paths) == 1 else Path(f"{len(tier1_paths)} runs: {tier1_paths}")

    js = earliest_reliable(frame)
    sens = sensitivity(frame, thresholds, n_full)

    tier2_rows: list[dict] = []
    for d in oracle_dirs:
        path = d / "tier2_success.jsonl"
        if path.is_file():
            tier2_rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    tier2 = pd.DataFrame(tier2_rows) if tier2_rows else None

    out_root = Path(args.output_root)
    plots = out_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    tables = out_root / "oracle" / oracle_dir.name / "analysis"
    tables.mkdir(parents=True, exist_ok=True)

    frame.to_csv(tables / "p2_branches.csv", index=False)
    js.to_csv(tables / "p2_earliest_reliable.csv", index=False)
    sens.to_csv(tables / "p2_sensitivity.csv", index=False)

    plot_cdf(js, plots / "earliest_reliable_step_cdf.png", n_full)
    plot_coverage(frame, plots / "oracle_coverage_by_prefix.png")
    plot_by_stage(js, plots / "oracle_opportunity_by_task_stage.png", n_full)

    report_path = Path(args.report) if args.report else REPO_ROOT / "reports" / "oracle_opportunity.md"
    write_report(report_path, tier1_path, frame, js, sens, tier2, thresholds, n_full)
    print(f"[p2-analysis] tables -> {tables}")
    print(f"[p2-analysis] report -> {report_path}")


if __name__ == "__main__":
    main()
