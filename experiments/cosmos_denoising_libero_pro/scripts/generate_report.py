#!/usr/bin/env python3
"""Generate the required 30-section Chinese report from real summaries only."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import markdown
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
EXPERIMENT = PROJECT / "experiments/cosmos_denoising_libero_pro"
SUMMARIES = EXPERIMENT / "summaries"
REPORTS = EXPERIMENT / "reports"
OUTPUT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")


def table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns:
        frame = frame[[column for column in columns if column in frame]]
    if frame.empty:
        return "（无可用真实数据）"
    return frame.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    analysis = json.loads((SUMMARIES / "analysis_summary.json").read_text(encoding="utf-8"))
    aggregate = pd.read_csv(SUMMARIES / "aggregate_metrics.csv")
    paired = pd.read_csv(SUMMARIES / "paired_mcnemar_bootstrap.csv")
    interactions = pd.read_csv(SUMMARIES / "interaction_analysis.csv")
    robustness = pd.read_csv(SUMMARIES / "robustness_metrics.csv")
    episodes = pd.read_csv(SUMMARIES / "episodes_deduplicated.csv")
    original = aggregate[
        (aggregate["domain"] == "libero")
        & (aggregate["perturbation_category"] == "none")
        & (aggregate["suite"] != "__all__")
    ]
    pro = aggregate[
        (aggregate["domain"] == "libero_pro")
        & (aggregate["perturbation_category"] != "__all__")
        & (aggregate["suite"] == "__all__")
    ]
    original_overall_table = aggregate[
        (aggregate["domain"] == "libero")
        & (aggregate["perturbation_category"] == "__all__")
        & (aggregate["suite"] == "__all__")
    ]
    original_overall = (
        episodes[episodes["domain"] == "libero"]
        .groupby("denoising_steps")
        .agg(episodes=("success", "size"), success_rate=("success", "mean"))
        .reset_index()
    )
    pro_overall = (
        episodes[episodes["domain"] == "libero_pro"]
        .groupby(["perturbation_category", "denoising_steps"])
        .agg(episodes=("success", "size"), success_rate=("success", "mean"))
        .reset_index()
    )
    selected = analysis["selected_stage2_steps"]
    selection = json.loads(
        (SUMMARIES / "second_stage_selection.json").read_text(encoding="utf-8")
    )
    original_rates = original_overall.set_index("denoising_steps")["success_rate"]
    native_rate = float(original_rates.get(5, float("nan")))
    observed_differences = {
        step: float(original_rates.get(step, float("nan")) - native_rate)
        for step in (1, 2, 3, 4, 6)
    }
    close_candidates = [
        step for step in (1, 2, 3, 4) if original_rates.get(step, -1) >= native_rate - 0.05
    ]
    lowest_close = min(close_candidates) if close_candidates else None
    low_penalties = robustness[robustness["denoising_steps"].isin([1, 2, 3, 4])]
    most_sensitive = (
        low_penalties.sort_values("denoising_penalty_vs_5").iloc[0][
            "perturbation_category"
        ]
        if not low_penalties.empty
        else "无"
    )
    overall_metrics = aggregate[
        (aggregate["perturbation_category"] == "__all__")
        & (aggregate["suite"] == "__all__")
    ]
    disagreement_total = int(
        paired["step_success_reference_failure"].sum()
        + paired["step_failure_reference_success"].sum()
    )
    stage2_path = SUMMARIES / "stage2_analysis_summary.json"
    stage2_text = (
        stage2_path.read_text(encoding="utf-8").strip()
        if stage2_path.is_file()
        else "第二轮尚未生成完成状态文件。"
    )
    stage2_original_path = SUMMARIES / "stage2_original_three_seed.csv"
    stage2_original = (
        pd.read_csv(stage2_original_path)
        if stage2_original_path.is_file()
        else pd.DataFrame()
    )
    stage2_hard_path = SUMMARIES / "stage2_hard_subset_metrics.csv"
    stage2_hard = (
        pd.read_csv(stage2_hard_path)
        if stage2_hard_path.is_file()
        else pd.DataFrame()
    )
    reduced_plan_path = SUMMARIES / "stage2_reduced_plan.json"
    reduced_plan = (
        json.loads(reduced_plan_path.read_text(encoding="utf-8"))
        if reduced_plan_path.is_file()
        else None
    )
    reduced_plan_text = (
        json.dumps(reduced_plan, indent=2, ensure_ascii=False)
        if reduced_plan is not None
        else "尚未建立缩减方案。"
    )
    direct_answers = (
        f"第一轮原始 LIBERO 相对 5-step 的观测成功率差（step→差值）为 "
        f"`{observed_differences}`；最低且距 native 不超过 5 个百分点的配置为 "
        f"`{lowest_close}`。6-step 相对 5-step 的观测差为 "
        f"{observed_differences[6]:+.4f}。PRO 中低步数相对 5-step 最负的类别为 "
        f"`{most_sensitive}`。配对表共记录 {disagreement_total} 个方向性 discordance；"
        "显著性与不确定性必须结合 Wilson、exact McNemar 和 base-task cluster bootstrap。"
    )
    complete = analysis["status"] == "complete"
    status_text = (
        "第一轮 1,410 个正式 episode 已全部完成。"
        if complete
        else f"当前是中间报告：已完成 {analysis['episodes']}/1,410 个正式 episode，未完成部分不作结论。"
    )
    sections = [
        ("摘要", f"{status_text} 本报告所有数值均由 JSONL/CSV 生成；不把 policy request 当作成功率独立样本。\n\n{direct_answers}"),
        ("研究问题", "评估 1–4 step 相对 native 5-step 的成功率与延迟，6-step 的增益，以及 PRO 单扰动是否放大低步数退化。"),
        ("现有 smoke 结果", "历史 joint-parallel 2 tasks × 3 seeds 六档均 6/6，只证明路径可运行；本阶段 correctness smoke 单独归档，不进入正式统计。"),
        ("为什么移除 8-step", "正式研究预算固定为 1–6；8-step 超出问题定义，旧 smoke 的 8-step 不进入 manifest、图或结论。"),
        ("为什么将 6-step 设为上限", "6-step 是高计算 upper-control，用于检验超过 native 5-step 是否有可观察收益，不预设其更准确。"),
        ("为什么采用 action-only usage", "保持 joint latent generation，关闭 future RGB VAE decode；所有正式请求的 future decode 必须为 0 ms。"),
        ("Cosmos 模型与 checkpoint", "Cosmos-Policy-LIBERO-Predict2-2B，BF16 eager；checkpoint SHA256 `8818528d…33e2`，参数量 1,956,413,440。"),
        ("原始 LIBERO 设置", "四个官方 suite 各 10 task，seed 195、init index 0、horizon 16；40×6=240 episode。"),
        ("LIBERO-PRO 设置", "官方 commit `eafdb809…`，隔离 package shim；五类单扰动。官方环境生成器对 5 个 living-room task 无实际变化，标为 unsupported 并排除正式分母。"),
        ("五类单一扰动", "object 40、position/swap 40、language 40、task 40、environment 有效 35；禁止多扰动组合。"),
        ("T5 paraphrase embedding 处理", "首轮 117 条精确指令 cache SHA256 `e8b46e19…05088`。第二轮扩展为 197 条精确指令（新增 80），cache SHA256 `d6cfd5c9…c71c1`；manifest 保存逐指令 tensor hash，无 canonical fallback。"),
        ("配对实验设计", "相同 task/variant/seed/init/BDDL 的六档配对；执行顺序按 task/category 循环旋转，避免总是 1→6。"),
        ("Forward-count correctness", "每请求断言 selected steps、完整 denoiser forward 数及 CUDA-event 数三者相等；任一失败即 invalid。future decode count 必须为 0。"),
        (
            "原始 LIBERO 成功率",
            table(
                original_overall_table,
                [
                    "denoising_steps",
                    "episodes",
                    "successes",
                    "success_rate",
                    "success_wilson_low",
                    "success_wilson_high",
                ],
            )
            + "\n\n![原始 LIBERO overall success](../plots/01_original_steps_vs_overall_success.png)"
            + "\n\n第二轮 selected steps 的三种子（每档 120 条）结果：\n\n"
            + table(stage2_original),
        ),
        (
            "LIBERO-PRO 成功率",
            table(pro_overall)
            + "\n\n![PRO success by perturbation](../plots/03_pro_steps_vs_success_by_perturbation.png)",
        ),
        ("各 suite 结果", table(original, ["suite", "denoising_steps", "episodes", "success_rate", "success_wilson_low", "success_wilson_high"])),
        (
            "各 perturbation 结果",
            table(
                pro,
                [
                    "perturbation_category",
                    "denoising_steps",
                    "episodes",
                    "success_rate",
                    "success_wilson_low",
                    "success_wilson_high",
                ],
            )
            + "\n\nMatched-base-task robustness gap 与相对 5-step 差：\n\n"
            + table(robustness),
        ),
        (
            "Denoising × perturbation interaction",
            table(interactions)
            + "\n\n![perturbation × steps heatmap](../plots/04_pro_perturbation_steps_success_heatmap.png)",
        ),
        (
            "Policy/DiT latency",
            table(overall_metrics, ["domain", "denoising_steps", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "dit_mean_ms", "dit_p95_ms"])
            + "\n\n![policy latency](../plots/07_policy_latency_vs_steps.png)\n\n"
            + "![DiT latency](../plots/08_dit_latency_vs_steps.png)",
        ),
        ("Episode time", table(aggregate, ["domain", "perturbation_category", "denoising_steps", "episode_time_mean_s", "episode_steps_mean"])),
        ("Success–latency Pareto", f"数据驱动第二轮候选为 `{selected}`。Pareto 图见 `../plots/10_success_latency_pareto.png`；逐 step 理由为 `{selection['steps']}`。"),
        ("动作平滑性和 jerk", "逐 episode 保存 float32 trajectory、smoothness、jerk、chunk-boundary discontinuity、gripper transitions/chatter、saturation 与 EEF no-progress duration。平滑性只作行为描述，不直接解释为任务质量。"),
        ("失败案例", f"paired-disagreement 方向性计数合计 {disagreement_total}。当前阶段性报告尚未完成 reduced 失败视频复放；计划从 833 条缩减为 32 条代表性/直接分歧端点。自动标签默认 `unclassified`；没有人工/事件证据时不依据 `success=False` 猜测原因。"),
        ("环境无效案例", "生成阶段 200/200 BDDL 最终可构造。官方 environment 的 5 个 no-op 是 unsupported，不计 policy failure。上游修复 ledger 已完整保存。"),
        ("统计局限", "第一轮每 task/variant 仅 1 episode；Wilson CI、exact McNemar 和 task-cluster bootstrap 仍不能替代更多 init-state 重复。全成功时普通 bootstrap 会退化。第二轮 hard 表目前是停止大 sweep 时的部分数据，不应用于最终类别间显著性结论。\n\n部分 hard 数据：\n\n" + table(stage2_hard)),
        ("第二轮候选选择", f"当前选择 `{selected}`；2/4 仅在 Pareto、非单调或特定扰动独特表现时保留。理由索引见 `../summaries/second_stage_selection.json`。\n\n第二轮完成状态：\n\n```json\n{stage2_text}\n```\n\n用户确认缩减后的方案：\n\n```json\n{reduced_plan_text}\n```"),
        ("对异步推理的启示", "只有在成功率/鲁棒性可接受且位于 success–latency frontier 的低 step 配置，才值得进入后续 steps × execution-horizon 联合实验；本阶段没有运行真异步。"),
        ("后续工作", "40-task 三 seed 复测已经完成。后续按 reduced manifest 补齐 21 个 hard 变体的 3 seeds × 3 init（还需以 validator 为准），运行官方所有 language paraphrase 480 条和 32 条代表性失败视频；继续保持 base-task cluster 统计。"),
        ("完整复现命令", "见 `../RUNBOOK.md`。主要顺序：prepare assets → build manifests → precompute T5 → validate → correctness → original → PRO categories → analyze → report。"),
        ("Manifest 与原始结果索引", f"Manifest：`../manifests/`；episode/request/action 原始数据：`{OUTPUT}`；16 组 PNG/PDF：`../plots/`。"),
    ]
    body = ["# Cosmos Policy denoising steps：LIBERO 与 LIBERO-PRO 评测\n"]
    for index, (heading, content) in enumerate(sections, start=1):
        body.append(f"\n## {index}. {heading}\n\n{content}\n")
    body.append(
        "\n> 未建立 expert-aligned gripper transition reference，因此不报告 gripper transition error。\n"
    )
    body.append("\n### 配对检验附表\n\n" + table(paired) + "\n")
    REPORTS.mkdir(parents=True, exist_ok=True)
    markdown_path = REPORTS / "report_zh.md"
    markdown_text = "".join(body)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    css = """
    body { font-family: sans-serif; line-height: 1.55; max-width: 1100px; margin: auto; padding: 28px; }
    table { border-collapse: collapse; font-size: 11px; width: 100%; }
    th, td { border: 1px solid #aaa; padding: 4px; }
    h1, h2 { page-break-after: avoid; }
    """
    html_body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    html_path = REPORTS / "report_zh.html"
    html_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{html_body}</body></html>",
        encoding="utf-8",
    )
    pdf_path = REPORTS / "report_zh.pdf"
    subprocess.run(
        ["/home/rxhuang/.local/bin/weasyprint", str(html_path), str(pdf_path)],
        check=True,
    )
    print(json.dumps({"markdown": str(markdown_path), "html": str(html_path), "pdf": str(pdf_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
