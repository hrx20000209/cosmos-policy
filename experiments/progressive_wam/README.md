# Progressive WAM Denoising and Action Commitment

早期可行性实验。目标不是提出完整算法，而是验证：**WAM 完整 denoising trajectory 的早期
checkpoint 是否已经包含可执行的近期 action**，以及提前执行是否存在真实机会。

先读 [`reports/progressive_denoising_audit.md`](../../reports/progressive_denoising_audit.md)，
它定义了本项目强制的三个概念口径（`standalone_one_step` / `first_checkpoint_of_full_schedule` /
`predicted_clean_action`），并给出两个模型的 parameterization 与最小修改点。

## 状态

| Step | 状态 | 规模 | 产物 |
|---|---|---|---|
| 1 Audit schedulers and parameterization | 完成 | 55 项单测 | `reports/progressive_denoising_audit.md` |
| 2 Baseline smoke | 完成 | 4 suites × 2 tasks × 3 eps = **24/24 成功** | `wam_progressive_outputs/baseline_smoke` |
| 3 Checkpoint trajectory dump（P1） | 完成 | 24 eps / **241 requests** / 5 checkpoint | `reports/trajectory_audit.md` |
| 4 Oracle branched rollout（P2） | 完成 | tier1 **3984** 条分支 + tier2 **129** 条成功率 | `reports/oracle_opportunity.md` |
| 5–10 | 未开始（按用户约束，先到 Step 4） | — | — |

**决策见 [`reports/feasibility_decision.md`](../../reports/feasibility_decision.md)：继续，但换机制。**
oracle median j\*=1、tier2 成功率无退化，但 checkpoint 1..5 的可靠率与 identical-action 对照
不可区分；prefix-wise 递进提交被逐 token 的 U 形误差否定；「读取中间 checkpoint 的 x0_pred」
被短 schedule 严格支配（相同 NFE 下误差 5×）。

LingBot-VA 侧：**没有 LIBERO post-train checkpoint**，全部实验为 `implemented_not_run`；
本轮 lingbot-va 仓库未做任何修改。

## 环境

```bash
cd /home/rxhuang/Projects/cosmos-policy
source experiments/progressive_wam/env.sh <gpu>     # 4 / 5 / 6 / 7
```

LIBERO 用 `/data/rxhuang` 下的用户态 OSMesa 做 CPU 渲染（本机无 `/dev/dri`），
CUDA 推理仍在选定 GPU。**不要把 OSMesa 的 CPU 渲染 wall time 外推到 Jetson。**

本机还装了 editable 的 LIBERO-plus；所有 runner 都用
`configure_repository_paths` 把 `/home/rxhuang/Projects/LIBERO` 顶到 `sys.path` 最前，
否则 `import libero` 会静默解析到 LIBERO-plus。

## 单元测试

```bash
.venv/bin/python -m pytest -q tests/test_progressive_checkpoint.py tests/test_progressive_oracle.py
```

覆盖：checkpoint 数 == denoiser forward 数（1/2/3/4/5/8 全分支）、hook 不改变输出、
官方 LIBERO σ 网格、`standalone_1 == 任意 schedule 的 j=1`、
hook 提取与官方 `extract_action_chunk_from_latent_sequence` 一致、
EDM `x0 = x_t - σ·eps` 恒等、LingBot flow-matching `x0 = x_σ - σ·v` 与
`scheduler.step(to_final=True)` 等价、oracle 标注与 j*(h)、阈值敏感性重标注。

## Step 2：baseline smoke

```bash
source experiments/progressive_wam/env.sh 4
for suite in libero_spatial libero_object libero_goal libero_10; do
  .venv/bin/python experiments/run_wam_libero_experiment.py \
    --model cosmos --task_suite "$suite" \
    --config experiments/configs/cosmos_baseline.yaml \
    --output_dir /data/rxhuang/wam_progressive_outputs/baseline_smoke \
    --set "run_id=baseline-smoke-${suite}" --set "evaluation.task_suite=${suite}" \
    --set 'evaluation.task_ids=[0,1]' --set evaluation.episodes_per_task=3 \
    --set 'evaluation.seeds=[195,196,197]' --set evaluation.record_video=false
done
```

## Step 3：P1 checkpoint trajectory dump

Episode 始终由 **官方 5-step** action 驱动，所以被审计的状态分布就是真实 baseline 分布。
每个 request 记录 5 个 checkpoint 的 predicted-clean action、value、future latent、
CUDA-event 时间和 **MuJoCo sim state**（P2 直接从这里分叉，不需要重跑 policy）。

```bash
source experiments/progressive_wam/env.sh 6
.venv/bin/python experiments/progressive_wam/run_p1_trajectory_dump.py \
  --run-id p1-full-A --task-suites libero_spatial libero_object \
  --task-ids 0 1 --seeds 195 196 197 --standalone-stride 4
```

`--standalone-stride N`：每 N 个 request 额外跑一组 `standalone_k_step`（k=1..5）探针，
用同一 observation 和同一 seed，因此初始噪声完全相同。

分析：

```bash
.venv/bin/python experiments/progressive_wam/analyze_p1.py \
  --trajectory-dir /data/rxhuang/wam_progressive_outputs/trajectories/p1-full-A
```

## Step 4：P2 simulator branched rollout oracle

```bash
# tier1：状态偏离 + 安全检查，不需要模型
source experiments/progressive_wam/env.sh 5
.venv/bin/python experiments/progressive_wam/run_p2_oracle.py \
  --trajectory-dir /data/rxhuang/wam_progressive_outputs/trajectories/p1-full-A \
  --tier tier1 --prefix-lengths 1 2 4 8

# tier2：分叉后继续跑真实 policy 到 episode 结束，比较成功率（贵，子采样）
.venv/bin/python experiments/progressive_wam/run_p2_oracle.py \
  --trajectory-dir ... --tier tier2 --tier2-checkpoints 1 2 3 --tier2-prefix-lengths 8
```

分析：

```bash
.venv/bin/python experiments/progressive_wam/analyze_p2.py \
  --oracle-dir /data/rxhuang/wam_progressive_outputs/oracle/<run> \
  --trajectory-dir /data/rxhuang/wam_progressive_outputs/trajectories/<run>
```

## 输出目录

```text
/data/rxhuang/wam_progressive_outputs/
  baseline_smoke/runs/<run_id>/          # Step 2
  trajectories/<run_id>/                 # P1: metadata.json checkpoints.pt actions.npy timing.jsonl
  oracle/<run_id>/                       # P2: tier1_branches.jsonl tier2_success.jsonl metadata.json
  audits/<run_id>/                       # P1 汇总表
  plots/                                 # 全部图
```

## 口径红线

1. 永远不要把 noisy solver state 当 action 执行；只用 predicted-clean。
2. Cosmos 的 σ 终点是 **4 不是 0**，"final action" 一律定义为官方 5-step 输出。
3. `standalone_k_step`（配置）与 `first_checkpoint_of_full_schedule`（截取）是两件事，
   即使在 k=1 上数值相同也必须分别报告。
4. P2 是 **feasibility upper bound**，用了 simulator 特权状态，在线系统拿不到。
5. tier1 不能回答成功率；只有 tier2 能，且样本量小。
6. 代码完成但实验未运行时标 `implemented_not_run`，不得伪造。
