# PV0 / Cosmos WAM 最终研究报告

## CURRENT DECISION: PIVOT：机制 GO；“PV0 可普遍替代 Fresh”主张 NO-GO

PV0 已经是一个有证据支持的系统机制雏形：它用当前物理视觉校正上一轮生成 latent 的视觉条件，并在固定 one-denoise Cosmos 中修复纯预测复用的漂移。它**还不是**一个可宣称全面替代 always-fresh inference 的系统：完整 40-task 基准里有两个 Fresh-only success。

## 已完成的冻结协议

| 阶段 | 覆盖 | 结论 |
| --- | ---: | --- |
| S1 state fidelity | 3,801 states / 40 tasks | `GO` |
| S2 predictive-prior decomposition | 3,801 states | `GO` |
| S3 execution-prefix alignment | K=4/8/12/16 | `GO` |
| S4 condition compile control | 40 fixed anchors | `GO` |
| S5 action-outcome mismatch | 1 preregistered scenario | preliminary only |
| Phase B closed loop | 40 tasks × 5 init × 3 routes = 600 | complete |

所有阶段都使用原始 pre-finetune Cosmos checkpoint（SHA-256 已锁定）、`denoise=1`，不读取 Cosmos value；没有 SO101 finetune、scheduler、threshold、hidden activation patch、fresh-prefix oracle 或 runtime privileged state。

## 三个运行时模块（均不训练）

1. **Speculative joint-latent reuse (P1)**：保存上一轮 generated joint latent，作为下一 request 的起点。
2. **Native Persistent Visual Condition (PV0)**：仅重编码最新物理视觉的因果 13-frame VAE prefix，并在唯一 denoiser forward 前替换 current visual condition slots。
3. **Fixed execution-prefix contract**：动作仍按固定 16-step prefix 执行；K=4/8/12/16 只是离线对齐审计，不是 adaptive scheduler。

训练注册表：**无训练模块**。主路径保持 frozen original policy；没有 residual head、value head 或 finetune checkpoint。

## 机制证据：状态层

- Held-out 1,066 states：PV0→F1 mean-step L2 的中位数 `0.0020`、p95 `0.0052`；P1→F1 对应中位数 `0.3440`、p95 `1.9149`。
- Held-out states 中 PV0 在 ε=0.05 内的 fidelity 为 `99.9%`，PV0 优于 P1 的比例为 `100.0%`；所有 held-out tasks 的首个 gripper sign 均匹配。
- S4 held-out 12 anchors：正确 fresh visual condition 优于 shuffled condition 的比例 `100.0%`；正确 PV0→F1 L2 mean `0.0037`，shuffled 为 `1.1811`。这表明效果来自正确物理视觉条件，而非 latent-reuse 假象。

## 闭环结果：应主张什么、不能主张什么

| Route | Success | Success rate |
| --- | ---: | ---: |
| Fresh (F1) | 19/200 | 9.5% |
| Predicted reuse (P1) | 11/200 | 5.5% |
| PV0 | 17/200 | 8.5% |

- PV0 vs P1：6 wins / 0 losses，Δ=3.0 pp，per-scenario exact McNemar p=0.0312；但 task-hierarchical bootstrap 95% CI 为 [0.0, 0.08]，下界为 0，因此不能过度表述 task-general significance。
- Held-out 12 tasks × 5 init：Fresh=12/60，P1=6/60，PV0=12/60；PV0 vs P1 为 6/0，p=0.0312；PV0 与 Fresh 逐对完全相同。
- PV0 vs Fresh（完整 200 scenarios）：0 wins / 2 losses，Δ=-1.0 pp。因此 PV0 的正确定位是 **recovery of predictive reuse**, 而不是 **universal Fresh replacement**。

## 成本证据（clean GPU，单一已存物理状态，25 warm repeats，轮换执行顺序）

- F1 median `199.3 ms`，P1 `70.9 ms`，PV0 `131.4 ms`；PV0 比 F1 低 `34.1%`。
- model-generate median：F1 `183.3 ms`，PV0 `115.3 ms`，PV0 降低 `37.1%`。
- 同一状态的 PV0 action recovery 为 `99.8%`（相对 P1→F1 discrepancy）。这是 4090 的同 worker route-cost 证据，不可外推为 end-to-end closed-loop latency，也不是 Thor latency claim。

## 负结果与 failure diagnosis

- 两个 base-seed Fresh success → PV0 failure 都是 LIBERO-10 多物体入篮任务；两条 P1 也同样失败并达 max-steps。
- 离线回放已 SHA 验证 saved actions：首例 Fresh 251 steps 成功，而 P1/PV0 都执行 520 steps 未完成；第二例 Fresh 331 steps 成功，而 P1/PV0 都执行 520 steps 未完成。
- 两例的首个 16-action prefix 与 Fresh 几乎一致，偏离在后续闭环累积后才出现。第二例 PV0 有显著 gripper chatter / chunk-boundary discontinuity；这是长期闭环残余失配的候选诊断，不构成单一因果证明。
- 代表性回放：`failure_replays/native_persistent_d7868de6b12fd8c81036d744b2e656af60afb1ff9ba27644c208c0ede8be54e3_seedoff0.mp4`，以及同一目录下 Fresh/P1 对照视频。

## S5 与种子稳健性

- S5 的一个预注册 zero-motion/preserve-gripper interruption 场景中三条路线均成功。样本量为 1，不能写成 disturbance-recovery rate。
- 额外 3 inference seeds 仅针对 Fresh-success / disagreement 条件子集（57 matched pairs），PV0 对 Fresh 为 3 wins / 1 loss；该选择性子集不能当成总体 success rate。

## 论文定位与下一步

可发展的论文中心问题是：**在 unified WAM 中，如何将到达的新鲜物理视觉作为 persistent condition 写入正在复用的世界 latent，从而恢复 prediction-only reuse 的控制正确性，并保留计算收益？**

当前可用的论文结论：PV0 将 P1 的 latent drift 拉回接近 Fresh action，并在 held-out closed loop 中保留 Fresh 成功、修复所有 P1-only losses。

当前不能写的结论：PV0 普遍优于 Fresh、已完成泛化硬件验证、或 action-outcome recovery 已被大样本证明。下一轮应是冻结机制下的更广 success/regression audit 与目标 Thor 测量，而不是添加 scheduler、value 或新 design family。

## 可复核资产

- `CLOSED_LOOP_PAIRED_ANALYSIS.json`：600 条路线、200 个 paired scenarios 的完整审计与 bootstrap。
- `PREFIX_VAE_ARCHITECTURE_AUDIT.md`：13-frame prefix 的源代码级解释。
- `failure_replays/`：6 个 exact saved-action MP4 与 replay summary。
- `thor_export/`：目标硬件测量合同，不含任何 Thor 性能声明。
- 当前 artifact audit：S1=3801/3801，S4=40/40，原始 checkpoint SHA 已通过。
