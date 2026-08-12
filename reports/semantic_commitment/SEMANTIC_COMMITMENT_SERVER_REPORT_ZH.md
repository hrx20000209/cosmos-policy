# Semantic Action Commitment：E11-A/B/C 服务器结果

## 最终决定：SEMANTIC_COMMITMENT_NO_GO

本轮没有运行 E11-B commitment oracle、E11-C LONG/SHORT counterfactual 或 E12 controller。停止原因是 E11-A 的因果 route-transfer gate 失败，而非把时间错位当成负结果。

## 已确认的固定时间合同

Cosmos-LIBERO 的 action chunk 和 future visual latent 都固定为 H=16 actions 后的目标。E11-B 所提出的 remaining-plan 比较本身是合法的：真实执行 K 后，`A_orig[K]` 与该同一物理状态 fresh policy 的 `A_fresh[0]` 都是“当前之后”的 7D relative EEF delta + gripper command，不涉及把 `z_hat(t+16)` 当作 `t+K`。

## E11-A：分数冻结成功，但 F1 route transfer 失败

原 E4 `INTERNAL_PLUS_ACTION` 被从原 discovery+validation 352 行确定性重建：89 个 feature、mean、scale、ridge 系数、intercept 与 checksum 全部固化；历史 prediction 最大绝对重建误差仅 **3.15e-14**。

但 action commitment 的 anchor 是刚产生 action chunk 的 F1 route。用全新的 8 个 discovery task、32 个 state 的预检：`S_F1` 与合法的 16-action retrospective target 的 task-balanced Spearman 仅 **0.175**，低于预注册 **0.50** 门槛；虽然 `S_F1` 与 `S_P1` 的 rank Spearman 为 **0.955**，也不能证明 F1 评分能预测 commitment risk。因此标为 **P1_ONLY_SENSITIVITY**，不能未经验证作为 F1 anchor 的 runtime decision signal。

## Native runtime 预检

原 scorer 84.44 ms，native fused reducer 85.05 ms（20-repeat preflight）；没有成本下降。action 完全一致。native 与 original score 最大差 **1.08e-05**，略超过预检 1e-5 tolerance，来自 GPU/CPU 浮点求和顺序；没有用删 feature、近似、重训来规避。因 route-transfer 已失败，未进行 100-repeat 正式 profiling。

## 科学边界

这并不否定 P1-route 内部 semantic risk 的上一轮证据；它否定的是“将那个 P1-only 分数直接挪到 F1 action-commitment anchor”这一必要前提。E11 validation 与 heldout 仍完全未触碰，24-task split 保留。未来若研究继续，必须先在 discovery 上开发并冻结一个独立的 F1-route estimator，然后才可以使用这批 clean validation/heldout 任务。
