# Sensitivity-Conditioned Prediction Validity：E8/E9/E10 服务器审计

## 最终决定：SENSITIVITY_HORIZON_NO_GO（协议有效性停止）

这不是对上一轮 E4 semantic state-risk 证据的经验性推翻；而是本轮提出的“语义敏感性决定预测随执行时间失效速度”无法在当前 P1 的时间语义下被合法检验，因此不运行 E8 age sweep、E9 优化或 E10 controller freeze。

## E8：为什么 age sweep 无效

当前 Cosmos-LIBERO P1 的未来视觉 slot 6/7 与 action chunk 都固定预测 `t0+16` 控制动作后的目标；P1 将这些 slot 复制进下一请求的当前视觉 slot 2/3。代码和既有采集都明确 source/target 间隔必须为 16 actions。

所以若从同一 anchor 执行 `K != 16` 个真实动作再比较 P1/F1，P1 仍在表达 `t0+16` 的世界，而 F1 已处于 `t0+K`。此差异混合了**时间错位**，不能解释为 prediction age。唯一对齐的 K=16 只有一个年龄点，无法估计 error-vs-age slope 或 `AGE×S`。

没有以 sleep 伪造年龄，没有采用错位 K 作为标签，也没有进行任何 E8 formal sample。

## 新的 confirmatory 资源

已预留此前 E4/E5/E6 从未使用的剩余 24 个任务，按 8/8/8 task-disjoint split 写入 `TASK_SPLIT_E8.json`。这只是资源冻结，尚未采样。

## E9：为何本轮未运行

先前 89-feature E4 的实测增量为 5.25 ms（2.13% F1）。E4 的确定性代码和 discovery/validation 原始表可以重建该 ridge 分数，尽管历史 artifact 没有单独保存 mean/scale/coef/intercept。本轮不运行重建、profile 或优化：严格的 E8→E9 顺序已在 E8 时间对齐 blocker 处停止。没有删 feature、重训或改变任何分数语义。

## 后续前提

若要恢复该研究问题，需要在新的、预注册的系统合同中提供真正的 multi-horizon WAM output（每个 K 都有对应对齐的预测目标），并在触碰新 held-out 前序列化 E4 score 的全部系数。那是新接口/训练或模型能力研究，不能事后修改当前 P1 来绕过本轮约束。
