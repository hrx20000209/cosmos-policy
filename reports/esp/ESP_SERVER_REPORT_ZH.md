# ESP Server 机制验证报告（阶段性 NO-GO）

## 结论

本轮将 Cosmos 当作通用 WAM，使用原始 pre-finetune checkpoint、1 denoise step，不读取 value，不做 finetune、action scaling、x0/hidden patch 或 scheduler。结论为 **ESP_NO_GO**：不能将当前的 early-layer finite-difference probe 发展为在线相机选择/刷新机制。

这不是因为没有任何视觉敏感性：E2 因果干预确实改变动作，且 12 个任务上的平均 task 内 top-1 camera switching 为 27.1%。否决来自两个独立因素：相机在当前联合 VAE 结构中不能独立刷新；并且 E3 即使 k=2 的双相机总 probe 成本也为一次 F1 的 29.5%，远高于预注册的 5% 上限。

## 合法性与语义检查

- 审计确认 action token 为 temporal slot 4 的 196×2048 hidden；模型共 28 blocks。
- E2 的 imagined/stale/shuffle 干预均只改变目标 camera slot；48-state pilot 的 scope checks 全部通过。
- E3 仅用前一决策已存在的 real condition；未读取目标当前 fresh condition。
- Early exit 在 k={2,4,6,8,12} 与完整前向同一 block 的 hidden RMS 均为 0；F1 重复输出为 0。因此成本不是完整前向伪装成 early exit。

## E1：架构与成本

50 次 clean-GPU CUDA-event 配对测量：F1 中位 246.5 ms，P1 中位 74.0 ms，二者端到端刷新差 172.6 ms（F1 的 70.0%）。该差额不被错误标称为独立 VAE kernel 时间。关键架构事实是 wrist/primary 通过时序拼接进入一次 joint VAE encode，没有能避免另一相机 VAE 工作的单相机路径；故 E1 为 `E1_SELECTIVE_SENSING_ARCH_NO_GO`。

## E2：因果相机重要性（pilot）

48 个 paired states / 12 个 task，严格 4/4/4 discovery/validation/heldout task split。primary 与 wrist 的 imagined intervention 平均动作差分别为 0.448 和 0.469（full-chunk mean-step L2）。这证明相机条件对动作具有因果影响，但 48 states 仅用于机制 smoke/pilot，不能替代协议的 ≥300-state E2 正式 gate。

## E3：early hidden 是否预测 E2 target

下表为 task-balanced Spearman（按 task 平均，而非将 state-camera 对假作 IID）：

| k | discovery | validation | heldout | total ESP/F1 |
|---:|---:|---:|---:|---:|
| 2 | 0.095 | -0.036 | 0.119 | 29.5% |
| 4 | 0.155 | -0.119 | 0.143 | 32.8% |
| 6 | 0.054 | -0.089 | 0.119 | 40.4% |
| 8 | 0.131 | 0.006 | 0.167 | 48.3% |
| 12 | 0.048 | -0.077 | 0.190 | 58.4% |

最好 discovery 深度 k=4 也仅为 0.155，validation 转为 −0.119，heldout 为 0.143；不存在跨 split 一致性。即使忽略相关性，k=2 的 29.5% 总成本也单独否决 runtime GO。因此不扩展到 300-state 来事后寻找有利结果。

## 下一步

保留 ESP 的负结果，停止 ESP-guided camera-refresh scheduler。若后续要继续，应提出新的、独立的可计算机制（不能重开 AGE-only、action amplification、hidden/x0 patch、固定 cache/skip 或当前 ESP 变体），并先做新的架构与成本可行性审计。

## 产物

- `reports/esp/E1_RESULT.json`：50-repeat clean-GPU E1。
- `reports/esp/E2_PILOT_RESULT.json`、`E3_PILOT_RESULT.json`：48-state pilot 汇总。
- `reports/esp/ESP_FINAL_DECISION.json`：可机读最终决策。
- `artifacts/esp/*_pilot.parquet`：原始可分析行；`reports/esp/shards/pilot_offline/`：原始 shard。
