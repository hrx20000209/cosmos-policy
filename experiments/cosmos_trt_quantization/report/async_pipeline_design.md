# Cosmos Policy 真异步推理—执行流水线设计

## 当前流程判定

现有 LIBERO evaluator 的控制流是：同步读取 observation，完整推理一个
action chunk，执行 `E=num_open_loop_steps` 个 action，停止执行，再读取
observation 并重新推理。推理期间机器人动作执行不前进，因此它是
open-loop chunk execution，不是 inference/execution overlap。单纯把
`E` 设小只会提高重规划频率，同时增加停顿占比。

本阶段不直接改动 evaluator；必须先让固定 shape TensorRT engine 稳定、
可重入并能复用 execution context，才值得引入并发状态机。

## 两 worker 架构

`inference_worker` 独占模型、CUDA stream 和可复用 engine context。输入为
带时间戳的 observation snapshot、当前执行游标和冻结前缀，输出为候选
action chunk。`action_execution_worker` 以固定控制周期提交 action，不因
推理开始而暂停，并原子地读取当前生效 chunk。

共享状态只存小对象：

- `observation_timestamp`：传感器完成同步采样的单调时钟；
- `inference_start_timestamp` / `inference_finish_timestamp`；
- `global_action_index`：从 episode 开始累计的动作编号；
- `chunk_start_index`：候选 chunk 对应的全局起点；
- `action_execution_cursor`：当前正在执行/即将提交的全局编号；
- `inference_deadline`：候选必须到达的最迟时刻；
- chunk 版本号、seed、observation ID 和 engine ID。

图像张量采用有界环形缓冲区，slot 用引用计数或 generation ID 防止
worker 读取时被覆盖。所有时钟必须使用同一 `monotonic_ns()` 域。

## Deadline 与 stale chunk

在推理开始时：

```text
chunk_start_index = action_execution_cursor + frozen_prefix_len
inference_deadline = predicted_time_of(chunk_start_index)
```

推理完成后，若 `action_execution_cursor > chunk_start_index`，候选前缀
已经过期。若剩余后缀长度低于安全阈值，或 observation age 超过阈值，
整块标记为 stale 并丢弃；绝不能把过期的第 0 个动作重新执行。

若仍有足够后缀，则按全局 index 对齐后替换：

1. 保留 execution worker 已取走的动作；
2. 冻结未来 `frozen_prefix_len` 个已承诺动作，避免控制不连续；
3. 用新 chunk 替换其余后缀；
4. 版本号加一并原子发布。

## Prefix freezing、inpainting 与 overlap

第一版采用 prefix freezing：把已执行和即将执行的短前缀作为硬约束，
只发布后缀。下一版可把冻结 action 编码回 diffusion condition，通过
action inpainting 让新 chunk 的前缀与旧轨迹严格一致；或者让新旧 chunk
在 2–4 个动作的 overlap 区间做带速度/加速度约束的平滑融合。没有训练或
明确的 inpainting mask 语义时，不应伪造 action inpainting。

## 必须记录的在线指标

- `deadline_miss_rate`：完成时刻晚于 deadline 的推理比例；
- `observation_age_ms`：action 实际提交时减 observation timestamp；
- `inference_execution_overlap_ratio`：推理区间内 execution worker
  实际执行时间 / 推理总时间；
- stale chunk rate、可裁剪后缀比例、chunk replacement 次数；
- action queue depth、冻结前缀长度；
- 推理 p50/p95/p99 与控制周期 jitter；
- chunk 边界 action/velocity/jerk 跳变；
- safety stop 和 engine error 次数。

## 推荐状态机

```text
IDLE -> SNAPSHOT_READY -> INFER_RUNNING -> CANDIDATE_READY
                                      \-> ENGINE_ERROR
CANDIDATE_READY -> PUBLISH_ALIGNED -> EXECUTING
                \-> DROP_STALE
EXECUTING -> SNAPSHOT_READY  (按重规划触发器循环)
```

engine context 由 inference worker 单线程持有；不要让两个线程同时调用同一
TensorRT context。推理和执行只通过有界队列、不可变 snapshot 与原子 chunk
句柄通信。先在离线 replay 上验证 index 对齐和 deadline 逻辑，再接入真实
LIBERO evaluator。

