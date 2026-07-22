# 任务：在 Jetson Thor 上部署 Cosmos Policy × Three Cubes（step 5000）真机推理

> 把本文件放到仓库根目录，在 Thor 设备上以同一份 `cosmos-policy` 仓库为工作目录启动 Claude Code，让它阅读本文件后执行。背景信息来自训练机（A100）上的 `outputs/cosmos_three_cubes_full_finetune/REPORT_ZH.md` 与 `REPORT_ZH_step100_to_5000.md`，遇到冲突以 Thor 上的实际环境和代码为准。

## 0. 背景（不要重新训练，只做推理部署）

- 已完成：Cosmos Policy（Predict2 2B Video2World 480p/16fps base）在 `hrx2000/Three_Cubes_1`（100 episodes、SO101 机械臂、6 DOF：shoulder_pan/lift、elbow_flex、wrist_flex/roll、gripper）上全参数微调，从 step 0 训练到 **step 5000**，DCP checkpoint 位于训练机的：
  `outputs/cosmos_three_cubes_full_finetune/runs/cosmos_policy/so101_lerobot/cosmos_predict2_2b_480p_so101_lerobot/checkpoints/iter_000005000`（约 11GB，含 model/optim/scheduler/trainer 四部分；推理只需要 `model/` 子目录，但**这条"只传 model 子目录"的路径本任务未验证过**，如果要用请先在 Thor 上自行确认 `load_model_from_checkpoint` 能否只从 `model/` 加载，不确定就传完整目录）。
- 训练时 action chunk 长度 `chunk_size=30`（不是仓库里部署脚本示例代码默认写的 50，见第 3 节，这是最容易踩的坑）。
- 诊断结论（训练集诊断，非真机闭环）：
  - 默认（无重叠拼接）评估下，action chunk 边界跳变可达真实轨迹的 30+ 倍，5/6 关节在单个 chunk 内趋于近似常数输出。
  - 用更高频重规划 + 时间集成（temporal ensembling，指数衰减加权融合重叠 chunk）评估后，跳变倍数降到 ~2.5-2.7 倍，说明大部分"跳变"是评估/执行方式的产物而非模型完全学坏，但**这只是离线模拟，从未在真实机器人上验证过**。
  - **这个 checkpoint 从未做过真机闭环测试**，请把它当作"第一次上机、行为完全未知"来对待，不要因为离线指标好看就跳过安全检查。

## 1. 目标

用仓库里已有的 SO101 async policy server（`cosmos_policy/experiments/robot/so101_async_deploy.py`，基于 LeRobot 的 async-inference gRPC 协议，机器人本体侧沿用 LeRobot 标准的 `robot_client`，只是把 policy server 换成这个 Cosmos checkpoint）在 Thor 上跑起来，先做不执行动作的冒烟测试，再在人工监督、安全裁剪开启的前提下做谨慎的小范围真机测试。

## 2. 前置检查（不要假设，逐项确认）

1. **硬件/依赖兼容性**：Thor 是 ARM 架构的 Jetson 设备，训练机是 x86 + A100。确认 Thor 上能装：PyTorch（含 CUDA 支持）、Transformer Engine 2.2.0、FlashAttention 2.7.3、LeRobot 0.4.4、draccus、grpc、json-numpy 等。这些在 Jetson 上不一定有现成 wheel，可能需要源码编译或版本降级；如果某个依赖在 Thor 上确实装不上或不兼容，如实报告卡在哪一步、报什么错，不要为了"能跑起来"去改训练/推理逻辑绕过。
2. **文件传输**：以下文件需要从训练机传到 Thor（scp/rsync 均可，具体方式和网络连通性由你决定，不要假设两台机器共享文件系统）：
   - `iter_000005000` checkpoint 目录（完整传输约 11GB，见第 0 节关于只传 `model/` 子目录的未验证说明）
   - `outputs/cosmos_three_cubes_full_finetune/pretrained/cosmos_official/tokenizer/tokenizer.pth`（VAE/tokenizer，约 485MB，推理时编码观测图像需要）
   - `outputs/cosmos_three_cubes_full_finetune/data/hrx2000/Three_Cubes_1/so101_dataset_statistics.json`（action/state 归一化统计，几 KB）
   - `outputs/cosmos_three_cubes_full_finetune/data/hrx2000/Three_Cubes_1/so101_t5_embeddings.pkl`（预计算 T5 text embedding，约 1MB）
   传完后在 Thor 上记录实际路径，第 3 节要用。
3. **代码同步**：确认 Thor 上的 `cosmos-policy` 仓库代码是最新的（含本次新增的 `cosmos_policy/scripts/eval_so101_full_episode_action_curve_ensembled.py`、`cosmos_policy/datasets/lerobot_full_adapter.py` 等），如果是独立 clone 而非同一份工作区，用 git 同步。

## 3. 已知需要修改的代码（否则会报错或行为错误）

`cosmos_policy/experiments/robot/so101_async_deploy.py` 里的 `SO101CosmosAsyncServerConfig`（约 127-165 行）和内部推理配置（约 200-210 行）是照着另一个 chunk_size=50 的 checkpoint 写的模板，用我们这个 checkpoint 之前必须核对/修改：

1. **`chunk_size` 不匹配**：`__init__` 里构造 `cosmos_cfg` 的 `SimpleNamespace` 硬编码了 `chunk_size=50, num_open_loop_steps=50`（约 204-205 行），但我们训练时 `chunk_size=30`。必须改成 `30`；同时 `_predict_action_chunk` 里的 shape 校验 `if action_array.shape != (50, 6)`（约 415 行附近）也要同步改成 `(30, 6)`，否则模型返回 `(30, 6)` 的 chunk 时会直接 `RuntimeError` 崩溃。
2. **默认路径**：`ckpt_path`、`dataset_stats_path`、`t5_text_embeddings_path` 三个字段默认值指向训练机上原作者的路径（`/data/rxhuang/...`），要改成第 2 节里实际传到 Thor 的路径。`cosmos_config` 默认值 `"cosmos_predict2_2b_480p_so101_lerobot"` 不用改，正好是我们训练用的 experiment 名。
3. **相机 key 映射**：`primary_camera_key="front"`、`left_wrist_camera_key="right"`、`right_wrist_camera_key="wrist"` 对应训练数据里的 `observation.images.{front,right,wrist}`。请核对 Thor 上 LeRobot 机器人客户端实际配置的相机名称是否也是这三个，不一致会在 `_build_cosmos_observation` 里直接 `KeyError`。
4. **安全裁剪参数**（`max_delta_from_observation=8.0`、`max_gripper_delta_from_observation=8.0`、`max_step_delta=4.0`、`max_gripper_step_delta=5.0`，单位：角度/gripper 用度数和 0-100 range）：这些是仓库作者留下的保守默认值，**不要因为想要更大动作幅度而放宽或设为 0 禁用**，尤其是鉴于第 0 节提到的 chunk 跳变问题，这是重要的安全边界，只应该更保守不应该放松。

## 4. 分阶段验证流程（不要跳步）

1. **冒烟测试（`dry_run_zero_actions=true`）**：先用这个模式启动 server（此模式下 `_predict_action_chunk` 会把动作强制设为当前 proprio，即不产生任何实际位移），验证：gRPC 连接正常、相机 key 匹配、`_validate_and_log_action_schema` 通过（joint_order/action_dim/归一化范围与训练时一致）、模型能成功加载不报错。这一步不应该有任何机器人运动。
2. **关闭 dry-run，人工近旁监督，小范围低速测试**：确认第 1 步全部通过后，才关闭 `dry_run_zero_actions`。测试时必须有人在机械臂旁边，随时准备物理急停/断电；建议先固定摄像头对准和训练数据里相近的场景摆放（三个方块 + 类似背景），因为这个 checkpoint 从未在训练分布之外的场景下测过，行为未知。
3. **如果观察到明显的抖动/跳变/危险动作**：立即停止，不要尝试通过调大 `actions_per_chunk`（当前默认 10，即每次推理后只执行 chunk 前 10 步就重新请求新观测重新规划）或调小安全裁剪来"压制"问题——先回退到冒烟测试模式，把现象记录下来。
4. **可选的进阶优化**（不是必须项，效果未在真机验证过）：训练机上验证过更高频重规划（离线模拟里用 stride=5，即约每 5 帧重新查询一次）配合时间集成能显著降低跳变；`actions_per_chunk` 调小到 5 左右、更频繁地重新请求观测，在闭环执行层面是同一个方向的思路，但这个 server 目前没有做重叠 chunk 的加权融合（只是简单的"每 N 步重新规划，直接截取新 chunk 的前 N 步"），效果不完全等价，如果要引入真正的时间集成融合需要改 `_predict_action_chunk` 的逻辑，这是一处可以做但本任务未做的改动。

## 5. 报告纪律

- 这个 checkpoint 只有训练集诊断（且是离线、非闭环），没有任何真机验证记录。任何在 Thor 上做的测试结论都应该明确标注"首次真机测试"，不要用训练机上的离线指标（MAE、collapse 等）来暗示或保证真机表现。
- 遇到环境/依赖/连通性问题，如实报告卡在哪一步、报什么错，不要为了推进任务而绕过第 4 节的安全检查顺序。
