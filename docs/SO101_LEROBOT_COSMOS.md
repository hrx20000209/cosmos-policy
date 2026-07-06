# SO101 LeRobot → Cosmos Policy 微调流程

## 1. 整体流程

```text
LeRobot SO101 dataset
→ SO101LeRobotCosmosDataset
→ Cosmos Policy 11-slot joint latent sequence
→ full / partial DiT fine-tuning
→ action latent frame 生成与 action chunk 提取
→ 通过离线物理曲线检查后再进入 SO101 部署
```

当前 adapter 只支持三路相机，并固定使用以下 latent 顺序：

```text
0 blank
1 current proprio
2 current wrist-left
3 current wrist-right
4 current primary
5 action chunk
6 future proprio
7 future wrist-left
8 future wrist-right
9 future primary
10 value placeholder
```

图像经过 4 次 temporal duplicate 后组成 `[3, 41, 224, 224]`。Action 和 proprio 不伪装成像素，而是在 VAE 编码后注入对应 latent frame。

## 2. 环境

Cosmos 环境需要能导入本地 LeRobot fork：

```bash
export PYTHONPATH=/home/rxhuang/Projects/lerobot/src:$PYTHONPATH
```

## 3. 生成 T5 embeddings

```bash
python -m cosmos_policy.datasets.save_so101_lerobot_t5_text_embeddings \
  --repo_id local/three_cubes_1 \
  --root /data/rxhuang/three_cubes_1 \
  --output /data/rxhuang/three_cubes_1/so101_t5_embeddings.pkl
```

Dataset 会检查每个 task string 是否存在 embedding，缺失时直接报错。

## 4. 生成 SO101 stats

```bash
python -m cosmos_policy.datasets.export_so101_lerobot_stats \
  --repo_id local/three_cubes_1 \
  --root /data/rxhuang/three_cubes_1 \
  --action_mode absolute \
  --output /data/rxhuang/three_cubes_1/so101_dataset_statistics.json
```

Stats 独立保存 action/proprio 的 min、max、mean、std。训练归一化与未来部署反归一化必须使用同一个文件，禁止替换为 ALOHA stats。

## 5. Dataset debug

```bash
python -m cosmos_policy.scripts.debug_so101_cosmos_dataset \
  --repo_id local/three_cubes_1 \
  --root /data/rxhuang/three_cubes_1 \
  --t5_text_embeddings_path /data/rxhuang/three_cubes_1/so101_t5_embeddings.pkl \
  --dataset_stats_path /data/rxhuang/three_cubes_1/so101_dataset_statistics.json \
  --output /data/rxhuang/three_cubes_1/so101_dataset_debug.png
```

脚本检查 episode 开头、中间、末尾的 current/future images、action padding、维度、物理范围和归一化范围。
它还会在训练前明确打印 action key、action mode、joint order、逐维 action range 和 gripper range。

## 6. Fine-tuning 模式

- `full_dit`：训练完整 `model.net`；tokenizer/VAE/text encoder 冻结。
- `partial_dit_last_n`：只训练最后 N 个 DiT blocks、`final_layer` 和可用的 final norm。
- `all`：训练 tokenizer/VAE 以外的模型参数。
- `action_only_head_if_exists`：由于 Cosmos Policy 没有独立 action head，会打印中文警告并回退到 partial DiT。

检查可训练参数：

```bash
torchrun --nproc_per_node=1 --master_port=12342 \
  -m cosmos_policy.scripts.debug_so101_trainable_params \
  --finetune_mode partial_dit_last_n \
  --train_last_n_dit_blocks 8
```

## 7. Loss 模式

- `action_only`：只计算 action latent frame 的 EDM loss。
- `future_state_only`：只计算 future proprio 和三路 future image。
- `joint_action_future_state`：同时计算 action 和 future state；默认不含 value。
- `all`：保留原始全部 latent loss。

可先在 CPU 上精确检查各模式选中的 latent slot：

```bash
python -m cosmos_policy.scripts.debug_so101_loss_masks
```

## 8. Partial DiT 微调

```bash
SO101_LEROBOT_ROOT=/data/rxhuang/three_cubes_1 \
SO101_LEROBOT_REPO_ID=local/three_cubes_1 \
PYTHONPATH=/home/rxhuang/Projects/lerobot/src:$PYTHONPATH \
torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_so101_lerobot \
  model.config.finetune_mode=partial_dit_last_n \
  model.config.train_last_n_dit_blocks=8 \
  model.config.so101_loss_mode=joint_action_future_state
```

## 9. Full DiT 微调

```bash
torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_so101_lerobot \
  model.config.finetune_mode=full_dit \
  model.config.so101_loss_mode=joint_action_future_state
```

Full DiT 的 optimizer state 显存需求远高于 partial 模式，单张 24 GB GPU 不保证可运行。

## 10. Action-only baseline

```bash
torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_so101_lerobot \
  model.config.finetune_mode=partial_dit_last_n \
  model.config.train_last_n_dit_blocks=8 \
  model.config.so101_loss_mode=action_only
```

## 11. 常见错误

- **camera key 不匹配**：显式传入三个角色 `primary/wrist_left/wrist_right`。
- **action/proprio dim 错误**：维度从 `dataset.meta.features` 读取，不由 ALOHA 常量决定。
- **T5 embedding 缺失**：先运行 embedding 脚本，task string 必须完全一致。
- **stats 不匹配**：stats 的 `action_mode` 必须与 dataset 一致。
- **absolute/delta 混淆**：训练前打印 action mode、names、range；部署必须使用相同语义。
- **episode 尾部**：LeRobot `_is_pad` 会复制最后有效 action/state/image。
- **物理曲线不对齐**：即使 latent loss 下降也不能进入真机；必须检查 30/50-step predicted-vs-GT 曲线和 gripper 方向。

## 12. 当前边界

本阶段只实现数据、训练、loss mask 与 debug。真实机器人 client/server 不作为当前验收重点；在离线 action 曲线通过前，不应发送电机命令。

当前数据 metadata 表明 action 是 6D absolute，顺序为
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`。
这只能证明训练数据 schema；在 SO101 runtime 端打印并逐项核对相同 key、维度、单位、顺序和 gripper 方向之前，不能认定可真机部署。

## 13. 本机已验证结果

- 真实数据共 51,387 帧、100 episodes、30 fps；开头/中间/结尾 sample 与 episode-end padding 均通过。
- `partial_dit_last_n=8` 解冻 555,358,208 / 1,956,413,440 参数（28.3865%）。
- 20-step joint-loss smoke test 完成，无 NaN/OOM，峰值 PyTorch GPU memory 约 13.35 GiB。
- checkpoint 保存于 `so101_partial8_joint_20step_20260706_163909/checkpoints/iter_000000020`。

20 step 只证明数据、反向传播、optimizer 和 checkpoint 链路可运行；loss 会随随机样本与 diffusion sigma 波动，不能据此判断策略成功。下一验收门是从该 checkpoint 离线生成 action chunk，反归一化后逐关节与 ground truth 曲线对齐。

## 14. 离线 action curve 验收

训练 checkpoint 必须先经过以下离线检查，不能只看 training loss：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/rxhuang/Projects/lerobot/src:$PYTHONPATH \
python -m cosmos_policy.scripts.eval_so101_action_curves \
  --repo_id local/three_cubes_1 \
  --root /data/rxhuang/three_cubes_1 \
  --checkpoint /path/to/checkpoints/iter_000002000 \
  --t5_text_embeddings_path /data/rxhuang/three_cubes_1/so101_t5_embeddings.pkl \
  --dataset_stats_path /data/rxhuang/three_cubes_1/so101_dataset_statistics.json \
  --num_samples 16 \
  --chunk_size 50 \
  --num_denoising_steps 10 \
  --output_dir /data/rxhuang/three_cubes_1/offline_eval_partial8_joint_2k
```

脚本保存每个样本的六关节 predicted/GT/absolute-error 曲线和原始 NPZ，并汇总 per-joint MAE、RMSE、direction agreement、尺度比、前 5 步/完整 50 步 MAE、gripper 方向、恒定输出与均值坍缩风险。

完整 episode 可用固定 stride 重新 query 并拼接 action chunk：

```bash
python -m cosmos_policy.scripts.eval_so101_full_episode_action_curve \
  --repo_id local/three_cubes_1 \
  --root /data/rxhuang/three_cubes_1 \
  --checkpoint /path/to/checkpoints/iter_000002000 \
  --t5_text_embeddings_path /data/rxhuang/three_cubes_1/so101_t5_embeddings.pkl \
  --dataset_stats_path /data/rxhuang/three_cubes_1/so101_dataset_statistics.json \
  --episode 0 --query_stride 10 --chunk_size 50 --num_denoising_steps 10 \
  --output_dir /data/rxhuang/three_cubes_1/offline_eval_full_episode
```

该评估只离线拼接预测，不发送机器人命令。每个 query 只取前 `query_stride` 步，因此比单个 50-step chunk 更接近未来短 open-loop 的使用方式。

## 15. 五实验 checkpoint sweep

先复制并填写 TSV manifest；每行固定为 `实验名<TAB>run 目录<TAB>GPU ID`：

```bash
cp cosmos_policy/scripts/so101_sweep_manifest.example.tsv /tmp/so101_sweep.tsv
```

确认每个 run 都包含 500、1000、1500、2000 checkpoint 后运行：

```bash
PYTHON_BIN=/home/rxhuang/Projects/cosmos-policy/.venv/bin/python \
bash cosmos_policy/scripts/eval_so101_parallel_sweep.sh /tmp/so101_sweep.tsv
```

脚本按实验并行、按 checkpoint 串行执行。每个 checkpoint 都生成一次 16-sample action curve，
并对 episode 0/10/30/60/90 生成 stride=10 的完整拼接曲线；缺少任一 checkpoint 会在推理前中止，
不会默认跳过 1500。汇总结果写入 `offline_eval_sweep/{summary,training_dynamics}.{csv,md}`，
best checkpoint 按 first-5 MAE、方向、gripper MAE、jump、collapse/constant/discontinuity 依次选择，
而不是默认使用 2000 step。

启动 sweep 前的 checkpoint 清理必须先执行只读 inventory（例如 `find ... -name 'iter_*'` 和
`du -sh`），逐路径报告并得到人工确认后才能删除。原始 LeRobot 数据、T5 embedding、SO101 stats、
dataset debug 图和所有 action curve 评估目录始终属于保护范围。
