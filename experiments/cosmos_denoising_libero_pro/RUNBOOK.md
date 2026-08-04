# Cosmos denoising × LIBERO / LIBERO-PRO

本目录只评测 Cosmos Policy 的 `action_only_usage`，固定 action horizon 16、同步执行、关闭 future RGB decode。正式步数为 `1,2,3,4,5,6`；不包含 LingBot-VA、量化、auto-regressive future、动态 replanning、真异步或 8-step。

大文件与逐 episode 原始结果位于：

`/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/`

代码树中保留 configs、manifest、汇总、图与报告。每个正式配置由独立进程运行；runner 以 `(config_id, task_uid, variant_id, seed, init_state_index)` 为去重键，已完成行自动跳过，episode 和 request JSONL 均即时追加。

## 固定口径

- Cosmos commit: `eaf393e5b5dbe8ab03a6ab08061c99aad6ba2e1e`
- LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- LIBERO-PRO commit: `eafdb809426b13153aa1e4c42d6601844217dfec`
- checkpoint SHA256: `8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2`
- seed: 195；init index: 0；PRO 生成 seed: 28
- deterministic: true；randomize_seed: false
- OSMesa:
  `/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu`

## 运行顺序

```bash
source .venv/bin/activate
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

python experiments/cosmos_denoising_libero_pro/scripts/prepare_assets.py
python experiments/cosmos_denoising_libero_pro/scripts/build_manifests.py
CUDA_VISIBLE_DEVICES=4 python experiments/cosmos_denoising_libero_pro/scripts/precompute_t5.py
python experiments/cosmos_denoising_libero_pro/scripts/validate_manifests.py
```

正式运行命令由 `scripts/launch_sweep.py` 写入 logs，允许重复执行以续跑。

## 缩减后的第二轮

初始自动规则把所有 “original success / PRO failure” 都纳入 hard 集，得到
104 个变体和 5,824 个新增 episode。经阶段审查后停止该大 sweep，已产生的
JSONL 保留。缩减方案固定为：保留所有 denoising 直接分歧变体，其他分布偏移
失败按 perturbation 固定哈希最多选 3 个代表；使用 selected steps
`1,3,5,6`、seeds `195,196,197` 和 init indices `0,1,2`。

```bash
python experiments/cosmos_denoising_libero_pro/scripts/prepare_reduced_stage2.py

python experiments/cosmos_denoising_libero_pro/scripts/launch_sweep.py \
  --manifest experiments/cosmos_denoising_libero_pro/manifests/stage2_hard_subset_reduced.jsonl \
  --config experiments/cosmos_denoising_libero_pro/configs/sweep_stage2.yaml \
  --phase stage2_hard_subset_reduced --gpus 0,1,2,3,4,4,5,5,6,6,7,7 \
  --shards-per-config 2 --retry-failures

python experiments/cosmos_denoising_libero_pro/scripts/launch_sweep.py \
  --manifest experiments/cosmos_denoising_libero_pro/manifests/stage2_language_all_paraphrases.jsonl \
  --config experiments/cosmos_denoising_libero_pro/configs/sweep_stage2.yaml \
  --phase stage2_language_all_paraphrases --gpus 0,1,2,3,4,4,5,5,6,6,7,7 \
  --shards-per-config 2

python experiments/cosmos_denoising_libero_pro/scripts/launch_sweep.py \
  --manifest experiments/cosmos_denoising_libero_pro/manifests/stage2_disagreement_video_replay_reduced.jsonl \
  --config experiments/cosmos_denoising_libero_pro/configs/sweep_stage2.yaml \
  --phase stage2_video_replay_reduced --gpus 4,5,6,7 --record-video
```

缩减决策与确切数量见
`summaries/stage2_reduced_plan.json`。旧的 5,824/833 行 manifest 仅作审计，
不再是完成目标。

## 官方 PRO 资产说明

锁定的官方提交注册了 `*_lan/object/swap/task/env` benchmark 名称，但未提交成品变体 BDDL/init。本实验直接调用该提交原样 `perturbation.py` 与五份 YAML，各次只开启一个 flag，生成文件和 SHA256 记录在 `assets/libero_pro_asset_audit.json`。官方 environment perturbator 固定目标为 `living_room_table`；原本已处于该环境的任务可能产生字节不变的 BDDL，这会标记为 `variant_applied=false`，不会静默伪装成有效扰动。
