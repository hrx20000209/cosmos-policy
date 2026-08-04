# WAM / LIBERO 统一实验

本目录在不复制模型实现的前提下，以 adapter 统一 Cosmos Policy 和 LingBot-VA。正式实验前先阅读 [`reports/model_runtime_audit.md`](../reports/model_runtime_audit.md)。

## 环境与 checkpoint

三个仓库默认位置已写入 YAML，可修改：

```text
/home/rxhuang/Projects/cosmos-policy
/home/rxhuang/Projects/lingbot-va
/home/rxhuang/Projects/LIBERO
```

Cosmos config 默认复用 `/data/rxhuang/models/cosmos-policy-libero-2b` 下的本地权重、statistics 和 T5 cache。LingBot config 中的 `/path/to/lingbot-va-posttrain-libero-long` 必须替换为真实 LIBERO checkpoint。不要用 RoboTwin、SO101 或 base checkpoint 冒充 LIBERO baseline。

Cosmos 与 LingBot 依赖环境不同。建议从对应 repo 的官方环境运行统一入口，并保证另外两个 repo 可从 YAML 路径访问。Jetson 不要求 H100 专用功能；默认只用常规 PyTorch CUDA stream/event，资源采样在 `nvidia-smi` 字段不可用时返回 null。

当前桌面机用户没有 `/dev/dri` 权限时，可用 `/data/rxhuang` 中的用户态 OSMesa 做 CPU MuJoCo rendering，同时保留 CUDA inference：

```bash
export CUDA_VISIBLE_DEVICES=5
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
```

不要把 OSMesa 的 CPU rendering 性能外推到 Jetson；GPU kernel latency 与 episode wall time应分别解释。

## Step 1：审计和基础测试

```bash
cd /home/rxhuang/Projects/cosmos-policy
python -m pytest -q tests
python -m compileall -q runtime adapters analysis experiments
```

只验证 harness 的 mock smoke（结果带 `"mock": true`，禁止用于论文表格）：

```bash
python experiments/run_wam_libero_experiment.py \
  --model cosmos \
  --task_suite libero_10 \
  --config experiments/configs/cosmos_baseline.yaml \
  --mock
```

## Step 2/3：baseline smoke 与 correctness

先执行 fail-fast preflight。`--load-model` 只有在 CUDA、资产、真实渲染全部通过后才构造模型：

```bash
.venv/bin/python experiments/preflight_wam_libero.py \
  --model cosmos \
  --config experiments/configs/cosmos_baseline.yaml \
  --task-id 0 \
  --load-model \
  --run-inference
```

Cosmos：

```bash
python experiments/run_wam_libero_experiment.py \
  --model cosmos \
  --task_suite libero_10 \
  --config experiments/configs/cosmos_baseline.yaml
```

LingBot-VA：

```bash
python experiments/run_wam_libero_experiment.py \
  --model lingbot_va \
  --task_suite libero_10 \
  --config experiments/configs/lingbot_baseline.yaml
```

Smoke 固定 task IDs `[0,1]`、每任务 3 episodes、seeds `[195,196,197]`。通过后再用 override 扩大规模：

```bash
python experiments/run_wam_libero_experiment.py \
  --model cosmos \
  --config experiments/configs/cosmos_baseline.yaml \
  --set evaluation.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --set evaluation.episodes_per_task=20
```

正式对比必须先人工核对视频与以下项目：action normalization、两个相机方向/顺序、crop/resize/JPEG、gripper sign、首 chunk placeholder、action horizon 和 control frequency。

## Step 4：denoising sweep

```bash
python experiments/run_denoising_sweep.py \
  --model cosmos \
  --base-config experiments/configs/cosmos_baseline.yaml \
  --cosmos-mode joint_parallel

python experiments/run_denoising_sweep.py \
  --model cosmos \
  --base-config experiments/configs/cosmos_baseline.yaml \
  --cosmos-mode action_only_usage

python experiments/run_denoising_sweep.py \
  --model cosmos \
  --base-config experiments/configs/cosmos_baseline.yaml \
  --cosmos-mode autoregressive_future

python experiments/run_denoising_sweep.py \
  --model lingbot_va \
  --base-config experiments/configs/lingbot_baseline.yaml
```

Cosmos 测 `{1,2,3,4,5,8}`。LingBot 源码默认是 video=20、action=50，因此默认 sweep 为 action `{5,10,20,35,50}`，不是假设的 16。
Cosmos sampler 的参数化单测要求每个公开 step 值都等于完整 denoiser forward 数；这也覆盖容易退化成单次调用的 2-step 边界。
所有 sweep runner 支持重复的 `--set`，并把 override 应用到每个 variant。例如先做固定 task/seed 的低成本 pilot：

```bash
python experiments/run_denoising_sweep.py \
  --model cosmos \
  --base-config experiments/configs/cosmos_baseline.yaml \
  --output-dir /data/rxhuang/wam_libero_outputs/denoising_pilot \
  --steps 1 2 3 4 5 8 \
  --set 'evaluation.task_ids=[0]' \
  --set evaluation.episodes_per_task=1 \
  --set 'evaluation.seeds=[195]' \
  --set evaluation.record_video=false
```

## Step 5：固定输出、动态执行 prefix

```bash
python experiments/run_chunk_prefix_sweep.py \
  --model cosmos \
  --base-config experiments/configs/cosmos_baseline.yaml
```

固定测试 `{1,2,4,8,16}`，并测试 proprio/visual/action-variance/task-stage 规则。模型输出 horizon 不变。`runtime/action_buffer.py` 明确支持 `replace`、`blend`、`temporal_aggregation`、`keep_prefix`。

## Step 6：latest/keyframe/history

```bash
python experiments/run_history_sweep.py \
  --model cosmos \
  --base-config experiments/configs/cosmos_baseline.yaml

python experiments/run_history_sweep.py \
  --model lingbot_va \
  --base-config experiments/configs/lingbot_baseline.yaml
```

Cosmos 只改变哪一个完整 observation 成为 current observation，adapter 会拒绝多个 history observations。LingBot 的 K/S 实验进入流式 VAE/KV cache，并记录 span/interval。

## Step 7：action-first pipeline

```bash
python experiments/run_pipeline_sweep.py \
  --model cosmos \
  --base-config experiments/configs/cosmos_baseline.yaml
```

模式为 `sync_baseline`、`action_first_sync`、`action_first_async_decode`、`async_encode_sync_dit`、`full_pipeline`。Cosmos 的 joint DiT 不变，仅通过 `decode_future_state=False` 延后 VAE decode。LingBot server baseline 已不 decode future RGB。

PyTorch profiler/Nsight：

```bash
nsys profile -t cuda,nvtx,osrt -o outputs/wam_libero/nsys/cosmos \
  python experiments/run_wam_libero_experiment.py \
  --model cosmos \
  --config experiments/configs/cosmos_baseline.yaml \
  --set pipeline.mode=action_first_async_decode
```

不要从 CPU enqueue 时间推断 kernel overlap；必须查看 CUDA events 和 Nsight timeline。多 stream overlap 结论需要在目标 Jetson AGX Thor 和桌面 GPU 分别验证。

## Step 8：动态 denoising

```bash
python experiments/run_dynamic_denoising.py \
  --model cosmos \
  --base-config experiments/configs/cosmos_baseline.yaml
```

统一比较 always-min、always-official、matched random 和 heuristic。

## Step 9：prediction surprise

`runtime/prediction_surprise.py` 提供：

- `PendingPrediction(target_control_step=...)`；
- 严格 checkpoint 对齐的 `SurpriseAligner`；
- latent L1/L2/cosine/channel-normalized distance。

在得到真实 future latent 的运行后再把 trigger 接入 action interruption；禁止把 `z_pred(t+H)` 与 `z_real(t+1)` 比较。

## 日志与分析

每个 run 包含：

```text
runs/<run_id>/
  config.yaml
  episodes.jsonl
  inference_trace.jsonl
  summary.json
  environment.txt
  git_state.txt
  stdout.log
  videos/*.mp4
  actions/*.npy
```

Baseline config 默认将送入模型的 post-flip primary/wrist 视图横向拼接并流式编码为 MP4；sweep 可用
`--set evaluation.record_video=false` 关闭，以避免大量视频 I/O。
每个 episode 默认保存实际执行的 float32 action trace，并在 `episodes.jsonl` 中记录路径与 SHA-256，
用于检查跨模式随机流和动作等价性。

统计和绘图：

```bash
python analysis/summarize_results.py \
  outputs/wam_libero/runs/<run-a> \
  outputs/wam_libero/runs/<run-b> \
  --output reports/summary.json

python analysis/plot_results.py reports/summary.json --output-dir plots
```

统计脚本输出 per-task、macro、aggregate、标准差和 bootstrap 95% CI。GPU stage 用 CUDA events；E2E 用 monotonic clock；默认排除前 10 个 request 并单独保存 cold start。

## 公平性

每个 run 的 config 必须保留 checkpoint、precision、resolution、camera views、history span、model/executed horizon、denoiser calls、VAE counts、cache、control frequency。分别报告：

- Native：各模型官方配置；
- Matched budget：显式说明被匹配的 calls/resolution/replan interval/horizon/precision，以及无法匹配的架构差异。

禁止将 mock、缺 checkpoint、不同 task subset 或不同 seeds 的结果合并。
