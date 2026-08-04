# Cosmos Policy 端侧低精度推理阶段报告

日期：2026-07-29  
分支：`exp/cosmos-tensorrt-fp8`  
硬件：NVIDIA GeForce RTX 4090 D，Ada sm_89，24 GB  
实验目录：`experiments/cosmos_trt_quantization/`

## 1. 摘要

本阶段实际完成了环境隔离、固定输入捕获、单步 DiT wrapper、六种 torchao
模式的 30+200 latency 测量、四种模式的 torch.profiler trace、BF16 Nsight
Systems trace、BF16 TensorRT full-graph 编译尝试、100+1000 单 denoiser
BF16 eager 基准和 1–5 步 100+300 sweep。

核心结论：

1. torchao W8A8 的主要瓶颈是 1400 次动态 activation scale 及其引发的
   50424 次 `_local_scalar_dense`、约 6.7 万次 D2H 小传输；INT4-WO 则有
   1400 次 tinygemm，累计 1246.17 ms。
2. Cosmos DiT 在替换 export-hostile 的 TransformerEngine 算子并增加一个
   BF16 cast converter 后，能够在内存中编译为单一 TensorRT 分区：
   6492/6492 算子支持、无 PyTorch fallback。
3. 该 BF16 engine 的序列化失败：PyTorch 2.7 的 ExportedProgram 路径对
   大于 4 GiB 的 engine string 使用了不足的 pickle protocol。随后已实现
   TorchScript 保存路径，但继续执行环境不再暴露 `/dev/nvidia*`，无法重建、
   重载和计时。因此 BF16 TensorRT 尚不是稳定 baseline。
4. 由于 BF16 TensorRT 稳定性门禁未通过，严格没有开始 ModelOpt FP8 PTQ、
   FP8 engine、原始 LIBERO rollout 或 LIBERO-Plus rollout；不能声称获得
   TensorRT FP8 加速。
5. BF16 eager 的 3-step policy p50 为 326.39 ms，相对 5-step 的
   448.42 ms 快 1.37×；固定输入 action cosine 为 0.9999992、L2 为
   0.00613，是当前最有希望的 latency–error 候选，但尚无 rollout 成功率
   证据。

## 2. 上一阶段结果回顾

上一阶段在原始 LIBERO-10 上得到 BF16 48/50，policy latency 约 492 ms；
torchao INT8-WO、W8A8、INT4-WO 分别约 608、2530、1506 ms。torch.compile
未改善这些量化路径。上述结果构成本阶段“停止扩大 torchao rollout、转向
静态 NVIDIA engine”的依据。

## 3. 为什么没有使用 llama.cpp

Cosmos Policy 是带视频 latent、时序条件、cross-attention 和扩散 scheduler
的 DiT policy，不是标准自回归 LLM。其主干输入为
`[B,C,T,H,W]` 视频/动作 latent，推理要重复调用 denoiser；llama.cpp 的
token-by-token KV-cache/GGUF 路径与该计算图不匹配，因此本阶段没有使用。

## 4. 环境审计

主环境保持不变，新增隔离环境 `/data/rxhuang/envs/cosmos-trt`：

| 组件 | 版本/状态 |
|---|---|
| GPU | RTX 4090 D，compute capability 8.9 |
| Driver | 560.35.03 |
| PyTorch / CUDA runtime | 2.7.0+cu128 / 12.8 |
| 系统 CUDA toolkit | 12.6.85 |
| cuDNN | 9.7.1 |
| torchao | 0.11.0 |
| TransformerEngine | 2.2 |
| TensorRT | 10.9.0.34（隔离环境） |
| Torch-TensorRT | 2.7.0（隔离环境） |
| NVIDIA ModelOpt | 0.27.0（隔离环境） |
| ONNX / ORT GPU | 1.17.0 / 1.20.1 |
| TensorRT-RTX | 未安装 |
| trtexec | 系统无可执行文件 |
| Nsight Systems / Compute | 2024.5.1 / 2024.3.2 |

Ada sm_89 可测试 FP8，但本报告没有测试或声称 NVFP4。完整安装命令见
`configs/environment.lock.md`；主 `.venv` 中的 PyTorch/CUDA 未被替换。

## 5. torchao 量化变慢的具体原因

统一 policy 基准（30 warm-up + 200 measured）：

| 模式 | p50 | p95 | p99 | 相对 BF16 |
|---|---:|---:|---:|---:|
| BF16 eager | 461.02 | 540.40 | 568.13 | 1.000× |
| fake INT4 | 451.19 | 511.77 | 555.60 | 1.022× |
| INT8-WO | 552.32 | 633.69 | 660.80 | 0.835× |
| torchao FP8 | 1021.65 | 1093.72 | 1130.93 | 0.451× |
| INT4-WO | 1494.88 | 1566.30 | 1589.56 | 0.308× |
| W8A8 | 2512.16 | 2581.07 | 2662.31 | 0.184× |

单位均为 ms。

### W8A8

- 1400 次 `ampere_igemm_int8_128x128_ldg4_nn` 只花约 219.20 ms；
- 动态 quantize kernel 9866 次、约 161.06 ms；
- cast/copy 83679 次、约 175.46 ms；
- 50423 次 D2H pinned、16801 次 D2H pageable；
- `_local_scalar_dense` 50424 次；
- 总 GPU 事件 115765 次，是 BF16 9370 次的 12.35 倍。

所以 W8A8 主要慢在动态 activation scale、device scalar 同步和 launch
开销，而非 INT8 GEMM 本身。

### INT8 weight-only

主 GEMM 仍是 `ampere_bf16_s1688gemm_bf16_*`。该路径把量化权重恢复/转换
后做 BF16 GEMM，额外增加 1401 次 direct-copy/cast 和大量 elementwise。
它是显存格式优化，不是该 shape 下的真实 INT8 计算加速。

### INT4 weight-only

1400 次
`tinygemm_m16n8k16_chunk_kernel<...,BLayout_TC_int4<8,128>,...>`
累计 1246.17 ms，占 trace GPU 时间 86.68%。INT4 unpack 与计算融合在该
kernel 内，无法诚实地拆出独立 unpack 时间；可以明确的是 batch=1、M=1764、
K/N=2048/8192 的 DiT shape 没有从 tinygemm 获益。

### BF16 融合路径和 torch.compile

BF16 已走 Tensor Core GEMM、FlashAttention、
`transformer_engine::normalization::rmsnorm_fwd_general_kernel` 和
`transformer_engine::fused_rope_forward_kernel`。替换 Linear 后产生的
scale/cast 边界破坏了连续 BF16 路径。

Dynamo 不能 trace `transformer_engine_torch.PyCapsule.rmsnorm_fwd`，并对
checkpoint `context_fn` 给出警告。上一阶段 compiled p50 与 eager 基本相同，
因为 compile 不能消除 torchao kernel 内部的 dynamic scale、D2H 同步或
tinygemm shape 问题。完整证据见
`report/torchao_latency_diagnosis.md`。

## 6. Cosmos DiT wrapper

`wrappers/cosmos_denoiser_wrapper.py` 直接包裹真实
`MinimalV1LVGDiT.forward`，不包含 LIBERO env、tokenizer、VAE、文件 I/O 或
Python scheduler。真实捕获输入：

| 张量 | shape | dtype |
|---|---|---|
| noisy/action latent | `[1,16,9,28,28]` | BF16 |
| timestep | `[1,9]` | BF16 |
| text/cross-attention | `[1,512,1024]` | BF16 |
| condition mask | `[1,1,9,28,28]` | FP32 |
| fps | `[1]` | BF16 |
| padding mask | `[1,1,224,224]` | BF16 |

wrapper 与原始 net 输出 shape/dtype 完全一致，cosine
1.000000119、max abs 0、重复执行 max abs 0，无 NaN/Inf。fixture 只有
1.4 MB，位于 `calibration/fixed_denoiser_inputs.pt`。

## 7. TensorRT 编译方法

按顺序实际尝试：

1. 原生 TE graph：FakeTensor export 在 TE RMSNorm 报
   `Input x is not allocated`。
2. 替换 113 个 TE RMSNorm：可继续导出，但仍遇到 TE attention/RoPE。
3. 再替换 56 个 TE attention 为 PyTorch SDPA，并关闭 fused RoPE：
   `torch.export` 成功。
4. Torch-TensorRT 内置 capability validator 把 454 个
   `aten._to_copy(dtype=BF16)` 错判为不支持；原始 dry-run 为
   6038/6492（93.01%），61 个 TRT 分区 + 61 个 PyTorch 分区。
5. 在仓库内注册仅匹配 BF16 的高优先级 cast converter 后，dry-run 为
   6492/6492、单一 TRT 分区。
6. 实际 build 又遇到 padding mask `repeat` 被分解成临时 10-D expand，
   超过 TensorRT shuffle rank。将其改为语义等价的 5-D `expand` 后，
   full graph engine 在内存中成功构建。

## 8. BF16 TensorRT baseline

### 成功部分

- build 时间：388.05 s；
- 节点覆盖率：100%（6492/6492）；
- 参数覆盖率：100%；
- TRT 分区：1；
- PyTorch fallback：0；
- engine boundary：0；
- 输出 finite；
- 相对重写后 eager：cosine 0.9999347，max abs 0.109375，L2 4.12242。

### 未通过部分

engine 未能持久化。`torch_tensorrt.save(..., exported_program)` 失败：

`OverflowError: serializing a string larger than 4 GiB requires pickle protocol 4 or higher`

已将保存实现改为 TorchScript zip 路径，但后续执行容器不再暴露
`/dev/nvidia*`，无法重新 build、reload 和 benchmark。因而：

- 不能给出 BF16 TRT p50/p95；
- 不能证明 engine 可重复加载；
- 不能把“内存中构建成功”写成“稳定 BF16 TensorRT baseline”；
- `engines/` 中只有 3.7 GB 的 exported program 中间产物，不是可用 engine。

另一个数值限制是 TE→ATen 重写本身相对原始 eager cosine 0.9999757，
未达到 wrapper 的严格 0.99999 等价门槛，但高于本阶段 engine action gate
0.999。engine 对原始未重写 graph 的组合误差没有直接实测，因此不外推。

## 9. FP8 PTQ 方法与 Calibration 数据

本阶段在 BF16 TensorRT 稳定性门禁处停止。没有采集 128–512 条 calibration
observation，没有生成 ModelOpt scale，也没有构建 FP8 engine。这样做是为了
避免在 BF16 engine 尚不能持久化/复载时产生不可比较的 FP8 结果。

计划保持不变：未来使用原始无扰动 LIBERO、多 suite/task/seed 的 agent view、
wrist、proprio 和文本 embedding，ModelOpt 静态 PTQ + 显式 Q/DQ；norm、
softmax、timestep、scheduler 和 action reconstruction 默认 BF16。

## 10. 实际层精度

由于 FP8 PTQ 未开始，没有可报告的逐层实际 TensorRT precision、Q/DQ scale
或 layer output error。不能把 torchao FP8 的 280 个动态量化 Linear 当作
TensorRT FP8 层精度审计。

## 11. TensorRT 覆盖率

| 图版本 | 节点覆盖 | 参数覆盖 | TRT/PyTorch 分区 | fallback |
|---|---:|---:|---:|---:|
| 原生 TE | export 失败 | 不适用 | 不适用 | 不适用 |
| TE 重写、内置 converter | 93.01% | 近似 100% | 61 / 61 | 454 个 cast |
| + BF16 cast converter + mask rewrite | 100% | 100% | 1 / 0 | 0 |

FLOP/latency 覆盖没有用虚构数字填充。最终图按节点和参数为 100%，但没有
持久化 engine 的 latency trace，故 latency coverage 标为不可用。

## 12. Microbenchmark

单 denoiser BF16 eager 使用 100 warm-up + 1000 measured：

- mean 61.00 ms；
- p50 60.58 ms；
- p95 63.94 ms；
- p99 68.11 ms。

BF16 TRT/FP8 TRT 因 engine 不可复载而没有计时。真正量化 speedup 必须比较
BF16 TRT 与 FP8 TRT；本报告没有用 BF16 eager 对 torchao FP8 的结果冒充
TensorRT precision speedup。

## 13. 完整 policy latency

完整 5-step policy 的 30+200 基准见第 5 节。BF16 p50 461.02 ms；torchao
FP8、INT8-WO、W8A8、INT4-WO 均更慢。没有 BF16 TRT/FP8 TRT p50/p95。

## 14. 显存

本轮单进程 profiler 的 `max_memory_allocated` 六种模式均约 7727 MB；该数字
包含完整 policy、tokenizer/缓存和 allocator 行为，不足以证明 torchao 显存
没有下降。由于没有持久化 TRT engine，也没有可比较的 engine peak memory
与 engine size。图表对缺失值留空，而不是填 0。

## 15. 数值误差

相对固定 BF16 action：

| 模式 | cosine | L1 | L2 | max abs | NaN/Inf |
|---|---:|---:|---:|---:|---:|
| INT8-WO | 0.9999960 | 0.000890 | 0.01422 | 0.00463 | 0 |
| W8A8 | 0.9999808 | 0.002116 | 0.03078 | 0.00762 | 0 |
| torchao FP8 | 0.9999405 | 0.004829 | 0.07526 | 0.02830 | 0 |
| INT4-WO | 0.9996271 | 0.007914 | 0.14068 | 0.05843 | 0 |
| fake INT4 | 0.9995420 | 0.009591 | 0.15210 | 0.05004 | 0 |

torchao FP8 不是 TensorRT FP8，不能回答 TensorRT FP8 是否降低 action 精度。

## 16. 去噪步数实验

每个配置 100 warm-up + 300 measured，同一 observation、seed 和 scheduler：

| 配置 | policy p50 / p95 (ms) | 实际 denoiser total p50 | cosine vs 5-step | L2 |
|---|---:|---:|---:|---:|
| 1-step | 200.69 / 243.37 | 60.22 | 0.9999612 | 0.04385 |
| 2-step | 201.57 / 266.61 | 60.19 | 0.9999612 | 0.04385 |
| 3-step | 326.39 / 388.22 | 180.32 | 0.9999992 | 0.00613 |
| 4-step | 387.79 / 446.32 | 240.15 | 0.9999994 | 0.00549 |
| 5-step | 448.42 / 513.93 | 300.00 | 1.0000000 | 0 |

当前 scheduler 下 1-step 和 2-step 都只发生一次 net forward，因此输出完全
相同；这不是测量错误，也不能把“2”解释为两次 denoiser。3-step 相对 5-step
p50 快 1.37×，且固定输入误差很小，是最合理的下一轮 rollout 候选。

不过单 observation 的 action cosine 不能替代成功率。当前最佳“已验证成功率”
配置仍是上一阶段 BF16 eager 5-step；当前最佳 microbenchmark Pareto 候选是
BF16 eager 3-step。

## 17. 原始 LIBERO pilot

未运行。原因不是算力不足，而是 speedup gate 明确要求稳定、可复载、已计时的
BF16/FP8 TensorRT engine。本阶段没有满足。

## 18. LIBERO-Plus pilot

未运行，也没有扩大已知很慢的 torchao W8A8/INT4 rollout。符合阶段门禁。

## 19. 失败配置

| 配置/阶段 | 失败原因 | 证据 |
|---|---|---|
| 原生 TE export | TE RMSNorm FakeTensor 无 storage | attempt0 log/report |
| TE attention/RoPE export | fused RoPE/export 不兼容 | attempt2 log/report |
| 内置 converter full compile | BF16 `_to_copy` validator 漏项 | dry-run report |
| 首次 custom converter build | 10-D expand 超 TRT shuffle rank | build log |
| 最终 full graph save | >4 GiB string / pickle protocol | partition report |
| TorchScript follow-up | 后续容器无 `/dev/nvidia*` | final status |
| W8A8 Nsight follow-up | 权限切换后报告未落盘 | nsys log |

## 20. 局限性

- 固定 observation microbenchmark 不能代表任务成功率分布；
- 没有 BF16 TRT 可复载 engine，因此没有 BF16 TRT latency；
- 没有 FP8 calibration、layer precision 或 TensorRT FP8 latency；
- BF16 TensorRT 使用了 TE→ATen 语义近似重写；
- profiler 会放大 CPU 调度时间，kernel 总时间和事件计数比 profile wall time
  更适合做归因；
- GPU clock 未锁定，但各正式模式固定单卡、独立进程并记录 p50/p95/p99；
- 当前共享 LIBERO-Plus task order 在实验期间漂移为 2519 项，固定输入文本用
  最长原始指令前缀匹配缓存，已记录在 RUNBOOK。

## 21. 是否值得继续量化研究

torchao 动态量化路径不值得继续扩大 rollout。TensorRT 路线仍有研究价值，
因为 full graph 已证明可以达到单分区、零 fallback；下一个工程问题是大
engine 持久化，而不是算子覆盖。只有解决保存/重载后，ModelOpt 静态 FP8
才值得继续。

## 22. 是否达到端侧加速目标

没有。当前没有 BF16 TRT vs FP8 TRT 的公平 latency 对照，也没有满足 FP8
≥1.20× speedup gate。不能声称获得低比特端侧加速。

去噪步数方面，3-step 在 microbenchmark 上已达到 1.37×，比现有 torchao
量化更有效；但没有 rollout 成功率，暂时只能称为候选。

## 23. 下一阶段异步框架计划

现有 open-loop 流程是“完整推理 chunk → 执行 E 个 action → 停止 →
再推理”，不是真正 inference/execution overlap。详细方案位于
`report/async_pipeline_design.md`，包含：

- inference worker 与 action execution worker；
- observation/inference 时间戳；
- global action index、chunk start、execution cursor；
- deadline、stale chunk、replacement；
- prefix freezing、action inpainting/overlap；
- deadline miss rate、observation age 和 overlap ratio。

在 engine 尚不可复载前不大改运行框架。3-step 可先在同步原始 LIBERO pilot
中验证；真正异步实验仍应等待稳定 backend。

## 24. 完整复现命令

```bash
cd /home/rxhuang/Projects/cosmos-policy
git switch exp/cosmos-tensorrt-fp8

# wrapper
CUDA_VISIBLE_DEVICES=5 .venv/bin/pytest -q \
  experiments/cosmos_trt_quantization/tests/test_cosmos_denoiser_wrapper.py

# torchao latency（每个 mode 独立进程）
source experiments/liberoplus_quantization/env.sh
CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 .venv/bin/python \
  experiments/cosmos_trt_quantization/scripts/profile_torchao.py \
  --mode int8_backbone --phase latency --warmup 30 --iters 200

# torch.profiler
CUDA_VISIBLE_DEVICES=7 MUJOCO_EGL_DEVICE_ID=7 .venv/bin/python \
  experiments/cosmos_trt_quantization/scripts/profile_torchao.py \
  --mode int4_weight_only --phase torchprof --warmup 30

# TensorRT（隔离环境）
source /data/rxhuang/envs/cosmos-trt/bin/activate
CUDA_VISIBLE_DEVICES=5 python \
  experiments/cosmos_trt_quantization/scripts/build_bf16_trt.py \
  --rewrite-te all --require-full --register-bf16-cast-converter \
  --workspace-gb 8

# 单 denoiser eager
CUDA_VISIBLE_DEVICES=5 python \
  experiments/cosmos_trt_quantization/scripts/benchmark_denoiser.py \
  --backend bf16_eager --warmup 100 --iters 1000

# 1–5 steps
source experiments/liberoplus_quantization/env.sh
CUDA_VISIBLE_DEVICES=6 MUJOCO_EGL_DEVICE_ID=6 .venv/bin/python \
  experiments/cosmos_trt_quantization/scripts/denoising_sweep.py \
  --warmup 100 --iters 300

# 汇总和图
python experiments/cosmos_trt_quantization/scripts/summarize_results.py
python experiments/cosmos_trt_quantization/scripts/make_figures.py
```

## 25. 结果索引

- 汇总：`summaries/backend_latency_summary.csv`
- speedup gate：`summaries/speedup_gate.json`
- BF16 TRT 最终状态：`summaries/bf16_trt_final_status.json`
- 去噪 sweep：`summaries/denoising_sweep_summary.csv`
- kernel 汇总：`profiles/torchao_kernel_summary.csv`
- graph break：`profiles/torchao_graph_breaks.txt`
- TRT coverage：`profiles/bf16_trt_dryrun*.txt`
- 构建日志：`logs/bf16_trt_build*.log`
- 图表：`figures/`（14 组 PNG 300 DPI + PDF）
- 异步设计：`report/async_pipeline_design.md`

