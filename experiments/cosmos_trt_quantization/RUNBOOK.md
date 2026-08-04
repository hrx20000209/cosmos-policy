# Cosmos Policy TensorRT / FP8 实验运行记录

> 分支：`exp/cosmos-tensorrt-fp8`  
> 主环境保持不变；TensorRT 实验环境：`/data/rxhuang/envs/cosmos-trt`

## 2026-07-29 17:30 HKT — 建立隔离实验工作区

- 命令：`git switch -c exp/cosmos-tensorrt-fp8`
- Git 基线：见 `git rev-parse HEAD`；工作树原有未提交改动均保留。
- 修改：建立 `experiments/cosmos_trt_quantization/` 目录结构。
- 输出：`engines/`、`calibration/`、`profiles/`、`raw/`、`logs/` 指向
  `/data/rxhuang/cosmos_trt_quantization/`，避免占满 `/home`。
- 状态：成功。
- 问题：GPU 7 被其他进程占用；GPU 5、6 可用。需要固定单卡、无竞争的实验使用 GPU 5。
- 下一步：审计环境，创建与 PyTorch 2.7/CUDA 12.8 匹配的隔离环境。

## 2026-07-29 17:37 HKT — 环境审计与依赖选择

- 命令：`nvidia-smi`、`nvcc --version`、Python 包导入审计。
- Git commit：见 `system_info.txt`。
- 输出：`system_info.txt`（审计脚本运行后生成）、安装日志 `logs/install_*.log`。
- 状态：主环境审计成功；TensorRT 依赖安装进行中。
- 发现：
  - 8 × RTX 4090 D，compute capability 8.9；driver 560.35.03。
  - 主环境 PyTorch 2.7.0+cu128、torchao 0.11.0、TransformerEngine 2.2。
  - 主环境无 TensorRT、Torch-TensorRT、ModelOpt、ONNX；系统无 `trtexec`。
  - CUDA toolkit 12.6，PyTorch 自带 CUDA 12.8 runtime；Nsight Systems/Compute 位于 CUDA 12.6 工具链。
- 版本决策：隔离环境使用 TensorRT 10.9.0.34、Torch-TensorRT 2.7.0、
  ModelOpt 0.27.0；复用但不修改主 `.venv` 的 PyTorch/Cosmos 依赖。
- 问题：TensorRT wheel 约 2.6 GB，安装耗时较长。
- 下一步：验证隔离环境导入和 GPU 执行，捕获真实单步 denoiser 输入。

## 2026-07-29 18:08 HKT — 真实 denoiser 输入捕获与 wrapper 验证

- 命令：`CUDA_VISIBLE_DEVICES=5 ... capture_denoiser_inputs.py`；
  `pytest -q tests/test_cosmos_denoiser_wrapper.py`。
- Git commit：`eaf393e5b5dbe8ab03a6ab08061c99aad6ba2e1e`（工作树含本实验新增文件）。
- 修改：`wrappers/cosmos_denoiser_wrapper.py`、
  `scripts/capture_denoiser_inputs.py`、wrapper 单测。
- 输出：`calibration/fixed_denoiser_inputs.pt`（1.4 MB）、
  `summaries/denoiser_wrapper_validation.json`、捕获日志。
- 状态：成功；测试 1/1 通过。
- 结果：真实输入为 latent `[1,16,9,28,28]`、timestep `[1,9]`、
  text `[1,512,1024]`、condition mask `[1,1,9,28,28]`、fps `[1]`、
  padding mask `[1,1,224,224]`。输出 cosine 约 1.0000001，
  max abs error 0，重复误差 0，无 NaN/Inf。
- 问题：
  - 当前共享 LIBERO-Plus task order 在运行期间已扩展为 2519 个 task，
    task 0 文本带 `table 1` 后缀；使用最长前缀匹配到原始预计算文本
    `turn on the stove and put the moka pot on it`，没有下载 T5-11B。
  - 首次尝试因模块重载导致 cache miss，误启动 T5-11B 下载；已终止并删除
    5.1 GB 无效 `.incomplete` 文件。
- 下一步：以该 fixture 尝试 BF16 TensorRT full compilation。

## 2026-07-29 18:09–18:30 HKT — torchao latency 与 kernel profiler

- 命令：`profile_torchao.py --phase latency --warmup 30 --iters 200`，
  六种模式各自独立进程；四个要求的模式再运行 `--phase torchprof`。
- GPU：5、6、7；每个正式计时进程固定一张卡。
- 修改：`scripts/profile_torchao.py`。
- 输出：
  - `raw/torchao_latency_*.csv`、`raw/torchao_action_*.npy`；
  - `summaries/torchao_latency_*.json`；
  - `profiles/torchao_{bf16,int8_wo,int8_w8a8,int4_wo}_trace.json`；
  - operator/kernel CSV。
- 状态：成功。
- 主要结果：BF16/W8A8/INT4-WO policy p50 为
  461.02/2512.16/1494.88 ms。W8A8 trace 有 115765 次 GPU 事件和
  50424 次 `_local_scalar_dense`；INT4 tinygemm 1400 次、累计
  1246.17 ms。
- 问题：W8A8 trace 达 411 MB，CSV 解析耗时明显；未覆盖 LIBERO env 的
  episode profile。
- 下一步：完成 BF16 TensorRT full-graph 支持审计。

## 2026-07-29 18:22–18:40 HKT — BF16 TensorRT graph 支持与实际构建

- 命令：
  - `analyze_trt_support.py` dry-run；
  - `build_bf16_trt.py --rewrite-te all --require-full
    --register-bf16-cast-converter`。
- 修改：
  - `scripts/build_bf16_trt.py`；
  - `scripts/analyze_trt_support.py`；
  - `wrappers/trt_bfloat16_cast_converter.py`；
  - `minimal_v4_dit.py` 中 padding mask `repeat` 改为等价 5-D `expand`。
- 第一次结果：
  - 原生 TE RMSNorm 无法 FakeTensor export；
  - 替换 113 RMSNorm、56 attention 并禁用 fused RoPE 后 export 成功；
  - 内置 converter 支持 6038/6492（93.01%），454 个 BF16 cast fallback，
    61 TRT + 61 PyTorch 分区。
- 修复结果：
  - 自定义 BF16 `_to_copy` converter 后 dry-run 6492/6492、单 TRT 分区；
  - 5-D mask rewrite 后 engine 在内存中 full compile 成功；
  - build 388.05 s，输出 cosine 0.9999347，finite，零 fallback。
- 失败：ExportedProgram 保存时报
  `serializing a string larger than 4 GiB requires pickle protocol 4 or higher`。
  因 engine 未落盘，未满足稳定 BF16 baseline。
- 输出：`profiles/bf16_trt_partition_report.json`、
  `profiles/bf16_trt_dryrun*.txt`、`logs/bf16_trt_build*.log`、
  `summaries/bf16_trt_final_status.json`。
- 下一步：改用 TorchScript zip 保存并重建；成功复载前不进入 FP8。

## 2026-07-29 18:31–18:41 HKT — 统一 eager 基准与 1–5 步 sweep

- 命令：
  - `benchmark_denoiser.py --backend bf16_eager --warmup 100 --iters 1000`；
  - `denoising_sweep.py --warmup 100 --iters 300`。
- GPU：单 denoiser GPU 5，sweep GPU 6。
- 输出：`raw/denoiser_microbenchmark.csv`、
  `raw/denoising_sweep_microbenchmark.csv`、对应 summary。
- 状态：成功。
- 结果：
  - 单 denoiser p50/p95/p99 = 60.58/63.94/68.11 ms；
  - 3-step policy p50 326.39 ms、cosine 0.9999992、L2 0.00613；
  - 5-step p50 448.42 ms；
  - 当前 scheduler 的 1-step 与 2-step 都只调用 1 次 denoiser，并输出相同。
- 下一步：只有可复载 BF16 TRT 存在时才做 TRT 统一基准。

## 2026-07-29 19:47 HKT — 继续执行环境变化与门禁结论

- 修改：实现 TorchScript engine 保存与 TRT full-policy benchmark adapter。
- 环境变化：
  - 继续执行容器不再暴露 `/dev/nvidia*`，`nvidia-smi` 无法连接驱动；
  - 新的写权限不允许继续向 `/data` 软链接目标落盘。
- 处理：
  - 将本实验已有 `engines/calibration/profiles/raw/logs` 完整复制回仓库本地
    目录；`/data` 原副本保留；
  - 已成功落盘的 BF16 Nsight report 保留；
  - W8A8 Nsight follow-up 未落盘，不伪装为成功。
- 门禁：
  - BF16 engine 未持久化/复载，稳定 baseline = false；
  - FP8 calibration/engine = 未开始；
  - 原始 LIBERO 和 LIBERO-Plus rollout = 未开始。
- 下一步：在重新获得 GPU 的执行环境中用 TorchScript 路径重建并复载；
  只有 BF16 1000/300 基准稳定后才开始 ModelOpt FP8。

## 2026-07-29 19:50 HKT — 汇总、图表和中文报告

- 命令：`summarize_results.py`、`make_figures.py`、Pandoc、WeasyPrint。
- 修改：汇总/绘图脚本、torchao 中文诊断、最终中文报告、FP8 gate 状态。
- 输出：
  - `summaries/backend_latency_summary.csv`、`speedup_gate.json`；
  - 14 组 300 DPI PNG + PDF；
  - `report/report_zh.md`、`.html`、`.pdf`；
  - `report/torchao_latency_diagnosis.md`；
  - `report/async_pipeline_design.md`。
- 状态：报告成功生成；总体实验结论为负门禁结果，未声称 TensorRT FP8
  speedup。
