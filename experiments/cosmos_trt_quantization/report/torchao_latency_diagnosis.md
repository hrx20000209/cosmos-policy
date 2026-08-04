# torchao 延迟退化诊断

## 结论

本轮使用同一条固定 LIBERO observation、同一 T5 embedding、batch=1、
chunk=16、5 个 action denoising steps。每个模式独立进程运行 30 次 warm-up
和 200 次正式测量；torch.profiler 在相同输入上另抓取 1 次完整 5-step
policy。

torchao 在这台 RTX 4090 D（Ada sm_89）上没有获得端到端加速：

| 模式 | policy p50 / p95 / p99 (ms) | 单 denoiser p50 (ms) | 相对 BF16 |
|---|---:|---:|---:|
| BF16 eager | 461.02 / 540.40 / 568.13 | 60.58（1000 次独立基准） | 1.00× |
| INT8 weight-only | 552.32 / 633.69 / 660.80 | 79.56 | 0.835× |
| FP8 torchao | 1021.65 / 1093.72 / 1130.93 | 172.09 | 0.451× |
| INT4 weight-only | 1494.88 / 1566.30 / 1589.56 | 268.46 | 0.308× |
| INT8 dynamic W8A8 | 2512.16 / 2581.07 / 2662.31 | 469.68 | 0.184× |
| fake INT4 | 451.19 / 511.77 / 555.60 | 59.83 | 1.022× |

fake INT4 只在加载时做 quantize→dequantize，计算仍为 BF16。它与 BF16
延迟接近，证明量化后的数值本身不是慢因；真实低精度 kernel 路径才是慢因。

## BF16 基线已经使用的高效路径

BF16 trace 一次 5-step 调用记录 9370 次 GPU kernel/传输事件，GPU kernel
累计 396.60 ms。其中：

- BF16 Tensor Core GEMM 约 273.45 ms（68.95%）；
- FlashAttention 约 35.86 ms（9.04%）；
- cast/copy 约 42.25 ms（10.65%）；
- norm 约 9.55 ms，其中可直接看到
  `transformer_engine::normalization::rmsnorm_fwd_general_kernel`；
- 还可看到 `transformer_engine::fused_rope_forward_kernel`。

主干矩阵 shape 包括：

- 840 次 `[1764,2048] × [2048,2048]`；
- 140 次 `[1764,2048] × [2048,8192]`；
- 140 次 `[1764,8192] × [8192,2048]`；
- attention 使用 `[1,1764,16,128]` 的 FlashAttention。

因此 BF16 不是低效的纯 eager 小算子基线；原模型已经同时受益于
Tensor Core GEMM、FlashAttention、TransformerEngine RMSNorm 和 fused RoPE。

## INT8 weight-only 为什么慢

INT8-WO trace 的 GEMM kernel 名仍是
`ampere_bf16_s1688gemm_bf16_*`，没有进入 INT8 GEMM。权重在每个 Linear
调用前被还原/转换，随后仍做 BF16 GEMM：

- 5 步共有 280 × 5 = 1400 次量化 Linear；
- 新增 1401 次 BF16 direct-copy/cast kernel，累计 33.22 ms；
- 额外 elementwise/copy 后，GPU 事件数由 9370 增到 12170；
- GPU kernel 累计由 396.60 ms 增到 440.19 ms；
- policy p50 增加 91.30 ms。

所以 INT8-WO 的主要问题是“存储为 INT8、运行时恢复成 BF16”，而不是使用
高效 INT8 GEMM。它能节省权重显存，但不能提供 batch=1 latency 加速。

## dynamic W8A8 为什么最慢

W8A8 确实进入 `ampere_igemm_int8_128x128_ldg4_nn`，1400 次 INT8 GEMM
累计约 219.20 ms；但动态 activation scale 和 Python/ATen 标量路径远大于
这部分收益：

- 每个 5-step policy 有 1400 次动态 Linear；
- 每次都执行 min/max reduction、scale、clamp、round、BF16→INT8 cast；
- activation quantize 相关 kernel 9866 次、累计约 161.06 ms；
- cast/copy 类事件 83679 次、累计约 175.46 ms；
- 其中 50423 次 D2H pinned copy、16801 次 D2H pageable copy；
- `aten::_local_scalar_dense` 出现 50424 次，说明大量 device scalar 被同步到
  CPU；
- 总 GPU 事件达到 115765 次，是 BF16 的 12.35 倍；
- profiler 中 GPU kernel 只累计约 755.04 ms，而被 profile 的 policy
  record_function 为约 4347 ms，差额主要是 CPU launch、标量同步和 profiler
  放大后的调度开销。

因此，W8A8 的主要慢因就是每次 Linear 动态求 activation scale，以及由标量
提取造成的 CPU/GPU 同步。INT8 GEMM 本身并不慢，但无法抵消这些开销。

## INT4 weight-only 为什么慢

INT4-WO 的决定性 kernel 是：

`at::native::tinygemm_m16n8k16_chunk_kernel<..., BLayout_TC_int4<8,128>, ...>`

它恰好执行 1400 次，累计 1246.17 ms，占该 trace GPU 时间的 86.68%。
对应 operator 与 shape：

- 840 次 `[1764,2048]` 主投影；
- 140 次 `[1764,2048] → 8192` MLP 上投影；
- 140 次 `[1764,8192] → 2048` MLP 下投影；
- 280 次 `[512,1024]` 文本 cross-attention K/V 投影。

这里 INT4 unpack/decode 与 GEMM 融合在 tinygemm kernel 内，不能从 kernel
时间中独立拆出“纯 unpack”。证据支持的准确表述是：这些 batch=1、M=1764、
大 K/N 的 DiT shape 没有从 tinygemm 获益；融合了 INT4 权重解码的 tinygemm
本身耗时 1.246 s，是 INT4 退化的主要原因，而不是 attention、norm 或
scheduler。

## TransformerEngine、dtype cast 与融合破坏

- BF16 路径保留 TE RMSNorm、fused RoPE 和 BF16 Tensor Core GEMM。
- INT8-WO/INT4-WO 仍保留 TE norm 与 RoPE，但替换 280 个 Linear 后破坏了
  原来的连续 BF16 GEMM 路径。
- W8A8 在线性层边界反复 BF16→INT8→BF16，并伴随 scale scalar 的 D2H；
  attention、norm、softmax 仍保持 BF16，因此每层都有 dtype 边界。
- 1400 次量化 Linear 是累计效应的核心。单层看似很小的 scale/cast/launch
  成本乘以 280 层、再乘以 5 步后成为主导项。

## torch.compile 为什么没有改善

上一阶段真实 compiled microbenchmark 的 denoise p50 为：

- BF16 424.9→429.9 ms；
- FP8 998.4→998.0 ms；
- W8A8 2458.9→2471.5 ms；
- INT4-WO 1455.3→1445.0 ms。

Dynamo 明确报告无法 trace
`transformer_engine_torch.PyCapsule.rmsnorm_fwd`，并对
`torch.utils.checkpoint(context_fn=...)` 给出警告。更重要的是，即使外层图
可编译，compile 也不能消除 torchao 内部每次调用的 dynamic scale、
`_local_scalar_dense` 同步或 tinygemm kernel 本身的 shape 效率问题。因此
四种模式都没有实质改善。

## 证据文件

- `profiles/torchao_{bf16,int8_wo,int8_w8a8,int4_wo}_trace.json`
- `profiles/torchao_*_operator_summary.csv`
- `profiles/torchao_kernel_summary.csv`
- `summaries/kernel_category_summary.csv`
- `profiles/torchao_graph_breaks.txt`
- `profiles/nsys_bf16.nsys-rep`
- `raw/torchao_latency_*.csv`

W8A8 的第二份 Nsight 报告在运行环境写权限切换后未能落盘；上述 W8A8
结论来自已完成的 411 MB Kineto trace 和 operator/kernel CSV，不将失败的
Nsight 抓取伪装为成功。
