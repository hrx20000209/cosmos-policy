# Cosmos Policy — 模型量化映射 (model_quantization_map)

模型: `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`
加载类: `CosmosPolicyVideo2WorldModel`,骨干网络 `model.net = MinimalV1LVGDiT`
基础权重: `nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt` (gated) + VAE `tokenizer/tokenizer.pth`
参数量: **net = 1.956 B**(占模型可训练参数的 100%;VAE/UMT5 文本编码器不在 `net` 内,推理时文本走预计算 T5 embedding `libero_t5_embeddings.pkl`,不加载 UMT5)。
数据来源: `report/model_structure.json`(由 `scripts/inspect_model.py` 从真实加载的模型导出)。

## 1. 顶层结构
| 模块 | 参数 | 类型 | 说明 |
|---|---|---|---|
| `net` | 1.9564 B | MinimalV1LVGDiT | 视频+动作扩散 Transformer(DiT),推理主体 |
| `conditioner` | 0 | Video2WorldConditioner | 条件封装(无独立权重) |
| `sampler` | 0 | CosmosPolicySampler | flow-matching 采样器(无权重) |

`net` 子模块:
| 子模块 | 参数 | 是否量化 | 理由 |
|---|---|---|---|
| `x_embedder` (PatchEmbed) | 0.15 M | 否 | 输入 patch 投影,数值敏感、占比极小 |
| `pos_embedder` (RoPE 3D) | 0 | 否 | 位置编码,无可量化 Linear |
| `t_embedder` (Sequential) | 16.8 M | 否 | timestep/sigma 嵌入,条件路径 |
| `blocks` (28× DiT block) | 1937.8 M | **部分量化** | 主体计算,见下 |
| `final_layer` | 1.7 M | 否 | 最终动作/输出投影,数值敏感 |
| `t_embedding_norm` (RMSNorm) | ~0 | 否 | 归一化 |

## 2. DiT block 内 Linear 清单(每类共 28 层)
| Linear 组 | 每层 shape | 28 层合计 | 量化? |
|---|---|---|---|
| `mlp.layer1` (FFN 上投影) | (8192, 2048) | 469.8 M | ✅ |
| `mlp.layer2` (FFN 下投影) | (2048, 8192) | 469.8 M | ✅ |
| `self_attn.q_proj` | (2048, 2048) | 117.4 M | ✅ |
| `self_attn.k_proj` | (2048, 2048) | 117.4 M | ✅ |
| `self_attn.v_proj` | (2048, 2048) | 117.4 M | ✅ |
| `self_attn.output_proj` | (2048, 2048) | 117.4 M | ✅ |
| `cross_attn.q_proj` | (2048, 2048) | 117.4 M | ✅ |
| `cross_attn.output_proj` | (2048, 2048) | 117.4 M | ✅ |
| `cross_attn.k_proj` | (2048, 1024) | 58.7 M | ✅ |
| `cross_attn.v_proj` | (2048, 1024) | 58.7 M | ✅ |
| `adaln_modulation_self_attn.{1,2}` | (256,2048)+(6144,256) | 58.7 M | ❌ 条件调制(adaLN) |
| `adaln_modulation_cross_attn.{1,2}` | 同上 | 58.7 M | ❌ 条件调制(adaLN) |
| `adaln_modulation_mlp.{1,2}` | 同上 | 58.7 M | ❌ 条件调制(adaLN) |
| `q_norm/k_norm` (RMSNorm) | — | — | ❌ 归一化 |

> 注:每个 block 被 `_checkpoint_wrapped_module`(激活重计算)包裹,量化 filter 用 `blocks.\d+.` 前缀匹配,不受影响。
> 注:checkpoint 中含 TransformerEngine 训练期 FP8 的 `*_extra_state` 键(q_norm/k_norm),推理加载时被安全跳过——说明该模型**训练时即用过 FP8**,与我们做 FP8 推理量化方向一致。

## 3. 量化覆盖率(实际选中)
- **量化 Linear: 280 个(10 组 × 28 block),= 1761.6 M 参数 = net 的 90.0%**
- 保持 bf16: 174 个 Linear + 所有 norm/embed = 194.8 M(10.0%)
- 覆盖的是全部注意力投影(self+cross 的 q/k/v/output)与 FFN——即计算量与显存带宽的绝对主体。

## 4. 各精度方案(Ada / sm_89)
| quant_mode | 后端 | 计算路径 | 硬件加速 | 用途 |
|---|---|---|---|---|
| `bf16` | 无 | bf16 tensor core | 基线 | 对照 |
| `fp8_backbone` | torchao Float8DynamicActivationFloat8Weight | `torch._scaled_mm`,Ada FP8 e4m3 | **是**(权重+激活) | 主 latency 结论 |
| `int8_backbone` | torchao Int8DynamicActivationInt8Weight (W8A8) | Ada INT8 tensor core | **是**(权重+激活) | 主 latency 结论 |
| `int4_weight_only` | torchao Int4WeightOnly (tinygemm) | int4 权重 unpack × bf16 激活 | **是**(权重带宽) | 主 latency 结论 |
| `fake_int4_backbone` | 自实现 quant→dequant 到 bf16 | bf16 计算 | **否** | 仅成功率敏感性,禁止用于加速结论 |

> NVFP4 不适用:Ada 无 FP4 tensor core(Blackwell 专属),不做。

## 5. 默认保持高精度的模块(第一轮部分量化)
LayerNorm/RMSNorm、softmax、flow-matching scheduler、timestep/sigma 计算、adaLN 条件调制、图像/proprio 预处理与归一化、patch 嵌入、最终 action 重建/输出投影、动作反归一化。这些要么数值敏感,要么参数占比极小(<10%),量化收益低、风险高。
