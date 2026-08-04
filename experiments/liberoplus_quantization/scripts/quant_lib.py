"""Partial quantization of the Cosmos Policy DiT backbone (model.net) via torchao.

Design (Ada / sm_89):
  bf16              baseline, no quantization (control)
  fp8_backbone      Float8 dynamic act + Float8 weight on backbone Linears.
                    Uses torch._scaled_mm -> Ada FP8 tensor cores. HW-accel: TRUE.
  int8_backbone     Int8 dynamic act + Int8 weight (W8A8) on backbone Linears.
                    Ada INT8 tensor cores. HW-accel: TRUE.
  int4_weight_only  Int4 weight-only (tinygemm _weight_int4pack_mm), bf16 act.
                    Weight-bandwidth win, real int4 unpack kernel. HW-accel: TRUE (weight).
  fake_int4_backbone  Simulated int4 weight (quant->dequant to bf16), bf16 compute.
                    For SUCCESS-RATE sensitivity ONLY. HW-accel: FALSE.

Only large Linear layers inside the transformer blocks are quantized. Norms,
softmax, timestep/sigma embeddings, conditioning, patch embed, and the final
action/output projection stay bf16 (numerically sensitive, tiny param share).
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import torch
import torch.nn as nn

# ---- name patterns ----------------------------------------------------------
# A Linear is a quantization CANDIDATE only if its qualified name matches INCLUDE
# and none of EXCLUDE. Tuned to Cosmos Predict2 DiT; verify against inspect_model.
INCLUDE = re.compile(r"(blocks?|layers?)\.\d+\.")           # inside transformer blocks
EXCLUDE = re.compile(
    r"(embed|embedder|patch|pos_|time|t_emb|sigma|cond|logvar|adaln|modulation|"
    r"norm|ln_|layernorm|rmsnorm|final|head|out_proj_action|proj_out|unpatch|x_embedder|t_embedder|affine)",
    re.IGNORECASE,
)
MIN_FEATURES = 256  # skip tiny Linears (adaLN modulation etc.)


SCOPE_PATTERNS = {
    "backbone": re.compile(r"(blocks?|layers?)\.\d+\."),
    "attention": re.compile(r"\.(self_attn|cross_attn)\."),
    "mlp": re.compile(r"\.mlp\."),
    "vision_input_proxy": re.compile(r"^x_embedder\.proj\.1$"),
    # Cosmos Policy has no independent action head. This is the shared output
    # projection and is exposed only as an explicitly named sensitivity proxy.
    "shared_output_proxy": re.compile(r"^final_layer\..*\.linear$"),
    "all_linear": re.compile(r"."),
}


def is_candidate(
    name: str,
    mod: nn.Module,
    scope: str = "backbone",
    include_patterns: Sequence[str] = (),
    exclude_patterns: Sequence[str] = (),
) -> bool:
    if not isinstance(mod, nn.Linear):
        return False
    if scope not in SCOPE_PATTERNS:
        raise ValueError(f"unknown quantization scope: {scope}")
    if not SCOPE_PATTERNS[scope].search(name):
        return False
    if include_patterns and not any(re.search(pattern, name) for pattern in include_patterns):
        return False
    if any(re.search(pattern, name) for pattern in exclude_patterns):
        return False
    if scope not in ("shared_output_proxy", "vision_input_proxy") and EXCLUDE.search(name):
        return False
    return max(mod.in_features, mod.out_features) >= MIN_FEATURES


# ---- fake int4 (sensitivity only) -------------------------------------------
class FakeInt4Linear(nn.Module):
    """Weight quantized to a symmetric per-group int4 grid then dequantized to
    bf16; compute stays bf16. Emulates int4 *error* with NO hardware speedup."""

    def __init__(self, lin: nn.Linear, group_size: int = 128):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        w = lin.weight.data.float()
        O, I = w.shape
        gs = group_size if I % group_size == 0 else I
        wg = w.reshape(O, I // gs, gs)
        amax = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / 7.0
        q = torch.clamp(torch.round(wg / scale), -8, 7)
        wdq = (q * scale).reshape(O, I).to(lin.weight.dtype)
        self.register_buffer("weight", wdq)
        self.bias = nn.Parameter(lin.bias.data.clone()) if lin.bias is not None else None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


def _apply_fake_int4(
    net: nn.Module,
    group_size: int = 128,
    scope: str = "backbone",
    include_patterns: Sequence[str] = (),
    exclude_patterns: Sequence[str] = (),
):
    n = 0
    for parent_name, parent in net.named_modules():
        for cname, child in list(parent.named_children()):
            full = f"{parent_name}.{cname}" if parent_name else cname
            if is_candidate(full, child, scope, include_patterns, exclude_patterns):
                setattr(parent, cname, FakeInt4Linear(child, group_size).to(child.weight.device))
                n += 1
    return n


def configure_model_precision(model: nn.Module, precision: str) -> None:
    """Keep wrapper tensor creation and net weights on the same native dtype."""
    dtype_by_name = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if precision not in dtype_by_name:
        return
    dtype = dtype_by_name[precision]
    model.precision = dtype
    if hasattr(model, "tensor_kwargs"):
        model.tensor_kwargs["dtype"] = dtype
    if hasattr(model, "config") and hasattr(model.config, "precision"):
        model.config.precision = "float16" if dtype == torch.float16 else "bfloat16"
    model.net.to(dtype=dtype)


# ---- main entry -------------------------------------------------------------
def apply_quantization(
    net: nn.Module,
    mode: str,
    group_size: int = 128,
    scope: str = "backbone",
    include_patterns: Sequence[str] = (),
    exclude_patterns: Sequence[str] = (),
) -> dict:
    """Quantize `net` in place. Returns an audit dict."""
    candidates = [
        (name, module)
        for name, module in net.named_modules()
        if is_candidate(name, module, scope, include_patterns, exclude_patterns)
    ]
    cand_params = sum(m.weight.numel() for _, m in candidates)
    total_params = sum(p.numel() for p in net.parameters())

    backend = "none"
    hw = False
    real_quantization = False
    if mode == "bf16":
        backend = "none (bf16 baseline)"
    elif mode == "fp16":
        net.to(dtype=torch.float16)
        backend = "native torch.float16 (no quantization)"
    elif mode in (
        "fp8_backbone",
        "int8_backbone",
        "int8_weight_only",
        "int4_weight_only",
    ):
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Int4WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
        if mode == "fp8_backbone":
            cfg = Float8DynamicActivationFloat8WeightConfig()
            backend = "torchao float8 (torch._scaled_mm, Ada FP8 e4m3)"
            hw = True
        elif mode == "int8_backbone":
            cfg = Int8DynamicActivationInt8WeightConfig()
            backend = "torchao int8 W8A8 (Ada INT8 tensor cores)"
            hw = True
        elif mode == "int8_weight_only":
            cfg = Int8WeightOnlyConfig()
            backend = "torchao int8 weight-only"
            hw = True
        else:
            cfg = Int4WeightOnlyConfig(group_size=group_size)
            backend = f"torchao int4 weight-only tinygemm (group={group_size})"
            hw = True
        real_quantization = True
        cand_names = {n for n, _ in candidates}
        quantize_(net, cfg, filter_fn=lambda m, name: name in cand_names)
    elif mode == "fake_int4_backbone":
        _apply_fake_int4(net, group_size, scope, include_patterns, exclude_patterns)
        backend = "FAKE int4 (quant->dequant, bf16 compute) — sensitivity only"
        hw = False
    else:
        raise ValueError(f"unknown quant mode: {mode}")

    return {
        "quant_mode": mode,
        "quantization_scope": scope,
        "include_patterns": list(include_patterns),
        "exclude_patterns": list(exclude_patterns),
        "quant_backend": backend,
        "hardware_accelerated": hw,
        "real_quantization": real_quantization,
        "n_candidate_linears": len(candidates),
        "candidate_module_names": [name for name, _ in candidates],
        "candidate_weight_params": cand_params,
        "net_total_params": total_params,
        "coverage_frac": cand_params / max(total_params, 1),
        "group_size": group_size if mode in ("int4_weight_only", "fake_int4_backbone") else None,
    }


def verify_quantized(net: nn.Module) -> dict:
    """Post-quantization audit: count modules whose weight is no longer a plain
    bf16/fp32 tensor (i.e. became an AQT / fake-int4)."""
    from collections import Counter
    kinds = Counter()
    quantized = 0
    for name, mod in net.named_modules():
        w = getattr(mod, "weight", None)
        if w is None:
            continue
        tn = type(w).__name__
        if isinstance(mod, FakeInt4Linear):
            kinds["FakeInt4Linear"] += 1
            quantized += 1
        elif "AffineQuantizedTensor" in tn or "LinearActivationQuantizedTensor" in tn or tn not in ("Parameter", "Tensor"):
            kinds[tn] += 1
            quantized += 1
    return {"quantized_modules": quantized, "weight_tensor_kinds": dict(kinds)}
