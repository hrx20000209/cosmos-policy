"""Cosmos Policy 主 DiT 的 full/partial fine-tuning 参数控制。"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from cosmos_policy._src.imaginaire.utils import log


@dataclass(frozen=True)
class FinetuneReport:
    mode: str
    total_params: int
    trainable_params: int
    trainable_module_names: tuple[str, ...]
    optimizer_module: nn.Module


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def apply_finetune_mode(
    model: nn.Module,
    finetune_mode: str,
    train_last_n_dit_blocks: int,
    action_head_fallback: str | None = None,
) -> FinetuneReport:
    """冻结全部参数后，按模式只解冻所需 DiT 模块。"""
    valid_modes = {
        "full_dit",
        "partial_dit_last_n",
        "final_layer_only",
        "all",
        "action_only_head_if_exists",
    }
    if finetune_mode not in valid_modes:
        raise ValueError(f"未知 finetune_mode={finetune_mode}，可选值为 {sorted(valid_modes)}")
    if not hasattr(model, "net") or not isinstance(model.net, nn.Module):
        raise RuntimeError("找不到 Cosmos Policy 主 DiT：模型没有 nn.Module 类型的 model.net")

    _set_module_trainable(model, False)
    trainable_modules: list[tuple[str, nn.Module]] = []
    optimizer_module: nn.Module = model.net

    if finetune_mode == "action_only_head_if_exists":
        explicit_head_names = ("action_head", "action_decoder", "policy_head")
        explicit_heads = [
            (name, module)
            for name, module in model.named_modules()
            if name.rsplit(".", 1)[-1] in explicit_head_names
        ]
        if explicit_heads:
            trainable_modules.extend(explicit_heads)
        else:
            message = (
                "Cosmos Policy 没有独立 action head；action 是通过共享 DiT 的 action latent frame 生成。"
            )
            log.warning(message)
            if action_head_fallback != "final_layer_only":
                raise RuntimeError(
                    message + " 如需近似 action-head baseline，请显式设置 "
                    "model.config.action_head_fallback=final_layer_only。"
                )
            log.warning("已按显式配置回退到共享 final_layer_only；不会解冻 DiT blocks。")
            finetune_mode = "final_layer_only"

    if trainable_modules:
        pass
    elif finetune_mode == "full_dit":
        trainable_modules.append(("net (完整 DiT)", model.net))
    elif finetune_mode == "final_layer_only":
        final_layer = getattr(model.net, "final_layer", None)
        if not isinstance(final_layer, nn.Module):
            raise RuntimeError("final_layer_only 找不到 model.net.final_layer")
        trainable_modules.append(("net.final_layer", final_layer))
    elif finetune_mode == "partial_dit_last_n":
        blocks = getattr(model.net, "blocks", None)
        if blocks is None or not hasattr(blocks, "__len__"):
            children = [name for name, _ in model.net.named_children()]
            raise RuntimeError(f"model.net 中找不到 transformer blocks；顶层模块为 {children}")
        num_blocks = len(blocks)
        if not 1 <= train_last_n_dit_blocks <= num_blocks:
            raise ValueError(
                f"train_last_n_dit_blocks 必须在 [1, {num_blocks}]，实际为 {train_last_n_dit_blocks}"
            )
        first_trainable = num_blocks - train_last_n_dit_blocks
        for index in range(first_trainable, num_blocks):
            trainable_modules.append((f"net.blocks.{index}", blocks[index]))
        for name in ("final_layer", "t_embedding_norm"):
            module = getattr(model.net, name, None)
            if isinstance(module, nn.Module):
                trainable_modules.append((f"net.{name}", module))
        if not any(name == "net.final_layer" for name, _ in trainable_modules):
            raise RuntimeError("partial DiT 模式找不到必要的 model.net.final_layer")
    elif finetune_mode == "all":
        # tokenizer/VAE 始终冻结；其余模型组件都交给 optimizer。
        trainable_modules.append(("model（不含 tokenizer/VAE）", model))
        optimizer_module = model

    for _, module in trainable_modules:
        _set_module_trainable(module, True)
    tokenizer = getattr(model, "tokenizer", None)
    if isinstance(tokenizer, nn.Module):
        _set_module_trainable(tokenizer, False)
    text_encoder = getattr(model, "text_encoder", None)
    if isinstance(text_encoder, nn.Module):
        _set_module_trainable(text_encoder, False)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_params == 0:
        raise RuntimeError(f"finetune_mode={finetune_mode} 没有解冻任何参数")
    names = tuple(name for name, _ in trainable_modules)
    log.critical(
        f"微调模式={finetune_mode}；总参数={total_params:,}；可训练参数={trainable_params:,}；"
        f"比例={100.0 * trainable_params / total_params:.4f}%"
    )
    log.critical("解冻模块: " + ", ".join(names))
    return FinetuneReport(finetune_mode, total_params, trainable_params, names, optimizer_module)
