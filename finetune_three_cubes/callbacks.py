"""three_cubes 全参微调的监控 callbacks（阶段 2.3 / 2.4 / 2.5）。

- DeterministicValLossCallback: 固定 val 样本 + 固定 sigma 网格 + 固定噪声种子，
  每次 validation 后计算**低方差、可读**的确定性验证 loss（video / action 分开），
  写入 metrics.jsonl(split="det_val")、维护 best 指针。
- CheckpointRetentionCallback: 每次存 checkpoint 后只保留最近 N 个 + best，
  删除其余，避免 2B FSDP checkpoint 撑爆磁盘。

两个 callback 都用 try/except 包住核心逻辑：监控代码永远不允许拖垮训练本身。
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from collections import defaultdict

import torch
import torch.distributed as dist

from cosmos_policy._src.imaginaire.model import ImaginaireModel
from cosmos_policy._src.imaginaire.utils import distributed, log
from cosmos_policy._src.imaginaire.utils.callback import Callback


def _to_cuda(batch: dict) -> dict:
    return {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}


def _cpu_clone(batch: dict) -> dict:
    return {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def _append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


class DeterministicValLossCallback(Callback):
    """固定输入 + 固定 sigma + 固定噪声的确定性验证 loss（不依赖 val 数据路径）。

    扩散训练 loss 因为每步随机采 timestep 方差极大，几乎看不出收敛。这里对一小批
    **固定样本**（缓存训练最初见到的前 N 个 batch）、一组固定 sigma、固定随机种子
    生成的噪声，重复用完全相同的输入计算 per-component EDM MSE（未加权），得到可读的
    收敛曲线。

    注意：Cosmos 的内置 validation_step 对 SO101 batch 会 KeyError('guidance') 直接崩，
    所以这里**不走 on_validation_* / val dataloader**，而是缓存训练 batch，在
    on_training_step_end 里按 det_val_every 的节奏、直接用训练 loss API 计算。
    用户已同意不划分验证集，固定探针取自训练数据即可（仍是可读的收敛指标）。
    """

    def __init__(
        self,
        capture_num_batches: int = 8,
        det_val_every: int = 500,
        sigma_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
        noise_seed: int = 1234,
        metrics_filename: str = "det_val_metrics.jsonl",
        best_filename: str = "best_det_val.json",
    ) -> None:
        super().__init__()
        self.capture_num_batches = int(capture_num_batches)
        self.det_val_every = int(det_val_every)
        self.sigma_grid = tuple(float(s) for s in sigma_grid)
        self.noise_seed = int(noise_seed)
        self.metrics_filename = metrics_filename
        self.best_filename = best_filename
        self._fixed_batches: list[dict] = []
        self._best_value = float("inf")
        self._best_iteration = -1

    # ---- 缓存固定探针 batch（训练最初见到的前 N 个） ----
    def on_training_step_start(self, model, data, iteration: int = 0) -> None:
        if len(self._fixed_batches) < self.capture_num_batches:
            self._fixed_batches.append(_cpu_clone(data))

    # ---- 按节奏计算确定性 loss ----
    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        if iteration <= 0 or self.det_val_every <= 0 or iteration % self.det_val_every != 0:
            return
        self._run(model, iteration)

    def _run(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not self._fixed_batches:
            return
        try:
            metrics = self._compute(model)
        except Exception as exc:  # 监控失败不影响训练
            log.warning(f"[det_val] 确定性验证 loss 计算失败 (iter {iteration}): {exc!r}")
            return

        if not distributed.is_rank0():
            return

        record = {"split": "det_val", "iteration": iteration, **metrics}
        path = os.path.join(self.config.job.path_local, self.metrics_filename)
        _append_jsonl(path, record)
        # 也进主 metrics.jsonl，方便统一画图
        _append_jsonl(os.path.join(self.config.job.path_local, "metrics.jsonl"), record)

        primary = metrics.get("det_val/action_mse", float("inf"))
        if primary < self._best_value:
            self._best_value = primary
            self._best_iteration = iteration
            with open(os.path.join(self.config.job.path_local, self.best_filename), "w") as fh:
                json.dump({"iteration": iteration, "det_val/action_mse": primary, **metrics}, fh, indent=2)
        log.info(
            f"[det_val] iter {iteration}: action_mse={primary:.6f} "
            f"video_mse={metrics.get('det_val/future_image_mse', float('nan')):.6f} "
            f"(best action_mse={self._best_value:.6f} @ iter {self._best_iteration})"
        )
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(record, step=iteration)
        except Exception:
            pass

    @torch.no_grad()
    def _compute(self, model: ImaginaireModel) -> dict:
        was_training = model.training
        model.eval()
        device = "cuda"
        sums: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros((), device=device))
        count = 0
        component_keys = {
            "det_val/action_mse": "demo_sample_action_mse_loss",
            "det_val/action_l1": "demo_sample_action_l1_loss",
            "det_val/future_image_mse": "demo_sample_future_image_mse_loss",
            "det_val/future_image_l1": "demo_sample_future_image_l1_loss",
            "det_val/future_wrist_image_mse": "demo_sample_future_wrist_image_mse_loss",
            "det_val/future_proprio_mse": "demo_sample_future_proprio_mse_loss",
            "det_val/edm_loss": "edm_loss",
        }
        for bi, cpu_batch in enumerate(self._fixed_batches):
            batch = _to_cuda(cpu_batch)
            if model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online:
                emb = model.text_encoder.compute_text_embeddings_online(batch, model.input_caption_key)
                batch["t5_text_embeddings"] = emb
                batch["t5_text_mask"] = torch.ones(emb.shape[0], emb.shape[1], device=device)
            _, x0, condition = model.get_data_and_condition(batch)
            for si, sigma_val in enumerate(self.sigma_grid):
                sigma_B_T = torch.full(
                    (x0.shape[0], x0.shape[2]), float(sigma_val), device=device, dtype=torch.float32
                )
                gen = torch.Generator(device=device).manual_seed(self.noise_seed + bi * 131 + si)
                epsilon = torch.randn(x0.size(), generator=gen, device=device, dtype=x0.dtype)
                x0s, cond_s, eps_s, sig_s = model.broadcast_split_for_model_parallelsim(
                    x0.clone(), condition, epsilon, sigma_B_T
                )
                output_batch, _, _, _ = model.compute_loss_with_epsilon_and_sigma(
                    x0s,
                    cond_s,
                    eps_s,
                    sig_s,
                    action_chunk=batch["actions"],
                    action_indices=batch["action_latent_idx"],
                    proprio=batch["proprio"],
                    current_proprio_indices=batch["current_proprio_latent_idx"],
                    future_proprio=batch["future_proprio"],
                    future_proprio_indices=batch["future_proprio_latent_idx"],
                    future_wrist_image_indices=batch["future_wrist_image_latent_idx"],
                    future_wrist_image2_indices=batch.get("future_wrist_image2_latent_idx"),
                    future_image_indices=batch["future_image_latent_idx"],
                    future_image2_indices=batch.get("future_image2_latent_idx"),
                    rollout_data_mask=batch["rollout_data_mask"],
                    world_model_sample_mask=batch["world_model_sample_mask"],
                    value_function_sample_mask=batch["value_function_sample_mask"],
                    value_function_return=batch["value_function_return"],
                    value_indices=batch["value_latent_idx"],
                )
                for out_key, batch_key in component_keys.items():
                    value = output_batch[batch_key]
                    if torch.is_tensor(value) and not torch.isnan(value):
                        sums[out_key] = sums[out_key] + value.detach().float()
                count += 1

        if was_training:
            model.train()
        if count == 0:
            return {}
        metrics = {}
        for key in component_keys:
            mean = sums[key] / count
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(mean, op=dist.ReduceOp.AVG)
            metrics[key] = float(mean.item())
        metrics["det_val/num_samples"] = len(self._fixed_batches)
        metrics["det_val/num_sigmas"] = len(self.sigma_grid)
        return metrics


class CheckpointRetentionCallback(Callback):
    """只保留最近 N 个 checkpoint + best（best 由 DeterministicValLossCallback 记录）。"""

    def __init__(self, keep_last_n: int = 5, best_filename: str = "best_det_val.json") -> None:
        super().__init__()
        self.keep_last_n = int(keep_last_n)
        self.best_filename = best_filename

    def on_save_checkpoint_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not distributed.is_rank0():
            return
        try:
            self._prune()
        except Exception as exc:
            log.warning(f"[ckpt_retention] 清理 checkpoint 失败: {exc!r}")

    def _iteration_of(self, path: str) -> int:
        base = os.path.basename(path.rstrip("/"))
        digits = "".join(c for c in base.replace("iter_", "") if c.isdigit())
        return int(digits) if digits else -1

    def _prune(self) -> None:
        ckpt_dir = os.path.join(self.config.job.path_local, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            return
        entries = sorted(
            {p for p in glob.glob(os.path.join(ckpt_dir, "iter_*"))},
            key=self._iteration_of,
        )
        # 去掉非法项
        entries = [p for p in entries if self._iteration_of(p) >= 0]
        best_iter = -1
        best_path = os.path.join(self.config.job.path_local, self.best_filename)
        if os.path.isfile(best_path):
            try:
                best_iter = int(json.load(open(best_path)).get("iteration", -1))
            except Exception:
                best_iter = -1
        # 按 iteration 分组（.pt 文件 + trained_data_record.json 同 iteration）
        keep = set(entries[-self.keep_last_n :]) if self.keep_last_n > 0 else set(entries)
        for path in entries:
            if path in keep:
                continue
            if self._iteration_of(path) == best_iter:
                continue  # 保护 best
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass
            log.info(f"[ckpt_retention] 删除旧 checkpoint: {os.path.basename(path)}")
