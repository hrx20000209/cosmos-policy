#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/cosmos-policy
OUT=$REPO/outputs/cosmos_three_cubes_full_finetune
RUN_ROOT=$OUT/overfit_runs
JOB=cosmos_predict2_2b_480p_so101_episode50_overfit_ga1
RUN=$RUN_ROOT/cosmos_policy/so101_lerobot/$JOB
DIAG=$OUT/diagnostics/episode50_overfit
CKPT_ROOT=$RUN/checkpoints

export PYTHONPATH=$REPO
export SO101_LEROBOT_ROOT=$OUT/data/hrx2000/Three_Cubes_1
export SO101_LEROBOT_REPO_ID=hrx2000/Three_Cubes_1
export SO101_LEROBOT_T5=$SO101_LEROBOT_ROOT/so101_t5_embeddings.pkl
export SO101_LEROBOT_STATS=$SO101_LEROBOT_ROOT/so101_dataset_statistics.json
export COSMOS_BASE_CHECKPOINT=$OUT/pretrained/model-480p-16fps.pt
export COSMOS_TOKENIZER_CHECKPOINT=$OUT/pretrained/cosmos_official/tokenizer/tokenizer.pth
export IMAGINAIRE_OUTPUT_ROOT=$RUN_ROOT
export COSMOS_ENABLE_WANDB=0

mkdir -p "$OUT/logs" "$DIAG"

for step in $(seq 100 100 1500); do
  step_pad=$(printf '%09d' "$step")
  checkpoint=$CKPT_ROOT/iter_$step_pad
  if [[ ! -d "$checkpoint" ]]; then
    torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
      --config=cosmos_policy/config/config.py -- \
      experiment=cosmos_predict2_2b_480p_so101_lerobot \
      job.name=$JOB optimizer=adamw \
      model.config.finetune_mode=all model.config.use_lora=false \
      dataloader_train.dataset.episodes=[50] \
      dataloader_train.dataset.use_image_aug=false \
      dataloader_train.dataset.use_stronger_image_aug=false \
      trainer.run_validation=false trainer.max_iter=$step \
      trainer.grad_accum_iter=1 optimizer.lr=1e-5 \
      scheduler.warm_up_steps=[50,0] checkpoint.save_iter=100 \
      2>&1 | tee -a "$OUT/logs/episode50_overfit_train.log"
  fi

  step_dir=$DIAG/step_$(printf '%04d' "$step")
  python cosmos_policy/scripts/eval_so101_full_episode_action_curve.py \
    --repo_id hrx2000/Three_Cubes_1 --root "$SO101_LEROBOT_ROOT" \
    --checkpoint "$checkpoint" --t5_text_embeddings_path "$SO101_LEROBOT_T5" \
    --dataset_stats_path "$SO101_LEROBOT_STATS" --episode 50 \
    --query_stride 30 --chunk_size 30 --num_denoising_steps 5 --seed 12345 \
    --output_dir "$step_dir" 2>&1 | tee "$OUT/logs/episode50_overfit_diag_step_${step}.log"
  python scripts/visualize_action_predictions.py \
    --predictions "$step_dir/episode_050_stride_30.npz" \
    --output-dir "$DIAG/standard" --step "$step" --chunk-index 8

  # Keep a compact time series independent of checkpoint retention.
  python - "$step" "$step_dir/episode_050_stride_30.json" "$DIAG/standard/step_${step}_metrics.json" "$DIAG/metrics.jsonl" <<'PY'
import json, sys
from pathlib import Path
step, native_path, extended_path, out_path = int(sys.argv[1]), *map(Path, sys.argv[2:])
native = json.loads(native_path.read_text())
extended = json.loads(extended_path.read_text())
row = {
    "step": step,
    "mae": extended["mae"],
    "rmse": extended["rmse"],
    "mean_pearson": extended["mean_pearson"],
    "per_dimension": extended["per_dimension"],
    "action_jump_excess_ratio": native["action_jump_excess_ratio"],
}
existing = []
if out_path.exists():
    existing = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
existing = [item for item in existing if item["step"] != step] + [row]
existing.sort(key=lambda item: item["step"])
out_path.write_text("".join(json.dumps(item) + "\n" for item in existing))
PY

  # Once the new checkpoint and diagnostics are complete, remove only the
  # previous regenerable overfit checkpoint to fit the 50 GB workspace.
  if (( step > 100 )); then
    previous=$(printf '%09d' $((step - 100)))
    previous_dir=$CKPT_ROOT/iter_$previous
    if [[ -d "$previous_dir" ]]; then
      find "$previous_dir" -type f -delete
      find "$previous_dir" -depth -type d -empty -delete
    fi
  fi
done
