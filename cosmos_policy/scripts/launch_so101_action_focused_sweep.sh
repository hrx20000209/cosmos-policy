#!/usr/bin/env bash
set -uo pipefail

ROOT=${SO101_LEROBOT_ROOT:-/data/rxhuang/three_cubes_1}
REPO_ID=${SO101_LEROBOT_REPO_ID:-local/three_cubes_1}
T5_PATH=${SO101_T5_PATH:-$ROOT/so101_t5_embeddings.pkl}
STATS_PATH=${SO101_STATS_PATH:-$ROOT/so101_dataset_statistics.json}
OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/data/rxhuang/cosmos_action_focused_runs}
EVAL_ROOT=${SO101_ACTION_EVAL_ROOT:-$ROOT/action_focused_eval}
LOG_ROOT=${SO101_ACTION_LOG_ROOT:-$ROOT/action_focused_train_logs}
PYTHON_BIN=${PYTHON_BIN:-/home/rxhuang/Projects/cosmos-policy/.venv/bin/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-/home/rxhuang/Projects/cosmos-policy/.venv/bin/torchrun}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
BASE_PORT=${BASE_PORT:-12600}

export SO101_LEROBOT_ROOT=$ROOT
export SO101_LEROBOT_REPO_ID=$REPO_ID
export PYTHONPATH="/home/rxhuang/Projects/lerobot/src:${PYTHONPATH:-}"

mkdir -p "$LOG_ROOT" "$EVAL_ROOT"
MANIFEST="$EVAL_ROOT/manifest_$RUN_STAMP.tsv"
printf '# exp_name\tfinetune_mode\tloss_mode\taction_loss_multiplier\tnormalize_masked_loss\tmax_step\trun_dir\tgpu\ttrain_log\n' >"$MANIFEST"

ALL_SPECS=(
  "A_final_layer_action|final_layer_only|action_only|1|false|2000|4|${GPU_A:-0}"
  "B_full_dit_action|full_dit|action_only|1|false|1000|1|${GPU_B:-1}"
  "C_full_dit_action_x10|full_dit|action_only|10|false|1000|1|${GPU_C:-2}"
  "D_partial16_action_x10|partial_dit_last_n|action_only|10|false|2000|1|${GPU_D:-3}"
  "E_partial16_joint_x10|partial_dit_last_n|joint_action_future_state|10|false|2000|1|${GPU_E:-4}"
  "F_partial8_all|partial_dit_last_n|all|1|false|2000|4|${GPU_F:-5}"
)
SPECS=()
FILTER=",${SO101_EXPERIMENT_FILTER:-},"
for spec in "${ALL_SPECS[@]}"; do
  exp=${spec%%|*}
  if [[ $FILTER == ",," || $FILTER == *",$exp,"* ]]; then
    SPECS+=("$spec")
  fi
done
if [[ ${#SPECS[@]} -eq 0 ]]; then
  echo "SO101_EXPERIMENT_FILTER 没有匹配任何实验" >&2
  exit 1
fi

run_eval() {
  local exp=$1 run_dir=$2 gpu=$3 max_step=$4 step checkpoint out episode
  for ((step = 500; step <= max_step; step += 500)); do
    checkpoint=$(printf 'iter_%09d' "$step")
    if [[ ! -f "$run_dir/checkpoints/$checkpoint/model/.metadata" ]]; then
      echo "[$exp] 缺少 $checkpoint，跳过该 checkpoint（训练可能 OOM/提前退出）"
      continue
    fi
    out="$EVAL_ROOT/$exp/$checkpoint"
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" -m cosmos_policy.scripts.eval_so101_action_curves \
      --repo_id "$REPO_ID" --root "$ROOT" \
      --checkpoint "$run_dir/checkpoints/$checkpoint" \
      --t5_text_embeddings_path "$T5_PATH" --dataset_stats_path "$STATS_PATH" \
      --num_samples 16 --chunk_size 50 --num_denoising_steps 10 \
      --output_dir "$out/sample_level"
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" \
      -m cosmos_policy.scripts.eval_so101_teacher_forced_action_reconstruction \
      --repo_id "$REPO_ID" --root "$ROOT" \
      --checkpoint "$run_dir/checkpoints/$checkpoint" \
      --t5_text_embeddings_path "$T5_PATH" --dataset_stats_path "$STATS_PATH" \
      --num_samples 8 --chunk_size 50 --low_sigma 0.1 \
      --output_dir "$out/teacher_forced"
    for episode in 0 10 30 60 90; do
      CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" -m cosmos_policy.scripts.eval_so101_full_episode_action_curve \
        --repo_id "$REPO_ID" --root "$ROOT" \
        --checkpoint "$run_dir/checkpoints/$checkpoint" \
        --t5_text_embeddings_path "$T5_PATH" --dataset_stats_path "$STATS_PATH" \
        --episode "$episode" --query_stride 10 --chunk_size 50 --num_denoising_steps 10 \
        --output_dir "$out/full_episode/episode_$episode"
    done
  done
}

run_pipeline() {
  local index=$1 spec=$2 exp mode loss multiplier normalize max_step batch gpu blocks job run_dir train_log
  IFS='|' read -r exp mode loss multiplier normalize max_step batch gpu <<<"$spec"
  blocks=8
  [[ $exp == D_* || $exp == E_* ]] && blocks=16
  job="action_focused_${exp}_${RUN_STAMP}"
  run_dir="$OUTPUT_ROOT/cosmos_policy/so101_lerobot/$job"
  train_log="$LOG_ROOT/$job.train.log"
  echo "[$exp] GPU=$gpu, mode=$mode, loss=$loss, multiplier=$multiplier, max_step=$max_step"
  set +e
  CUDA_VISIBLE_DEVICES=$gpu IMAGINAIRE_OUTPUT_ROOT=$OUTPUT_ROOT \
    "$TORCHRUN_BIN" --nproc_per_node=1 --master_port=$((BASE_PORT + index)) \
    -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py -- \
    experiment=cosmos_predict2_2b_480p_so101_lerobot \
    job.name="$job" job.wandb_mode=disabled \
    model.config.finetune_mode="$mode" \
    model.config.train_last_n_dit_blocks="$blocks" \
    model.config.so101_loss_mode="$loss" \
    model.config.action_loss_multiplier="$multiplier" \
    model.config.normalize_so101_masked_loss="$normalize" \
    dataloader_train.batch_size="$batch" trainer.max_iter="$max_step" \
    checkpoint.save_iter=500 trainer.logging_iter=5 trainer.run_validation=false \
    >"$train_log" 2>&1
  status=$?
  set -e
  echo "[$exp] training exit=$status；开始评估现有 checkpoint"
  run_eval "$exp" "$run_dir" "$gpu" "$max_step"
}

declare -A USED_GPUS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r exp mode loss multiplier normalize max_step batch gpu <<<"$spec"
  if [[ -n ${USED_GPUS[$gpu]:-} ]]; then
    echo "GPU $gpu 同时分配给 ${USED_GPUS[$gpu]} 和 $exp；拒绝共享 GPU。" >&2
    exit 1
  fi
  USED_GPUS[$gpu]=$exp
  job="action_focused_${exp}_${RUN_STAMP}"
  run_dir="$OUTPUT_ROOT/cosmos_policy/so101_lerobot/$job"
  train_log="$LOG_ROOT/$job.train.log"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$exp" "$mode" "$loss" "$multiplier" "$normalize" "$max_step" "$run_dir" "$gpu" "$train_log" \
    >>"$MANIFEST"
done

pids=()
for index in "${!SPECS[@]}"; do
  exp=${SPECS[$index]%%|*}
  run_pipeline "$index" "${SPECS[$index]}" >"$LOG_ROOT/pipeline_${exp}_${RUN_STAMP}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

"$PYTHON_BIN" -m cosmos_policy.scripts.summarize_so101_action_focused_sweep \
  --manifest "$MANIFEST" --eval_root "$EVAL_ROOT" --output_dir "$EVAL_ROOT"
echo "action-focused sweep 完成；pipeline_failed=$failed；manifest=$MANIFEST"
