#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "用法: $0 /path/to/sweep_manifest.tsv" >&2
  echo "manifest 每行: EXPERIMENT_NAME<TAB>RUN_DIRECTORY<TAB>GPU_ID" >&2
  exit 2
fi

MANIFEST=$1
ROOT=${SO101_LEROBOT_ROOT:-/data/rxhuang/three_cubes_1}
REPO_ID=${SO101_LEROBOT_REPO_ID:-local/three_cubes_1}
T5_PATH=${SO101_T5_PATH:-$ROOT/so101_t5_embeddings.pkl}
STATS_PATH=${SO101_STATS_PATH:-$ROOT/so101_dataset_statistics.json}
SAMPLE_ROOT=${SO101_SAMPLE_EVAL_ROOT:-$ROOT/offline_eval_sweep}
EPISODE_ROOT=${SO101_EPISODE_EVAL_ROOT:-$ROOT/offline_eval_sweep_full_episode}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
EXPECTED_EXPERIMENTS=${EXPECTED_EXPERIMENTS:-5}
CHECKPOINTS=(iter_000000500 iter_000001000 iter_000001500 iter_000002000)
EPISODES=(0 10 30 60 90)

export PYTHONPATH="/home/rxhuang/Projects/lerobot/src:${PYTHONPATH:-}"

for protected in "$ROOT" "$T5_PATH" "$STATS_PATH"; do
  if [[ ! -e "$protected" ]]; then
    echo "保护路径不存在，拒绝启动评估: $protected" >&2
    exit 1
  fi
done

mapfile -t ROWS < <(awk -F '\t' 'NF && $1 !~ /^#/ {print $0}' "$MANIFEST")
if [[ ${#ROWS[@]} -ne $EXPECTED_EXPERIMENTS ]]; then
  echo "manifest 实验数=${#ROWS[@]}，期望=$EXPECTED_EXPERIMENTS；拒绝静默漏评。" >&2
  exit 1
fi

validate_row() {
  local row=$1 name run_dir gpu checkpoint
  IFS=$'\t' read -r name run_dir gpu <<< "$row"
  if [[ -z "$name" || -z "$run_dir" || -z "$gpu" ]]; then
    echo "manifest 行格式错误: $row" >&2
    return 1
  fi
  for checkpoint in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "$run_dir/checkpoints/$checkpoint/model/.metadata" ]]; then
      echo "缺少必须 checkpoint: $run_dir/checkpoints/$checkpoint" >&2
      return 1
    fi
  done
}

for row in "${ROWS[@]}"; do
  validate_row "$row"
done

run_experiment() {
  local row=$1 name run_dir gpu checkpoint checkpoint_path sample_out episode episode_out
  IFS=$'\t' read -r name run_dir gpu <<< "$row"
  echo "[$name] GPU=$gpu run=$run_dir"
  for checkpoint in "${CHECKPOINTS[@]}"; do
    checkpoint_path="$run_dir/checkpoints/$checkpoint"
    sample_out="$SAMPLE_ROOT/$name/$checkpoint/sample_level"
    mkdir -p "$sample_out"
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" -m cosmos_policy.scripts.eval_so101_action_curves \
      --repo_id "$REPO_ID" \
      --root "$ROOT" \
      --checkpoint "$checkpoint_path" \
      --t5_text_embeddings_path "$T5_PATH" \
      --dataset_stats_path "$STATS_PATH" \
      --num_samples 16 \
      --chunk_size 50 \
      --num_denoising_steps 10 \
      --output_dir "$sample_out"

    for episode in "${EPISODES[@]}"; do
      episode_out="$EPISODE_ROOT/$name/$checkpoint/episode_$episode"
      mkdir -p "$episode_out"
      CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" -m cosmos_policy.scripts.eval_so101_full_episode_action_curve \
        --repo_id "$REPO_ID" \
        --root "$ROOT" \
        --checkpoint "$checkpoint_path" \
        --t5_text_embeddings_path "$T5_PATH" \
        --dataset_stats_path "$STATS_PATH" \
        --episode "$episode" \
        --query_stride 10 \
        --chunk_size 50 \
        --num_denoising_steps 10 \
        --output_dir "$episode_out"
    done
  done
  echo "[$name] 所有 500/1000/1500/2000 sample + episode 评估完成"
}

mkdir -p "$SAMPLE_ROOT/logs" "$EPISODE_ROOT"
pids=()
names=()
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r name _ _ <<< "$row"
  run_experiment "$row" >"$SAMPLE_ROOT/logs/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "实验评估失败: ${names[$index]}，查看 $SAMPLE_ROOT/logs/${names[$index]}.log" >&2
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  exit 1
fi

"$PYTHON_BIN" -m cosmos_policy.scripts.summarize_so101_sweep \
  --sample_root "$SAMPLE_ROOT" \
  --episode_root "$EPISODE_ROOT" \
  --output_root "$SAMPLE_ROOT"

echo "全部 sweep 评估与 training dynamics 汇总完成。"
