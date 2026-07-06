# Three Cubes SO101 post-training and deployment

This path is intentionally schema-gated. The policy target is a 30-step chunk of six **absolute position targets** in this exact order:

`shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos`

The first five values are LeRobot calibrated degrees. The gripper is calibrated percent range (`0..100`) and larger values open it. Runtime refuses a different calibration or schema digest. Dataset `wrist_flex` contains a few targets up to `100.484°`; runtime clips those to the calibrated `100°` limit and records the clip.

Run commands from `/home/rxhuang/Projects/cosmos-policy`.

## 1. Inspect the dataset and action schema

```bash
PYTHONPATH=$PWD python -m cosmos_policy.experiments.robot.aloha.inspect_lerobot_three_cubes \
  --dataset-root /data/rxhuang/three_cubes_1 \
  --output-json /data/rxhuang/cosmos_three_cubes_runs/dataset_inspection.json
```

This prints camera/state/action keys, dimensions, physical ranges, joint order, gripper direction, fps, episode lengths, and statistics.

## 2. Generate the task embedding and dry-run

```bash
uv run --extra cu128 --group aloha --python 3.10 \
  python -m cosmos_policy.datasets.save_so101_t5_text_embeddings \
  --data-dir /data/rxhuang/three_cubes_1

uv run --extra cu128 --group aloha --python 3.10 \
  python -m cosmos_policy.experiments.robot.aloha.dryrun_three_cubes_so101 \
  --dataset-root /data/rxhuang/three_cubes_1 --overfit-episodes 1

RUN_NAME="dryrun_$(date +%Y%m%d_%H%M%S)"
IMAGINAIRE_OUTPUT_ROOT=/data/rxhuang/cosmos_three_cubes_runs \
uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py --dryrun -- \
  experiment=cosmos_predict2_2b_480p_three_cubes_so101_posttrain \
  job.name="$RUN_NAME"
```

The first dry-run decodes actual AV1 frames and checks `[B,3,41,224,224]` video, `[B,30,6]` action and `[B,6]` proprio tensors. The second resolves and saves the complete model/training config.

## 3. Two-episode overfit sanity check

```bash
RUN_NAME="overfit_$(date +%Y%m%d_%H%M%S)"
IMAGINAIRE_OUTPUT_ROOT=/data/rxhuang/cosmos_three_cubes_runs \
uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_three_cubes_so101_posttrain \
  job.name="$RUN_NAME" trainer.max_iter=500 trainer.run_validation=false \
  checkpoint.save_iter=100 \
  dataloader_train.dataset.overfit_num_episodes=2 \
  dataloader_train.dataset.val_episodes=0
```

Do not promote this checkpoint merely because total loss falls. Run the offline action curve below on both overfit episodes and require all six physical curves—including gripper direction—to follow ground truth.

## 4. Full post-training

The checked-in configuration uses rank-32 LoRA so the first pass is practical on a single 24 GB card; it still trains the Cosmos policy action/future-state objectives and loads the released 2B base checkpoint.

```bash
RUN_NAME="posttrain_$(date +%Y%m%d_%H%M%S)"
IMAGINAIRE_OUTPUT_ROOT=/data/rxhuang/cosmos_three_cubes_runs \
uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_three_cubes_so101_posttrain \
  job.name="$RUN_NAME"
```

Check `stdout.log` and W&B for total training/validation loss, demo action MSE/L1, future proprio loss, future wrist/front image loss and value loss. Checkpoints are saved every 500 steps plus the final checkpoint under:

`/data/rxhuang/cosmos_three_cubes_runs/cosmos_policy/three_cubes_so101/<RUN_NAME>/checkpoints/`

The same component metrics are always appended locally to `metrics.jsonl`. Plot them with:

```bash
uv run --extra cu128 --group aloha --python 3.10 \
  python -m cosmos_policy.experiments.robot.aloha.plot_three_cubes_losses --run-dir "$RUN_DIR"
```

## 5. Predicted versus ground-truth action curves

```bash
RUN_DIR=/data/rxhuang/cosmos_three_cubes_runs/cosmos_policy/three_cubes_so101/<RUN_NAME>
CKPT="$RUN_DIR/checkpoints/<CHECKPOINT>"

uv run --extra cu128 --group aloha --python 3.10 \
  python -m cosmos_policy.experiments.robot.aloha.eval_three_cubes_offline_action_curve \
  --checkpoint "$CKPT" --episode 0 --timestep 100 --output-dir "$RUN_DIR/offline_eval"
```

The output directory contains a six-panel PNG, an NPZ with predicted/GT/error arrays, and JSON containing overall/per-joint MSE. Repeat at grasp/release timesteps; a single aggregate MSE can hide a reversed or static gripper.

## 6. Start the policy server

```bash
uv run --extra cu128 --group aloha --python 3.10 \
  python -m cosmos_policy.experiments.robot.aloha.deploy_so101_three_cubes \
  --ckpt_path "$CKPT" --host 0.0.0.0 --port 8777 \
  --dataset_stats_path cosmos_policy/experiments/robot/aloha/three_cubes_so101_dataset_statistics.json \
  --t5_text_embeddings_path /data/rxhuang/three_cubes_1/t5_embeddings.pkl
```

Every request prints observation keys/shapes, action shape/range/mean/std and inference time. Returned actions are already unnormalized absolute targets with shape `[30,6]`.

## 7. Start the SO101 client

Observation/query-only safety run (no motor commands):

```bash
PYTHONPATH=/home/rxhuang/Projects/cosmos-policy:/home/rxhuang/Projects/lerobot/src \
/home/rxhuang/anaconda3/envs/lerobot/bin/python \
  -m cosmos_policy.experiments.robot.aloha.run_so101_three_cubes_eval \
  --server http://127.0.0.1:8777/act --robot-port /dev/ttyACM1
```

Only after inspecting the timeline and first predicted chunk, enable commands:

```bash
PYTHONPATH=/home/rxhuang/Projects/cosmos-policy:/home/rxhuang/Projects/lerobot/src \
/home/rxhuang/anaconda3/envs/lerobot/bin/python \
  -m cosmos_policy.experiments.robot.aloha.run_so101_three_cubes_eval \
  --server http://127.0.0.1:8777/act --robot-port /dev/ttyACM1 \
  --max-joint-delta 8 --max-gripper-delta 12 --execute
```

`--prefetch-threshold 5` optionally queries early. A new chunk is accepted only if its first target is within `--handoff-max-delta` of the last executed target. Timeline JSONL includes observations, predicted chunks, executed/clipped targets, qpos, latency and queue size; query images are saved alongside it.

## Troubleshooting

- **Camera mismatch:** runtime names must be `front`, `right`, `wrist`; they map to Cosmos primary, wrist-1 and wrist-2 in that fixed order.
- **Action dimension/order mismatch:** both server and client must print the six-key schema and the same SHA-256 digest. Never reorder by dictionary iteration.
- **Normalization stats missing:** use the checked-in statistics JSON; server must have `unnormalize_actions=true`.
- **Calibration/schema mismatch:** runtime deliberately aborts unless `follower_arm.json` exactly matches the approved calibration.
- **Gripper reversed:** approved direction is larger=open. Confirm predicted and GT gripper curves at both grasp and release before `--execute`.
- **Queue empty / latency high:** first measure server latency. Use a small positive prefetch threshold only after the basic empty-queue mode is stable.
- **Near-zero model output:** inspect server action statistics in physical units and the saved curve. Absolute joint targets should resemble dataset ranges, not values clustered around zero or `[-1,1]`.
- **Safety delta exceeded:** do not raise thresholds reflexively. Compare current qpos, target, calibration, normalization and camera scene first.
