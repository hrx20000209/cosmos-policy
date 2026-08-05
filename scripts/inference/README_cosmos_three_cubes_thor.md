# Cosmos Policy × Three Cubes (step 5000) on Jetson Thor

Deployment record for the step-5000 full-finetune checkpoint on the real SO-101.
Written 2026-07-22 on the Thor device itself.

**Status: Stage 1a (no-robot smoke test) PASSED. The checkpoint has still never driven a
physical arm.** Stages 1b and 2 below are not done and require a person at the robot.

---

## 1. What is on this device

| Artifact | Path | Provenance |
|---|---|---|
| Checkpoint (step 5000) | `/home/hrx/Projects/models/cosmos_policy/iter_000005000` | transferred, full 11 GB (all 4 subdirs) |
| T5 text embeddings | `/home/hrx/Projects/models/cosmos_policy/so101_t5_embeddings.pkl` | transferred from the training machine |
| Dataset statistics | `/home/hrx/Projects/models/cosmos_policy/so101_dataset_statistics.json` | **regenerated on Thor**, see §3 |
| VAE / tokenizer | HF cache, `models--nvidia--Cosmos-Predict2-2B-Video2World/.../tokenizer/tokenizer.pth` | already present; resolved via `hf://`, no transfer needed |

The T5 pkl was verified to contain exactly one key, matching the Three_Cubes_1 task string
byte-for-byte, with value shape `(1, 512, 1024)` bfloat16:

```
'go to red cube. take the red cube. go to box. put the red cube in box.'
```

That is the same string `run_cosmos_three_cubes_client.sh` passes as `--task`. Changing the
task string changes the conditioning and would fall back to on-demand T5 computation, which
would require downloading `google-t5/t5-11b` (~45 GB). Don't edit it.

## 2. Code changes (`so101_async_deploy.py`)

Only what §3 of the deployment brief required — safety clamps were **not** touched:

* `chunk_size` / `num_open_loop_steps`: `50 → 30` (this checkpoint was trained at 30).
* Chunk shape check: hardcoded `(50, 6)` → derived from `cosmos_cfg`, so it can never drift
  out of sync with `chunk_size` again. Without this fix the server raises `RuntimeError` on
  the very first inference.
* Default `ckpt_path` / `dataset_stats_path` / `t5_text_embeddings_path` → the Thor paths above.
* Camera keys needed **no change**: `front` / `right` / `wrist` already match both the
  training data (`observation.images.{front,right,wrist}`) and the robot client config.
* **Action chunks are now pickled with stock torch tensor serialization** — see §2.1.

### 2.1 The `No module named 'megatron'` client crash

Hit on the first real robot-client run:

```
File "lerobot/async_inference/robot_client.py", line 342, in receive_actions
    timed_actions = pickle.loads(actions_chunk.data)
ModuleNotFoundError: No module named 'megatron'
```

`megatron.core` monkeypatches `torch.storage._load_from_bytes` at import time
(`megatron/core/__init__.py:58`), and Cosmos Policy imports megatron transitively.
`torch.storage._StorageBase.__reduce__` resolves that name as a **module global at call
time**, so every tensor this server pickled recorded
`megatron.core.safe_globals.safe_load_from_bytes` as its storage rebuild function. The
robot client runs in the `lerobot` conda env, which has no megatron, so it died on unpickle.

Sending numpy instead is not an option — the client reads
`timed_actions[0].get_action().device.type` (`robot_client.py:347`), so it requires a real
`torch.Tensor`.

Fix: `_plain_torch_tensor_pickling()` restores the stock `torch.storage._load_from_bytes`
for the duration of the `pickle.dumps` only, under a lock (the gRPC server uses a 4-thread
pool). The emitted pickle then references `torch.storage._load_from_bytes`, which any plain
torch env resolves. Nothing else in the process is affected, and the checkpoint is already
loaded before any of this runs.

Verified by unpickling a captured chunk **in the `lerobot` env** (megatron confirmed absent):
10 actions, `torch.Tensor`, `device.type='cpu'`, `float32`, shape `(6,)`, correct
`joint_order` — and in dry run the values still equal proprio exactly.

This is not Thor-specific: any deployment serving Cosmos to a stock LeRobot client hits it.

## 3. Why the dataset statistics file was regenerated

The brief expected `so101_dataset_statistics.json` to be transferred; it was not. The
similar-looking in-repo file
`cosmos_policy/experiments/robot/aloha/three_cubes_so101_dataset_statistics.json` is an
**older schema** — it has `joint_order` but not `action_names` / `state_names` /
`action_dim`, so it makes the server's startup safety gate
(`_validate_and_log_action_schema`) abort. That gate was left intact.

Instead the file was regenerated on Thor from the same source of truth the training machine
used — the LeRobot `Three_Cubes_1` metadata — via the repo's own
`compute_so101_statistics(action_mode="absolute")`, and cross-checked against the older
file before being written:

* `actions_min`, `actions_max`, `proprio_min`, `proprio_max` — match **bit-exactly** (`max|diff| = 0`).
  These are the only stats the inference path reads (`cosmos_utils.py:637-666`), and they
  also drive the server's safety clipping.
* `actions_mean/std`, `proprio_mean/std` — differ by ~1e-3, from float32 accumulation order
  over 51387 frames. These fields are **not read anywhere** under `experiments/robot/`; they
  are carried for schema auditing only.

## 4. Environment work required on Thor (aarch64)

None of this changed training or inference logic.

* **Transformer Engine** — PyPI ships `transformer_engine_cu12/cu13` for **x86_64 only**, so
  the pip path cannot work here. Built **TE 2.16.0 from the NVIDIA source repo** for
  `sm_110` / CUDA 13.0 / torch 2.10. Needed `nvidia-nccl-cu13` first (TE's `logging.h`
  includes `nccl.h` unconditionally).
* **A phantom TE was shadowing the real one.** `cosmos-policy/transformer_engine/` contained
  only stale `__pycache__` `.pyc` files with no source — Python treated it as an empty
  namespace package, so `import transformer_engine` "succeeded" while silently providing
  nothing. It was moved aside. If you re-clone or something recreates that directory, TE
  will break again in exactly this confusing way.
* `triton` 3.7.1, `tensorboard` — plain installs, both have aarch64 wheels.
* **`av` pinned to 15.1.0.** cosmos-policy asks for `av>=16.0.1`, LeRobot pins `<16.0.0`, and
  LeRobot's `pyav_utils` breaks on av 18 (`av.option`). `av` is only used for video
  encode/decode in training/analysis paths in both packages; inference touches neither. The
  shared LeRobot source was deliberately not patched.
* **`decord` is an import-only stub** (`site-packages/decord.py`). No aarch64/py3.12 wheel
  exists and upstream is unmaintained. It is needed only because checkpoint loading
  recursively imports every config module, including training-only dataset providers; the
  real-robot path receives live camera frames over gRPC and never decodes a video file. The
  stub raises loudly on any actual use, so it cannot silently substitute wrong data.

`cosmos_policy` is not pip-installed — both launch scripts set `PYTHONPATH` to the repo plus
`/home/hrx/Projects/lerobot/src`.

LeRobot here is **0.5.2**, not the 0.4.4 the brief assumed. The hand-rolled pickle-compat
dataclasses in `so101_async_deploy.py` (`TimedData` / `TimedAction` / `TimedObservation` /
`RemotePolicyConfig`) were checked field-by-field against 0.5.2's
`lerobot/async_inference/helpers.py` — identical field names and order, so the wire format
is compatible.

## 5. Port

`8080` on this host is normally the **LingBot-VA** policy server. These scripts default to
**8081** so the two can coexist. Keep server and client in sync.

## 6. Measured inference latency (this is the big operational constraint)

From the Stage 1a synthetic run (bf16, 3 cameras, `num_denoising_steps_action=10`):

| | latency |
|---|---|
| first chunk (warmup) | ~5.3 s |
| steady state | ~2.6–2.8 s per chunk (~0.37 Hz) |

With `actions_per_chunk=10`, staying ahead of inference needs `10 / FPS >= 2.7`, so
**FPS <= ~3.7**. At FPS=30 the arm would move for 0.33 s then stall ~2.4 s, repeatedly.
Both scripts therefore default to **FPS=3** — the same rate the LingBot-VA deployment
settled on. Actions are absolute joint targets, so the demonstrated path is preserved, just
executed ~10x slower than the 30 fps training data. For a first real-robot run that is a
feature, not a problem.

## 7. Staged validation

### Stage 1a — server exercised over gRPC, no robot — **PASSED 2026-07-22**

Ran the real gRPC protocol against the server with synthetic observations
(`scratchpad/synthetic_client.py`): no cameras, no serial port, nothing that can move.

* model loaded: 13.7 s, 4.20 GB VRAM, `missing_keys=[] unexpected_keys=[] incorrect_shapes=[]`
  (the skipped `_extra_state` keys are TE FP8 bookkeeping, handled by the loader by design)
* startup schema gate passed: correct `joint_order`, `action_dim=6`, gripper at index 5
* camera keys `front`/`right`/`wrist` accepted — no `KeyError`
* action chunk shape correct: model returned `(30, 6)`, server truncated to the configured
  10 — this is the direct proof that the `chunk_size` fix was both necessary and right
* dry run behaved exactly as specified: `max|action − proprio| = 0.0` on **all six joints**,
  i.e. zero commanded motion
* zero errors in the server log

Caveat: inputs were synthetic noise, so the **action values themselves are meaningless**.
This validated plumbing, schema, shapes and latency — nothing about policy quality.

### Stage 1b — dry run against the real robot — NOT DONE

```bash
# terminal 1 (DRY_RUN defaults to true -- the arm must not move)
scripts/inference/run_cosmos_three_cubes_server.sh
# wait for "Loaded Cosmos SO101 checkpoint", then terminal 2
scripts/inference/run_cosmos_three_cubes_client.sh
```

Verify the camera indices first (`lerobot_find_cameras opencv`) — front=2 / right=0 /
wrist=4 on this host, and a swapped order silently degrades the policy. Expect **no motion
at all**. If the arm moves during Stage 1b, stop immediately: something is wrong with the
dry-run path.

### Stage 2 — live, supervised — NOT DONE

Only after 1b is clean. `DRY_RUN=false`, a person beside the arm with a hand on the e-stop,
and the three cubes staged like the training scenes. This checkpoint has never run outside
its training distribution and its behavior is genuinely unknown.

Keep both safety layers:
* server-side clamps (`max_delta_from_observation=8.0`, `max_gripper_delta_from_observation=8.0`,
  `max_step_delta=4.0`, `max_gripper_step_delta=5.0`)
* client-side `--robot.max_relative_target=5.0`, which is independent of the server and still
  protects you if the server config is wrong

**Only ever tighten these. Never widen them and never set them to 0** (0 disables the clamp).

If you see jitter, jumps or anything dangerous: stop, return to dry run, and write down what
happened. Do **not** try to suppress it by raising `actions_per_chunk` or loosening the
clamps.

## 7.5 Why the arm only jittered in place (2026-07-22)

Diagnosed with `cosmos_policy/scripts/replay_so101_async_action_curve.py --episode 0`,
which replays a dataset episode through the *actual deployment server object* (same config,
same safety filter) in the async predict-30 / execute-10 / replan pattern.

**Under teacher forcing the model looks fine** — predictions track ground truth across the
whole episode including the large swings, MAE 2.2–7.9° against GT ranges of 46–146°.
**That number is misleading; see §7.6.** Teacher forcing hands the model the true state at
every replan, which flatters a policy that mostly echoes its proprio input.

**The `max_delta_from_observation=8.0` clamp is the cause.** It caps every action in a chunk
to ±8° of the observation at chunk start. Measured over all 100 episodes, the motion the
*human demonstrations themselves* require within one 10-step chunk is:

| joint | p99 | max | vs the 8° cap |
|---|---|---|---|
| shoulder_pan | 19.7 | 36.6 | 2.5x too small |
| shoulder_lift | 46.4 | 73.1 | **5.8x too small** |
| elbow_flex | 44.9 | 81.8 | **5.6x too small** |
| wrist_flex | 22.2 | 34.8 | 2.8x |
| wrist_roll | 31.8 | 47.1 | 4.0x |
| gripper | 18.4 | 46.6 | 2.3x |

So 8° is smaller than what the demonstrations need by ~6x on the two joints that do most of
the work. In the replay the clamp altered 22.3% of all commanded values, 40% on
shoulder_lift/elbow_flex, with max |clamped − raw| ≈ 90°. Every single chunk saturated at
exactly 8.00°.

By contrast `max_step_delta=4.0` is roughly the right order: GT per-step deltas are p99
3.6–3.9°, max 9.5°. This is the clamp worth keeping — at FPS=3 it directly bounds angular
velocity at `max_step_delta × FPS = 12 °/s`.

Note the client's `--robot.max_relative_target=5.0` caps per-step motion independently, so
relaxing the server alone still leaves that ceiling in place.

## 7.6 Why it approaches the cube but cannot grasp it (2026-07-22)

Run with `--closed_loop`, which feeds the model's own last executed action back as proprio
instead of the dataset state. With the clamps relaxed to 50 (verified non-binding —
`clamped max == raw max` on nearly every chunk):

* tracking error vs GT grows monotonically: 32° → 42° → 73° → **121°** by end of episode
* the predicted step away from proprio *shrinks* as it drifts, to 2.7–6°: the policy stops
  trying and holds near wherever it currently is
* in the figure (`outputs/so101_relaxed_ep0_closedloop.png`) every joint flattens after
  ~2.5 s while GT executes the full task; gripper sits at ~35 and never runs its open/close

So the safety clamps are **not** the reason grasping fails — this is with them wide open.
The behaviour is consistent with a **proprio shortcut** (copycat problem): the policy has
largely learned `action ≈ current state + small delta`, which scores well under
teacher-forced evaluation (where the true state is handed back every replan) and collapses
once it has to drive the state itself.

Supporting evidence: within one chunk the model produces only 50–80% of the motion GT does
(shoulder_lift 3.35° vs 5.99°, elbow_flex 3.24° vs 5.58°, wrist_flex 1.22° vs 2.45°).

**Caveat, and it matters:** in this replay the camera frames still come from the dataset, so
once the fed-back proprio diverges, images and proprio describe different arm poses. That
contradiction could itself cause the freeze. It is strong evidence, not proof. The clean
discriminator is the real robot, where images and proprio stay consistent.

Separately, a real config bug for grasping: GT gripper transitions are 10–12° over 5–6
frames, so `max_gripper_delta_from_observation=8.0` **cannot complete a single open or
close** within one chunk. Raise it (≥25) regardless of the above.

If the shortcut diagnosis holds, no deployment-side tuning fixes it — it is a training-side
issue (proprio dropout/noise augmentation, action-focused loss weighting, or more steps).

## 8. Known-but-not-done

The offline diagnostics that reduced chunk-boundary jumps from ~30x to ~2.5x used temporal
ensembling — exponentially-weighted blending of overlapping chunks. **This server does not
do that.** It just executes the first N steps of each fresh chunk. Lowering
`actions_per_chunk` (10 → 5) replans more often and pushes in the same direction, but it is
not equivalent. Real ensembling would mean reworking `_predict_action_chunk` to retain and
blend overlapping chunks — a real change, deliberately not made here, and in any case
unvalidated on hardware.
