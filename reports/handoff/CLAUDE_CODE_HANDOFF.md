# Claude Code Handoff — Cosmos WAM / LIBERO

Generated from main commit `95f1c265354921bc22435158f469f81ffc0d490a`. This is an execution-context handoff, not authorization to run a new study. Start by preserving the clean E11 validation/heldout split and by reading the frozen negative results.

## 1. Executive Handoff Summary

Main repo: `/home/rxhuang/Projects/cosmos-policy` on branch `research/pv0-closed-loop-feedback-20260812`, commit `95f1c265354921bc22435158f469f81ffc0d490a`. Remote is `git@github.com:hrx20000209/cosmos-policy.git`. The worktree is intentionally dirty only through untracked runtime artifacts/logs:

```
?? artifacts/
?? experiments/handoff/
?? logs/esp/
?? logs/semantic_commitment_e11a_native_gpu6.log
?? logs/semantic_commitment_e11a_native_gpu6_egl.log
?? logs/semantic_commitment_e11a_native_gpu6_egl2.log
?? logs/semantic_commitment_e11a_route_transfer.log
?? logs/semantic_commitment_e11a_route_transfer_gpu2.log
?? logs/semantic_commitment_e11a_route_transfer_gpu5.log
?? logs/semantic_commitment_e11a_route_transfer_gpu5_egl.log
?? logs/semantic_risk/
```

Do not delete, reset, or casually add them. The last result is **SEMANTIC_COMMITMENT_NO_GO**, caused by E11-A F1-route transfer failure, not by an illegal temporal comparison. E11-B/C and E12 were not run. The 24 E11 tasks retain 8 discovery / 8 validation / 8 heldout task-disjoint hygiene; E11 validation and heldout have no sampled outcomes.

## 2. Current Scientific State

The stable positive systems mechanism is PV0: reuse the generated joint latent while inserting a causal fresh visual prefix into current condition slots before the single denoiser forward. Full scale result: F1 19/200, P1 11/200, PV0 17/200; PV0→P1 paired wins/losses 6/0. This supports *recovery of predicted reuse*, not universal Fresh replacement.

Frozen NO-GO / stop boundaries: physical EEF/proprio-only feedback; action amplification; hidden/x0/sparse patches; ESP finite difference and camera scheduler/per-camera selective VAE; raw optical innovation as primary estimator; S×I; illegal prediction-age labels; fixed R3+ as final method; denoise/value/deep scheduler; direct use of the old P1 E4 score at an F1 action anchor. Do not revive these without an entirely new protocol.

Semantic Risk E4 found P1 internal+action score heldout rho 0.651 vs action-only 0.563, but the 89-summary implementation cost 5.25 ms / 2.13% F1. E5 was weak; E6 product lost to additive. Sensitivity Horizon E8 stopped because K!=16 compares a t+16 predicted latent to wrong physical time. E11 correctly avoided that: its planned remaining-action oracle would be temporally legal, but it requires an F1-anchor signal. On 32 new discovery-only states, frozen `S_F1` vs the legal t+16 retrospective target was task-balanced rho 0.175 (<0.50). Thus this estimator is **P1_ONLY_SENSITIVITY** for commitment.

## 3. Repository Map

| Role | Path | Revision / status |
| --- | --- | --- |
| Main Cosmos WAM | `/home/rxhuang/Projects/cosmos-policy` | `research/pv0-closed-loop-feedback-20260812` / `95f1c265354921bc22435158f469f81ffc0d490a` |
| Active LIBERO-PRO selected by bank rows | `/data/rxhuang/repos/LIBERO-PRO` | `master` / `eafdb809426b13153aa1e4c42d6601844217dfec` |
| Extra LIBERO checkout | `/home/rxhuang/Projects/LIBERO` | `master` / `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Server raw output roots | `/data/rxhuang/wam_server_deep_validation`, `/data/rxhuang/wam_full_scale_server` | data roots, not git repos |
| Model root | `/data/rxhuang/models/cosmos-policy-libero-2b` | checkpoint/statistics |
| assets root | `/data/rxhuang/wam_libero_outputs` | T5, BDDL/init assets |

## 4. Environment / Startup Commands

```bash
cd /home/rxhuang/Projects/cosmos-policy
source .venv/bin/activate
export PYTHONPATH=.
export HF_HOME=/data/hf_cache
export HF_HUB_OFFLINE=1
# Headless LIBERO workers: explicit EGL worked in E11-A; OSMesa failed here.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

Environment observed at handoff: Ubuntu 22.04 kernel 6.8, Python 3.10.12 in `.venv`, torch 2.7.0+cu128, CUDA 12.8, driver 560.35.03, eight RTX 4090 D (24564 MiB each). Before any approved GPU job set `CUDA_VISIBLE_DEVICES=<id>`, `EVAL_PHYSICAL_GPU=<id>`, and `MUJOCO_EGL_DEVICE_ID=<id>`. Query `nvidia-smi` and do not kill other users’ processes. The helper defaults to OSMesa, so explicit EGL must be present before imports that initialize MuJoCo.

## 5. Checkpoint / Model Contract

- Original checkpoint: `/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt`
- SHA256: `8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2`
- Dataset stats: `/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json`
- T5 cache: `/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/cosmos_libero_pro_t5_embeddings.pkl`
- Model load: `experiments/server_deep_validation/pv0_overnight_common.py::build_model` → `experiments/progressive_wam/run_p1_trajectory_dump.py::load_model` → `cosmos_policy.experiments.robot.cosmos_utils::get_model`.
- Inference: `cosmos_policy/experiments/robot/cosmos_utils.py::get_action`; it calls `generate_samples_from_batch` with one step, extracts slot-4 action latent by `extract_action_chunk_from_latent_sequence`, and unnormalizes to float32 numpy.
- Do not use a SO101/finetuned checkpoint, Cosmos value, trainable module, privileged simulator state as policy input, or denoise count other than 1.

## 6. Temporal H=16 Contract

The model is 28 DiT blocks, hidden width 2048. The 9 temporal VAE slots are `{"0": "temporal VAE leading placeholder", "1": "current proprio", "2": "current wrist image", "3": "current primary image", "4": "action chunk", "5": "future proprio", "6": "future wrist image", "7": "future primary image", "8": "value (structurally present; never read)"}`. The policy returns `A_t` of shape 16×7 and future visual slots 6/7. In `cosmos_policy/datasets/libero_dataset.py`, `next_relative_step_idx = relative_step_idx + chunk_size`; config sets chunk_size=16. Thus `A_t[0:16]` and `z_hat_(t+16)` are bound to the same temporal horizon.

Action queue execution is one action per `env.step`: `runtime/action_buffer.py::ActionBuffer.install/pop` and `experiments/libero_harness.py::run_episode`. Actions are 7D OSC-POSE **relative EEF delta plus gripper command** (see `run_pv0_closed_loop_episode.py` execution feedback contract). The logical control frequency defaults to 20 Hz in `run_episode`; exact simulator physical dt was not separately verified and is an open item.

P1 moves prior generated slots 6/7 into current slots 2/3 through `adapters/cosmos_adapter.py::CosmosAdapter._predicted_visual_latent`. It is only temporally aligned after 16 executed actions. Do not produce age labels by comparing that fixed t+16 condition with F1 at K!=16.

## 7. F1 / P1 / STALE / PV0 Implementation Map

All route choice/cache logic is `adapters/cosmos_adapter.py::CosmosAdapter._visual_input_for_request` and `infer`.

- **F1**: current RGB+proprio, VAE encode, no previous latent. Sets `previous_real_latent`; always overwrites `previous_generated_latent` after inference.
- **P1**: `predicted_reuse`; prior generated latent is cloned, predicted slots 6/7 copied to current 2/3, VAE/camera preprocessing bypassed. Result overwrites `previous_generated_latent`.
- **STALE**: `stale_r2`; stale phases use `last_physical_condition_latent`, retained from a PV0 request, rather than future slots.
- **PV0**: `native_persistent`/`pv0_r*`; gets prior generated base condition plus `persistent_visual_correction_prefix_frames=13`, with arrival at denoiser forward 0. It records `persistent_condition_latent` as `last_physical_condition_latent`, then overwrites generated cache.

Route contracts are also explicit in `experiments/server_deep_validation/pv0_overnight_common.py::route_contract` and `experiments/server_deep_validation/run_pv0_closed_loop_episode.py::route_contract`. `CosmosAdapter.infer_shadow_validity_labels` is retrospective: it calls `get_action` direct and does not mutate adapter cache fields; new code should still isolate RNG/model hooks when adding a shadow route.

## 8. State Replay Infrastructure

State registry: `reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl` (3801 rows, 40 tasks). It points to `collection_episode` `.pt` bundles under `/data/rxhuang/wam_full_scale_server/queue_a/f1_collection`; request records contain `sim_state`, generated latent, fresh action, control step, seed, instruction, and source/target indices. Simulator state reconstructs observations only; it is never passed to the WAM.

Use `experiments/esp/run_e2_e3_pilot_shard.py::load_request` to enforce source/target 16-action alignment and `::render` to restore/re-render observation. Robust restore is `experiments/progressive_wam/run_p2_oracle.py::restore`: regenerates MuJoCo state, resets done/timestep, zeroes warmstart, forwards simulation, and resets OSC controller goal. Preserve `state_key`, `episode_key`, instruction, init state index, seed, observation/proprio hashes, and invalid reasons in future paired artifacts.

## 9. Task Splits and Data Hygiene

The new E11 split is the same reserved E8 24-task split:

- discovery:
  - `libero_10:KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
  - `libero_10:KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`
  - `libero_10:LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`
  - `libero_10:LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`
  - `libero_10:LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
  - `libero_10:LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
  - `libero_10:STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`
  - `libero_goal:open_the_top_drawer_and_put_the_bowl_inside`
- heldout:
  - `libero_object:pick_up_the_salad_dressing_and_place_it_in_the_basket`
  - `libero_spatial:pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
  - `libero_spatial:pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`
  - `libero_spatial:pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`
  - `libero_spatial:pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`
  - `libero_spatial:pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`
  - `libero_spatial:pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`
  - `libero_spatial:pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`
- validation:
  - `libero_goal:push_the_plate_to_the_front_of_the_stove`
  - `libero_goal:put_the_bowl_on_the_stove`
  - `libero_goal:put_the_cream_cheese_in_the_bowl`
  - `libero_goal:put_the_wine_bottle_on_the_rack`
  - `libero_object:pick_up_the_bbq_sauce_and_place_it_in_the_basket`
  - `libero_object:pick_up_the_butter_and_place_it_in_the_basket`
  - `libero_object:pick_up_the_chocolate_pudding_and_place_it_in_the_basket`
  - `libero_object:pick_up_the_ketchup_and_place_it_in_the_basket`

Discovery was touched only by E11-A’s 32 rows (`artifacts/semantic_commitment/e11a_route_transfer.parquet`). **Validation and heldout are UNTOUCHED**: do not inspect their outcome artifacts, run scorers on them, or use them for feature/model/threshold choices before a new protocol freezes a candidate. Older ESP/E4 splits are historical formal resources, not fresh confirmation; paths are `reports/esp/TASK_SPLIT.json` and `reports/semantic_risk/TASK_SPLIT.json`.

## 10. E4 Semantic Sensitivity Implementation

Original extraction: `experiments/semantic_risk/run_semantic_shard.py::run_route`. It installs `model.intermediate_feature_ids` for blocks `(4,8,12,16,20,24,27)` (zero-indexed IDs passed as block-1). The reducer observes each selected post-block tensor and emits 12 scalar terms: action-slot RMS/dispersion; future wrist/primary RMS; current wrist/primary RMS; and six pooled cosine relations between action/current/future slots. That makes 84 internal scalars. `action_geometry` supplies five action terms: `action_norm`, `endpoint_displacement`, `action_curvature`, `action_jerk`, `gripper_transition`.

`reports/semantic_commitment/FROZEN_SENSITIVITY_MODEL.json` serializes exact feature names/order, mean, scale, coefficients, intercept, target transform and checksum `3299a5ec2308f990ff68e6f80f2d28ca922b4aef8c72c2f708da139bd6745d4b`. It was reconstructed by `experiments/semantic_commitment/freeze_sensitivity.py` from the original 352 discovery+validation E4 rows, with historical prediction max absolute error `3.15e-14`. Fit code: `experiments/semantic_risk/analyze_semantic_risk.py::fit_predict`, ridge λ=10, target `log1p(causal_sensitivity_target)`, output `expm1(linear)`.

All 89 features are mechanically definable on either F1 or P1 because the same passive reducer/action geometry runs on both. Observed, not explained: on E11 discovery preflight S_F1/S_P1 rank rho was 0.955, yet S_F1 vs legal target task-balanced rho was 0.175. Do not infer a causal explanation not present in data.

## 11. E11 Route-Transfer Result

Read `reports/semantic_commitment/SEMANTIC_COMMITMENT_SERVER_REPORT_ZH.md`, `E11A_ROUTE_TRANSFER.json`, and `SEMANTIC_COMMITMENT_FINAL_DECISION.json`. Artifact reconstruction passed. Discovery-only preflight used 8 clean-split discovery tasks × 4 states = 32 states. Results: S_F1 vs S_P1 pooled Spearman 0.9553; S_F1 vs legal t+16 retrospective target task-balanced Spearman 0.175; S_P1 equivalent 0.350. Preregistered F1 transfer gate was >=0.50. Therefore the score is `P1_ONLY_SENSITIVITY` for commitment. E11-B’s remaining-plan oracle would be temporally legal, but was not authorized; E11-C and E12 are not authorized.

## 12. Runtime Scorer Implementation

Original passive summary path is `run_semantic_shard.py::run_route`; it materializes each selected block’s 12 values with `.detach().cpu().tolist()` after timed `get_action`. Original clean E4 profile is `reports/semantic_risk/E4_COST_RESULT.json`: P1 73.61 ms, P1+89 summaries 78.87 ms, +5.25 ms (2.13% clean F1), so runtime NO-GO under <1% gate.

Native preflight: `experiments/semantic_commitment/profile_native_scorer.py::native_route` weights each block’s 12 GPU reductions and returns seven scalar contributions, then applies the exact `expm1` transform. On 12 states/20 repeats: action max error 0; score max abs error 1.075e-5 (slightly over 1e-5 preflight tolerance due to floating summation order); baseline 77.61 ms, original 84.44 ms, native 85.05 ms. This is **not** a formal 100-repeat profile, no component kernel/D2H breakdown was completed, and it did not reduce cost.

## 13. Existing Experiment / Artifact Index

| Experiment | Decision | Report | Raw Data | Split | Notes |
| --- | --- | --- | --- | --- | --- |
| PV0 full-scale / 600 route episodes | mechanism GO; universal Fresh replacement NO-GO | [FINAL_PV0_REPORT_ZH.md](/home/rxhuang/Projects/cosmos-policy/reports/pv0_overnight/FINAL_PV0_REPORT_ZH.md) | `/data/rxhuang/wam_server_deep_validation and /data/rxhuang/wam_full_scale_server` | `/home/rxhuang/Projects/cosmos-policy/reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl` |
| fixed R1/R2/R3 + matched stale/execution feedback | fixed R2 candidate only; adaptive feedback NO-GO | [FIXED_REUSE_PILOT_REPORT_ZH.md](/home/rxhuang/Projects/cosmos-policy/reports/pv0_execution_feedback/fixed_reuse_pilot/FIXED_REUSE_PILOT_REPORT_ZH.md) | `/home/rxhuang/Projects/cosmos-policy/reports/pv0_execution_feedback` | `experiment-local manifests/reports` |
| ESP E1/E2/E3 | ESP_NO_GO | [ESP_SERVER_REPORT_ZH.md](/home/rxhuang/Projects/cosmos-policy/reports/esp/ESP_SERVER_REPORT_ZH.md) | `/home/rxhuang/Projects/cosmos-policy/reports/esp/shards` | `/home/rxhuang/Projects/cosmos-policy/reports/esp/TASK_SPLIT.json` |
| Semantic risk E4/E5/E6 | SEMANTIC_RISK_NO_GO | [SEMANTIC_RISK_SERVER_REPORT_ZH.md](/home/rxhuang/Projects/cosmos-policy/reports/semantic_risk/SEMANTIC_RISK_SERVER_REPORT_ZH.md) | `/home/rxhuang/Projects/cosmos-policy/artifacts/semantic_risk/e4_e5_e6_raw.parquet` | `/home/rxhuang/Projects/cosmos-policy/reports/semantic_risk/TASK_SPLIT.json` |
| Sensitivity horizon E8/E9/E10 | SENSITIVITY_HORIZON_NO_GO: invalid age sweep | [SENSITIVITY_HORIZON_SERVER_REPORT_ZH.md](/home/rxhuang/Projects/cosmos-policy/reports/sensitivity_horizon/SENSITIVITY_HORIZON_SERVER_REPORT_ZH.md) | `none; formal samples=0` | `/home/rxhuang/Projects/cosmos-policy/reports/sensitivity_horizon/TASK_SPLIT_E8.json` |
| Semantic commitment E11 | SEMANTIC_COMMITMENT_NO_GO at F1 route-transfer gate | [SEMANTIC_COMMITMENT_SERVER_REPORT_ZH.md](/home/rxhuang/Projects/cosmos-policy/reports/semantic_commitment/SEMANTIC_COMMITMENT_SERVER_REPORT_ZH.md) | `/home/rxhuang/Projects/cosmos-policy/artifacts/semantic_commitment/e11a_route_transfer.parquet` | `/home/rxhuang/Projects/cosmos-policy/reports/semantic_commitment/TASK_SPLIT_E11.json` |

## 14. Safe Discovery Resources

Safe under a future, explicit protocol: the eight E11 discovery tasks above; Foundation V2 registry (with provenance check); `artifacts/semantic_commitment/e11a_route_transfer.parquet`; prior raw tables as historical evidence; frozen original checkpoint; F1/P1/PV0 route code; state restore/replay helpers. New experiments must not silently reuse historic heldout tasks as “new confirmation.”

## 15. Clean Validation / Heldout Resources

E11 validation and heldout tasks are reserved clean resources, as listed above. They have no E8/E11 outcome artifacts. Do not run even a “quick” model probe that creates outcome labels, because that consumes the task-disjoint confirmation resource. Only inspect split metadata during handoff.

## 16. Existing Smoke / Unit Tests

Relevant existing smoke/preflight files: `experiments/esp/run_esp_smoke.py`, `experiments/preflight_wam_libero.py`, `experiments/server_deep_validation/run_native_persistent_condition_preflight.py`, and `experiments/server_deep_validation/run_modular_worker_preflight.py`. **NOT RUN DURING THIS HANDOFF**: they can load models/simulators and consume GPU. Safe no-inference checks are `python -m py_compile experiments/semantic_commitment/*.py`, JSON schema inspection, and `git diff --check`.

## 17. GPU / Job Scheduler

Existing multi-GPU mechanisms: `experiments/server_deep_validation/pv0_overnight_supervisor.py` (args `--run-dir --manifest --collection-root --ablation-root --duration-hours --poll-seconds`), `pv0_fixed_reuse_pilot_supervisor.py` (`--manifest --run-dir --gpu-ids --poll-seconds`), `execution_validity_shadow_supervisor.py` (`--manifest --run-dir --poll-seconds`), plus `server_queue_supervisor.py` and `server_stage_supervisor.py`. Workers conventionally set `CUDA_VISIBLE_DEVICES`, `EVAL_PHYSICAL_GPU`, memory fraction and write resumable per-shard JSON/log outputs. Query free VRAM at launch; never kill other users. In E11-A, GPU 1/2 failed model construction under shared-memory pressure; GPU 5/6 worked with explicit EGL.

## 18. Logging and Output Conventions

Reports live under `reports/<experiment>/`; raw/parquet under `artifacts/<experiment>/`; plots under `reports/<experiment>/plots/`; resumable shard JSON under `reports/<experiment>/shards/`; logs usually under `logs/` or report-local `logs/`. JSON reports use `{status, checkpoint, checkpoint_sha256, denoising_steps, value_used, finetuning_used, gpu, rows/...}`. State IDs are `<episode_hash>:req<index>`. Task ID is `suite:task_name`. Keep one row per paired unit, include split, seed, state/observation hashes, `valid`, and `invalid_reason`; do not discard failures silently.

## 19. Open Implementation Questions

1. **[REQUIRES_RUNTIME_VERIFICATION or simulator source audit]** Exact physical control dt. `control_frequency_hz=20` is a logical fallback; `video_fps` and model dataset fps are not proof of simulator dt.
2. A separately valid F1-route internal signal has not been developed. Evidence and code are `E11A_ROUTE_TRANSFER.json` and `run_e11a_route_transfer.py`; this must be discovery-only before touching E11 validation/heldout.
3. Why F1/P1 score calibration differs despite rank consistency is unknown. No causal explanation exists in current artifacts.
4. Original E4’s 5.25 ms component-level GPU/D2H/Python attribution was not measured. `profile_e4_hook.py` only measures paired route timing.

## 20. Do-Not-Touch List

- E11 clean validation/heldout tasks and any future outcomes on them.
- The frozen E4 artifact and prior formal JSON/parquet tables: no in-place refit, feature selection, score rewrite, or overwritten output.
- Original checkpoint/state banks and historic NO-GO results.
- `artifacts/` and `logs/` are untracked user/runtime material; retain them. Current E11 logs include OOM/OSMesa attempts as well as successful EGL runs.
- No E11-B/C/E12, no new scheduler, no hidden/action patches, no ESP/camera/E5/S×I revival, before a new user research protocol.

## 21. Claude First-30-Minutes Checklist

1. Activate `.venv` and export `PYTHONPATH=.`.
2. Run `git status --short`, `git log -12 --oneline`, `git branch --show-current`; do not clean the worktree.
3. Read: this file; `reports/semantic_commitment/SEMANTIC_COMMITMENT_SERVER_REPORT_ZH.md`; `reports/semantic_commitment/FROZEN_SENSITIVITY_MODEL.json`; `reports/sensitivity_horizon/SENSITIVITY_HORIZON_SERVER_REPORT_ZH.md`; `reports/pv0_overnight/FINAL_PV0_REPORT_ZH.md`.
4. Inspect: `adapters/cosmos_adapter.py`; `cosmos_policy/experiments/robot/cosmos_utils.py`; `experiments/semantic_risk/run_semantic_shard.py`; `experiments/progressive_wam/run_p2_oracle.py`; `experiments/semantic_commitment/run_e11a_route_transfer.py`.
5. Inspect `TASK_SPLIT_E11.json`, verify no outcome files exist for E11 validation/heldout; do not run a model.
6. Query GPU state only with `nvidia-smi`; no job launch.
7. Wait for a new researcher protocol before changing code or collecting data.

## 22. Exact Files Claude Should Read First

1. `reports/handoff/CLAUDE_HANDOFF.json`
2. `reports/semantic_commitment/SEMANTIC_COMMITMENT_SERVER_REPORT_ZH.md`
3. `reports/semantic_commitment/FROZEN_SENSITIVITY_MODEL.json`
4. `reports/semantic_commitment/TASK_SPLIT_E11.json`
5. `reports/sensitivity_horizon/AGE_SEMANTICS.json`
6. `reports/pv0_overnight/FINAL_PV0_REPORT_ZH.md`
7. `adapters/cosmos_adapter.py`
8. `experiments/semantic_risk/run_semantic_shard.py`
9. `experiments/progressive_wam/run_p2_oracle.py`
10. `experiments/semantic_commitment/run_e11a_route_transfer.py`
