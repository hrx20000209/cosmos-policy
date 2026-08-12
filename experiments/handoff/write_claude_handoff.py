#!/usr/bin/env python3
"""Generate a read-only execution handoff for a future Claude Code session."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path('/home/rxhuang/Projects/cosmos-policy')
REPORT = ROOT / 'reports' / 'handoff'


def cmd(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def p(rel: str) -> str:
    return str((ROOT / rel).resolve())


def write(path: Path, value: str | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    else:
        path.write_text(value.strip() + '\n', encoding='utf-8')


def main() -> None:
    e11split = read_json('reports/semantic_commitment/TASK_SPLIT_E11.json')['splits']
    e4split = read_json('reports/semantic_risk/TASK_SPLIT.json')['splits']
    esp_split = read_json('reports/esp/TASK_SPLIT.json')['splits']
    frozen = read_json('reports/semantic_commitment/FROZEN_SENSITIVITY_MODEL.json')
    e11 = read_json('reports/semantic_commitment/SEMANTIC_COMMITMENT_FINAL_DECISION.json')
    risk = read_json('reports/semantic_risk/SEMANTIC_RISK_FINAL_DECISION.json')
    horizon = read_json('reports/sensitivity_horizon/FINAL_DECISION.json')
    main_sha = cmd('git', 'rev-parse', 'HEAD')
    dirty = cmd('git', 'status', '--short')
    handoff = {
        'schema_version': 1,
        'generated_from_commit': main_sha,
        'generated_for': 'Claude Code execution handoff; read/inspect/document context only',
        'repos': [
            {'role':'main Cosmos WAM repo','path':str(ROOT),'branch':cmd('git','branch','--show-current'),'sha':main_sha,'dirty_status':dirty,'remote':'git@github.com:hrx20000209/cosmos-policy.git'},
            {'role':'active LIBERO-PRO checkout selected by state-bank rows','path':'/data/rxhuang/repos/LIBERO-PRO','branch':'master','sha':'eafdb809426b13153aa1e4c42d6601844217dfec','remote':'https://github.com/Zxy-MLlab/LIBERO-PRO.git'},
            {'role':'additional LIBERO checkout','path':'/home/rxhuang/Projects/LIBERO','branch':'master','sha':'8f1084e3132a39270c3a13ebe37270a43ece2a01','remote':'https://github.com/Lifelong-Robot-Learning/LIBERO.git'},
        ],
        'environment': {
            'os':'Ubuntu 22.04 kernel 6.8.0-110-generic (host user-ESC8000A-E11)',
            'venv':str(ROOT/'.venv'), 'python':str(ROOT/'.venv/bin/python'), 'python_version':'3.10.12',
            'torch':'2.7.0+cu128','cuda':'12.8','driver':'560.35.03','gpus':'8 x NVIDIA GeForce RTX 4090 D, 24564 MiB each',
            'startup':[
                'cd /home/rxhuang/Projects/cosmos-policy',
                'source .venv/bin/activate',
                'export PYTHONPATH=.',
                'export HF_HOME=/data/hf_cache',
                'export HF_HUB_OFFLINE=1',
                '# For headless simulator workers on this server, explicit EGL succeeded whereas the helper default OSMesa failed in E11-A:',
                'export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
                '# Set CUDA_VISIBLE_DEVICES and EVAL_PHYSICAL_GPU to the same chosen physical GPU; do not kill other users\' processes.',
            ],
            'requirements':[p('pyproject.toml'),p('cosmos_policy/requirements.txt'),p('cosmos_policy/pyproject.toml')],
            'runtime_note':'GPU availability is dynamic. Query nvidia-smi immediately before jobs; a model worker needs about 6 GB allocated but sharing can still OOM because of per-process 40% memory caps and fragmentation.'
        },
        'checkpoint': {
            'path':'/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt',
            'sha256':'8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2',
            'dataset_stats':'/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json',
            't5_embeddings':'/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/cosmos_libero_pro_t5_embeddings.pkl',
            'contract':'original pre-finetune LIBERO checkpoint only; deny SO101/finetuned paths; denoise=1; value unused; no training.'
        },
        'model_contract': {
            'model_load':[p('experiments/server_deep_validation/pv0_overnight_common.py')+'::build_model',p('experiments/progressive_wam/run_p1_trajectory_dump.py')+'::load_model',p('cosmos_policy/experiments/robot/cosmos_utils.py')+'::get_model'],
            'inference':[p('cosmos_policy/experiments/robot/cosmos_utils.py')+'::get_action',p('cosmos_policy/models/policy_video2world_model.py')+'::CosmosPolicyVideo2WorldModel',p('cosmos_policy/_src/predict2/networks/minimal_v4_dit.py')+'::MiniTrainDIT.forward'],
            'blocks':28,'hidden_dimension':2048,'precision':'BF16 weights (runtime audit); action output float32 numpy after extraction/unnormalization','action_shape':[16,7],
            'slots':{'0':'temporal VAE leading placeholder','1':'current proprio','2':'current wrist image','3':'current primary image','4':'action chunk','5':'future proprio','6':'future wrist image','7':'future primary image','8':'value (structurally present; never read)'},
            'temporal_contract':'H=16. Dataset builds future target via next_relative_step_idx = relative_step_idx + chunk_size; chunk_size=16. z_hat future slots and A[0:16] both target t+16.'
        },
        'routes': {
            'entry':p('adapters/cosmos_adapter.py')+'::CosmosAdapter._visual_input_for_request / infer',
            'F1':'fresh current RGB/proprio; skip_vae_encoding=False; previous_generated_latent=None; after infer sets previous_real_latent and always replaces previous_generated_latent with new generated latent.',
            'P1':'closed_loop_mode=predicted_reuse; previous_generated_latent is cloned and slots 6/7 copied to current slots 2/3 by _predicted_visual_latent; skip VAE/camera preprocessing; after infer replaces previous_generated_latent.',
            'STALE':'closed_loop_mode=stale_r2; uses last_physical_condition_latent on stale phases rather than future slots; state set by prior native_persistent route.',
            'PV0':'closed_loop_mode=native_persistent (or pv0_r*); prior generated latent base plus persistent_visual_correction_prefix_frames=13; fresh current visual slots arrive before sole denoiser forward; last_physical_condition_latent set from result persistent_condition_latent; generated cache replaced afterwards.',
            'shadow':'CosmosAdapter.infer_shadow_validity_labels invokes get_action directly for F1/P1/PV0 using fixed adapter seed and does not update adapter cache fields; it is retrospective/offline only.'
        },
        'state_banks':[
            {'name':'Foundation V2 registry','path':p('reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl'),'rows':3801,'tasks':40,'schema':'state_key, episode_key, source/target request indices, sim-state-bearing collection episode, init path/index, instruction, seed, task_uid; source/target aligned 16 actions.'},
            {'name':'Semantic-risk bank','path':p('reports/semantic_risk/SEMANTIC_RISK_STATE_BANK.jsonl'),'rows':512,'tasks':16,'status':'prior formal E4/E5/E6 resource; do not reuse as pristine confirmation'},
            {'name':'E11 A discovery rows','path':str(ROOT/'artifacts/semantic_commitment/e11a_route_transfer.parquet'),'rows':32,'tasks':8,'status':'new E11 discovery only; contains F1/P1 preflight scores and retrospective t+16 target.'},
            {'name':'F1 collection episode bundles','path':'/data/rxhuang/wam_full_scale_server/queue_a/f1_collection','schema':'*.pt has requests entries including sim_state, fresh_action, generated_latent, control_step, state_key; raw simulator state is replay-only, never policy input.'}
        ],
        'task_splits': {'esp':{'path':p('reports/esp/TASK_SPLIT.json'),'splits':esp_split},'semantic_risk':{'path':p('reports/semantic_risk/TASK_SPLIT.json'),'splits':e4split},'e11_clean':{'path':p('reports/semantic_commitment/TASK_SPLIT_E11.json'),'splits':e11split,'discovery_status':'E11-A preflight used 32 states from these 8 tasks. May be used for future discovery only under a new protocol.','validation_status':'UNTOUCHED by E8/E11; do not read outcomes or use design.','heldout_status':'UNTOUCHED by E8/E11; do not read outcomes or use design.'}},
        'semantic_sensitivity': {
            'frozen_model':p('reports/semantic_commitment/FROZEN_SENSITIVITY_MODEL.json'),'checksum':frozen['checksum_sha256'],'features':89,
            'feature_order_source':p('reports/semantic_commitment/FROZEN_SENSITIVITY_MODEL.json')+'::feature_names','extractor':p('experiments/semantic_risk/run_semantic_shard.py')+'::run_route/reducer/action_geometry','fit':p('experiments/semantic_risk/analyze_semantic_risk.py')+'::fit_predict','reconstruction':p('experiments/semantic_commitment/freeze_sensitivity.py'),'target':'log1p(causal_sensitivity_target), ridge lambda=10, original post-selection refit on E4 discovery+validation (352 rows), output expm1(linear).','route_validity':'P1_ONLY_SENSITIVITY for commitment. Do not apply it at F1 anchor without a new discovery protocol.'
        },
        'e11': {'final_decision':p('reports/semantic_commitment/SEMANTIC_COMMITMENT_FINAL_DECISION.json'),'report':p('reports/semantic_commitment/SEMANTIC_COMMITMENT_SERVER_REPORT_ZH.md'),'route_transfer':p('reports/semantic_commitment/E11A_ROUTE_TRANSFER.json'),'runtime':p('reports/semantic_commitment/E11A_RUNTIME_RESULT.json'),'raw_preflight':str(ROOT/'artifacts/semantic_commitment/e11a_route_transfer.parquet'),'result':e11,'e11b_e11c':'NOT RUN; zero E11 validation/heldout outcome reads.'},
        'reports':[
            {'experiment':'PV0 full-scale / 600 route episodes','decision':'mechanism GO; universal Fresh replacement NO-GO','report':p('reports/pv0_overnight/FINAL_PV0_REPORT_ZH.md'),'raw':'/data/rxhuang/wam_server_deep_validation and /data/rxhuang/wam_full_scale_server','split':p('reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl')},
            {'experiment':'fixed R1/R2/R3 + matched stale/execution feedback','decision':'fixed R2 candidate only; adaptive feedback NO-GO','report':p('reports/pv0_execution_feedback/fixed_reuse_pilot/FIXED_REUSE_PILOT_REPORT_ZH.md'),'raw':p('reports/pv0_execution_feedback'),'split':'experiment-local manifests/reports'},
            {'experiment':'ESP E1/E2/E3','decision':'ESP_NO_GO','report':p('reports/esp/ESP_SERVER_REPORT_ZH.md'),'raw':p('reports/esp/shards'),'split':p('reports/esp/TASK_SPLIT.json')},
            {'experiment':'Semantic risk E4/E5/E6','decision':'SEMANTIC_RISK_NO_GO','report':p('reports/semantic_risk/SEMANTIC_RISK_SERVER_REPORT_ZH.md'),'raw':str(ROOT/'artifacts/semantic_risk/e4_e5_e6_raw.parquet'),'split':p('reports/semantic_risk/TASK_SPLIT.json')},
            {'experiment':'Sensitivity horizon E8/E9/E10','decision':'SENSITIVITY_HORIZON_NO_GO: invalid age sweep','report':p('reports/sensitivity_horizon/SENSITIVITY_HORIZON_SERVER_REPORT_ZH.md'),'raw':'none; formal samples=0','split':p('reports/sensitivity_horizon/TASK_SPLIT_E8.json')},
            {'experiment':'Semantic commitment E11','decision':'SEMANTIC_COMMITMENT_NO_GO at F1 route-transfer gate','report':p('reports/semantic_commitment/SEMANTIC_COMMITMENT_SERVER_REPORT_ZH.md'),'raw':str(ROOT/'artifacts/semantic_commitment/e11a_route_transfer.parquet'),'split':p('reports/semantic_commitment/TASK_SPLIT_E11.json')},
        ],
        'scripts': [
            {'purpose':'restore MuJoCo/controller state safely','path':p('experiments/progressive_wam/run_p2_oracle.py')+'::restore'},
            {'purpose':'replay a recorded request and render observation','path':p('experiments/esp/run_e2_e3_pilot_shard.py')+'::load_request/render'},
            {'purpose':'state capture','path':p('experiments/server_deep_validation/run_server_f1_collection.py')+'::collect_episode'},
            {'purpose':'route primitive/contracts','path':p('experiments/server_deep_validation/pv0_overnight_common.py')+'::route_contract/run_route'},
            {'purpose':'closed-loop runner and trace contract','path':p('experiments/server_deep_validation/run_pv0_closed_loop_episode.py')},
            {'purpose':'multi-GPU PV0 supervisor','path':p('experiments/server_deep_validation/pv0_overnight_supervisor.py')},
            {'purpose':'fixed reuse supervisor','path':p('experiments/server_deep_validation/pv0_fixed_reuse_pilot_supervisor.py')},
            {'purpose':'existing ESP smoke (NOT RUN during handoff)','path':p('experiments/esp/run_esp_smoke.py')},
        ],
        'safe_commands':[
            'cd /home/rxhuang/Projects/cosmos-policy && source .venv/bin/activate && export PYTHONPATH=.',
            'git status --short && git log -12 --oneline && git branch --show-current',
            'nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader',
            'sed -n "1,240p" reports/handoff/CLAUDE_CODE_HANDOFF.md',
            'python -m py_compile experiments/semantic_commitment/*.py  # syntax only; no inference',
            'PYTHONPATH=. .venv/bin/python experiments/semantic_commitment/freeze_sensitivity.py  # deterministic E4 artifact reconstruction; rewrites frozen JSON with same contents; NOT needed during handoff',
            '# Do not run model/simulator smoke or formal runners until a new research protocol explicitly authorizes it.'
        ],
        'do_not_touch':[
            'E11 validation tasks and heldout tasks in TASK_SPLIT_E11.json: untouched clean confirmation resource.',
            'FROZEN_SENSITIVITY_MODEL.json and prior formal result tables: do not modify/refit/reselect in place.',
            'Original checkpoint and original state banks: never overwrite or replace with SO101/finetuned checkpoint.',
            'Existing NO-GO branches: ESP/camera scheduler, E5 main estimator, S×I, illegal K!=16 age labels, P1 E4 score at F1 anchor, action/hidden/x0 patch, fixed R3+ final method, denoise scheduler/value/deep scheduler.',
            'Untracked artifacts/ and logs/: user/run artifacts; preserve, do not add/delete blindly. Logs include failed E11-A OSMesa/OOM attempts and successful EGL records.'
        ],
        'unknowns':[
            {'question':'Exact physical simulator control dt, distinct from logical control_frequency_hz=20 and video fps metadata.','evidence_paths':[p('experiments/libero_harness.py'),p('experiments/server_deep_validation/server_sweep.yaml')],'status':'[REQUIRES_RUNTIME_VERIFICATION or source-level simulator audit]'},
            {'question':'Whether an independently designed F1-route internal signal can be made valid for a future commitment protocol.','evidence_paths':[p('reports/semantic_commitment/E11A_ROUTE_TRANSFER.json'),p('artifacts/semantic_commitment/e11a_route_transfer.parquet')],'status':'not answered; must remain discovery-only before touching E11 validation/heldout'},
            {'question':'Where/why F1 and P1 score-target calibration differs despite S_F1/S_P1 rank rho 0.955.','evidence_paths':[p('reports/semantic_commitment/E11A_ROUTE_TRANSFER.json'),p('experiments/semantic_commitment/run_e11a_route_transfer.py')],'status':'observed only; no causal explanation established'},
            {'question':'Exact component-level kernel/D2H attribution of the original 5.25 ms E4 overhead.','evidence_paths':[p('reports/semantic_risk/E4_COST_RESULT.json'),p('experiments/semantic_risk/profile_e4_hook.py')],'status':'not completed; do not infer from 20-repeat native preflight'}
        ]
    }
    write(REPORT/'CLAUDE_HANDOFF.json', handoff)
    table='\n'.join(f"| {x['experiment']} | {x['decision']} | [{Path(x['report']).name}]({x['report']}) | `{x['raw']}` | `{x['split']}` |" for x in handoff['reports'])
    task_list='\n'.join(f"- {kind}:\n"+'\n'.join(f"  - `{t}`" for t in tasks) for kind,tasks in e11split.items())
    markdown=f'''# Claude Code Handoff — Cosmos WAM / LIBERO

Generated from main commit `{main_sha}`. This is an execution-context handoff, not authorization to run a new study. Start by preserving the clean E11 validation/heldout split and by reading the frozen negative results.

## 1. Executive Handoff Summary

Main repo: `{ROOT}` on branch `research/pv0-closed-loop-feedback-20260812`, commit `{main_sha}`. Remote is `git@github.com:hrx20000209/cosmos-policy.git`. The worktree is intentionally dirty only through untracked runtime artifacts/logs:

```
{dirty}
```

Do not delete, reset, or casually add them. The last result is **SEMANTIC_COMMITMENT_NO_GO**, caused by E11-A F1-route transfer failure, not by an illegal temporal comparison. E11-B/C and E12 were not run. The 24 E11 tasks retain 8 discovery / 8 validation / 8 heldout task-disjoint hygiene; E11 validation and heldout have no sampled outcomes.

## 2. Current Scientific State

The stable positive systems mechanism is PV0: reuse the generated joint latent while inserting a causal fresh visual prefix into current condition slots before the single denoiser forward. Full scale result: F1 19/200, P1 11/200, PV0 17/200; PV0→P1 paired wins/losses 6/0. This supports *recovery of predicted reuse*, not universal Fresh replacement.

Frozen NO-GO / stop boundaries: physical EEF/proprio-only feedback; action amplification; hidden/x0/sparse patches; ESP finite difference and camera scheduler/per-camera selective VAE; raw optical innovation as primary estimator; S×I; illegal prediction-age labels; fixed R3+ as final method; denoise/value/deep scheduler; direct use of the old P1 E4 score at an F1 action anchor. Do not revive these without an entirely new protocol.

Semantic Risk E4 found P1 internal+action score heldout rho 0.651 vs action-only 0.563, but the 89-summary implementation cost 5.25 ms / 2.13% F1. E5 was weak; E6 product lost to additive. Sensitivity Horizon E8 stopped because K!=16 compares a t+16 predicted latent to wrong physical time. E11 correctly avoided that: its planned remaining-action oracle would be temporally legal, but it requires an F1-anchor signal. On 32 new discovery-only states, frozen `S_F1` vs the legal t+16 retrospective target was task-balanced rho 0.175 (<0.50). Thus this estimator is **P1_ONLY_SENSITIVITY** for commitment.

## 3. Repository Map

| Role | Path | Revision / status |
| --- | --- | --- |
| Main Cosmos WAM | `{ROOT}` | `research/pv0-closed-loop-feedback-20260812` / `{main_sha}` |
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

The model is 28 DiT blocks, hidden width 2048. The 9 temporal VAE slots are `{json.dumps(handoff['model_contract']['slots'], ensure_ascii=False)}`. The policy returns `A_t` of shape 16×7 and future visual slots 6/7. In `cosmos_policy/datasets/libero_dataset.py`, `next_relative_step_idx = relative_step_idx + chunk_size`; config sets chunk_size=16. Thus `A_t[0:16]` and `z_hat_(t+16)` are bound to the same temporal horizon.

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

{task_list}

Discovery was touched only by E11-A’s 32 rows (`artifacts/semantic_commitment/e11a_route_transfer.parquet`). **Validation and heldout are UNTOUCHED**: do not inspect their outcome artifacts, run scorers on them, or use them for feature/model/threshold choices before a new protocol freezes a candidate. Older ESP/E4 splits are historical formal resources, not fresh confirmation; paths are `reports/esp/TASK_SPLIT.json` and `reports/semantic_risk/TASK_SPLIT.json`.

## 10. E4 Semantic Sensitivity Implementation

Original extraction: `experiments/semantic_risk/run_semantic_shard.py::run_route`. It installs `model.intermediate_feature_ids` for blocks `(4,8,12,16,20,24,27)` (zero-indexed IDs passed as block-1). The reducer observes each selected post-block tensor and emits 12 scalar terms: action-slot RMS/dispersion; future wrist/primary RMS; current wrist/primary RMS; and six pooled cosine relations between action/current/future slots. That makes 84 internal scalars. `action_geometry` supplies five action terms: `action_norm`, `endpoint_displacement`, `action_curvature`, `action_jerk`, `gripper_transition`.

`reports/semantic_commitment/FROZEN_SENSITIVITY_MODEL.json` serializes exact feature names/order, mean, scale, coefficients, intercept, target transform and checksum `{frozen['checksum_sha256']}`. It was reconstructed by `experiments/semantic_commitment/freeze_sensitivity.py` from the original 352 discovery+validation E4 rows, with historical prediction max absolute error `{frozen['reconstruction_validation']['max_abs_prediction_error_vs_e4_saved_predictions']:.3g}`. Fit code: `experiments/semantic_risk/analyze_semantic_risk.py::fit_predict`, ridge λ=10, target `log1p(causal_sensitivity_target)`, output `expm1(linear)`.

All 89 features are mechanically definable on either F1 or P1 because the same passive reducer/action geometry runs on both. Observed, not explained: on E11 discovery preflight S_F1/S_P1 rank rho was 0.955, yet S_F1 vs legal target task-balanced rho was 0.175. Do not infer a causal explanation not present in data.

## 11. E11 Route-Transfer Result

Read `reports/semantic_commitment/SEMANTIC_COMMITMENT_SERVER_REPORT_ZH.md`, `E11A_ROUTE_TRANSFER.json`, and `SEMANTIC_COMMITMENT_FINAL_DECISION.json`. Artifact reconstruction passed. Discovery-only preflight used 8 clean-split discovery tasks × 4 states = 32 states. Results: S_F1 vs S_P1 pooled Spearman 0.9553; S_F1 vs legal t+16 retrospective target task-balanced Spearman 0.175; S_P1 equivalent 0.350. Preregistered F1 transfer gate was >=0.50. Therefore the score is `P1_ONLY_SENSITIVITY` for commitment. E11-B’s remaining-plan oracle would be temporally legal, but was not authorized; E11-C and E12 are not authorized.

## 12. Runtime Scorer Implementation

Original passive summary path is `run_semantic_shard.py::run_route`; it materializes each selected block’s 12 values with `.detach().cpu().tolist()` after timed `get_action`. Original clean E4 profile is `reports/semantic_risk/E4_COST_RESULT.json`: P1 73.61 ms, P1+89 summaries 78.87 ms, +5.25 ms (2.13% clean F1), so runtime NO-GO under <1% gate.

Native preflight: `experiments/semantic_commitment/profile_native_scorer.py::native_route` weights each block’s 12 GPU reductions and returns seven scalar contributions, then applies the exact `expm1` transform. On 12 states/20 repeats: action max error 0; score max abs error 1.075e-5 (slightly over 1e-5 preflight tolerance due to floating summation order); baseline 77.61 ms, original 84.44 ms, native 85.05 ms. This is **not** a formal 100-repeat profile, no component kernel/D2H breakdown was completed, and it did not reduce cost.

## 13. Existing Experiment / Artifact Index

| Experiment | Decision | Report | Raw Data | Split | Notes |
| --- | --- | --- | --- | --- | --- |
{table}

## 14. Safe Discovery Resources

Safe under a future, explicit protocol: the eight E11 discovery tasks above; Foundation V2 registry (with provenance check); `artifacts/semantic_commitment/e11a_route_transfer.parquet`; prior raw tables as historical evidence; frozen original checkpoint; F1/P1/PV0 route code; state restore/replay helpers. New experiments must not silently reuse historic heldout tasks as “new confirmation.”

## 15. Clean Validation / Heldout Resources

E11 validation and heldout tasks are reserved clean resources, as listed above. They have no E8/E11 outcome artifacts. Do not run even a “quick” model probe that creates outcome labels, because that consumes the task-disjoint confirmation resource. Only inspect split metadata during handoff.

## 16. Existing Smoke / Unit Tests

Relevant existing smoke/preflight files: `experiments/esp/run_esp_smoke.py`, `experiments/preflight_wam_libero.py`, `experiments/server_deep_validation/run_native_persistent_condition_preflight.py`, and `experiments/server_deep_validation/run_modular_worker_preflight.py`. **NOT RUN DURING THIS HANDOFF**: they can load models/simulators and consume GPU. Safe no-inference checks are `python -m py_compile experiments/semantic_commitment/*.py`, JSON schema inspection, and `git diff --check`.

## 17. GPU / Job Scheduler

Existing multi-GPU mechanisms: `experiments/server_deep_validation/pv0_overnight_supervisor.py` (args `--run-dir --manifest --collection-root --ablation-root --duration-hours --poll-seconds`), `pv0_fixed_reuse_pilot_supervisor.py` (`--manifest --run-dir --gpu-ids --poll-seconds`), `execution_validity_shadow_supervisor.py` (`--manifest --run-dir --poll-seconds`), plus `server_queue_supervisor.py` and `server_stage_supervisor.py`. Workers conventionally set `CUDA_VISIBLE_DEVICES`, `EVAL_PHYSICAL_GPU`, memory fraction and write resumable per-shard JSON/log outputs. Query free VRAM at launch; never kill other users. In E11-A, GPU 1/2 failed model construction under shared-memory pressure; GPU 5/6 worked with explicit EGL.

## 18. Logging and Output Conventions

Reports live under `reports/<experiment>/`; raw/parquet under `artifacts/<experiment>/`; plots under `reports/<experiment>/plots/`; resumable shard JSON under `reports/<experiment>/shards/`; logs usually under `logs/` or report-local `logs/`. JSON reports use `{{status, checkpoint, checkpoint_sha256, denoising_steps, value_used, finetuning_used, gpu, rows/...}}`. State IDs are `<episode_hash>:req<index>`. Task ID is `suite:task_name`. Keep one row per paired unit, include split, seed, state/observation hashes, `valid`, and `invalid_reason`; do not discard failures silently.

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
'''
    write(REPORT/'CLAUDE_CODE_HANDOFF.md', markdown)
    print(json.dumps({'markdown':str(REPORT/'CLAUDE_CODE_HANDOFF.md'),'json':str(REPORT/'CLAUDE_HANDOFF.json')}))


if __name__ == '__main__':
    main()
