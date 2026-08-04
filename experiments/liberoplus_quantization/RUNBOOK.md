# RUNBOOK — LIBERO-Plus WAM Quantization Experiment

Branch: `exp/liberoplus-quantization` (repo: `~/Projects/cosmos-policy`)
Heavy data (raw/logs/profiles) symlinked to `/data/rxhuang/liberoplus_quantization/` (/home is 96% full).

## Hardware / Software (see system_info.txt)
- 8× RTX 4090 D, 24 GB, sm_89 (Ada Lovelace). FP8/INT8/INT4 real kernels: YES. NVFP4: NO.
- cosmos-policy `.venv`: python 3.10.12, torch 2.7.0+cu128, transformers 4.57.1, peft 0.18.0.
- torchao/tensorrt/modelopt/bitsandbytes: NOT installed at audit time.

## Precision plan (Ada-appropriate)
- A. BF16 baseline (official, no quant)
- B. Medium: FP8 backbone (torchao float8, Ada tensor cores) — real HW accel
- C. Aggressive: INT4 weight-only (torchao tinygemm/Marlin) — real HW accel (weight-bandwidth)
- D. Sensitivity: fake-int4 broader coverage — hardware_accelerated=false, sensitivity only

## Log
| # | time (HKT) | action | result | next |
|---|-----------|--------|--------|------|
| 1 | 2026-07-29 01:41 | Phase 0 audit: GPU, sw versions, repos, disk | done — 8×4090D sm_89; torchao missing; LIBERO-Plus missing | install LIBERO-Plus |
| 2 | 2026-07-29 01:44 | Created exp dir + /data symlinks + system_info.txt + branch | done | verify Cosmos venv + LIBERO-Plus |
| 3 | 2026-07-29 01:50 | Cloned LIBERO-plus -> /data/rxhuang/LIBERO-plus; read README/reqs | done — drop-in, robosuite 1.4.0 (matches cosmos lock!), needs assets.zip from HF | install after baseline |
| 4 | 2026-07-29 01:52 | Started `uv sync --extra cu128 --group libero` (UV_CACHE_DIR=/data) + HF checkpoint download (HF_HOME=/data) | running in bg | run sanity check |
| 5 | 2026-07-29 01:55 | Built pilot task list (360 tasks, 36 strata x10, seed195) -> task_lists/liberoplus_pilot_seed195.json | done | use in Phase 4 |
| 6 | 2026-07-29 02:00 | Read cosmos_utils: inject quant at get_model on `model.net` (DiT); latency wraps `model.generate_samples_from_batch` | mapped | implement Phase 3 |

## Key architecture notes (for Phase 3/8)
- `get_model(cfg)` -> (model, config); model is Cosmos Predict2 diffusion wrapper. DiT = `model.net`. VAE/text-embed cache separate.
- Denoising forward = `model.generate_samples_from_batch(data_batch, num_steps=num_denoising_steps_action, ...)`. This is the dominant cost; wrap with CUDA events.
- Eval uses multi-GPU worker pool (`WorkerPoolManager`/`query_model_parallel`), `num_trials_per_task` (=1 for LIBERO-Plus). Success-rate eval can use the pool; precise single-GPU latency uses a dedicated 1-process harness.
- Quant plan on Ada: quantize large Linear in `model.net` (attention qkv/o, FFN); keep norms/softmax/timestep/VAE/action-decode in bf16.

| 7 | 2026-07-29 09:55 | Fixed hung HF checkpoint dl (Xet finalize hang); re-downloading .pt via curl to /data/rxhuang/models/cosmos-policy-libero-2b | in progress | — |
| 8 | 2026-07-29 09:56 | robosuite /tmp/robosuite.log PermissionError (shared host, owned by hlliu) -> set FILE_LOGGING_LEVEL=None in robosuite/macros.py [venv patch] | fixed | — |
| 9 | 2026-07-29 10:05 | Phase 1 min env test: mesa EGL blocked (not in video/render group). Forced NVIDIA EGL (10_nvidia.json, /dev/nvidia* world-rw). ENV_TEST_OK | DONE | Phase 2 |

## CRITICAL run knobs (every LIBERO run)
- `MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json` (see env.sh) + per-proc `MUJOCO_EGL_DEVICE_ID=<gpu> CUDA_VISIBLE_DEVICES=<gpu>`.
- Checkpoint LOCAL dir: `/data/rxhuang/models/cosmos-policy-libero-2b/` (Cosmos-Policy-LIBERO-Predict2-2B.pt + config.json + libero_dataset_statistics.json + libero_t5_embeddings.pkl). Pass this as `--ckpt_path` to avoid re-download.
- **Eval parallelism**: `run_libero_eval` worker pool = best-of-N sampling (uses `available_gpus[:num_queries_best_of_n]`), NOT task-parallel. One process => 1 GPU, episodes serial (`for task_id in range(num_tasks)`, line 901). To use 8 GPUs: shard tasks across 8 processes. This also gives clean single-GPU latency for Phase 8.
- Plan: Phase 2 runs stock `run_libero_eval.py` (verified) for baseline SR; Phase 3+ uses a custom instrumented sharded driver mirroring `run_episode`.

## Phase 4 LIBERO-Plus drop-in setup (done)
1. Gated base model `nvidia/Cosmos-Predict2-2B-Video2World` needed HF token (user provided; acct hrx2000). Downloaded model-480p-16fps.pt (3.9GB)+tokenizer.pth(508MB) via curl -> HF cache (Xet finalize hangs -> HF_HUB_DISABLE_XET=1).
2. ckpt_path MUST be the .pt FILE (loader treats dir as DCP -> appends /model). Use .../Cosmos-Policy-LIBERO-Predict2-2B.pt.
3. LIBERO-Plus: clone sylvestf/LIBERO-plus; assets.zip 6.4GB (deep path inspire/.../assets) extracted+mv to $LP/libero/libero/assets (457k files).
4. Missing top-level libero/__init__.py in repo -> copied empty one from PyPI wheel cache; `uv pip install -e LIBERO-plus --no-deps`; repoint ~/.libero/config.yaml.
5. env_wrapper needs `wand`+libMagickWand (no sudo) -> conda imagemagick to /data/rxhuang/envs/imagemagick; env.sh sets MAGICK_HOME + appends LD_LIBRARY_PATH.
6. torch 2.7 weights_only=True breaks LIBERO-Plus init-state torch.load -> patched benchmark/__init__.py lines 186,242 -> weights_only=False.
7. Verified: libero_10 perturbed n_tasks=2519; perturbed env creates+renders (agentview/wrist 256).
torchao 0.11.0 (torch2.7-compatible; 0.17 needs torch>=2.11).

## Phase 3b microbench findings (EAGER, GPU0 4090, denoise=5, batch=1, action-only)
| mode | denoise p50 | vs bf16 | peak mem | action MSE vs bf16 | cosine | hw-accel |
|---|---|---|---|---|---|---|
| bf16 | 424.9 ms | 1.00× | 8145 MB | 0 | 1.0 | baseline |
| fp8_backbone | 998.4 ms | 2.35× SLOWER | 12357 MB | 3.9e-5 | 0.99995 | yes(_scaled_mm) |
| int8_backbone | 2458.9 ms | 5.79× SLOWER | 14793 MB | 1.4e-5 | 0.99996 | yes(int8 mm) |
| int4_weight_only | (pending) | | | | | |
| fake_int4 | (pending) | | | | | |

KEY: at batch=1 with small per-step GEMMs, torchao EAGER dynamic-quant overhead (cast+scale
per matmul) dominates -> slowdown, higher mem. Numerics excellent (cosine>0.9999).
NEXT: fair latency needs torch.compile (fuses quant+matmul). Must test compiled fp8/int8/int4
before any speedup conclusion. Record BOTH eager and compiled.

## Open questions / risks
- Cosmos checkpoint not cached; needs HF download (~2B model).
- LIBERO-Plus must be installed as drop-in; verify it does not break the existing LIBERO env used by cosmos venv.
- Whether cosmos `.venv` sees LIBERO (installed in a separate conda env? check import path).

## 2026-07-29 interface audit and unified evaluator

- The original custom `driver_liberoplus.py` did not complete an episode. The
  first non-language perturbation appended `tb 2` to the task language, missed
  the checkpoint's 40-entry T5 cache, and entered an on-demand T5-11B load.
- `run_libero_plus_eval.py` now resolves non-language implementation suffixes
  to the longest exact original instruction prefix. Genuine language
  perturbations require exact paraphrase embeddings and fail before policy
  loading if absent.
- Fixed upstream open-loop semantics: `deque(maxlen=N).extend(chunk)` retained
  the last N actions. The evaluator now executes `chunk[:N]`.
- Added per-episode, per-chunk and per-step JSONL, initial-state hashing,
  CUDA-event timings, clean process memory, optional NVML power integration,
  resume, timeout, dynamic replanning and dynamic denoising.
- Added typed inherited YAML configs and sweep launchers for BF16/FP16,
  INT8/INT4, branch proxies, open-loop, denoising and dynamic strategies.
- Dry-run resolves the four representative LIBERO-Plus tasks without T5 cache
  misses. Framework unit tests: 7 passed.
- No new rollout was launched after the fix because all eight GPUs were occupied
  by an unrelated 8-GPU job (~19.2 GB/card, 100% utilization). Aggregation
  therefore reports zero completed LIBERO-Plus episodes; missing plots are
  labeled rather than imputed.

## Unattended GPU monitor

- Script: `monitor_and_run.py`
- Active PID: see `logs/monitor/status.json` (launched in an independent session).
- Poll: every 60 seconds.
- Idle gate: at least 12 GiB free and at most 10% utilization for three
  consecutive checks; uses at most four GPUs because the smoke list has four tasks.
- Pipeline: BF16 gate → FP16/INT8/INT4 → branch proxies → BF16 open-loop;
  INT8 open-loop runs only if its four-episode pilot is within one failure of BF16.
- Every config re-enters the idle gate, so a new external job between configs is
  not raced.
- Stop gracefully before the next config/check:
  `touch experiments/liberoplus_quantization/logs/monitor/monitor.stop`
