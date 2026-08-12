# E12 P1-Native Semantic Verify-and-Correct — Pre-Experiment Audit

Audit performed before any GPU job. Base commit `009af72` on
`research/pv0-closed-loop-feedback-20260812` (handoff commit `95f1c26` plus the
handoff-document commit). Worktree dirty only through the untracked
`artifacts/` and `logs/` runtime material named in the handoff; nothing was
deleted, reset, or added.

## 1. Why this round is causally legal

E11 froze `P1_ONLY_SENSITIVITY`: the E4 score does **not** transfer to an F1
anchor (`S_F1` vs legal t+16 target task-balanced rho 0.175 < 0.50 gate). This
round never places the score at an F1 anchor. It reads the score **out of the
P1 forward that produced the speculative action being judged**, and asks only
whether that same forward's semantics predict that same forward's own action
risk. The estimator and the judged object are route-identical, so no route
transfer is claimed or required.

## 2. Frozen scorer reproducibility — PASS (offline, no GPU)

Independent reconstruction from `FROZEN_SENSITIVITY_MODEL.json` applied to
`artifacts/semantic_risk/e4_e5_e6_raw.parquet`:

| check | result |
| --- | --- |
| `checksum_sha256` recomputed over the model body | matches `3299a5ec…6d5be` |
| max abs error vs saved E4 predictions, 352 train rows | `3.15e-14` |
| max abs error vs saved E4 predictions, 160 heldout rows (not used in fit) | `3.82e-14` |
| feature count / order | 89 = 84 `internal_b{4,8,12,16,20,24,27}_{0..11}` + 5 action-geometry |

The frozen scorer is therefore exactly reproducible in float64 from the
serialized artifact. `experiments/p1_semantic_verify/common.py::frozen_score`
uses the same arithmetic and is the only scoring path in this round. No refit,
no feature selection, no lambda change, no layer change.

## 3. Task-disjointness of the frozen scorer's training data — PASS

The frozen model was fit on 352 rows from the 11 E4 discovery+validation tasks.
Intersection of the 16 E4 tasks with the 24 E11 tasks is **empty**. Every E12
row — discovery, validation and heldout — is therefore a genuinely
task-disjoint test of the frozen scorer, not a re-read of its training set.

## 4. Historical reference for the new primary target (prior formal table only)

E12-A's primary target is raw `R_P1 = mean-step L2(A_P1, A_F1)`, not E4's
innovation-normalized `causal_sensitivity_target`. Re-scoring the historical
E4 table (a prior formal resource, explicitly listed as safe historical
evidence; **no E11 data touched**) establishes what the frozen score did
against each target on E4's own tasks:

| E4 split | n | rho(S, causal_sensitivity_target) | rho(S, raw R_P1) |
| --- | ---: | ---: | ---: |
| discovery | 192 | 0.629 | 0.582 |
| validation | 160 | 0.740 | 0.750 |
| heldout | 160 | 0.651 | 0.642 |

ACTION_ONLY refit on E4 discovery against raw `R_P1`, evaluated on E4 heldout:
**0.563**. So the historical S-over-action-only margin against the *raw* target
is 0.079 — just under this round's preregistered +0.08 improvement gate. That
margin is a historical reference point on E4's tasks, not a prediction for E11
tasks, and the gate is not being relaxed to accommodate it; the gate's
disjunctive second arm (task-hierarchical bootstrap support for a positive
increment) exists precisely for this regime.

Also observed: raw `R_P1` and `causal_sensitivity_target` have pooled Spearman
0.985 on the E4 table, because latent innovation is nearly constant at a fixed
16-action gap. The two targets are close to rank-equivalent; the change of
target is a simplification, not a new estimand.

## 5. Known risk carried into E12-A

E11-A reported `S_P1` vs the legal target at task-balanced rho **0.350** on the
new E11 discovery tasks — well below E4's historical 0.651. Two non-exclusive
explanations exist and E12-A is powered to separate them:

1. **Statistical.** E11-A used 4 states/task. A per-task Spearman over 4 points
   takes values in a coarse discrete set and has enormous variance; averaging 8
   such estimates does not fix it. E12-A uses 16 states/task.
2. **Real transfer loss.** The frozen score may genuinely degrade on the E11
   task family (which is LIBERO-10-heavy in discovery and libero_spatial-heavy
   in heldout, unlike E4's mix).

If discovery at 16 states/task reproduces ~0.35, that is evidence for (2) and
the honest outcome is a NO-GO at the E12-A validation gate. This is recorded
before collection so it cannot be reinterpreted afterwards.

## 6. Determinism and hook-invariance — PASS (historical, re-verified in SMOKE-A/B)

`artifacts/semantic_risk/e4_e5_e6_raw.parquet` carries a 24-state repeat audit:
`p1_repeat_action_l2`, `f1_repeat_action_l2`, `p1_hook_off_action_l2` and
`p1_hook_feature_max_abs_diff` are all exactly `0.0`. The intermediate-feature
reducer is action-invariant and inference is bit-repeatable at fixed seed.
SMOKE-A/B re-establish this on the E12 bank rather than inheriting it.

## 7. Transaction safety — source-level finding

The speculative-P1 transaction does **not** need snapshot/rollback. Source
audit shows the inference path has no durable side effects:

- `cosmos_utils.get_action` takes `seed` explicitly and this round always passes
  `randomize_seed=False`.
- Noise is `misc.arch_invariant_rand(..., seed)`, which builds a fresh
  `np.random.RandomState(seed)` per call. Global torch/numpy RNG is never read.
- `use_variance_scale=False` in `run_p1_trajectory_dump.build_cfg`, so the only
  `torch.manual_seed` call in `policy_text2world_model.generate_samples_from_batch`
  is not reached.
- `inference_condition_transform` and `sampler.x0_transform` are saved and
  restored inside `get_action`; `intermediate_feature_ids/_reducer` are cleared
  in the caller's `finally`.

All mutable route state lives in `CosmosAdapter`: `previous_generated_latent`,
`previous_real_latent`, `last_physical_condition_latent`, `request_index`,
`_shadow_prior_generated_latent`. Therefore the transaction is implemented as
**speculate-without-mutating, commit-on-accept**: the speculative P1 forward
writes nothing, so the reject path is exact by construction rather than by a
restore that could drift. SMOKE-C and SMOKE-D verify accept-path and
reject-path equivalence numerically anyway; the design is not trusted on
argument alone.

## 8. State bank legality — PASS

`reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl` (3801 rows,
40 tasks) is built by `pv0_overnight_common.build_state_index`, which *skips*
any (source, target) pair whose `control_step` difference is not 16.
`run_e2_e3_pilot_shard.load_request` re-asserts the same 16-action gap and the
target `state_key` at load time. Every E12 row is therefore H=16 aligned by
two independent checks. Simulator state is used only by
`run_p2_oracle.restore` + `regenerate_obs_from_state` to reproduce a camera
observation; it never enters the policy.

Per-task aligned-state availability for the 24 E11 tasks (min 25, max 160, 5
stored episodes each) supports 16 states/task everywhere. Selection is
round-robin over episodes with maximal within-episode control-step spacing to
avoid stacking near-duplicate adjacent requests.

## 8b. AUDIT FINDING — the E4/E11-A collection route is not the deployed P1

Found while building SMOKE-B, recorded before any E12 correlation was computed.

`experiments/semantic_risk/run_semantic_shard.py::collect` calls its `run_route`
with `previous=prev`, where `prev` is the **raw** prior generated joint latent.
It never moves predicted future slots 6/7 into current slots 2/3. In that file
`predicted_condition()` is used *only* to compute the innovation denominator,
never to build the action condition. `experiments/semantic_commitment/run_e11a_route_transfer.py`
inherits the same call.

The deployed P1 route — `CosmosAdapter._visual_input_for_request`'s
`predicted_reuse`, `pv0_overnight_common.route_contract("P1")` as used by the
PV0 fidelity/closed-loop line, and `run_e2_e3_pilot_shard.process_state` — does
apply the slot move. So the frozen scorer's 84 internal features were collected
from a *stale-condition* reuse variant, while every deployment claim in this
project is about *predicted* reuse.

Measured divergence on one discovery state (SMOKE-B):

| quantity | value |
| --- | ---: |
| action max abs diff, E4 variant vs deployed P1 | `0.598` |
| internal feature max abs diff | `0.986` |
| frozen score on E4 variant | `1.248` |
| frozen score on deployed P1 | `1.844` |

**Decision, taken before seeing any E12-A result.** E12 uses the **deployed P1**
route for both the speculative action and the score. This is required by the
round's causal contract: the score must be read out of the very forward that
produced the action being judged. Scoring an E4-variant forward while executing
a deployed-P1 action would be exactly the kind of cross-route borrowing E11
ruled out.

Consequence: E12-A is a strictly harder test than a replication. The frozen
estimator is applied to a forward variant it was not collected on, on tasks it
was not fit on. The estimator is still used completely unmodified. The
E4-variant score is recorded as a **discovery-only diagnostic column**
(`s_p1_e4variant`) so that a NO-GO can be attributed between "does not transfer
to new tasks" and "does not survive the route-variant change" — it is *not* a
candidate policy in this round, and cannot become one here.

SMOKE-B also proves the extractor itself is the frozen one: fed the identical
predicted condition, the historical E4 reducer and this round's reducer agree to
`0.0` on all 84 features and `0.0` on the action.

## 8c. AUDIT FINDING — bit-exactness holds within a process, not across processes

SMOKE-A: F1/P1/PV0 each repeat to exactly `0.0`, and the feature hook changes
the action by exactly `0.0`, within one process.

SMOKE-F: the same requests rendered in two separate processes produce identical
observation hashes, so the renderer is reproducible. But driving the identical
frozen observation chain through the identical bootstrap F1 route in two
separate processes produced action differences of order `1e-3` and different
`previous_real_latent` hashes. Within a process the same call repeats exactly.
The behaviour is consistent with per-process GPU kernel/algorithm selection; no
causal explanation is claimed here.

Consequences, both applied:

1. SMOKE-C/D compare the transactional adapter against the reference route
   **inside one process, sharing one resident model object**. That is the
   correct scope for a transaction test anyway. Both then pass bit-exactly
   (`0.0` at every step, identical adapter-state hashes).
2. E12-A collects all three routes for a state inside one process, so the
   paired comparison is unaffected.
3. For E12-C, paired episodes necessarily run in separate processes (as in the
   prior 600-episode PV0 full-scale run). Episode success, not bit-exactness,
   is the outcome measure there; this is stated rather than assumed away.

## 8d. Smoke results

| gate | result |
| --- | --- |
| SMOKE-A paired determinism | PASS — all repeat/hook diffs `0.0`; P1 vs F1 differs by `0.061` so the routes are genuinely distinct |
| SMOKE-B frozen score reconstruction | PASS — extractor equivalence `0.0`, 84 features, score vs independent reference `0.0` |
| SMOKE-C accept path | PASS — bit-exact vs `predicted_reuse` at every step, adapter state identical |
| SMOKE-D reject path | PASS — bit-exact vs `native_persistent` at every step; 2/2 speculative forwards discarded |
| SMOKE-E H=16 alignment | PASS — all 128 discovery rows have gap exactly 16 |
| SMOKE-F render reproducibility | PASS (diagnostic) — 3/3 observation hashes identical across processes |

Artifact: `reports/p1_semantic_verify/SMOKE_RESULT.json`.

## 9. GPU state at audit time

GPUs 1, 5, 6 free (19 MiB each); 0, 2, 3, 4, 7 held by other users' processes
(17–23 GiB). E12 workers will take 5, 6 and 1, with `CUDA_VISIBLE_DEVICES`,
`EVAL_PHYSICAL_GPU`, `MUJOCO_EGL_DEVICE_ID` set explicitly and per-process
memory fraction capped. No other user's process is touched. Formal latency
profiling runs on a single otherwise-idle GPU.

## 10. Boundaries re-affirmed

No F1-anchored use of the E4 score; no refit of the frozen model; no new
estimator family; no hidden/x0/sparse patch; no ESP probe; no camera scheduler;
no S×I; no value; no finetuned checkpoint; no denoise count other than 1; no
K != 16 action commitment; no task-specific thresholds; no feature selection on
validation or heldout. E11 validation is read only after discovery operating
points are frozen; E11 heldout is read only if both E12-A and E12-B are GO.
