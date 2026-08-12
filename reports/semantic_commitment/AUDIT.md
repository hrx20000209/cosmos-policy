# Semantic-commitment Phase-0 / E11-A audit

- Commit `ef8728718f417cc0046dc062a0974bf362f90580`, original pre-finetune checkpoint SHA `8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2`, denoise=1, no value.
- H=16: dataset `next_relative_step_idx = relative_step_idx + chunk_size`; the action chunk and future visual slots are both t+16 targets. `A_orig[K]` is the next relative 7D OSC-POSE delta+gripper command after exactly K executed commands. ActionBuffer executes actions sequentially, one simulator `env.step` per command.
- E11-B's proposed comparison is temporally legal: at state t+K it compares `A_orig[K:K+M]` to F1 `A_fresh[0:M]`, both commands for the same physical state/time. It was not run.
- State restore uses the existing full MuJoCo/controller restore helper: physics state, done/timestep, solver warmstart, `sim.forward`, controller update/reset-goal. EGL was required on this headless server; OSMesa failed before any collection.

## E11-A result

The original E4 post-selection discovery+validation ridge was reconstructed exactly (max saved-prediction error `3.15e-14`) and serialized. On 32 states from the new **discovery-only** 8-task split, S_F1 versus the legal 16-action retrospective target was `0.175`, below the predeclared 0.50 route-transfer gate. S_F1 and S_P1 are rank-consistent (`0.955`) but F1 calibration/predictive validity is insufficient. Therefore frozen E4 is P1-only for this purpose.

E11-B/C, validation, and heldout were not touched.
