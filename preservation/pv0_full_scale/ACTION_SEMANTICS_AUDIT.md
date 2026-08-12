# Action semantics audit — PV0 execution-feedback runtime

## Frozen contract

- Model: `Cosmos-Policy-LIBERO-Predict2-2B`, original pre-finetune checkpoint
  `8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2`.
- Horizon: 16; action dimension: 7; denoising: exactly one forward.
- Environment: LIBERO-PRO `OffScreenRenderEnv`, whose default controller is
  `OSC_POSE`.

## Interpretation

The first six action coordinates are the relative operational-space
end-effector command consumed by OSC_POSE: Cartesian translation plus
orientation increment.  Coordinate seven is the gripper command.  The action
is a command increment, not an absolute robot pose and not joint position.

Accordingly, a nominal trajectory is formed by accumulating the first six
commands from the observed EEF pose.  The future feedback loop compares its
observed EEF displacement against the corresponding nominal increment.  The
gripper is recorded separately and is never scaled.

## Permitted control operations

- Retiming / selecting the nearest nominal waypoint from EEF proprioception.
- Measuring EEF tracking residual, progress residual, arm direction cosine,
  boundary discontinuity, and gripper switching.
- If and only if a separately frozen action-progress gate passes, scaling the
  first six coordinates on the P1 route.  The current offline oracle failed
  that gate, so no scale control is enabled.

## Explicitly excluded

No runtime component reads object pose, contact, reward, success, task
predicate, semantic stage, ground-truth subgoal, Cosmos value, or a finetuned
SO101 checkpoint.
