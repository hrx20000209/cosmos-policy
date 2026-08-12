# PV0 Prefix-VAE Architecture Audit

## Conclusion

PV0 is a native **condition-slot replacement**, not a hidden activation patch. It starts from the prior generated joint latent, encodes only the causal prefix needed to recover the current wrist/primary visual slots, and writes those slots into the condition before denoiser forward 0. No trainable module is introduced.

## Why 13 frames

The policy temporal VAE uses a 4-frame compression factor. The source builds the first 13 raw frames as one structural blank plus four duplicated frames each for proprio, wrist image, and primary image. That yields latent slots 0–3; PV0 selects only current visual slots 2 and 3.

## Latent slot layout

- 0: structural blank
- 1: current proprio
- 2: current wrist image
- 3: current primary image
- 4: action chunk
- 5: future proprio
- 6: future wrist image
- 7: future primary image
- 8: value slot (never read by this experiment)

PV0 leaves current proprio and every predicted/action/future/value slot inherited from the previous generated latent. The value slot is structurally present in the model layout but is never read by this experiment.

## Source evidence

- Temporal compression factor: `cosmos_policy/experiments/robot/cosmos_utils.py:52`.
- Prefix-only VAE encode: `cosmos_policy/experiments/robot/cosmos_utils.py:1364`.
- Condition `gt_frames` slot replacement: `cosmos_policy/experiments/robot/cosmos_utils.py:1317`.
- Arrival 0 for native persistent route: `adapters/cosmos_adapter.py:233`.
- Condition hook occurs before the denoiser: `cosmos_policy/models/policy_video2world_model.py:787`.
- Nine-slot policy layout: `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py:129`.

## Guardrails

- Original pre-finetune checkpoint only; `denoise=1`.
- No Cosmos value, simulator state runtime input, scheduler, threshold, hidden patch, learned head, or dynamic denoise.
- The static audit establishes interface semantics; Phase-A/Phase-B artifacts establish numerical and closed-loop behavior.
