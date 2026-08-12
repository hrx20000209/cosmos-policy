# ESP Phase-0 codebase audit

## Model and inference entry

`CosmosAdapter.infer` calls `get_action`, which calls `CosmosPolicyVideo2WorldModel.generate_samples_from_batch`. The one-step sampler invokes `x0_fn`, then `model.denoise`; `denoise` constructs the video-conditioned tensor and invokes `self.net` at `policy_video2world_model.py:401-471`.

The loaded original LIBERO checkpoint instantiates `MinimalV1LVGDiT`: 28 transformer blocks, hidden size 2048, 16 heads, BF16, `minimal_a2a` attention. The base transformer is `MiniTrainDIT`; its exact block loop is in `minimal_v4_dit.py:1715-1863`.

## Condition path

Video evidence is injected before block 0: `denoise` overwrites `net_state_in` with `condition.gt_frames` where the video-condition mask is one (`policy_video2world_model.py:420-446`). It is not re-injected at each block. Every block performs AdaLN-modulated self-attention over unified spatio-temporal tokens, text cross-attention, and an MLP with residual connections (`minimal_v4_dit.py:1131-1350`). Q/K/V are projected by `Attention.compute_qkv`; RoPE applies to self-attention, not text cross-attention (`minimal_v4_dit.py:390-590`).

## ESP legality

Condition-slot finite differences are legal because existing PV0 already uses native `condition.gt_frames.index_copy_` before the sole denoiser forward. ESP must construct E0/E^c through this condition mechanism, never by hidden/x0 patching. The existing intermediate feature API is passive, but it continues through the full suffix. Consequently shallow capture is source-legal; cheap early-exit is **not yet confirmed** and is a mandatory smoke-test gate.

## E1 architecture decision

Current visual slots are independently replaceable, but the VAE is not camera-compute-separable for LIBERO. It encodes one joint temporal video; primary evidence requires prior temporal frames. Therefore `E1_SELECTIVE_SENSING_ARCH_NO_GO`. This does not invalidate E2 causal importance or E3 offline sensitivity validation, but it prohibits claims of selective per-camera VAE savings on this layout.
