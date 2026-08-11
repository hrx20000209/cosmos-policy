# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extended Sampler for Cosmos Policy with special handling for num_steps.

This sampler extends the base Sampler class to add:
- Adjusted num_steps logic when sample_clean is enabled
- Special case handling for num_steps==1
"""

from typing import Callable, List, Optional

import torch

from cosmos_policy._src.imaginaire.functional.multi_step import is_multi_step_fn_supported
from cosmos_policy._src.imaginaire.functional.runge_kutta import is_runge_kutta_fn_supported
from cosmos_policy._src.imaginaire.modules.res_sampler import (
    Sampler,
    SamplerConfig,
    SolverConfig,
    SolverTimestampConfig,
    differential_equation_solver,
    get_rev_ts,
)


class CosmosPolicySampler(Sampler):
    """
    Extended Sampler for Cosmos Policy.

    Adds special handling for:
    - Adjusting num_steps when sample_clean is enabled (subtracts 1 for num_steps > 1)
    - Special case for num_steps==1 where we directly denoise without the solver loop
    """

    def __init__(self, cfg: Optional[SamplerConfig] = None):
        super().__init__(cfg)

    @torch.no_grad()
    def forward(
        self,
        x0_fn: Callable,
        x_sigma_max: torch.Tensor,
        num_steps: int = 35,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        solver_option: str = "2ab",
    ) -> torch.Tensor:
        in_dtype = x_sigma_max.dtype
        denoiser_forward_index = 0

        def float64_x0_fn(x_B_StateShape: torch.Tensor, t_B: torch.Tensor) -> torch.Tensor:
            nonlocal denoiser_forward_index
            pre_hook = getattr(self, "pre_denoise_hook", None)
            if pre_hook is not None:
                pre_hook(
                    denoiser_forward_index=denoiser_forward_index,
                    solver_state=x_B_StateShape,
                    sigma=t_B,
                )
            timing_events = getattr(self, "step_timing_events", None)
            if timing_events is None:
                output = x0_fn(x_B_StateShape.to(in_dtype), t_B.to(in_dtype))
            else:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output = x0_fn(x_B_StateShape.to(in_dtype), t_B.to(in_dtype))
                end_event.record()
                timing_events.append((start_event, end_event))
            transform = getattr(self, "x0_transform", None)
            if transform is not None:
                transformed = transform(
                    denoiser_forward_index=denoiser_forward_index,
                    solver_state=x_B_StateShape,
                    sigma=t_B,
                    predicted_clean=output,
                )
                if transformed.shape != output.shape:
                    raise ValueError(
                        f"x0_transform changed shape {tuple(output.shape)} -> {tuple(transformed.shape)}"
                    )
                output = transformed.to(device=output.device, dtype=output.dtype)
            denoiser_forward_index += 1
            return output.to(torch.float64)

        is_multistep = is_multi_step_fn_supported(solver_option)
        is_rk = is_runge_kutta_fn_supported(solver_option)
        assert is_multistep or is_rk, f"Only support multistep or Runge-Kutta method, got {solver_option}"

        solver_cfg = SolverConfig(
            s_churn=S_churn,
            s_t_max=S_max,
            s_t_min=S_min,
            s_noise=S_noise,
            is_multi=is_multistep,
            rk=solver_option,
            multistep=solver_option,
        )
        if num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}")
        requested_num_steps = num_steps
        # The final clean prediction is a full denoiser call. Reserve one NFE
        # for it so the public num_steps value always equals actual forwards.
        sample_clean = True
        if sample_clean and num_steps > 1:
            num_steps = num_steps - 1
        timestamps_cfg = SolverTimestampConfig(nfe=num_steps, t_min=sigma_min, t_max=sigma_max, order=rho)
        sampler_cfg = SamplerConfig(solver=solver_cfg, timestamps=timestamps_cfg, sample_clean=sample_clean)

        # Progressive-denoising instrumentation hook. Callers may set
        # `sampler.checkpoint_hook` to a callable to observe every denoiser
        # forward. When it is unset the value is None, `if callback_fns:`
        # short-circuits everywhere below, and the control flow is identical
        # to the un-instrumented path.
        hook = getattr(self, "checkpoint_hook", None)
        callback_fns = [hook] if hook is not None else None

        return self._forward_impl(
            float64_x0_fn,
            x_sigma_max,
            sampler_cfg,
            callback_fns=callback_fns,
            requested_num_steps=requested_num_steps,
        ).to(in_dtype)

    @torch.no_grad()
    def _forward_impl(
        self,
        denoiser_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noisy_input_B_StateShape: torch.Tensor,
        sampler_cfg: Optional[SamplerConfig] = None,
        callback_fns: Optional[List[Callable]] = None,
        requested_num_steps: int = 35,
    ) -> torch.Tensor:
        """
        Internal implementation of the forward pass.

        Args:
            denoiser_fn: Function to denoise the input.
            noisy_input_B_StateShape: Input tensor with noise.
            sampler_cfg: Configuration for the sampler.
            callback_fns: List of callback functions to be called during sampling.
            requested_num_steps: Public number of denoiser forwards requested.

        Returns:
            torch.Tensor: Denoised output tensor.
        """
        sampler_cfg = self.cfg if sampler_cfg is None else sampler_cfg
        solver_order = 1 if sampler_cfg.solver.is_multi else int(sampler_cfg.solver.rk[0])
        num_timestamps = sampler_cfg.timestamps.nfe // solver_order

        sigmas_L = get_rev_ts(
            sampler_cfg.timestamps.t_min, sampler_cfg.timestamps.t_max, num_timestamps, sampler_cfg.timestamps.order
        ).to(noisy_input_B_StateShape.device)

        if requested_num_steps > 1:
            # Normal sampling
            denoised_output = differential_equation_solver(
                denoiser_fn, sigmas_L, sampler_cfg.solver, callback_fns=callback_fns
            )(noisy_input_B_StateShape)

            if sampler_cfg.sample_clean:
                # Override denoised_output with fully denoised version
                ones = torch.ones(denoised_output.size(0), device=denoised_output.device, dtype=denoised_output.dtype)
                solver_state = denoised_output
                denoised_output = denoiser_fn(solver_state, sigmas_L[-1] * ones)
                if callback_fns:
                    # The terminal clean call is a full denoiser forward too, so
                    # emit it as the final checkpoint. Keeping the same keyword
                    # names as the solver's `callback_fn(**locals())` lets one
                    # hook serve both call sites.
                    for callback_fn in callback_fns:
                        callback_fn(
                            i_th=len(sigmas_L) - 1,
                            input_x_B_StateShape=solver_state,
                            sigma_cur_0=sigmas_L[-1],
                            sigma_next_0=sigmas_L[-1],
                            x0_pred_B_StateShape=denoised_output,
                            output_x_B_StateShape=denoised_output,
                            x0_preds=None,
                            is_terminal_clean=True,
                        )
        else:
            # Special case: a one-step policy request is one direct x0 call.
            denoised_output = noisy_input_B_StateShape
            ones = torch.ones(denoised_output.size(0), device=denoised_output.device, dtype=denoised_output.dtype)
            solver_state = denoised_output
            denoised_output = denoiser_fn(solver_state, sigmas_L[0] * ones)
            if callback_fns:
                for callback_fn in callback_fns:
                    callback_fn(
                        i_th=0,
                        input_x_B_StateShape=solver_state,
                        sigma_cur_0=sigmas_L[0],
                        sigma_next_0=sigmas_L[0],
                        x0_pred_B_StateShape=denoised_output,
                        output_x_B_StateShape=denoised_output,
                        x0_preds=None,
                        is_terminal_clean=True,
                    )

        return denoised_output
