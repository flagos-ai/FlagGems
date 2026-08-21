# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from flag_gems.ops._fused_adam import _fused_adam as _generic_fused_adam

logger = logging.getLogger(__name__)


def _fused_adam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    eps=1e-8,
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    logger.debug("GEMS_KUNLUNXIN FUSED_ADAM")
    kernel_params = []
    kernel_grads = []
    kernel_exp_avgs = []
    kernel_exp_avg_sqs = []
    kernel_max_exp_avg_sqs = []
    kernel_state_steps = []

    for index, param in enumerate(params):
        # A single-program 1024-element launch miscomputes index zero on XPU.
        # Preserve the known-good 512-element launches only for this small case.
        chunks = param.reshape(-1).split(512) if param.numel() <= 1024 else [param]
        kernel_params.extend(chunks)
        kernel_grads.extend(
            grads[index].reshape(-1).split(512)
            if param.numel() <= 1024
            else [grads[index]]
        )
        kernel_exp_avgs.extend(
            exp_avgs[index].reshape(-1).split(512)
            if param.numel() <= 1024
            else [exp_avgs[index]]
        )
        kernel_exp_avg_sqs.extend(
            exp_avg_sqs[index].reshape(-1).split(512)
            if param.numel() <= 1024
            else [exp_avg_sqs[index]]
        )
        if max_exp_avg_sqs:
            kernel_max_exp_avg_sqs.extend(
                max_exp_avg_sqs[index].reshape(-1).split(512)
                if param.numel() <= 1024
                else [max_exp_avg_sqs[index]]
            )
        kernel_state_steps.extend([state_steps[index]] * len(chunks))

    _generic_fused_adam(
        kernel_params,
        kernel_grads,
        kernel_exp_avgs,
        kernel_exp_avg_sqs,
        kernel_max_exp_avg_sqs,
        kernel_state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
    return params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs


def _fused_adam_(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    eps=1e-8,
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    logger.debug("GEMS_KUNLUNXIN FUSED_ADAM_")
    _fused_adam(
        self,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
    return None
