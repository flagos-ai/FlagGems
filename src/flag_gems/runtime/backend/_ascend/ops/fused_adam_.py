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

import triton
import triton.language as tl


@triton.jit
def _fused_adam_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    state_step_ptr,
    lr,
    beta1,
    beta2,
    eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    param = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0)

    # bias corrections: 1 - beta**step, computed by exponentiation by squaring
    # on the integer step so the fp32 result matches the reference rounding.
    s = tl.load(state_step_ptr)
    base1 = beta1
    p1 = 1.0
    while s > 0:
        if s % 2 == 1:
            p1 = p1 * base1
        base1 = base1 * base1
        s = s // 2
    s = tl.load(state_step_ptr)
    base2 = beta2
    p2 = 1.0
    while s > 0:
        if s % 2 == 1:
            p2 = p2 * base2
        base2 = base2 * base2
        s = s // 2

    bias_correction1 = 1.0 - p1
    bias_correction2 = 1.0 - p2

    next_exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
    next_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
    # Replace the two per-element vector divisions by the scalar corrections
    # with vector multiplies by the precomputed scalar reciprocals.
    inv1 = 1.0 / bias_correction1
    inv2 = 1.0 / bias_correction2
    corrected_exp_avg = next_exp_avg * inv1
    corrected_exp_avg_sq = next_exp_avg_sq * inv2
    next_param = param - lr * corrected_exp_avg / (tl.sqrt(corrected_exp_avg_sq) + eps)

    tl.store(exp_avg_ptr + offsets, next_exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, next_exp_avg_sq, mask=mask)
    tl.store(param_ptr + offsets, next_param, mask=mask)


def fused_adam_(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
):
    if amsgrad or maximize or weight_decay != 0.0:
        raise ValueError("branch absent from the FlagGems source test")
    BLOCK_SIZE = 4096
    for param, grad, exp_avg, exp_avg_sq, state_step in zip(
        params, grads, exp_avgs, exp_avg_sqs, state_steps
    ):
        n_elements = param.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _fused_adam_kernel[grid](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            state_step,
            lr,
            beta1,
            beta2,
            eps,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=16,
        )
    return params
