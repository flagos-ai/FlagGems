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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def _round_to_bfloat16(value):
    # Match PyTorch bf16 operator boundaries without allowing XPU lowering to fuse
    # intermediate calculations at fp32 precision.
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def _margin_ranking_loss_elementwise(input1, input2, target, margin):
    if input1.dtype == tl.bfloat16:
        input1 = input1.to(tl.float32)
        input2 = input2.to(tl.float32)
        target = target.to(tl.float32)
        difference = _round_to_bfloat16(input1 - input2)
        product = _round_to_bfloat16(-target * difference)
        loss = _round_to_bfloat16(product + margin)
        return tl.maximum(loss, 0.0)
    return tl.maximum(-target * (input1 - input2) + margin, 0.0)


@libentry()
@triton.jit
def _margin_ranking_loss_kernel(
    input1,
    input2,
    target,
    out,
    n_elements,
    margin,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input1_val = tl.load(input1 + offsets, mask=mask, other=0.0)
    input2_val = tl.load(input2 + offsets, mask=mask, other=0.0)
    target_val = tl.load(target + offsets, mask=mask, other=0.0)
    margin_val = tl.full([BLOCK_SIZE], margin, input1_val.dtype)
    zero = tl.zeros([BLOCK_SIZE], input1_val.dtype)
    difference = (input1_val - input2_val).to(input1_val.dtype)
    product = (-target_val * difference).to(input1_val.dtype)
    loss = tl.maximum((product + margin_val).to(input1_val.dtype), zero)
    tl.store(out + offsets, loss, mask=mask)


@libentry()
@triton.jit
def _margin_ranking_loss_partial_sum_kernel(
    input1,
    input2,
    target,
    partial_sums,
    n_elements,
    margin,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input1_val = tl.load(input1 + offsets, mask=mask, other=0.0)
    input2_val = tl.load(input2 + offsets, mask=mask, other=0.0)
    target_val = tl.load(target + offsets, mask=mask, other=0.0)
    if input1_val.dtype == tl.bfloat16:
        input1_val = input1_val.to(tl.float32)
        input2_val = input2_val.to(tl.float32)
        target_val = target_val.to(tl.float32)
        difference = _round_to_bfloat16(input1_val - input2_val)
        product = _round_to_bfloat16(-target_val * difference)
        loss = tl.maximum(_round_to_bfloat16(product + margin), 0.0)
    else:
        margin_val = tl.full([BLOCK_SIZE], margin, input1_val.dtype)
        zero = tl.zeros([BLOCK_SIZE], input1_val.dtype)
        difference = (input1_val - input2_val).to(input1_val.dtype)
        product = (-target_val * difference).to(input1_val.dtype)
        loss = tl.maximum((product + margin_val).to(input1_val.dtype), zero)
    tl.store(
        partial_sums + pid, tl.sum(tl.where(mask, loss, 0.0).to(tl.float32), axis=0)
    )


@libentry()
@triton.jit
def _margin_ranking_loss_finalize_kernel(
    partial_sums,
    out,
    n_elements,
    n_partials,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    partial = tl.load(partial_sums + offsets, mask=offsets < n_partials, other=0.0)
    total = tl.sum(partial, axis=0)
    if reduction == 1:
        total = total / n_elements
    tl.store(out, total)


@libentry()
@triton.jit
def _margin_ranking_loss_backward_kernel(
    input1,
    input2,
    target,
    grad_output,
    grad_input1,
    grad_input2,
    n_elements,
    margin,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input1_val = tl.load(input1 + offsets, mask=mask, other=0.0)
    input2_val = tl.load(input2 + offsets, mask=mask, other=0.0)
    target_val = tl.load(target + offsets, mask=mask, other=0.0)
    if REDUCTION == 0:
        grad = tl.load(grad_output + offsets, mask=mask, other=0.0)
    else:
        grad = tl.load(grad_output)
        if REDUCTION == 1:
            grad = grad / n_elements

    if input1_val.dtype == tl.bfloat16:
        input1_float = input1_val.to(tl.float32)
        input2_float = input2_val.to(tl.float32)
        target_float = target_val.to(tl.float32)
        difference = _round_to_bfloat16(input1_float - input2_float)
        product = _round_to_bfloat16(-target_float * difference)
        active = _round_to_bfloat16(product + margin) > 0.0
    else:
        difference = (input1_val - input2_val).to(input1_val.dtype)
        product = (-target_val * difference).to(input1_val.dtype)
        active = (product + margin).to(input1_val.dtype) > 0.0

    grad_input1_val = tl.where(active, -target_val * grad, 0.0)
    grad_input2_val = tl.where(active, target_val * grad, 0.0)
    tl.store(grad_input1 + offsets, grad_input1_val.to(input1_val.dtype), mask=mask)
    tl.store(grad_input2 + offsets, grad_input2_val.to(input1_val.dtype), mask=mask)


def _margin_ranking_loss_forward(input1, input2, target, margin, reduction):
    logger.debug("GEMS_KUNLUNXIN MARGIN_RANKING_LOSS")
    if reduction not in (0, 1, 2):
        raise ValueError("reduction must be 0 (none), 1 (mean), or 2 (sum)")

    input1, input2, target = torch.broadcast_tensors(input1, input2, target)
    output_shape = input1.shape
    input1 = input1.contiguous().view(-1)
    input2 = input2.contiguous().view(-1)
    target = target.contiguous().view(-1)
    n_elements = input1.numel()
    block_size = 2048

    if reduction == 0:
        return _margin_ranking_loss_elementwise(
            input1, input2, target, float(margin)
        ).view(output_shape)

    n_partials = triton.cdiv(n_elements, block_size)
    partial_sums = torch.empty((n_partials,), dtype=torch.float32, device=input1.device)
    out = torch.empty((), dtype=input1.dtype, device=input1.device)
    final_block_size = triton.next_power_of_2(n_partials)
    with torch_device_fn.device(input1.device):
        _margin_ranking_loss_partial_sum_kernel[(n_partials,)](
            input1,
            input2,
            target,
            partial_sums,
            n_elements,
            float(margin),
            BLOCK_SIZE=block_size,
            buffer_size_limit=2048,
        )
        _margin_ranking_loss_finalize_kernel[(1,)](
            partial_sums,
            out,
            n_elements,
            n_partials,
            reduction,
            BLOCK_SIZE=final_block_size,
            buffer_size_limit=2048,
        )
    return out


class _MarginRankingLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, target, margin, reduction):
        input1, input2, target = torch.broadcast_tensors(input1, input2, target)
        ctx.save_for_backward(
            input1.contiguous(), input2.contiguous(), target.contiguous()
        )
        ctx.margin = float(margin)
        ctx.reduction = reduction
        return _margin_ranking_loss_forward(input1, input2, target, margin, reduction)

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, target = ctx.saved_tensors
        grad_input1 = torch.empty_like(input1)
        grad_input2 = torch.empty_like(input2)
        n_elements = input1.numel()
        block_size = 2048
        with torch_device_fn.device(input1.device):
            _margin_ranking_loss_backward_kernel[
                (triton.cdiv(n_elements, block_size),)
            ](
                input1,
                input2,
                target,
                grad_output.contiguous(),
                grad_input1,
                grad_input2,
                n_elements,
                ctx.margin,
                REDUCTION=ctx.reduction,
                BLOCK_SIZE=block_size,
                buffer_size_limit=2048,
            )
        return grad_input1, grad_input2, None, None, None


def margin_ranking_loss(input1, input2, target, margin=0.0, reduction=1):
    if reduction not in (0, 1, 2):
        raise ValueError("reduction must be 0 (none), 1 (mean), or 2 (sum)")
    return _MarginRankingLoss.apply(input1, input2, target, margin, reduction)
