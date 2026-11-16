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

import functools
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Unmasked stage-1 tile: the largest tl.sum tile exact on this XPU with
# buffer_size_limit=2048 (same as mse_loss's _FULL_BLOCK).
_FULL_BLOCK = 32768
# Tensors up to this size use the single-launch mean/sum kernel.
_SMALL_MAX = 4096


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
def _margin_ranking_loss_partial_sum_unmasked_kernel(
    input1,
    input2,
    target,
    partial_sums,
    n_elements,
    margin,
    BLOCK_SIZE: tl.constexpr,
):
    # Unmasked stage-1 program over a full BLOCK_SIZE (=32768) lane block.
    # Requires n_elements % BLOCK_SIZE == 0 on the host; skips the masked
    # memory path entirely (masked 2048-lane tiles run ~40x slower than the
    # vendor engine on big shapes). tl.sum at 32768 lanes is only complete
    # with buffer_size_limit=2048 enforced at launch (kunlunxin reduction
    # ceiling, same as mse_loss).
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input1_val = tl.load(input1 + offsets)
    input2_val = tl.load(input2 + offsets)
    target_val = tl.load(target + offsets)
    if input1_val.dtype == tl.bfloat16:
        input1_val = input1_val.to(tl.float32)
        input2_val = input2_val.to(tl.float32)
        target_val = target_val.to(tl.float32)
        loss = tl.maximum(-target_val * (input1_val - input2_val) + margin, 0.0)
    else:
        difference = (input1_val - input2_val).to(input1_val.dtype)
        product = (-target_val * difference).to(input1_val.dtype)
        loss = tl.maximum((product + margin).to(input1_val.dtype), 0.0)
    tl.store(partial_sums + pid, tl.sum(loss.to(tl.float32), axis=0))


@libentry()
@triton.jit
def _margin_ranking_loss_small_kernel(
    input1,
    input2,
    target,
    out,
    n_elements,
    margin,
    REDUCTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Single-launch mean/sum for small tensors (n_elements <= _SMALL_MAX),
    # where the two-launch partial+finalize path is launch-bound.
    offsets = tl.arange(0, BLOCK_SIZE)
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
    total = tl.sum(tl.where(mask, loss, 0.0).to(tl.float32), axis=0)
    if REDUCTION == 1:
        total = total / n_elements
    tl.store(out, total)


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


# Backward is a pure memory-bound flat elementwise kernel.  Geometry sweep on
# XPU3: the masked load/store path costs ~40% even when the mask is all-true,
# so MASKED is a constexpr chosen host-side (only tails take the slow path).
# 32K is the bandwidth plateau for every dtype (64K is never faster and is
# ~3% slower for bf16); below _BWD_TINY a single program beats any grid.
_BWD_TINY = 4096
_BWD_BLOCK_MIN = 8192
_BWD_BLOCK_MAX = 32768


@functools.lru_cache(maxsize=1024)
def _pick_backward_geometry(n_elements):
    # Memoized: the pure-python next_power_of_2/cdiv path costs ~5.5us per call,
    # which dominates the launch floor for tiny shapes in KERNEL-mode
    # benchmarks (the kernel itself runs in ~6us at n=4096).
    if n_elements <= _BWD_TINY:
        block = max(256, triton.next_power_of_2(max(n_elements, 1)))
    else:
        # ~8 programs: wide tiles without starving the SMs on tiny grids.
        block = triton.next_power_of_2((n_elements + 7) // 8)
        block = min(_BWD_BLOCK_MAX, max(_BWD_BLOCK_MIN, block))
    masked = (n_elements % block) != 0
    grid = max(1, triton.cdiv(n_elements, block))
    return block, grid, masked


@libentry()
@triton.jit
def _margin_ranking_loss_backward_kernel(
    input1,
    input2,
    target,
    grad_output,
    grad_input1,
    grad_input2,
    grad_target,
    n_elements,
    margin,
    REDUCTION: tl.constexpr,
    WRITE_TARGET: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if MASKED:
        # Strict "<" against the runtime bound (never "<="): XPU3 can fold a
        # non-strict runtime tail mask away and store the full tile OOB
        # (wiki pattern-xpu3-masked-store-constexpr-tail).
        mask = offsets < n_elements
        input1_val = tl.load(input1 + offsets, mask=mask, other=0.0)
        input2_val = tl.load(input2 + offsets, mask=mask, other=0.0)
        target_val = tl.load(target + offsets, mask=mask, other=0.0)
        if REDUCTION == 0:
            grad = tl.load(grad_output + offsets, mask=mask, other=0.0)
        else:
            grad = tl.load(grad_output)
    else:
        input1_val = tl.load(input1 + offsets)
        input2_val = tl.load(input2 + offsets)
        target_val = tl.load(target + offsets)
        if REDUCTION == 0:
            grad = tl.load(grad_output + offsets)
        else:
            grad = tl.load(grad_output)
    if REDUCTION != 0:
        # Keep the upstream scalar in fp32 for the divide; the gradient itself
        # is produced in fp32 below.
        grad = grad.to(tl.float32)
        if REDUCTION == 1:
            grad = grad / n_elements

    # Evaluate the active mask and the gradients entirely in fp32.  Two XPU3
    # traps are avoided this way: a half-precision tl.where/scalar-multiply is
    # miscompiled (fp16/bf16 gradients come out zero or with the mask inverted,
    # and the failure depends on the surrounding kernel body), and every
    # tl.where branch must share one element type
    # (wiki language-xpu3-tl-where-scalar-literal-select-type-mismatch).
    # Bit-exact vs ATen on fp16/bf16/fp32 (harness/perf_ir/probe_mrl_bwd_formulations).
    input1_32 = input1_val.to(tl.float32)
    input2_32 = input2_val.to(tl.float32)
    target_32 = target_val.to(tl.float32)
    grad_32 = grad.to(tl.float32)
    active = (-target_32 * (input1_32 - input2_32) + margin) > 0.0

    zero = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    grad_input1_val = tl.where(active, -target_32 * grad_32, zero).to(input1_val.dtype)
    grad_input2_val = tl.where(active, target_32 * grad_32, zero).to(input1_val.dtype)
    if MASKED:
        tl.store(grad_input1 + offsets, grad_input1_val, mask=mask)
        tl.store(grad_input2 + offsets, grad_input2_val, mask=mask)
        if WRITE_TARGET:
            # d/dtarget = (input2 - input1) where the loss is active.
            grad_target_val = tl.where(
                active, (input2_32 - input1_32) * grad_32, zero
            ).to(input1_val.dtype)
            tl.store(grad_target + offsets, grad_target_val, mask=mask)
    else:
        tl.store(grad_input1 + offsets, grad_input1_val)
        tl.store(grad_input2 + offsets, grad_input2_val)
        if WRITE_TARGET:
            grad_target_val = tl.where(
                active, (input2_32 - input1_32) * grad_32, zero
            ).to(input1_val.dtype)
            tl.store(grad_target + offsets, grad_target_val)


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

    if n_elements <= _SMALL_MAX:
        # Tiny tensors: single-launch mean/sum (two launches are launch-bound).
        small_block_size = max(triton.next_power_of_2(n_elements), 1)
        out = torch.empty((), dtype=input1.dtype, device=input1.device)
        with torch_device_fn.device(input1.device):
            _margin_ranking_loss_small_kernel[(1,)](
                input1,
                input2,
                target,
                out,
                n_elements,
                float(margin),
                REDUCTION=reduction,
                BLOCK_SIZE=small_block_size,
                buffer_size_limit=2048,
            )
        return out

    if n_elements >= _FULL_BLOCK and n_elements % _FULL_BLOCK == 0:
        # Fully divisible tensors take the unmasked 32768-lane stage-1 path
        # (block DMA instead of the masked-memory path, which runs ~40x
        # slower than the vendor engine on the big shapes). Masked 2048-lane
        # tiles are kept for everything else.
        block_size = _FULL_BLOCK
        partial_sum_kernel = _margin_ranking_loss_partial_sum_unmasked_kernel
        n_partials = n_elements // _FULL_BLOCK
    else:
        partial_sum_kernel = _margin_ranking_loss_partial_sum_kernel
        n_partials = triton.cdiv(n_elements, block_size)
    partial_sums = torch.empty((n_partials,), dtype=torch.float32, device=input1.device)
    out = torch.empty((), dtype=input1.dtype, device=input1.device)
    final_block_size = triton.next_power_of_2(n_partials)
    with torch_device_fn.device(input1.device):
        partial_sum_kernel[(n_partials,)](
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
        # The benchmark/autograd harness may mark ``target`` (a float tensor)
        # as requiring grad; autograd.grad then demands a non-None gradient for
        # it.  ATen's margin_ranking_loss_backward does produce d/dtarget, so
        # record whether the caller actually needs it.
        ctx.needs_target_grad = bool(target.requires_grad)
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
        n_elements = input1.numel()

        if n_elements == 0:
            grad_input1 = torch.empty_like(input1)
            grad_input2 = torch.empty_like(input2)
            grad_target = torch.empty_like(target) if ctx.needs_target_grad else None
            return grad_input1, grad_input2, grad_target, None, None

        grad_input1 = torch.empty_like(input1)
        grad_input2 = torch.empty_like(input2)
        grad_target = torch.empty_like(target) if ctx.needs_target_grad else None

        block_size, grid, masked = _pick_backward_geometry(n_elements)
        target_ptr = grad_target if ctx.needs_target_grad else grad_input1
        with torch_device_fn.device(input1.device):
            _margin_ranking_loss_backward_kernel[(grid,)](
                input1,
                input2,
                target,
                grad_output.contiguous(),
                grad_input1,
                grad_input2,
                target_ptr,
                n_elements,
                ctx.margin,
                REDUCTION=ctx.reduction,
                WRITE_TARGET=ctx.needs_target_grad,
                MASKED=masked,
                BLOCK_SIZE=block_size,
                buffer_size_limit=2048,
            )
        return grad_input1, grad_input2, grad_target, None, None


def _normalize_reduction(reduction):
    # Accept both string and int/enum forms: 0=none, 1=mean, 2=sum.
    if isinstance(reduction, str):
        name = reduction.lower()
        if name in ("none", "mean", "sum"):
            return {"none": 0, "mean": 1, "sum": 2}[name]
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def margin_ranking_loss(input1, input2, target, margin=0.0, reduction=1):
    reduction = _normalize_reduction(reduction)
    return _MarginRankingLoss.apply(input1, input2, target, margin, reduction)
