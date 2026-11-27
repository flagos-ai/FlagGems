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

"""GCU400-specific BCEWithLogits.

Default path uses fused 1D flat kernels: elementwise for reduction=none, and a
single-pass elementwise+reduction kernel (which never materializes the loss
tensor) for reduction=mean/sum. Other shapes/weights use the common kernels.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_PROGRAMS = 48
BLOCK_SIZE = 8192
# Tuned on GCU400 (kernel-mode do_bench, 2^30-element sweep): fp32 streams
# fastest with a 4096-wide tile (2.0 TB/s); fp16/bf16 need the wider 16384
# tile to keep enough bytes in flight per iteration (1.19 TB/s).
REDUCE_BLOCK_SIZE_32 = 4096
REDUCE_BLOCK_SIZE_16 = 16384
REDUCE_NUM_WARPS = 4
FINALIZE_BLOCK = 64


def _reduce_block_size(input_tensor):
    if input_tensor.element_size() >= 4:
        return REDUCE_BLOCK_SIZE_32
    return REDUCE_BLOCK_SIZE_16


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _bce_logits_flat_kernel(
    input_ptr,
    target_ptr,
    output_ptr,
    num_elements,
    BLOCK: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(program_id, tl.cdiv(num_elements, BLOCK), num_programs):
        element_offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        valid_mask = element_offsets < num_elements
        logits = tl.load(input_ptr + element_offsets, mask=valid_mask, other=0).to(
            tl.float32
        )
        targets = tl.load(target_ptr + element_offsets, mask=valid_mask, other=0).to(
            tl.float32
        )
        # Numerically stable form: max(x,0) - x*y + log(1 + exp(-|x|))
        loss = (
            tl.maximum(logits, 0.0)
            - logits * targets
            + tl.log(1.0 + tl.exp(-tl.abs(logits)))
        )
        tl.store(output_ptr + element_offsets, loss, mask=valid_mask)


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _bce_logits_reduce_kernel(
    input_ptr,
    target_ptr,
    partial_ptr,
    out_ptr,
    num_elements,
    BLOCK: tl.constexpr,
    IS_MEAN: tl.constexpr,
    SINGLE_PROGRAM: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    # Wide lane-parallel accumulator: no cross-lane reduction inside the loop
    # (a full-width tl.sum per iteration serializes the load pipeline with
    # barriers); the single reduction happens once after the loop. Per-lane
    # sequential depth stays tiny, so fp32 error is ~1e-7 relative even at
    # 2^31 elements.
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    num_full_blocks = num_elements // BLOCK
    # Mask-free main loop over full blocks; only the single tail block
    # (< BLOCK elements) needs masking, and program 0 covers it alone.
    for block_id in tl.range(program_id, num_full_blocks, num_programs):
        # 1D contiguous offsets: the 2D (row, col) offset form wraps when the
        # byte offset passes 2^31 (e.g. fp32 numel 2^30) on this backend.
        element_offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        logits = tl.load(input_ptr + element_offsets).to(tl.float32)
        targets = tl.load(target_ptr + element_offsets).to(tl.float32)
        # Numerically stable form: max(x,0) - x*y + log(1 + exp(-|x|))
        loss = (
            tl.maximum(logits, 0.0)
            - logits * targets
            + tl.log(1.0 + tl.exp(-tl.abs(logits)))
        )
        acc += loss
    tail_start = num_full_blocks * BLOCK + program_id * BLOCK
    if tail_start < num_elements:
        element_offsets = tail_start + tl.arange(0, BLOCK)
        valid_mask = element_offsets < num_elements
        logits = tl.load(input_ptr + element_offsets, mask=valid_mask, other=0).to(
            tl.float32
        )
        targets = tl.load(target_ptr + element_offsets, mask=valid_mask, other=0).to(
            tl.float32
        )
        loss = (
            tl.maximum(logits, 0.0)
            - logits * targets
            + tl.log(1.0 + tl.exp(-tl.abs(logits)))
        )
        # Tail lanes load 0/0 and would contribute log(2); zero them out.
        loss = tl.where(valid_mask, loss, 0.0)
        acc += loss
    total = tl.sum(acc, axis=0)
    if SINGLE_PROGRAM:
        if IS_MEAN:
            total = total / num_elements.to(tl.float32)
        tl.store(out_ptr, total)
    else:
        tl.store(partial_ptr + program_id, total)


@libentry()
@triton.jit(do_not_specialize=["num_elements", "num_partials"])
def _bce_reduce_finalize_kernel(
    partial_ptr,
    out_ptr,
    num_elements,
    num_partials,
    IS_MEAN: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_MID)
    valid_mask = offsets < num_partials
    partials = tl.load(partial_ptr + offsets, mask=valid_mask, other=0.0)
    total = tl.sum(partials, axis=0)
    if IS_MEAN:
        total = total / num_elements.to(tl.float32)
    tl.store(out_ptr, total)


def _can_use_fused(input_tensor, target, weight, pos_weight):
    return (
        weight is None
        and pos_weight is None
        and input_tensor.is_contiguous()
        and target.is_contiguous()
        and input_tensor.shape == target.shape
    )


def _launch_fused(input_tensor, target, output_tensor):
    num_elements = input_tensor.numel()
    if num_elements == 0:
        return output_tensor
    grid_size = min(triton.cdiv(num_elements, BLOCK_SIZE), NUM_PROGRAMS)
    with torch_device_fn.device(input_tensor.device):
        _bce_logits_flat_kernel[(grid_size,)](
            input_tensor,
            target,
            output_tensor,
            num_elements,
            BLOCK=BLOCK_SIZE,
            num_warps=4,
        )
    return output_tensor


def _launch_fused_reduction(input_tensor, target, reduction):
    """Elementwise loss + reduction in one pass: never materializes the loss tensor."""
    num_elements = input_tensor.numel()
    is_mean = reduction == 1
    if num_elements == 0:
        if is_mean:
            return torch.full(
                [], float("nan"), dtype=input_tensor.dtype, device=input_tensor.device
            )
        return torch.zeros([], dtype=input_tensor.dtype, device=input_tensor.device)
    block = _reduce_block_size(input_tensor)
    grid_size = min(triton.cdiv(num_elements, block), NUM_PROGRAMS)
    out = torch.empty([], dtype=input_tensor.dtype, device=input_tensor.device)
    with torch_device_fn.device(input_tensor.device):
        if grid_size == 1:
            _bce_logits_reduce_kernel[(1,)](
                input_tensor,
                target,
                out,
                out,
                num_elements,
                BLOCK=block,
                IS_MEAN=is_mean,
                SINGLE_PROGRAM=True,
                num_warps=REDUCE_NUM_WARPS,
            )
        else:
            partial = torch.empty(
                (grid_size,), dtype=torch.float32, device=input_tensor.device
            )
            _bce_logits_reduce_kernel[(grid_size,)](
                input_tensor,
                target,
                partial,
                out,
                num_elements,
                BLOCK=block,
                IS_MEAN=is_mean,
                SINGLE_PROGRAM=False,
                num_warps=REDUCE_NUM_WARPS,
            )
            _bce_reduce_finalize_kernel[(1,)](
                partial,
                out,
                num_elements,
                grid_size,
                IS_MEAN=is_mean,
                BLOCK_MID=FINALIZE_BLOCK,
                num_warps=1,
            )
    return out


def _common_elementwise(self, target, weight, pos_weight, out0=None):
    """Keep the common implementation's weight / pos_weight / broadcast semantics; all paths remain Triton kernels."""
    from flag_gems.ops.binary_cross_entropy_with_logits import (
        binary_cross_entropy_with_logits_kernel,
        binary_cross_entropy_with_logits_pos_weight_kernel,
        binary_cross_entropy_with_logits_weight_kernel,
        binary_cross_entropy_with_logits_weight_pos_weight_kernel,
    )

    if weight is not None and pos_weight is not None:
        kernel = binary_cross_entropy_with_logits_weight_pos_weight_kernel
        args = (self, target, weight, pos_weight)
    elif weight is not None:
        kernel = binary_cross_entropy_with_logits_weight_kernel
        args = (self, target, weight)
    elif pos_weight is not None:
        kernel = binary_cross_entropy_with_logits_pos_weight_kernel
        args = (self, target, pos_weight)
    else:
        kernel = binary_cross_entropy_with_logits_kernel
        args = (self, target)
    if out0 is None:
        return kernel(*args)
    return kernel(*args, out0=out0)


def _apply_reduction(output_tensor, reduction):
    # reduction: 0=NONE, 1=MEAN, 2=SUM; matches the common implementation.
    if reduction == 1:
        return output_tensor.mean()
    if reduction == 2:
        return output_tensor.sum()
    return output_tensor


def binary_cross_entropy_with_logits(
    self, target, weight=None, pos_weight=None, reduction=1
):
    logger.debug("GEMS_ENFLAME BINARY_CROSS_ENTROPY_WITH_LOGITS")
    if _can_use_fused(self, target, weight, pos_weight):
        if reduction == 1 or reduction == 2:
            return _launch_fused_reduction(self, target, reduction)
        out = torch.empty_like(self)
        _launch_fused(self, target, out)
        return out
    out = _common_elementwise(self, target, weight, pos_weight)
    return _apply_reduction(out, reduction)


def binary_cross_entropy_with_logits_(
    self, target, weight=None, pos_weight=None, reduction=1
):
    logger.debug("GEMS_ENFLAME BINARY_CROSS_ENTROPY_WITH_LOGITS_")
    if _can_use_fused(self, target, weight, pos_weight):
        _launch_fused(self, target, self)
        out = self
    else:
        out = _common_elementwise(self, target, weight, pos_weight, out0=self)
    return _apply_reduction(out, reduction)
