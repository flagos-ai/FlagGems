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

TOTAL_CORE_NUM = torch_device_fn.get_device_properties().multi_processor_count
BLOCK_SIZE = 16384

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["numel", "input_offset"])
def slice_backward_range_copy_kernel(
    grad_output_ptr,
    grad_input_ptr,
    numel,
    input_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    loop_step = (num_jobs * BLOCK_SIZE).to(tl.int64)

    for block_start_offset in range(block_start, numel, loop_step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        grad = tl.load(grad_output_ptr + offsets, mask=mask)
        tl.store(grad_input_ptr + input_offset + offsets, grad, mask=mask)


@triton.jit(do_not_specialize=["numel", "input_offset"])
def slice_backward_range_zero_kernel(
    grad_input_ptr,
    numel,
    input_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    loop_step = (num_jobs * BLOCK_SIZE).to(tl.int64)

    for block_start_offset in range(block_start, numel, loop_step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        tl.store(grad_input_ptr + input_offset + offsets, 0.0, mask=mask)


@triton.jit(do_not_specialize=["outer", "segment_numel", "input_offset"])
def slice_backward_outer_copy_kernel(
    grad_output_ptr,
    grad_input_ptr,
    outer,
    segment_numel,
    input_block,
    input_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid_outer_start = tl.program_id(0)
    outer_jobs = tl.num_programs(axis=0)
    pid_block = tl.program_id(1)
    offsets = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < segment_numel

    for pid_outer in range(pid_outer_start, outer, outer_jobs):
        grad = tl.load(grad_output_ptr + pid_outer * segment_numel + offsets, mask=mask)
        tl.store(
            grad_input_ptr + pid_outer * input_block + input_offset + offsets,
            grad,
            mask=mask,
        )


@triton.jit(do_not_specialize=["outer", "segment_numel", "input_offset"])
def slice_backward_outer_zero_kernel(
    grad_input_ptr,
    outer,
    segment_numel,
    input_block,
    input_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid_outer_start = tl.program_id(0)
    outer_jobs = tl.num_programs(axis=0)
    pid_block = tl.program_id(1)
    offsets = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < segment_numel

    for pid_outer in range(pid_outer_start, outer, outer_jobs):
        tl.store(
            grad_input_ptr + pid_outer * input_block + input_offset + offsets,
            0.0,
            mask=mask,
        )


@triton.jit(do_not_specialize=["outer", "inner", "dim_size", "start", "step"])
def slice_backward_strided_copy_kernel(
    grad_output_ptr,
    grad_input_ptr,
    outer,
    inner,
    dim_size,
    start,
    step,
    BLOCK_SIZE: tl.constexpr,
):
    pid_outer_start = tl.program_id(0)
    outer_jobs = tl.num_programs(axis=0)
    slice_idx = tl.program_id(1)
    pid_inner = tl.program_id(2)
    offsets = pid_inner * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < inner
    slice_len = tl.num_programs(axis=1)
    input_dim_idx = start + slice_idx * step

    for pid_outer in range(pid_outer_start, outer, outer_jobs):
        grad = tl.load(
            grad_output_ptr + (pid_outer * slice_len + slice_idx) * inner + offsets,
            mask=mask,
        )
        tl.store(
            grad_input_ptr + (pid_outer * dim_size + input_dim_idx) * inner + offsets,
            grad,
            mask=mask,
        )


@triton.jit
def slice_backward_scatter_kernel(
    grad_output_ptr,
    grad_input_ptr,
    numel,
    inner,
    slice_len,
    dim_size,
    start,
    step,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    loop_step = (num_jobs * BLOCK_SIZE).to(tl.int64)

    for block_start_offset in range(block_start, numel, loop_step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        grad = tl.load(grad_output_ptr + offsets, mask=mask)

        outer_idx = offsets // (slice_len * inner)
        slice_idx = (offsets // inner) % slice_len
        inner_idx = offsets % inner
        dim_index = start + slice_idx * step
        input_offset = outer_idx * dim_size * inner + dim_index * inner + inner_idx

        tl.store(grad_input_ptr + input_offset, grad, mask=mask)


@triton.jit
def slice_backward_copy_kernel(
    grad_output_ptr,
    grad_input_ptr,
    numel,
    inner,
    slice_len,
    dim_size,
    start,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    loop_step = (num_jobs * BLOCK_SIZE).to(tl.int64)
    slice_block = slice_len * inner
    input_block = dim_size * inner
    input_start = start * inner

    for block_start_offset in range(block_start, numel, loop_step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        grad = tl.load(grad_output_ptr + offsets, mask=mask)
        outer_idx = offsets // slice_block
        inner_offset = offsets % slice_block
        input_offset = outer_idx * input_block + input_start + inner_offset

        tl.store(grad_input_ptr + input_offset, grad, mask=mask)


@triton.jit
def slice_backward_zero_kernel(
    grad_input_ptr,
    numel,
    inner,
    dim_size,
    start,
    slice_len,
    zero_dim_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    loop_step = (num_jobs * BLOCK_SIZE).to(tl.int64)
    zero_block = zero_dim_len * inner
    input_block = dim_size * inner
    copied_end = start + slice_len

    for block_start_offset in range(block_start, numel, loop_step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        outer_idx = offsets // zero_block
        zero_offset = offsets % zero_block
        zero_dim_idx = zero_offset // inner
        inner_idx = zero_offset % inner
        dim_index = tl.where(
            zero_dim_idx < start,
            zero_dim_idx,
            copied_end + zero_dim_idx - start,
        )
        input_offset = outer_idx * input_block + dim_index * inner + inner_idx

        tl.store(grad_input_ptr + input_offset, 0.0, mask=mask)


def _normalize_slice(dim_size, start, end, step):
    if start < 0:
        start += dim_size
    if end < 0:
        end += dim_size
    start = max(0, min(start, dim_size))
    end = max(0, min(end, dim_size))
    if end < start:
        end = start
    return start


def _launch_grid(numel):
    return (min(triton.cdiv(numel, BLOCK_SIZE), TOTAL_CORE_NUM),)


def _launch_outer_grid(outer, segment_numel):
    return (min(outer, TOTAL_CORE_NUM), triton.cdiv(segment_numel, BLOCK_SIZE))


def _launch_strided_grid(outer, slice_len, inner):
    return (min(outer, TOTAL_CORE_NUM), slice_len, triton.cdiv(inner, BLOCK_SIZE))


def slice_backward(
    grad_output,
    input_sizes,
    dim,
    start,
    end,
    step,
):
    logger.debug("GEMS_TSINGMICRO SLICE_BACKWARD")
    shape = list(input_sizes)
    if dim < 0:
        dim += len(shape)

    inner = 1
    for i in range(dim + 1, len(shape)):
        inner *= shape[i]

    outer = 1
    for i in range(dim):
        outer *= shape[i]

    dim_size = shape[dim]
    start = _normalize_slice(dim_size, start, end, step)
    slice_len = grad_output.shape[dim]
    numel = grad_output.numel()

    grad_input = torch.empty(
        input_sizes, device=grad_output.device, dtype=grad_output.dtype
    )

    with torch_device_fn.device(grad_output.device):
        if step == 1:
            input_block = dim_size * inner
            prefix_numel = start * inner
            copied_offset = start * inner
            copied_numel = slice_len * inner
            suffix_offset = (start + slice_len) * inner
            suffix_numel = (dim_size - start - slice_len) * inner
            if prefix_numel > 0:
                slice_backward_outer_zero_kernel[
                    _launch_outer_grid(outer, prefix_numel)
                ](
                    grad_input,
                    outer,
                    prefix_numel,
                    input_block,
                    0,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=1,
                )
            if suffix_numel > 0:
                slice_backward_outer_zero_kernel[
                    _launch_outer_grid(outer, suffix_numel)
                ](
                    grad_input,
                    outer,
                    suffix_numel,
                    input_block,
                    suffix_offset,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=1,
                )
            if numel > 0:
                slice_backward_outer_copy_kernel[
                    _launch_outer_grid(outer, copied_numel)
                ](
                    grad_output,
                    grad_input,
                    outer,
                    copied_numel,
                    input_block,
                    copied_offset,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=1,
                )
        else:
            zero_numel = outer * dim_size * inner
            slice_backward_range_zero_kernel[_launch_grid(zero_numel)](
                grad_input,
                zero_numel,
                0,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=1,
            )
            if numel > 0:
                slice_backward_strided_copy_kernel[
                    _launch_strided_grid(outer, slice_len, inner)
                ](
                    grad_output,
                    grad_input,
                    outer,
                    inner,
                    dim_size,
                    start,
                    step,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=1,
                )

    return grad_input
