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

# Kunlunxin (XPU) override of slice_scatter.
#
# The previous override built the result with two `copy_slice` passes
# (copy inp -> out, then copy src into the strided view out[..start:end:step..]).
# The second pass writes to a step-strided view through pointwise_dynamic, which
# on XPU degrades to fully discrete stores -> ~76ms for [4096,4096] step=2
# (gems speedup ~0.001-0.005). The generic single fused kernel is contiguous on
# read/write but reconstructs the src gather index as `pre*src_dim_size + ...`
# with a runtime `step` division; the compiler cannot prove that access is affine
# so it also falls to a discrete src gather (~9ms).
#
# Fix: keep a single fused kernel that reads inp contiguously and writes out
# contiguously (block DMA), but for the common "scatter every step-th element
# from 0 over the full last dim" case (start==0, dim is last, step divides the
# dim, slice covers the whole dim -- exactly the benchmark) route to `ss_fast`
# where the src index collapses to the affine `idx // step` and the mask to
# `idx % step == 0`, WITH `step` as a constexpr so the division is compiled away.
# That turns the src gather into a recognizable strided load (~0.13-0.5ms). All
# other shapes (arbitrary dim/start/partial slice) take the correct general
# fused kernel.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)


@triton.jit
def slice_scatter_fast_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    total_elements,
    step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Fast path: dim is the last dim (dim_prod_post==1), start==0 and the slice
    # covers the whole dim with `step` dividing it. Then the src flat index is
    # exactly idx // step and the slice mask is idx % step == 0 (both affine in
    # idx, with step constexpr -> no runtime division).
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    inp_data = tl.load(inp_ptr + idx, mask=mask)
    src_data = tl.load(src_ptr + (idx // step), mask=mask)
    result = tl.where(idx % step == 0, src_data, inp_data)
    tl.store(out_ptr + idx, result, mask=mask)


@triton.jit
def slice_scatter_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    total_elements,
    dim_size,
    dim_prod_post,
    start,
    step,
    src_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # General path (correct for arbitrary dim / start / partial slice / step).
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < total_elements

    pre_idx = idx // (dim_size * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % dim_size
    post_idx = idx % dim_prod_post

    slice_mask = (
        (dim_idx >= start)
        & (dim_idx < start + src_dim_size * step)
        & ((dim_idx - start) % step == 0)
    )

    inp_data = tl.load(inp_ptr + idx, mask=mask)

    src_dim_idx = (dim_idx - start) // step
    src_idx = (
        pre_idx * src_dim_size * dim_prod_post + src_dim_idx * dim_prod_post + post_idx
    )
    src_data = tl.load(src_ptr + src_idx, mask=mask & slice_mask)
    result = tl.where(slice_mask, src_data, inp_data)
    tl.store(out_ptr + idx, result, mask=mask)


def _block_for(numel):
    if numel <= (1 << 14):
        return 1024, 4
    if numel <= (1 << 18):
        return 8192, 8
    return 16384, 8


def slice_scatter(inp, src, dim=0, start=None, end=None, step=1):
    logger.debug("GEMS_KUNLUNXIN SLICE_SCATTER")
    assert src.device == inp.device, "inp and src reside on different devices."
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert step > 0, "slice step must be positive"
    dim = dim % inp.ndim

    start = start or 0
    end = end or inp.size(dim)
    if end < 0:
        end = end % inp.size(dim)

    valid_shape = list(inp.shape)
    valid_shape[dim] = triton.cdiv(end - start, step)
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    if has_internal_overlapping(inp) == MemOverlap.Yes:
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )

    inp = inp.contiguous()
    src = src.contiguous()

    total_elements = inp.numel()
    dim_size = inp.size(dim)
    src_dim_size = src.size(dim)

    dim_prod_post = 1
    for d in range(dim + 1, inp.ndim):
        dim_prod_post *= inp.size(d)

    block_size, num_warps = _block_for(total_elements)

    fast = (
        dim_prod_post == 1
        and start == 0
        and dim_size % step == 0
        and src_dim_size * step == dim_size
    )

    if fast:
        grid = (triton.cdiv(total_elements, block_size),)
        slice_scatter_fast_kernel[grid](
            out,
            inp,
            src,
            total_elements,
            step=step,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return out

    grid = (triton.cdiv(total_elements, block_size),)
    slice_scatter_kernel[grid](
        out,
        inp,
        src,
        total_elements,
        dim_size,
        dim_prod_post,
        start,
        step,
        src_dim_size,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out
