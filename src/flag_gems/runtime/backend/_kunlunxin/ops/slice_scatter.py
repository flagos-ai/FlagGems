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
# Fix (v2): the "full last dim" fast case (start==0, dim is last, step divides
# the dim, slice covers the whole dim -- exactly the benchmark) is routed to
# `slice_scatter_pack_kernel`, which reinterprets the contiguous output as one
# wide element per `step`-sized unit and only patches the first element of each
# unit with the src value. This keeps every global access contiguous (block-DMA,
# isDiscrete=false in the generated IR) and removes the per-lane division,
# modulo, compare and select of the generic kernel. Measured on XPU 3:
# (4096,4096) fp16 slice_scatter 976us (fast_gather) -> 107us (packed).
# `slice_scatter_fast_kernel` (per-lane gather) is retained for fast cases whose
# unit does not fit a u32/u64 (step>2 for 4-byte dtypes, step>4 for 2-byte
# dtypes) and for non-contiguous/unaligned inputs; `slice_scatter_kernel` remains
# the correct general path for arbitrary dim / start / partial slice / step.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

# Tuned on XPU 3 for the slice_scatter benchmark matrix (card 2, 2026-10-26).
_PACK_SMALL_NUMEL = 1 << 14
_PACK_MID_NUMEL = 1 << 18
_PACK_SMALL_BLOCK = 1024
_PACK_MID_BLOCK = 8192
_PACK_BIG_BLOCK = 16384
_PACK_SMALL_WARPS = 4
_PACK_MID_WARPS = 8
_PACK_BIG_WARPS = 16
# num_warps kept at 8 for the generic/`fast` gather kernels (pre-existing).
_PACK_WARPS_GENERIC = 8


@triton.jit
def slice_scatter_fast_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    total_elements,
    step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Fast path (gather form): dim is the last dim (dim_prod_post==1), start==0
    # and the slice covers the whole dim with `step` dividing it. The src flat
    # index is exactly idx // step and the slice mask is idx % step == 0 (both
    # affine in idx, with step constexpr -> no runtime division). The src load
    # still lowers to a discrete gather on XPU, so this is only kept for fast
    # cases the packed kernel below cannot serve.
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    inp_data = tl.load(inp_ptr + idx, mask=mask)
    src_data = tl.load(src_ptr + (idx // step), mask=mask)
    result = tl.where(idx % step == 0, src_data, inp_data)
    tl.store(out_ptr + idx, result, mask=mask)


@triton.jit
def slice_scatter_pack_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    n_unit,
    UNIT_BITS: tl.constexpr,
    ELEM_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Packed fast path (contiguous block-DMA variant).
    #
    # For the fast case `out[step*k + t] = (t == 0) ? src[k] : inp[step*k + t]`
    # on contiguous tensors, `step` consecutive output elements form one unit
    # that differs from the `step` consecutive input elements only in the first
    # element: `unit = (inp_unit & ~low_mask) | src_elem_bits`. Reinterpreting
    # the unit as a single u32/u64 makes every global access contiguous, so the
    # lowered IR has stride=1 / isDiscrete=false (validated on XPU 3; the
    # per-lane `idx // step` gather of slice_scatter_fast_kernel is ~9x slower).
    # NOTE: the "first element" occupies the low-order bits (little-endian).
    r = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = r < n_unit
    if UNIT_BITS == 32:
        in_u = tl.load(inp_ptr.to(tl.pointer_type(tl.uint32)) + r, mask=mask)
        src_bits = tl.load(src_ptr.to(tl.pointer_type(tl.uint16)) + r, mask=mask).to(
            tl.uint32
        )
        out_u = (in_u & 0xFFFF0000) | src_bits
        tl.store(out_ptr.to(tl.pointer_type(tl.uint32)) + r, out_u, mask=mask)
    else:
        in_u = tl.load(inp_ptr.to(tl.pointer_type(tl.uint64)) + r, mask=mask)
        if ELEM_BITS == 32:
            src_bits = tl.load(
                src_ptr.to(tl.pointer_type(tl.uint32)) + r, mask=mask
            ).to(tl.uint64)
            out_u = (in_u & 0xFFFFFFFF00000000) | src_bits
        else:
            src_bits = tl.load(
                src_ptr.to(tl.pointer_type(tl.uint16)) + r, mask=mask
            ).to(tl.uint64)
            out_u = (in_u & 0xFFFFFFFFFFFF0000) | src_bits
        tl.store(out_ptr.to(tl.pointer_type(tl.uint64)) + r, out_u, mask=mask)


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
    NEED_CLAMP: tl.constexpr,
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
    if NEED_CLAMP:
        src_idx = tl.where(slice_mask, src_idx, 0)
    src_data = tl.load(src_ptr + src_idx, mask=mask & slice_mask, other=0.0)
    result = tl.where(slice_mask, src_data, inp_data)
    tl.store(out_ptr + idx, result, mask=mask)


def _block_for(numel):
    if numel <= _PACK_SMALL_NUMEL:
        return _PACK_SMALL_BLOCK, _PACK_SMALL_WARPS
    if numel <= _PACK_MID_NUMEL:
        return _PACK_MID_BLOCK, _PACK_MID_WARPS
    return _PACK_BIG_BLOCK, _PACK_WARPS_GENERIC


def _pack_block_for(numel):
    """Block/launch shape for the packed kernel (tuned on n_unit, XPU 3)."""
    if numel <= _PACK_SMALL_NUMEL:
        return _PACK_SMALL_BLOCK, _PACK_SMALL_WARPS
    if numel <= _PACK_MID_NUMEL:
        return _PACK_MID_BLOCK, _PACK_MID_WARPS
    return _PACK_BIG_BLOCK, _PACK_BIG_WARPS


def _needs_clamp(total_elements, dim_size, dim_prod_post, start, step, src_dim_size):
    """Exact O(1) bounds check on the src index `slice_scatter_kernel` computes.

    `src_idx` is monotonic in each of pre_idx / dim_idx / post_idx, so its
    extremes over the whole launch domain are reached at the corners. If both
    stay inside `src`, masked-out lanes can never form an illegal address and
    the in-kernel clamp is dead weight -> compile it out.
    """
    outer = total_elements // (dim_size * dim_prod_post)
    n_src = outer * src_dim_size * dim_prod_post
    # floor division, matching the kernel's `//` on negative values
    lo = ((0 - start) // step) * dim_prod_post
    hi = (
        (outer - 1) * src_dim_size * dim_prod_post
        + ((dim_size - 1 - start) // step) * dim_prod_post
        + (dim_prod_post - 1)
    )
    return lo < 0 or hi >= n_src


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
        elem_bits = inp.element_size() * 8
        # Packed path: one u32/u64 unit per `step` output elements. Requires a
        # contiguous dense layout (else the linear unit indexing would not match
        # the logical element order) and u32/u64-aligned base pointers.
        pack_unit_bits = None
        if elem_bits == 16 and step == 2:
            pack_unit_bits = 32
        elif elem_bits == 16 and step == 4:
            pack_unit_bits = 64
        elif elem_bits == 32 and step == 2:
            pack_unit_bits = 64
        if (
            pack_unit_bits is not None
            and inp.is_contiguous()
            and src.is_contiguous()
            and out.is_contiguous()
            and inp.data_ptr() % (pack_unit_bits // 8) == 0
            and src.data_ptr() % (pack_unit_bits // 8) == 0
            and out.data_ptr() % (pack_unit_bits // 8) == 0
        ):
            n_unit = total_elements // step
            pblock, pwarps = _pack_block_for(n_unit)
            grid = (triton.cdiv(n_unit, pblock),)
            slice_scatter_pack_kernel[grid](
                out,
                inp,
                src,
                n_unit,
                UNIT_BITS=pack_unit_bits,
                ELEM_BITS=elem_bits,
                BLOCK_SIZE=pblock,
                num_warps=pwarps,
            )
            return out

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
        NEED_CLAMP=_needs_clamp(
            total_elements, dim_size, dim_prod_post, start, step, src_dim_size
        ),
        num_warps=num_warps,
    )
    return out
