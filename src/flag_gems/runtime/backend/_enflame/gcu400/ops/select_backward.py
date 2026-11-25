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
import math

import torch
import triton
import triton.language as tl

_FILL_BLOCK = 8192
logger = logging.getLogger(__name__)


@triton.jit
def _select_bwd_fill_kernel(out_ptr, n_out, BLOCK: tl.constexpr, WIDE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if WIDE:
        base0 = pid.to(tl.int64) * BLOCK
        step = nprog.to(tl.int64) * BLOCK
        for off0 in tl.range(base0, n_out, step):
            offs = off0 + tl.arange(0, BLOCK)
            tl.store(
                out_ptr + offs,
                tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty),
                mask=offs < n_out,
            )
    else:
        for off0 in tl.range(pid * BLOCK, n_out, nprog * BLOCK):
            offs = off0 + tl.arange(0, BLOCK)
            tl.store(
                out_ptr + offs,
                tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty),
                mask=offs < n_out,
            )


@triton.jit
def _select_bwd_patch_rowblock_kernel(
    grad_ptr,
    out_ptr,
    outer,
    inner_size: tl.constexpr,
    dim_stride: tl.constexpr,
    lo: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    npr = tl.num_programs(0)
    pid_b = tl.program_id(1)
    npb = tl.num_programs(1)
    for row in tl.range(pid_r, outer, npr):
        row64 = row.to(tl.int64)
        row_base = row64 * dim_stride
        for g0 in tl.range(pid_b * BLOCK, inner_size, npb * BLOCK):
            gidx = g0 + tl.arange(0, BLOCK)
            m = gidx < inner_size
            vals = tl.load(grad_ptr + row64 * inner_size + gidx, mask=m)
            tl.store(out_ptr + row_base + lo + gidx, vals, mask=m)


@triton.jit
def _select_bwd_patch_flat_kernel(
    grad_ptr,
    out_ptr,
    total,
    inner_size: tl.constexpr,
    dim_stride: tl.constexpr,
    lo: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    safe_outer = tl.where(mask, offs // inner_size, 0).to(tl.int64)
    inner = offs % inner_size
    vals = tl.load(grad_ptr + offs, mask=mask)
    out_offset = safe_outer * dim_stride + lo + inner
    tl.store(out_ptr + out_offset, vals, mask=mask)


def _next_pow2(x):
    x = int(x)
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def select_backward(grad, input_sizes, dim, index, out=None):
    logger.debug("GEMS_ENFLAME_GCU400 SELECT_BACKWARD")
    dim = int(dim)
    index = int(index)
    sizes = list(input_sizes)
    ndim = len(sizes)

    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError("invalid dim")

    dim_size = sizes[dim]

    if index < 0:
        index += dim_size
    if index < 0 or index >= dim_size:
        raise ValueError("index out of range")

    if out is None:
        out = torch.empty(sizes, dtype=grad.dtype, device=grad.device)
    else:
        if tuple(out.shape) != tuple(sizes):
            raise ValueError("out shape mismatch")
        if out.dtype != grad.dtype:
            raise ValueError("dtype mismatch")
        if out.device != grad.device:
            raise ValueError("device mismatch")

    outer_size = math.prod(sizes[:dim]) if dim > 0 else 1
    inner_size = math.prod(sizes[dim + 1 :]) if dim < ndim - 1 else 1
    dim_stride = dim_size * inner_size
    n_out = outer_size * dim_stride
    if n_out == 0:
        return out

    grad_c = grad.contiguous()
    total = outer_size * inner_size
    lo = index * inner_size

    wide = n_out >= 2**31
    fill_block = max(1024, min(_FILL_BLOCK, _next_pow2(n_out)))
    n_fill_blocks = triton.cdiv(n_out, fill_block)
    fill_grid = (min(48, n_fill_blocks),)
    _select_bwd_fill_kernel[fill_grid](out, n_out, fill_block, wide, num_warps=4)

    if total > 0:
        if inner_size >= 1024:
            rblock = min(8192, _next_pow2(inner_size))
            n_rblocks = triton.cdiv(inner_size, rblock)
            patch_grid = (min(48, outer_size), min(48, n_rblocks))
            _select_bwd_patch_rowblock_kernel[patch_grid](
                grad_c,
                out,
                outer_size,
                inner_size,
                dim_stride,
                lo,
                rblock,
                num_warps=4,
            )
        else:
            pblock = max(256, min(16384, _next_pow2(triton.cdiv(total, 96))))
            n_pblocks = triton.cdiv(total, pblock)
            patch_grid = (n_pblocks,)
            _select_bwd_patch_flat_kernel[patch_grid](
                grad_c,
                out,
                total,
                inner_size,
                dim_stride,
                lo,
                pblock,
                wide,
                num_warps=4,
            )
    return out
