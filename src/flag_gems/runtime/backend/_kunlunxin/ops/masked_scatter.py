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
from flag_gems.utils import broadcastable, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# The old kunlunxin masked_scatter used a TWO-PASS scheme launched on only
# `multi_processor_count` programs, each looping serially over blocks_per_row with a
# data-dependent `advance += tl.sum(...)` accumulator, plus a MASKED store that only
# produced correct results with the TRITONXPU_OTHER_SIM / TRITONXPU_STORE_MASK_SIM
# env flags. On XPU this serializes hard -> a fixed ~0.9 GBPS wall (4096^2 ~133ms,
# speedup ~0.003) REGARDLESS of shape.
#
# masked_scatter is exactly
#   out[p] = where(mask[p], source[cumsum(mask)[p] - 1], inp[p])
# The fix reformulates this as a SINGLE-PASS where-store on MANY parallel programs
# (grid = n_blocks), doing a discrete READ of source at the per-position index and an
# UNMASKED contiguous store of the `tl.where` result (no SIM flags needed).
#
# The per-position source index is built WITHOUT a giant global cumsum (a full
# `torch.cumsum` over the flattened mask hangs / returns OOB garbage on XPU past a few
# million ints -> kernel-exception status 299). Instead: pass 1 computes each block's
# True-count, a tiny `torch.cumsum` over just the n_blocks counts gives each block's
# base offset, and pass 2 recomputes the index in-block with `base + tl.cumsum(mask)`.
# This lifts the huge shapes from ~133/530ms to ~10.7/42ms (~12x).


@libentry()
@triton.jit
def masked_scatter_single_block_kernel(
    inp_ptr, mask_ptr, src_ptr, N, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N
    inp_val = tl.load(inp_ptr + offsets, mask=block_mask, other=0)
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)
    src_indices = tl.cumsum(mask_val.to(tl.int32), axis=0) - 1
    active = block_mask & mask_val
    src_val = tl.load(src_ptr + src_indices, mask=active, other=0)
    out_val = tl.where(mask_val, src_val, inp_val)
    tl.store(inp_ptr + offsets, out_val, mask=block_mask)


@libentry()
@triton.jit
def mask_block_count_kernel(mask_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    sel = tl.load(mask_ptr + offsets, mask=offsets < N, other=0).to(tl.int32)
    tl.store(counts_ptr + pid, tl.sum(sel, axis=0))


@libentry()
@triton.jit
def masked_scatter_where_kernel(
    inp_ptr, mask_ptr, src_ptr, base_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N
    base = tl.load(base_ptr + pid)
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)
    src_indices = base + tl.cumsum(mask_val.to(tl.int32), axis=0) - 1
    active = block_mask & mask_val
    inp_val = tl.load(inp_ptr + offsets, mask=block_mask, other=0)
    src_val = tl.load(src_ptr + src_indices, mask=active, other=0)
    tl.store(inp_ptr + offsets, tl.where(active, src_val, inp_val), mask=block_mask)


def masked_scatter_impl(inp, mask, source, N):
    flat_inp = inp.ravel()
    flat_mask = mask.ravel()
    flat_src = source.ravel()

    if N <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
        with torch_device_fn.device(inp.device):
            masked_scatter_single_block_kernel[(1,)](
                flat_inp,
                flat_mask,
                flat_src,
                N,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        return inp

    BLOCK_SIZE = 8192
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    idx_dtype = torch.int32 if N < 2**31 else torch.int64
    with torch_device_fn.device(inp.device):
        # pass 1: per-block True-count -> exclusive prefix sum (tiny cumsum, n_blocks)
        counts = torch.empty(n_blocks, dtype=idx_dtype, device=inp.device)
        mask_block_count_kernel[(n_blocks,)](
            flat_mask, counts, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=16
        )
        base = torch.zeros(n_blocks, dtype=idx_dtype, device=inp.device)
        torch.cumsum(counts[:-1], dim=0, out=base[1:])
        # pass 2: single-pass where-store, index = base + in-block cumsum
        masked_scatter_where_kernel[(n_blocks,)](
            flat_inp,
            flat_mask,
            flat_src,
            base,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=16,
        )
    return inp


def masked_scatter(inp, mask, source):
    logger.debug("GEMS_KUNLUNXIN MASKED_SCATTER")

    assert broadcastable(
        inp.shape, mask.shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"

    _, mask = torch.broadcast_tensors(inp, mask)

    out = inp.clone()
    if not out.is_contiguous():
        out = out.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()
    if not source.is_contiguous():
        source = source.contiguous()

    N = out.numel()
    masked_scatter_impl(out, mask, source, N)
    return out


def masked_scatter_(inp, mask, source):
    logger.debug("GEMS_KUNLUNXIN MASKED_SCATTER_")

    assert broadcastable(inp.shape, mask.shape)
    _, mask = torch.broadcast_tensors(inp, mask)

    if not inp.is_contiguous():
        raise RuntimeError(
            "in-place operation currently requires contiguous input tensor. "
        )

    mask = mask if mask.is_contiguous() else mask.contiguous()
    source = source if source.is_contiguous() else source.contiguous()

    N = inp.numel()
    masked_scatter_impl(inp, mask, source, N)
    return inp
