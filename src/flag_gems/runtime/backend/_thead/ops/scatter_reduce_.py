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

"""PPU product/minimum fast paths with current generic fallback semantics."""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.scatter_reduce import (
    _decode_nan_min,
    _encode_nan_min,
    _rowwise_result_is_safe,
)
from flag_gems.ops.scatter_reduce import scatter_reduce as _generic_scatter_reduce
from flag_gems.ops.scatter_reduce import scatter_reduce_ as _generic_scatter_reduce_
from flag_gems.ops.scatter_reduce import (
    scatter_reduce_out as _generic_scatter_reduce_out,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# Use integer exchanges to build source lists, avoiding contended FP32 product
# CAS retries. The same algorithm is used by the Hygon specialization.
@libentry()
@triton.jit(do_not_specialize=["heads_ptr"])
def ppu_scatter_reduce_init_heads_kernel(heads_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(heads_ptr + offsets, -2, mask=offsets < N)


@libentry()
@triton.jit(do_not_specialize=["index_ptr", "src_ptr", "heads_ptr", "next_ptr"])
def ppu_scatter_reduce_prod_build_lists_kernel(
    index_ptr,
    src_ptr,
    heads_ptr,
    next_ptr,
    N,
    index_ncols: tl.constexpr,
    src_ncols: tl.constexpr,
    out_ncols: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
    NEXT_OFFSET: tl.constexpr = 0,
):
    """Build one lock-free source list per output using global integer exchange."""
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    lanes = tl.arange(0, BLOCK).to(tl.int64)
    base = pid * (BLOCK * LOOP) + lanes

    for loop_idx in range(LOOP):
        offsets = base + loop_idx * BLOCK
        mask = offsets < N
        row = offsets // index_ncols
        col = offsets % index_ncols
        source = tl.load(
            src_ptr + row * src_ncols + col,
            mask=mask,
            other=1.0,
        ).to(tl.float32)
        changes_value = source != 1.0
        active = mask & changes_value
        index_mask = active if INCLUDE_SELF else mask
        index = tl.load(index_ptr + offsets, mask=index_mask, other=0).to(tl.int64)
        out_offsets = row * out_ncols + index
        if not INCLUDE_SELF:
            # -2 is untouched; -1 marks identity-only updates. A real node
            # is nonnegative, so this marker never overwrites an existing list.
            tl.atomic_max(
                heads_ptr + out_offsets,
                -1,
                mask=mask & ~changes_value,
                sem="relaxed",
            )
        previous = tl.atomic_xchg(
            heads_ptr + out_offsets,
            offsets.to(tl.int32),
            mask=active,
            sem="relaxed",
        )
        tl.store(next_ptr + NEXT_OFFSET + offsets, previous, mask=active)


@libentry()
@triton.jit(
    do_not_specialize=["inp_ptr", "src_ptr", "heads_ptr", "next_ptr", "result_ptr"]
)
def ppu_scatter_reduce_prod_finalize_lists_kernel(
    inp_ptr,
    src_ptr,
    heads_ptr,
    next_ptr,
    result_ptr,
    out_numel,
    index_ncols: tl.constexpr,
    src_ncols: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
    NEXT_OFFSET: tl.constexpr = 0,
):
    """Traverse independent source lists and multiply each source exactly once."""
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < out_numel
    node = tl.load(heads_ptr + offsets, mask=mask, other=-1)
    touched = node != -2
    if INCLUDE_SELF:
        value = tl.load(inp_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    else:
        value = tl.full((BLOCK,), 1.0, tl.float32)

    active = mask & (node >= 0)
    done = ~active
    all_done = False
    while not all_done:
        safe_node = tl.where(active, node, 0).to(tl.int64)
        row = safe_node // index_ncols
        col = safe_node % index_ncols
        source = tl.load(
            src_ptr + row * src_ncols + col,
            mask=active,
            other=1.0,
        ).to(tl.float32)
        value = tl.where(active, value * source, value)
        node = tl.load(next_ptr + NEXT_OFFSET + safe_node, mask=active, other=-1)
        active &= node >= 0
        done |= ~active
        all_done = tl.sum(done.to(tl.int32)) == BLOCK

    if not INCLUDE_SELF:
        inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.where(touched, value, inp)
    tl.store(result_ptr + offsets, value, mask=mask)


def _split_grid(programs):
    rows = triton.cdiv(programs, 65535)
    return triton.cdiv(programs, rows), rows


@libentry()
@triton.jit
def ppu_scatter_min_init_kernel(
    inp, scratch, N, INCLUDE_SELF: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(inp + offsets, mask=offsets < N, other=float("inf")).to(tl.float32)
    # INT_MAX cannot represent an encoded non-NaN float (even +inf is smaller).
    # NaNs map to INT_MIN, so this sentinel also tracks untouched outputs.
    initial = _encode_nan_min(value) if INCLUDE_SELF else 0x7FFFFFFF
    tl.store(scratch + offsets, initial, mask=offsets < N)


@libentry()
@triton.jit
def ppu_scatter_min_update_kernel(
    index,
    src,
    scratch,
    N,
    INDEX_COLS: tl.constexpr,
    SRC_COLS: tl.constexpr,
    OUT_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    row = offsets // INDEX_COLS
    col = offsets % INDEX_COLS
    target = row * OUT_COLS + tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    value = tl.load(src + row * SRC_COLS + col, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_min(scratch + target, _encode_nan_min(value), mask=mask, sem="relaxed")


@libentry()
@triton.jit
def ppu_scatter_min_finalize_kernel(
    inp, scratch, result, N, INCLUDE_SELF: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    ordered = tl.load(scratch + offsets, mask=mask, other=0)
    value = _decode_nan_min(ordered)
    if not INCLUDE_SELF:
        original = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.where(ordered != 0x7FFFFFFF, value, original)
    tl.store(result + offsets, value, mask=mask)


def _encoded_min(inp, index, src, include_self, result=None):
    with torch_device_fn.device(inp.device):
        if result is None:
            result = torch.empty_like(inp)
        scratch = torch.empty(
            inp.numel(),
            dtype=torch.int32,
            device=inp.device,
        )
        ppu_scatter_min_init_kernel[_split_grid(triton.cdiv(inp.numel(), 1024))](
            inp, scratch, inp.numel(), include_self, BLOCK=1024
        )
        ppu_scatter_min_update_kernel[_split_grid(triton.cdiv(index.numel(), 512))](
            index,
            src,
            scratch,
            index.numel(),
            index.shape[-1],
            src.shape[-1],
            inp.shape[-1],
            BLOCK=512,
        )
        ppu_scatter_min_finalize_kernel[_split_grid(triton.cdiv(inp.numel(), 1024))](
            inp, scratch, result, inp.numel(), include_self, BLOCK=1024
        )
    return result


def _can_use_contiguous_reduction(inp, dim, index, src, reduce, result=None):
    if (
        reduce not in ("prod", "amin")
        or inp.ndim not in (1, 2)
        or index.ndim != inp.ndim
        or src.ndim != inp.ndim
        or dim not in (-1, inp.ndim - 1)
        or inp.dtype not in (torch.float16, torch.float32, torch.bfloat16)
        or src.dtype != inp.dtype
        or index.dtype != torch.int64
        or src.device != inp.device
        or index.device != inp.device
        or not inp.is_contiguous()
        or not index.is_contiguous()
        or not src.is_contiguous()
        or inp.numel() == 0
        or index.numel() == 0
        or index.numel() >= 1 << 31
        or max(inp.shape[-1], index.shape[-1]) <= (4096 if reduce == "amin" else 64)
        or (inp.ndim == 2 and index.shape[0] > inp.shape[0])
        or (inp.ndim == 2 and index.shape[0] > src.shape[0])
        or index.shape[-1] > src.shape[-1]
    ):
        return False
    return result is None or (
        result.shape == inp.shape
        and result.dtype == inp.dtype
        and result.device == inp.device
        and result.is_contiguous()
        and _rowwise_result_is_safe(inp, index, src, result)
    )


def _linked_product(inp, index, src, include_self, result=None):
    with torch_device_fn.device(inp.device):
        if result is None:
            result = torch.empty_like(inp)
        scratch = torch.empty(
            inp.numel() + index.numel(), dtype=torch.int32, device=inp.device
        )
        ppu_scatter_reduce_init_heads_kernel[
            _split_grid(triton.cdiv(inp.numel(), 1024))
        ](scratch, inp.numel(), BLOCK=1024)
        ppu_scatter_reduce_prod_build_lists_kernel[
            _split_grid(triton.cdiv(index.numel(), 512))
        ](
            index,
            src,
            scratch,
            scratch,
            index.numel(),
            index.shape[-1],
            src.shape[-1],
            inp.shape[-1],
            include_self,
            BLOCK=128,
            LOOP=4,
            NEXT_OFFSET=inp.numel(),
        )
        final_block = 256 if max(inp.shape[-1], index.shape[-1]) > 4096 else 512
        ppu_scatter_reduce_prod_finalize_lists_kernel[
            _split_grid(triton.cdiv(inp.numel(), final_block))
        ](
            inp,
            src,
            scratch,
            scratch,
            result,
            inp.numel(),
            index.shape[-1],
            src.shape[-1],
            include_self,
            BLOCK=final_block,
            NEXT_OFFSET=inp.numel(),
        )
    return result


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_THEAD SCATTER_REDUCE_TWO")
    if _can_use_contiguous_reduction(inp, dim, index, src, reduce):
        if reduce == "amin":
            return _encoded_min(inp, index, src, include_self)
        return _linked_product(inp, index, src, include_self)
    return _generic_scatter_reduce(
        inp, dim, index, src, reduce, include_self=include_self
    )


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_THEAD SCATTER_REDUCE_TWO_")
    if _can_use_contiguous_reduction(inp, dim, index, src, reduce, inp):
        if reduce == "amin":
            return _encoded_min(inp, index, src, include_self, inp)
        return _linked_product(inp, index, src, include_self, inp)
    return _generic_scatter_reduce_(
        inp, dim, index, src, reduce, include_self=include_self
    )


def scatter_reduce_out(inp, dim, index, src, reduce, *, include_self=True, out=None):
    logger.debug("GEMS_THEAD SCATTER_REDUCE_TWO_OUT")
    if _can_use_contiguous_reduction(inp, dim, index, src, reduce, out):
        if reduce == "amin":
            return _encoded_min(inp, index, src, include_self, out)
        return _linked_product(inp, index, src, include_self, out)
    return _generic_scatter_reduce_out(
        inp, dim, index, src, reduce, include_self=include_self, out=out
    )
