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

from flag_gems.ops.nanmedian import (
    CUDA_SUPPORTS_MASKED_HISTOGRAM,
    FLAT_RADIX_BITS,
    FLAT_RADIX_BLOCK_N,
    INT32_MAX,
    MAX_BLOCK_N,
    RADIX_SELECT_DTYPES,
    _check_supported_dtype,
    _is_not_nan,
    _launch_radix_select,
    _nanmedian_dim_impl,
    _nanmedian_flat_impl,
    _to_order_key,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

# Runtime loads vendor modules under _thead, while use_gems records flag_gems.
logger = logging.getLogger("flag_gems.runtime.backend._thead.ops.nanmedian")

_SORT_LIMIT = 4096
_SELECT_DTYPES = RADIX_SELECT_DTYPES + (torch.float64, torch.int64)


@libentry()
@triton.jit
def nanmedian_sort_select_kernel(
    inp,
    out_values,
    out_indices,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tle.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    row_inp = inp + row * N
    dtype = inp.dtype.element_ty
    vals = tl.load(row_inp + cols, cols < N, other=0)
    valid = cols < N
    if dtype.is_floating():
        valid = valid & _is_not_nan(vals, False)

    keys = _to_order_key(vals, valid)
    ordered = tl.sort(keys, descending=False)
    count = tl.sum(valid.to(tl.int32), axis=0)
    rank = tl.maximum((count - 1) // 2, 0)
    key = tl.gather(ordered, tl.full((1,), 0, tl.int32) + rank, 0).reshape(())
    selected = tl.min(tl.where(valid & (keys == key), cols, N), axis=0)
    if dtype.is_floating():
        result = tl.load(row_inp + selected, count > 0, other=float("nan"))
        selected = tl.where(count > 0, selected, 0)
    else:
        result = tl.load(row_inp + selected)
    tl.store(out_values + row, result)
    tl.store(out_indices + row, selected)


@libentry()
@triton.jit(do_not_specialize=["digit_pos", "step"])
def nanmedian_ppu_flat_count_kernel(
    inp,
    bin_counts,
    state,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    digit_pos,
    step,
    FIRST: tl.constexpr,
):
    BINS: tl.constexpr = 1 << RADIX_BITS
    offsets = tle.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offsets < N
    vals = tl.load(inp + offsets, valid, other=0)
    dtype = inp.dtype.element_ty
    if dtype.is_floating():
        valid = valid & _is_not_nan(vals, False)

    keys = _to_order_key(vals, valid)
    if not FIRST:
        selected = tl.load(state).to(keys.dtype)
        valid = valid & ((keys >> (digit_pos + RADIX_BITS)) == selected)
    digit = ((keys >> digit_pos) & (BINS - 1)).to(tl.int32)
    counts = tl.histogram(digit, BINS, valid)
    bins = tl.arange(0, BINS)
    tl.atomic_add(bin_counts + step * BINS + bins, counts, counts > 0, sem="relaxed")


@libentry()
@triton.jit(do_not_specialize=["step"])
def nanmedian_ppu_flat_select_kernel(
    bin_counts,
    state,
    out,
    RADIX_BITS: tl.constexpr,
    step,
    FIRST: tl.constexpr,
    LAST: tl.constexpr,
):
    BINS: tl.constexpr = 1 << RADIX_BITS
    bins = tl.arange(0, BINS)
    counts = tl.load(bin_counts + step * BINS + bins)
    cumsum = tl.cumsum(counts, axis=0)
    before = cumsum - counts
    if FIRST:
        count = tl.sum(counts, axis=0)
        rank = count // 2 + (count & 1)
        prefix = tl.full((), 0, tl.uint64)
    else:
        rank = tl.load(state + 1).to(tl.int32)
        prefix = tl.load(state).to(tl.uint64)

    is_bucket = (before < rank) & (rank <= cumsum)
    bucket = tl.min(tl.where(is_bucket, bins, BINS - 1), axis=0)
    prefix = (prefix << RADIX_BITS) | bucket.to(tl.uint64)
    if LAST:
        dtype = out.dtype.element_ty
        bits: tl.constexpr = dtype.primitive_bitwidth
        sign: tl.constexpr = 1 << (bits - 1)
        if dtype.is_floating():
            key = prefix.to(tl.dtype(f"uint{bits}"))
            mask = tl.where((key & sign) != 0, sign, (1 << bits) - 1).to(key.dtype)
            result = (key ^ mask).to(dtype, bitcast=True)
            tl.store(out, tl.where(rank > 0, result, float("nan")))
        elif dtype.is_int_signed():
            key = prefix.to(tl.dtype(f"uint{bits}"))
            tl.store(out, (key ^ sign).to(dtype, bitcast=True))
        else:
            tl.store(out, prefix.to(dtype))
    else:
        remaining_rank = rank - tl.max(tl.where(is_bucket, before, 0), axis=0)
        tl.store(state, prefix.to(tl.int64))
        tl.store(state + 1, remaining_rank)


def _nanmedian_ppu_flat_radix_select(inp, out):
    flat = inp.reshape(-1).contiguous()
    n = flat.numel()
    if out is None:
        out = torch.empty((), dtype=flat.dtype, device=flat.device)
    block_n = FLAT_RADIX_BLOCK_N
    bins = 1 << FLAT_RADIX_BITS
    steps = flat.element_size()
    # Each pass gets independent zeroed counters. The first histogram also
    # counts valid values, and the final prefix encodes the result directly.
    counts = torch.zeros((steps, bins), dtype=torch.int32, device=flat.device)
    state = torch.empty((2,), dtype=torch.int64, device=flat.device)
    grid = (triton.cdiv(n, block_n),)
    with torch_device_fn.device(flat.device):
        for step in range(steps):
            digit_pos = (steps - 1 - step) * FLAT_RADIX_BITS
            nanmedian_ppu_flat_count_kernel[grid](
                flat,
                counts,
                state,
                n,
                block_n,
                FLAT_RADIX_BITS,
                digit_pos,
                step,
                step == 0,
                num_warps=8,
                num_stages=1,
            )
            nanmedian_ppu_flat_select_kernel[(1,)](
                counts,
                state,
                out,
                FLAT_RADIX_BITS,
                step,
                step == 0,
                step == steps - 1,
                num_warps=4,
                num_stages=1,
            )
    return out


def _launch_sort_select(inp, values, indices, M, N):
    block_n = triton.next_power_of_2(N)
    with torch_device_fn.device(inp.device):
        nanmedian_sort_select_kernel[(M,)](inp, values, indices, N, block_n)


def _get_row_selector(inp, M, N):
    # Selecting through kthvalue can enter Gems Top-K with an oversized JIT tile.
    if (
        inp.is_cuda
        and inp.dtype in _SELECT_DTYPES
        and M * N <= INT32_MAX
        and (N > MAX_BLOCK_N or inp.dtype is torch.float64)
    ):
        if N <= _SORT_LIMIT:
            return _launch_sort_select
        return _launch_radix_select
    return None


def _flat(inp, out=None):
    if (
        inp.is_cuda
        and CUDA_SUPPORTS_MASKED_HISTOGRAM
        and inp.dtype in _SELECT_DTYPES
        and _SORT_LIMIT < inp.numel() <= INT32_MAX
    ):
        return _nanmedian_ppu_flat_radix_select(inp, out)

    return _nanmedian_flat_impl(inp, out=out, get_row_selector=_get_row_selector)


def nanmedian(inp):
    logger.debug("GEMS_THEAD NANMEDIAN")
    _check_supported_dtype(inp)
    return _flat(inp)


def nanmedian_out(inp, *, out):
    logger.debug("GEMS_THEAD NANMEDIAN_OUT")
    _check_supported_dtype(inp)
    return _flat(inp, out=out)


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_THEAD NANMEDIAN_DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim, get_row_selector=_get_row_selector)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_THEAD NANMEDIAN_DIM_VALUES")
    return _nanmedian_dim_impl(
        inp, dim, keepdim, out=(values, indices), get_row_selector=_get_row_selector
    )
