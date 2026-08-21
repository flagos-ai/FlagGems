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

from flag_gems.ops.nanmedian import (
    INT32_MAX,
    RADIX_SELECT_DTYPES,
    NanMedian,
    _check_supported_dtype,
)
from flag_gems.ops.nanmedian import _nanmedian_dim_impl as _generic_nanmedian_dim_impl
from flag_gems.ops.nanmedian import _nanmedian_flat_impl as _generic_nanmedian_flat_impl
from flag_gems.ops.nanmedian import _normalize_dim, _to_order_key
from flag_gems.ops.topk import _get_iinfo_val
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)

_TL_SORT_LIMIT = 4096
_RADIX_BITS = 4
_RADIX_SIZE = 1 << _RADIX_BITS
_RADIX_BLOCK_N = 4096


@libentry()
@triton.jit
def _nanmedian_tl_sort_kernel(
    inp,
    out_values,
    out_indices,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < N
    dtype = inp.dtype.element_ty
    if dtype.is_floating():
        high = float("inf")
    else:
        high = _get_iinfo_val(dtype, return_max=True)
    values = tl.load(inp + row * N + columns, mask=mask, other=high)

    if dtype.is_floating():
        valid = mask & (values == values)
        valid_count = tl.sum(valid.to(tl.int32), axis=0)
        sortable = tl.where(valid, values, high)
    else:
        valid = mask
        valid_count = N
        sortable = values

    ordered = tl.sort(sortable, descending=False)
    rank = tl.maximum(valid_count - 1, 0) // 2
    selected = tl.sum(
        tl.where(columns == rank, ordered, tl.zeros_like(ordered)), axis=0
    )
    first_index = tl.argmax((valid & (values == selected)).to(tl.int32), axis=0)

    if dtype.is_floating():
        all_nan = valid_count == 0
        selected = tl.where(all_nan, float("nan"), selected)
        first_index = tl.where(all_nan, 0, first_index)

    tl.store(out_values + row, selected)
    tl.store(out_indices + row, first_index.to(tl.int64))


@libentry()
@triton.jit
def _radix_init_kernel(states):
    row = tl.program_id(0)
    tl.store(states + row * 4, 0)
    tl.store(states + row * 4 + 1, 0)
    tl.store(states + row * 4 + 2, 0)
    tl.store(states + row * 4 + 3, 0)


@libentry()
@triton.jit
def _radix_partial_count_kernel(
    inp,
    partial_counts,
    states,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    DIGIT_POS: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    columns = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = columns < N
    values = tl.load(inp + row * N + columns, mask=mask, other=0.0)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    radix_mask: tl.constexpr = RADIX_SIZE - 1

    if dtype.is_floating():
        valid = mask & (values == values)
    else:
        valid = mask
    desired = tl.load(states + row * 4).to(utype)
    desired_mask = tl.load(states + row * 4 + 1).to(utype)
    keys = _to_order_key(values, valid)
    active = valid & ((keys & desired_mask) == desired)
    digit = ((keys >> DIGIT_POS) & radix_mask).to(tl.int32)

    bins = tl.arange(0, RADIX_SIZE)
    counts = tl.zeros((RADIX_SIZE,), dtype=tl.int32)
    for radix_bin in tl.static_range(0, RADIX_SIZE):
        count = tl.sum((active & (digit == radix_bin)).to(tl.int32), axis=0)
        counts += tl.where(bins == radix_bin, count, 0)
    base = (row * NUM_CHUNKS + chunk) * RADIX_SIZE
    tl.store(partial_counts + base + bins, counts)


@libentry()
@triton.jit
def _radix_update_kernel(
    partial_counts,
    states,
    NUM_CHUNKS: tl.constexpr,
    DIGIT_POS: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
    FIRST_DIGIT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    bins = tl.arange(0, RADIX_SIZE)
    counts = tl.zeros((RADIX_SIZE,), dtype=tl.int32)
    for chunk in tl.range(0, NUM_CHUNKS):
        base = (row * NUM_CHUNKS + chunk) * RADIX_SIZE
        counts += tl.load(partial_counts + base + bins)

    if FIRST_DIGIT:
        valid_count = tl.sum(counts, axis=0)
        remaining = (valid_count + 1) // 2
        tl.store(states + row * 4 + 3, valid_count)
    else:
        remaining = tl.load(states + row * 4 + 2).to(tl.int32)

    selected_bin = tl.full((), 0, dtype=tl.int32)
    found = tl.full((), 0, dtype=tl.int1)
    for radix_bin in tl.static_range(0, RADIX_SIZE):
        count = tl.sum(tl.where(bins == radix_bin, counts, 0), axis=0)
        take = (~found) & (remaining > 0) & (remaining <= count)
        selected_bin = tl.where(take, radix_bin, selected_bin)
        remaining = tl.where((~found) & (~take), remaining - count, remaining)
        found = found | take

    desired = tl.load(states + row * 4)
    desired_mask = tl.load(states + row * 4 + 1)
    radix_mask: tl.constexpr = RADIX_SIZE - 1
    tl.store(states + row * 4, desired | (selected_bin.to(tl.int64) << DIGIT_POS))
    tl.store(states + row * 4 + 1, desired_mask | (radix_mask << DIGIT_POS))
    tl.store(states + row * 4 + 2, remaining)


@libentry()
@triton.jit
def _radix_partial_index_kernel(
    inp,
    partial_indices,
    states,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    columns = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = columns < N
    values = tl.load(inp + row * N + columns, mask=mask, other=0.0)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    if dtype.is_floating():
        valid = mask & (values == values)
    else:
        valid = mask
    desired = tl.load(states + row * 4).to(utype)
    keys = _to_order_key(values, valid)
    local_index = tl.min(tl.where(valid & (keys == desired), columns, N), axis=0)
    tl.store(partial_indices + row * NUM_CHUNKS + chunk, local_index)


@libentry()
@triton.jit
def _radix_store_kernel(
    inp,
    partial_indices,
    states,
    out_values,
    out_indices,
    N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    index = tl.full((), N, dtype=tl.int32)
    for chunk in tl.range(0, NUM_CHUNKS):
        candidate = tl.load(partial_indices + row * NUM_CHUNKS + chunk)
        index = tl.minimum(index, candidate)
    valid_count = tl.load(states + row * 4 + 3)
    dtype = inp.dtype.element_ty
    if dtype.is_floating():
        value = tl.load(
            inp + row * N + index,
            mask=valid_count > 0,
            other=float("nan"),
        )
        value = tl.where(valid_count > 0, value, float("nan"))
        index = tl.where(valid_count > 0, index, 0)
    else:
        value = tl.load(inp + row * N + index)
    tl.store(out_values + row, value)
    tl.store(out_indices + row, index.to(tl.int64))


def _tl_sort_nanmedian(rows, values, indices):
    M, N = rows.shape
    block_n = triton.next_power_of_2(N)
    if block_n <= 1024:
        num_warps = 1
    elif rows.dtype.is_floating_point:
        num_warps = 8
    else:
        num_warps = 2
    with torch_device_fn.device(rows.device):
        _nanmedian_tl_sort_kernel[(M,)](
            rows,
            values,
            indices,
            N,
            block_n,
            num_warps=num_warps,
            num_stages=1,
        )


def _radix_nanmedian(rows, values, indices):
    M, N = rows.shape
    block_n = min(triton.next_power_of_2(N), _RADIX_BLOCK_N)
    num_chunks = triton.cdiv(N, block_n)
    states = torch.empty((M, 4), dtype=torch.int64, device=rows.device)
    partial_counts = torch.empty(
        (M, num_chunks, _RADIX_SIZE), dtype=torch.int32, device=rows.device
    )
    partial_indices = torch.empty(
        (M, num_chunks), dtype=torch.int32, device=rows.device
    )
    nbits = rows.element_size() * 8

    with torch_device_fn.device(rows.device):
        _radix_init_kernel[(M,)](states, num_warps=1, num_stages=1)
        first_digit = True
        for digit_pos in range(nbits - _RADIX_BITS, -1, -_RADIX_BITS):
            _radix_partial_count_kernel[(M, num_chunks)](
                rows,
                partial_counts,
                states,
                N,
                block_n,
                num_chunks,
                digit_pos,
                _RADIX_BITS,
                _RADIX_SIZE,
                num_warps=8,
                num_stages=1,
            )
            _radix_update_kernel[(M,)](
                partial_counts,
                states,
                num_chunks,
                digit_pos,
                _RADIX_SIZE,
                first_digit,
                num_warps=1,
                num_stages=1,
            )
            first_digit = False
        _radix_partial_index_kernel[(M, num_chunks)](
            rows,
            partial_indices,
            states,
            N,
            block_n,
            num_chunks,
            num_warps=8,
            num_stages=1,
        )
        _radix_store_kernel[(M,)](
            rows,
            partial_indices,
            states,
            values,
            indices,
            N,
            num_chunks,
            num_warps=1,
            num_stages=1,
        )


def _nanmedian_dim_impl(inp, dim, keepdim, out=None):
    dim = _normalize_dim(dim, inp.ndim)
    if inp.ndim == 0:
        return _generic_nanmedian_dim_impl(inp, dim, keepdim, out=out)

    shape = list(inp.shape)
    N = shape[dim]
    out_shape = shape[:dim] + shape[dim + 1 :]
    M = math.prod(out_shape)
    if N <= 1 or M == 0 or N > INT32_MAX or inp.dtype not in RADIX_SELECT_DTYPES:
        return _generic_nanmedian_dim_impl(inp, dim, keepdim, out=out)

    keepdim_shape = shape.copy()
    keepdim_shape[dim] = 1
    output_shape = keepdim_shape if keepdim else out_shape
    compute_shape = output_shape if out is not None else keepdim_shape
    if out is None:
        values = torch.empty(compute_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(compute_shape, dtype=torch.long, device=inp.device)
    else:
        values, indices = out

    rows = dim_compress(inp, dim).reshape(M, N)
    values_contiguous = values.is_contiguous()
    indices_contiguous = indices.is_contiguous()
    flat_values = values.reshape(M) if values_contiguous else values.new_empty((M,))
    flat_indices = indices.reshape(M) if indices_contiguous else indices.new_empty((M,))

    if N <= _TL_SORT_LIMIT:
        _tl_sort_nanmedian(rows, flat_values, flat_indices)
    else:
        # Partial histograms avoid both whole-row sorting and cross-program
        # atomics, which are costly on MetaX for long reductions.
        _radix_nanmedian(rows, flat_values, flat_indices)

    if not values_contiguous:
        values.copy_(flat_values.reshape(values.shape))
    if not indices_contiguous:
        indices.copy_(flat_indices.reshape(indices.shape))
    if out is None and not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)
    return NanMedian(values=values, indices=indices)


def _nanmedian_flat_impl(inp, out=None):
    if (
        inp.numel() == 0
        or inp.numel() > INT32_MAX
        or inp.dtype not in RADIX_SELECT_DTYPES
    ):
        return _generic_nanmedian_flat_impl(inp, out=out)

    flat = inp.reshape(-1).contiguous()
    if out is None:
        return _nanmedian_dim_impl(flat, 0, False).values

    indices = torch.empty((), dtype=torch.long, device=inp.device)
    _nanmedian_dim_impl(flat, 0, False, out=(out, indices))
    return out


def nanmedian(inp):
    logger.debug("GEMS_METAX NANMEDIAN")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp)


def nanmedian_out(inp, *, out):
    logger.debug("GEMS_METAX NANMEDIAN OUT")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp, out=out)


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_METAX NANMEDIAN DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_METAX NANMEDIAN DIM VALUES")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim, out=(values, indices))
