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
    MAX_BLOCK_N,
    RADIX_SELECT_DTYPES,
    NanMedian,
    _check_supported_dtype,
    _is_not_nan,
)
from flag_gems.ops.nanmedian import _nanmedian_dim_impl as _generic_nanmedian_dim_impl
from flag_gems.ops.nanmedian import _nanmedian_flat_impl as _generic_nanmedian_flat_impl
from flag_gems.ops.nanmedian import _normalize_dim, _to_order_key
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_RADIX_BITS = 4
_RADIX_SIZE = 1 << _RADIX_BITS
_RADIX_BLOCK_N = 4096


@libentry()
@triton.jit
def _radix_init_kernel(
    valid_counts,
    states,
    result_indices,
    bin_counts,
    N: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    bins = tl.arange(0, RADIX_SIZE)
    tl.store(valid_counts + row, 0 if IS_FLOAT else N)
    tl.store(states + row * 3, 0)
    tl.store(states + row * 3 + 1, 0)
    tl.store(states + row * 3 + 2, 0)
    tl.store(result_indices + row, N)
    tl.store(bin_counts + row * RADIX_SIZE + bins, 0)


@libentry()
@triton.jit
def _radix_count_valid_kernel(
    inp,
    valid_counts,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    cols = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    values = tl.load(inp + row * N + cols, mask=mask, other=float("nan"))
    valid = mask & _is_not_nan(values, False)
    count = tl.sum(valid.to(tl.int64), axis=0)
    tl.atomic_add(valid_counts + row, count, sem="relaxed")


@libentry()
@triton.jit
def _radix_init_rank_kernel(valid_counts, states):
    row = tl.program_id(0)
    valid_count = tl.load(valid_counts + row)
    tl.store(states + row * 3 + 2, (valid_count + 1) // 2)


@libentry()
@triton.jit
def _radix_count_kernel(
    inp,
    bin_counts,
    states,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIGIT_POS: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    cols = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    values = tl.load(inp + row * N + cols, mask=mask, other=0.0)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    radix_mask: tl.constexpr = RADIX_SIZE - 1

    if dtype.is_floating():
        valid = mask & _is_not_nan(values, False)
    else:
        valid = mask

    desired = tl.load(states + row * 3).to(utype)
    desired_mask = tl.load(states + row * 3 + 1).to(utype)
    keys = _to_order_key(values, valid)
    active = valid & ((keys & desired_mask) == desired)
    digit = ((keys >> DIGIT_POS) & radix_mask).to(tl.int32)

    bins = tl.arange(0, RADIX_SIZE)
    counts = tl.zeros((RADIX_SIZE,), dtype=tl.int64)
    for radix_bin in tl.static_range(0, RADIX_SIZE):
        count = tl.sum((active & (digit == radix_bin)).to(tl.int64), axis=0)
        counts += tl.where(bins == radix_bin, count, 0)
    tl.atomic_add(bin_counts + row * RADIX_SIZE + bins, counts, sem="relaxed")


@libentry()
@triton.jit
def _radix_update_kernel(
    bin_counts,
    states,
    DIGIT_POS: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    bins = tl.arange(0, RADIX_SIZE)
    counts_ptr = bin_counts + row * RADIX_SIZE + bins
    counts = tl.load(counts_ptr)
    k_to_find = tl.load(states + row * 3 + 2)
    cumsum = tl.cumsum(counts, axis=0)
    previous = cumsum - counts
    take = (k_to_find <= cumsum) & (k_to_find > previous)
    selected_bin = tl.min(tl.where(take, bins, RADIX_SIZE - 1), axis=0).to(tl.int64)
    counts_before = tl.max(tl.where(take, previous, 0), axis=0)

    desired = tl.load(states + row * 3)
    desired_mask = tl.load(states + row * 3 + 1)
    radix_mask: tl.constexpr = RADIX_SIZE - 1
    tl.store(states + row * 3, desired | (selected_bin << DIGIT_POS))
    tl.store(states + row * 3 + 1, desired_mask | (radix_mask << DIGIT_POS))
    tl.store(states + row * 3 + 2, k_to_find - counts_before)
    tl.store(counts_ptr, 0)


@libentry()
@triton.jit
def _radix_find_index_kernel(
    inp,
    states,
    valid_counts,
    result_indices,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    if tl.load(valid_counts + row) > 0:
        cols = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        values = tl.load(inp + row * N + cols, mask=mask, other=0.0)
        dtype = inp.dtype.element_ty
        nbits: tl.constexpr = dtype.primitive_bitwidth
        utype = tl.dtype(f"uint{nbits}")
        if dtype.is_floating():
            valid = mask & _is_not_nan(values, False)
        else:
            valid = mask

        desired = tl.load(states + row * 3).to(utype)
        keys = _to_order_key(values, valid)
        local_index = tl.min(tl.where(valid & (keys == desired), cols, N), axis=0).to(
            tl.int64
        )
        tl.atomic_min(result_indices + row, local_index, sem="relaxed")


@libentry()
@triton.jit
def _radix_store_result_kernel(
    inp,
    out_values,
    out_indices,
    valid_counts,
    result_indices,
    N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    dtype = inp.dtype.element_ty
    valid_count = tl.load(valid_counts + row)
    index = tl.load(result_indices + row)
    if dtype.is_floating():
        value = tl.load(inp + row * N + index, mask=valid_count > 0, other=float("nan"))
        index = tl.where(valid_count > 0, index, 0)
    else:
        value = tl.load(inp + row * N + index)
    tl.store(out_values + row, value)
    tl.store(out_indices + row, index)


def _radix_nanmedian(rows, values, indices):
    M, N = rows.shape
    block_n = min(triton.next_power_of_2(N), _RADIX_BLOCK_N)
    num_chunks = triton.cdiv(N, block_n)
    num_warps = 4 if block_n <= 1024 else 8
    is_float = rows.dtype.is_floating_point
    valid_counts = torch.empty((M,), dtype=torch.int64, device=rows.device)
    states = torch.empty((M, 3), dtype=torch.int64, device=rows.device)
    result_indices = torch.empty((M,), dtype=torch.int64, device=rows.device)
    bin_counts = torch.empty((M, _RADIX_SIZE), dtype=torch.int64, device=rows.device)

    with torch_device_fn.device(rows.device):
        _radix_init_kernel[(M,)](
            valid_counts,
            states,
            result_indices,
            bin_counts,
            N,
            is_float,
            _RADIX_SIZE,
            num_warps=4,
            num_stages=1,
        )
        if is_float:
            _radix_count_valid_kernel[(M, num_chunks)](
                rows,
                valid_counts,
                N,
                block_n,
                num_warps=num_warps,
                num_stages=1,
            )
        _radix_init_rank_kernel[(M,)](valid_counts, states, num_warps=4, num_stages=1)
        for digit_pos in range(rows.element_size() * 8 - _RADIX_BITS, -1, -_RADIX_BITS):
            _radix_count_kernel[(M, num_chunks)](
                rows,
                bin_counts,
                states,
                N,
                block_n,
                digit_pos,
                _RADIX_BITS,
                _RADIX_SIZE,
                num_warps=num_warps,
                num_stages=1,
            )
            _radix_update_kernel[(M,)](
                bin_counts,
                states,
                digit_pos,
                _RADIX_SIZE,
                num_warps=4,
                num_stages=1,
            )
        _radix_find_index_kernel[(M, num_chunks)](
            rows,
            states,
            valid_counts,
            result_indices,
            N,
            block_n,
            num_warps=num_warps,
            num_stages=1,
        )
        _radix_store_result_kernel[(M,)](
            rows,
            values,
            indices,
            valid_counts,
            result_indices,
            N,
            num_warps=4,
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
    if (
        N <= MAX_BLOCK_N
        or M == 0
        or N > INT32_MAX
        or inp.dtype not in RADIX_SELECT_DTYPES
    ):
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

    rows = torch.movedim(inp, dim, -1).contiguous().reshape(M, N)
    values_contiguous = values.is_contiguous()
    indices_contiguous = indices.is_contiguous()
    flat_values = (
        values.reshape(M)
        if values_contiguous
        else torch.empty((M,), dtype=values.dtype, device=values.device)
    )
    flat_indices = (
        indices.reshape(M)
        if indices_contiguous
        else torch.empty((M,), dtype=indices.dtype, device=indices.device)
    )

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
        inp.numel() <= MAX_BLOCK_N
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
    logger.debug("GEMS_HYGON NANMEDIAN")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp)


def nanmedian_out(inp, *, out):
    logger.debug("GEMS_HYGON NANMEDIAN OUT")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp, out=out)


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_HYGON NANMEDIAN DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_HYGON NANMEDIAN DIM VALUES")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim, out=(values, indices))
